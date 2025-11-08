#!/usr/bin/env python3
"""Generate booking conversations via self-play between user and chat agents."""

from __future__ import annotations

import os
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml
import sys

# Ensure the project root is on sys.path so `agents/...` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.chat.module import FlightBookingChatAgent
from agents.user.module import FlightBookingUserAgent
from agents.verifier.module import BookingVerifierAgent
from agents.judge.module import PolicyJudgeAgent
from utils import BookingObjectiveGenerator


DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "training", "configs", "training_config.yaml")
QUIT_WORDS = {"quit", "exit", "bye", "goodbye"}


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
    
def fmt_secs(sec: float) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

def load_resources() -> Dict[str, Path]:
    return {
        "chat_config": "agents/chat/config.yaml",
        "db_path": "database/flights.db",
        "server_path": "mcp-server/database_server.py",
        "user_config": "agents/user/config.yaml",
        "verifier_config": "agents/verifier/config.yaml",
        "judge_config": "agents/judge/config.yaml",
    }


def reset_agents(chat: FlightBookingChatAgent, user: FlightBookingUserAgent, objective: str) -> None:
    chat.conversation_history = []
    chat.booking_context = {
        "summary": "",
        "current_requirements": {},
        "search_history": [],
        "preferences": {},
    }
    user.booking_objective = objective


def detect_confabulation(
    chat: FlightBookingChatAgent,
    verifier: BookingVerifierAgent,
) -> int:

    summary = chat.generate_booking_summary()
    report = verifier.verify_booking(summary, chat.mcp_client)
    if report.get("verification_complete") and (not report.get("booking_exists")):
        return 1
    else:
        return 0


def detect_non_compliance(
    judge: Optional[PolicyJudgeAgent],
    policy_bundle: Dict[str, Any],
    messages: Sequence[Dict[str, str]],
) -> tuple[int, str]:
    if judge is None:
        return 0, ""

    result = judge.evaluate(messages, policy_bundle)
    violations = result.violations if isinstance(getattr(result, "violations", None), list) else []
    titles = [str(v.get("policy_title")) for v in violations]

    non_compliance_flag = result.reward
    reasons = "; ".join(titles)
    return non_compliance_flag, reasons


def run_episode(
    objective: str,
    user: FlightBookingUserAgent,
    chat: FlightBookingChatAgent,
    verifier: BookingVerifierAgent,
    judge: Optional[PolicyJudgeAgent],
    policy_bundle: Dict[str, Any],
    max_dialogues: int,
    attempts: int,
) -> Optional[Dict[str, Any]]:

    reset_agents(chat, user, objective)

    dialogues = 0
    confabulation_flag = 0
    non_compliance_flag = 0

    attempts += 1
    if attempts > 3:
        return None

    while True:
        
        user_message = user.generate_user_message(chat.conversation_history, objective)
        normalized_user = user_message.lower()
        if any(q in normalized_user for q in QUIT_WORDS):
            confabulation_flag = detect_confabulation(chat, verifier)
            non_compliance_flag, reasons = detect_non_compliance(judge, policy_bundle, chat.conversation_history)
            break
        assistant_message = chat.generate_chat_message(user_message)

        dialogues += 1
        if dialogues > max_dialogues:
            return run_episode(objective, user, chat, verifier, judge, policy_bundle, max_dialogues, attempts)

    reward = {
        "non_compliance": int(non_compliance_flag),
        "confabulation": int(confabulation_flag),
    }

    return {
        "objective": objective,
        "messages": chat.conversation_history,
        "reward": reward,
        "reason": reasons if non_compliance_flag else "confabulation",
    }


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="dataset/conversations.jsonl")
    parser.add_argument("--max_dialogues", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=1, help="Print progress every N episodes")
    args = parser.parse_args()
    resources = load_resources()
    policy_bundle = load_yaml(resources["policy_path"])
    objective_generator = BookingObjectiveGenerator(db_path=resources["db_path"])
    max_diag = args.max_dialogues

    chat = FlightBookingChatAgent(
        config_path=str(resources["chat_config"]),
        db_path=str(resources["db_path"]),
        server_path=str(resources["server_path"]),
    )

    user = FlightBookingUserAgent(str(resources["user_config"]))
    verifier = BookingVerifierAgent(str(resources["verifier_config"]))
    judge = PolicyJudgeAgent(str(resources["judge_config"]))

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        (
            f"[Dataset] Start | episodes={args.episodes} | out={output_path} | "
            f"max_dialogues={args.max_dialogues} | log_every={args.log_every}"
        )
    )

    saved = 0
    failed = 0
    total_confab = 0
    total_noncompliance = 0
    start_ts = time.time()

    with output_path.open("a", encoding="utf-8") as handle:
        while saved < args.episodes:

            objective = objective_generator.generate()
            print(f"[Dataset] Episode {saved + 1} | Objective: {objective}")
            result = run_episode(objective, user, chat, verifier, judge, policy_bundle, max_diag, 0)
            if not result:
                continue

            episode = {
                "episode_id": str(saved + 1),
                "objective": result["objective"],
                "messages": result["messages"],
                "reward": result["reward"],
                "reason": result["reason"],
            }

            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
            saved += 1
            total_confab += int(episode["reward"].get("confabulation", 0))
            total_noncompliance += int(episode["reward"].get("non_compliance", 0))

            if saved % args.log_every == 0 or saved == args.episodes:
                elapsed = time.time() - start_ts
                eps_rate = saved / elapsed if elapsed > 0 else 0.0
                remaining = args.episodes - saved
                eta = remaining / eps_rate if eps_rate > 0 else 0.0
                print(
                    (
                        f"[Dataset] saved={saved}/{args.episodes} | "
                        f"confab={total_confab} | non_compliance={total_noncompliance} | "
                        f"failures={failed} | eps={eps_rate:.2f}/s | "
                        f"elapsed={fmt_secs(elapsed)} | eta={fmt_secs(eta)}"
                    )
                )


if __name__ == "__main__":
    main()
