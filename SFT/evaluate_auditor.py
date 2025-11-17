#!/usr/bin/env python3
"""
Evaluate the trained SFT auditor model in real-time against the chat agent.
"""

from __future__ import annotations

import re
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import sys

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.chat import FlightBookingChatAgent
from agents.auditor import AuditorAgent, AuditorConfig
from agents.verifier import BookingVerifierAgent
from agents.judge import PolicyJudgeAgent
from utils import BookingObjectiveGenerator

from dotenv import load_dotenv

load_dotenv(".env", override=True)

QUIT_WORDS = {"quit", "exit", "bye", "goodbye"}


def fmt_secs(sec: float) -> str:
    sec = max(0, int(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_resources() -> Dict[str, Path]:
    return {
        "db_path": Path("database/flights.db"),
        "server_path": Path("mcp-server/database_server.py"),
    }


def sanitize_for_filename(value: str) -> str:
    """Sanitize a string for use in filenames"""
    sanitized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip())
    sanitized = sanitized.strip("-").lower()
    return sanitized or "na"


def _config_value(cfg: Any, key: str, default: Any = None) -> Any:
    """Extract value from config object or dict"""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def computed_dataset_path(auditor_cfg: Any, chat_cfg: Any) -> Path:
    auditor_model = sanitize_for_filename(str(_config_value(auditor_cfg, "model_name", "auditor")))
    chat_model = sanitize_for_filename(str(_config_value(chat_cfg, "model_name", "chat")))
    temperature = float(_config_value(auditor_cfg, "temperature"))
    temp_suffix = int(round(temperature * 10))
    filename = f"data_auditor-{auditor_model}_chat-{chat_model}_temp-{temp_suffix}.jsonl"
    return Path("SFT/eval_data") / filename


def reset_agents(chat: FlightBookingChatAgent, objective: str) -> None:
    """Reset chat agent state for new episode"""
    chat.conversation_history = []
    chat.booking_context = {
        "summary": "",
        "current_requirements": {},
        "search_history": [],
        "preferences": {},
    }


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
    messages: Sequence[Dict[str, str]],
) -> tuple[int, str, list]:
    if judge is None:
        return 0, "", [""]

    result = judge.evaluate(messages)
    violations = result.violations if isinstance(getattr(result, "violations", None), list) else []
    reason = [str(v.get("reason")) for v in violations]
    confidence = [str(v.get("confidence")) for v in violations]
    non_compliance_flag = result.reward
    reasons = "; ".join(reason)
    return non_compliance_flag, reasons, confidence


def format_conversation_for_auditor(
    conversation_history: list[dict[str, str]],
    objective: str
) -> str:
    """
    Format conversation history into a state string for the auditor model.
    This should match the format used during training.
    """
    # Build the state string with objective and recent conversation
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history

    state_parts = [f"Booking Objective: {objective}\n"]

    if recent_history:
        state_parts.append("Recent Conversation:")
        for msg in recent_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            state_parts.append(f"{role}: {content}")

    state_parts.append("\nYour response:")

    return "\n".join(state_parts)


def run_episode(
    objective: str,
    auditor: AuditorAgent,
    chat: FlightBookingChatAgent,
    verifier: BookingVerifierAgent,
    judge: Optional[PolicyJudgeAgent],
    max_dialogues: int,
    attempts: int,
) -> Optional[Dict[str, Any]]:
    """Run a single episode with the trained auditor model"""

    reset_agents(chat, objective)

    dialogues = 0
    confabulation_flag = 0
    non_compliance_flag = 0

    attempts += 1
    if attempts > 3:
        return None

    while True:
        # Format state for auditor
        state = format_conversation_for_auditor(chat.conversation_history, objective)
        auditor_message = auditor.get_action(state)
        print("[Auditor]:", auditor_message)
        # Check for quit words
        normalized_auditor = auditor_message.lower()
        if any(q in normalized_auditor for q in QUIT_WORDS):
            confabulation_flag = detect_confabulation(chat, verifier)
            non_compliance_flag, reasons, confidence = detect_non_compliance(judge, chat.conversation_history)
            break
        # Generate assistant response
        assistant_message = chat.generate_chat_message(auditor_message)
        print("[Chat Agent]:", assistant_message)

        dialogues += 1
        if dialogues > max_dialogues:
            return run_episode(objective, auditor, chat, verifier, judge, max_dialogues, attempts)

    reward = {
        "non_compliance": int(non_compliance_flag),
        "confabulation": int(confabulation_flag),
    }

    if non_compliance_flag and confabulation_flag:
        reasons = reasons + "; " + "confabulation detected"
    elif confabulation_flag and not non_compliance_flag:
        reasons = "confabulation detected"
    else:
        reasons = reasons if non_compliance_flag else ""

    return {
        "objective": objective,
        "messages": chat.conversation_history,
        "reward": reward,
        "reason": reasons,
        "confidence": confidence if non_compliance_flag else [""],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SFT auditor model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--max_dialogues", type=int, default=30, help="Max dialogue turns per episode")
    parser.add_argument("--model_path", type=str, default="SFT/trained_models/qwen3-4b-sft-os1-epoch",
                        help="Path to trained model (default: final model, use --use-checkpoint for latest checkpoint)")
    parser.add_argument("--chat-model", type=str, default=None, help="Override chat agent model name")
    parser.add_argument("--temperature", type=float, default=0.1, help="Auditor sampling temperature")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")

    args = parser.parse_args()
    max_diag = args.max_dialogues

    # Determine which model to load
    model_base_path = Path(args.model_path)
    
    model_path = args.model_path
    print("Using final trained model")

    # Load resources
    resources = load_resources()
    objective_generator = BookingObjectiveGenerator(db_path=resources["db_path"])

    # Initialize chat agent
    chat = FlightBookingChatAgent(
        db_path=str(resources["db_path"]),
        server_path=str(resources["server_path"]),
        model_name=args.chat_model,
    )

    # Initialize trained auditor agent
    print(f"\n{'='*60}")
    print(f"Loading trained SFT auditor model...")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # Read system prompt from config
    system_prompt_path = Path("agents/system_prompts/auditor.txt")
    system_prompt = system_prompt_path.read_text(encoding="utf-8") if system_prompt_path.exists() else ""

    auditor_config = AuditorConfig(
        model_name=model_path,  # Path to the LoRA adapter
        max_length=128,  # Max new tokens to generate
        temperature=args.temperature,
        top_p=0.9,
        do_sample=True,
        device=args.device,
        system_prompt=system_prompt,
    )

    auditor = AuditorAgent(auditor_config)
    auditor.initialize_model()

    # Initialize verifier and judge
    verifier = BookingVerifierAgent()
    judge = PolicyJudgeAgent()

    # Print evaluation info
    model_info = auditor.get_model_info()
    print(f"\n{'='*60}")
    print(f"Model Information:")
    print(f"{'='*60}")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print(f"{'='*60}\n")

    # Setup output path
    output_path = computed_dataset_path(auditor.config, chat.config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[Evaluation] Start | episodes={args.episodes} | out={output_path} | "
        f"max_dialogues={args.max_dialogues}"
    )

    # Run evaluation
    saved = 0
    failed = 0
    total_confab = 0
    total_noncompliance = 0
    start_ts = time.time()

    with output_path.open("a", encoding="utf-8") as handle:
        while saved < args.episodes:
            try:
                objective = objective_generator.generate()
                print(f"[Evaluation] Episode {saved + 1} | Objective: {objective}")

                result = run_episode(objective, auditor, chat, verifier, judge, max_diag, 0)

                if not result:
                    failed += 1
                    continue

                episode = {
                    "episode_id": str(saved + 1),
                    "objective": result["objective"],
                    "messages": result["messages"],
                    "reward": result["reward"],
                    "reason": result["reason"],
                    "confidence": result["confidence"],
                }

                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
                handle.flush()  # Ensure data is written immediately
                saved += 1
                total_confab += int(episode["reward"].get("confabulation", 0))
                total_noncompliance += int(episode["reward"].get("non_compliance", 0))

                elapsed = time.time() - start_ts
                eps_rate = saved / elapsed if elapsed > 0 else 0.0
                remaining = args.episodes - saved
                eta = remaining / eps_rate if eps_rate > 0 else 0.0

                print(
                    f"[Evaluation] saved={saved}/{args.episodes} | "
                    f"confab={total_confab} | non_compliance={total_noncompliance} | "
                    f"failures={failed} | eps={eps_rate:.2f}/s | "
                    f"elapsed={fmt_secs(elapsed)} | eta={fmt_secs(eta)}"
                )

                break

            except Exception as e:
                failed += 1
                continue

    # Final summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total episodes: {saved}")
    print(f"Total confabulations: {total_confab}")
    print(f"Total non-compliance: {total_noncompliance}")
    print(f"Failed episodes: {failed}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
