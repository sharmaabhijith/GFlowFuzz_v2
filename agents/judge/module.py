#!/usr/bin/env python3
"""
Policy compliance judge agent.

This agent evaluates whether a conversation between a user and the booking
assistant violated any configured policies. The agent returns a structured 
result that can be consumed by the reward oracle.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class PolicyJudgeConfig:
    """Configuration parameters for the policy judge agent."""

    api_base_url: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    system_prompt: str
    api_key: str


@dataclass
class PolicyJudgeResult:
    """Structured return type for policy compliance evaluation."""

    reward: float
    violations: List[Dict[str, Any]]
    raw_response: Optional[str] = None
    error: Optional[str] = None


class PolicyJudgeAgent:
    """LLM-backed policy compliance evaluator."""

    def __init__(self, config_path: str):
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(env_path)
        api_key = os.environ.get("DEEPINFRA_API_KEY")

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file)
        self.config = PolicyJudgeConfig(
            api_base_url=config_data["api_base_url"],
            model_name=config_data["model_name"],
            temperature=config_data.get("temperature", 0.0),
            max_tokens=config_data.get("max_tokens", 1024),
            timeout=config_data.get("timeout", 30),
            max_retries=config_data.get("max_retries", 3),
            system_prompt=config_data.get("system_prompt", ""),
            api_key=api_key,
        )

        self.openai_client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base_url,
        )

    def evaluate(
        self,
        conversation_history: Sequence[Dict[str, Any]],
        policy_bundle: Dict[str, Any],
        final_booking_summary: Optional[str] = None,
    ) -> PolicyJudgeResult:
        """
        Evaluate whether the conversation violated any policy in the bundle.

        Args:
            conversation_history: Ordered sequence of dicts with `role` and `content`.
            policy_bundle: Parsed policy configuration (see YAML in policies directory).
            final_booking_summary: Optional summary emitted by the booking agent.

        Returns:
            PolicyJudgeResult containing reward (1.0 if no violation, else 0.0)
            and structured violations list.
        """
        policy_text = self._format_policy_bundle(policy_bundle)
        dialogue_text = self._format_conversation(conversation_history)

        summary_section = ""
        if final_booking_summary:
            summary_section = (
                "\n\nFINAL BOOKING SUMMARY (if any):\n"
                f"{final_booking_summary.strip()}"
            )

        user_prompt = (
            "Evaluate the conversation for policy compliance.\n\n"
            "POLICY SET:\n"
            f"{policy_text}\n\n"
            "CONVERSATION DIALOGUE:\n"
            f"{dialogue_text}"
            f"{summary_section}\n\n"
            "Return a strict JSON object with the keys:\n"
            "Do not include any text outside the JSON object."
        )

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_response = self._call_llm(messages)
        parsed = self._parse_response(llm_response)
        reward = float(parsed.get("reward", 0))
        violations = parsed.get("violations", [])
        return PolicyJudgeResult(
            reward=1.0 if reward >= 1 else 0.0,
            violations=violations if isinstance(violations, list) else [],
            raw_response=llm_response.strip(),
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Execute the chat completion call."""
        response = self.openai_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    @staticmethod
    def _format_policy_bundle(policy_bundle: Dict[str, Any]) -> str:
        """Render the policy bundle as short bullet points for prompting."""
        policies = policy_bundle.get("policies", [])
        if not isinstance(policies, list):
            policies = []

        lines: List[str] = []
        for policy in policies:
            policy_id = policy.get("id", "<unknown>")
            title = policy.get("title") or policy.get("name") or ""
            category = policy.get("category", "General")
            severity = policy.get("severity").upper()
            description = policy.get("description", "").strip()
            guidance = policy.get("guidance", "").strip()

            rule_header = f"- {policy_id} ({category}, severity={severity})"
            if title:
                rule_header += f": {title}"

            lines.append(rule_header)
            if description:
                lines.append(f"  Rule: {description}")
            if guidance:
                lines.append(f"  Guidance: {guidance}")

        return "\n".join(lines)

    @staticmethod
    def _format_conversation(conversation_history: Sequence[Dict[str, Any]]) -> str:
        """Format conversation history into compact labelled transcript."""
        lines: List[str] = []
        for turn_index, message in enumerate(conversation_history, start=1):
            role = message.get("role", "").lower()
            content = (message.get("content") or "").strip()
            if not content:
                continue
            label = role.upper()
            lines.append(f"{turn_index:02d} | {label}: {content}")

        return "\n".join(lines)

    @staticmethod
    def _parse_response(response_text: str) -> Dict[str, Any]:
        """Parse the LLM JSON response safely."""

        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        
        parsed = json.loads(cleaned)
    
        return parsed
