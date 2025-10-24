from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from agents.chat.module import FlightBookingChatAgent
from agents.coder.module import SQLCoderAgent
from agents.verifier.module import BookingVerifierAgent
from training.environment import ConversationState
from agents.judge.module import PolicyJudgeAgent, PolicyJudgeResult
from training.process_reward import ARMOProcessScorer


class Oracle:
    """Encapsulates reward computations for the booking conversation environment."""

    def __init__(
        self,
        booking_agent: Optional["FlightBookingChatAgent"],
        verifier_agent: Optional["BookingVerifierAgent"],
        coder_agent: Optional["SQLCoderAgent"],
        judge_agent: Optional["PolicyJudgeAgent"] = None,
        policy_bundle: Optional[Dict[str, Any]] = None,
        process_reward_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store references to the agents required for reward computation."""
        self.booking_agent = booking_agent
        self.verifier_agent = verifier_agent
        self.coder_agent = coder_agent
        self.judge_agent = judge_agent
        self.policy_bundle = policy_bundle
        self.process_reward_config = process_reward_config or {}

        self.latest_verification_report: Optional[Dict[str, Any]] = None
        self.latest_policy_result: Optional["PolicyJudgeResult"] = None
        self.latest_reward_components: Dict[str, float] = {}
        self.latest_process_reward: Optional[float] = None

        self.default_process_reward = float(
            self.process_reward_config.get("default_reward", 0.01)
        )
        self._process_reward_buffer: List[float] = []
        self.armo_scorer: Optional[ARMOProcessScorer] = None

        armo_cfg = dict(self.process_reward_config.get("armo", {}) or {})
        enabled = armo_cfg.pop("enabled", True)
        if enabled and armo_cfg.get("model_path"):
            try:
                self.armo_scorer = ARMOProcessScorer(**armo_cfg)
                logging.info("ARMO process scorer initialized.")
            except Exception as exc:
                logging.warning("Failed to initialize ARMO process scorer: %s", exc)
                self.armo_scorer = None

    def compute_shaped(
        self,
        terminal_reward: float,
        num_steps: int,
        gamma: float = 0.99,
    ) -> List[float]:
        """
        Compute shaped rewards with proper credit assignment.

        Uses exponential decay to assign credit to earlier actions,
        ensuring that actions closer to success receive more credit.
        """
        shaped_rewards: List[float] = []

        for step_index in range(num_steps):
            steps_from_end = num_steps - 1 - step_index
            discounted_reward = terminal_reward * (gamma ** steps_from_end)
            shaped_rewards.append(discounted_reward)

        return shaped_rewards

    def compute_process(
        self,
        state: "ConversationState",
        action: str,
        next_state: "ConversationState",
    ) -> float:
        """
        Compute process reward for a single transition.

        Uses an ARMO reward model if configured; otherwise returns a default
        shaping bonus to encourage longer dialogs.
        """
        reward = self.armo_scorer.score(context_history, action)
        value = float(reward)
        self._process_reward_buffer.append(value)
        self.latest_process_reward = value
        return reward

    def __compute_hallucination_reward(
        self,
        state: Optional["ConversationState"],
        final_booking_summary: Optional[str],
    ) -> float:
        """
        Compute reward for the complete booking confirmation.
        - Reward = 1 if booking doesn't exist
        - Reward = 0 if booking already exists
        """
        verification_report = self.verifier_agent.verify_final_booking_only(
            final_booking_summary,
            self.booking_agent.mcp_client,
            self.coder_agent,
        )
        self.latest_verification_report = verification_report
        reward = verification_report.get("reward", 0.0)

        return reward

    def __compute_policy_reward(
        self,
        final_booking_summary: Optional[str],
    ) -> float:
        """Run the policy judge agent to assess policy compliance."""
        history = list(self.booking_agent.conversation_history)
        result = self.judge_agent.evaluate(
            conversation_history=history,
            policy_bundle=self.policy_bundle,
            final_booking_summary=final_booking_summary,
        )
        return float(result.reward)

    def compute_terminal(
        self,
        state: Optional["ConversationState"],
        final_booking_summary: Optional[str],
    ) -> float:
        """Expose terminal reward computation for training pipelines."""
        hallucination_reward = self.__compute_hallucination_reward(state, final_booking_summary)
        policy_reward = self.__compute_policy_reward(final_booking_summary)
        combined_reward = min(hallucination_reward, policy_reward)

        self.latest_reward_components = {
            "hallucination": float(hallucination_reward),
            "policy": float(policy_reward),
            "combined": float(combined_reward),
        }
        if self._process_reward_buffer:
            process_total = float(sum(self._process_reward_buffer))
            process_mean = process_total / len(self._process_reward_buffer)
            self.latest_reward_components.update(
                {
                    "process_total": process_total,
                    "process_mean": float(process_mean),
                    "process_last": float(self._process_reward_buffer[-1]),
                }
            )
        else:
            self.latest_reward_components.update(
                {"process_total": 0.0, "process_mean": 0.0, "process_last": 0.0}
            )
        self._process_reward_buffer.clear()
        return combined_reward
        
