from __future__ import annotations
from typing import Any, Dict, List, Optional

from agents.chat.module import FlightBookingChatAgent
from agents.coder.module import SQLCoderAgent
from agents.verifier.module import BookingVerifierAgent
from agents.judge.module import PolicyJudgeAgent


class Oracle:
    """Encapsulates reward computations for the booking conversation environment."""

    def __init__(
        self,
        booking_agent: Optional["FlightBookingChatAgent"],
        verifier_agent: Optional["BookingVerifierAgent"],
        coder_agent: Optional["SQLCoderAgent"],
        judge_agent: Optional["PolicyJudgeAgent"] = None,
        policy_bundle: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store references to the agents required for reward computation."""
        self.booking_agent = booking_agent
        self.verifier_agent = verifier_agent
        self.coder_agent = coder_agent
        self.judge_agent = judge_agent
        self.policy_bundle = policy_bundle
        self.latest_verification_report: Optional[Dict[str, Any]] = None

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

    def __compute_hallucination_reward(
        self,
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
        final_booking_summary: Optional[str],
    ) -> float:
        """Expose terminal reward computation for training pipelines."""
        hallucination_reward = self.__compute_hallucination_reward(final_booking_summary)
        policy_reward = self.__compute_policy_reward(final_booking_summary)
        combined_reward = max(hallucination_reward, policy_reward)

        return combined_reward
        
