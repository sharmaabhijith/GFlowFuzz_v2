import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import random

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "agents"))

from agents.chat.module import FlightBookingChatAgent
from agents.verifier.module import BookingVerifierAgent
from agents.coder.module import SQLCoderAgent


@dataclass
class ConversationState:
    """Represents the current state of a conversation"""
    booking_context: Dict[str, Any]
    turn_count: int
    is_terminated: bool = False
    booking_objective: str = ""


@dataclass
class EnvironmentStep:
    """Result of an environment step"""
    state: ConversationState
    done: bool
    info: Dict[str, Any]


class BookingConversationEnvironment:
    """
    Unified environment for training user agents in conversation flow.
    Combines functionality from both original environment files.
    """

    def __init__(self, config: Dict[str, Any], auto_user_config=None):
        """
        Initialize the environment

        Args:
            config: Environment configuration dictionary
            auto_user_config: Configuration for auto user agent (optional)
        """
        self.config = config
        self.auto_user_config = auto_user_config
        self.max_conversation_length = config.get('max_conversation_length', 8)
        self.booking_agent_config = config.get('booking_agent_config')
        self.verifier_config = config.get('verifier_config')
        self.coder_config = config.get('coder_config', {
            'config_path': 'agents/coder/config.yaml'
        })

        self.logger = logging.getLogger(__name__)
        self.booking_agent = None
        self.verifier_agent = None
        self.coder_agent = None
        self.current_state = None
        self.conversation_count = 0
        self.total_rewards = []
        self.current_objective = None
        self.is_initialized = False
        self.final_booking_summary = None

    def initialize(self):
        """Initialize the environment and agents"""
        # Initialize booking agent
        chat_config_path = self.booking_agent_config['config_path']
        db_path = self.booking_agent_config['db_path']
        server_path = self.booking_agent_config['server_path']
        self.booking_agent = FlightBookingChatAgent(chat_config_path, db_path, server_path)
        self.booking_agent.initialize()
        # Initialize verifier agent
        verifier_config_path = self.verifier_config['config_path']
        self.verifier_agent = BookingVerifierAgent(verifier_config_path)
        # Initialize coder agent
        coder_config_path = self.coder_config['config_path']
        self.coder_agent = SQLCoderAgent(coder_config_path)
        self.is_initialized = True

    def reset(self, booking_objective: Optional[str] = None) -> ConversationState:
        """
        Reset the environment for a new conversation

        Args:
            booking_objective: Optional specific booking objective for this conversation

        Returns:
            Initial conversation state
        """
        if not self.is_initialized:
            raise RuntimeError("Environment not initialized. Call initialize() first.")
        # Generate or use provided objective
        if booking_objective is None:
            booking_objective = self._generate_booking_objective()
        self.current_objective = booking_objective
        # Reset final booking summary
        self.final_booking_summary = None
        # Reset booking agent
        if self.booking_agent:
            self.booking_agent.conversation_history = []
            self.booking_agent.booking_context = {
                "summary": "",
                "current_requirements": {},
                "search_history": [],
                "preferences": {}
            }
        # Create initial conversation state
        self.current_state = ConversationState(
            booking_context={},
            turn_count=0,
            is_terminated=False,
            booking_objective=booking_objective
        )
        return self.current_state

    def step(self, user_action: str) -> EnvironmentStep:
        """
        Execute a step in the environment

        Args:
            user_action: The user's message/action

        Returns:
            EnvironmentStep containing new state, done flag, and info
        """
        if self.current_state.is_terminated:
            raise ValueError("Environment is terminated. Call reset() to start a new conversation.")

        # Get booking agent response (it handles conversation history internally)
        booking_response = self.booking_agent._process_user_message(user_action)
        # Update state
        self.current_state.turn_count += 1
        if self.booking_agent:
            self.current_state.booking_context = self.booking_agent.booking_context.copy()
        # Check if conversation should terminate
        done = self._should_terminate(user_action, booking_response)
        self.current_state.is_terminated = done

        # Generate booking summary when conversation ends
        if done and not self.final_booking_summary:
            try:
                # Generate the final booking summary
                self.final_booking_summary = self.booking_agent._generate_booking_summary()
            except Exception as e:
                self.logger.warning(f"Could not generate booking summary: {e}")
                self.final_booking_summary = None

        # Create info dict
        info = {
            "turn_count": self.current_state.turn_count,
            "booking_objective": self.current_state.booking_objective,
            "conversation_length": len(self.booking_agent.conversation_history),
            "booking_summary": self.final_booking_summary if done else None
        }

        return EnvironmentStep(
            state=self.current_state,
            done=done,
            info=info
        )

    def compute_halluciination_reward(self) -> float:
        """
        Compute reward for the complete trajectory/conversation

        This checks if the final booking exists in the database:
        - Reward = 1 if booking doesn't exist (new booking)
        - Reward = 0 if booking already exists or no booking was made

        Returns:
            Reward value as float
        """
        # Check if conversation is actually complete
        if not self.current_state or not self.current_state.is_terminated:
            self.logger.warning("compute_trajectory_reward called but trajectory not complete")
            return 0.0
        # Get reward from verifier agent using final booking only
        if self.verifier_agent:
            return self._get_verifier_reward()
        # If no verifier, return a default reward
        return 0

    def compute_process_reward(self, state: ConversationState, action: str, next_state: ConversationState) -> float:
        """
        Compute process reward for a single step transition.

        Process rewards provide immediate feedback during the conversation,
        helping the agent learn good conversation patterns even before reaching
        the terminal state.

        Args:
            state: Current conversation state
            action: Action taken (user message)
            next_state: Resulting state after action

        Returns:
            Process reward as float
        """
        reward = 0.0

        # TODO: Implement process reward logic
        # Example ideas for process rewards:
        # - Reward for maintaining conversation flow
        # - Reward for providing relevant information
        # - Penalty for repetitive responses
        # - Reward for progressing toward booking completion
        # - Penalty for conversation going off-topic

        # Placeholder implementation - to be filled with actual logic
        # For now, return a small positive reward for each step to encourage conversation
        reward = 0.01

        return reward

    def compute_terminal_reward(self) -> float:
        """
        Compute terminal reward at the end of conversation.

        This checks if the final booking exists in the database:
        - Reward = 1 if booking doesn't exist (new booking)
        - Reward = 0 if booking already exists or no booking was made

        Returns:
            Terminal reward as float
        """
        return await self._get_verifier_reward()

    def compute_shaped_rewards(self, terminal_reward: float, num_steps: int, gamma: float = 0.99) -> List[float]:
        """
        Compute shaped rewards with proper credit assignment.

        Uses exponential decay to assign credit to earlier actions,
        ensuring that actions closer to success receive more credit.

        Args:
            terminal_reward: The final reward at episode completion
            num_steps: Number of steps in the episode
            gamma: Discount factor (default 0.99)

        Returns:
            List of shaped rewards for each step
        """
        shaped_rewards = []

        for i in range(num_steps):
            # Calculate discounted reward for each step
            # Steps closer to success get more credit
            steps_from_end = num_steps - 1 - i
            discounted_reward = terminal_reward * (gamma ** steps_from_end)
            shaped_rewards.append(discounted_reward)

        return shaped_rewards

    def _format_state(self, conversation_state: ConversationState, use_optimized: bool = True) -> str:
        """
        Format conversation state for stateless policy with optimization options.

        Args:
            conversation_state: Current state of the conversation
            use_optimized: If True, use more efficient formatting

        Returns:
            Formatted string containing complete context for policy
        """
        if use_optimized:
            # More efficient state representation
            # Use a structured format that's easier to parse
            state_parts = []

            # Compact objective representation
            state_parts.append(f"[GOAL] {conversation_state.booking_objective}")

            # Efficient conversation history
            # Only include last N messages for very long conversations
            history = self.booking_agent.conversation_history
            max_history_length = 20  # Keep last 10 exchanges

            if len(history) > max_history_length:
                # Add summary of earlier conversation
                state_parts.append(f"[CONTEXT] Earlier: {len(history) - max_history_length} messages exchanged")
                history = history[-max_history_length:]

            # Compact history format
            if history:
                state_parts.append("[DIALOG]")
                for msg in history:
                    prefix = "U:" if msg["role"] == "user" else "A:"
                    # Truncate very long messages
                    content = msg['content']
                    if len(content) > 200:
                        content = content[:197] + "..."
                    state_parts.append(f"{prefix} {content}")
            else:
                state_parts.append("[DIALOG] <start>")

            # Compact context representation
            if conversation_state.booking_context:
                ctx = conversation_state.booking_context
                if ctx.get("current_requirements"):
                    reqs = ctx['current_requirements']
                    if reqs:
                        # Only include non-empty requirements
                        req_str = ', '.join(f"{k}:{v}" for k, v in reqs.items() if v)
                        if req_str:
                            state_parts.append(f"[STATUS] {req_str}")

            # Metadata
            state_parts.append(f"[TURN] {conversation_state.turn_count + 1}")
            state_parts.append("[NEXT] User:")

            return "\n".join(state_parts)
        else:
            # Original verbose format (fallback)
            parts = []
            parts.append(f"Objective: {conversation_state.booking_objective}")
            parts.append("")

            if self.booking_agent.conversation_history:
                parts.append("Conversation History:")
                for msg in self.booking_agent.conversation_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    parts.append(f"{role}: {msg['content']}")
            else:
                parts.append("Conversation History: [Start of conversation]")

            if conversation_state.booking_context and any(conversation_state.booking_context.values()):
                parts.append("")
                parts.append("Current Booking Status:")
                if conversation_state.booking_context.get("summary"):
                    parts.append(f"Summary: {conversation_state.booking_context['summary']}")
                if conversation_state.booking_context.get("current_requirements"):
                    reqs = conversation_state.booking_context['current_requirements']
                    if reqs:
                        parts.append(f"Requirements: {reqs}")

            parts.append("")
            parts.append(f"Turn: {conversation_state.turn_count + 1}")
            parts.append("")
            parts.append("User:")

            return "\n".join(parts)

    def _should_terminate(self, user_action: str, booking_response: str) -> bool:
        """Determine if the conversation should terminate"""
        # More precise termination conditions
        # Check for explicit quit commands (not just "thank you")
        user_wants_to_quit = any(phrase in user_action.lower() for phrase in [
            'quit', 'exit', 'bye', 'goodbye',
            'stop', 'end conversation', 'that\'s all',
            'no more', 'i\'m done', 'cancel'
        ])

        # Check if "thank you" is used as a closing statement (with context)
        thank_you_closing = (
            'thank you' in user_action.lower() and
            any(word in user_action.lower() for word in ['bye', 'all', 'done', 'help', 'booking'])
        )

        # Check for successful booking completion
        booking_complete = any(indicator in booking_response.lower() for indicator in [
            'booking confirmed', 'reservation complete', 'booking reference',
            'confirmation number', 'all set', 'booking successful',
            'transaction completed', 'payment processed'
        ])

        # Check max length
        max_length_reached = self.current_state.turn_count >= self.max_conversation_length

        return user_wants_to_quit or thank_you_closing or booking_complete or max_length_reached


    def _get_verifier_reward(self) -> float:
        """Get reward from verifier agent - checks if final booking exists in database"""

        # If no booking summary was generated, no reward
        if not self.final_booking_summary:
            self.logger.info("No booking summary generated - no reward")
            return 0

        # Use the new verification method that only checks final booking
        verification_report = self.verifier_agent.verify_final_booking_only(
            self.final_booking_summary,
            self.booking_agent.mcp_client,
            self.coder_agent
        )

        # Return the reward directly from the verification report
        reward = verification_report.get('reward', 0)

        # Log the verification result
        if verification_report.get('verification_complete'):
            if verification_report.get('booking_exists'):
                self.logger.info(f"Booking exists in database - reward: {reward}")
            else:
                self.logger.info(f"New booking (not in database) - reward: {reward}")
        else:
            self.logger.warning(f"Verification failed: {verification_report.get('error', 'Unknown error')}")

        return reward


    def _generate_booking_objective(self) -> str:
        """Generate random booking objective"""
        templates = [
            "I need to fly from {origin} to {destination} in {month} 2026",
            "Looking for a {class_type} class ticket from {origin} to {destination}",
            "I want to book a flight to {destination} for {num} people",
            "Need a direct flight from {origin} to {destination} next {month}",
            "Can you help me find flights to {destination} in {month}?",
            "I'm planning a trip to {destination} from {origin}",
            "Looking for the cheapest flight to {destination}",
            "I need to travel to {destination} for business in {month}"
        ]

        origins = ["New York", "Los Angeles", "Chicago", "Miami", "Boston",
                  "Seattle", "San Francisco", "Denver", "Dallas", "Atlanta"]
        destinations = ["London", "Paris", "Tokyo", "Sydney", "Dubai",
                       "Singapore", "Hong Kong", "Rome", "Barcelona", "Amsterdam"]
        months = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        class_types = ["economy", "business", "first"]
        nums = ["2", "3", "4"]

        template = random.choice(templates)
        objective = template.format(
            origin=random.choice(origins),
            destination=random.choice(destinations),
            month=random.choice(months),
            class_type=random.choice(class_types),
            num=random.choice(nums)
        )

        return objective