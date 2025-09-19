#!/usr/bin/env python3

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

        self.logger = logging.getLogger(__name__)
        self.booking_agent = None
        self.verifier_agent = None
        self.current_state = None
        self.conversation_count = 0
        self.total_rewards = []
        self.current_objective = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the environment and agents"""
        # Initialize booking agent
        chat_config_path = self.booking_agent_config['config_path']
        db_path = self.booking_agent_config['db_path']
        server_path = self.booking_agent_config['server_path']
        self.booking_agent = FlightBookingChatAgent(chat_config_path, db_path, server_path)
        await self.booking_agent.initialize()
        # Initialize verifier agent
        verifier_config_path = self.verifier_config['config_path']
        self.verifier_agent = BookingVerifierAgent(verifier_config_path)
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

    async def step(self, user_action: str) -> EnvironmentStep:
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
        booking_response = await self.booking_agent._process_user_message(user_action)
        # Update state
        self.current_state.turn_count += 1
        if self.booking_agent:
            self.current_state.booking_context = self.booking_agent.booking_context.copy()
        # Check if conversation should terminate
        done = self._should_terminate(user_action, booking_response)
        self.current_state.is_terminated = done
        # Create info dict
        info = {
            "turn_count": self.current_state.turn_count,
            "booking_objective": self.current_state.booking_objective,
            "conversation_length": len(self.booking_agent.conversation_history)
        }

        return EnvironmentStep(
            state=self.current_state,
            done=done,
            info=info
        )

    async def compute_reward(self) -> float:
        """
        Compute reward for the complete trajectory/conversation

        This should be called after the trajectory is complete (done=True).
        It evaluates the entire conversation based on how well the booking
        objective was achieved.

        Returns:
            Reward value as float
        """
        # Check if conversation is actually complete
        if not self.current_state or not self.current_state.is_terminated:
            self.logger.warning("compute_trajectory_reward called but trajectory not complete")
            return 0.0
        # Get reward from verifier agent if available
        if self.verifier_agent:
            return await self._get_verifier_reward()
        # If no verifier, return a default reward
        # Could be extended with other reward mechanisms
        return 0

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


    async def _get_verifier_reward(self) -> float:
        """Get reward from verifier agent with graduated rewards"""

        # Get verification report using booking agent's conversation history
        verification_report = await self.verifier_agent.verify_bookings(
            self.booking_agent.conversation_history,
            self.booking_agent.mcp_client
        )
        # Calculate reward based on verification
        summary = verification_report.get('summary', {})
        total_claims = summary.get('total_claims', 0)
        verified_claims = summary.get('verified', 0)

        if total_claims == 0:
            # Check if conversation was productive even without booking
            if len(self.booking_agent.conversation_history) >= 4:
                return 0.1  # Small reward for engagement
            return 0  # Neutral if no booking claims and short conversation

        # Graduated rewards based on verification rate
        verification_rate = verified_claims / total_claims

        if verification_rate == 1.0:
            return 1.0  # Full reward for perfect verification
        elif verification_rate >= 0.8:
            return 0.7  # Good performance
        elif verification_rate >= 0.5:
            return 0.3  # Partial success
        elif verification_rate > 0:
            return 0.1  # Some progress made
        else:
            return -0.1  # Small penalty for complete failure


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