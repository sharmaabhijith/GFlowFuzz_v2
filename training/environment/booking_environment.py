#!/usr/bin/env python3

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "agents"))

from agents.chat.module import FlightBookingChatAgent
from agents.verifier.module import BookingVerifierAgent

@dataclass
class ConversationState:
    """Represents the current state of a conversation"""
    conversation_history: List[Dict[str, str]]
    booking_context: Dict[str, Any]
    turn_count: int
    is_terminated: bool = False
    booking_objective: str = ""

@dataclass
class EnvironmentStep:
    """Result of an environment step"""
    state: ConversationState
    reward: float
    done: bool
    info: Dict[str, Any]


class BookingConversationEnvironment:
    """
    Simplified environment for training user agents.
    Focuses on basic conversation flow with proper reward timing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the environment

        Args:
            config: Environment configuration dictionary
        """
        self.config = config
        self.max_conversation_length = config.get('max_conversation_length', 8)
        self.booking_agent_config = config.get('booking_agent_config')
        self.verifier_config = config.get('verifier_config')
        self.logger = logging.getLogger(__name__)
        self.booking_agent = None
        self.verifier_agent = None
        self.current_state = None
        self.conversation_count = 0
        self.total_rewards = []

    async def initialize(self):
        """Initialize the environment and agents"""
        chat_config_path = self.booking_agent_config['config_path']
        db_path = self.booking_agent_config['db_path']
        server_path = self.booking_agent_config['server_path']
        self.booking_agent = FlightBookingChatAgent(chat_config_path, db_path, server_path)
        await self.booking_agent.initialize()
        if self.verifier_config:
            verifier_config_path = self.verifier_config['config_path']
            self.verifier_agent = BookingVerifierAgent(verifier_config_path)
        self.logger.info("Environment initialized successfully")

    def reset(self, booking_objective: Optional[str] = None) -> ConversationState:
        """
        Reset the environment for a new conversation

        Args:
            booking_objective: Optional specific booking objective for this conversation

        Returns:
            Initial conversation state
        """
        if booking_objective is None:
            booking_objective = self._generate_simple_booking_objective()
        if self.booking_agent:
            self.booking_agent.conversation_history = []
            self.booking_agent.booking_context = {
                "summary": "",
                "current_requirements": {},
                "search_history": [],
                "preferences": {}
            }
        self.current_state = ConversationState(
            conversation_history=[],
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
            EnvironmentStep containing new state, reward, done flag, and info
        """
        if self.current_state.is_terminated:
            raise ValueError("Environment is terminated. Call reset() to start a new conversation.")
        self.current_state.conversation_history.append({
            "role": "user",
            "content": user_action
        })
        booking_response = await self.booking_agent._process_user_message(user_action)
        self.current_state.conversation_history.append({
            "role": "assistant",
            "content": booking_response
        })
        self.current_state.turn_count += 1
        if self.booking_agent:
            self.current_state.booking_context = self.booking_agent.booking_context.copy()
        done = self._should_terminate(user_action, booking_response)
        self.current_state.is_terminated = done
        reward = self._calculate_simple_reward(user_action, booking_response, done)
        info = {
            "turn_count": self.current_state.turn_count,
            "booking_objective": self.current_state.booking_objective,
            "conversation_length": len(self.current_state.conversation_history),
        }

        return EnvironmentStep(
            state=self.current_state,
            reward=reward,
            done=done,
            info=info
        )

    def _should_terminate(self, user_action: str, booking_response: str) -> bool:
        """Determine if the conversation should terminate"""
        user_wants_to_quit = any(word in user_action.lower() for word in ['quit', 'exit', 'bye', 'goodbye', 'thank you'])
        booking_complete = any(indicator in booking_response.lower() for indicator in [
            'booking confirmed', 'reservation complete', 'booking reference',
            'confirmation number', 'all set', 'booking successful'
        ])
        max_length_reached = self.current_state.turn_count >= self.max_conversation_length
        return user_wants_to_quit or booking_complete or max_length_reached

    def _calculate_simple_reward(self, user_action: str, booking_response: str, done: bool) -> float:
        """Calculate simplified reward for training"""
        reward = 0.0
        if self._is_natural_response(user_action):
            reward += 0.1
        if done:
            length = self.current_state.turn_count
            if length < 2:
                reward -= 0.5  # Too short
            elif length > self.max_conversation_length:
                reward -= 0.3  # Too long
            else:
                reward += 0.5
        return reward

    def _is_natural_response(self, user_action: str) -> bool:
        """Check if user response seems natural"""
        # Simple heuristics for natural responses
        natural_patterns = [
            'please', 'thank', 'could', 'would', 'i need', 'i want',
            'looking for', 'help me', 'can you'
        ]
        user_lower = user_action.lower()
        word_count = len(user_action.split())
        good_length = 2 <= word_count <= 15
        has_natural_pattern = any(pattern in user_lower for pattern in natural_patterns)
        return good_length and (has_natural_pattern or word_count >= 3)

    def _generate_simple_booking_objective(self) -> str:
        """Generate a simple booking objective for training"""
        import random

        origins = ["New York", "Los Angeles", "Chicago", "Miami", "Boston"]
        destinations = ["London", "Paris", "Tokyo", "Sydney", "Dubai"]
        months = ["March", "April", "May", "June", "July"]

        origin = random.choice(origins)
        destination = random.choice([d for d in destinations if d != origin])
        month = random.choice(months)

        objectives = [
            f"I need to fly from {origin} to {destination} in {month} 2026",
            f"Looking for a ticket to {destination} from {origin}",
            f"I want to travel to {destination} next month"
        ]

        return random.choice(objectives)