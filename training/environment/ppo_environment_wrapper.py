#!/usr/bin/env python3

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
import random
import logging

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from training.environment.booking_environment import BookingConversationEnvironment, ConversationState
from agents.auto_user.module import AutoUserAgent, AutoUserConfig

logger = logging.getLogger(__name__)


@dataclass
class PPOExperience:
    """Single experience for PPO training"""
    state: str  # Conversation context
    action: str  # User response
    reward: float
    next_state: str
    done: bool
    info: Dict[str, Any]


class PPOEnvironmentWrapper:
    """
    Wrapper to make the booking environment compatible with PPO training.
    Handles conversation state management and reward computation.
    """

    def __init__(self, env_config: Dict[str, Any], auto_user_config: AutoUserConfig):
        """
        Initialize PPO environment wrapper

        Args:
            env_config: Configuration for booking environment
            auto_user_config: Configuration for auto user agent
        """
        self.env_config = env_config
        self.auto_user_config = auto_user_config
        self.booking_env = BookingConversationEnvironment(env_config)
        self.auto_user = None
        self.current_objective = None
        self.conversation_buffer = []
        self.is_initialized = False

    async def initialize(self):
        """Initialize the environment and agents"""
        await self.booking_env.initialize()
        self.auto_user = AutoUserAgent(self.auto_user_config)
        self.is_initialized = True
        logger.info("PPO environment wrapper initialized")

    def reset(self, objective: Optional[str] = None) -> str:
        """
        Reset environment for new episode

        Args:
            objective: Optional booking objective

        Returns:
            Initial state (conversation context)
        """
        if not self.is_initialized:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Generate or use provided objective
        if objective is None:
            objective = self._generate_booking_objective()

        self.current_objective = objective
        self.conversation_buffer = []

        # Reset booking environment
        conversation_state = self.booking_env.reset(objective)

        # Reset auto user agent
        self.auto_user.reset_conversation(objective)

        # Return initial context as state
        return self._format_state(conversation_state)

    async def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute environment step with user action

        Args:
            action: User response/action

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute step in booking environment
        env_step = await self.booking_env.step(action)

        # Update conversation buffer
        self.conversation_buffer.append({
            "role": "user",
            "content": action
        })
        if env_step.state.conversation_history:
            last_assistant = env_step.state.conversation_history[-1]
            if last_assistant["role"] == "assistant":
                self.conversation_buffer.append(last_assistant)

        # Format next state
        next_state = self._format_state(env_step.state)

        # Compute immediate reward
        immediate_reward = self._compute_immediate_reward(action, env_step)

        # If conversation is done, get verifier reward
        final_reward = immediate_reward
        if env_step.done and self.booking_env.verifier_agent:
            verifier_reward = await self._get_verifier_reward()
            final_reward = self._combine_rewards(immediate_reward, verifier_reward, env_step.done)

        info = env_step.info
        info["immediate_reward"] = immediate_reward
        info["final_reward"] = final_reward

        return next_state, final_reward, env_step.done, info

    def _format_state(self, conversation_state: ConversationState) -> str:
        """Format conversation state for model input"""
        parts = [f"Objective: {conversation_state.booking_objective}"]

        # Add conversation history
        for msg in conversation_state.conversation_history[-10:]:  # Last 10 messages
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")

        # Add prompt for next user response
        parts.append("User:")

        return "\n".join(parts)

    def _compute_immediate_reward(self, action: str, env_step) -> float:
        """Compute immediate reward for the action"""
        reward = 0.0

        # Length reward
        action_length = len(action.split())
        if 3 <= action_length <= 20:
            reward += 0.1
        elif action_length < 3:
            reward -= 0.2
        else:
            reward -= 0.1

        # Naturalness reward
        if self._is_natural_response(action):
            reward += 0.2

        # Progress reward (conversation moving forward)
        if not env_step.done:
            reward += 0.05
        else:
            # Ending reward
            if self._is_polite_ending(action):
                reward += 0.3

        return reward

    def _is_natural_response(self, response: str) -> bool:
        """Check if response seems natural"""
        natural_patterns = [
            'please', 'thank', 'could', 'would', 'i need', 'i want',
            'looking for', 'help me', 'can you', "i'd like", "i'm"
        ]
        response_lower = response.lower()
        has_pattern = any(pattern in response_lower for pattern in natural_patterns)
        proper_length = 3 <= len(response.split()) <= 30
        return has_pattern and proper_length

    def _is_polite_ending(self, response: str) -> bool:
        """Check if response is a polite ending"""
        ending_patterns = ['thank', 'thanks', 'perfect', 'great', 'excellent']
        quit_patterns = ['quit', 'exit', 'bye', 'goodbye']
        response_lower = response.lower()

        has_thanks = any(pattern in response_lower for pattern in ending_patterns)
        has_quit = any(pattern in response_lower for pattern in quit_patterns)

        return has_thanks and has_quit

    async def _get_verifier_reward(self) -> float:
        """Get reward from verifier agent"""
        if not self.booking_env.verifier_agent:
            return 0.5

        try:
            # Get verification report
            verification_report = await self.booking_env.verifier_agent.verify_bookings(
                self.conversation_buffer,
                self.booking_env.booking_agent.mcp_client
            )

            # Calculate reward based on verification
            summary = verification_report.get('summary', {})
            total_claims = summary.get('total_claims', 0)
            verified_claims = summary.get('verified', 0)

            if total_claims == 0:
                return 0.5  # Neutral if no booking claims

            # Binary reward for verification
            verification_rate = verified_claims / total_claims
            return 1.0 if verification_rate == 1.0 else 0.0

        except Exception as e:
            logger.error(f"Error getting verifier reward: {e}")
            return 0.5

    def _combine_rewards(self, immediate: float, verifier: float, done: bool) -> float:
        """Combine immediate and verifier rewards"""
        if not done:
            return immediate

        # Weight the rewards
        immediate_weight = 0.3
        verifier_weight = 0.7

        return immediate * immediate_weight + verifier * verifier_weight

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

    async def collect_episode(self, model=None, tokenizer=None, max_steps: int = 10) -> List[PPOExperience]:
        """
        Collect a full episode of experiences

        Args:
            model: Optional model to use for action generation
            tokenizer: Optional tokenizer
            max_steps: Maximum steps per episode

        Returns:
            List of PPO experiences
        """
        experiences = []

        # Reset environment
        state = self.reset()

        # If model provided, update auto user's model
        if model and tokenizer:
            self.auto_user.model = model
            self.auto_user.tokenizer = tokenizer

        done = False
        step = 0

        while not done and step < max_steps:
            # Generate action
            action = await self.auto_user.generate_response(context=state)

            # Take environment step
            next_state, reward, done, info = await self.step(action)

            # Store experience
            experience = PPOExperience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info
            )
            experiences.append(experience)

            # Update state
            state = next_state
            step += 1

        return experiences

    async def evaluate_model(self, model, tokenizer, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        total_rewards = []
        verification_rates = []
        conversation_lengths = []

        for _ in range(num_episodes):
            experiences = await self.collect_episode(model, tokenizer)

            # Calculate metrics
            episode_reward = sum(exp.reward for exp in experiences)
            total_rewards.append(episode_reward)

            conversation_lengths.append(len(experiences))

            # Get verification rate from last experience info
            if experiences:
                last_info = experiences[-1].info
                if "final_reward" in last_info:
                    # Verifier reward of 1.0 means all claims verified
                    verification_rates.append(1.0 if last_info["final_reward"] >= 0.7 else 0.0)

        metrics = {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "avg_length": np.mean(conversation_lengths),
            "verification_rate": np.mean(verification_rates) if verification_rates else 0.0
        }

        return metrics


class PPOBatchCollector:
    """Collects batches of experiences for PPO training"""

    def __init__(self, env_wrapper: PPOEnvironmentWrapper):
        """
        Initialize batch collector

        Args:
            env_wrapper: PPO environment wrapper
        """
        self.env_wrapper = env_wrapper

    async def collect_batch(self, batch_size: int, model=None, tokenizer=None) -> Dict[str, List]:
        """
        Collect batch of experiences

        Args:
            batch_size: Number of experiences to collect
            model: Optional model for action generation
            tokenizer: Optional tokenizer

        Returns:
            Batch data for PPO training
        """
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []  # Value estimates if needed

        episodes_needed = batch_size // 5  # Estimate episodes needed

        for _ in range(episodes_needed):
            experiences = await self.env_wrapper.collect_episode(model, tokenizer)

            for exp in experiences:
                all_states.append(exp.state)
                all_actions.append(exp.action)
                all_rewards.append(exp.reward)
                all_dones.append(exp.done)

                # Break if we have enough
                if len(all_states) >= batch_size:
                    break

            if len(all_states) >= batch_size:
                break

        # Trim to exact batch size
        batch_data = {
            "states": all_states[:batch_size],
            "actions": all_actions[:batch_size],
            "rewards": all_rewards[:batch_size],
            "dones": all_dones[:batch_size]
        }

        return batch_data