#!/usr/bin/env python3

import os
import asyncio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "training"))

@dataclass
class AutoUserConfig:
    """Simplified configuration for the auto user agent"""
    model_name: str = "gpt2"
    tokenizer_name: str = "gpt2"
    max_length: int = 30
    temperature: float = 0.7
    do_sample: bool = True
    device: str = "auto"

class AutoUserAgent:
    """
    Simplified trainable user agent for PPO training.
    Compatible with HuggingFace TRL PPO trainer.
    """

    def __init__(self, config: AutoUserConfig):
        """Initialize the auto user agent"""
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.tokenizer = None
        self.model = None
        self.conversation_history: List[Dict[str, str]] = []
        self.booking_objective: str = ""

    def initialize_model(self, model_path: Optional[str] = None, use_value_head: bool = False):
        """
        Initialize the language model

        Args:
            model_path: Path to a saved model, if None uses base model
            use_value_head: Whether to use model with value head for PPO
        """
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if model_path:
            if use_value_head:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            if use_value_head:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)

        # Move to device
        self.model.to(self.device)

    def set_training_mode(self, training: bool = True):
        """Set the model to training or evaluation mode"""
        if self.model:
            self.model.train(training)

    def reset_conversation(self, booking_objective: str):
        """Reset for a new conversation"""
        self.conversation_history = []
        self.booking_objective = booking_objective

    def update_conversation_history(self, role: str, message: str):
        """Update conversation history"""
        self.conversation_history.append({"role": role, "content": message})

    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        context_parts = [f"Objective: {self.booking_objective}"]
        recent_history = self.conversation_history[-100:] if len(self.conversation_history) > 100 else self.conversation_history
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Agent"
            context_parts.append(f"{role}: {msg['content']}")

        context_parts.append("User:")
        return "\n".join(context_parts)

    async def generate_response(self,
                              assistant_message: Optional[str] = None,
                              context: Optional[str] = None) -> str:
        """
        Generate a user response

        Args:
            assistant_message: Latest message from assistant (if any)
            context: Optional context override

        Returns:
            Generated user response
        """
        # Update conversation if assistant message provided
        if assistant_message:
            self.update_conversation_history("assistant", assistant_message)
        if context is None:
            context = self.get_conversation_context()
        inputs = self.tokenizer.encode(
            context,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.ne(self.tokenizer.pad_token_id)
            )
        user_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.update_conversation_history("user", user_response)
        return user_response

    def get_model_logits(self, text: str) -> torch.Tensor:
        """
        Get model logits for given text

        Args:
            text: Input text

        Returns:
            Model logits
        """
        inputs = self.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            return outputs.logits

    def save_model(self, save_path: str):
        """Save the trained model"""
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_model(self, model_path: str, use_value_head: bool = False):
        """Load a trained model"""
        if use_value_head:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)

    def get_parameters(self):
        """Get model parameters for optimization"""
        if self.model:
            return self.model.parameters()
        return []

    def prepare_training_data(self, conversations: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Prepare conversation data for training (simplified)

        Args:
            conversations: List of conversation dictionaries

        Returns:
            Prepared training data
        """
        states = []
        actions = []
        rewards = []
        for conv in conversations:
            conv_states = conv.get('states', [])
            conv_actions = conv.get('actions', [])
            conv_rewards = conv.get('rewards', [])
            states.extend(conv_states)
            actions.extend(conv_actions)
            rewards.extend(conv_rewards)
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }

    async def get_verifier_reward(self, conversation_history: List[Dict[str, str]], verifier_agent, mcp_client) -> float:
        """
        Get reward from verifier agent based on conversation validation

        Args:
            conversation_history: List of conversation messages
            verifier_agent: BookingVerifierAgent instance
            mcp_client: MCP client for database verification

        Returns:
            Reward (1.0 for valid, 0.0 for hallucinated)
        """
        try:
            # Get verification report from verifier agent
            verification_report = await verifier_agent.verify_bookings(conversation_history, mcp_client)

            if not verification_report.get('verification_complete', False):
                return 0.5  # Neutral reward if verification fails

            summary = verification_report.get('summary', {})
            total_claims = summary.get('total_claims', 0)
            verified_claims = summary.get('verified', 0)
            not_found_claims = summary.get('not_found', 0)

            # If no claims made, give neutral reward
            if total_claims == 0:
                return 0.5

            # Calculate verification rate
            verification_rate = verified_claims / total_claims

            # Binary reward: 1.0 if all claims verified, 0.0 if any hallucinations
            if verification_rate == 1.0:
                return 1.0
            else:
                return 0.0

        except Exception as e:
            # Return neutral reward on error
            return 0.5

    async def evaluate_on_objective(self, booking_objective: str, max_turns: int = 5) -> Dict[str, Any]:
        """
        Evaluate the agent on a specific booking objective

        Args:
            booking_objective: The booking goal
            max_turns: Maximum number of conversation turns

        Returns:
            Evaluation results
        """
        self.reset_conversation(booking_objective)
        conversation = []
        for turn in range(max_turns):
            if turn == 0:
                user_msg = booking_objective
            else:
                user_msg = await self.generate_response()
            conversation.append({"role": "user", "content": user_msg})
            if any(word in user_msg.lower() for word in ['quit', 'exit', 'thank']):
                break
        return {
            "objective": booking_objective,
            "conversation": conversation,
            "turns": len(conversation)
        }


class AutoUserPPOWrapper:
    """Simple wrapper for PPO training compatibility"""

    def __init__(self, auto_user: AutoUserAgent):
        self.auto_user = auto_user

    async def act(self, state: str) -> str:
        """Generate action for PPO"""
        return await self.auto_user.generate_response(context=state)

    def get_parameters(self):
        """Get model parameters"""
        return self.auto_user.get_parameters()