#!/usr/bin/env python3
"""
Unified PPO Training System for Auto User Agent

This module provides a complete PPO training implementation that combines:
- Core PPO algorithm with HuggingFace TRL integration
- Training orchestration and conversation collection
- Environment interaction and reward calculation
- Verifier agent integration for accurate feedback
- Professional monitoring and checkpointing

Key Components:
- PPOConfig: Comprehensive configuration for all aspects of training
- PPOTrainer: Complete training system with algorithm and orchestration
- ConversationProcessor: Handles conversation data processing
- VerifierRewardCalculator: Integrates verifier agent feedback

Architecture:
- All-in-one solution combining algorithm and training logic
- Professional implementation following RL best practices
- Integration with booking environment and verifier agent
- Comprehensive logging, evaluation, and checkpointing
"""

import torch
import numpy as np
import asyncio
import json
import os
import logging
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# HuggingFace imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Add parent directories to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from environment.booking_environment import BookingConversationEnvironment


@dataclass
class PPOConfig:
    """
    Comprehensive configuration for PPO training system.

    This configuration covers all aspects of PPO training including:
    - Core PPO hyperparameters
    - Model and tokenizer settings
    - Training schedule and checkpointing
    - Environment and verifier integration
    - Performance optimization settings
    """

    # Model Configuration
    model_name: str = "gpt2"
    tokenizer_name: str = "gpt2"
    load_from_checkpoint: Optional[str] = None

    # Core PPO Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 2
    gamma: float = 0.99                    # Discount factor
    lam: float = 0.95                      # GAE lambda
    cliprange: float = 0.2                 # PPO clipping parameter
    vf_coef: float = 0.1                   # Value function coefficient
    ent_coef: float = 0.01                 # Entropy coefficient
    max_grad_norm: float = 0.5             # Gradient clipping

    # Training Schedule
    max_epochs: int = 10
    conversations_per_epoch: int = 20
    eval_conversations: int = 5
    save_frequency: int = 2
    eval_frequency: int = 1

    # Environment Configuration
    max_conversation_turns: int = 8
    max_response_length: int = 30
    max_context_length: int = 256

    # Verifier Integration
    use_verifier_rewards: bool = True
    reward_scale: float = 1.0

    # Performance Optimization
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    device: str = "auto"

    # Logging and Output
    output_dir: str = "./ppo_training_output"
    log_level: str = "INFO"
    log_interval: int = 5
    save_conversation_logs: bool = True


class ConversationProcessor:
    """
    Processes conversation data into format suitable for PPO training.

    This class handles:
    - Tokenization of conversation contexts and responses
    - Reward assignment and processing
    - Batch preparation for PPO updates
    """

    def __init__(self, tokenizer, config: PPOConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process_conversation(self, conversation_data: Dict[str, Any]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Process a single conversation into PPO training format.

        Args:
            conversation_data: Dictionary containing conversation history and rewards

        Returns:
            Tuple of (queries, responses, rewards) as tensors
        """
        queries = []
        responses = []
        rewards = []

        user_responses = conversation_data.get('user_responses', [])
        conversation_history = conversation_data.get('conversation_history', [])
        total_reward = conversation_data.get('total_reward', 0.0)

        # Process each user response in the conversation
        for i, user_response in enumerate(user_responses):
            # Build context up to this point
            context_history = conversation_history[:i*2]  # User and assistant alternate
            context = self._build_context(
                conversation_data.get('booking_objective', ''),
                context_history
            )

            # Tokenize query (context)
            query_tokens = self.tokenizer.encode(
                context,
                return_tensors="pt",
                max_length=self.config.max_context_length,
                truncation=True,
                padding=True
            ).squeeze()

            # Tokenize response (user action)
            response_tokens = self.tokenizer.encode(
                user_response,
                return_tensors="pt",
                max_length=self.config.max_response_length,
                truncation=True,
                padding=True
            ).squeeze()

            queries.append(query_tokens)
            responses.append(response_tokens)

            # Assign reward (scaled by configuration)
            scaled_reward = total_reward * self.config.reward_scale
            rewards.append(scaled_reward)

        return queries, responses, rewards

    def _build_context(self, objective: str, history: List[Dict[str, str]]) -> str:
        """Build context string from objective and conversation history"""
        context_parts = [f"Objective: {objective}"]

        for msg in history:
            role = "User" if msg["role"] == "user" else "Agent"
            context_parts.append(f"{role}: {msg['content']}")

        context_parts.append("User:")
        return "\n".join(context_parts)

    def process_batch(self, conversations: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Process multiple conversations into a batch for training.

        Args:
            conversations: List of conversation data dictionaries

        Returns:
            Tuple of (all_queries, all_responses, all_rewards)
        """
        all_queries = []
        all_responses = []
        all_rewards = []

        for conversation in conversations:
            queries, responses, rewards = self.process_conversation(conversation)
            all_queries.extend(queries)
            all_responses.extend(responses)
            all_rewards.extend(rewards)

        return all_queries, all_responses, all_rewards


class VerifierRewardCalculator:
    """
    Calculates rewards using verifier agent feedback.

    This class integrates the verifier agent to provide accurate reward signals
    based on booking information verification.
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def calculate_reward(self, conversation_history: List[Dict[str, str]],
                             verifier_agent, mcp_client) -> float:
        """
        Calculate reward using verifier agent feedback.

        Args:
            conversation_history: List of conversation messages
            verifier_agent: BookingVerifierAgent instance
            mcp_client: MCP client for database verification

        Returns:
            Reward value (1.0 for verified, 0.0 for hallucinated, 0 for neutral)
        """
        if not self.config.use_verifier_rewards or not verifier_agent:
            return 0  # Neutral reward when verifier disabled

        # Get verification report from verifier agent
        verification_report = await verifier_agent.verify_bookings(
            conversation_history, mcp_client
        )

        if not verification_report.get('verification_complete', False):
            return 0  # Neutral reward if verification fails

        summary = verification_report.get('summary', {})
        total_claims = summary.get('total_claims', 0)
        verified_claims = summary.get('verified', 0)

        # If no claims made, give neutral reward
        if total_claims == 0:
            return 0

        # Calculate verification rate
        verification_rate = verified_claims / total_claims

        # Binary reward: 1.0 if all claims verified, 0.0 if any hallucinations
        if verification_rate == 1.0:
            return 1.0
        else:
            return 0.0


class PPOTrainer:
    """
    Unified PPO training system for auto user agent.

    This class combines the PPO algorithm with training orchestration to provide
    a complete training solution including:
    - Environment interaction and conversation collection
    - PPO algorithm with HuggingFace TRL integration
    - Verifier agent reward calculation
    - Comprehensive evaluation and monitoring
    - Professional checkpointing and statistics tracking

    Key Features:
    - All-in-one training solution
    - Professional implementation following RL best practices
    - Integration with booking environment and verifier agent
    - Comprehensive logging, evaluation, and checkpointing
    """

    def __init__(self, config: PPOConfig, environment_config: Dict[str, Any], mcp_client=None):
        """
        Initialize the PPO trainer.

        Args:
            config: PPO training configuration
            environment_config: Environment configuration
            mcp_client: Optional MCP client for database verification
        """
        self.config = config
        self.environment_config = environment_config
        self.mcp_client = mcp_client or self._create_mock_mcp_client()

        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Setup logging
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_eval_reward = float('-inf')

        # Training statistics
        self.training_stats = {
            'epoch_rewards': [],
            'epoch_verification_rates': [],
            'epoch_conversation_lengths': [],
            'eval_rewards': [],
            'eval_verification_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'step_rewards': []
        }

        # Core components (initialized in setup)
        self.model = None
        self.tokenizer = None
        self.ppo_trainer = None
        self.environment = None
        self.verifier_agent = None
        self.conversation_processor = None
        self.reward_calculator = None

        self.logger.info(f"PPOTrainer initialized on {self.device}")

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # File handler for training logs
        log_file = os.path.join(self.config.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers if not already present
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def _create_mock_mcp_client(self):
        """Create mock MCP client for testing"""
        class MockMCPClient:
            async def query_database(self, query: str):
                class MockResult:
                    def __init__(self):
                        self.success = True
                        self.result = '{"results": [], "row_count": 0}'
                        self.error_message = None
                return MockResult()
        return MockMCPClient()

    async def initialize(self):
        """
        Initialize all training components.

        Initializes:
        - Model and tokenizer with value head for PPO
        - Conversation environment for data collection
        - Verifier agent for reward calculation
        - PPO trainer using HuggingFace TRL
        - Conversation processor and reward calculator
        """
        self.logger.info("Initializing PPO training components...")

        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "conversations"), exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model with value head
        if self.config.load_from_checkpoint:
            self.logger.info(f"Loading model from checkpoint: {self.config.load_from_checkpoint}")
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.load_from_checkpoint)
        else:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_name)

        self.model.to(self.device)

        # Initialize environment
        self.environment = BookingConversationEnvironment(self.environment_config)
        await self.environment.initialize()

        # Initialize verifier agent
        if self.config.use_verifier_rewards:
            from agents.verifier.module import BookingVerifierAgent
            verifier_config_path = self.environment_config.get('verifier_config', {}).get('config_path')
            if verifier_config_path:
                self.verifier_agent = BookingVerifierAgent(verifier_config_path)
                self.logger.info("Verifier agent initialized")
            else:
                self.logger.warning("Verifier rewards enabled but no config path provided")
                self.verifier_agent = None
        else:
            self.verifier_agent = None

        # Initialize processors
        self.conversation_processor = ConversationProcessor(self.tokenizer, self.config)
        self.reward_calculator = VerifierRewardCalculator(self.config)

        # Initialize PPO trainer
        await self._initialize_ppo_trainer()

        self.logger.info("All PPO training components initialized successfully")

    async def _initialize_ppo_trainer(self):
        """Initialize the HuggingFace TRL PPO trainer"""
        # Create TRL PPO configuration
        ppo_config = PPOConfig(
            model_name=self.config.model_name,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            gamma=self.config.gamma,
            lam=self.config.lam,
            cliprange=self.config.cliprange,
            vf_coef=self.config.vf_coef,
            ent_coef=self.config.ent_coef,
            max_grad_norm=self.config.max_grad_norm,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with=None,  # Disable external logging
        )

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.logger.info("HuggingFace TRL PPO trainer initialized successfully")

    async def train(self):
        """
        Execute the complete training pipeline.

        Training Pipeline:
        1. Collect conversations in environment using current policy
        2. Calculate verifier-based rewards for collected conversations
        3. Train PPO algorithm on collected data
        4. Evaluate periodically and save checkpoints
        5. Track comprehensive statistics throughout training
        """
        self.logger.info(f"Starting PPO training for {self.config.max_epochs} epochs")

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = datetime.now()

            self.logger.info(f"=== Epoch {epoch + 1}/{self.config.max_epochs} ===")

            # Training phase
            epoch_stats = await self._train_epoch()
            self._update_epoch_stats(epoch_stats)

            # Evaluation phase
            if epoch % self.config.eval_frequency == 0:
                eval_stats = await self._evaluate()
                self._update_eval_stats(eval_stats)

                # Save best model
                if eval_stats['mean_reward'] > self.best_eval_reward:
                    self.best_eval_reward = eval_stats['mean_reward']
                    self._save_best_model()

            # Checkpointing
            if epoch % self.config.save_frequency == 0:
                self._save_checkpoint(epoch)

            # Memory cleanup
            self._cleanup_memory()

            # Log epoch summary
            epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
            self.logger.info(
                f"Epoch {epoch + 1} completed in {epoch_duration:.1f}s - "
                f"Reward: {epoch_stats['mean_reward']:.3f}, "
                f"Verification: {epoch_stats['mean_verification_rate']:.3f}"
            )

        # Final checkpoint
        self._save_final_model()
        self.logger.info("PPO training completed successfully!")

    async def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch by collecting conversations and updating policy.

        Returns:
            Dictionary containing epoch training statistics
        """
        self.logger.info("Collecting training conversations...")

        # Collect conversations
        conversations = []
        for conv_idx in range(self.config.conversations_per_epoch):
            conversation_data = await self._collect_conversation()
            conversations.append(conversation_data)

            # Log progress
            if (conv_idx + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f"Collected {conv_idx + 1}/{self.config.conversations_per_epoch} conversations - "
                    f"Latest reward: {conversation_data['total_reward']:.3f}"
                )

        if not conversations:
            self.logger.warning("No conversations collected this epoch")
            return {'mean_reward': 0.0, 'mean_verification_rate': 0.0, 'mean_length': 0.0}

        # Train PPO algorithm on collected conversations
        self.logger.info("Training PPO algorithm...")
        await self._train_on_batch(conversations)

        # Calculate epoch statistics
        epoch_rewards = [conv['total_reward'] for conv in conversations]
        epoch_verification_rates = [conv['verification_rate'] for conv in conversations]
        epoch_lengths = [len(conv['conversation_history']) for conv in conversations]

        return {
            'mean_reward': sum(epoch_rewards) / len(epoch_rewards),
            'mean_verification_rate': sum(epoch_verification_rates) / len(epoch_verification_rates),
            'mean_length': sum(epoch_lengths) / len(epoch_lengths),
            'total_conversations': len(conversations)
        }

    async def _collect_conversation(self) -> Dict[str, Any]:
        """
        Collect a single conversation trajectory in the environment.

        Returns:
            Dictionary containing conversation data and computed rewards
        """
        # Reset environment for new conversation
        state = self.environment.reset()

        conversation_data = {
            'booking_objective': state.booking_objective,
            'conversation_history': [],
            'user_responses': [],
            'environment_rewards': [],
            'total_reward': 0.0,
            'verification_rate': 0.0
        }

        # Conversation loop
        done = False
        while not done:
            # Generate user response using current policy
            if len(state.conversation_history) == 0:
                # First message is the booking objective
                user_response = state.booking_objective
            else:
                # Generate contextual response using current policy
                context = self._build_conversation_context(state)
                user_response = await self._generate_user_response(context)

            conversation_data['user_responses'].append(user_response)

            # Take environment step
            step_result = await self.environment.step(user_response)

            # Update state and collect data
            state = step_result.state
            done = step_result.done

            conversation_data['conversation_history'] = state.conversation_history.copy()
            conversation_data['environment_rewards'].append(step_result.reward)

        # Calculate final rewards using verifier agent
        if self.config.use_verifier_rewards and self.verifier_agent:
            verifier_reward = await self.reward_calculator.calculate_reward(
                conversation_data['conversation_history'],
                self.verifier_agent,
                self.mcp_client
            )
            conversation_data['total_reward'] = verifier_reward
            conversation_data['verification_rate'] = verifier_reward  # Binary: 0 or 1
        else:
            # Use environment rewards
            env_reward_sum = sum(conversation_data['environment_rewards'])
            conversation_data['total_reward'] = env_reward_sum * self.config.reward_scale
            conversation_data['verification_rate'] = 0.5  # Neutral when not using verifier

        # Log conversation if enabled
        if self.config.save_conversation_logs:
            self._log_conversation(conversation_data)

        return conversation_data

    def _build_conversation_context(self, state) -> str:
        """Build context string for user response generation"""
        context_parts = [f"Objective: {state.booking_objective}"]

        # Add recent conversation history
        recent_history = state.conversation_history[-6:] if len(state.conversation_history) > 6 else state.conversation_history

        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Agent"
            context_parts.append(f"{role}: {msg['content']}")

        context_parts.append("User:")
        return "\n".join(context_parts)

    async def _generate_user_response(self, context: str) -> str:
        """
        Generate user response using the current policy.

        Args:
            context: Conversation context string

        Returns:
            Generated user response
        """
        # Tokenize context
        inputs = self.tokenizer.encode(
            context,
            return_tensors="pt",
            max_length=self.config.max_context_length,
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_response_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.ne(self.tokenizer.pad_token_id)
            )

        # Decode and extract response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        user_response = self._extract_user_response(generated_text, context)

        return user_response

    def _extract_user_response(self, generated_text: str, context: str) -> str:
        """Extract user response from generated text"""
        # Remove context from generated text
        if context in generated_text:
            response_part = generated_text[len(context):].strip()
        else:
            response_part = generated_text.strip()

        # Clean up response
        lines = response_part.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Agent:', 'User:', 'Objective:')):
                # Limit length and return first valid line
                words = line.split()[:15]  # Reasonable length limit
                return " ".join(words)

        # Default response if extraction fails
        return "I need help with flight booking."

    async def _train_on_batch(self, conversations: List[Dict[str, Any]]):
        """
        Train the PPO model on a batch of conversations.

        Args:
            conversations: List of conversation data for training
        """
        if not conversations or self.ppo_trainer is None:
            self.logger.warning("Cannot train: no conversations or PPO trainer unavailable")
            return

        # Process conversations into training format
        queries, responses, rewards = self.conversation_processor.process_batch(conversations)

        if not queries:
            self.logger.warning("No valid training data in batch")
            return

        # Prepare data for PPO trainer
        reward_tensors = [torch.tensor(r, dtype=torch.float32, device=self.device) for r in rewards]

        # Ensure all tensors are on correct device
        queries = [q.to(self.device) for q in queries]
        responses = [r.to(self.device) for r in responses]

        # Execute PPO training step
        stats = self.ppo_trainer.step(queries, responses, reward_tensors)

        self.global_step += 1

        # Update training statistics
        if isinstance(stats, dict):
            if 'loss/policy' in stats:
                self.training_stats['policy_losses'].append(stats['loss/policy'])
            if 'loss/value' in stats:
                self.training_stats['value_losses'].append(stats['loss/value'])

        if rewards:
            self.training_stats['step_rewards'].append(np.mean(rewards))

        # Log training progress
        if self.global_step % self.config.log_interval == 0:
            avg_reward = np.mean(rewards)
            self.logger.info(f"PPO Step {self.global_step} - Avg reward: {avg_reward:.3f}")

    async def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current model performance.

        Returns:
            Dictionary containing evaluation statistics
        """
        self.logger.info("Running evaluation...")

        eval_rewards = []
        eval_verification_rates = []
        eval_lengths = []

        # Set model to evaluation mode
        self.model.eval()

        for i in range(self.config.eval_conversations):
            conversation_data = await self._collect_conversation()

            eval_rewards.append(conversation_data['total_reward'])
            eval_verification_rates.append(conversation_data['verification_rate'])
            eval_lengths.append(len(conversation_data['conversation_history']))

        # Set model back to training mode
        self.model.train()

        eval_stats = {
            'mean_reward': sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0,
            'mean_verification_rate': sum(eval_verification_rates) / len(eval_verification_rates) if eval_verification_rates else 0.0,
            'mean_length': sum(eval_lengths) / len(eval_lengths) if eval_lengths else 0.0,
            'num_conversations': len(eval_rewards)
        }

        self.logger.info(
            f"Evaluation complete - Mean reward: {eval_stats['mean_reward']:.3f}, "
            f"Verification rate: {eval_stats['mean_verification_rate']:.3f}"
        )

        return eval_stats

    def _update_epoch_stats(self, epoch_stats: Dict[str, float]):
        """Update epoch training statistics"""
        self.training_stats['epoch_rewards'].append(epoch_stats['mean_reward'])
        self.training_stats['epoch_verification_rates'].append(epoch_stats['mean_verification_rate'])
        self.training_stats['epoch_conversation_lengths'].append(epoch_stats['mean_length'])

    def _update_eval_stats(self, eval_stats: Dict[str, float]):
        """Update evaluation statistics"""
        self.training_stats['eval_rewards'].append(eval_stats['mean_reward'])
        self.training_stats['eval_verification_rates'].append(eval_stats['mean_verification_rate'])

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints", f"epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training configuration
        config_path = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save training statistics
        stats_path = os.path.join(checkpoint_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        self.logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def _save_best_model(self):
        """Save the best performing model"""
        best_dir = os.path.join(self.config.output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)

        self.model.save_pretrained(best_dir)
        self.tokenizer.save_pretrained(best_dir)

        # Save metadata
        metadata = {
            'best_eval_reward': self.best_eval_reward,
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat()
        }

        metadata_path = os.path.join(best_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Best model saved with reward: {self.best_eval_reward:.3f}")

    def _save_final_model(self):
        """Save the final trained model"""
        final_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save final statistics
        final_stats_path = os.path.join(final_dir, "final_training_stats.json")
        with open(final_stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        self.logger.info(f"Final model saved: {final_dir}")

    def _log_conversation(self, conversation_data: Dict[str, Any]):
        """Log conversation for analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': self.current_epoch,
            'booking_objective': conversation_data['booking_objective'],
            'conversation_history': conversation_data['conversation_history'],
            'total_reward': conversation_data['total_reward'],
            'verification_rate': conversation_data['verification_rate']
        }

        # Save to conversation log file
        conv_log_path = os.path.join(
            self.config.output_dir,
            "conversations",
            f"epoch_{self.current_epoch}_conversations.jsonl"
        )
        with open(conv_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load training statistics if available
        stats_path = os.path.join(checkpoint_path, "training_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)

        self.logger.info("Checkpoint loaded successfully")

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_eval_reward': self.best_eval_reward,
            'training_stats': self.training_stats,
            'config': asdict(self.config)
        }