#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import random
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Set HuggingFace cache directory
hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache, "hub")
os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from torch.utils.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperienceBuffer:
    """Buffer for storing conversation experiences"""
    max_size: int = 10000
    adversarial_buffer: deque = field(default_factory=lambda: deque(maxlen=5000))
    normal_buffer: deque = field(default_factory=lambda: deque(maxlen=5000))

    def add_experience(self, experience: Dict, is_adversarial: bool):
        """Add an experience to the appropriate buffer"""
        if is_adversarial:
            self.adversarial_buffer.append(experience)
        else:
            self.normal_buffer.append(experience)

    def sample_batch(self, batch_size: int, adversarial_ratio: float = 0.3) -> List[Dict]:
        """Sample a batch with specified ratio of adversarial examples"""
        adversarial_size = int(batch_size * adversarial_ratio)
        normal_size = batch_size - adversarial_size

        batch = []

        # Sample from adversarial buffer
        if len(self.adversarial_buffer) > 0:
            adversarial_samples = min(adversarial_size, len(self.adversarial_buffer))
            batch.extend(random.sample(list(self.adversarial_buffer), adversarial_samples))

        # Sample from normal buffer
        if len(self.normal_buffer) > 0:
            normal_samples = min(normal_size, len(self.normal_buffer))
            batch.extend(random.sample(list(self.normal_buffer), normal_samples))

        random.shuffle(batch)
        return batch

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            "adversarial_size": len(self.adversarial_buffer),
            "normal_size": len(self.normal_buffer),
            "total_size": len(self.adversarial_buffer) + len(self.normal_buffer)
        }


class PPOAutoUserTrainer:
    """PPO trainer for auto user agent using TRL library"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize PPO trainer with configuration"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize buffers
        self.experience_buffer = ExperienceBuffer(
            max_size=config.get("buffer_size", 10000)
        )

        # Model configuration
        self.model_name = config["model"]["base_model"]
        self.use_quantization = config["model"].get("use_quantization", True)
        self.use_lora = config["model"].get("use_lora", True)

        # Training configuration
        self.ppo_config = self._create_ppo_config()

        # Initialize components
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.ppo_trainer = None

    def _create_ppo_config(self) -> PPOConfig:
        """Create PPO configuration from training config"""
        ppo_params = self.config["ppo"]

        return PPOConfig(
            model_name=self.model_name,
            learning_rate=ppo_params["learning_rate"],
            batch_size=ppo_params["batch_size"],
            mini_batch_size=ppo_params["mini_batch_size"],
            gradient_accumulation_steps=ppo_params.get("gradient_accumulation_steps", 1),
            ppo_epochs=ppo_params["ppo_epochs"],
            gamma=ppo_params["gamma"],
            lam=ppo_params["lam"],
            cliprange=ppo_params["cliprange"],
            cliprange_value=ppo_params.get("cliprange_value", 0.2),
            vf_coef=ppo_params["vf_coef"],
            seed=42,
            log_with="tensorboard",
            tracker_project_name="ppo_auto_user",
            remove_unused_columns=False
        )

    def initialize_models(self):
        """Initialize models with quantization and LoRA if configured"""
        logger.info("Initializing models...")

        cache_dir = os.path.join(os.environ.get("HF_HOME"), "hub")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config if enabled
        bnb_config = None
        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            logger.info("Using 4-bit quantization")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )

        # Apply LoRA if enabled
        if self.use_lora:
            # Prepare model for k-bit training if quantized
            if self.use_quantization:
                base_model = prepare_model_for_kbit_training(base_model)

            lora_config = LoraConfig(
                r=self.config["model"].get("lora_r", 16),
                lora_alpha=self.config["model"].get("lora_alpha", 32),
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config["model"].get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            base_model = get_peft_model(base_model, lora_config)
            logger.info("Applied LoRA configuration")

            # Print trainable parameters
            base_model.print_trainable_parameters()

        # Create model with value head for PPO
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        # Create reference model (frozen copy)
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        logger.info("Models initialized successfully")

    def prepare_conversation_for_training(self, conversation: List[Dict]) -> Dict:
        """Convert conversation to training format"""
        # Build conversation context
        context_parts = []
        for msg in conversation[:-1]:  # All messages except last user response
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")

        # Last assistant message is the prompt
        if conversation[-1]["role"] == "assistant":
            context_parts.append(f"Assistant: {conversation[-1]['content']}")
            context_parts.append("User:")

        query = "\n".join(context_parts)

        # The response we want to generate (next user message)
        # For training, we need to have the actual next user response
        response = ""  # This will be generated during training

        return {
            "query": query,
            "response": response
        }

    def tokenize_batch(self, batch: List[str]) -> Dict:
        """Tokenize a batch of texts"""
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

    def compute_rewards(self, responses: List[str], conversation_contexts: List[str],
                       verifier_rewards: List[float]) -> torch.Tensor:
        """Compute combined rewards for responses"""
        rewards = []

        for response, context, verifier_reward in zip(responses, conversation_contexts, verifier_rewards):
            # Base reward from verifier (0 or 1)
            reward = verifier_reward

            # Length penalty
            response_length = len(response.split())
            if response_length < 3:
                reward -= 0.2
            elif response_length > 50:
                reward -= 0.3
            else:
                reward += 0.1

            # Naturalness bonus
            if self._is_natural_response(response):
                reward += 0.2

            # Ending bonus (polite ending)
            if any(word in response.lower() for word in ["thank", "thanks", "please"]):
                reward += 0.1

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32).to(self.device)

    def _is_natural_response(self, response: str) -> bool:
        """Check if response is natural"""
        natural_patterns = [
            'please', 'thank', 'could', 'would', 'i need', 'i want',
            'looking for', 'help me', 'can you', "i'd like"
        ]
        response_lower = response.lower()
        return any(pattern in response_lower for pattern in natural_patterns)

    def create_ppo_trainer(self):
        """Create PPO trainer instance"""
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer
        )
        logger.info("PPO trainer created")

    async def collect_rollout_data(self, environment, num_episodes: int = 10) -> List[Dict]:
        """Collect rollout data from environment interactions"""
        from agents.auto_user.module import AutoUserAgent, AutoUserConfig

        rollout_data = []

        # Create auto user agent for data collection
        auto_config = AutoUserConfig(
            model_name=self.model_name,
            max_length=self.config["model"]["max_response_length"],
            temperature=0.7,
            do_sample=True
        )
        auto_user = AutoUserAgent(auto_config)

        # Use the current model for rollouts
        auto_user.model = self.model.pretrained_model
        auto_user.tokenizer = self.tokenizer

        for episode in range(num_episodes):
            # Reset environment with random objective
            state = environment.reset()
            episode_data = {
                "states": [],
                "actions": [],
                "rewards": [],
                "conversation": []
            }

            done = False
            while not done:
                # Get current context
                context = auto_user.get_conversation_context()
                episode_data["states"].append(context)

                # Generate action (user response)
                action = await auto_user.generate_response()
                episode_data["actions"].append(action)
                episode_data["conversation"].append({"role": "user", "content": action})

                # Step environment
                step_result = await environment.step(action)
                episode_data["conversation"].append({
                    "role": "assistant",
                    "content": step_result.state.conversation_history[-1]["content"]
                })

                # Store reward (will be updated with verifier at end)
                episode_data["rewards"].append(step_result.reward)
                done = step_result.done

            # Get final verifier reward
            verifier_reward = await self._get_verifier_reward(
                episode_data["conversation"],
                environment
            )

            # Update rewards with verifier result
            is_adversarial = verifier_reward == 0.0  # Hallucination detected
            episode_data["verifier_reward"] = verifier_reward
            episode_data["is_adversarial"] = is_adversarial

            rollout_data.append(episode_data)

            # Add to buffer
            self.experience_buffer.add_experience(episode_data, is_adversarial)

            logger.info(f"Episode {episode + 1}/{num_episodes} completed. "
                       f"Verifier reward: {verifier_reward}")

        return rollout_data

    async def _get_verifier_reward(self, conversation: List[Dict], environment) -> float:
        """Get reward from verifier agent"""
        if not environment.verifier_agent:
            return 0.5  # Neutral reward if no verifier

        try:
            # Get verification report
            verification_report = await environment.verifier_agent.verify_bookings(
                conversation,
                environment.booking_agent.mcp_client
            )

            # Calculate reward based on verification
            summary = verification_report.get('summary', {})
            total_claims = summary.get('total_claims', 0)
            verified_claims = summary.get('verified', 0)

            if total_claims == 0:
                return 0.5  # Neutral if no claims

            # Binary reward: 1.0 if all verified, 0.0 if any hallucinations
            verification_rate = verified_claims / total_claims
            return 1.0 if verification_rate == 1.0 else 0.0

        except Exception as e:
            logger.error(f"Error getting verifier reward: {e}")
            return 0.5

    def train_step(self, batch_data: List[Dict]) -> Dict[str, float]:
        """Execute one training step with PPO"""
        # Prepare batch
        queries = []
        responses = []

        for episode in batch_data:
            # Use conversation states as queries
            for i in range(len(episode["states"]) - 1):
                queries.append(episode["states"][i])
                responses.append(episode["actions"][i])

        if not queries:
            return {"loss": 0.0}

        # Tokenize
        query_tensors = []
        response_tensors = []

        for q, r in zip(queries, responses):
            q_tokens = self.tokenizer.encode(q, return_tensors="pt", truncation=True, max_length=256)
            r_tokens = self.tokenizer.encode(r, return_tensors="pt", truncation=True, max_length=64)
            query_tensors.append(q_tokens.squeeze())
            response_tensors.append(r_tokens.squeeze())

        # Compute rewards
        rewards = []
        for episode in batch_data:
            episode_reward = episode["verifier_reward"]
            for _ in episode["states"][:-1]:
                rewards.append(episode_reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)

        return {
            "loss": stats.get("ppo/loss/total", 0.0),
            "rewards_mean": rewards_tensor.mean().item()
        }

    async def train(self, environment, num_epochs: int = None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config["ppo"]["max_epochs"]

        logger.info(f"Starting PPO training for {num_epochs} epochs")

        # Initialize models and trainer
        self.initialize_models()
        self.create_ppo_trainer()

        # Training metrics
        all_metrics = []

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")

            # Collect rollout data
            rollout_episodes = self.config["ppo"].get("rollout_episodes", 10)
            logger.info(f"Collecting {rollout_episodes} rollout episodes...")
            rollout_data = await self.collect_rollout_data(environment, rollout_episodes)

            # Sample from buffer if using off-policy data
            use_buffer = self.config["ppo"].get("use_buffer", True)
            buffer_ratio = self.config["ppo"].get("buffer_ratio", 0.5)

            if use_buffer and self.experience_buffer.get_stats()["total_size"] > 0:
                buffer_size = int(len(rollout_data) * buffer_ratio)
                buffer_batch = self.experience_buffer.sample_batch(
                    buffer_size,
                    adversarial_ratio=self.config["ppo"].get("adversarial_ratio", 0.3)
                )
                rollout_data.extend(buffer_batch)
                logger.info(f"Added {len(buffer_batch)} samples from buffer")

            # Training steps
            batch_size = self.config["ppo"]["batch_size"]
            num_batches = len(rollout_data) // batch_size

            epoch_metrics = {"loss": [], "rewards": []}

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(rollout_data))
                batch_data = rollout_data[batch_start:batch_end]

                # Train on batch
                metrics = self.train_step(batch_data)
                epoch_metrics["loss"].append(metrics["loss"])
                epoch_metrics["rewards"].append(metrics["rewards_mean"])

                if batch_idx % 5 == 0:
                    logger.info(f"Batch {batch_idx}/{num_batches}: "
                               f"Loss={metrics['loss']:.4f}, "
                               f"Reward={metrics['rewards_mean']:.4f}")

            # Epoch summary
            avg_loss = np.mean(epoch_metrics["loss"]) if epoch_metrics["loss"] else 0
            avg_reward = np.mean(epoch_metrics["rewards"]) if epoch_metrics["rewards"] else 0

            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Average Loss: {avg_loss:.4f}")
            logger.info(f"  Average Reward: {avg_reward:.4f}")
            logger.info(f"  Buffer Stats: {self.experience_buffer.get_stats()}")

            all_metrics.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_reward": avg_reward,
                "buffer_stats": self.experience_buffer.get_stats()
            })

            # Save checkpoint
            if (epoch + 1) % self.config["ppo"].get("save_freq", 5) == 0:
                self.save_checkpoint(epoch + 1)

        # Save final model
        self.save_final_model()

        # Save training metrics
        self.save_metrics(all_metrics)

        logger.info("Training completed!")
        return all_metrics

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        output_dir = self.config["ppo"]["output_dir"]
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save buffer
        buffer_path = os.path.join(checkpoint_dir, "buffer.json")
        buffer_data = {
            "adversarial": list(self.experience_buffer.adversarial_buffer),
            "normal": list(self.experience_buffer.normal_buffer)
        }
        with open(buffer_path, "w") as f:
            json.dump(buffer_data, f)

        logger.info(f"Checkpoint saved at {checkpoint_dir}")

    def save_final_model(self):
        """Save final trained model"""
        output_dir = self.config["ppo"]["output_dir"]
        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)

        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        logger.info(f"Final model saved at {final_dir}")

    def save_metrics(self, metrics: List[Dict]):
        """Save training metrics"""
        output_dir = self.config["ppo"]["output_dir"]
        metrics_path = os.path.join(output_dir, "training_metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved at {metrics_path}")