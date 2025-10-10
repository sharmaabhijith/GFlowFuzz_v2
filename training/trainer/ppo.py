#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Algorithm Implementation

This module contains all PPO-specific functions and classes for training
conversational agents using reinforcement learning. The implementation follows
a modular design to allow easy addition of other RL algorithms in the future.

Architecture:
- PPO with actor-critic networks (policy + value function)
- KL divergence regularization to prevent catastrophic forgetting
- Experience replay buffer for improved sample efficiency
- Stateless policy design for clean RL implementation
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

# TRL imports for PPO with Unsloth
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from unsloth import FastLanguageModel

# Rich UI components
from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)


class PPOAlgorithm:
    """
    PPO algorithm implementation with enhanced features for conversation training.

    This class encapsulates all PPO-specific logic, making it easy to swap
    algorithms or add new ones without modifying the main training loop.
    """

    def __init__(self, config: Dict, console: Console = None):
        """
        Initialize PPO algorithm with configuration.

        Args:
            config: Algorithm configuration dictionary
            console: Rich console for enhanced output (optional)
        """
        self.config = config
        self.console = console or Console()
        self.trainer = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_trainer(self, policy_agent) -> PPOTrainer:
        """
        Initialize PPO trainer with proper actor-critic architecture.

        Architecture Details:
        - Actor (Policy): The base language model that generates actions (responses)
        - Critic (Value): An additional head on the model that estimates state values
        - Reference Model: A frozen copy of the initial model for KL divergence calculation

        The KL divergence term prevents the policy from diverging too far from the
        initial behavior, maintaining coherent language generation while optimizing
        for task-specific rewards.

        Args:
            policy_agent: The agent whose underlying model will be trained

        Returns:
            Configured PPO trainer ready for optimization
        """
        ppo_params = self.config["ppo"]
        model_name = policy_agent.config.model_name
        tokenizer_name = policy_agent.config.tokenizer_name or model_name

        # PPO hyperparameter configuration with careful defaults
        ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=ppo_params["learning_rate"],  # Typically 1e-5 to 1e-4 for LLMs
            batch_size=ppo_params["batch_size"],  # Total batch size across all devices
            mini_batch_size=ppo_params["mini_batch_size"],  # Size for gradient updates
            gradient_accumulation_steps=ppo_params.get("gradient_accumulation_steps", 1),
            ppo_epochs=ppo_params["ppo_epochs"],  # Number of optimization epochs per batch
            gamma=ppo_params["gamma"],  # Discount factor for future rewards (0.99 = long-term focus)
            lam=ppo_params["lam"],  # GAE lambda for advantage estimation (balance bias/variance)
            cliprange=ppo_params["cliprange"],  # PPO clipping to prevent large policy updates
            cliprange_value=ppo_params.get("cliprange_value", 0.2),  # Value function clipping
            vf_coef=ppo_params["vf_coef"],  # Weight of value function loss in total loss
            seed=42,  # Fixed seed for reproducibility
            log_with="tensorboard",  # Enable tensorboard logging for monitoring
            tracker_project_name="ppo_conversation_policy",
            remove_unused_columns=False,  # Keep all data columns for debugging
            optimize_cuda_cache=True,  # Memory optimization for large models
            early_stopping=ppo_params.get("early_stopping", False),  # Stop if KL diverges
            target_kl=ppo_params.get("target_kl", 0.1)  # KL divergence threshold
        )

        # Load model and tokenizer using Unsloth's FastLanguageModel
        max_seq_length = 2048  # Unsloth auto supports RoPE Scaling internally
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_in_4bit = True  # 4bit quantization enabled to save memory space

        # Load base model with Unsloth for efficient training
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Prepare model for k-bit training with Unsloth (enables LoRA for efficiency)
        base_model = FastLanguageModel.get_peft_model(
            base_model,
            r=16,  # LoRA rank - suggested values 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=42,
        )

        # Wrap with value head for PPO (Actor-Critic architecture)
        model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

        # Reference model: We need a frozen copy for KL divergence
        # Load a separate instance for the reference model
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Freezing ensures reference distribution remains constant during training
        for param in ref_model.parameters():
            param.requires_grad = False

        # Create PPO trainer with all model components
        self.trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value,
            ref_model=ref_model,
            tokenizer=tokenizer  # Use the tokenizer from FastLanguageModel
        )

        self.tokenizer = tokenizer

        if self.console:
            self.console.print("[green]✓[/green] PPO trainer initialized successfully")

        return self.trainer

    def prepare_batch(self, trajectories: List[Dict]) -> Tuple[List, List, torch.Tensor]:
        """
        Convert trajectory data into tokenized tensors for PPO optimization.

        Data Processing Strategy:
        - Flatten episode trajectories into individual (state, action, reward) tuples
        - Tokenize text data with appropriate truncation for model limits
        - Now uses combined process and terminal rewards

        Process rewards provide immediate feedback during conversation,
        while terminal rewards evaluate overall success.

        Args:
            trajectories: List of collected episode trajectories

        Returns:
            Tuple of (query_tensors, response_tensors, reward_tensor) for PPO
        """
        queries = []      # States/contexts that prompt actions
        responses = []    # Actions taken (agent responses)
        rewards = []      # Combined rewards for each state-action pair

        for traj in trajectories:
            num_turns = traj.get("num_turns", len(traj["states"]))
            if num_turns == 0:
                continue

            for i in range(num_turns):
                queries.append(traj["states"][i])
                responses.append(traj["actions"][i])

                # Use the combined rewards that include both process and terminal rewards
                # These were already computed in collect_trajectories
                if "rewards" in traj and i < len(traj["rewards"]):
                    rewards.append(traj["rewards"][i])
                else:
                    # Fallback for backward compatibility
                    if i == num_turns - 1:  # Terminal state
                        rewards.append(traj.get("episode_reward", 0.0))
                    else:
                        rewards.append(0.0)

        if not queries:
            return None, None, None

        # Tokenize queries and responses
        query_tensors = []
        response_tensors = []

        for q, r in zip(queries, responses):
            # Tokenize query (state/context)
            q_tokens = self.tokenizer.encode(q, return_tensors="pt", truncation=True, max_length=512)
            # Tokenize response (action)
            r_tokens = self.tokenizer.encode(r, return_tensors="pt", truncation=True, max_length=128)

            query_tensors.append(q_tokens.squeeze())
            response_tensors.append(r_tokens.squeeze())

        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        return query_tensors, response_tensors, rewards_tensor

    def train_step(self, trajectories: List[Dict], epoch: int = 0) -> Dict:
        """
        Perform a single PPO training step with collected trajectories.

        This method handles:
        - Batch preparation and tokenization
        - PPO optimization step
        - Metrics extraction and logging

        Args:
            trajectories: Fresh trajectories from current rollouts
            epoch: Current training epoch for logging

        Returns:
            Dictionary containing training metrics
        """

        # Prepare data for PPO trainer
        query_tensors, response_tensors, rewards_tensor = self.prepare_batch(trajectories)

        # PPO optimization phase
        if self.console:
            self.console.print("\n[cyan]Running PPO optimization...[/cyan]")

        ppo_loss = 0.0
        mean_reward = 0.0
        kl_divergence = 0.0
        value_loss = 0.0
        policy_loss = 0.0

        if query_tensors is not None and len(query_tensors) > 0:
            # PPO step performs multiple epochs of minibatch updates
            # This extracts more learning signal from each trajectory
            stats = self.trainer.step(query_tensors, response_tensors, rewards_tensor)

            # Extract training metrics for monitoring
            if isinstance(stats, dict):
                ppo_loss = stats.get("ppo/loss/total", 0.0)
                kl_divergence = stats.get("ppo/mean_non_score_reward", 0.0)
                value_loss = stats.get("ppo/loss/value", 0.0)
                policy_loss = stats.get("ppo/loss/policy", 0.0)

            mean_reward = rewards_tensor.mean().item()

            if self.console:
                self.console.print(f"[green]PPO optimization complete[/green] - Loss: {ppo_loss:.4f}, KL: {kl_divergence:.4f}")
        else:
            logger.warning(f"No valid data for PPO training in epoch {epoch + 1}")

        return {
            "loss": ppo_loss,
            "mean_reward": mean_reward,
            "kl_divergence": kl_divergence,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "num_samples": len(query_tensors) if query_tensors else 0
        }

    def save_checkpoint(self, epoch: int, output_dir: str):
        """
        Save training checkpoint with model and optimizer state.

        Args:
            epoch: Current epoch number
            output_dir: Directory to save checkpoint
        """
        checkpoint_dir = Path(output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.trainer.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state_dict = {
            "epoch": epoch,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }

        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(state_dict, f, indent=2)

        if self.console:
            self.console.print(f"[green]✓ Checkpoint saved for epoch {epoch}[/green]")

        logger.info(f"Checkpoint saved at {checkpoint_dir}")

    def save_final_model(self, output_dir: str):
        """
        Save the final trained model.

        Args:
            output_dir: Directory to save the final model
        """
        final_dir = Path(output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)

        self.trainer.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save final configuration
        with open(final_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        if self.console:
            self.console.print(f"[green]✓ Final model saved to {final_dir}[/green]")

        logger.info(f"Final model saved at {final_dir}")

    def get_model(self):
        """Get the current model for external use."""
        return self.trainer.model if self.trainer else None

    def get_tokenizer(self):
        """Get the tokenizer for external use."""
        return self.tokenizer

    def display_epoch_summary(self, epoch: int, metrics: Dict):
        """
        Display a formatted summary of epoch metrics.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of training metrics
        """
        if not self.console:
            return

        # Create summary table
        summary_table = Table(title=f"Epoch {epoch} Summary - PPO", box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Add metrics to table
        summary_table.add_row("Average Total Reward", f"{metrics.get('avg_reward', 0):.3f}")
        summary_table.add_row("Terminal Reward", f"{metrics.get('avg_terminal_reward', 0):.3f}")
        summary_table.add_row("Process Reward", f"{metrics.get('avg_process_reward', 0):.3f}")
        summary_table.add_row("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        summary_table.add_row("Loss", f"{metrics.get('loss', 0):.4f}")
        summary_table.add_row("KL Divergence", f"{metrics.get('kl_divergence', 0):.4f}")
        summary_table.add_row("Value Loss", f"{metrics.get('value_loss', 0):.4f}")
        summary_table.add_row("Policy Loss", f"{metrics.get('policy_loss', 0):.4f}")
        summary_table.add_row("Total Trajectories", str(metrics.get('num_trajectories', 0)))

        if 'validation' in metrics and metrics['validation']:
            summary_table.add_row("Val Mean Reward", f"{metrics['validation']['mean_reward']:.3f}")

        self.console.print(summary_table)
        self.console.print()