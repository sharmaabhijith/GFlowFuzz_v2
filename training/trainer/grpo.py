#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Algorithm Implementation using TRL

This module uses TRL's official GRPOTrainer for efficient group-based policy optimization.
GRPO is particularly effective for language model fine-tuning as it:
- Uses relative rewards within groups to reduce variance
- Doesn't require a value network (policy-only)
- Is more sample-efficient than PPO for language tasks

The TRL implementation handles all the complex optimization details while
we focus on trajectory collection and reward shaping.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

# TRL imports for GRPO
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# Rich UI components
from rich.console import Console
from rich.table import Table
from rich import box

logger = logging.getLogger(__name__)


class GRPOAlgorithm:
    """
    GRPO algorithm implementation using TRL's GRPOTrainer.

    This class provides a clean interface to TRL's GRPO implementation,
    matching the API of our PPO implementation for easy swapping.
    """

    def __init__(self, config: Dict, console: Console = None):
        """
        Initialize GRPO algorithm with configuration.

        Args:
            config: Algorithm configuration dictionary
            console: Rich console for enhanced output
        """
        self.config = config
        self.console = console or Console()
        self.trainer = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_trainer(self, policy_agent) -> GRPOTrainer:
        """
        Initialize GRPO trainer using TRL's implementation.

        GRPO doesn't require a value head, making it simpler and more
        memory-efficient than PPO. The group-based advantage computation
        provides natural variance reduction.

        Args:
            policy_agent: The agent whose underlying model will be trained

        Returns:
            Configured GRPO trainer ready for optimization
        """
        grpo_params = self.config["grpo"]
        model_name = policy_agent.config.model_name
        tokenizer_name = policy_agent.config.tokenizer_name or model_name

        # GRPO configuration using TRL's GRPOConfig
        grpo_config = GRPOConfig(
            # Model configuration
            model_name=model_name,

            # Training hyperparameters
            learning_rate=grpo_params["learning_rate"],
            batch_size=grpo_params["batch_size"],
            gradient_accumulation_steps=grpo_params.get("gradient_accumulation_steps", 1),

            # GRPO specific parameters
            num_train_epochs=grpo_params["grpo_epochs"],
            gamma=grpo_params["gamma"],  # Discount factor
            lam=grpo_params["lam"],  # GAE lambda
            kl_coef=grpo_params.get("kl_coef", 0.05),  # KL penalty coefficient

            # Optimization parameters
            max_grad_norm=grpo_params.get("max_grad_norm", 0.5),
            warmup_steps=grpo_params.get("warmup_steps", 100),

            # Logging and saving
            logging_steps=grpo_params.get("logging_steps", 10),
            save_steps=grpo_params.get("save_steps", 500),
            eval_steps=grpo_params.get("eval_steps", 500),

            # Output configuration
            output_dir=grpo_params["output_dir"],
            remove_unused_columns=False,

            # Hardware optimization
            fp16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),

            # Random seed for reproducibility
            seed=42,

            # Group size for relative rewards
            num_sample_generations=grpo_params.get("group_size", 4),

            # Temperature for generation during training
            temperature=grpo_params.get("temperature", 0.7),

            # Maximum sequence length
            max_length=512,
            max_prompt_length=256,

            # Training control
            do_train=True,
            do_eval=True,

            # Report to tensorboard
            report_to=["tensorboard"],

            # Push to hub settings
            push_to_hub=False,
        )

        # Load the model (no value head needed for GRPO)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )

        # Load reference model for KL divergence
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )

        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create GRPO trainer
        self.trainer = GRPOTrainer(
            config=grpo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=self.tokenizer
        )

        if self.console:
            self.console.print("[green]✓[/green] GRPO trainer initialized successfully")
            self.console.print(f"  - Model: {model_name}")
            self.console.print(f"  - Learning rate: {grpo_params['learning_rate']}")
            self.console.print(f"  - Batch size: {grpo_params['batch_size']}")
            self.console.print(f"  - Group size: {grpo_params.get('group_size', 4)}")

        return self.trainer

    def prepare_batch(self, trajectories: List[Dict]) -> Tuple[List, List, torch.Tensor]:
        """
        Convert trajectory data into format expected by GRPO trainer.

        GRPO uses group-based comparison, so we organize trajectories
        into groups for relative reward computation.

        Args:
            trajectories: List of collected episode trajectories

        Returns:
            Tuple of (queries, responses, rewards) for GRPO
        """
        queries = []
        responses = []
        rewards = []

        for traj in trajectories:
            num_turns = traj.get("num_turns", len(traj["states"]))
            if num_turns == 0:
                continue

            for i in range(num_turns):
                # Extract state and action
                queries.append(traj["states"][i])
                responses.append(traj["actions"][i])

                # Use combined rewards (process + terminal)
                if "rewards" in traj and i < len(traj["rewards"]):
                    rewards.append(traj["rewards"][i])
                else:
                    # Fallback to episode reward for last step
                    if i == num_turns - 1:
                        rewards.append(traj.get("terminal_reward", traj.get("episode_reward", 0.0)))
                    else:
                        rewards.append(traj.get("process_rewards", [0.0])[i] if "process_rewards" in traj else 0.0)

        if not queries:
            return None, None, None

        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        return queries, responses, rewards_tensor

    def train_step(self, trajectories: List[Dict], experience_buffer=None, epoch: int = 0) -> Dict:
        """
        Perform a single GRPO training step with collected trajectories.

        Uses TRL's GRPO trainer for efficient group-based optimization.

        Args:
            trajectories: Fresh trajectories from current rollouts
            experience_buffer: Optional replay buffer
            epoch: Current training epoch

        Returns:
            Dictionary containing training metrics
        """
        # Mix with experience buffer if configured
        if experience_buffer and self.config["grpo"].get("use_buffer", False) and epoch > 0:
            buffer_trajectories = experience_buffer.sample_batch(len(trajectories) // 2)
            if buffer_trajectories:
                trajectories.extend(buffer_trajectories)
                if self.console:
                    self.console.print(f"[cyan]Added {len(buffer_trajectories)} trajectories from buffer[/cyan]")

        # Prepare data for GRPO
        queries, responses, rewards = self.prepare_batch(trajectories)

        if queries is None or len(queries) == 0:
            logger.warning(f"No valid data for GRPO training in epoch {epoch + 1}")
            return {
                "grpo_loss": 0.0,
                "mean_reward": 0.0,
                "kl_divergence": 0.0,
                "num_samples": 0
            }

        if self.console:
            self.console.print("\n[cyan]Running GRPO optimization...[/cyan]")

        # Prepare data in the format expected by TRL's GRPO
        # GRPO expects tokenized inputs, so we'll tokenize here
        tokenized_queries = []
        tokenized_responses = []

        for q, r in zip(queries, responses):
            # Tokenize query
            q_tokens = self.tokenizer(q, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
            # Tokenize response
            r_tokens = self.tokenizer(r, return_tensors="pt", truncation=True, max_length=128, padding="max_length")

            tokenized_queries.append(q_tokens.input_ids.squeeze())
            tokenized_responses.append(r_tokens.input_ids.squeeze())

        # Run GRPO training step
        stats = self.trainer.step(tokenized_queries, tokenized_responses, rewards)

        # Extract metrics
        grpo_loss = 0.0
        kl_divergence = 0.0
        mean_reward = rewards.mean().item()

        if isinstance(stats, dict):
            grpo_loss = stats.get("loss", 0.0)
            kl_divergence = stats.get("kl", 0.0)

        # Calculate additional metrics
        process_rewards = []
        terminal_rewards = []
        for traj in trajectories:
            if "process_rewards" in traj:
                process_rewards.extend(traj["process_rewards"])
            if "terminal_reward" in traj:
                terminal_rewards.append(traj["terminal_reward"])

        mean_process_reward = sum(process_rewards) / len(process_rewards) if process_rewards else 0.0
        mean_terminal_reward = sum(terminal_rewards) / len(terminal_rewards) if terminal_rewards else 0.0

        if self.console:
            self.console.print(f"[green]GRPO optimization complete[/green] - Loss: {grpo_loss:.4f}, KL: {kl_divergence:.4f}")

        return {
            "grpo_loss": grpo_loss,
            "mean_reward": mean_reward,
            "mean_process_reward": mean_process_reward,
            "mean_terminal_reward": mean_terminal_reward,
            "kl_divergence": kl_divergence,
            "num_samples": len(queries)
        }

    def save_checkpoint(self, epoch: int, output_dir: str, experience_buffer=None):
        """
        Save training checkpoint with model and training state.

        Args:
            epoch: Current epoch number
            output_dir: Directory to save checkpoint
            experience_buffer: Optional buffer to save
        """
        checkpoint_dir = Path(output_dir) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.trainer.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save experience buffer if provided
        if experience_buffer:
            experience_buffer.save(checkpoint_dir / "buffer.json")

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
        summary_table = Table(title=f"Epoch {epoch} Summary - GRPO", box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Add metrics to table
        summary_table.add_row("Average Total Reward", f"{metrics.get('avg_reward', 0):.3f}")
        summary_table.add_row("Terminal Reward", f"{metrics.get('avg_terminal_reward', 0):.3f}")
        summary_table.add_row("Process Reward", f"{metrics.get('avg_process_reward', 0):.3f}")
        summary_table.add_row("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        summary_table.add_row("GRPO Loss", f"{metrics.get('grpo_loss', 0):.4f}")
        summary_table.add_row("KL Divergence", f"{metrics.get('kl_divergence', 0):.4f}")
        summary_table.add_row("Total Trajectories", str(metrics.get('num_trajectories', 0)))

        if 'validation' in metrics and metrics['validation']:
            summary_table.add_row("Val Mean Reward", f"{metrics['validation']['mean_reward']:.3f}")

        self.console.print(summary_table)
        self.console.print()