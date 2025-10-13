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
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import logging
import json
from datetime import datetime

# TRL imports
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        tokenizer = policy_agent.tokenizer
        model = policy_agent.model
        use_bf16 = False

        # Helpers to coerce numeric types from config (handle quoted values)
        def _as_float(v: Any, key: str) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except Exception as e:
                    raise TypeError(f"GRPO config '{key}' must be a number, got: {v!r}") from e
            raise TypeError(f"GRPO config '{key}' must be a number, got: {type(v).__name__}")

        def _as_int(v: Any, key: str) -> int:
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
            if isinstance(v, str):
                try:
                    return int(float(v.strip()))
                except Exception as e:
                    raise TypeError(f"GRPO config '{key}' must be an integer, got: {v!r}") from e
            raise TypeError(f"GRPO config '{key}' must be an integer, got: {type(v).__name__}")

        # GRPO configuration using TRL's GRPOConfig (v0.23.0)
        grpo_config = GRPOConfig(
            # Training hyperparameters
            learning_rate=_as_float(grpo_params["learning_rate"], "learning_rate"),
            per_device_train_batch_size=_as_int(grpo_params["batch_size"], "batch_size"),
            per_device_eval_batch_size=_as_int(grpo_params.get("batch_size", 4), "batch_size"),
            gradient_accumulation_steps=_as_int(grpo_params.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps"),

            # Epochs
            num_train_epochs=_as_int(grpo_params.get("grpo_epochs", grpo_params.get("max_epochs", 1)), "grpo_epochs"),

            # Optimization parameters
            max_grad_norm=_as_float(grpo_params.get("max_grad_norm", 0.5), "max_grad_norm"),
            warmup_steps=_as_int(grpo_params.get("warmup_steps", 100), "warmup_steps"),
            max_steps=_as_int(grpo_params.get("max_steps", 1), "max_steps"),

            # Logging and saving
            logging_steps=float(_as_int(grpo_params.get("logging_steps", 10), "logging_steps")),
            save_steps=float(_as_int(grpo_params.get("save_steps", 500), "save_steps")),
            eval_steps=float(_as_int(grpo_params.get("eval_steps", 500), "eval_steps")),

            # Output configuration
            output_dir=grpo_params["output_dir"],
            remove_unused_columns=False,

            # Hardware optimization
            fp16=torch.cuda.is_available() and not use_bf16,
            bf16=torch.cuda.is_available() and use_bf16,
            gradient_checkpointing=bool(grpo_params.get("gradient_checkpointing", False)),

            # Random seed for reproducibility
            seed=42,

            # Group size for relative rewards
            num_generations=_as_int(grpo_params.get("group_size", 4), "group_size"),

            # Temperature for generation during training
            temperature=_as_float(grpo_params.get("temperature", 0.7), "temperature"),

            # Sequence lengths
            max_prompt_length=_as_int(grpo_params.get("max_prompt_length", 256), "max_prompt_length"),
            max_completion_length=_as_int(grpo_params.get("max_completion_length", 512), "max_completion_length"),

            # Training control
            do_train=True,
            do_eval=True,

            # Report to tensorboard
            report_to=["tensorboard"],

            # Push to hub settings
            push_to_hub=False,
        )

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        policy_agent.tokenizer = tokenizer

        # Prepare model for LoRA fine-tuning under 4-bit quantization
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if not hasattr(model, "peft_config"):
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=int(grpo_params.get("lora_rank", 16)),
                lora_alpha=int(grpo_params.get("lora_alpha", 32)),
                target_modules=grpo_params.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                lora_dropout=float(grpo_params.get("lora_dropout", 0.05)),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            policy_agent.model = model

        # Build a minimal prompt dataset. Prefer explicit evaluation objectives from config.
        eval_cfg = self.config.get("evaluation", {})
        default_prompts = [
            "I need to fly from New York to London",
            "Looking for a ticket to Paris",
            "Book a flight to Tokyo",
            "Find me a cheap flight to Rome",
        ]
        prompts: List[str] = eval_cfg.get("eval_objectives") or default_prompts
        train_ds = Dataset.from_dict({"prompt": prompts})

        # Define a very simple reward function (placeholder): reward longer, non-empty completions slightly higher.
        def simple_reward_func(prompts: List[str], completions: List[str], completion_ids=None, trainer_state=None, **kwargs) -> List[float]:
            rewards: List[float] = []
            for c in completions:
                c = c or ""
                # Basic heuristic: encourage informative replies up to a cap
                rewards.append(min(len(c.strip()) / 64.0, 1.0))
            return rewards

        # Create GRPO trainer per TRL 0.23.0 API (model can be a repo id)
        self.trainer = GRPOTrainer(
            model=model,
            reward_funcs=simple_reward_func,
            args=grpo_config,
            train_dataset=train_ds,
            processing_class=self.tokenizer,
        )

        # Align model's config and generation config with tokenizer special tokens
        try:
            model = self.trainer.model
            tok = self.tokenizer
            if getattr(tok, "pad_token_id", None) is not None:
                model.config.pad_token_id = tok.pad_token_id
                if getattr(model, "generation_config", None) is not None:
                    model.generation_config.pad_token_id = tok.pad_token_id
            if getattr(tok, "eos_token_id", None) is not None:
                model.config.eos_token_id = tok.eos_token_id
                if getattr(model, "generation_config", None) is not None:
                    model.generation_config.eos_token_id = tok.eos_token_id
            if getattr(tok, "bos_token_id", None) is not None:
                model.config.bos_token_id = tok.bos_token_id
                if getattr(model, "generation_config", None) is not None:
                    model.generation_config.bos_token_id = tok.bos_token_id
        except Exception:
            # Best-effort alignment; ignore if trainer/model not ready yet
            pass

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

    def train_step(self, trajectories: List[Dict], epoch: int = 0) -> Dict:
        """
        Perform a single GRPO training step with collected trajectories.

        Uses TRL's GRPO trainer for efficient group-based optimization.

        Args:
            trajectories: Fresh trajectories from current rollouts
            epoch: Current training epoch

        Returns:
            Dictionary containing training metrics
        """

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

        # Use Trainer API: run a small number of steps per call if max_steps is set
        train_result = self.trainer.train()
        # Extract metrics
        metrics = getattr(train_result, "metrics", {}) if train_result is not None else {}
        grpo_loss = float(metrics.get("train_loss", metrics.get("loss", 0.0)))
        kl_divergence = float(metrics.get("eval_kl", 0.0))
        mean_reward = float(rewards.mean().item()) if rewards is not None else 0.0

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

    def save_checkpoint(self, epoch: int, output_dir: str):
        """
        Save training checkpoint with model and training state.

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
