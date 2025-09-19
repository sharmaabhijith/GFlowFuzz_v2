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

# TRL imports for PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM

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

        # Actor-Critic model: Base LLM + value head for state value estimation
        # The value head learns to predict expected returns from each state
        model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        # Reference model: Frozen copy of the initial policy
        # Used to compute KL divergence penalty, preventing catastrophic forgetting
        # and maintaining language coherence during RL training
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        # Freezing ensures reference distribution remains constant during training
        for param in ref_model.parameters():
            param.requires_grad = False

        # Create PPO trainer with all model components
        self.trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value,
            ref_model=ref_model,
            tokenizer=policy_agent.tokenizer
        )

        self.tokenizer = policy_agent.tokenizer

        if self.console:
            self.console.print("[green]✓[/green] PPO trainer initialized successfully")

        return self.trainer

    def prepare_batch(self, trajectories: List[Dict]) -> Tuple[List, List, torch.Tensor]:
        """
        Convert trajectory data into tokenized tensors for PPO optimization.

        Data Processing Strategy:
        - Flatten episode trajectories into individual (state, action, reward) tuples
        - Tokenize text data with appropriate truncation for model limits
        - Apply sparse rewards only at episode boundaries

        Why Sparse Rewards: In conversation tasks, intermediate turns don't have
        clear quality signals. Only the final outcome (objective achieved or not)
        provides reliable feedback.

        Args:
            trajectories: List of collected episode trajectories

        Returns:
            Tuple of (query_tensors, response_tensors, reward_tensor) for PPO
        """
        queries = []      # States/contexts that prompt actions
        responses = []    # Actions taken (agent responses)
        rewards = []      # Rewards for each state-action pair

        for traj in trajectories:
            num_turns = traj.get("num_turns", len(traj["states"]))
            if num_turns == 0:
                continue

            for i in range(num_turns):
                queries.append(traj["states"][i])
                responses.append(traj["actions"][i])

                # Sparse reward assignment: Only the final action gets the episode reward
                # This is because we can't evaluate partial conversation success
                if i == num_turns - 1:  # Terminal state
                    rewards.append(traj["episode_reward"])
                else:
                    rewards.append(0.0)  # Non-terminal states have no immediate reward

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

    def train_step(self, trajectories: List[Dict], experience_buffer=None, epoch: int = 0) -> Dict:
        """
        Perform a single PPO training step with collected trajectories.

        This method handles:
        - Experience replay mixing (if enabled)
        - Batch preparation and tokenization
        - PPO optimization step
        - Metrics extraction and logging

        Args:
            trajectories: Fresh trajectories from current rollouts
            experience_buffer: Optional replay buffer for off-policy learning
            epoch: Current training epoch for logging

        Returns:
            Dictionary containing training metrics
        """
        # Off-policy learning: Mix current trajectories with replay buffer
        # This stabilizes training and improves sample efficiency
        if experience_buffer and self.config["ppo"]["use_buffer"] and epoch > 0:
            buffer_trajectories = experience_buffer.sample_batch(len(trajectories) // 2)
            if buffer_trajectories:
                trajectories.extend(buffer_trajectories)
                if self.console:
                    self.console.print(f"[cyan]Added {len(buffer_trajectories)} trajectories from replay buffer[/cyan]")

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
            "ppo_loss": ppo_loss,
            "mean_reward": mean_reward,
            "kl_divergence": kl_divergence,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "num_samples": len(query_tensors) if query_tensors else 0
        }

    def save_checkpoint(self, epoch: int, output_dir: str, experience_buffer=None):
        """
        Save training checkpoint with model and optimizer state.

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
        summary_table = Table(title=f"Epoch {epoch} Summary - PPO", box=box.SIMPLE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Add metrics to table
        summary_table.add_row("Average Reward", f"{metrics.get('avg_reward', 0):.3f}")
        summary_table.add_row("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        summary_table.add_row("PPO Loss", f"{metrics.get('ppo_loss', 0):.4f}")
        summary_table.add_row("KL Divergence", f"{metrics.get('kl_divergence', 0):.4f}")
        summary_table.add_row("Value Loss", f"{metrics.get('value_loss', 0):.4f}")
        summary_table.add_row("Policy Loss", f"{metrics.get('policy_loss', 0):.4f}")
        summary_table.add_row("Total Trajectories", str(metrics.get('num_trajectories', 0)))

        if 'validation' in metrics and metrics['validation']:
            summary_table.add_row("Val Mean Reward", f"{metrics['validation']['mean_reward']:.3f}")

        self.console.print(summary_table)
        self.console.print()


async def collect_trajectories(
    policy,
    environment,
    num_episodes: int = 10,
    gamma: float = 0.99,
    logger=None,
    console: Console = None,
    trajectory_log_dir: str = None
) -> List[Dict]:
    """
    Collect trajectories using Monte Carlo rollouts with a stateless policy.

    Critical Architecture Point: The policy is treated as a pure function
    mapping states to actions. All conversational context and history is
    maintained by the environment, ensuring clean separation of concerns.

    Reward Structure:
    - Sparse rewards: Only given at episode termination (success/failure)
    - Credit assignment: Uses discounted rewards to propagate signal back

    Args:
        policy: Stateless policy agent (pure state->action mapping)
        environment: Stateful environment managing conversation flow
        num_episodes: Number of complete conversations to collect
        gamma: Discount factor for temporal credit assignment
        logger: Logger for debugging and monitoring
        console: Rich console for enhanced UI output
        trajectory_log_dir: Directory to save successful trajectories

    Returns:
        List of trajectory dictionaries with states, actions, and shaped rewards
    """
    trajectories = []
    successful_trajectories = []  # Track successful conversations for logging

    for episode in range(num_episodes):
        # Environment reset: Generates new user objective and clears conversation
        # The policy never needs resetting as it maintains no internal state
        state_obj = environment.reset()

        # State formatting: Environment packages complete context including:
        # - User objective (what the user wants to achieve)
        # - Full conversation history (all previous turns)
        # - System prompts and constraints
        state = environment._format_state(state_obj)

        if console:
            console.print(f"[cyan]Episode {episode + 1}/{num_episodes}[/cyan] starting...")

        trajectory = {
            "states": [],      # Complete environment states at each timestep
            "actions": [],     # Agent responses/actions taken
            "rewards": [],     # Shaped rewards (computed post-hoc)
            "dones": [],       # Episode termination flags
            "episode_reward": 0.0,  # Final sparse reward (success=1, failure=0)
            "num_turns": 0,    # Total conversation turns for this episode
            "conversation_history": []  # Store for logging successful conversations
        }

        # Episode rollout: Collect a complete conversation trajectory
        done = False
        turn_count = 0

        while not done:
            trajectory["states"].append(state)

            # Pure functional policy: Maps current state to action without memory
            # This ensures no information leakage between episodes and maintains
            # Markov property required for valid RL training
            action = await policy.get_action(state)
            trajectory["actions"].append(action)

            # Store conversation for potential logging
            trajectory["conversation_history"].append({
                "turn": turn_count,
                "state_preview": state[:200] + "..." if len(state) > 200 else state,
                "action": action
            })

            # Environment step
            step_result = await environment.step(action)
            done = step_result.done
            trajectory["dones"].append(done)

            # Get next COMPLETE state from environment
            state = environment._format_state(step_result.state)
            turn_count += 1

        trajectory["num_turns"] = turn_count

        # Terminal reward computation: Evaluate if conversation achieved objective
        # This is the sparse signal that drives learning (1=success, 0=failure)
        episode_reward = await environment.compute_reward()
        trajectory["episode_reward"] = episode_reward

        # Temporal credit assignment: Distribute reward across trajectory
        # Uses exponential discounting to assign more credit to recent actions
        # This solves the credit assignment problem in sparse reward settings
        trajectory["rewards"] = environment.compute_shaped_rewards(
            terminal_reward=episode_reward,
            num_steps=trajectory["num_turns"],
            gamma=gamma  # Controls how much to value immediate vs future rewards
        )

        # Log successful trajectories for analysis
        if episode_reward > 0.5:  # Threshold for "successful" conversation
            successful_trajectories.append(trajectory)
            if console:
                console.print(f"[green]✓ Episode {episode + 1} successful![/green] Reward: {episode_reward:.2f}")
        elif console:
            console.print(f"[yellow]Episode {episode + 1} completed.[/yellow] Reward: {episode_reward:.2f}")

        trajectories.append(trajectory)

    # Save successful trajectories to log file if requested
    if trajectory_log_dir and successful_trajectories:
        os.makedirs(trajectory_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(trajectory_log_dir, f"successful_trajectories_{timestamp}.json")

        with open(log_file, 'w') as f:
            json.dump([
                {
                    "episode_reward": t["episode_reward"],
                    "num_turns": t["num_turns"],
                    "conversation": t["conversation_history"]
                }
                for t in successful_trajectories
            ], f, indent=2)

        if console:
            console.print(f"[green]Saved {len(successful_trajectories)} successful trajectories to {log_file}[/green]")

    return trajectories