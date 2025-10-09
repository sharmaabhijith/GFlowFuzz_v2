import os
import sys
import asyncio
import argparse
import yaml
import logging
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import traceback
from rich.logging import RichHandler
# Path configuration: Establish project structure to ensure imports work correctly
# This allows the training module to access all project components regardless of where it's run from
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
# Environment configuration: Load API keys and environment-specific settings
# The .env file should contain sensitive data like API keys that shouldn't be in version control
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# HuggingFace cache configuration: Centralize model downloads to avoid redundant storage
# This is critical for managing disk space when working with large language models
# HuggingFace uses multiple cache directories - we unify them to a single location
# This prevents duplicate model downloads and makes cleanup easier
hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache, "hub")  # Model weights
os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")  # Hub metadata
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")  # Dataset storage
# Core training components
from training.environment import BookingConversationEnvironment  # Manages conversation state and rewards
from training.utils import ExperienceBuffer, save_metrics  # General utilities
from agents.auto_user.module import AutoUserAgent, AutoUserConfig  # The policy agent being trained
from training.trainer.ppo import PPOAlgorithm, collect_trajectories
from training.trainer.grpo import GRPOAlgorithm  # GRPO algorithm implementation
# Rich helps in enhanced user experience
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.text import Text


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def initialize_environment(config: Dict) -> BookingConversationEnvironment:
    """
    Initialize the training environment with stateless architecture.

    The environment is responsible for ALL state management, including conversation 
    history, user objectives, and context. The policy agentremains stateless 
    (markovian), treating each action request as independent.

    Args:
        config: Environment configuration including conversation parameters

    Returns:
        Initialized environment ready for trajectory collection
    """
    # auto_user_config=None signals that the environment manages its own state
    # rather than delegating to the agent
    env = BookingConversationEnvironment(
        config=config["environment"],
        auto_user_config=None
    )

    await env.initialize()
    return env


async def run_training(config: Dict):
    """
    Main training function with proper RL architecture:
    - Stateless policy
    - Environment manages all state
    """
    console = Console()
    logging.basicConfig(
        level=config.get("logging", {}).get("level", "INFO"),
        format='%(message)s',
        handlers=[
            RichHandler(console=console, rich_tracebacks=True)
        ]
    )
    logger = logging.getLogger(__name__)

    # Determine which algorithm to use
    algorithm_name = config.get("algorithm", "ppo").lower()

    # Display training configuration
    config_table = Table(title="Training Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Algorithm", algorithm_name.upper())
    config_table.add_row("Model", config["model"]["base_model"])

    # Algorithm-specific parameters
    if algorithm_name == "grpo":
        algo_config = config["grpo"]
        config_table.add_row("Learning Rate", str(algo_config["learning_rate"]))
        config_table.add_row("Batch Size", str(algo_config["batch_size"]))
        config_table.add_row("Group Size", str(algo_config["group_size"]))
        config_table.add_row("Max Epochs", str(algo_config["max_epochs"]))
    else:  # PPO
        algo_config = config["ppo"]
        config_table.add_row("Learning Rate", str(algo_config["learning_rate"]))
        config_table.add_row("Batch Size", str(algo_config["batch_size"]))
        config_table.add_row("Max Epochs", str(algo_config["max_epochs"]))

    config_table.add_row("Rollout Episodes", str(algo_config.get("max_conversations_per_epoch", 30)))
    config_table.add_row("Architecture", "Stateless Policy with Stateful Environment")

    console.print(config_table)
    console.print()

    # Output directory setup with organized structure
    output_dir = algo_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for better organization
    trajectory_log_dir = os.path.join(output_dir, "trajectory_logs")
    os.makedirs(trajectory_log_dir, exist_ok=True)

    # Save configuration for reproducibility
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    console.print(f"[green]Output directory:[/green] {output_dir}")
    console.print(f"[green]Trajectory logs:[/green] {trajectory_log_dir}")
    console.print()

    # Training Setup Phase 1: Environment initialization
    # The environment is the source of truth for all state management
    console.print(Panel("[bold cyan]Initializing Training Components[/bold cyan]", expand=False))
    environment = await initialize_environment(config)
    console.print("[green]✓[/green] Environment initialized")

    # Training Setup Phase 2: Policy agent initialization
    policy_config = AutoUserConfig(
        model_name=config["model"]["base_model"],
        tokenizer_name=config["model"].get("tokenizer", config["model"]["base_model"]),
        max_length=config["model"]["max_response_length"],
        temperature=0.7,
        do_sample=True,
        device=config["model"].get("device", "auto"),
        top_p=0.9  # Added missing parameter
    )
    policy = AutoUserAgent(policy_config)
    policy.initialize_model()  # Load the base model weights
    console.print("[green]✓[/green] Policy model loaded")

    # Training Setup Phase 3: Initialize trainer based on selected algorithm
    # This modular design allows easy swapping of algorithms
    if algorithm_name == "grpo":
        trainer = GRPOAlgorithm(config, console=console)
        trainer.setup_trainer(policy)  # Both use setup_trainer now
        console.print("[green]✓[/green] Trainer (GRPO) initialized")
    else:  # PPO
        trainer = PPOAlgorithm(config, console=console)
        trainer.setup_trainer(policy)
        console.print("[green]✓[/green] Trainer (PPO) initialized")

    # Experience replay buffer: Stores past trajectories for sample efficiency
    # Reusing past experiences reduces sample complexity and stabilizes training
    experience_buffer = ExperienceBuffer(max_size=config.get("buffer_size", 10000))
    console.print("[green]✓[/green] Experience buffer initialized")
    console.print()

    # Enhanced progress tracking with time estimation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        training_task = progress.add_task(
            f"[bold cyan]{algorithm_name.upper()} Training Progress[/bold cyan]",
            total=algo_config["max_epochs"]
        )

        num_epochs = algo_config["max_epochs"]
        all_metrics = []

        for epoch in range(num_epochs):
            progress.update(training_task, description=f"[bold cyan]Epoch {epoch+1}/{num_epochs}[/bold cyan]")

            # Display epoch panel
            console.print(Panel(
                f"[bold]Starting Epoch {epoch+1}[/bold]\n"
                f"Collecting {config['ppo']['rollout_episodes']} trajectories...",
                title="[cyan]Training Status[/cyan]",
                expand=False
            ))

            # Trajectory collection phase: Generate new experience through rollouts
            rollout_episodes = algo_config.get("max_conversations_per_epoch", 30)
            trajectories = await collect_trajectories(
                policy,
                environment,
                num_episodes=rollout_episodes,
                gamma=algo_config.get("gamma", 0.99),
                logger=logger,
                console=console,
                trajectory_log_dir=trajectory_log_dir if epoch % 5 == 0 else None  # Log every 5 epochs
            )

            # Validation phase: Evaluate policy performance without exploration
            # This gives us an unbiased estimate of policy quality
            validation_metrics = None
            if (epoch + 1) % algo_config.get("eval_freq", 1) == 0:
                console.print("\n[yellow]Running validation...[/yellow]")
                val_trajectories = await collect_trajectories(
                    policy,
                    environment,
                    num_episodes=config["evaluation"].get("num_eval_episodes", 5),
                    gamma=algo_config.get("gamma", 0.99),
                    logger=logger
                )
                val_rewards = [t["episode_reward"] for t in val_trajectories]
                validation_metrics = {
                    "mean_reward": sum(val_rewards) / len(val_rewards) if val_rewards else 0,
                    "max_reward": max(val_rewards) if val_rewards else 0,
                    "min_reward": min(val_rewards) if val_rewards else 0
                }

            # Experience replay: Store trajectories for future reuse
            # This improves sample efficiency by learning from past experiences
            for traj in trajectories:
                experience_buffer.add_episode(traj)

            # Perform trainer training step (delegated to trainer module)
            # This abstraction allows easy swapping of different RL algorithms
            train_metrics = trainer.train_step(
                trajectories=trajectories,
                experience_buffer=experience_buffer,
                epoch=epoch
            )

            # Extract algorithm-specific loss
            if algorithm_name == "grpo":
                algo_loss = train_metrics.get("grpo_loss", 0.0)
            else:
                algo_loss = train_metrics.get("ppo_loss", 0.0)
            mean_reward = train_metrics.get("mean_reward", 0.0)
            # Epoch statistics: Aggregate metrics for monitoring progress
            episode_rewards = [traj["episode_reward"] for traj in trajectories]
            terminal_rewards = [traj.get("terminal_reward", traj["episode_reward"]) for traj in trajectories]
            process_rewards = [sum(traj.get("process_rewards", [])) for traj in trajectories]

            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            avg_terminal = sum(terminal_rewards) / len(terminal_rewards) if terminal_rewards else 0
            avg_process = sum(process_rewards) / len(process_rewards) if process_rewards else 0
            success_rate = sum(1 for r in terminal_rewards if r > 0.5) / len(terminal_rewards) if terminal_rewards else 0

            # Display trainer-specific epoch summary
            epoch_display_metrics = {
                "avg_reward": avg_reward,
                "avg_terminal_reward": avg_terminal,
                "avg_process_reward": avg_process,
                "success_rate": success_rate,
                "kl_divergence": train_metrics.get("kl_divergence", 0),
                "num_trajectories": len(trajectories),
                "validation": validation_metrics
            }

            # Add algorithm-specific metrics
            if algorithm_name == "grpo":
                epoch_display_metrics["grpo_loss"] = algo_loss
            else:  # PPO
                epoch_display_metrics["ppo_loss"] = algo_loss
                epoch_display_metrics["value_loss"] = train_metrics.get("value_loss", 0)
                epoch_display_metrics["policy_loss"] = train_metrics.get("policy_loss", 0)
            trainer.display_epoch_summary(epoch + 1, epoch_display_metrics)

            # Comprehensive metrics tracking
            epoch_summary = {
                "epoch": epoch + 1,
                "algorithm": algorithm_name,
                "avg_episode_reward": avg_reward,
                "avg_terminal_reward": avg_terminal,
                "avg_process_reward": avg_process,
                "success_rate": success_rate,
                "algo_loss": algo_loss,
                "mean_reward": mean_reward,
                "num_trajectories": len(trajectories),
                "total_steps": sum(traj.get("num_turns", len(traj["states"])) for traj in trajectories),
                "buffer_stats": experience_buffer.get_stats(),
                "architecture": "stateless",
                "validation": validation_metrics
            }

            # Add algorithm-specific metrics to summary
            if algorithm_name == "grpo":
                epoch_summary["grpo_loss"] = algo_loss
            else:
                epoch_summary["ppo_loss"] = algo_loss
            all_metrics.append(epoch_summary)
            # Checkpoint saving: Delegated to trainer for trainer-specific handling
            if (epoch + 1) % algo_config["save_freq"] == 0:
                trainer.save_checkpoint(
                    epoch + 1,
                    output_dir,
                    experience_buffer
                )

            # Progress logging with validation comparison
            if validation_metrics:
                logger.info(
                    f"Epoch {epoch + 1} Complete | "
                    f"Train: {avg_reward:.3f} (Success: {success_rate:.1%}) | "
                    f"Val: {validation_metrics['mean_reward']:.3f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1} Complete | "
                    f"Reward: {avg_reward:.3f} | "
                    f"Success Rate: {success_rate:.1%}"
                )

            progress.advance(training_task)

    # Training completion: Save final artifacts
    console.print(Panel(
        "[bold green]Training Complete![/bold green]",
        title="Status",
        expand=False
    ))

    # Save final model using trainer-specific method
    trainer.save_final_model(output_dir)
    save_metrics(all_metrics, output_dir)

    # Display final summary
    final_table = Table(title="Training Results", box=box.DOUBLE)
    final_table.add_column("Metric", style="cyan")
    final_table.add_column("Value", style="green")

    if all_metrics:
        final_metrics = all_metrics[-1]
        final_table.add_row("Final Epoch", str(final_metrics["epoch"]))
        final_table.add_row("Final Avg Reward", f"{final_metrics['avg_episode_reward']:.3f}")
        final_table.add_row("Final Success Rate", f"{final_metrics.get('success_rate', 0):.1%}")
        final_table.add_row("Total Steps Trained", str(sum(m["total_steps"] for m in all_metrics)))

    final_table.add_row("Model Saved To", output_dir)
    console.print(final_table)

    console.print(f"\n[green]Trajectory logs saved to:[/green] {trajectory_log_dir}")
    console.print("[cyan]Review the logs to understand conversation patterns and success cases.[/cyan]")


def main():
    """
    Main entry point for PPO training.

    Architecture Philosophy:
    This implementation follows clean RL principles with strict separation:
    - Policy: Stateless function mapping observations to actions
    - Environment: Maintains all state and computes rewards
    - Training: PPO with experience replay for sample efficiency

    This design ensures reproducibility, debuggability, and theoretical soundness.
    """
    parser = argparse.ArgumentParser(
        description="PPO Training for Conversation Policy (Stateless Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python train.py --config configs/ppo_training_config.yaml
  python train.py --config custom_config.yaml --debug
  python train.py --resume checkpoints/epoch_10.pt
        """
    )
    parser.add_argument("--config", type=str, default="configs/ppo_training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug:
        config["logging"]["level"] = "DEBUG"

    asyncio.run(run_training(config))


if __name__ == "__main__":
    main()