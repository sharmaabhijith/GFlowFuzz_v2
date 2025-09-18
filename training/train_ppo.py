#!/usr/bin/env python3

"""
PPO Training Script for Auto User Agent
Trains an RL-based user agent to have natural conversations with the booking agent
"""

import os
import sys
import asyncio
import argparse
import yaml
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json
import traceback

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / '.env'
load_dotenv(env_path)

# Configure HuggingFace cache
hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache, "hub")
os.environ["HF_HUB_CACHE"] = os.path.join(hf_cache, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")

# Imports
from training.algorithms.ppo_trainer import PPOAutoUserTrainer
from training.environment.ppo_environment_wrapper import PPOEnvironmentWrapper, PPOBatchCollector
from agents.auto_user.module import AutoUserConfig
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

# Initialize console
console = Console()

# Configure logging
def setup_logging(config: Dict):
    """Setup logging configuration"""
    log_level = config.get("logging", {}).get("level", "INFO")
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"ppo_training_{timestamp}.log")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def display_training_config(config: Dict):
    """Display training configuration in a nice table"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]PPO Training Configuration[/bold cyan]",
        box=box.DOUBLE
    ))

    # Model configuration table
    model_table = Table(title="Model Configuration", box=box.SIMPLE)
    model_table.add_column("Parameter", style="cyan")
    model_table.add_column("Value", style="yellow")

    model_config = config["model"]
    model_table.add_row("Base Model", model_config["base_model"])
    model_table.add_row("Quantization", str(model_config.get("use_quantization", False)))
    model_table.add_row("LoRA", str(model_config.get("use_lora", False)))
    model_table.add_row("Device", model_config.get("device", "auto"))

    console.print(model_table)
    console.print()

    # Training configuration table
    training_table = Table(title="Training Parameters", box=box.SIMPLE)
    training_table.add_column("Parameter", style="cyan")
    training_table.add_column("Value", style="yellow")

    ppo_config = config["ppo"]
    training_table.add_row("Learning Rate", str(ppo_config["learning_rate"]))
    training_table.add_row("Batch Size", str(ppo_config["batch_size"]))
    training_table.add_row("Max Epochs", str(ppo_config["max_epochs"]))
    training_table.add_row("PPO Epochs", str(ppo_config["ppo_epochs"]))
    training_table.add_row("Output Directory", ppo_config["output_dir"])

    console.print(training_table)
    console.print()


async def initialize_environment(config: Dict) -> PPOEnvironmentWrapper:
    """Initialize the training environment"""
    console.print("[cyan]Initializing environment...[/cyan]")

    # Create auto user config
    auto_user_config = AutoUserConfig(
        model_name=config["model"]["base_model"],
        tokenizer_name=config["model"].get("tokenizer", config["model"]["base_model"]),
        max_length=config["model"]["max_response_length"],
        temperature=0.7,
        do_sample=True,
        device=config["model"].get("device", "auto")
    )

    # Create environment wrapper
    env_wrapper = PPOEnvironmentWrapper(
        env_config=config["environment"],
        auto_user_config=auto_user_config
    )

    # Initialize environment
    await env_wrapper.initialize()

    console.print("[green]✓ Environment initialized successfully[/green]")
    return env_wrapper


async def run_training(config: Dict):
    """Main training function"""
    logger = setup_logging(config)
    logger.info("Starting PPO training")

    # Display configuration
    display_training_config(config)

    # Create output directory
    output_dir = config["ppo"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    try:
        # Initialize environment
        console.print("\n[bold cyan]Step 1: Environment Setup[/bold cyan]")
        env_wrapper = await initialize_environment(config)

        # Initialize PPO trainer
        console.print("\n[bold cyan]Step 2: Initializing PPO Trainer[/bold cyan]")
        ppo_trainer = PPOAutoUserTrainer(config)

        # Start training
        console.print("\n[bold cyan]Step 3: Starting Training Loop[/bold cyan]")
        console.print(f"Training for {config['ppo']['max_epochs']} epochs...")
        console.print()

        # Training with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:

            training_task = progress.add_task(
                "[cyan]Training Progress",
                total=config["ppo"]["max_epochs"]
            )

            # Custom training loop with environment
            num_epochs = config["ppo"]["max_epochs"]
            all_metrics = []

            # Initialize models
            ppo_trainer.initialize_models()
            ppo_trainer.create_ppo_trainer()

            for epoch in range(num_epochs):
                progress.update(training_task, description=f"[cyan]Epoch {epoch+1}/{num_epochs}")

                # Collect rollouts
                logger.info(f"Epoch {epoch+1}: Collecting rollouts...")
                rollout_data = await ppo_trainer.collect_rollout_data(
                    env_wrapper.booking_env,
                    num_episodes=config["ppo"]["rollout_episodes"]
                )

                # Add buffer data if configured
                if config["ppo"]["use_buffer"] and epoch > 0:
                    buffer_batch = ppo_trainer.experience_buffer.sample_batch(
                        batch_size=len(rollout_data) // 2,
                        adversarial_ratio=config["ppo"]["adversarial_ratio"]
                    )
                    if buffer_batch:
                        rollout_data.extend(buffer_batch)

                # Train on collected data
                epoch_metrics = {"losses": [], "rewards": []}

                batch_size = config["ppo"]["batch_size"]
                for i in range(0, len(rollout_data), batch_size):
                    batch = rollout_data[i:i+batch_size]
                    if batch:
                        metrics = ppo_trainer.train_step(batch)
                        epoch_metrics["losses"].append(metrics.get("loss", 0))
                        epoch_metrics["rewards"].append(metrics.get("rewards_mean", 0))

                # Calculate epoch statistics
                avg_loss = sum(epoch_metrics["losses"]) / len(epoch_metrics["losses"]) if epoch_metrics["losses"] else 0
                avg_reward = sum(epoch_metrics["rewards"]) / len(epoch_metrics["rewards"]) if epoch_metrics["rewards"] else 0

                # Log metrics
                epoch_summary = {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "avg_reward": avg_reward,
                    "buffer_stats": ppo_trainer.experience_buffer.get_stats()
                }
                all_metrics.append(epoch_summary)

                logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")

                # Evaluation
                if (epoch + 1) % config["ppo"]["eval_freq"] == 0:
                    logger.info(f"Running evaluation...")
                    eval_metrics = await env_wrapper.evaluate_model(
                        ppo_trainer.model,
                        ppo_trainer.tokenizer,
                        num_episodes=config["evaluation"]["num_eval_episodes"]
                    )
                    logger.info(f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.4f}, "
                               f"Verification Rate: {eval_metrics['verification_rate']:.2%}")
                    epoch_summary["eval_metrics"] = eval_metrics

                # Save checkpoint
                if (epoch + 1) % config["ppo"]["save_freq"] == 0:
                    ppo_trainer.save_checkpoint(epoch + 1)

                progress.advance(training_task)

        # Save final model
        console.print("\n[cyan]Saving final model...[/cyan]")
        ppo_trainer.save_final_model()

        # Save training metrics
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        # Display final summary
        display_training_summary(all_metrics)

        console.print("\n[bold green]✓ Training completed successfully![/bold green]")
        console.print(f"[dim]Output saved to: {output_dir}[/dim]")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        console.print(f"\n[bold red]✗ Training failed: {str(e)}[/bold red]")
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        raise


def display_training_summary(metrics: list):
    """Display training summary"""
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]Training Summary[/bold green]",
        box=box.DOUBLE
    ))

    if not metrics:
        console.print("[yellow]No metrics available[/yellow]")
        return

    summary_table = Table(box=box.SIMPLE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")

    # Calculate summary statistics
    final_metrics = metrics[-1]
    avg_reward = sum(m.get("avg_reward", 0) for m in metrics) / len(metrics)

    summary_table.add_row("Total Epochs", str(len(metrics)))
    summary_table.add_row("Final Loss", f"{final_metrics.get('avg_loss', 0):.4f}")
    summary_table.add_row("Final Reward", f"{final_metrics.get('avg_reward', 0):.4f}")
    summary_table.add_row("Average Reward", f"{avg_reward:.4f}")

    if "eval_metrics" in final_metrics:
        eval = final_metrics["eval_metrics"]
        summary_table.add_row("Final Verification Rate", f"{eval.get('verification_rate', 0):.2%}")

    if "buffer_stats" in final_metrics:
        buffer = final_metrics["buffer_stats"]
        summary_table.add_row("Buffer Size", f"{buffer['total_size']}")
        summary_table.add_row("Adversarial Samples", f"{buffer['adversarial_size']}")

    console.print(summary_table)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PPO Training for Auto User Agent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with debug if specified
    if args.debug:
        config["logging"]["level"] = "DEBUG"

    # Display header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🚀 PPO Training for Auto User Agent 🚀[/bold cyan]\n" +
        "[dim]Training an RL agent for natural flight booking conversations[/dim]",
        box=box.DOUBLE_EDGE
    ))

    # Run training
    try:
        asyncio.run(run_training(config))
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Training failed with error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()