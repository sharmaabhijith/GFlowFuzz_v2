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
# Repo Imports
from training.environment import BookingConversationEnvironment
from training.utils import ExperienceBuffer, save_checkpoint, save_final_model, save_metrics, prepare_ppo_batch
from agents.auto_user.module import AutoUserAgent, AutoUserConfig
# UI Imports
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
# TRL imports for PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

async def initialize_environment(config: Dict) -> BookingConversationEnvironment:
    """Initialize the training environment"""
    # Note: auto_user_config is None since environment manages state
    env = BookingConversationEnvironment(
        config=config["environment"],
        auto_user_config=None
    )

    await env.initialize()
    return env


def setup_ppo_trainer(policy_agent: AutoUserAgent, config: Dict) -> PPOTrainer:
    """
    Setup PPO trainer with stateless policy agent

    The policy agent is stateless - it only maps states to actions
    """
    ppo_params = config["ppo"]
    model_name = policy_agent.config.model_name
    tokenizer_name = policy_agent.config.tokenizer_name or model_name

    # Create PPO config
    ppo_config = PPOConfig(
        model_name=model_name,
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
        tracker_project_name="ppo_conversation_policy",
        remove_unused_columns=False,
        optimize_cuda_cache=True,
        early_stopping=ppo_params.get("early_stopping", False),
        target_kl=ppo_params.get("target_kl", 0.1)
    )
    # Create policy model with value head
    model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,  # Fixed: use model_name instead of undefined 'policy'
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    # Create reference model (frozen copy for KL divergence)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    # Freeze reference model parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    # Create PPO trainer with all model components
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model_with_value,  # Fixed: Use model with value head
        ref_model=ref_model,  # Reference model (frozen for KL)
        tokenizer=policy_agent.tokenizer  # Use the agent's tokenizer
    )
    return ppo_trainer


async def collect_trajectories(
    policy: AutoUserAgent,
    environment: BookingConversationEnvironment,
    num_episodes: int = 10,
    gamma: float = 0.99,
    logger=None
) -> List[Dict]:
    """
    Collect trajectories from environment using STATELESS policy

    Key point: No policy.reset_conversation() needed!
    Environment handles all state management.
    """
    trajectories = []

    for episode in range(num_episodes):
        # Only environment resets - policy has no state to reset
        state_obj = environment.reset()
        # Environment provides COMPLETE state (objective + history + context)
        state = environment._format_state(state_obj)

        trajectory = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "episode_reward": 0.0,
            "num_turns": 0  # Track actual conversation turns
        }
        # Collect trajectory until episode done
        done = False
        turn_count = 0

        while not done:
            trajectory["states"].append(state)
            # Policy is pure function: complete_state -> action
            # No internal state or memory needed!
            action = await policy.get_action(state)
            trajectory["actions"].append(action)
            # Environment step
            step_result = await environment.step(action)
            done = step_result.done
            trajectory["dones"].append(done)
            # Get next COMPLETE state from environment
            state = environment._format_state(step_result.state)
            turn_count += 1

        # Store the actual number of conversation turns
        trajectory["num_turns"] = turn_count
        # Compute reward for complete trajectory after episode is done
        episode_reward = await environment.compute_reward()
        trajectory["episode_reward"] = episode_reward
        # Use the environment's method for computing shaped rewards
        # This implements proper credit assignment with discounting
        trajectory["rewards"] = environment.compute_shaped_rewards(
            terminal_reward=episode_reward,
            num_steps=trajectory["num_turns"],
            gamma=gamma  # Now passed as parameter
        )

        trajectories.append(trajectory)

    return trajectories


def prepare_ppo_batch(trajectories: List[Dict], tokenizer, device) -> tuple:
    """
    Prepare trajectories for PPO training

    Converts collected trajectories into format expected by PPO trainer
    """
    queries = []
    responses = []
    rewards = []

    for traj in trajectories:
        num_turns = traj.get("num_turns", len(traj["states"]))
        if num_turns == 0:
            continue

        for i in range(num_turns):
            queries.append(traj["states"][i])
            responses.append(traj["actions"][i])

            # Sparse rewards - only at episode end
            if i == num_turns - 1:  # Last turn
                rewards.append(traj["episode_reward"])
            else:
                rewards.append(0.0)  # Intermediate turns

    if not queries:
        return None, None, None

    # Tokenize queries and responses
    query_tensors = []
    response_tensors = []

    for q, r in zip(queries, responses):
        # Tokenize query (state/context)
        q_tokens = tokenizer.encode(q, return_tensors="pt", truncation=True, max_length=512)
        # Tokenize response (action)
        r_tokens = tokenizer.encode(r, return_tensors="pt", truncation=True, max_length=128)

        query_tensors.append(q_tokens.squeeze())
        response_tensors.append(r_tokens.squeeze())

    # Convert rewards to tensor
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)

    return query_tensors, response_tensors, rewards_tensor


async def run_training(config: Dict):
    """
    Main training function with proper RL architecture:
    - Stateless policy (pure function)
    - Environment manages all state
    - Clean separation of concerns
    """
    # Setup logging
    logging.basicConfig(
        level=config.get("logging", {}).get("level", "INFO"),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = config["ppo"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    # Save configuration
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Step 1: Initialize environment (manages all conversation state)
    environment = await initialize_environment(config)
    # Step 2: Initialize policy agent
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
    policy.initialize_model()  # Load the base model
    # Step 3: Setup PPO trainer
    ppo_trainer = setup_ppo_trainer(policy, config)
    tokenizer = policy.tokenizer
    # Initialize experience buffer
    experience_buffer = ExperienceBuffer(max_size=config.get("buffer_size", 10000))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:

        training_task = progress.add_task(
            "[cyan]Training Progress (Stateless Policy)",
            total=config["ppo"]["max_epochs"]
        )

        num_epochs = config["ppo"]["max_epochs"]
        all_metrics = []

        for epoch in range(num_epochs):
            progress.update(training_task, description=f"[cyan]Epoch {epoch+1}/{num_epochs}")
            # Collect training trajectories
            trajectories = await collect_trajectories(
                policy,
                environment,
                num_episodes=config["ppo"]["rollout_episodes"],
                gamma=config["ppo"].get("gamma", 0.99),
                logger=logger
            )

            # Collect validation trajectories if it's an evaluation epoch
            validation_metrics = None
            if (epoch + 1) % config["ppo"].get("eval_freq", 1) == 0:
                val_trajectories = await collect_trajectories(
                    policy,
                    environment,
                    num_episodes=config["evaluation"].get("num_eval_episodes", 5),
                    gamma=config["ppo"].get("gamma", 0.99),
                    logger=logger
                )
                val_rewards = [t["episode_reward"] for t in val_trajectories]
                validation_metrics = {
                    "mean_reward": sum(val_rewards) / len(val_rewards) if val_rewards else 0,
                    "max_reward": max(val_rewards) if val_rewards else 0,
                    "min_reward": min(val_rewards) if val_rewards else 0
                }

            # Add trajectories to buffer
            for traj in trajectories:
                experience_buffer.add_episode(traj)

            # Add buffer samples if configured
            if config["ppo"]["use_buffer"] and epoch > 0:
                buffer_trajectories = experience_buffer.sample_batch(len(trajectories) // 2)
                if buffer_trajectories:
                    trajectories.extend(buffer_trajectories)

            # Prepare data for PPO trainer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            query_tensors, response_tensors, rewards_tensor = prepare_ppo_batch(
                trajectories, tokenizer, device
            )

            # PPO training step
            ppo_loss = 0.0
            mean_reward = 0.0

            if query_tensors is not None and len(query_tensors) > 0:
                stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
                # Extract metrics
                ppo_loss = stats.get("ppo/loss/total", 0.0) if isinstance(stats, dict) else 0.0
                mean_reward = rewards_tensor.mean().item()
            else:
                logger.warning(f"No valid data for PPO training in epoch {epoch + 1}")
            # Calculate epoch statistics
            episode_rewards = [traj["episode_reward"] for traj in trajectories]
            avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
            # Log metrics
            epoch_summary = {
                "epoch": epoch + 1,
                "avg_episode_reward": avg_reward,
                "ppo_loss": ppo_loss,
                "mean_reward": mean_reward,
                "num_trajectories": len(trajectories),
                "total_steps": sum(traj.get("num_turns", len(traj["states"])) for traj in trajectories),
                "buffer_stats": experience_buffer.get_stats(),
                "architecture": "stateless",  # Mark as stateless architecture
                "validation": validation_metrics  # Add validation metrics
            }
            all_metrics.append(epoch_summary)
            # Save checkpoint
            if (epoch + 1) % config["ppo"]["save_freq"] == 0:
                save_checkpoint(
                    ppo_trainer.model,
                    tokenizer,
                    experience_buffer,
                    epoch + 1,
                    output_dir
                )

            # Log progress
            if validation_metrics:
                logger.info(f"Epoch {epoch + 1}: Train Reward={avg_reward:.3f}, Val Reward={validation_metrics['mean_reward']:.3f}")
            else:
                logger.info(f"Epoch {epoch + 1}: Train Reward={avg_reward:.3f}")

            progress.advance(training_task)

    save_final_model(ppo_trainer.model, tokenizer, output_dir)
    save_metrics(all_metrics, output_dir)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PPO Training for Conversation Policy (Stateless Architecture)")
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