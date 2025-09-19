#!/usr/bin/env python3

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import deque
import torch

logger = logging.getLogger(__name__)


@dataclass
class ExperienceBuffer:
    """Simple buffer for storing RL episodes"""
    max_size: int = 10000
    episodes: deque = field(default_factory=lambda: deque(maxlen=10000))

    def add_episode(self, episode: Dict):
        """Add an episode to the buffer"""
        self.episodes.append(episode)

    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch from the buffer"""
        if len(self.episodes) < batch_size:
            return list(self.episodes)
        return random.sample(list(self.episodes), batch_size)

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            "buffer_size": len(self.episodes),
            "max_size": self.max_size
        }

    def save(self, path: str):
        """Save buffer to file"""
        with open(path, 'w') as f:
            json.dump(list(self.episodes), f, indent=2)

    def load(self, path: str):
        """Load buffer from file"""
        try:
            with open(path, 'r') as f:
                episodes = json.load(f)
                self.episodes.extend(episodes)
        except FileNotFoundError:
            logger.warning(f"Buffer file not found: {path}")


def save_checkpoint(model, tokenizer, buffer, epoch: int, output_dir: str):
    """Save training checkpoint"""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save buffer
    buffer.save(checkpoint_dir / "buffer.json")

    logger.info(f"Checkpoint saved at {checkpoint_dir}")


def save_final_model(model, tokenizer, output_dir: str):
    """Save final trained model"""
    final_dir = Path(output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info(f"Final model saved at {final_dir}")


def save_metrics(metrics: List[Dict], output_dir: str):
    """Save training metrics"""
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved at {metrics_path}")


def prepare_ppo_batch(episodes: List[Dict], tokenizer, device: torch.device, max_length: int = 256):
    """Prepare batch for PPO training"""
    queries = []
    responses = []
    rewards = []

    for episode in episodes:
        # Use conversation states as queries and actions as responses
        for i in range(len(episode["states"])):
            queries.append(episode["states"][i])
            responses.append(episode["actions"][i])
            # Fixed: Access individual rewards for each step
            rewards.append(episode["rewards"][i] if "rewards" in episode else episode.get("episode_reward", 0.0))

    if not queries:
        return None, None, None

    # Tokenize
    query_tensors = []
    response_tensors = []

    for q, r in zip(queries, responses):
        q_tokens = tokenizer.encode(q, return_tensors="pt", truncation=True, max_length=max_length)
        r_tokens = tokenizer.encode(r, return_tensors="pt", truncation=True, max_length=64)
        query_tensors.append(q_tokens.squeeze())
        response_tensors.append(r_tokens.squeeze())

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)

    return query_tensors, response_tensors, rewards_tensor