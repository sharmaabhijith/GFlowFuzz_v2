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


# Note: save_checkpoint and save_final_model have been moved to algorithms/ppo.py
# These functions are now part of the PPOAlgorithm class for better encapsulation


def save_metrics(metrics: List[Dict], output_dir: str):
    """Save training metrics"""
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved at {metrics_path}")