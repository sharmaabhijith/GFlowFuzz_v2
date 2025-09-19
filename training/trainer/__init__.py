#!/usr/bin/env python3
"""
Reinforcement Learning Algorithms Module

This module provides modular implementations of various RL algorithms
for training conversational agents. The design allows easy addition
of new algorithms without modifying the core training infrastructure.

Available Algorithms:
- PPO (Proximal Policy Optimization): Stable policy gradient method
- [Future] DPO (Direct Preference Optimization)
- [Future] REINFORCE: Simple policy gradient
- [Future] A2C (Advantage Actor-Critic)
"""

from .ppo import PPOAlgorithm, collect_trajectories

__all__ = [
    'PPOAlgorithm',
    'collect_trajectories'
]