"""
Go2 Gait Learning Example using Periodic Reward Composition.

This example implements natural gait learning for quadruped robots
based on the paper "Sim-to-Real Learning of All Common Bipedal Gaits
via Periodic Reward Composition" (Siekmann et al., 2020).
"""

from .environment import Go2GaitEnv
from .gait_command_manager import GaitCommandManager
from . import periodic_rewards

__all__ = [
    "Go2GaitEnv",
    "GaitCommandManager",
    "periodic_rewards",
]
