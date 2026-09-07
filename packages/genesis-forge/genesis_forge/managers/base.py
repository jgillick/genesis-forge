from __future__ import annotations

from typing import Literal

import torch

from genesis_forge.genesis_env import GenesisEnv

ManagerType = Literal[
    "action",
    "actuator",
    "reward",
    "termination",
    "contact",
    "terrain",
    "entity",
    "command",
    "observation",
]


class BaseManager:
    """
    The base class used to define the interface for all other managers
    """

    def __init__(
        self,
        env: GenesisEnv,
        type: ManagerType,
        enabled: bool = True,
    ):
        self.env = env
        self.enabled = True
        self.type = type
        if hasattr(env, "add_manager"):
            env.add_manager(type, self)

    def build(self):
        """Called when the scene is built"""

    def step(self):
        """Called when the environment is stepped"""

    def reset(self, envs_idx: torch.Tensor | None = None):
        """One or more environments have been reset"""
