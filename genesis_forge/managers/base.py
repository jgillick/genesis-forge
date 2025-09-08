from genesis_forge.genesis_env import GenesisEnv
from typing import Literal

ManagerType = Literal[
    "action",
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
    A base manager describing the interface for all other managers
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
        pass

    def step(self):
        """Called when the environment is stepped"""
        pass

    def reset(self, envs_idx: list[int] | None = None):
        """One or more environments have been reset"""
        pass
