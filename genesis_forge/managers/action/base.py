import torch
import numpy as np
from gymnasium import spaces

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class BaseActionManager(BaseManager):
    """
    Base for managers that handle actions.

    Args:
        env: The environment to manage the DOF actuators for.
    """

    def __init__(self, env: GenesisEnv):
        super().__init__(env, type="action")
        self._actions = None
        self._last_actions = None

    """
    Properties
    """

    @property
    def num_actions(self) -> int:
        """
        The total number of actions.
        """
        return 0

    @property
    def action_space(self) -> tuple[float, float]:
        """
        If using the default action handler, the action space is [-1, 1].
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    def step(self, actions: torch.Tensor) -> None:
        """
        Handle the received actions.
        """
        self._last_actions = self._actions
        self._actions = actions

    def reset(self, envs_idx: list[int] | None):
        """Reset environments."""
        pass

    def get_actions(self) -> torch.Tensor:
        """
        Get the current actions for the environments.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions
