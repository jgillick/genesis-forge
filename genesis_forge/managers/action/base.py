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
        self._raw_actions = None
        self._actions = None

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
            low=-np.inf,
            high=np.inf,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    @property
    def actions(self) -> torch.Tensor:
        """
        The actions for for the current step.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions
    
    @property
    def raw_actions(self) -> torch.Tensor:
        """
        The actions received from the policy, before being converted.
        """
        if self._raw_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._raw_actions

    def step(self, actions: torch.Tensor) -> None:
        """
        Handle the received actions.
        """
        # Copy the actions into the manager buffer
        if self._actions is None:
            self._actions = actions.detach().clone()
            self._raw_actions = actions.detach().clone()
        else:
            self._actions[:] = actions[:]
            self._raw_actions[:] = actions[:]
        return self._actions

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
