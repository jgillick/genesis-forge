from typing import Any

import torch
from gymnasium import spaces
from skrl.envs.wrappers.torch.base import Wrapper as SkrlWrapper

from genesis_forge.wrappers.wrapper import Wrapper as GenesisWrapper


class SkrlEnvWrapper(SkrlWrapper, GenesisWrapper):
    """
    A wrapper that makes your genesis forge environment compatible with the skrl training framework.
    """

    can_be_wrapped = False

    @property
    def action_space(self) -> spaces.Space:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> spaces.Space:
        """The observation space of the environment."""
        return self._env.observation_space

    def reset(self) -> tuple[torch.Tensor, Any]:
        """Reset the environment

        Raises:
            NotImplementedError: Not implemented

        Returns:
            tuple: Observation (tensor), info (dict)
        """
        return self._env.reset()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        Args:
            actions: The actions to perform

        Returns:
            tuple of tensors and a dict:
                Observation (tensor) , reward (tensor), terminated (tensor), truncated (tensor), info (dict)
        """
        obs, rewards, terminations, timeouts, extras = self._env.step(actions)

        # Expand rewards, terminations and timeouts to the shape (num_envs, 1)
        rewards = rewards.unsqueeze(1)
        terminations = terminations.unsqueeze(1)
        timeouts = timeouts.unsqueeze(1)

        return obs, rewards, terminations, timeouts, extras

    def state(self) -> torch.Tensor:
        """Get the environment state

        Returns:
            State (torch.Tensor)
        """
        return self.env.state()

    def render(self, *args, **kwargs) -> Any:
        """
        Not implemented for Genesis Forge environments.
        """

    def close(self) -> None:
        """Close the environment"""
        return self._env.close()

    def build(self) -> None:
        """Build the environment"""
        self._env.build()
