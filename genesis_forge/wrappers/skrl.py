import torch
from gymnasium import spaces
from typing import Any, Tuple

from skrl.envs.wrappers.torch.base import Wrapper as SkrlWrapper
from genesis_forge.wrappers.wrapper import Wrapper as GenesisWrapper


class SkrlEnvWapper(SkrlWrapper, GenesisWrapper):
    """
    A wrapper that makes your genesis forge environment compatible with the skrl training framework.
    """

    can_be_wrapped = False

    @property
    def action_space(self) -> spaces:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> spaces:
        """The observation space of the environment."""
        return self._env.observation_space

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment"""
        return self._env.reset()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        Args:
            actions: The actions to perform

        Returns:
            obs: The observations
            rewards: The rewards
            terminations: The terminations
            timeouts: The timeouts
            extras: The extras
        """
        obs, rewards, terminations, timeouts, extras = self._env.step(actions)

        # Expand rewards, terminations and timeouts to the shape (num_envs, 1)
        rewards = rewards.unsqueeze(1)
        terminations = terminations.unsqueeze(1)
        timeouts = timeouts.unsqueeze(1)

        return obs, rewards, terminations, timeouts, extras

    def render(self, *args, **kwargs) -> Any:
        """
        Not implemented for Genesis Forge environments.
        """
        pass

    def close(self) -> None:
        """Close the environment"""
        return self._env.close()

    def build(self) -> None:
        """Build the environment"""
        self._env.build()
