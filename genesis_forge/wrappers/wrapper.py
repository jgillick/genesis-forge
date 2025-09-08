from typing import Any, TypeVar, Sequence, Callable

import torch
import genesis as gs
from gymnasium import spaces
from genesis_forge.genesis_env import GenesisEnv

RenderFrame = TypeVar("RenderFrame")


class Wrapper:
    """
    The core wrapper class that provides the basic functionality for all wrappers.
    """

    env: GenesisEnv = None

    def __init__(self, env: GenesisEnv):
        """Initialize the logging wrapper with the function to use for data logging."""
        self.env = env
        if not isinstance(env, GenesisEnv) and not isinstance(env, Wrapper):
            raise ValueError(
                f"Expected env to be a `GenesisEnv` or `Wrapper` but got {type(env)}"
            )

    """
    Properties
    """

    @property
    def dt(self) -> float:
        """The time step of the environment."""
        return self.env.dt

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self.env.num_envs

    @property
    def scene(self) -> gs.Scene:
        """Get the environment scene."""
        return self.env.scene

    @property
    def robot(self) -> Any:
        """Get the environment robot."""
        return self.env.robot

    @property
    def action_space(self) -> spaces:
        """The action space of the environment."""
        return self.env.action_space

    @property
    def observation_space(self) -> spaces:
        """The observation space of the environment."""
        return self.env.observation_space

    @property
    def unwrapped(self) -> GenesisEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`GenesisEnv` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Operations
    """

    def build(self) -> None:
        """
        Builds the scene and other supporting components necessary for the training environment.
        This assumes that the scene has already been constructed and assigned to the <env>.scene attribute.
        """
        self.env.build()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.step(actions)

    def reset(
        self,
        env_ids: Sequence[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.reset(env_ids)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self.env.render()

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        return self.env.close()
