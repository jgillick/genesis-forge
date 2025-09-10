"""
Simplified Spider Robot Environment with Curriculum Learning
Focuses on core objectives with progressive difficulty
"""

import math
import torch
import genesis as gs
from gymnasium import spaces
from genesis.engine.entities import RigidEntity
from typing import Any, Literal

EnvMode = Literal["train", "eval", "play"]


class GenesisEnv:
    """
    Base vectorized environment for Genesis.

    Args:
        num_envs: Number of parallel environments.
        dt: Simulation time step.
        max_episode_length_sec: Maximum episode length in seconds.
        max_episode_random_scaling: Scale the maximum episode length by this amount (+/-) so that not all environments reset at the same time.
        headless: Whether to run the environment in headless mode.
        extras_logging_key: The key used, in info/extras dict, which is returned by step and reset functions, to send data to tensorboard by the RL agent.
    """

    action_space: spaces.Space | None = None
    observation_space: spaces.Space | None = None
    can_be_wrapped: bool = True

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_sec: int | None = 10,
        max_episode_random_scaling: float = 0.0,
        headless: bool = True,
        mode: EnvMode = "train",
        extras_logging_key: str = "episode",
    ):
        self.dt = dt
        self.device = gs.device
        self.num_envs = num_envs
        self.headless = headless
        self.mode = mode
        self.scene: gs.Scene = None
        self.robot: RigidEntity = None
        self.terrain: RigidEntity = None

        self.extras_logging_key = extras_logging_key
        self._extras = {}
        self._extras[extras_logging_key] = {}

        self._actions: torch.Tensor = None
        self.last_actions: torch.Tensor = None

        self.step_count: int = 0
        self.episode_length = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.int32
        )
        self.max_episode_length: torch.Tensor = None

        self._max_episode_length_sec = 0.0
        self._max_episode_random_scaling = 0.0
        self._base_max_episode_length = None
        if max_episode_length_sec and max_episode_length_sec > 0:
            self._max_episode_random_scaling = max_episode_random_scaling / self.dt
            self.max_episode_length = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_int
            )
            self.max_episode_length[:] = self.set_max_episode_length(
                max_episode_length_sec
            )

    """
    Properties
    """

    @property
    def unwrapped(self):
        """Returns the base non-wrapped environment.

        Returns:
            Env: The base non-wrapped :class:`GenesisEnv` instance
        """
        return self

    @property
    def max_episode_length_sec(self) -> int | None:
        """The max episode length, in seconds, for each environment."""
        return self._max_episode_length_sec

    @property
    def extras(self) -> dict:
        """
        The extras/infos dictionary that should be returned by the step and reset functions.
        This dictionary will be cleared at the start of every step.
        """
        return self._extras

    @property
    def actions(self) -> torch.Tensor:
        """The current actions for each environment for this step."""
        return self._actions

    @actions.setter
    def actions(self, actions: torch.Tensor):
        """Set the actions for each environment for this step."""
        self._actions = actions

    @property
    def num_actions(self) -> int:
        """The number of actions for each environment."""
        if self.action_space is not None:
            return self.action_space.shape[0]
        return 0

    @property
    def num_observations(self) -> int:
        """The number of observations for each environment."""
        if self.observation_space is not None:
            return self.observation_space.shape[0]
        return 0

    """
    Utilities
    """

    def set_max_episode_length(self, max_episode_length_sec: int) -> int:
        """
        Set or change the maximum episode length.

        Args:
            max_episode_length_sec: The maximum episode length in seconds.

        Returns:
            The maximum episode length in steps.
        """
        self._max_episode_length_sec = max_episode_length_sec
        self._base_max_episode_length = math.ceil(max_episode_length_sec / self.dt)
        return self._base_max_episode_length

    """
    Operations
    """

    def build(self) -> None:
        """
        Builds the scene and other supporting components necessary for the training environment.
        This assumes that the scene has already been constructed and assigned to the <env>.scene attribute.
        """
        assert (
            self.scene is not None
        ), "The scene must be constructed and assigned to the <env>.scene attribute before building."
        self.scene.build(n_envs=self.num_envs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Take an action for each parallel environment.

        Args:
            actions: Batch of actions with the :attr:`action_space` shape.

        Returns:
            Batch of (observations, rewards, terminations, truncations, infos/extras)
        """
        self._extras = {}
        self._extras[self.extras_logging_key] = {}
        self.step_count += 1
        self.episode_length += 1

        self.last_actions[:] = self.actions[:]
        self._actions = actions

        # Clip actions within bounds of action space
        if self.action_space is not None:
            self._actions = self.actions.clamp(
                torch.tensor(self.action_space.low, device=gs.device),
                torch.tensor(self.action_space.high, device=gs.device),
            )

        return None, None, None, None, self._extras

    def reset(
        self,
        envs_idx: list[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or all parallel environments.

        Args:
            envs_idx: The environment ids to reset. If None, all environments are reset.

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        # Initial reset, set buffers
        if self.step_count == 0:
            self.actions = torch.zeros(
                (self.num_envs, self.action_space.shape[0]),
                device=gs.device,
                dtype=gs.tc_float,
            )
            self.last_actions = torch.zeros_like(self.actions, device=gs.device)

        # Actions
        if envs_idx.numel() > 0:
            self.actions[envs_idx] = 0.0
            self.last_actions[envs_idx] = 0.0

            # Episode length
            self.episode_length[envs_idx] = 0

        # Randomize max episode length for env_ids
        if (
            len(envs_idx) > 0
            and self._max_episode_random_scaling > 0.0
            and self._base_max_episode_length
        ):
            scale = torch.rand((envs_idx.numel(),)) * self._max_episode_random_scaling
            self.max_episode_length[envs_idx] = torch.round(
                self._base_max_episode_length + scale
            ).to(gs.tc_int)

        return None, self.extras

    def get_observations(self) -> torch.Tensor:
        """
        Return the current observations for each environment.
        """
        if self.observation_space is not None:
            return torch.zeros(
                (self.num_envs, self.observation_space.shape[0]),
                device=gs.device,
                dtype=gs.tc_float,
            )
        return None

    def close(self):
        """Close the environment."""
        pass

    def render(self):
        """Not implemented."""
        pass
