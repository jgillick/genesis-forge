from __future__ import annotations
import math
import torch
import genesis as gs
from gymnasium import spaces
from genesis.engine.entities import RigidEntity
from typing import Any, Literal

EnvMode = Literal["train", "eval", "play"]


class GenesisEnv:
    """
    Base environment class for your simulated robot environment.

    Args:
        num_envs: Number of parallel environments.
        dt: Simulation time step.
        max_episode_length_sec: Maximum episode length in seconds.
        max_episode_random_scaling: Scale the maximum episode length by this amount (+/-) so that not all environments reset at the same time.
        extras_logging_key: The key used, in info/extras dict, which is returned by step and reset functions, to send data to tensorboard by the RL agent.

    Example::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # ...Define scene here...
                self.scene = gs.Scene()
                self.terrain = self.scene.add_entity(gs.morphs.Plane())
                self.robot = self.scene.add_entity( ... )

            def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
                # ...step logic here...
                return obs, rewards, terminations, truncations, info

            def reset(self, envs_idx: list[int] = None) -> tuple[torch.Tensor, dict[str, Any]]:
                # ...reset logic here...
                return obs, info

            def get_observations(self) -> torch.Tensor:
                # ...define current observations here...
                return obs

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
        extras_logging_key: str = "episode",
    ):
        self.dt = dt
        self.device = gs.device
        self.num_envs = num_envs
        self.scene: gs.Scene = None
        self.robot: RigidEntity = None
        self.terrain: RigidEntity = None

        self.extras_logging_key = extras_logging_key
        self._extras = {}
        self._extras[extras_logging_key] = {}

        self._actions: torch.Tensor = None
        self._last_actions: torch.Tensor = None

        self.step_count: int = 0
        self.episode_length = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.int32
        )
        self.max_episode_length: torch.Tensor = None

        self._max_episode_length_sec = 0.0
        self._base_max_episode_length = None
        self._max_episode_random_scaling = max_episode_random_scaling
        if max_episode_length_sec and max_episode_length_sec > 0:
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
        """Returns this environment, not a wrapped version of it."""
        return self

    @property
    def max_episode_length_sec(self) -> int | None:
        """The max episode length, in seconds, for each environment."""
        return self._max_episode_length_sec

    @property
    def extras(self) -> dict:
        """
        The extras/infos dictionary reset at the start of every step, and contains additional data about the environment during that step.
        """
        return self._extras

    @property
    def actions(self) -> torch.Tensor:
        """
        The actions for each environment for this step.
        If you're using an action manager, these are the actions prior to being handled by the action manager.
        """
        return self._actions

    @property
    def last_actions(self) -> torch.Tensor:
        """
        The actions for for the previous step.
        """
        return self._last_actions

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
    
    @property
    def max_episode_length_steps(self) -> int | None:
        """
        The max episode length, in steps, for each environment.
        If episode randomization scaling is enabled, this will be the base max episode length before scaling.
        """
        return self._base_max_episode_length

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
        Builds the environment before the first step.
        The Genesis scene and all the scene entities must be added before calling this method.
        """
        assert (
            self.scene is not None
        ), "The scene must be constructed and assigned to the <env>.scene attribute before building."
        self.scene.build(n_envs=self.num_envs)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Performs a step in all environments with the given actions.

        Args:
            actions: Batch of actions for each environment with the :attr:`action_space` shape.

        Returns:
            Batch of (observations, rewards, terminations, truncations, info/extras)
        """
        self._extras = {}
        self._extras[self.extras_logging_key] = {}
        self.step_count += 1
        self.episode_length += 1

        if self._actions is None:
            self._actions = actions.detach().clone()
            self._last_actions = torch.zeros_like(actions, device=gs.device)
        else:
            self._last_actions[:] = self._actions[:]
            self._actions[:] = actions[:]

        return None, None, None, None, self._extras

    def reset(
        self,
        envs_idx: list[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or more environments.
        Each of the registered managers will also be reset for those environments.

        Args:
            env_ids: The environment ids to reset. If None, all environments are reset.

        Returns:
            A batch of observations and info from the vectorized environment.
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        # Initial reset, set buffers
        if self.step_count == 0 and self.action_space is not None:
            self._actions = torch.zeros(
                (self.num_envs, self.action_space.shape[0]),
                device=gs.device,
                dtype=gs.tc_float,
            )
            self._last_actions = torch.zeros_like(self.actions, device=gs.device)

        # Actions
        if envs_idx.numel() > 0:
            if self.actions is not None:
                self.actions[envs_idx] = 0.0
                self._last_actions[envs_idx] = 0.0

            # Episode length
            self.episode_length[envs_idx] = 0

        # Randomize max episode length for env_ids
        if (
            len(envs_idx) > 0
            and self._max_episode_random_scaling > 0.0
            and self._base_max_episode_length is not None
        ):
            max_random_scaling = self._base_max_episode_length * self._max_episode_random_scaling
            randomization = torch.empty((envs_idx.numel(),)).uniform_(-1.0, 1.0) * max_random_scaling 
            self.max_episode_length[envs_idx] = torch.round(
                self._base_max_episode_length + randomization
            ).to(gs.tc_int)

        return None, self.extras

    def get_observations(self) -> torch.Tensor:
        """
        Returns the current observations for each environment.
        Override this method to return the observations for your environment.

        Example::

            def get_observations(self) -> torch.Tensor:
                return torch.cat(
                [
                    self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                    self.projected_gravity,  # 3
                    self.commands * self.commands_scale,  # 3
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                    self.dof_vel * self.obs_scales["dof_vel"],  # 12
                    self.actions,  # 12
                ],
                axis=-1,
            )
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
