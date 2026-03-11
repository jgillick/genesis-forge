import torch
import numpy as np
from gymnasium import spaces
import genesis as gs
from typing import TypedDict, Callable, Any
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.config import ObservationConfigItem
from genesis_forge.rolling_buffer import RollingBuffer


class ObservationConfig(TypedDict):
    """Defines an observation item."""

    fn: Callable[[GenesisEnv, ...], torch.Tensor]
    """Function that will be called to generate an observation, returning a value for each environment."""

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    scale: float | None
    """The scale to apply to the observation. If None, no scale will be applied."""

    noise: float | None
    """The noise scale to add to the observation. If None, no noise will be added.
    This will randomly choose a number between -1 and 1, multiply it by the noise scale, and add the result to the observation values."""


class ObservationManager(BaseManager):
    """
    Defines the observations and observation space for the environment.

    Args:
        env: The environment.
        cfg: The configuration for the observation manager.
        name: The name to categorize the observations under, generally used for asymmetrical RL.
              It's required to have one observation manager named "policy".
        noise: The range of random noise to add to all observations.
        history_len: The number of previous observations to include in the observation.

    Example with ManagedEnvironment::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            config(self):
                ObservationManager(
                    self,
                    cfg={
                        "velocity_cmd": {"fn": self.velocity_command.observation},
                        "robot_ang_vel": {
                            "fn": utils.entity_ang_vel,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_lin_vel": {
                            "fn": utils.entity_lin_vel,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_projected_gravity": {
                            "fn": utils.entity_projected_gravity,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_dofs_position": {
                            "fn": self.action_manager.get_dofs_position,
                            "noise": 0.01,
                        },
                        "actions": {"fn": lambda: env.actions},
                    },
                )

    Example using the observation manager directly::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.observation_manager = ObservationManager(
                    self,
                    cfg={
                        "velocity_cmd": {"fn": self.velocity_command.observation},
                        "robot_ang_vel": {
                            "fn": utils.entity_ang_vel,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_lin_vel": {
                            "fn": utils.entity_lin_vel,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_projected_gravity": {
                            "fn": utils.entity_projected_gravity,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_dofs_position": {
                            "fn": self.action_manager.get_dofs_position,
                            "noise": 0.01,
                        },
                        "actions": {"fn": lambda: env.actions},
                    },
                )

            @property
            observation_space(self):
                return self.obs_manager.observation_space

            def build(self):
                super().build()
                self.obs_manager.build()

            def step(self, actions: torch.Tensor):
                super().step(actions)

                # ... step logic ...

                obs = self.observation_manager.observation()
                return obs, rewards, terminations, timeouts, info

            def reset(self, envs_idx: list[int] | None = None):
                super().reset(envs_idx)

                # ... reset logic ...

                obs = self.observation_manager.observation()
                return obs, info

    """

    def __init__(
        self,
        env: GenesisEnv,
        cfg: dict[str, ObservationConfig],
        name: str = "policy",
        history_len: int | None = None,
        noise: tuple[float, float] | None = None,
    ):
        super().__init__(env, "observation")
        self._name = name
        self.cfg = cfg
        self.noise = noise
        self._observation_size = 1
        self._observation_space = None

        if history_len is not None and history_len < 1:
            raise ValueError("history_len must be greater than 0")
        self._history_len = history_len if history_len is not None else 1
        self._history = []

        # Per-observation slot dimensions; populated during build().
        # Each entry is (observation_name, dim) in the same order as self.cfg.
        self._observation_dims: list[tuple[str, int]] = []

        # Wrap config items
        self.cfg: dict[str, ObservationConfigItem] = {}
        for name, cfg in cfg.items():
            self.cfg[name] = ObservationConfigItem(cfg, env)

    """
    Properties
    """

    @property
    def name(self) -> str:
        """
        The name to categorize the observations under
        This is generally used for asymmetrical RL and it's required to have
        one observation manager named "policy".
        """
        return self._name

    @property
    def observation_space(self) -> spaces.Space:
        """The observation space."""
        return self._observation_space

    """
    Public methods
    """

    def build(self):
        """
        Determine the observation space and setup the buffers.
        """
        if not self.enabled:
            self._observation_size = 1
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            )
            return

        # Setup observation functions and the observation space
        single_obs_size = self._setup_observation_functions()
        self._observation_size = single_obs_size * self._history_len
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._observation_size,),
            dtype=np.float32,
        )

        # Fill history buffer
        init_frame = torch.zeros((self.env.num_envs, single_obs_size), device=gs.device)
        self._history = RollingBuffer(self._history_len, init_frame)
        self._history_output = torch.zeros(
            (self.env.num_envs, self._observation_size),
            device=gs.device,
        )

    def get_observations(
        self, values: dict[str, float | torch.Tensor] | None = None
    ) -> torch.Tensor:
        """
        Generate current observations for all environments.

        When you deploy to a real robot, you can provide the observations directly as a dictionary of values,
        and this method will return the formatted & scaled tensor that you can pass to the policy.

        Args:
            values: (optional) If provided, use these values instead of fetching observations from the config functions.
                    It's expected that this dict contains a key for every observation configuration.
                    These values will be scaled, based on the configuration, but not receive any noise.
                    This is useful for providing observations for deployment.

        Returns:
            The observations for all environments.
        """
        if not self.enabled:
            return torch.zeros((self.env.num_envs, self._observation_size))

        buffer = self._history.rotate()
        self._perform_observation(buffer, values)
        self._history.push(buffer)
        self._history.output(self._history_output)
        return self._history_output.clone()

    """
    Private methods.
    """

    def _setup_observation_functions(self) -> int:
        """Build all the observation function classes, and determine the observation space."""
        size = 0
        self._observation_dims = []
        for name, cfg in self.cfg.items():
            try:
                cfg.build()
                assert callable(cfg.fn), f"Observation function {name} is not callable"
                value = cfg.fn(env=self.env, **cfg.params)
                value_size = value.shape[-1]
                if value_size > 0:
                    size += value_size
                    self._observation_dims.append((name, value_size))
            except Exception as e:
                print(f"Error generating observation for '{name}'")
                raise e
        return size

    def _perform_observation(
        self,
        output: torch.Tensor,
        override_values: dict[str, float | torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Perform a round of observations.

        Args:
            output: The output tensor to fill with the observations.
        """
        offset = 0
        has_overrides = override_values is not None
        for name, cfg in self.cfg.items():
            try:
                # Get values
                params = cfg.params
                if override_values is not None:
                    if name not in override_values:
                        raise ValueError(f"Value '{name}' not found in override values")
                    value = override_values[name]
                    if not isinstance(value, torch.Tensor):
                        value = torch.tensor(value, device=gs.device)
                else:
                    value = cfg.fn(env=self.env, **params)

                # Apply scale
                scale = cfg.scale
                if scale is not None and scale != 1.0:
                    value *= scale

                # Add noise, if the value is not an override
                if not has_overrides:
                    noise = cfg.noise or self.noise
                    if noise is not None and noise != 0.0:
                        noise_value = torch.empty_like(value).uniform_(-1, 1) * noise
                        value += noise_value

                # Copy directly into output buffer
                value_size = value.shape[-1]
                if value_size > 0:
                    output[:, offset : offset + value_size] = value
                    offset += value_size
            except Exception as e:
                print(f"Error generating observation for '{name}'")
                raise e
        return output
