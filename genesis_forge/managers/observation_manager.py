import re
import torch
import numpy as np
from gymnasium import spaces
import genesis as gs
from typing import TypedDict, Tuple, Callable, Any
from genesis.engine.entities import RigidEntity

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


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
        noise: tuple[float, float] | None = None,
    ):
        """
        Args:
            env: The environment.
            cfg: The configuration for the observation manager.
            noise: The range of random noise to add to all observations.
        """
        super().__init__(env, "observation")
        self.cfg = cfg
        self.noise = noise
        self._observation_size = 1
        self._observation_space = None

    @property
    def observation_space(self) -> spaces.Space:
        """The observation space."""
        return self._observation_space

    def build(self):
        """Make an initial observation in order to determine the observation space"""
        if not self.enabled:
            self._observation_size = 1
            self._observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            )
            return

        obs = self.get_observations()
        self._observation_size = obs.shape[1]
        self._observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._observation_size,),
            dtype=np.float32,
        )

    def get_observations(self) -> torch.Tensor:
        """Generate current observations for all environments."""
        if not self.enabled:
            return torch.zeros((self.env.num_envs, self._observation_size))

        obs = []
        for name, cfg in self.cfg.items():
            try:
                fn = cfg["fn"]
                params = cfg.get("params", dict())
                noise = cfg.get("noise", self.noise)
                scale = cfg.get("scale", 1.0)

                # Get values
                assert callable(fn), f"Observation function {name} is not callable"
                value = fn(env=self.env, **params)

                # Convert to tensor, if necessary
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, device=gs.device, dtype=gs.tc_float)

                # Apply scale
                if scale is not None and scale != 1.0:
                    value = value * scale

                # Add noise
                if noise is not None and noise != 0.0:
                    noise_value = torch.empty_like(value).uniform_(-1, 1) * noise
                    value = value + noise_value

                obs.append(value)
            except Exception as e:
                print(f"Error generating observation for '{name}'")
                raise e

        return torch.cat(obs, dim=-1)

    def step(self):
        """Called when the environment is stepped"""
        pass

    def reset(self, env_ids: list[int] | None = None):
        """One or more environments have been reset"""
        pass
