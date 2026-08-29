import torch
import numpy as np
import collections
from gymnasium import spaces
import genesis as gs
from typing import TypedDict, Callable, Any
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.config import ObservationConfigItem


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
        obs_to_discard: list[str]=None,
        noise: tuple[float, float] | None = None,
    ):
        super().__init__(env, "observation")
        self._name = name
        self.cfg = cfg
        self.noise = noise
        self.obs_to_discard=obs_to_discard if obs_to_discard is not None else []
        self._observation_size = 1
        self._observation_space = None

        if history_len is not None and history_len < 1:
            raise ValueError("history_len must be greater than 0")
        self._history_len = history_len if history_len is not None else 1
        self._history = []

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

        # Build any config item function classes.
        for name, cfg in self.cfg.items():
            cfg.build()
            assert callable(cfg.fn), f"Observation function {name} is not callable"

        # Make an initial observation and create the observation space
        obs = self._perform_observation(single_agent=True)
        single_obs_size=0
        if len(self.obs_to_discard)>0:
            for key,value in obs.items(): 
                if key not in self.obs_to_discard:
                    single_obs_size +=value.shape[0]
        else:
            for key,value in obs.items():
                if value.ndim<=2:
                    single_obs_size +=value.shape[0]
        
        self._observation_space=self._spec_to_space(obs)
        
        if self._history_len>1:
            self._observation_size = single_obs_size * self._history_len

            # Fill history buffer
            shape = (self.env.num_envs, single_obs_size)
            self._history = [
                torch.zeros(shape, device=gs.device) for _ in range(self._history_len)
            ]

    def get_observations(self) -> torch.Tensor:
        """Generate current observations for all environments."""
        if not self.enabled:
            return torch.zeros((self.env.num_envs, self._observation_size))
        
        obs = self._perform_observation()
        obs_flattened=torch.cat([
            obs_value.view(self.env.num_envs, -1) if obs_value.dim() != 2 else obs_value 
            for obs_value in obs.values()
        ],dim=-1)
        # only use the _history if _history_len is grater than 1
        if self._history_len>1:
            self._history.pop()
            self._history.insert(0, obs_flattened)
            return torch.cat(self._history, dim=-1)
        return obs_flattened

    """
    Private methods.
    """

    def _perform_observation(self,single_agent=False) -> torch.Tensor:
        """Perform a round of observations."""
        obs = {}
        for name, cfg in self.cfg.items():
            try:
                # Get values
                params = cfg.params
                value = cfg.fn(env=self.env, **params)

                # Apply scale
                scale = cfg.scale
                if scale is not None and scale != 1.0:
                    value *= scale

                # Add noise
                noise = cfg.noise or self.noise
                if noise is not None and noise != 0.0:
                    noise_value = torch.empty_like(value).uniform_(-1, 1) * noise
                    value += noise_value
                obs[name]=value[0] if single_agent else value
            except Exception as e:
                print(f"Error generating observation for '{name}'")
                raise e
        return collections.OrderedDict(obs)

    def _spec_to_space(self, spec: Any) -> spaces.Space:
        """returns an gymnasium.space action space based on the observation """
        if type(spec) is tuple:
            return spaces.Box(shape=spec[0].shape, dtype=np.float64, low=spec[0], high=spec[1])
        elif isinstance(spec, np.ndarray):
            return spaces.Box(
                shape=spec.shape,
                dtype=np.float64,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, torch.Tensor):
            spec.detach().cpu().numpy()
            return spaces.Box(
                shape=spec.shape,
                dtype=np.float64,
                low=np.full(spec.shape, float("-inf")),
                high=np.full(spec.shape, float("inf")),
            )
        elif isinstance(spec, collections.OrderedDict):
            return spaces.Dict({k: self._spec_to_space(v) for k, v in spec.items()})
        else:
            raise ValueError(f"Spec type {type(spec)} not supported. Please report this issue")