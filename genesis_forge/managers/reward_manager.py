import torch
import genesis as gs
from typing import TypedDict, Callable, Any

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class RewardConfig(TypedDict):
    """Defines a reward item."""

    fn: Callable[[GenesisEnv, ...], torch.Tensor]
    """Function that will be called to calculate a reward for the environments."""

    params: dict[str, Any]
    """Additional parameters to pass to the function."""

    weight: float
    """The weight of the reward item."""


class RewardManager(BaseManager):
    """
    Handles calculating and logging the rewards for the environment.

    This works with a dictionary configuration of reward handlers. For each dictionary item,
    a function will be called to calculate a reward value for the environment.

    Args:
        env: The environment to manage the rewards for.
        reward_cfg: A dictionary of reward conditions.
        logging_enabled: Whether to log the rewards to tensorboard.
        logging_tag: The section name used to log the rewards to tensorboard.

    Example with ManagedEnvironment::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.reward_manager = RewardManager(
                    self,
                    cfg={
                        "Default pose": {
                            "fn": mdp.rewards.dof_similar_to_default,
                            "weight": -0.1,
                        },
                        "Base height": {
                            "fn": mdp.rewards.base_height,
                            "params": { "target_height": 0.135 },
                            "weight": -100.0,
                        },
                    },
                )

    Example using the reward manager directly::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.reward_manager = RewardManager(
                    self,
                    cfg={
                        "Base height": {
                            "fn": mdp.rewards.base_height,
                            "params": { "target_height": 0.135 },
                            "weight": -100.0,
                        },
                    },
                )

            def build(self):
                super().build()
                self.reward_manager.build()

            def step(self, actions: torch.Tensor):
                super().step(actions)
                rewards = self.reward_manager.step()
                # ... other step logic ...
                return obs, rewards, terminations, timeouts, info

            def reset(self, envs_idx: list[int] | None = None):
                super().reset(envs_idx)
                # ... other reset logic ...
                return obs, info

    """

    def __init__(
        self,
        env: GenesisEnv,
        cfg: dict[str, RewardConfig],
        logging_enabled: bool = True,
        logging_tag: str = "Rewards",
    ):
        super().__init__(env, type="reward")

        self.cfg = cfg
        self.logging_enabled = logging_enabled
        self.logging_tag = logging_tag

        # Initialize buffers
        self._reward_buf = torch.zeros(
            (env.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self._episode_length = torch.zeros(
            (self.env.num_envs,), device=gs.device, dtype=torch.int32
        )
        self._episode_data: dict[str, torch.Tensor] = dict()
        for name in self.cfg.keys():
            self._episode_data[name] = torch.zeros(
                (env.num_envs,), device=gs.device, dtype=gs.tc_float
            )

    """
    Operations
    """

    def step(self) -> torch.Tensor:
        """
        Calculate the rewards for this step

        Returns:
            The rewards for the environments. Shape is (num_envs,).
        """
        self._reward_buf[:] = 0.0
        self._episode_length += 1
        if not self.enabled:
            return self._reward_buf

        dt = self.env.dt
        for name, cfg in self.cfg.items():
            fn = cfg["fn"]
            weight = cfg.get("weight", 0.0)
            params = cfg.get("params", dict())

            # Don't calculate reward if the weight is zero
            if weight == 0:
                continue

            # Get reward value from function
            value = fn(self.env, **params).detach()
            if weight != 1.0:
                value = value * weight
            if dt != 1.0:
                value = value * dt

            # Add to reward buffer
            self._reward_buf += value

            # Add to episode data for logging (if enabled)
            if self.logging_enabled:
                self._episode_data[name] += value

        return self._reward_buf

    def reset(self, envs_idx: list[int] | None = None):
        """Log the reward mean values at the end of the episode"""
        if envs_idx is None:
            envs_idx = torch.arange(self.env.num_envs, device=gs.device)

        if self.enabled and self.logging_enabled:
            logging_dict = self.env.extras[self.env.extras_logging_key]

            # Get episode lengths for the reset environments
            episode_lengths = self._episode_length[envs_idx]
            valid_episodes = episode_lengths > 0

            # Only proceed if there are valid episodes to log
            if torch.any(valid_episodes):
                for name, value in self._episode_data.items():
                    # Get values for reset environments
                    episode_values = value[envs_idx]

                    # Calculate episode averages efficiently
                    # Use in-place operations to avoid creating temporary tensors
                    episode_avg = episode_values.clone()
                    episode_avg[valid_episodes] /= episode_lengths[valid_episodes]

                    # Calculate mean across valid episodes only
                    episode_mean = episode_avg[valid_episodes].mean()
                    logging_dict[f"{self.logging_tag} / {name}"] = episode_mean

                # Reset episodic data
                self._episode_data[name][envs_idx] = 0.0

        # Reset episode lengths for the reset environments
        self._episode_length[envs_idx] = 0
