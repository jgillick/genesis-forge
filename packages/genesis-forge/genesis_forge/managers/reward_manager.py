from __future__ import annotations

from collections.abc import Iterator

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.config import ConfigItemDict, RewardConfigItem


class RewardConfig(ConfigItemDict):
    """Defines a reward item."""

    weight: float
    """The weight of the reward item."""


class RewardManager(BaseManager):
    """
    Handles calculating and logging the rewards for the environment.

    This works with a dictionary configuration of reward handlers. For each dictionary item,
    a function will be called to calculate a reward value for the environment.

    Args:
        env: The environment to manage the rewards for.
        cfg: A dictionary of reward conditions.
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

            def reset(self, envs_idx: torch.Tensor | Sequence[int] | None = None):
                super().reset(envs_idx)
                self.reward_manager.reset(envs_idx)
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

        self.logging_enabled = logging_enabled
        self.logging_tag = logging_tag

        # Wrap config items
        self.cfg: dict[str, RewardConfigItem] = {}
        for name, item in cfg.items():
            self.cfg[name] = RewardConfigItem(item, env)

        # Initialize buffers
        self._reward_buf = torch.zeros(
            (env.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        # Nearly zero, like after a reset, so a reset before the first step doesn't divide by zero
        self._episode_seconds = torch.full(
            (self.env.num_envs,), 1e-10, device=gs.device, dtype=gs.tc_float
        )
        # Kept on the device, so a reset doesn't wait on the GPU; read as floats when asked for
        self._episode_mean: dict[str, torch.Tensor] = {}
        self._episode_sums: dict[str, torch.Tensor] = {}
        for name in self.cfg:
            self._episode_sums[name] = torch.zeros(
                (env.num_envs,), device=gs.device, dtype=gs.tc_float
            )

    @property
    def rewards(self) -> torch.Tensor:
        """
        The rewards calculated for the most recent step. Shape is (num_envs,).
        """
        return self._reward_buf

    @property
    def episode_data(self) -> dict[str, torch.Tensor]:
        """
        Get the accumulated reward data for the current episode of all environments,
        at each reward's current weight.
        """
        return {
            name: sums * self.cfg[name].weight
            for name, sums in self._episode_sums.items()
        }

    """
    Helpers
    """

    def last_episode_mean_reward(self, name: str, before_weight: bool = True) -> float:
        """
        Get the last mean reward for an episode for a given reward name.
        The mean reward is only calculated when episodes end/reset, and is the mean across
        the environments that reset together.

        Args:
            name: The name of the reward to get the mean for.
            before_weight: If True, this will be the base reward value before the weight was applied.

        Returns:
            The last mean reward for an episode for a given reward name.
        """
        rew = float(self._episode_mean.get(name, 0.0))
        if not before_weight:
            rew *= self.cfg[name].weight
        return rew

    def episode_mean_reward_for_envs(
        self,
        name: str,
        envs_idx: torch.Tensor | None = None,
        before_weight: bool = True,
    ) -> torch.Tensor:
        """
        Get each environment's mean reward, per second, for its current episode so far, for a
        given reward name. Call it before the environments reset, such as from a curriculum,
        to grade the episodes that just ended.

        Args:
            name: The name of the reward to get the mean for.
            envs_idx: The environments to get the mean for. If None, all environments.
            before_weight: If True, this will be the base reward value before the weight was applied.

        Returns:
            The mean reward for each environment. Shape is (len(envs_idx),).
        """
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx

        rew = self._episode_sums[name][envs_idx] / self._episode_seconds[envs_idx]
        if not before_weight:
            rew = rew * self.cfg[name].weight
        return rew

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Build any config item function classes.
        """
        for cfg in self.cfg.values():
            cfg.build()

    def step(self) -> torch.Tensor:
        """
        Calculate the rewards for this step

        Returns:
            The rewards for the environments. Shape is (num_envs,).
        """
        if not self.enabled:
            return self._reward_buf

        dt = self.env.dt
        self._reward_buf[:] = 0.0
        self._episode_seconds += dt
        for name, cfg in self.cfg.items():
            value = cfg.execute() * dt

            # Add to the episode sums, even at zero weight, so the mean is always known
            self._episode_sums[name] += value

            # Add the weighted value to the reward buffer
            if cfg.weight != 0:
                self._reward_buf += value * cfg.weight

        return self._reward_buf

    def reset(self, envs_idx: torch.Tensor | None = None):
        """Record, and log, the reward mean values at the end of the episode"""
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx

        # Reset function classes
        for cfg in self.cfg.values():
            cfg.reset(envs_idx)

        for name in self.cfg:
            if self.enabled and len(envs_idx) > 0:
                # The mean across the episodes that just ended, per second
                episode_mean = torch.mean(
                    self.episode_mean_reward_for_envs(name, envs_idx)
                )
                self._episode_mean[name] = episode_mean

                # Log items with a weight, at their weight
                weight = self.cfg[name].weight
                if self.logging_enabled and weight != 0:
                    logging_dict = self.env.extras[self.env.extras_logging_key]
                    logging_dict[f"{self.logging_tag} / {name}"] = episode_mean * weight

            # Reset episodic data
            self._episode_sums[name][envs_idx] = 0.0

        # Reset episode seconds to nearly zero, to prevent divide by zero errors
        self._episode_seconds[envs_idx] = 1e-10

    """
    Dict-like operations
    """

    def __getitem__(self, name: str) -> RewardConfigItem:
        """Get a reward config item by name."""
        return self.cfg[name]

    def __setitem__(self, name: str, value: RewardConfigItem):
        """Set a reward config item by name."""
        self.cfg[name] = value

    def __delitem__(self, name: str):
        """Delete a reward config item by name."""
        del self.cfg[name]

    def __contains__(self, name: str) -> bool:
        """Check if a reward config item exists by name."""
        return name in self.cfg

    def __iter__(self) -> Iterator[str]:
        """Iterate over the reward config item names."""
        return iter(self.cfg.keys())

    def __len__(self) -> int:
        """Get the number of reward config items."""
        return len(self.cfg)
