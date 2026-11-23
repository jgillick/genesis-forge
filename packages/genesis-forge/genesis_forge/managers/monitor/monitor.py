from __future__ import annotations

from collections.abc import Iterator

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager

from .monitor_config import (
    _PERCENTILE,
    MonitorConfig,
    MonitorConfigItem,
    MonitorReduction,
)


class Monitor(BaseManager):
    """
    Calculates values about your robot on every step, and logs them.

    Each item's function returns either:
        - a value per environment, shape (num_envs,)
        - a labeled row per environment, shape (num_envs, N)
        - or a single number, like a curriculum level.
    Values are logged combined across environments, as set by `reduce`.

    Rewards, terminations, and observations can read the latest value by name, like
    `monitor["height"]`. Values update every step and again after a reset, so an item
    can run twice in a step: calculate it from the current state, not a running total.

    Args:
        env: The environment to monitor.
        cfg: A dictionary of monitor items.
        logging_enabled: Whether to add the values to the training logs.
        logging_tag: The section name the values are logged under.

    Example::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.body_monitor = Monitor(
                    self,
                    logging_tag="Body",
                    cfg={
                        "height": {
                            "fn": lambda env: self.robot.get_pos()[:, 2],
                        },
                        "velocity": {
                            "fn": lambda env: self.robot.get_vel(),
                            "labels": ["x", "y", "z"],
                            "reduce": ["mean", "max"],
                        },
                    },
                )

                RewardManager(
                    self,
                    cfg={
                        "height": {
                            "weight": -10.0,
                            "fn": lambda env: torch.square(self.body_monitor["height"] - 0.3),
                        },
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        cfg: dict[str, MonitorConfig] | None = None,
        logging_enabled: bool = True,
        logging_tag: str = "Monitor",
    ):
        super().__init__(env, type="monitor")
        self.logging_enabled = logging_enabled
        self.logging_tag = logging_tag

        self.cfg: dict[str, MonitorConfigItem] = {}
        for name, item in (cfg or {}).items():
            self.cfg[name] = MonitorConfigItem(item, env)
        self._values: dict[str, torch.Tensor] = {}

    """
    Values
    """

    def labels(self, name: str) -> list[str] | None:
        """
        The column labels of a monitor value, or None if it has no columns.
        """
        return self.cfg[name].labels

    def get(self, name: str, label: str | None = None) -> torch.Tensor:
        """
        The latest value of a monitor item, or one of its labeled columns.

        Args:
            name: The monitor item name.
            label: The column to return, for a value shaped (num_envs, N).

        Returns:
            The value, shape (num_envs,), (num_envs, N), or () for a single number.
            Shape (num_envs,) when a label is given.
        """
        value = self[name]
        if label is None:
            return value
        labels = self.labels(name) or []
        if label not in labels:
            raise KeyError(f"Monitor item '{name}' has no column '{label}': {labels}")
        return value[:, labels.index(label)]

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Build the monitor item functions, then check that no earlier monitor logs
        an item under the same name.
        """
        for cfg in self.cfg.values():
            cfg.build()
        self._check_log_names()

    def step(self):
        """
        Calculate each monitor value for this step, and log them.
        """
        if not self.enabled:
            return
        self.update()
        if self.logging_enabled:
            for name, cfg in self.cfg.items():
                if cfg.log:
                    self._log(name, cfg, self._values[name])

    def update(self):
        """
        Calculate each monitor value, in config order, without logging them.
        """
        for name, cfg in self.cfg.items():
            value = cfg.execute()
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=gs.device)
            self._check_shape(name, cfg, value)
            self._values[name] = value

    def reset(self, envs_idx: torch.Tensor | None = None):
        """
        Reset the monitor item functions for the given environments, then
        recalculate the values, so observations see the new episode.
        """
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx
        for cfg in self.cfg.values():
            cfg.reset(envs_idx)
        if self.enabled:
            self.update()

    """
    Dict-like operations
    """

    def __getitem__(self, name: str) -> torch.Tensor:
        """The latest value of a monitor item."""
        if name not in self.cfg:
            raise KeyError(
                f"Monitor has no item '{name}'. Items: {list(self.cfg.keys())}"
            )
        if name not in self._values:
            raise RuntimeError(
                f"Monitor item '{name}' has not been calculated yet. A monitor item "
                "can only read monitors created before its own, and items listed before it."
            )
        return self._values[name]

    def __contains__(self, name: str) -> bool:
        """Check if a monitor item exists."""
        return name in self.cfg

    def __iter__(self) -> Iterator[str]:
        """Iterate over the monitor item names."""
        return iter(self.cfg)

    def __len__(self) -> int:
        """Get the number of monitor items."""
        return len(self.cfg)

    """
    Internal methods
    """

    def _check_log_names(self):
        """
        Raise if an earlier monitor logs an item with the same tag and name, since one
        would overwrite the other in the training logs.
        """
        if not self.logging_enabled:
            return
        managers = getattr(self.env, "managers", None) or {}
        for other in managers.get("monitor", []):
            if other is self:
                return
            if not other.logging_enabled or other.logging_tag != self.logging_tag:
                continue
            shared = [
                name
                for name, cfg in self.cfg.items()
                if cfg.log and name in other.cfg and other.cfg[name].log
            ]
            if shared:
                raise ValueError(
                    f"Two monitors log {shared} under '{self.logging_tag}'. "
                    "Give one of them a different logging_tag."
                )

    def _check_shape(self, name: str, cfg: MonitorConfigItem, value: torch.Tensor):
        """
        Raise if a value is not a single number, (num_envs,), or (num_envs, N) with N labels.
        """
        num_envs = self.env.num_envs
        if value.ndim == 0:
            if cfg.labels is not None:
                raise ValueError(
                    f"Monitor item '{name}' returned a single number, so it can't "
                    "have labels. Remove 'labels' from its config."
                )
            return
        if value.ndim == 1 and value.shape[0] == num_envs:
            if cfg.labels is not None:
                raise ValueError(
                    f"Monitor item '{name}' returned one value per environment, so it "
                    "can't have labels. Remove 'labels' from its config."
                )
            return
        if value.ndim == 2 and value.shape[0] == num_envs:
            if cfg.labels is None:
                raise ValueError(
                    f"Monitor item '{name}' returned shape {tuple(value.shape)}, so it "
                    f"needs a label for each of its {value.shape[1]} columns. "
                    "Add 'labels' to its config."
                )
            if len(cfg.labels) != value.shape[1]:
                raise ValueError(
                    f"Monitor item '{name}' returned {value.shape[1]} columns, "
                    f"but has {len(cfg.labels)} labels: {cfg.labels}"
                )
            return
        raise ValueError(
            f"Monitor item '{name}' returned shape {tuple(value.shape)}. "
            f"Expected a single number, ({num_envs},), or ({num_envs}, N)."
        )

    def _log(self, name: str, cfg: MonitorConfigItem, value: torch.Tensor):
        """
        Log a value, combined across environments, once for each reduction.
        The mean is logged under the item's name, and the others add a suffix.
        A single number is logged as-is.
        """
        logging_dict = self.env.extras[self.env.extras_logging_key]
        if value.ndim == 0:
            logging_dict[f"{self.logging_tag} / {name}"] = value
            return
        for reduction in cfg.reduce:
            reduced = self._reduce(value.float(), reduction)
            suffix = "" if reduction == "mean" else f"_{reduction}"
            if cfg.labels is None:
                logging_dict[f"{self.logging_tag} / {name}{suffix}"] = reduced
            else:
                for col, label in enumerate(cfg.labels):
                    key = f"{self.logging_tag} / {name}_{label}{suffix}"
                    logging_dict[key] = reduced[col]

    @staticmethod
    def _reduce(value: torch.Tensor, reduction: MonitorReduction) -> torch.Tensor:
        """
        Combine a value across environments (dim 0).
        """
        if reduction == "mean":
            return value.mean(dim=0)
        if reduction == "min":
            return value.min(dim=0).values
        if reduction == "max":
            return value.max(dim=0).values
        if reduction == "std":
            return value.std(dim=0, correction=0)
        if reduction == "rms":
            return torch.sqrt(value.square().mean(dim=0))
        percentile = _PERCENTILE.match(reduction)
        assert percentile is not None, f"Unknown reduction '{reduction}'"
        return torch.quantile(value, float(percentile.group(1)) / 100.0, dim=0)
