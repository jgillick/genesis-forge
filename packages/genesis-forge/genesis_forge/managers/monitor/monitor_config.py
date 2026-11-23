from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.config import ConfigItem, ConfigItemDict, MdpFn

MonitorReduction = Literal[
    "mean", "min", "max", "std", "rms", "p50", "p90", "p95", "p99"
]
"""
A way to combine the environments' values for the log. Other percentiles, like "p97",
also work, but aren't part of the type.
"""

_REDUCTIONS = ("mean", "min", "max", "std", "rms")
_PERCENTILE = re.compile(r"^p(\d+(?:\.\d+)?)$")


class MonitorFn(Protocol):
    """
    A function that calculates a monitor value. Unlike rewards and observations, it can
    return a plain number as well as a tensor.
    """

    def __call__(
        self, env: GenesisEnv, *params: Any, **kwargs: Any
    ) -> torch.Tensor | float: ...


class MonitorConfig(TypedDict):
    """Defines a monitor item."""

    fn: MonitorFn | MdpFn
    """The function that calculates the value, from the environment."""

    params: NotRequired[dict[str, Any]]
    """Additional parameters to pass to the function."""

    reduce: NotRequired[MonitorReduction | list[MonitorReduction]]
    """
    How to combine the environments' values for the log: any of "mean", "min", "max",
    "std", "rms", or a percentile like "p99", or a list of them. The mean is logged
    under the item's name, and the others add a suffix, like `<name>_p99`. A single
    number is logged as-is. Defaults to "mean".
    """

    log: NotRequired[bool]
    """
    Whether to add this item to the training logs. Set it to False for a value only
    your rewards, terminations, or observations read. Defaults to True.
    """

    labels: NotRequired[list[str] | Callable[[], list[str] | None]]
    """
    A name for each column, when the function returns shape (num_envs, N). Each column
    is logged as `<name>_<label>`. A callable is called at build, for names that are
    only known once the scene is built.
    """


class MonitorConfigItem(ConfigItem):
    """
    A config item for a monitor value.
    """

    def __init__(self, cfg: MonitorConfig, env: GenesisEnv):
        super().__init__(cast(ConfigItemDict, cfg), env)
        self.log: bool = cfg.get("log", True)

        self._labels_cfg = cfg.get("labels", None)
        self._labels: list[str] | None = None

        reduce = cfg.get("reduce", "mean")
        self.reduce: list[MonitorReduction] = (
            [reduce] if isinstance(reduce, str) else reduce
        )
        if not self.reduce:
            raise ValueError(
                "Monitor 'reduce' can't be empty. To skip logging an item, set 'log': False."
            )
        for reduction in self.reduce:
            percentile = _PERCENTILE.match(reduction)
            if percentile and float(percentile.group(1)) <= 100:
                continue
            if reduction not in _REDUCTIONS:
                raise ValueError(
                    f"Unknown monitor reduction '{reduction}'. Use one of "
                    f"{list(_REDUCTIONS)}, or a percentile like 'p99'."
                )

    def build(self, **kwargs):
        """Build the function, then resolve the column labels."""
        super().build(**kwargs)
        labels = self._labels_cfg
        if callable(labels):
            labels = labels()
        self._labels = list(labels) if labels is not None else None

    @property
    def labels(self) -> list[str] | None:
        """
        The column labels from the config, or else the function's own `labels`, read
        each time so they stay current when a curriculum rebuilds the function.
        """
        if self._labels is not None:
            return self._labels
        labels = getattr(self.fn, "labels", None)
        return list(labels) if labels is not None else None
