from __future__ import annotations

import math
from typing import Any, cast

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.config import (
    ConfigItem,
    ConfigItemDict,
    DurationEventMdpFn,
    EventMdpFn,
    MdpFn,
)

from .config import EventConfig, TimeRange


class EventConfigItem(ConfigItem):
    """
    A config item for an event, which also holds the event's state for each environment.
    """

    def __init__(self, cfg: EventConfig, env: GenesisEnv):
        super().__init__(cast(ConfigItemDict, cfg), env)

        self._duration_fn = (
            self._fn if isinstance(self._fn, DurationEventMdpFn) else None
        )
        if isinstance(self._fn, MdpFn):
            self._validate_fn()
        self._interval_range = self._parse_time_range(
            "interval_range_s", cfg.get("interval_range_s")
        )
        self.duration_range_s = cfg.get("duration_range_s")

        # Buffers
        self._active = torch.zeros(env.num_envs, device=gs.device, dtype=torch.bool)
        self._pending = torch.zeros_like(self._active)
        self._wait_steps = torch.zeros(
            env.num_envs, device=gs.device, dtype=torch.int32
        )
        self._remaining_steps = torch.zeros_like(self._wait_steps)
        self._sample_interval(env.all_envs_idx)

    """
    Properties
    """

    @property
    def interval_range_s(self) -> TimeRange | None:
        """
        The (min, max) time to wait before the event fires, or None if it is only fired manually.
        A changed interval is used the next time each environment starts waiting.
        """
        return self._interval_range

    @interval_range_s.setter
    def interval_range_s(self, interval_range_s: TimeRange | None):
        was_manual = self._interval_range is None
        self._interval_range = self._parse_time_range(
            "interval_range_s", interval_range_s
        )
        if was_manual:
            self._sample_interval(self._env.all_envs_idx)

    @property
    def duration_range_s(self) -> TimeRange | None:
        """
        The (min, max) time the event lasts, or None if it happens all at once.
        A changed duration is used the next time the event fires.
        """
        return self._duration_range

    @duration_range_s.setter
    def duration_range_s(self, duration_range_s: TimeRange | None):
        if self._duration_fn is not None and duration_range_s is None:
            raise ValueError(
                f"duration_range_s is required for a DurationEventMdpFn ({type(self._fn).__name__})"
            )
        if self._duration_fn is None and duration_range_s is not None:
            raise ValueError(
                "duration_range_s can only be set for a DurationEventMdpFn function"
            )
        self._duration_range = self._parse_time_range(
            "duration_range_s", duration_range_s
        )

    @property
    def has_debug_visualization(self) -> bool:
        """Whether the event supports the debug visualizer."""
        return (
            isinstance(self._fn, EventMdpFn)
            and type(self._fn).debug_draw is not EventMdpFn.debug_draw
        )

    @property
    def active(self) -> torch.Tensor:
        """Which environments the event is active in. Shape is (num_envs,)."""
        return self._active

    """
    Operations
    """

    def trigger(self, envs_idx: torch.Tensor):
        """Fire the event for these environments on the next step."""
        self._pending[envs_idx] = True

    def step(self):
        """Advance the timers, and run the event's function where it's due."""
        self._remaining_steps -= self._active.int()
        ended = self._active & (self._remaining_steps <= 0)

        # Count down the environments that are waiting, not including the ones that just ended
        due = self._pending.clone()
        if self._interval_range is not None:
            idle = ~self._active
            self._wait_steps -= idle.int()
            due |= idle & (self._wait_steps <= 0)
        self._pending[:] = False

        # A manual trigger restarts an event that is still active
        ended |= due & self._active
        active = (self._active & ~ended) | due

        ended_idx, due_idx, active_idx = self._masks_to_idx(ended, due, active)

        self._end(ended_idx)

        if due_idx.numel() > 0:
            self._active[due_idx] = True
            self._remaining_steps[due_idx] = (
                1
                if self._duration_range is None
                else self._sample_steps(self._duration_range, due_idx)
            )
            if self._duration_fn is not None:
                self._duration_fn.start(due_idx)
            else:
                self.execute(envs_idx=due_idx)

        if self._duration_fn is not None and active_idx.numel() > 0:
            self._duration_fn.step(active_idx)

    def stop(self):
        """End the event everywhere it's active, and discard any pending triggers."""
        self._pending[:] = False
        self._end(self._active.nonzero().flatten())

    def reset(self, envs_idx: torch.Tensor):
        """End the event for these environments, and restart their wait."""
        self._end(envs_idx[self._active[envs_idx]])
        self._pending[envs_idx] = False
        self._sample_interval(envs_idx)
        super().reset(envs_idx)

    def debug_draw(self, envs_idx: torch.Tensor) -> list[Any]:
        """Draw the event, and return the debug objects that were drawn."""
        if isinstance(self._fn, EventMdpFn):
            return self._fn.debug_draw(envs_idx)
        return []

    """
    Implementation
    """

    def _validate_fn(self):
        """Check that the function class implements the methods that will be called."""
        cls = type(self._fn)
        if not isinstance(self._fn, EventMdpFn):
            raise TypeError(
                f"{cls.__name__} must be a subclass of EventMdpFn to be used as an event"
            )
        if self._duration_fn is None:
            if cls.__call__ is EventMdpFn.__call__:
                raise TypeError(
                    f"{cls.__name__} must implement __call__(), or be a subclass of "
                    "DurationEventMdpFn if it lasts for a length of time"
                )
            return

        if cls.__call__ is not DurationEventMdpFn.__call__:
            raise TypeError(
                f"{cls.__name__} is a DurationEventMdpFn, so __call__() will not be "
                "called. Implement start() instead."
            )
        if all(
            getattr(cls, name) is getattr(DurationEventMdpFn, name)
            for name in ("start", "step", "end")
        ):
            raise TypeError(
                f"{cls.__name__} must implement at least one of start(), step(), or end()"
            )

    @staticmethod
    def _masks_to_idx(*masks: torch.Tensor) -> list[torch.Tensor]:
        """Get the environment ids where each mask is True."""
        # On a GPU, each nonzero() call has to wait for the GPU to finish, since it can't
        # know the size of its result. Apple Silicon (MPS) is slow to wait, so copy all the
        # masks to the CPU at once and only wait once. With 4,096 environments, this takes
        # 1.0-1.5 ms per step on MPS, compared with 2.6-3.0 ms for a nonzero() call per mask.
        # On CUDA, waiting is cheap, so the copies back to the GPU would cost more than they save.
        if masks[0].device.type == "mps":
            device = masks[0].device
            cpu_masks = torch.stack(masks).cpu()
            return [mask.nonzero().flatten().to(device) for mask in cpu_masks]
        return [mask.nonzero().flatten() for mask in masks]

    def _end(self, envs_idx: torch.Tensor):
        """End the event for these environments, and start their next wait."""
        if envs_idx.numel() == 0:
            return
        self._active[envs_idx] = False
        self._sample_interval(envs_idx)
        if self._duration_fn is not None:
            self._duration_fn.end(envs_idx)

    def _sample_interval(self, envs_idx: torch.Tensor):
        if self._interval_range is not None:
            self._wait_steps[envs_idx] = self._sample_steps(
                self._interval_range, envs_idx
            )

    def _sample_steps(
        self, time_range: tuple[float, float], envs_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sample a number of whole steps (at least one) for each environment."""
        seconds = torch.empty(len(envs_idx), device=gs.device).uniform_(*time_range)
        return torch.round(seconds / self._env.dt).clamp(min=1).to(torch.int32)

    @staticmethod
    def _parse_time_range(name: str, value: Any) -> tuple[float, float] | None:
        """Validate a time value, and convert it to a (min, max) tuple."""
        if value is None:
            return None
        try:
            low, high = (value, value) if isinstance(value, (int, float)) else value
            low, high = float(low), float(high)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must be a number or a (min, max) pair of numbers, got {value!r}"
            ) from error
        if not (math.isfinite(high) and 0 <= low <= high and high > 0):
            raise ValueError(
                f"{name} must be finite and satisfy 0 <= min <= max, with max > 0, got {value!r}"
            )
        return (low, high)
