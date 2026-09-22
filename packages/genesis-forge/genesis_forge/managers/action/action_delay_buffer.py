from __future__ import annotations

import genesis as gs
import numpy as np
import torch


class ActionDelayBuffer:
    """
    Delays actions by a whole number of steps, to emulate the latency between the policy
    emitting an action and the actuator acting on it.

    Each step, :meth:`push` records the incoming actions and returns the ones each
    environment should act on now: the actions it received ``delay`` steps ago, or
    zeros while its first real actions are still queued. The delay can be one fixed
    value for every environment, or a ``(min, max)`` range that each environment draws
    from on build and again whenever it resets, so a policy trains against a band of
    latencies rather than one exact timing.

    The buffer is simulation-only. It never enters the deployment bundle, since the
    real robot supplies its own latency.

    Args:
        delay_step: Steps to delay actions by. An int delays every environment by the
            same amount; a ``(min, max)`` tuple draws each environment's delay from
            that inclusive range.

    Example::

        delay = ActionDelayBuffer((0, 2))
        delay.build(num_envs=1024, num_actions=12)

        # Each step
        actions = delay.push(actions)

        # On reset
        delay.reset(envs_idx)
    """

    def __init__(self, delay_step: int | tuple[int, int] = 0):
        self._range = self._normalize_range(delay_step)
        self._delay_steps: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        # Slot in `_history` holding the most recent step's actions
        self._current_slot = 0
        self._env_idx: torch.Tensor | None = None

    @staticmethod
    def _normalize_range(delay_step: int | tuple[int, int]) -> tuple[int, int]:
        """Turn a `delay_step` argument into an inclusive ``(min, max)`` range of whole steps.

        A bare int is a fixed delay, so it becomes ``(n, n)``.

        Raises:
            ValueError: The delay is negative, not a whole number of steps, or the
                range's minimum exceeds its maximum.
        """
        if isinstance(delay_step, (tuple, list)):
            if len(delay_step) != 2:
                raise ValueError(
                    f"delay_step must be an int or a (min, max) tuple, got {delay_step!r}."
                )
            low, high = delay_step
        else:
            low = high = delay_step

        for value in (low, high):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"delay_step must be a whole number of steps, got {value!r}."
                )
            if value < 0:
                raise ValueError(f"delay_step cannot be negative, got {value!r}.")
        if low > high:
            raise ValueError(
                f"delay_step range minimum {low} is greater than its maximum {high}."
            )
        return int(low), int(high)

    """
    Properties
    """

    @property
    def range(self) -> tuple[int, int]:
        """
        The inclusive ``(min, max)`` range of steps an environment's actions can be
        delayed by. Both ends are equal when `delay_step` was given as a single int.
        """
        return self._range

    @property
    def enabled(self) -> bool:
        """
        Whether any delay is applied at all. False when the maximum delay is zero, in
        which case :meth:`push` returns its input untouched.
        """
        return self._range[1] > 0

    @property
    def delay_steps(self) -> torch.Tensor:
        """
        How many steps each environment's actions are currently delayed by, shape
        ``(num_envs,)``. Redrawn from `range` whenever an environment resets. Useful
        as a privileged observation for an asymmetric critic.
        """
        return self._delay_steps

    """
    Lifecycle Operations
    """

    def build(self, num_envs: int, num_actions: int):
        """
        Allocate the action history and draw every environment's delay.

        The history holds the current step's actions plus the ``max`` before it, all
        zero to begin with, so an environment's first ``delay`` steps send no-op
        actions while its real ones are still queued.
        """
        self._delay_steps = torch.zeros(num_envs, dtype=torch.long, device=gs.device)
        self._env_idx = torch.arange(num_envs, dtype=torch.long, device=gs.device)
        self._current_slot = 0
        if not self.enabled:
            self._history = None
            return
        self._history = torch.zeros(
            (self._range[1] + 1, num_envs, num_actions), device=gs.device
        )
        self._sample(self._env_idx)

    def push(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Record this step's actions and return the ones to act on now.

        Args:
            actions: This step's actions, shape ``(num_envs, num_actions)``.

        Returns:
            For each environment, the actions received `delay_steps` steps ago, shape
            ``(num_envs, num_actions)``. With no delay, `actions` itself.
        """
        if self._history is None:
            return actions
        self._current_slot = (self._current_slot + 1) % self._history.shape[0]
        self._history[self._current_slot] = actions
        delayed_slot = (self._current_slot - self._delay_steps) % self._history.shape[0]
        return self._history[delayed_slot, self._env_idx]

    def reset(self, envs_idx: torch.Tensor):
        """
        Clear the queued actions of the reset environments, so the previous episode's
        actions are not delivered, and draw them a new delay from `range`.
        """
        if self._history is None:
            return
        self._history[:, envs_idx] = 0.0
        self._sample(envs_idx)

    def _sample(self, envs_idx: torch.Tensor):
        """Draw a fresh delay for each of `envs_idx` from `range`."""
        low, high = self._range
        if low == high:
            self._delay_steps[envs_idx] = high
        else:
            self._delay_steps[envs_idx] = torch.randint(
                low, high + 1, (len(envs_idx),), device=gs.device
            )
