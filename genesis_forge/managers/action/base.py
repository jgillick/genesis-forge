import torch
import numpy as np
from gymnasium import spaces
import genesis as gs
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager


class BaseActionManager(BaseManager):
    """
    Base for managers that handle actions.

    Args:
        env: The environment to manage the DOF actuators for.
        delay_step: The number of steps to delay actions for, to emulate the latency in the system.
                    This can be an integer for a fixed delay, or a tuple (min, max) for a per-environment random delay range.
    """

    def __init__(self, env: GenesisEnv, delay_step: int | tuple[int, int] = 0):
        super().__init__(env, type="action")
        self._raw_actions = None
        self._actions = None
        self._envs_idx: torch.Tensor | None = None

        self._delay_step = delay_step
        self._delay_ring_buffer_head = 0
        self._delay_ring_buffer: torch.Tensor | None = None
        self._delay_step_idx: torch.Tensor | None = None

        # Validate the delay_step tuple
        if isinstance(delay_step, tuple):
            min_delay, max_delay = delay_step
            if min_delay < 0 or max_delay < min_delay:
                raise ValueError(
                    f"Invalid delay_step range: {self._delay_step}. Must be (min, max) where min >= 0 and max >= min"
                )
        elif isinstance(delay_step, int):
            if delay_step < 0:
                raise ValueError(
                    f"Invalid delay_step: {self._delay_step}. Must be >= 0"
                )
            if delay_step == 0:
                self._delay_step = None

    """
    Properties
    """

    @property
    def num_actions(self) -> int:
        """
        The total number of actions.
        """
        return 0

    @property
    def action_space(self) -> tuple[float, float]:
        """
        If using the default action handler, the action space is [-1, 1].
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    @property
    def actions(self) -> torch.Tensor:
        """
        The actions for for the current step.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions

    @property
    def raw_actions(self) -> torch.Tensor:
        """
        The actions received from the policy, before being converted.
        """
        if self._raw_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._raw_actions

    """
    Lifecycle Operations
    """

    def build(self):
        """Initialize the action delay buffers."""
        self._envs_idx = torch.arange(self.env.num_envs, device=gs.device)
        if self._delay_step is not None:
            if isinstance(self._delay_step, tuple):
                max_delay = self._delay_step[1]
            else:
                max_delay = self._delay_step

            self._delay_ring_buffer = torch.zeros(
                (self.env.num_envs, max_delay + 1, self.num_actions),
                dtype=torch.float32,
                device=gs.device,
            )
            self._delay_step_idx = torch.zeros(
                self.env.num_envs, dtype=torch.int32, device=gs.device
            )

    def step(self, actions: torch.Tensor) -> None:
        """
        Handle the received actions.
        """
        self._raw_actions = self._apply_action_delay(actions)

        # Copy the actions into the manager buffer
        if self._actions is None:
            self._actions = self._raw_actions.clone()
        else:
            self._actions[:] = self._raw_actions[:]
        return self._actions

    def reset(self, envs_idx: list[int] | None):
        """Reset environments."""

        # Per-environment random action delay
        if isinstance(self._delay_step, tuple) and self._delay_step_idx is not None:
            min_delay, max_delay = self._delay_step
            if envs_idx is None:
                envs_idx = self._envs_idx

            self._delay_step_idx[envs_idx] = torch.randint(
                min_delay,
                max_delay + 1,
                (len(envs_idx),),
                device=gs.device,
                dtype=torch.int32,
            )

    def get_actions(self) -> torch.Tensor:
        """
        Get the current actions for the environments.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions

    """
    Internal Operations
    """

    def _apply_action_delay(self, actions: torch.Tensor) -> torch.Tensor:
        """
        When action delay is enabled (via `delay_step`), the actions will be pushed onto a ring buffer,
        and then an older action-set will be returned. If the delay step is a tuple range, each environment
        will randomly be assigned a delay step within that range.
        """
        if self._delay_step is None or self._delay_ring_buffer is None:
            return actions

        # Ring buffer head index
        # This is the index of the ring buffer that we should add the new action set to
        delay_buffer_len = self._delay_ring_buffer.shape[1]
        head = (self._delay_ring_buffer_head + 1) % delay_buffer_len

        # Add to delay ring buffer
        self._delay_ring_buffer[:, head, :] = actions
        self._delay_ring_buffer_head = head

        # Get the buffer index for the delayed actions
        if isinstance(self._delay_step, tuple):
            idx = (head - self._delay_step_idx) % delay_buffer_len
            actions = self._delay_ring_buffer[self._envs_idx, idx, :]
        # Fixed delay for all environments
        else:
            idx = (head - self._delay_step) % delay_buffer_len
            actions = self._delay_ring_buffer[:, idx, :]

        return actions
