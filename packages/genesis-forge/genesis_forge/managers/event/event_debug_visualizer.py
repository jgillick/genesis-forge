from __future__ import annotations

import math
from typing import Any

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import debug_envs_idx

from .config import DEFAULT_VISUALIZER_CONFIG, EventDebugVisualizerConfig
from .config_item import EventConfigItem


class EventDebugVisualizer:
    """
    Decides when, and in which environments, each event is drawn, and removes what was drawn
    before. The drawing itself is done by the event function's ``debug_draw`` method.

    Args:
        env: The environment instance.
        items: The events to visualize.
        cfg: The visualizer config.
    """

    def __init__(
        self,
        env: GenesisEnv,
        items: dict[str, EventConfigItem],
        cfg: EventDebugVisualizerConfig,
    ):
        self._env = env
        self._cfg: dict[str, Any] = {**DEFAULT_VISUALIZER_CONFIG, **cfg}
        self._items = [item for item in items.values() if item.has_debug_visualization]
        self._nodes = []
        self._built = False

    def step(self):
        """Update which environments each event is visible in, and draw if it's time to."""
        if not self._built:
            self._build()

        for i, item in enumerate(self._items):
            active = item.active[self._envs_idx]
            started = active & ~self._active[i]
            self._remaining[i] = torch.where(
                started, self._display_steps, (self._remaining[i] - 1).clamp(min=0)
            )
            self._active[i] = active

        if self._env.step_count % self._steps_per_render == 0:
            self._render()

    def reset(self, envs_idx: torch.Tensor):
        """Remove what's drawn in these environments."""
        if not self._built:
            return
        done = torch.isin(self._envs_idx, envs_idx)
        if not done.any():
            return
        self._active[:, done] = False
        self._remaining[:, done] = 0
        self._render()

    def clear(self):
        """Remove everything that has been drawn."""
        for node in self._nodes:
            self._env.scene.clear_debug_object(node)
        self._nodes = []

    """
    Implementation
    """

    def _build(self):
        """Initialize on the first step, so that the visualizer can be turned on at any time."""
        env = self._env
        envs_idx = debug_envs_idx(env.scene, env.num_envs, self._cfg["envs_idx"])
        self._envs_idx = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.long)
        self._steps_per_render = math.ceil(1.0 / self._cfg["fps"] / env.dt)
        self._display_steps = max(1, round(self._cfg["min_display_s"] / env.dt))

        # Shape is (num events, num visualized envs)
        self._active = torch.zeros(
            (len(self._items), len(self._envs_idx)), device=gs.device, dtype=torch.bool
        )
        self._remaining = torch.zeros_like(self._active, dtype=torch.int32)
        self._built = True

    def _render(self):
        self.clear()
        visible = self._active | (self._remaining > 0)
        for item, mask in zip(self._items, visible, strict=True):
            envs_idx = self._envs_idx[mask]
            if envs_idx.numel() > 0:
                nodes = item.debug_draw(envs_idx)
                self._nodes.extend(node for node in nodes if node is not None)
