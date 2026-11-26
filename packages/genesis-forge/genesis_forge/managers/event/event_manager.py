from __future__ import annotations

from collections.abc import Sequence

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager

from .config import EventConfig, EventDebugVisualizerConfig
from .config_item import EventConfigItem
from .event_debug_visualizer import EventDebugVisualizer


class EventManager(BaseManager):
    """
    Runs events in the middle of an episode, like random pushes.

    Each environment has its own random timer for each event. Events run right before the
    simulation step.

    To set the state of an entity at the start of an episode, use the :class:`EntityManager`
    ``on_reset`` config instead.

    Setting ``enabled`` to False ends the active events on the next step, pauses the timers, and
    ignores triggers.

    Args:
        env: The environment instance.
        cfg: A dictionary of events.
        debug_visualizer: Draw the events that support it (see :meth:`EventMdpFn.debug_draw`).
                          To see this in camera renders or recorded videos, the camera needs
                          to be created with ``debug=True``.
        debug_visualizer_cfg: The configuration for the debug visualizer.

    Example::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.robot_manager = EntityManager(self, entity=self.robot)
                self.event_manager = EventManager(
                    self,
                    cfg={
                        # Instantly change the robot's velocity every 5-10 seconds
                        "kick": {
                            "fn": events.push_velocity(
                                entity_manager=self.robot_manager,
                                linear={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
                            ),
                            "interval_range_s": (5.0, 10.0),
                        },
                        # Push the robot with 10-40 newtons, for 0.1-0.3 seconds
                        "shove": {
                            "fn": events.push_force(
                                entity_manager=self.robot_manager,
                                force=(10.0, 40.0),
                            ),
                            "interval_range_s": (4.0, 8.0),
                            "duration_range_s": (0.1, 0.3),
                        },
                    },
                )

    Example - Change an event during training::

        self.event_manager["shove"].interval_range_s = (2.0, 5.0)
        self.event_manager["shove"].fn.force = (20.0, 60.0)
    """

    def __init__(
        self,
        env: GenesisEnv,
        cfg: dict[str, EventConfig],
        debug_visualizer: bool = False,
        debug_visualizer_cfg: EventDebugVisualizerConfig | None = None,
    ):
        super().__init__(env, type="event")
        self.debug_visualizer = debug_visualizer

        # Wrap config items
        self.cfg: dict[str, EventConfigItem] = {}
        for name, item in cfg.items():
            try:
                self.cfg[name] = EventConfigItem(item, env)
            except (TypeError, ValueError) as e:
                e.add_note(f"In the event '{name}'")
                raise

        self._visualizer = EventDebugVisualizer(
            env, self.cfg, debug_visualizer_cfg or {}
        )

    """
    Helpers
    """

    def __getitem__(self, name: str) -> EventConfigItem:
        """Get an event config item by name."""
        return self.cfg[name]

    def get_active(self, name: str) -> torch.Tensor:
        """
        Which environments an event is active in. Shape is (num_envs,).

        An event is active from the step it fires, through the observations of its last step.
        """
        return self.cfg[name].active

    def trigger(
        self, name: str, envs_idx: torch.Tensor | Sequence[int] | None = None
    ) -> None:
        """
        Fire an event on the next step, without waiting for its timer.
        If the event is already active, it will be restarted.

        Args:
            name: The name of the event.
            envs_idx: The environment ids to fire the event for. All environments, if None.
        """
        item = self.cfg[name]
        if not self.enabled:
            return
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx
        elif not isinstance(envs_idx, torch.Tensor):
            envs_idx = torch.as_tensor(envs_idx, device=gs.device, dtype=torch.long)
        item.trigger(envs_idx)

    """
    Operations
    """

    def build(self):
        """Build the event function classes."""
        for item in self.cfg.values():
            item.build()

    def step(self):
        """Run the events. Call this right before the simulation step."""
        for name, item in self.cfg.items():
            try:
                if self.enabled:
                    item.step()
                else:
                    item.stop()
            except Exception:
                print(f"Error running event: '{name}'")
                raise

        if self.debug_visualizer:
            self._visualizer.step()
        else:
            self._visualizer.clear()

    def reset(self, envs_idx: torch.Tensor | None = None):
        """End the events for the given environments, and restart their timers."""
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx
        for item in self.cfg.values():
            item.reset(envs_idx)
        if self.debug_visualizer:
            self._visualizer.reset(envs_idx)
