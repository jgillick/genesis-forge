from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, TypedDict

from genesis_forge.managers.config import ConfigItemDict, EventMdpFn

TimeRange = float | tuple[float, float]
"""A fixed time, or a (min, max) range to randomize within, in seconds."""


class EventConfig(ConfigItemDict):
    """Defines an event."""

    fn: Callable[..., None] | EventMdpFn  # pyright: ignore[reportIncompatibleVariableOverride]
    """
    What to do when the event fires: a constructed EventMdpFn or DurationEventMdpFn, or a
    plain function.

    Args:
        env: The environment instance.
        envs_idx: The environment ids the event fired for.
        **params: Additional parameters to pass to a plain function from the params dictionary.
    """

    interval_range_s: NotRequired[TimeRange | None]
    """
    The time to wait before the event fires, counted from the environment's reset or the
    end of its previous event. A (min, max) tuple is sampled separately for each
    environment, every time. Leave it out to only fire it with :meth:`EventManager.trigger`.
    """

    duration_range_s: NotRequired[TimeRange]
    """
    How long the event lasts, rounded to whole steps. A (min, max) tuple is sampled each
    time the event fires. Required for a DurationEventMdpFn, and not allowed for anything else.
    """


class EventDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the event debug visualizer."""

    envs_idx: NotRequired[list[int] | None]
    """
    The indices of the environments to visualize. Defaults to the environments the scene renders
    (``rendered_envs_idx`` in the scene's ``VisOptions``), or all of them if it isn't set.
    """

    fps: NotRequired[int]
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""

    min_display_s: NotRequired[float]
    """The shortest time to show an event, in seconds, so that very short events can be seen"""


DEFAULT_VISUALIZER_CONFIG: EventDebugVisualizerConfig = {
    "envs_idx": None,
    "fps": 30,
    "min_display_s": 0.3,
}
