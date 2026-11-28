from typing import NotRequired, TypedDict


class ContactDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the contact debug visualizer."""

    envs_idx: NotRequired[list[int] | None]
    """
    The indices of the environments to visualize. Defaults to the environments the scene renders
    (``rendered_envs_idx`` in the scene's ``VisOptions``), or all of them if it isn't set.
    """

    color: NotRequired[tuple[float, float, float, float]]
    """The color of the contact ball"""

    radius: NotRequired[float]
    """The radius of the visualization sphere"""

    force_threshold: NotRequired[float]
    """The threshold, in Newtons, for the contact force to be visualized"""

    fps: NotRequired[int]
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""


DEFAULT_VISUALIZER_CONFIG: ContactDebugVisualizerConfig = {
    "envs_idx": None,
    "size": 0.03,
    "color": (0.5, 0.0, 0.0, 1.0),
    "force_threshold": 1.0,
    "fps": 30,
}
