from typing import TypedDict, Tuple


class ContactDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the contact debug visualizer."""

    envs_idx: list[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    color: Tuple[float, float, float, float]
    """The color of the contact ball"""

    radius: float
    """The radius of the visualization sphere"""

    force_threshold: float
    """The threshold, in Newtons, for the contact force to be visualized"""


DEFAULT_VISUALIZER_CONFIG: ContactDebugVisualizerConfig = {
    "envs_idx": None,
    "size": 0.03,
    "color": (0.5, 0.0, 0.0, 1.0),
    "force_threshold": 1.0,
}
