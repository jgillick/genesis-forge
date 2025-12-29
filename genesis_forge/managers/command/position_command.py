from typing import Tuple, TypedDict

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

from .command_manager import CommandManager, CommandRangeValue


class PositionCommandRange(TypedDict):
    x: CommandRangeValue
    y: CommandRangeValue
    z: CommandRangeValue


class PositionDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: list[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    sphere_offset: float
    """The vertical offset of the debug sphere from the top of the robot"""

    sphere_radius: float
    """The radius of the shaft of the debug sphere"""

    commanded_color: Tuple[float, float, float, float]
    """The color of the commanded position sphere"""


DEFAULT_VISUALIZER_CONFIG: PositionDebugVisualizerConfig = {
    "envs_idx": None,
    "sphere_offset": 0.1,
    "sphere_radius": 0.02,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
}


class PositionCommandManager(CommandManager):
    """
    Generates a position command from uniform distribution.
    The command comprises of a position in the x, y, and z axes.

    IMPORTANT: The position commands are interpreted as world-relative coordinates:
    - X-axis: x coordinate of the target position
    - Y-axis: y coordinate of the target position
    - Z-axis: z coordinate of the target position

    :::{admonition} Debug Visualization

        If you set `debug_visualizer` to True, target sphere will be rendered above the target pos

        Sphere meanings:

        - GREEN: Commanded position for the robot in the world frame

    Args:
        env: The environment to control
        range: The ranges of linear & angular velocities
        resample_time_sec: The time interval between changing the command
        debug_visualizer: Enable the debug sphere visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(GenesisEnv):
            def config(self):
                # Create a position command manager
                self.position_command_manager = PositionCommandManager(
                    self,
                    debug_visualizer=True,
                    range = {
                        "x": (-5.0, 5.0),
                        "y": (-5.0, 5.0),
                        "z": (0.29, 0.31),
                    }
                )

                RewardManager(
                    self,
                    logging_enabled=True,
                    cfg={
                        "tracking_position": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_pos,
                            "params": {
                                "position_cmd_manager": self.position_command_manager,
                            },
                        },
                        # ... other rewards ...
                    },
                )

                # Observations
                ObservationManager(
                    self,
                    cfg={
                        "position_cmd": {"fn": self.position_command_manager.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: PositionCommandRange,
        resample_time_sec: float = 5.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: PositionDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env, range=range, resample_time_sec=resample_time_sec)
        self._sphere_nodes: list = []
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}
        self.debug_envs_idx = None

    """
    Lifecycle Operations
    """

    def build(self):
        """Build the position command manager"""
        super().build()

        # If debug envs_idx is not set, attempt to use the vis_options rendered_envs_idx
        if (
            not self.debug_visualizer
            or self.visualizer_cfg is None
            or self.env.scene is None
        ):
            return
        self.debug_envs_idx = self.visualizer_cfg.get("envs_idx", None)
        if self.debug_envs_idx is None and self.env.scene.vis_options is not None:
            self.debug_envs_idx = self.env.scene.vis_options.rendered_envs_idx
        if self.debug_envs_idx is None:
            self.debug_envs_idx = list[int](range(self.env.num_envs))

    def step(self):
        """Render the command spheres"""
        if not self.enabled:
            return
        super().step()
        self._render_sphere()

    def use_gamepad(
        self,
        gamepad: Gamepad,
        x: int = 0,
        y: int = 1,
        z: int = 2,
    ):
        """
        Use a connected gamepad to control the command.

        Args:
            gamepad: The gamepad to use.
            x: Map this gamepad axis index to the position in the x-direction.
            y: Map this gamepad axis index to the position in the y-direction.
            z: Map this gamepad axis index to the position in the z-direction.
        """
        super().use_gamepad(
            gamepad,
            range_axis={
                "x": x,
                "y": y,
                "z": z,
            },
        )

    """
    Internal Implementation
    """

    def _render_sphere(self):
        """
        Render the command sphere showing position commands.

        The commanded position sphere (green) shows the position in the world frame
        """
        if not self.debug_visualizer:
            return

        # Remove existing spheres
        for sphere in self._sphere_nodes:
            self.env.scene.clear_debug_object(sphere)
        self._sphere_nodes = []

        for i in self.debug_envs_idx:
            # Target sphere in the world frame for visualization)
            self._draw_sphere(
                pos=self.command[i],
                color=self.visualizer_cfg["commanded_color"],
            )

    def _draw_sphere(
        self,
        pos: torch.Tensor,
        color: list[float],
    ):
        try:
            node = self.env.scene.draw_debug_sphere(
                pos=pos.cpu().numpy(),
                color=color,
                radius=self.visualizer_cfg["sphere_radius"],
            )
            if node:
                self._sphere_nodes.append(node)
        except Exception as e:
            print(f"Error adding debug visualizing in PositionCommandManager: {e}")
