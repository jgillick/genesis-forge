from typing import Tuple, TypedDict

import os
import torch
import genesis as gs
from genesis.utils.geom import euler_to_R
import numpy as np
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

from .command_manager import CommandManager, CommandRangeValue


class PoseCommandRange(TypedDict):
    pos_x: CommandRangeValue
    pos_y: CommandRangeValue
    pos_z: CommandRangeValue
    euler_x: CommandRangeValue
    euler_y: CommandRangeValue
    euler_z: CommandRangeValue


class PoseDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: list[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    arrow_offset: float
    """The vertical offset of the debug arrows from the top of the robot"""

    arrow_radius: float
    """The radius of the shaft of the debug arrows"""

    commanded_color: Tuple[float, float, float, float]
    """The color of the pose arrow"""


DEFAULT_VISUALIZER_CONFIG: PoseDebugVisualizerConfig = {
    "envs_idx": None,
    "arrow_offset": 0.03,
    "arrow_radius": 0.02,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
}


class PoseCommandManager(CommandManager):
    """
    Generates a pose command from uniform distribution.
    The command comprises of a (x,y,z) position and (x,y,z) euler angles

    IMPORTANT: The pose commands are interpreted as world-relative coordinates:
    - pos-X: x coordinate of the target position
    - pos-Y: y coordinate of the target position
    - pos-Z: z coordinate of the target position
    - euler-X: x coordinate of the target orientation
    - euler-Y: y coordinate of the target orientation
    - euler-Z: z coordinate of the target orientation

    :::{admonition} Debug Visualization

        If you set `debug_visualizer` to True, target arrow will be rendered above the target pose

        Arrow meanings:

        - GREEN: Commanded pose for the robot in the world frame

    Args:
        env: The environment to control
        range: The ranges of positions and orientation
        resample_time_sec: The time interval between changing the command
        debug_visualizer: Enable the debug arrow visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(GenesisEnv):
            def config(self):
                # Create a pose command manager
                self.pose_command_manager = PoseCommandManager(
                    self,
                    debug_visualizer=True,
                    range = {
                        "pos_x": (-5.0, 5.0),
                        "pos_y": (-5.0, 5.0),
                        "euler_z": (-1.57, 1.57),
                    }
                )

                RewardManager(
                    self,
                    logging_enabled=True,
                    cfg={
                        "tracking_pose": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_pose,
                            "params": {
                                "pose_cmd_manager": self.pose_command_manager,
                            },
                        },
                        # ... other rewards ...
                    },
                )

                # Observations
                ObservationManager(
                    self,
                    cfg={
                        "pose_cmd": {"fn": self.pose_command_manager.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: PoseCommandRange,
        resample_time_sec: float = 5.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: PoseDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env, range=range, resample_time_sec=resample_time_sec)
        self._arrow_nodes: list = []
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}
        self.debug_envs_idx = None

    """
    Lifecycle Operations
    """

    def build(self):
        """Build the pose command manager"""
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
        """Render the command arrows"""
        if not self.enabled:
            return
        super().step()
        self._render_arrow()

    def use_gamepad(
        self,
        gamepad: Gamepad,
        pos_x: int = 0,
        pos_y: int = 1,
        pos_z: int = 2,
        euler_x: int = 3,
        euler_y: int = 4,
        euler_z: int = 5,
    ):
        """
        Use a connected gamepad to control the command.

        Args:
            gamepad: The gamepad to use.
            pos_x: Map this gamepad axis index to the position in the x-direction.
            pos_y: Map this gamepad axis index to the position in the y-direction.
            pos_z: Map this gamepad axis index to the position in the z-direction.
            euler_x: Map this gamepad axis index to the orientation in the x-direction.
            euler_y: Map this gamepad axis index to the orientation in the y-direction.
            euler_z: Map this gamepad axis index to the orientation in the z-direction.
        """
        super().use_gamepad(
            gamepad,
            range_axis={
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": pos_z,
                "euler_x": euler_x,
                "euler_y": euler_y,
                "euler_z": euler_z,
            },
        )

    """
    Internal Implementation
    """

    def _render_arrow(self):
        """
        Render the command arrow showing pose commands.

        The commanded pose arrow (green) shows the pose in the world frame
        """
        if not self.debug_visualizer:
            return

        # Remove existing arrows
        for arrow in self._arrow_nodes:
            self.env.scene.clear_debug_object(arrow)
        self._arrow_nodes = []

        for i in self.debug_envs_idx:
            # Target arrow (robot-relative command transformed to world coordinates for visualization)
            self._draw_arrow(
                pos=self.command[i],
                color=self.visualizer_cfg["commanded_color"],
            )

    def _draw_arrow(
        self,
        pos: torch.Tensor,
        euler: torch.Tensor,
        color: list[float],
    ):
        try:
            node = self.env.scene.draw_debug_arrow(
                pos=pos.cpu().numpy(),
                vec=np.tile([0, 0, 1], (pos.shape[0], 1)) @ euler_to_R(euler),
                color=color,
                radius=self.visualizer_cfg["arrow_radius"],
            )
            if node:
                self._sphere_nodes.append(node)
        except Exception as e:
            print(f"Error adding debug visualizing in PoseCommandManager: {e}")
