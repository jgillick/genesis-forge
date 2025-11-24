import math
from typing import Tuple, TypedDict

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import transform_by_quat
from genesis_forge.gamepads import Gamepad

from .command_manager import CommandManager, CommandRangeValue


class VelocityCommandRange(TypedDict):
    lin_vel_x: CommandRangeValue
    lin_vel_y: CommandRangeValue
    ang_vel_z: CommandRangeValue


class VelocityDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: list[int]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    arrow_offset: float
    """The vertical offset of the debug arrows from the top of the robot"""

    arrow_radius: float
    """The radius of the shaft of the debug arrows"""

    arrow_max_length: float
    """The maximum length of the debug arrows"""

    commanded_color: Tuple[float, float, float, float]
    """The color of the commanded velocity arrow"""

    actual_color: Tuple[float, float, float, float]
    """The color of the actual robot velocity arrow"""

    fps: int
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""


DEFAULT_VISUALIZER_CONFIG: VelocityDebugVisualizerConfig = {
    "envs_idx": None,
    "arrow_offset": 0.12,
    "arrow_radius": 0.02,
    "arrow_max_length": 0.15,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
    "actual_color": (0.0, 0.0, 0.5, 1.0),
    "fps": 30,
}


class VelocityCommandManager(CommandManager):
    """
    Generates a velocity command from uniform distribution.
    The command comprises of a linear velocity in x and y direction and an angular velocity around the z-axis.

    IMPORTANT: The velocity commands are interpreted as robot-relative coordinates:
    - X-axis: Forward/backward relative to robot's current orientation
    - Y-axis: Left/right relative to robot's current orientation
    - Z-axis: Yaw rotation around robot's vertical axis

    :::{admonition} Debug Visualization

        If you set `debug_visualizer` to True, arrows will be rendered above your robot
        showing the commanded velocity vs the actual velocity.

        Arrow meanings:

        - GREEN: Commanded velocity (robot-relative, transformed to world coordinates for visualization)
          When joystick is "forward", this arrow points in the robot's forward direction
        - BLUE: Actual robot velocity in world coordinates

    Args:
        env: The environment to control
        range: The ranges of linear & angular velocities
        standing_probability: The probability of all velocities being zero for an environment (0.0 = never, 1.0 = always)
        resample_time_sec: The time interval between changing the command
        debug_visualizer: Enable the debug arrow visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(GenesisEnv):
            def config(self):
                # Create a velocity command manager
                self.command_manager = VelocityCommandManager(
                    self,
                    visualize=True,
                    range = {
                        "lin_vel_x_range": (-1.0, 1.0),
                        "lin_vel_y_range": (-1.0, 1.0),
                        "ang_vel_z_range": (-0.5, 0.5),
                    }
                )

                RewardManager(
                    self,
                    logging_enabled=True,
                    cfg={
                        "tracking_lin_vel": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_lin_vel,
                            "params": {
                                "vel_cmd_manager": self.velocity_command,
                            },
                        },
                        "tracking_ang_vel": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_ang_vel,
                            "params": {
                                "vel_cmd_manager": self.velocity_command,
                            },
                        },
                        # ... other rewards ...
                    },
                )

                # Observations
                ObservationManager(
                    self,
                    cfg={
                        "velocity_cmd": {"fn": self.velocity_command.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: VelocityCommandRange,
        resample_time_sec: float = 5.0,
        standing_probability: float = 0.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: VelocityDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env, range=range, resample_time_sec=resample_time_sec)
        self._arrow_nodes: list = []
        self.standing_probability = standing_probability
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}
        self.debug_envs_idx = None

        self._is_standing_env = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

    """
    Lifecycle Operations
    """

    def resample_command(self, env_ids: list[int]):
        """
        Overwrites commands for environments that should be standing still.
        """
        super().resample_command(env_ids)
        if not self.enabled:
            return

        # Set standing environments
        rand_buffer = torch.empty(len(env_ids), device=gs.device).uniform_(0.0, 1.0)
        self._is_standing_env[env_ids] = rand_buffer <= self.standing_probability
        standing_envs_idx = self._is_standing_env.nonzero(as_tuple=False).flatten()
        self._command[standing_envs_idx, :] = 0.0

    def build(self):
        """Build the velocity command manager"""
        super().build()
        self.build_debug()

    def build_debug(self):
        """Build the debug components of the velocity command manager"""
        if not self.debug_visualizer or self.visualizer_cfg is None:
            return

        # Pre-allocate buffers
        self._arrow_pos_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._actual_vec_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._vec_3d_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._scene_env_offset = torch.from_numpy(self.env.scene.envs_offset).to(
            gs.device
        )

        # If debug envs_idx is not set, attempt to use the vis_options rendered_envs_idx
        self.debug_envs_idx = self.visualizer_cfg.get("envs_idx", None)
        if self.debug_envs_idx is None and self.env.scene.vis_options is not None:
            self.debug_envs_idx = self.env.scene.vis_options.rendered_envs_idx
        if self.debug_envs_idx is None:
            self.debug_envs_idx = list[int](range(self.env.num_envs))

        # Calculate the number of steps per debug render
        fps = self.visualizer_cfg.get("fps", 30)
        self._steps_per_debug_render = math.ceil(1.0 / fps / self.env.dt)

        # Arrow scale factor
        # Scales the arrow size based on the maximum target velocity range
        self._arrow_scale_factor = self.visualizer_cfg["arrow_max_length"] / max(
            *self._range["lin_vel_x"],
            *self._range["lin_vel_y"],
            *self._range["ang_vel_z"],
        )

    def step(self):
        """Render the command arrows"""
        if not self.enabled:
            return
        super().step()
        self._render_arrows()

    def use_gamepad(
        self,
        gamepad: Gamepad,
        lin_vel_y_axis: int = 0,
        lin_vel_x_axis: int = 1,
        ang_vel_z_axis: int = 2,
    ):
        """
        Use a connected gamepad to control the command.

        Args:
            gamepad: The gamepad to use.
            lin_vel_x_axis: Map this gamepad axis index to the linear velocity in the x-direction.
            lin_vel_y_axis: Map this gamepad axis index to the linear velocity in the y-direction.
            ang_vel_z_axis: Map this gamepad axis index to the angular velocity in the z-direction.
        """
        super().use_gamepad(
            gamepad,
            range_axis={
                "lin_vel_x": lin_vel_x_axis,
                "lin_vel_y": lin_vel_y_axis,
                "ang_vel_z": ang_vel_z_axis,
            },
        )

    """
    Internal Implementation
    """

    def _render_arrows(self):
        """
        Render the command arrows showing velocity commands and actual robot velocities.

        The commanded velocity arrow (green) shows the robot-relative velocity command
        transformed to world coordinates for visualization. The blue arrow is the robot's actual velocity.
        """
        # Is the debug visualizer enabled?
        if not self.debug_visualizer or len(self.debug_envs_idx) == 0:
            return

        # Don't update for every step
        if self.env.step_count % self._steps_per_debug_render != 0:
            return

        # Remove existing arrows
        for arrow in self._arrow_nodes:
            self.env.scene.clear_debug_object(arrow)
        self._arrow_nodes = []

        # Calculate the arrow position over the robot
        self._arrow_pos_buffer[:] = self.env.robot.get_pos()
        self._arrow_pos_buffer[:, 2] += self.visualizer_cfg["arrow_offset"]
        self._arrow_pos_buffer += self._scene_env_offset

        # Transform robot-relative velocity commands to world coordinates for visualization
        target_velocity = self._target_velocity_in_world_frame()

        # Actual robot velocity (already in world coordinates)
        self._actual_vec_buffer[:] = self.env.robot.get_vel() * self._arrow_scale_factor
        self._actual_vec_buffer[:, 2] = 0.0

        for i in self.debug_envs_idx:
            # Target arrow (robot-relative command transformed to world coordinates for visualization)
            self._draw_arrow(
                pos=self._arrow_pos_buffer[i],
                vec=target_velocity[i],
                color=self.visualizer_cfg["commanded_color"],
            )

            # Actual arrow
            self._draw_arrow(
                pos=self._arrow_pos_buffer[i],
                vec=self._actual_vec_buffer[i],
                color=self.visualizer_cfg["actual_color"],
            )

    def _target_velocity_in_world_frame(self) -> torch.Tensor:
        """
        Converts robot-relative XY velocity commands to world coordinates for visualization.

        This method follows the IsaacLab pattern:
        1. Takes robot-relative velocity commands (base frame)
        2. Transforms them to world coordinates using the robot's current orientation
        3. Scales them for visualization

        Returns:
            World-frame velocity vectors scaled for visualization, shape (num_envs, 3)
        """
        # Create 3D velocity tensor with Z component zeroed for 2D visualization
        self._vec_3d_buffer[:, :2] = self.command[:, :2]
        self._vec_3d_buffer[:, 2] = 0.0

        # Transform from robot-relative (base) frame to world frame using quaternion
        # This is the inverse of what robot_lin_vel does
        robot_quat = self.env.robot.get_quat()
        vec_world = transform_by_quat(self._vec_3d_buffer, robot_quat)

        # Scale the transformed world velocity vector
        vec_world[:, :] *= self._arrow_scale_factor

        return vec_world

    def _draw_arrow(
        self,
        pos: torch.Tensor,
        vec: torch.Tensor,
        color: list[float],
    ):
        # If velocity is zero, don't draw the arrow
        if not torch.any(vec != 0.0):
            return
        try:
            node = self.env.scene.draw_debug_arrow(
                pos=pos.cpu().numpy(),
                vec=vec.cpu().numpy(),
                color=color,
                radius=self.visualizer_cfg["arrow_radius"],
            )
            if node:
                self._arrow_nodes.append(node)
        except Exception as e:
            print(f"Error adding debug visualizing in VelocityCommandManager: {e}")
