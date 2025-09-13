from typing import Tuple, TypedDict

import os
import torch
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import entity_lin_vel, transform_by_quat
from genesis_forge.gamepads import Gamepad

from .command_manager import CommandManager, CommandRangeValue


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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


DEFAULT_VISUALIZER_CONFIG: VelocityDebugVisualizerConfig = {
    "envs_idx": None,
    "arrow_offset": 0.02,
    "arrow_radius": 0.02,
    "arrow_max_length": 0.15,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
    "actual_color": (0.0, 0.0, 0.5, 1.0),
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
        resample_time_s: The time interval between changing the command
        debug_visualizer: Enable the debug arrow visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.command_manager = VelocityCommandManager(
                    self,
                    visualize=True,
                    range = {
                        "lin_vel_x_range": (-1.0, 1.0),
                        "lin_vel_y_range": (-1.0, 1.0),
                        "ang_vel_z_range": (-0.5, 0.5),
                    }
                )

            def step(self, actions: torch.Tensor):
                super().step(actions)
                # ...handle actions and rewards calculations ...

                self.command_manager.step()
                obs = self.get_observations()
                return obs, rewards, terminations, timeouts, info


            def reset(self, env_ids: list[int] = None):
                super().reset(env_ids)
                # ...do reset logic here...

                self.command_manager.reset(envs_ids)
                obs = self.get_observations()
                return obs, info

            def calculate_rewards():
                target_cmd = self.command_manager.command

                # Tracking of linear velocity commands (xy axes)
                actual_vel = self.robot.get_vel()
                lin_vel_error = torch.sum(
                    torch.square(target_cmd[:, :2] - actual_vel[:, :2]), dim=1
                )
                line_vel_reward = torch.exp(-lin_vel_error / 0.25)

                # Tracking of angular velocity commands (yaw)
                ang_vel_error = torch.square(target_cmd[:, 2] - self.base_ang_vel[:, 2])
                ang_vel_reward = torch.exp(-ang_vel_error / 0.25)

                # ...additional reward calculations here...

            def get_observations(self):
                return torch.cat(
                    [
                        self.command_manager.command,
                        # ...additional observations here...
                    ],
                    dim=-1,
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: VelocityCommandRange,
        resample_time_s: float = 5.0,
        standing_probability: float = 0.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: VelocityDebugVisualizerConfig = DEFAULT_VISUALIZER_CONFIG,
    ):
        super().__init__(env, range=range, resample_time_sec=resample_time_s)
        self._arrow_nodes: list = []
        self.standing_probability = standing_probability
        self.debug_visualizer = debug_visualizer
        self.visualizer_cfg = {**DEFAULT_VISUALIZER_CONFIG, **debug_visualizer_cfg}

        self._is_standing_env = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

    """
    Properties
    """

    """
    Operations
    """

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
    Implementation
    """

    def _resample_command(self, env_ids: list[int]):
        """
        Overwrites commands for environments that should be standing still.
        """
        super().resample_command(env_ids)
        if not self.enabled:
            return

        num = torch.empty(len(env_ids), device=gs.device)
        self._is_standing_env[env_ids] = (
            num.uniform_(0.0, 1.0) <= self.standing_probability
        )
        standing_envs_idx = self._is_standing_env.nonzero(as_tuple=False).flatten()
        self._command[standing_envs_idx, :] = 0.0

    def _render_arrows(self):
        """
        Render the command arrows showing velocity commands and actual robot velocities.

        The commanded velocity arrow (green) shows the robot-relative velocity command
        transformed to world coordinates for visualization. The blue arrow is the robot's actual velocity.
        """
        if not self.debug_visualizer:
            return

        # Remove existing arrows
        for arrow in self._arrow_nodes:
            self.env.scene.clear_debug_object(arrow)
        self._arrow_nodes = []

        # Scale the arrow size based on the maximum target velocity range
        scale_factor = self.visualizer_cfg["arrow_max_length"] / max(
            *self._range["lin_vel_x"],
            *self._range["lin_vel_y"],
            *self._range["ang_vel_z"],
        )

        # Calculate the center of the robot
        aabb = self.env.robot.get_AABB()
        min_aabb = aabb[:, 0]  # [min_x, min_y, min_z]
        max_aabb = aabb[:, 1]  # [max_x, max_y, max_z]
        robot_x = (min_aabb[:, 0] + max_aabb[:, 0]) / 2
        robot_y = (min_aabb[:, 1] + max_aabb[:, 1]) / 2
        robot_z = max_aabb[:, 2]

        # Set the arrow position over the center of the robot
        arrow_pos = torch.zeros(self.env.num_envs, 3, device=gs.device)
        arrow_pos[:, 0] = robot_x
        arrow_pos[:, 1] = robot_y
        arrow_pos[:, 2] = robot_z + self.visualizer_cfg["arrow_offset"]
        arrow_pos += torch.from_numpy(self.env.scene.envs_offset).to(gs.device)

        # Get robot orientation (quaternion) for coordinate transformation
        robot_quat = self.env.robot.get_quat()

        # Transform robot-relative velocity commands to world coordinates for visualization
        target_velocity = self._resolve_xy_velocity_to_world_frame(
            self.command[:, :2], robot_quat, scale_factor
        )

        # Actual robot velocity (already in world coordinates)
        actual_vec = self.env.robot.get_vel().clone()
        actual_vec[:, 2] = 0.0
        actual_vec[:, :] *= scale_factor

        debug_envs = (
            self.visualizer_cfg["envs_idx"]
            if self.visualizer_cfg["envs_idx"] is not None
            else range(self.env.num_envs)
        )
        for i in debug_envs:
            # Target arrow (robot-relative command transformed to world coordinates for visualization)
            self._draw_arrow(
                pos=arrow_pos[i],
                vec=target_velocity[i],
                color=self.visualizer_cfg["commanded_color"],
            )
            # Actual arrow
            self._draw_arrow(
                pos=arrow_pos[i],
                vec=actual_vec[i],
                color=self.visualizer_cfg["actual_color"],
            )

    def _resolve_xy_velocity_to_world_frame(
        self, xy_velocity: torch.Tensor, robot_quat: torch.Tensor, scale_factor: float
    ) -> torch.Tensor:
        """
        Converts robot-relative XY velocity commands to world coordinates for visualization.

        This method follows the IsaacLab pattern:
        1. Takes robot-relative velocity commands (base frame)
        2. Transforms them to world coordinates using the robot's current orientation
        3. Scales them for visualization

        Args:
            xy_velocity: Robot-relative velocity commands in base frame, shape (num_envs, 2)
            robot_quat: Robot's current orientation quaternion, shape (num_envs, 4)
            scale_factor: Scaling factor for visualization

        Returns:
            World-frame velocity vectors scaled for visualization, shape (num_envs, 3)
        """
        # Create 3D velocity tensor with Z component zeroed for 2D visualization
        vec_3d = torch.zeros(xy_velocity.shape[0], 3, device=xy_velocity.device)
        vec_3d[:, :2] = xy_velocity

        # Transform from robot-relative (base) frame to world frame using quaternion
        # This is the inverse of what robot_lin_vel does
        vec_world = transform_by_quat(vec_3d, robot_quat)

        # Scale the transformed world velocity vector
        vec_world[:, :] *= scale_factor

        return vec_world

    def _draw_arrow(
        self,
        pos: torch.Tensor,
        vec: torch.Tensor,
        color: list[float],
    ):
        # If velocity is zero, don't draw the arrow
        if torch.all(vec == 0.0):
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
