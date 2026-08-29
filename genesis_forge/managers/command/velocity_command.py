import math
from collections.abc import Callable
from typing import NotRequired, TypedDict, cast

import genesis as gs
import numpy as np
import torch
import trimesh
from deprecated import deprecated, deprecated_params
from genesis.utils.geom import quat_to_xyz

from genesis_forge.gamepads import Gamepad
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import transform_by_quat

from .command_manager import CommandManager, CommandRangeValue


class VelocityCommandRange(TypedDict):
    lin_vel_x: CommandRangeValue
    lin_vel_y: CommandRangeValue
    ang_vel_z: CommandRangeValue


class VelocityDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: NotRequired[list[int]]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    fps: NotRequired[int]
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""

    arrow_offset: NotRequired[float]
    """The vertical offset of the debug visuals from the top of the robot"""

    arrow_radius: NotRequired[float]
    """The radius of the shaft of the debug arrows"""

    arrow_max_length: NotRequired[float]
    """The length of the linear velocity arrow at the maximum of the linear velocity range"""

    commanded_color: NotRequired[tuple[float, float, float, float]]
    """The color of the commanded velocity arrow and arc"""

    actual_color: NotRequired[tuple[float, float, float, float]]
    """The color of the actual robot velocity arrow and arc"""

    stopped_color: NotRequired[tuple[float, float, float, float]]
    """The color of the ball shown when no linear velocity is commanded"""

    stopped_ball_radius: NotRequired[float]
    """The radius of the ball shown when no linear velocity is commanded"""

    ang_arc_enabled: NotRequired[bool]
    """Show the angular velocity arcs"""

    ang_arc_gap: NotRequired[float]
    """The space between the tip of a full-length linear velocity arrow and the inner angular velocity arc, and between the two arcs"""

    ang_arc_max_sweep: NotRequired[float]
    """The sweep angle of an angular velocity arc, in radians, at the maximum of the ang_vel_z range (per rotation direction)"""


DEFAULT_VISUALIZER_CONFIG = {
    "envs_idx": [],
    "fps": 30,
    "arrow_offset": 0.12,
    "arrow_radius": 0.01,
    "arrow_max_length": 0.15,
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
    "actual_color": (0.0, 0.0, 0.5, 1.0),
    "stopped_color": (1.0, 0.0, 0.0, 1.0),
    "stopped_ball_radius": 0.03,
    "ang_arc_enabled": True,
    "ang_arc_gap": 0.01,
    "ang_arc_max_sweep": math.radians(45),
}

# Angular velocity arc geometry
_ARC_TUBE_SIDES = 8
"""Number of sides of the arc tube's cross-section polygon"""
_ARC_SECTIONS_PER_FULL_SWEEP = 16
"""Number of segments along an arc of the maximum sweep angle"""
_ARC_MIN_VISIBLE_SWEEP = 0.05
"""Arcs with a smaller sweep (in radians) are too small to see, so they are not drawn"""
_ARC_HEAD_RADIUS_RATIO = 2.5
"""Base radius of the arc's arrowhead cone, relative to the arc tube radius"""
_ARC_HEAD_LENGTH_RATIO = 5.0
"""Length of the arc's arrowhead cone, relative to the arc tube radius"""


class VelocityCommandManager(CommandManager):
    """
    Generates a velocity command from uniform distribution.
    The command comprises of a linear velocity in x and y direction and an angular velocity around the z-axis.

    IMPORTANT: The velocity commands are interpreted as robot-relative coordinates:
    - X-axis: Forward/backward relative to robot's current orientation
    - Y-axis: Left/right relative to robot's current orientation
    - Z-axis: Yaw rotation around robot's vertical axis

    !!! note "Debug Visualization"
        If you set `debug_visualizer` to True, the commanded and actual velocities are
        rendered above your robot.

        Linear velocity:

        - GREEN ARROW: Commanded velocity (robot-relative, transformed to world coordinates for visualization)
          When joystick is "forward", this arrow points in the robot's forward direction
        - BLUE ARROW: Actual robot velocity in world coordinates
        - RED BALL: Shown instead of the green arrow when no linear velocity is commanded

        Angular velocity (yaw rate), shown as flat arcs circling the vertical axis just
        beyond the arrows. The arc length is the yaw rate magnitude and the arc direction
        (clockwise/counter-clockwise from above) is the rotation direction. Arcs start from
        the direction the commanded linear velocity points, or from the robot's forward
        direction when no linear velocity is commanded.

        - GREEN ARC (outer): Commanded angular velocity
        - BLUE ARC (inner): Actual angular velocity

    Args:
        env: The environment to control
        range: The ranges of linear & angular velocities
        stopped_probability: The probability of all velocities being zero for an environment (0.0 = never, 1.0 = always)
        standing_probability: (deprecated) The probability of all velocities being zero for an environment (0.0 = never, 1.0 = always)
        resample_time_sec: The time interval between changing the command
        debug_visualizer: Enable the debug visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(GenesisEnv):
            def config(self):
                # Create a velocity command manager
                self.velocity_command = VelocityCommandManager(
                    self,
                    range={
                        "lin_vel_x": (-1.0, 1.0),
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-0.5, 0.5),
                    },
                    debug_visualizer=True,
                )

                RewardManager(
                    self,
                    logging_enabled=True,
                    cfg={
                        "tracking_lin_vel": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_lin_vel(
                                vel_cmd_manager=self.velocity_command,
                            ),
                        },
                        "tracking_ang_vel": {
                            "weight": 1.0,
                            "fn": rewards.command_tracking_ang_vel(
                                vel_cmd_manager=self.velocity_command,
                            ),
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

    @deprecated_params(
        "standing_probability",
        reason="Use 'stopped_probability' instead"
    )
    def __init__(
        self,
        env: GenesisEnv,
        range: VelocityCommandRange,
        resample_time_sec: float = 5.0,
        stopped_probability: float = 0.0,
        standing_probability: float = 0.0,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: VelocityDebugVisualizerConfig | None = None,
    ):
        super().__init__(
            env,
            range=range,
            resample_time_sec=resample_time_sec,
        )

        self.stopped_probability = stopped_probability or standing_probability
        self.debug_visualizer = debug_visualizer
        self.debug_envs_idx: list | None = None
        self.visualizer_cfg = debug_visualizer_cfg if debug_visualizer_cfg is not None else {}
        self._debug_nodes: list = []

        self._is_stopped_env = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

    """
    Properties
    """
    @property
    def range(self) -> VelocityCommandRange:
        """The velocity range dict."""
        return cast(VelocityCommandRange, self._range)

    @range.setter
    def range(self, range: VelocityCommandRange, *_args, **_kwargs):
        """Update the velocity ranges."""
        CommandManager.range.fset(self, range)

    @property
    def stopped_envs(self):
        """
        A tensor which has the "stopped" state (1 or 0) of all the environments.
        If the state is 1, the command has no movement commanded, linear or angular.
        """
        return self._is_stopped_env

    @property
    @deprecated("Use 'stopped_envs' instead")
    def standing_envs(self):
        return self.stopped_envs

    @property
    @deprecated("Use 'stopped_probability' instead")
    def standing_probability(self) -> float:
        return self.stopped_probability

    @standing_probability.setter
    @deprecated("Use 'stopped_probability' instead")
    def standing_probability(self, value: float) -> None:
        self.stopped_probability = value

    """
    Lifecycle Operations
    """

    def resample_command(self, env_ids: list[int]):
        """
        Overwrites commands for environments that should be stopped.
        """
        super().resample_command(env_ids)
        if not self.enabled:
            return

        # Select the stopped environments and zero their commands
        rand_buffer = torch.empty(len(env_ids), device=gs.device).uniform_(0.0, 1.0)
        self._is_stopped_env[env_ids] = rand_buffer <= self.stopped_probability
        stopped_envs_idx = self._is_stopped_env.nonzero(as_tuple=False).flatten()
        self._command[stopped_envs_idx, :] = 0.0

    def build(self):
        """Build the velocity command manager"""
        super().build()
        self.build_debug()

    def build_debug(self):
        """Build the debug visualizer: buffers, render throttle, and the visual scale factors"""
        if not self.debug_visualizer:
            return

        # Pre-allocate buffers
        self._origin_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._commanded_vec_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._actual_vec_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._scene_env_offset = torch.from_numpy(self.env.scene.envs_offset).to(
            gs.device
        )

        # If debug envs_idx is not set, attempt to use the vis_options rendered_envs_idx
        self.debug_envs_idx = self.visualizer_cfg.get("envs_idx", None)
        if self.debug_envs_idx is None and self.env.scene.vis_options is not None:
            if self.env.scene.vis_options.rendered_envs_idx is not None:
                self.debug_envs_idx = list(self.env.scene.vis_options.rendered_envs_idx)
            else:
                self.debug_envs_idx = list[int](range(self.env.num_envs))

        # Calculate the number of steps per debug render
        self._steps_per_debug_render = math.ceil(1.0 / self._vis_cfg("fps") / self.env.dt)

        # Linear velocity arrows: a full-length arrow is the fastest linear velocity in the range
        velocity_range = self.range
        arrow_max_length = self._vis_cfg("arrow_max_length")
        max_lin_vel = max(
            abs(v) for v in (*velocity_range["lin_vel_x"], *velocity_range["lin_vel_y"])
        )
        self._arrow_scale_factor = (
            arrow_max_length / max_lin_vel if max_lin_vel > 0.0 else 1.0
        )

        # Angular velocity arcs: a full sweep is the fastest angular velocity in the range,
        # and both arcs sit just beyond the tip of a full-length linear velocity arrow
        max_ang_vel = max(abs(v) for v in velocity_range["ang_vel_z"])
        self._ang_arc_enabled = self._vis_cfg("ang_arc_enabled") and max_ang_vel > 0.0
        if self._ang_arc_enabled:
            self._ang_arc_max_sweep = self._vis_cfg("ang_arc_max_sweep")
            self._ang_arc_scale_factor = self._ang_arc_max_sweep / max_ang_vel
            arc_gap = self._vis_cfg("ang_arc_gap")
            self._actual_arc_radius = arrow_max_length + arc_gap
            self._commanded_arc_radius = self._actual_arc_radius + arc_gap

            # Unit-circle cross-section of the arc tube, used by trimesh.creation.revolve.
            # The last point repeats the first exactly, so revolve sees a closed profile.
            theta = np.linspace(0.0, 2.0 * np.pi, _ARC_TUBE_SIDES + 1)
            self._arc_cross_section = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            self._arc_cross_section[-1] = self._arc_cross_section[0]

    def step(self):
        """Render the debug visualization"""
        if not self.enabled:
            return
        super().step()
        self._render_debug()

    def use_gamepad(
        self,
        gamepad: Gamepad,
        lin_vel_y_axis: int = 0,
        lin_vel_x_axis: int = 1,
        ang_vel_z_axis: int = 2,
        *args, **kwargs
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
            invert_axis={
                "lin_vel_x": True,
                "lin_vel_y": True,
                "ang_vel_z": True,
            },
        )

    """
    Internal Implementation
    """

    def _vis_cfg(self, key: str):
        """A debug visualizer config value, or its default when not configured"""
        return self.visualizer_cfg.get(key, DEFAULT_VISUALIZER_CONFIG[key])

    def _render_debug(self):
        """
        Draw the debug visuals above each debug environment's robot: the linear velocity
        arrows (or the stopped ball) and the angular velocity arcs.
        """
        if not self.debug_visualizer or not self.debug_envs_idx:
            return

        # Don't update for every step
        if self.env.step_count % self._steps_per_debug_render != 0:
            return

        self._clear_debug_objects()

        # Compute the values for all environments at once, then draw the debug envs
        robot_quat = self.env.robot.get_quat()
        origin = self._debug_origin()
        commanded_vec = self._commanded_velocity_in_world_frame(robot_quat)
        actual_vec = self._actual_velocity_in_world_frame()
        has_lin_cmd = (self.command[:, :2] != 0.0).any(dim=1)
        commanded_color = self._vis_cfg("commanded_color")
        actual_color = self._vis_cfg("actual_color")

        if self._ang_arc_enabled:
            arc_anchor = self._arc_anchor_angles(robot_quat, commanded_vec, has_lin_cmd)
            commanded_ang_vel = self.command[:, 2].cpu().numpy()
            actual_ang_vel = self.env.robot.get_ang()[:, 2].cpu().numpy()

        for i in self.debug_envs_idx:
            # Commanded linear velocity: an arrow, or a ball when no linear velocity is commanded
            if has_lin_cmd[i]:
                self._draw_arrow(origin[i], commanded_vec[i], commanded_color)
            else:
                self._draw_stopped_ball(origin[i])

            # Actual linear velocity
            self._draw_arrow(origin[i], actual_vec[i], actual_color)

            # Commanded and actual angular velocity
            if self._ang_arc_enabled:
                self._draw_ang_vel_arc(
                    origin[i],
                    commanded_ang_vel[i],
                    arc_anchor[i],
                    self._commanded_arc_radius,
                    commanded_color,
                )
                self._draw_ang_vel_arc(
                    origin[i],
                    actual_ang_vel[i],
                    arc_anchor[i],
                    self._actual_arc_radius,
                    actual_color,
                )

    def _debug_origin(self) -> torch.Tensor:
        """
        The world position above each robot that the debug visuals are drawn from,
        shape (num_envs, 3)
        """
        self._origin_buffer[:] = self.env.robot.get_pos()
        self._origin_buffer[:, 2] += self._vis_cfg("arrow_offset")
        self._origin_buffer += self._scene_env_offset
        return self._origin_buffer

    def _commanded_velocity_in_world_frame(self, robot_quat: torch.Tensor) -> torch.Tensor:
        """
        The commanded XY linear velocity, rotated from the robot frame into the world frame
        and scaled to arrow length, shape (num_envs, 3)

        Args:
            robot_quat: The robot's current orientation quaternion, shape (num_envs, 4)
        """
        self._commanded_vec_buffer[:, :2] = self.command[:, :2]
        self._commanded_vec_buffer[:, 2] = 0.0
        vec_world = cast(
            torch.Tensor, transform_by_quat(self._commanded_vec_buffer, robot_quat)
        )
        return vec_world * self._arrow_scale_factor

    def _actual_velocity_in_world_frame(self) -> torch.Tensor:
        """
        The robot's actual XY linear velocity (already in the world frame) scaled to arrow
        length, shape (num_envs, 3)
        """
        self._actual_vec_buffer[:] = self.env.robot.get_vel() * self._arrow_scale_factor
        self._actual_vec_buffer[:, 2] = 0.0
        return self._actual_vec_buffer

    def _arc_anchor_angles(
        self,
        robot_quat: torch.Tensor,
        commanded_vec: torch.Tensor,
        has_lin_cmd: torch.Tensor,
    ) -> np.ndarray:
        """
        The world-frame angle (radians) that each environment's angular velocity arcs start
        from: the direction of the commanded linear velocity, or the robot's heading when no
        linear velocity is commanded. Shape (num_envs,)
        """
        yaw = quat_to_xyz(robot_quat)[:, 2]
        commanded_dir = torch.atan2(commanded_vec[:, 1], commanded_vec[:, 0])
        return torch.where(has_lin_cmd, commanded_dir, yaw).cpu().numpy()

    def _draw_arrow(
        self,
        origin: torch.Tensor,
        vec: torch.Tensor,
        color: tuple[float, float, float, float],
    ):
        """Draw an arrow from `origin` along `vec`; nothing is drawn for a zero vector"""
        if not torch.any(vec != 0.0):
            return
        self._add_debug_object(
            self.env.scene.draw_debug_arrow,
            pos=origin.cpu().numpy(),
            vec=vec.cpu().numpy(),
            radius=self._vis_cfg("arrow_radius"),
            color=color,
        )

    def _draw_stopped_ball(self, origin: torch.Tensor):
        """Draw the ball that shows no linear velocity is commanded"""
        self._add_debug_object(
            self.env.scene.draw_debug_sphere,
            pos=origin.cpu().numpy(),
            radius=self._vis_cfg("stopped_ball_radius"),
            color=self._vis_cfg("stopped_color"),
        )

    def _draw_ang_vel_arc(
        self,
        origin: torch.Tensor,
        ang_vel: float,
        anchor_angle: float,
        arc_radius: float,
        color: tuple[float, float, float, float],
    ):
        """
        Draw a flat arc around the vertical axis through `origin`, representing a yaw rate.

        The arc starts at `anchor_angle` (world frame, radians) and sweeps counter-clockwise
        for a positive yaw rate or clockwise for a negative one, with a length proportional
        to the yaw rate. A small arrowhead at the end of the arc marks the sweep direction.
        """
        sweep = ang_vel * self._ang_arc_scale_factor
        sweep = max(-self._ang_arc_max_sweep, min(self._ang_arc_max_sweep, sweep))
        if abs(sweep) < _ARC_MIN_VISIBLE_SWEEP:
            return

        # The arc tube: the circular cross-section, offset from the vertical axis by the arc
        # radius, revolved around that axis by the sweep angle. The tube is thinner than the
        # arrows so the two concentric arcs stay visually distinct.
        tube_radius = 0.5 * self._vis_cfg("arrow_radius")
        cross_section = self._arc_cross_section * tube_radius
        cross_section[:, 0] += arc_radius
        sections = max(
            4,
            math.ceil(abs(sweep) / self._ang_arc_max_sweep * _ARC_SECTIONS_PER_FULL_SWEEP),
        )
        arc = trimesh.creation.revolve(cross_section, angle=abs(sweep), sections=sections)

        # trimesh revolves counter-clockwise from the +X axis, so the arc covers the local
        # angles [0, |sweep|]. A counter-clockwise sweep travels from 0 to |sweep|, a
        # clockwise sweep from |sweep| back to 0, so the arrowhead cone goes on the
        # corresponding end, pointing along the direction of travel.
        head_angle = abs(sweep) if sweep > 0.0 else 0.0
        head = self._arc_head_mesh(arc_radius, tube_radius, head_angle, math.copysign(1.0, sweep))
        mesh = trimesh.util.concatenate([arc, head])
        mesh.visual.vertex_colors = color

        # Rotate the mesh about Z so the arc starts at the anchor for a counter-clockwise
        # sweep, or ends at the anchor for a clockwise sweep, and translate it to the origin
        start_angle = anchor_angle if sweep > 0.0 else anchor_angle + sweep
        cos_a = math.cos(start_angle)
        sin_a = math.sin(start_angle)
        center = origin.cpu().numpy()
        transform = np.array([
            [cos_a, -sin_a, 0.0, center[0]],
            [sin_a, cos_a, 0.0, center[1]],
            [0.0, 0.0, 1.0, center[2]],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._add_debug_object(self.env.scene.draw_debug_mesh, mesh, T=transform)

    def _arc_head_mesh(
        self,
        arc_radius: float,
        tube_radius: float,
        angle: float,
        direction: float,
    ) -> trimesh.Trimesh:
        """
        The arrowhead cone for an angular velocity arc, in the arc's local frame: its base
        sits on the arc at `angle` (radians from the +X axis) and it points along the arc's
        tangent, counter-clockwise for `direction` +1 or clockwise for -1.
        """
        cone = trimesh.creation.cone(
            radius=tube_radius * _ARC_HEAD_RADIUS_RATIO,
            height=tube_radius * _ARC_HEAD_LENGTH_RATIO,
        )

        # The cone is built along +Z. Build a rotation that maps +Z onto the tangent
        # (its columns are the images of the X, Y, and Z axes) and place it on the arc.
        tangent_x = -math.sin(angle) * direction
        tangent_y = math.cos(angle) * direction
        cone.apply_transform(np.array([
            [-tangent_y, 0.0, tangent_x, arc_radius * math.cos(angle)],
            [tangent_x, 0.0, tangent_y, arc_radius * math.sin(angle)],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))
        return cone

    def _add_debug_object(self, draw_fn: Callable, *args, **kwargs):
        """
        Call one of the scene's `draw_debug_*` functions and keep the node it returns, so
        the object is removed on the next render
        """
        try:
            node = draw_fn(*args, **kwargs)
        except Exception as e: # noqa
            print(f"Error adding debug visualizing in VelocityCommandManager: {e}")
            return
        if node:
            self._debug_nodes.append(node)

    def _clear_debug_objects(self):
        """Remove all debug objects drawn by the previous render"""
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []
