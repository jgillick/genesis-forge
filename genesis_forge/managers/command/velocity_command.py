import math
import warnings
from typing import NotRequired, TypedDict, cast

import genesis as gs
import numpy as np
import torch
from deprecated import deprecated, deprecated_params
from genesis.utils.geom import quat_to_xyz

from genesis_forge.gamepads import Gamepad
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.meshes import arrow_mesh, flat_arc_arrow_mesh
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

    ang_arc_enabled: NotRequired[bool]
    """Show the angular velocity arcs"""

    ang_arc_width: NotRequired[float]
    """The radial width of the actual angular velocity arc"""

    ang_arc_height: NotRequired[float]
    """The vertical thickness of the actual angular velocity arc"""

    ang_arc_max_sweep: NotRequired[float]
    """The sweep angle of an angular velocity arc, in radians, at the maximum of the ang_vel_z range (per rotation direction)"""

    stopped_color: NotRequired[tuple[float, float, float, float]]
    """The color of the ball shown when no linear velocity is commanded"""

    stopped_ball_radius: NotRequired[float]
    """The radius of the ball shown when no linear velocity is commanded"""

    commanded_color: NotRequired[tuple[float, float, float, float]]
    """The color of the commanded velocity arrow and arc"""

    actual_color: NotRequired[tuple[float, float, float, float]]
    """The color of the actual robot velocity arrow and arc"""

    ##
    # DEPRECATED
    #
    standing_ball_radius: NotRequired[float]
    """Deprecated: use `stopped_ball_radius` instead"""

    standing_color: NotRequired[tuple[float, float, float, float]]
    """Deprecated: use `stopped_color` instead"""


_DEPRECATED_VISUALIZER_KEYS = {
    "standing_color": "stopped_color",
    "standing_ball_radius": "stopped_ball_radius",
}


DEFAULT_VISUALIZER_CONFIG = {
    "envs_idx": [],
    "fps": 30,
    "arrow_offset": 0.12,
    "arrow_radius": 0.02,
    "arrow_max_length": 0.15,
    "ang_arc_enabled": True,
    "ang_arc_width": 0.03,
    "ang_arc_height": 0.0025,
    "ang_arc_max_sweep": math.radians(45),
    "commanded_color": (0.0, 0.5, 0.0, 1.0),
    "actual_color": (0.0, 0.0, 0.5, 1.0),
    "stopped_ball_radius": 0.03,
    "stopped_color": (1.0, 0.0, 0.0, 1.0),
}

DEBUG_ARC_MIN_SWEEP = 0.05
"""Angular velocity arcs with a smaller sweep (in radians) are too small to see, so they are not drawn"""

DEBUG_ARC_COMMANDED_WIDTH_RATIO = 0.5
"""The width of the commanded angular velocity arc, relative to the actual angular velocity arc"""

DEBUG_ARC_COMMANDED_HEIGHT_RATIO = 1.25
"""The height of the commanded angular velocity arc, relative to the actual angular velocity arc"""


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

        Color meanings:

        - GREEN ARROW: Commanded velocity (robot-relative, transformed to world coordinates for visualization)
          When joystick is "forward", this arrow points in the robot's forward direction
        - BLUE ARROW: Actual robot velocity in world coordinates
        - RED BALL: Shown when no linear velocity is commanded

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
        "standing_probability", reason="Use 'stopped_probability' instead"
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
        self.visualizer_cfg = self._convert_deprecated_visualizer_keys(
            debug_visualizer_cfg if debug_visualizer_cfg is not None else {}
        )
        self._debug_nodes: list = []

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
    @deprecated("Use 'stopped_envs()' instead")
    def standing_envs(self):
        return self.stopped_envs()

    @property
    @deprecated("Use 'stopped_probability' instead")
    def standing_probability(self) -> float:
        return self.stopped_probability

    @standing_probability.setter
    @deprecated("Use 'stopped_probability' instead")
    def standing_probability(self, value: float) -> None:
        self.stopped_probability = value

    def stopped_envs(self, threshold: float = 0.0025) -> torch.Tensor:
        """
        The environments whose command is effectively stopped: no movement commanded,
        linear or angular.

        An environment counts as stopped when the norm of its full command (linear xy
        and angular z) is below ``threshold`` — whether the environment was selected by
        ``stopped_probability`` or its command is near zero for any other reason
        (sampling, a curriculum range, a centered gamepad stick).

        Args:
            threshold: The command is considered stopped when the norm of all its
                       components does not exceed this value. Pass 0.0 to match
                       only exactly-zero commands.

        Returns:
            Boolean tensor, shape (num_envs,)
        """
        return torch.norm(self.command, dim=1) <= threshold

    """
    Lifecycle Operations
    """

    def resample_command(self, env_ids: torch.Tensor):
        """
        Overwrites commands for environments that should be stopped.
        """
        super().resample_command(env_ids)
        if not self.enabled:
            return

        # Select the stopped environments and zero their commands.
        # torch.rand samples [0, 1), so `<` makes probability 0.0 never stop an
        # environment and 1.0 always stop it.
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=gs.device)
        stopped = torch.rand(len(env_ids), device=gs.device) < self.stopped_probability
        self._command[env_ids[stopped], :] = 0.0

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
        self._steps_per_debug_render = math.ceil(
            1.0 / self._debug_cfg("fps") / self.env.dt
        )

        # Linear velocity arrows: a full-length arrow is the fastest linear velocity in the range
        velocity_range = self.range
        arrow_max_length = self._debug_cfg("arrow_max_length")
        max_lin_vel = max(
            abs(v) for v in (*velocity_range["lin_vel_x"], *velocity_range["lin_vel_y"])
        )
        self._arrow_scale_factor = (
            arrow_max_length / max_lin_vel if max_lin_vel > 0.0 else 1.0
        )

        # Angular velocity arcs: a full sweep is the fastest angular velocity in the range,
        # and both arcs share a radius just beyond the tip of a full-length linear velocity
        # arrow
        max_ang_vel = max(abs(v) for v in velocity_range["ang_vel_z"])
        self._ang_arc_enabled = self._debug_cfg("ang_arc_enabled") and max_ang_vel > 0.0
        if self._ang_arc_enabled:
            self._ang_arc_max_sweep = self._debug_cfg("ang_arc_max_sweep")
            self._ang_arc_scale_factor = self._ang_arc_max_sweep / max_ang_vel

            # Place the arc outside of the full-length linear velocity arrow
            radius_gap = self._debug_cfg("arrow_radius") * 1.6
            self._ang_arc_radius = arrow_max_length + radius_gap

    def step(self):
        """Render the debug visualization"""
        if not self.enabled:
            return
        super().step()
        self._render_debug()

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Resample the commands of the reset environments and redraw the debug visualization.

        The environment resets after the managers have stepped, so the visuals drawn during
        this step show the robot's pre-reset position and command. Redraw them now so they
        match the reset state, rather than waiting for the next throttled render.
        """
        super().reset(env_ids)
        if not self.enabled or not self.debug_visualizer or not self.debug_envs_idx:
            return
        if env_ids is None or set(self.debug_envs_idx).intersection(env_ids.tolist()):
            self._render_debug(force=True)

    def use_gamepad(
        self,
        gamepad: Gamepad,
        lin_vel_y_axis: int = 0,
        lin_vel_x_axis: int = 1,
        ang_vel_z_axis: int = 2,
        *args,
        **kwargs,
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

    def _convert_deprecated_visualizer_keys(
        self, cfg: VelocityDebugVisualizerConfig
    ) -> VelocityDebugVisualizerConfig:
        """
        Convert deprecated debug visualizer config keys to their replacements, with a
        deprecation warning. The config is read with `.get()`, so without this an old
        key would silently fall back to the default value.
        """
        if not any(key in cfg for key in _DEPRECATED_VISUALIZER_KEYS):
            return cfg
        # Work on a copy so popping keys doesn't mutate the caller's config dict
        converted: dict[str, object] = dict(cfg)
        for old_key, new_key in _DEPRECATED_VISUALIZER_KEYS.items():
            if old_key in converted:
                warnings.warn(
                    f"The '{old_key}' debug visualizer config key is deprecated; use '{new_key}' instead",
                    DeprecationWarning,
                    stacklevel=3,
                )
                converted.setdefault(new_key, converted.pop(old_key))
        return cast(VelocityDebugVisualizerConfig, converted)

    def _debug_cfg(self, key: str):
        """A debug visualizer config value, or its default when not configured"""
        return self.visualizer_cfg.get(key, DEFAULT_VISUALIZER_CONFIG[key])

    def _render_debug(self, force: bool = False):
        """
        Draw the debug visuals above each debug environment's robot: the linear velocity
        arrows (or the stopped ball) and the angular velocity arcs.

        Args:
            force: Draw now, even if this step is not a scheduled render for the configured FPS
        """
        if not self.debug_visualizer or not self.debug_envs_idx:
            return

        # Don't update for every step
        if not force and self.env.step_count % self._steps_per_debug_render != 0:
            return

        self._clear_debug_objects()

        # Compute the values for all environments at once, then draw the debug envs
        robot_quat = self.env.robot.get_quat()
        origin = self._debug_origin()
        commanded_vec = self._commanded_velocity_in_world_frame(robot_quat)
        actual_vec = self._actual_velocity_in_world_frame()
        has_lin_cmd = (self.command[:, :2] != 0.0).any(dim=1)
        commanded_color = self._debug_cfg("commanded_color")
        actual_color = self._debug_cfg("actual_color")

        if self._ang_arc_enabled:
            arc_anchor = self._arc_anchor_angles(robot_quat, commanded_vec, has_lin_cmd)
            commanded_ang_vel = self.command[:, 2].cpu().numpy()
            actual_ang_vel = self.env.robot.get_ang()[:, 2].cpu().numpy()

        for i in self.debug_envs_idx:
            try:
                # Commanded linear velocity: an arrow, or a ball when no linear velocity is commanded
                if has_lin_cmd[i]:
                    self._draw_arrow(origin[i], commanded_vec[i], commanded_color)
                else:
                    self._draw_stopped_ball(origin[i])

                # Actual linear velocity
                self._draw_arrow(origin[i], actual_vec[i], actual_color)

                # Actual angular velocities
                if self._ang_arc_enabled:
                    self._draw_ang_vel_arc(
                        origin[i],
                        commanded_ang_vel[i],
                        arc_anchor[i],
                        commanded_color,
                        commanded=True,
                    )
                    self._draw_ang_vel_arc(
                        origin[i],
                        actual_ang_vel[i],
                        arc_anchor[i],
                        actual_color,
                        commanded=False,
                    )
            except Exception as e:  # noqa
                print(f"Error drawing debug visuals in VelocityCommandManager: {e}")

    def _debug_origin(self) -> torch.Tensor:
        """
        The world position above each robot that the debug visuals are drawn from,
        shape (num_envs, 3)
        """
        self._origin_buffer[:] = self.env.robot.get_pos()
        self._origin_buffer[:, 2] += self._debug_cfg("arrow_offset")
        self._origin_buffer += self._scene_env_offset
        return self._origin_buffer

    def _commanded_velocity_in_world_frame(
        self, robot_quat: torch.Tensor
    ) -> torch.Tensor:
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
        vec_np = vec.cpu().numpy()
        length = float(np.linalg.norm(vec_np))
        if length == 0.0:
            return
        mesh = arrow_mesh(
            origin.cpu().numpy(),
            vec_np,
            self._debug_cfg("arrow_radius"),
            color=color,
        )
        node = self.env.scene.draw_debug_mesh(mesh)
        self._debug_nodes.append(node)

    def _draw_stopped_ball(self, origin: torch.Tensor):
        """Draw the ball that shows no linear velocity is commanded"""
        node = self.env.scene.draw_debug_sphere(
            pos=origin.cpu().numpy(),
            radius=self._debug_cfg("stopped_ball_radius"),
            color=self._debug_cfg("stopped_color"),
        )
        self._debug_nodes.append(node)

    def _draw_ang_vel_arc(
        self,
        origin: torch.Tensor,
        ang_vel: float,
        anchor_angle: float,
        color: tuple[float, float, float, float],
        commanded: bool,
    ):
        """
        Draw a flat arc around the vertical axis representing a yaw rate.

        The arc starts at `anchor_angle` (world frame, radians) and sweeps counter-clockwise
        for a positive yaw rate or clockwise for a negative one, with a length proportional
        to the yaw rate. A small arrowhead at the end of the arc marks the sweep direction.
        """
        sweep = ang_vel * self._ang_arc_scale_factor
        sweep = max(-self._ang_arc_max_sweep, min(self._ang_arc_max_sweep, sweep))
        if abs(sweep) < DEBUG_ARC_MIN_SWEEP:
            return

        width = self._debug_cfg("ang_arc_width")
        thickness = self._debug_cfg("ang_arc_height")
        if commanded:
            width *= DEBUG_ARC_COMMANDED_WIDTH_RATIO
            thickness *= DEBUG_ARC_COMMANDED_HEIGHT_RATIO
        mesh = flat_arc_arrow_mesh(
            origin.cpu().numpy(),
            self._ang_arc_radius,
            sweep,
            width,
            thickness,
            start_angle=anchor_angle,
            color=color,
        )
        node = self.env.scene.draw_debug_mesh(mesh)
        self._debug_nodes.append(node)

    def _clear_debug_objects(self):
        """Remove all debug objects drawn by the previous render"""
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []
