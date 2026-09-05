import math
from collections.abc import Callable
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import genesis as gs
import torch
from genesis.utils.geom import inv_quat, transform_by_quat

from genesis_forge.genesis_env import GenesisEnv

from .command_manager import CommandManager, CommandRangeValue

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


MAX_RESAMPLE_ATTEMPTS = 10
"""
How many times a goal that landed on top of something is redrawn before the manager gives
up and keeps the last draw. Without a limit, a scene with no free space left would loop
forever.
"""


class Pose2dCommandRange(TypedDict):
    """The ranges a goal pose is drawn from."""

    x: CommandRangeValue
    y: CommandRangeValue
    heading: CommandRangeValue


class Pose2dDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: NotRequired[list[int]]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    fps: NotRequired[int]
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""

    arrow_length: NotRequired[float]
    """The length of the goal arrow"""

    arrow_radius: NotRequired[float]
    """The thickness of the goal arrow"""

    arrow_height: NotRequired[float]
    """The height above the ground to draw the goal arrow at"""

    goal_color: NotRequired[tuple[float, float, float, float]]
    """The color of the goal arrow"""

    reached_color: NotRequired[tuple[float, float, float, float]]
    """The color of the goal arrow when the goal has been reached"""


DEFAULT_VISUALIZER_CONFIG = {
    "envs_idx": [],
    "fps": 30,
    "arrow_length": 0.25,
    "arrow_radius": 0.02,
    "arrow_height": 0.05,
    "goal_color": (0.0, 0.5, 0.0, 1.0),
    "reached_color": (1.0, 0.0, 0.0, 1.0),
}


def heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    Which way an entity is facing, seen from above: the compass direction of its nose, in
    radians, ignoring any tilt forwards, backwards, or to the side.

    Args:
        quat: Orientation quaternions in (w, x, y, z) order, shape (num_envs, 4)

    Returns:
        torch.Tensor: Heading in radians, shape (num_envs,)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # The standard way to pull the flat, around-the-vertical-axis part out of a quaternion
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def shortest_turn_to(target_angle: torch.Tensor, current_angle: torch.Tensor) -> torch.Tensor:
    """
    How far `current_angle` has to turn to face `target_angle`, taking the shorter way around.

    Angles wrap around at half a turn, so plain subtraction can give a silly answer: going
    from 179 degrees to -179 degrees looks like a 358 degree turn, when really it is a 2
    degree turn the other way. Shifting by half a turn, taking the remainder, and shifting
    back folds the answer into "at most half a turn, in whichever direction is closer".

    Returns:
        torch.Tensor: Radians to turn, positive to the left and negative to the right
    """
    difference = target_angle - current_angle
    return (difference + math.pi) % (2 * math.pi) - math.pi


class Pose2dCommand(CommandManager):
    """
    Generates a goal pose command: a point to drive to, and the direction to be facing
    once there. Use this for navigation tasks where the robot should arrive somewhere
    lined up for whatever it does next, like backing into a charging dock or pulling up
    to a shelf facing it.

    The X/Y position and the heading are drawn independently, so the heading is not
    simply "whichever way you happened to arrive from" -- the robot has to both get there
    and turn to face the right way. If you only care about the position, leave
    `heading_reached_threshold` unset and give the heading a small weight (or no reward
    at all); the goal then counts as reached on position alone.

    A goal is never placed on top of anything else in the scene. Every entity, including
    the robot itself, is given a circle of clear space around it, so goals don't spawn
    inside an obstacle or right under the robot's wheels. This is automatic -- there is
    nothing to configure.

    The goal for each environment is resampled when the environment resets, when the
    goal is reached (if `resample_on_reached` is True), and on a timer
    (if `resample_time_sec` is set).

    !!! note "Debug Visualization"
        If you set `debug_visualizer` to True, an arrow is drawn at each goal: it starts
        at the goal position and points the way to face on arrival, changing color when
        the goal has been reached.

    Args:
        env: The environment to control
        range: The X/Y and heading ranges to draw goal poses from, in the environment's local frame
        resample_time_sec: The time interval between changing the goal.
                           Defaults to None: the goal only changes on reset or when reached.
        goal_reached_threshold: The distance (in meters) at which the goal counts as reached.
        heading_reached_threshold: How closely (in radians) the entity must be facing the goal
                                   heading for the goal to count as reached. Defaults to None:
                                   arriving at the position is enough, whichever way it is facing.
        resample_on_reached: Sample a new goal for an environment when its goal is reached.
        entity: The entity that is navigating to the goal. Defaults to `env.robot`.
        debug_visualizer: Enable the debug visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.pose_command = Pose2dCommand(
                    self,
                    range={
                        "x": (-2.0, 2.0),
                        "y": (-2.0, 2.0),
                        "heading": (-math.pi, math.pi),
                    },
                    debug_visualizer=True,
                )

                RewardManager(
                    self,
                    cfg={
                        "position_progress": {
                            "weight": 1.0,
                            "fn": rewards.position_progress(
                                pose_cmd_manager=self.pose_command,
                            ),
                        },
                        "heading_progress": {
                            "weight": 0.5,
                            "fn": rewards.heading_progress(
                                pose_cmd_manager=self.pose_command,
                            ),
                        },
                        "reached_goal": {
                            "weight": 10.0,
                            "fn": rewards.reached_goal(
                                pose_cmd_manager=self.pose_command,
                            ),
                        },
                        # ... other rewards ...
                    },
                )

                # Observations
                ObservationManager(
                    self,
                    cfg={
                        # The goal pose, from the robot's own point of view
                        "goal_pose": {"fn": self.pose_command.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: Pose2dCommandRange,
        resample_time_sec: float | None = None,
        goal_reached_threshold: float = 0.15,
        heading_reached_threshold: float | None = None,
        resample_on_reached: bool = True,
        entity: "RigidEntity | None" = None,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: Pose2dDebugVisualizerConfig | None = None,
    ):
        # The resampled mask is written by resample_command, which the base constructor
        # can indirectly trigger, so it must exist first
        self._resampled_last_step = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

        # Filled in by build(), once the scene exists and can be inspected
        self._avoided_entities: list = []
        self._avoided_margins: list[float] = []

        super().__init__(
            env,
            range=range,
            resample_time_sec=resample_time_sec,
        )

        self.goal_reached_threshold = goal_reached_threshold
        self.heading_reached_threshold = heading_reached_threshold
        self.resample_on_reached = resample_on_reached
        self._entity = entity

        self.debug_visualizer = debug_visualizer
        self.debug_envs_idx: list | None = None
        self.visualizer_cfg = (
            debug_visualizer_cfg if debug_visualizer_cfg is not None else {}
        )
        self._debug_nodes: list = []

    """
    Properties
    """

    @property
    def range(self) -> Pose2dCommandRange:
        """The goal pose range dict."""
        return cast(Pose2dCommandRange, self._range)

    @range.setter
    def range(self, range: Pose2dCommandRange, *_args, **_kwargs):
        """Update the goal pose ranges."""
        CommandManager.range.fset(self, range)

    @property
    def resample_time_sec(self) -> float | None:
        """
        The time interval (in seconds) between changing the goal for each environment,
        or None to only change the goal on reset or when it is reached.
        """
        return self._resample_time_sec

    @resample_time_sec.setter
    def resample_time_sec(self, resample_time_sec: float | None):
        """Set the time interval (in seconds) between changing the goal, or None to disable."""
        self._resample_time_sec = resample_time_sec
        self._resample_steps = (
            0 if resample_time_sec is None else int(resample_time_sec / self.env.dt)
        )

    @property
    def entity(self) -> "RigidEntity":
        """The entity that is navigating to the goal."""
        return self._entity if self._entity is not None else self.env.robot

    @property
    def goal_position(self) -> torch.Tensor:
        """The XY position each environment is driving to. Shape is (num_envs, 2)."""
        return self.command[:, :2]

    @property
    def goal_heading(self) -> torch.Tensor:
        """The direction (in radians) to be facing at the goal. Shape is (num_envs,)."""
        return self.command[:, 2]

    @property
    def distance_to_goal(self) -> torch.Tensor:
        """The XY distance from the entity to its goal position. Shape is (num_envs,)."""
        return torch.norm(self.goal_position - self.entity.get_pos()[:, :2], dim=-1)

    @property
    def goal_vec_local(self) -> torch.Tensor:
        """
        The vector from the entity to its goal position, in the entity's own frame:
        how far ahead the goal is, and how far to the left. Shape is (num_envs, 2).
        """
        entity = self.entity
        self._goal_vec_buffer[:, :2] = self.goal_position - entity.get_pos()[:, :2]
        self._goal_vec_buffer[:, 2] = 0.0
        rotated = transform_by_quat(self._goal_vec_buffer, inv_quat(entity.get_quat()))
        return rotated[:, :2]

    @property
    def bearing_error(self) -> torch.Tensor:
        """
        How far the entity has to turn to be pointing straight *at* its goal position, in
        radians: positive to the left, negative to the right. Shape is (num_envs,).

        This is the "which way do I drive" angle, and is not the same as
        `heading_error`, which is the "which way do I face once I get there" angle.
        """
        vec = self.goal_vec_local
        return torch.atan2(vec[:, 1], vec[:, 0])

    @property
    def heading_error(self) -> torch.Tensor:
        """
        How far the entity still has to turn to be facing its goal heading, in radians:
        positive to the left, negative to the right. Shape is (num_envs,).
        """
        current_heading = heading_from_quat(self.entity.get_quat())
        return shortest_turn_to(self.goal_heading, current_heading)

    @property
    def goal_reached(self) -> torch.Tensor:
        """
        Whether each environment's entity has arrived: within `goal_reached_threshold` of
        its goal position, and -- if `heading_reached_threshold` is set -- also facing
        close enough to the goal heading. Shape is (num_envs,).
        """
        reached = self.distance_to_goal < self.goal_reached_threshold
        if self.heading_reached_threshold is not None:
            reached &= self.heading_error.abs() < self.heading_reached_threshold
        return reached

    @property
    def resampled_last_step(self) -> torch.Tensor:
        """
        Boolean tensor marking the environments whose goal was resampled since the previous
        step's rewards were computed. Reward functions that track per-step goal history
        (e.g. progress toward the goal) use this to skip the step that spans a goal change.
        Shape is (num_envs,).
        """
        return self._resampled_last_step

    """
    Lifecycle Operations
    """

    def build(self):
        """Build the pose command manager"""
        super().build()
        self._goal_vec_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self._observation_buffer = torch.zeros(self.env.num_envs, 7, device=gs.device)
        self._find_entities_to_avoid()
        self.build_debug()

    def resample_command(self, env_ids: torch.Tensor):
        """Draw new goal poses for the given environment ids."""
        super().resample_command(env_ids)
        if self._avoided_entities:
            self._redraw_blocked_goals(env_ids)
        self._resampled_last_step[env_ids] = True

    def step(self):
        """Resample goals on the timer (if enabled) and for reached environments."""
        if not self.enabled:
            return

        # Rewards for this step have already been computed, so the resampled marks
        # from the previous step have been consumed
        self._resampled_last_step[:] = False

        # Timer-based resampling (the base implementation), only when enabled
        if self._resample_steps > 0:
            super().step()

        # Resample environments that reached their goal
        if self.resample_on_reached and self._external_controller is None:
            reached_envs_idx = self.goal_reached.nonzero(as_tuple=False).flatten()
            if len(reached_envs_idx) > 0:
                self.resample_command(reached_envs_idx)

        self._render_debug()

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Resample the goals of the reset environments and redraw the debug visualization.
        """
        super().reset(env_ids)
        if not self.enabled or not self.debug_visualizer or not self.debug_envs_idx:
            return
        if env_ids is None or set(self.debug_envs_idx).intersection(env_ids.tolist()):
            self._render_debug(force=True)

    def observation(self, env: GenesisEnv) -> torch.Tensor:
        """
        The goal from the entity's own point of view, in seven numbers:

        | # | Value | Meaning |
        |---|-------|---------|
        | 0, 1 | ahead, left | the goal vector in the entity's frame |
        | 2 | distance | how far away the goal is |
        | 3, 4 | cos, sin of the bearing | which way to drive to reach it |
        | 5, 6 | cos, sin of the heading error | which way to turn to face the goal heading |

        The goal vector on its own would be enough to locate the goal, but it mixes up
        *how far* with *which way*: at 3m away it is a long vector, and a few centimeters
        out it is a tiny one. That makes the steering signal fade away exactly where
        steering has to be most precise. Splitting the distance out from the bearing
        keeps the direction at full strength all the way in.

        Angles are given as cosine/sine pairs rather than raw radians so there is no jump
        where they wrap around.

        Returns:
            torch.Tensor: Shape (num_envs, 7)
        """
        goal_vec = self.goal_vec_local
        distance = torch.norm(goal_vec, dim=-1)
        heading_error = self.heading_error

        # Dividing the goal vector by its own length leaves the bearing as a unit vector,
        # which is the cosine and sine of the angle to turn through to face the goal. The
        # floor keeps an entity sitting exactly on its goal from dividing by zero.
        unit_bearing = goal_vec / distance.clamp(min=1e-6).unsqueeze(-1)

        obs = self._observation_buffer
        obs[:, :2] = goal_vec
        obs[:, 2] = distance
        obs[:, 3:5] = unit_bearing
        obs[:, 5] = torch.cos(heading_error)
        obs[:, 6] = torch.sin(heading_error)
        return obs

    """
    Internal Implementation
    """

    def _find_entities_to_avoid(self):
        """
        Work out what a goal must not be placed on top of, and how much room each one needs.

        Everything in the scene counts -- the obstacles, the robot itself, anything else
        that was added -- except the ground, which every goal sits on by definition.

        The room an entity needs is half the diagonal of its footprint, so the goal lands
        outside it whichever way it is turned, plus the reach threshold, so that arriving
        at the goal doesn't mean driving into it.
        """
        ground = self._ground_entities()
        self._avoided_entities = []
        self._avoided_margins = []

        for entity in self.env.scene.entities:
            if any(entity is ground_entity for ground_entity in ground):
                continue
            self._avoided_entities.append(entity)
            self._avoided_margins.append(
                self._footprint_radius(entity) + self.goal_reached_threshold
            )

    def _ground_entities(self) -> list:
        """
        The terrain entities in the scene. Goals are placed *on* these, so unlike
        everything else they are not kept clear of.
        """
        ground = []
        if getattr(self.env, "terrain", None) is not None:
            ground.append(self.env.terrain)
        for manager in getattr(self.env, "managers", {}).get("terrain", []):
            ground.append(manager.terrain)
        return ground

    def _footprint_radius(self, entity: "RigidEntity") -> float:
        """
        The radius of a circle drawn around the entity's footprint, big enough to cover it
        from every angle: half the diagonal of its bounding box, seen from above.
        """
        aabb = entity.get_AABB()
        # One bounding box per parallel environment; they are all the same size
        if aabb.ndim == 3:
            aabb = aabb[0]
        (x_min, y_min, _), (x_max, y_max, _) = aabb.tolist()
        return math.hypot(x_max - x_min, y_max - y_min) / 2

    def _redraw_blocked_goals(self, env_ids: torch.Tensor):
        """
        Redraw any goal in `env_ids` that landed too close to something in the scene,
        retrying up to `MAX_RESAMPLE_ATTEMPTS` times. Environments still blocked after
        that keep their last draw rather than looping forever.
        """
        remaining = env_ids
        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            remaining = remaining[self._blocked_goals(remaining)]
            if len(remaining) == 0:
                return
            super().resample_command(remaining)

    def _blocked_goals(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Which of `env_ids` drew a goal that is too close to something in the scene.
        Returns a boolean mask lined up with `env_ids`.
        """
        goal_xy = self._command[env_ids, :2]
        blocked = torch.zeros(len(env_ids), dtype=torch.bool, device=gs.device)
        for entity, margin in zip(self._avoided_entities, self._avoided_margins):
            distance = torch.norm(goal_xy - entity.get_pos()[env_ids, :2], dim=-1)
            blocked |= distance < margin
        return blocked

    def build_debug(self):
        """Build the debug visualizer buffers and render throttle"""
        if not self.debug_visualizer:
            return

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
            1.0 / self._vis_cfg("fps") / self.env.dt
        )

    def _vis_cfg(self, key: str):
        """A debug visualizer config value, or its default when not configured"""
        return self.visualizer_cfg.get(key, DEFAULT_VISUALIZER_CONFIG[key])

    def _render_debug(self, force: bool = False):
        """
        Draw an arrow at each debug environment's goal: it starts at the goal position and
        points the way to be facing on arrival.

        Args:
            force: Draw now, even if this step is not a scheduled render for the configured FPS
        """
        if not self.debug_visualizer or not self.debug_envs_idx:
            return

        # Don't update for every step
        if not force and self.env.step_count % self._steps_per_debug_render != 0:
            return

        self._clear_debug_objects()

        arrow_height = self._vis_cfg("arrow_height")
        arrow_length = self._vis_cfg("arrow_length")
        arrow_radius = self._vis_cfg("arrow_radius")
        goal_color = self._vis_cfg("goal_color")
        reached_color = self._vis_cfg("reached_color")
        goal_reached = self.goal_reached

        for i in self.debug_envs_idx:
            pos = (
                self.command[i, 0].item() + self._scene_env_offset[i, 0].item(),
                self.command[i, 1].item() + self._scene_env_offset[i, 1].item(),
                arrow_height + self._scene_env_offset[i, 2].item(),
            )
            heading = self.goal_heading[i].item()
            self._add_debug_object(
                self.env.scene.draw_debug_arrow,
                pos=pos,
                vec=(
                    math.cos(heading) * arrow_length,
                    math.sin(heading) * arrow_length,
                    0.0,
                ),
                radius=arrow_radius,
                color=reached_color if goal_reached[i] else goal_color,
            )

    def _add_debug_object(self, draw_fn: Callable, *args, **kwargs):
        """
        Call one of the scene's `draw_debug_*` functions and keep the node it returns, so
        the object is removed on the next render
        """
        try:
            node = draw_fn(*args, **kwargs)
        except Exception as e:  # noqa
            print(f"Error adding debug visualizing in Pose2dCommand: {e}")
            return
        if node:
            self._debug_nodes.append(node)

    def _clear_debug_objects(self):
        """Remove all debug objects drawn by the previous render"""
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []
