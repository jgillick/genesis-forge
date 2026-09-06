import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import genesis as gs
import numpy as np
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.meshes import arrow_mesh

from .command_manager import CommandManager, CommandRange, CommandRangeValue

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

    from genesis_forge.managers.entity_manager import EntityManager
    from genesis_forge.managers.terrain_manager import TerrainManager


MAX_RESAMPLE_ATTEMPTS = 10
"""
How many times a goal that landed on top of something is resampled before giving up and
keeping the last sample.
"""


class Pose2dCommandRange(TypedDict):
    """The ranges a goal pose is sampled from."""

    x: CommandRangeValue
    y: CommandRangeValue

    heading: NotRequired[CommandRangeValue | None]
    """
    The range the goal heading is sampled from. Leave it out, or set it to None, for a
    position-only goal, which can be arrived at facing any direction.
    """


class Pose2dDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: NotRequired[list[int]]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    fps: NotRequired[int]
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""

    marker_height: NotRequired[float]
    """The height above the ground to draw the goal marker at"""

    arrow_length: NotRequired[float]
    """The length of the goal arrow, when the goal has a heading"""

    arrow_radius: NotRequired[float]
    """The thickness of the goal arrow, when the goal has a heading"""

    ball_radius: NotRequired[float]
    """The radius of the goal ball, when the goal has no heading"""

    goal_color: NotRequired[tuple[float, float, float, float]]
    """The color of the goal marker"""

    reached_color: NotRequired[tuple[float, float, float, float]]
    """The color of the goal marker when the goal has been reached"""


DEFAULT_VISUALIZER_CONFIG = {
    "envs_idx": [],
    "fps": 30,
    "marker_height": 0.05,
    "arrow_length": 0.25,
    "arrow_radius": 0.02,
    "ball_radius": 0.05,
    "goal_color": (0.0, 0.5, 0.0, 1.0),
    "reached_color": (1.0, 0.0, 0.0, 1.0),
}


def _normalize_range(range: "Pose2dCommandRange | CommandRange") -> CommandRange:
    """
    Remove heading from the range dict if it is None.
    Otherwise, the base command manager will attempt to sample from None
    """
    if not isinstance(range, Mapping):
        return range
    ranges = dict(range)
    if ranges.get("heading") is None:
        ranges.pop("heading", None)
    return cast(dict[str, CommandRangeValue], ranges)


def heading_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """
    The compass direction the entity's nose points, seen from above, in radians.

    This follows the nose, so tilting the entity moves it. It is not the yaw of an euler
    decomposition (`quat_to_xyz(quat)[:, 2]`), which tilting leaves where it was.

    Args:
        quat: Orientation quaternions in (w, x, y, z) order, shape (num_envs, 4)

    Returns:
        torch.Tensor: Heading in radians, shape (num_envs,)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # The arctangent of the rotated +X axis: (1 - 2(y^2 + z^2), 2(xy + wz))
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def shortest_turn_to(target_angle: torch.Tensor, current_angle: torch.Tensor) -> torch.Tensor:
    """
    How far `current_angle` has to turn to face `target_angle`, taking the shorter way
    around: turning from 179 degrees to -179 degrees is 2 degrees, not 358.

    Returns:
        torch.Tensor: Radians to turn, positive to the left and negative to the right
    """
    difference = target_angle - current_angle
    return (difference + math.pi) % (2 * math.pi) - math.pi


class Pose2dCommand(CommandManager):
    """
    Generates a goal pose command: a point to drive to, and (optionally) the direction
    to be facing once there. Use this for navigation tasks where the robot should arrive
    somewhere lined up for whatever it does next, like backing into a charging dock or
    pulling up to a shelf facing it.

    The X/Y position and the heading are sampled independently, so the robot has to both
    get there and turn to face the right way. If you only care about the position, leave
    the `heading` range out (or set it to None): the goal is then a point to reach,
    arrived at facing any direction.

    The goal for each environment is resampled when the environment resets, when the
    goal is reached (if `resample_on_reached` is True), and when it has taken longer than
    `resample_time_sec` (if one is set).

    !!! note "Debug Visualization"
        If you set `debug_visualizer` to True, a marker is drawn at each goal: 
        an arrow pointing the way to face on arrival when the goal has a heading, 
        and a ball when it does not.

    Args:
        env: The environment to control
        range: The X/Y and (optional) heading ranges to sample the goal poses from, in the
               environment's local frame
        goal_reached_threshold: The distance (in meters) at which the goal counts as reached.
        heading_reached_threshold: How closely (in radians) the entity must be facing the
                                   goal heading for the goal to count as reached. Ignored
                                   by a goal with no heading.
        resample_on_reached: Sample a new goal for an environment when its goal is reached.
        resample_time_sec: How long an environment may spend on one goal before it is
                           given up on and replaced. The clock restarts with each new goal. 
                           Defaults to None: the goal only changes on reset or when reached.
        entity: The entity that is navigating to the goal. Defaults to `env.robot`.
                This isn't necessary if `entity_manager` is provided.
        entity_manager: The entity manager for the entity that is navigating to the goal.
                        This is more performant than the `entity` parameter: the pose comes
                        from the manager's per-step cache instead of the solver.
        terrain_manager: The terrain manager, so the debug marker sits above the terrain.
                         Defaults to None: the ground is assumed to be flat at z=0.
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
        heading_reached_threshold: float = math.radians(30),
        resample_on_reached: bool = True,
        entity: "RigidEntity | None" = None,
        entity_manager: "EntityManager | None" = None,
        terrain_manager: "TerrainManager | None" = None,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: Pose2dDebugVisualizerConfig | None = None,
    ):
        super().__init__(
            env,
            range=_normalize_range(range),
            resample_time_sec=resample_time_sec,
        )

        # A goal with no heading range is position-only: the command is x and y alone
        self._has_heading = "heading" in self._range

        self.goal_reached_threshold = goal_reached_threshold
        self.heading_reached_threshold = heading_reached_threshold
        self.resample_on_reached = resample_on_reached
        self._entity = entity
        self._entity_manager = entity_manager
        self.terrain_manager = terrain_manager

        self.debug_visualizer = debug_visualizer
        self.debug_envs_idx: list | None = None
        self.visualizer_cfg = (
            debug_visualizer_cfg if debug_visualizer_cfg is not None else {}
        )
        self._debug_nodes: list = []

        # Which environments had their goal replaced since rewards were last computed
        self._resampled_last_step = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

        # How long each environment has been working on its current goal
        self._steps_on_goal = torch.zeros(
            env.num_envs, dtype=torch.long, device=gs.device
        )

        # Filled in by build(), once the scene exists and can be inspected
        self._avoided_entities: list = []
        self._avoided_margins = torch.zeros(0, device=gs.device)

    """
    Properties
    """

    @property
    def range(self) -> dict[str, CommandRangeValue]:
        """The goal pose ranges: `x`, `y`, and (optionally) `heading`."""
        return cast(dict[str, CommandRangeValue], self._range)

    @range.setter
    def range(self, range: "Pose2dCommandRange | CommandRange"):
        """
        Update the goal pose ranges. A heading can be retuned, but not added or dropped:
        whether the goal has one is fixed at construction.
        """
        # `__set__` calls the base setter through the descriptor, which a type checker can
        # follow where the optional `fset` cannot
        CommandManager.range.__set__(self, _normalize_range(range))

    @property
    def goal_position(self) -> torch.Tensor:
        """The XY position each environment is driving to. Shape is (num_envs, 2)."""
        return self.command[:, :2]

    @property
    def goal_heading(self) -> torch.Tensor:
        """The direction (in radians) to be facing at the goal. Shape is (num_envs,)."""
        self._assert_has_heading("goal_heading")
        return self.command[:, 2]

    @property
    def distance_to_goal(self) -> torch.Tensor:
        """The XY distance from the entity to its goal position. Shape is (num_envs,)."""
        return torch.norm(self.goal_position - self._entity_pos[:, :2], dim=-1)

    @property
    def goal_vec_local(self) -> torch.Tensor:
        """
        The vector from the entity to its goal position, in the entity's heading frame:
        how far ahead the goal is, and how far to the left. Shape is (num_envs, 2).

        The frame turns with the entity but does not tilt with it, so leaning or pitching
        neither swings the goal around nor shortens the distance to it.
        """
        entity_pos = self._entity_pos
        goal_position = self.goal_position
        ahead = goal_position[:, 0] - entity_pos[:, 0]
        left = goal_position[:, 1] - entity_pos[:, 1]

        # Turn the world-frame offset by minus the heading
        heading = heading_from_quat(self._entity_quat)
        cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
        return torch.stack(
            (
                cos_heading * ahead + sin_heading * left,
                cos_heading * left - sin_heading * ahead,
            ),
            dim=-1,
        )

    @property
    def bearing_error(self) -> torch.Tensor:
        """
        How far the entity has to turn to be pointing straight *at* its goal position, in
        radians: positive to the left, negative to the right. Shape is (num_envs,).

        Not to be confused with `heading_error`: this is which way to drive, that is
        which way to face once there.
        """
        vec = self.goal_vec_local
        return torch.atan2(vec[:, 1], vec[:, 0])

    @property
    def heading_error(self) -> torch.Tensor:
        """
        How far the entity still has to turn to be facing its goal heading, in radians:
        positive to the left, negative to the right. Shape is (num_envs,).
        """
        self._assert_has_heading("heading_error")
        current_heading = heading_from_quat(self._entity_quat)
        return shortest_turn_to(self.goal_heading, current_heading)

    @property
    def goal_reached(self) -> torch.Tensor:
        """
        Whether each environment's entity has arrived: within `goal_reached_threshold` of
        its goal position, and -- for a goal with a heading -- also facing within
        `heading_reached_threshold` of the goal heading. 
        Shape is (num_envs,).
        """
        reached = self.distance_to_goal < self.goal_reached_threshold
        if self._has_heading:
            reached &= self.heading_error.abs() < self.heading_reached_threshold
        return reached

    @property
    def resampled_last_step(self) -> torch.Tensor:
        """
        Which environments had their goal resampled since the last rewards were computed.
        Reward functions that compare against the previous step (e.g. progress toward the
        goal) use this to skip the step that spans a goal change. Shape is (num_envs,).
        """
        return self._resampled_last_step

    @property
    def _navigating_entity(self) -> "RigidEntity":
        """The entity that is navigating to the goal: the one given, or the env's robot."""
        if self._entity_manager is not None:
            return self._entity_manager.entity
        return self._entity if self._entity is not None else self.env.robot

    @property
    def _entity_pos(self) -> torch.Tensor:
        """
        Where the navigating entity is, shape (num_envs, 3).
        """
        if self._entity_manager is not None:
            return self._entity_manager.base_pos
        return cast(torch.Tensor, self._navigating_entity.get_pos())

    @property
    def _entity_quat(self) -> torch.Tensor:
        """Which way the navigating entity is turned, shape (num_envs, 4)."""
        if self._entity_manager is not None:
            return self._entity_manager.base_quat
        return cast(torch.Tensor, self._navigating_entity.get_quat())

    """
    Lifecycle Operations
    """

    def build(self):
        """Build the pose command manager"""
        super().build()
        num_obs = 7 if self._has_heading else 5
        self._observation_buffer = torch.zeros(
            self.env.num_envs, num_obs, device=gs.device
        )
        self._find_entities_to_avoid()
        self.build_debug()

    def resample_command(self, env_ids: torch.Tensor):
        """Draw new goal poses for the given environment ids."""
        super().resample_command(env_ids)
        if self._avoided_entities:
            self._resample_blocked_goals(env_ids)
        self._resampled_last_step[env_ids] = True
        self._steps_on_goal[env_ids] = 0

    def step(self):
        """Resample the goals of environments that reached theirs, or ran out of time."""
        if not self.enabled:
            return

        # This value is used for rewards, and rewards for this step have already been computed, 
        # so the resampled marks from the previous step have been consumed
        self._resampled_last_step[:] = False

        # An external controller owns the command, so the manager doesn't replace it
        if self._external_controller is None:
            self._steps_on_goal += 1
            envs_idx = self._needs_new_goal().nonzero(as_tuple=False).flatten()
            if len(envs_idx) > 0:
                self.resample_command(envs_idx)

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
        The goal from the entity's own point of view, in five numbers, or seven when the
        goal has a heading:

        | # | Value | Meaning |
        |---|-------|---------|
        | 0, 1 | ahead, left | the goal vector in the entity's heading frame |
        | 2 | distance | how far away the goal is |
        | 3, 4 | cos, sin of the bearing | which way to drive to reach it |
        | 5, 6 | cos, sin of the heading error | which way to turn to face the goal heading |

        Distance and bearing are reported separately because the goal vector alone shrinks
        as the entity closes in, fading the steering signal exactly where it has to be most
        precise. Angles are cosine/sine pairs so there is no jump where they wrap around.

        Returns:
            torch.Tensor: Shape (num_envs, 7), or (num_envs, 5) for a goal with no heading
        """
        goal_vec = self.goal_vec_local
        distance = torch.norm(goal_vec, dim=-1)

        # The unit goal vector is the cosine and sine of the bearing; the floor keeps an
        # entity sitting exactly on its goal from dividing by zero
        unit_bearing = goal_vec / distance.clamp(min=1e-6).unsqueeze(-1)

        obs = self._observation_buffer
        obs[:, :2] = goal_vec
        obs[:, 2] = distance
        obs[:, 3:5] = unit_bearing
        if self._has_heading:
            heading_error = self.heading_error
            obs[:, 5] = torch.cos(heading_error)
            obs[:, 6] = torch.sin(heading_error)
        return obs

    """
    Internal Implementation
    """

    def _needs_new_goal(self) -> torch.Tensor:
        """
        Which environments are due a new goal: they reached the one they had (when
        `resample_on_reached` is set), or they have spent longer than `resample_time_sec`
        on it. Shape is (num_envs,).
        """
        needs_new_goal = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=gs.device
        )
        if self.resample_on_reached:
            needs_new_goal |= self.goal_reached
        if self._resample_steps > 0:
            needs_new_goal |= self._steps_on_goal >= self._resample_steps
        return needs_new_goal

    def _assert_has_heading(self, name: str):
        """Raise a helpful error when a heading value is asked of a position-only goal"""
        if not self._has_heading:
            raise ValueError(
                f"{name} is not available: the goal range has no heading. Add a "
                "'heading' range if the entity should arrive facing a particular direction."
            )

    def _find_entities_to_avoid(self):
        """
        Avoid putting the goal on top of any other entity in the scene.

        The room an entity needs, between it and any other entity, is half 
        the diagonal of its footprint, plus the reach threshold, so that arriving
        at the goal doesn't mean driving into it.
        """
        ground = self._ground_entities()
        self._avoided_entities = []
        margins = []

        for entity in cast(list["RigidEntity"], self.env.scene.entities):
            if any(entity is ground_entity for ground_entity in ground):
                continue

            # Get the entity footprint radius
            aabb = entity.get_AABB()
            if aabb.ndim == 3:
                aabb = aabb[0]
            (x_min, y_min, _), (x_max, y_max, _) = aabb.tolist()
            footprint_radius = math.hypot(x_max - x_min, y_max - y_min) / 2

            self._avoided_entities.append(entity)
            margins.append(footprint_radius + self.goal_reached_threshold)

        self._avoided_margins = torch.tensor(
            margins, device=gs.device, dtype=gs.tc_float
        ) # shape (num_entities, 1)

    def _ground_entities(self) -> list:
        """
        The entities that make up the ground in the scene.
        """
        ground = []
        if getattr(self.env, "terrain", None) is not None:
            ground.append(self.env.terrain)
        for manager in getattr(self.env, "managers", {}).get("terrain", []):
            ground.append(manager.terrain)
        for entity in self.env.scene.entities:
            morph = getattr(entity, "main_morph", None)
            if morph is None:
                morph = getattr(entity, "morph", None)
            if morph and isinstance(morph, (gs.morphs.Plane, gs.morphs.Terrain)):
                ground.append(entity)
        return ground

    def _resample_blocked_goals(self, env_ids: torch.Tensor):
        """
        Resample any goal in `env_ids` that landed too close to something in the scene,
        retrying up to `MAX_RESAMPLE_ATTEMPTS` times. Environments still blocked after
        that keep their last sample rather than looping forever.
        """

        # Put the X/Y position of all the entities to avoid in a stack
        avoid_xy = torch.stack([ 
            cast(torch.Tensor, entity.get_pos())[:, :2] 
            for entity in self._avoided_entities
        ])

        remaining = env_ids
        for _ in range(MAX_RESAMPLE_ATTEMPTS):
            remaining = remaining[self._blocked_goals(remaining, avoid_xy)]
            if len(remaining) == 0:
                return
            super().resample_command(remaining)

    def _blocked_goals(
        self, env_ids: torch.Tensor, avoid_xy: torch.Tensor
    ) -> torch.Tensor:
        """
        Which of `env_ids` sampled a goal that is too close to something in the scene.
        Returns a boolean mask lined up with `env_ids`.

        Args:
            env_ids: The environments to check
            avoided_xy: The XY position of everything to keep clear of, one row per
                        entity, shape (entities, num_envs, 2)
        """
        goal_xy = self._command[env_ids, :2]
        distance = torch.norm(avoid_xy[:, env_ids] - goal_xy, dim=-1)

        # Distances are (entities, envs); the margins are turned on their side to match,
        # so each entity's margin is compared against every environment
        return (distance < self._avoided_margins.unsqueeze(1)).any(dim=0)

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

        self._debug_envs_tensor = torch.tensor(
            self.debug_envs_idx or [], dtype=torch.long, device=gs.device
        )

        # Calculate the number of steps per debug render
        self._steps_per_debug_render = math.ceil(
            1.0 / self._vis_cfg("fps") / self.env.dt
        )

    def _vis_cfg(self, key: str):
        """Get a debug visualizer config value, or its default when not configured"""
        return self.visualizer_cfg.get(key, DEFAULT_VISUALIZER_CONFIG[key])

    def _render_debug(self, force: bool = False):
        """
        Draw a marker at each debug environment's goal: an arrow pointing the way to be
        facing on arrival, or a ball when the goal has no heading.

        Args:
            force: Draw now, even if this step is not a scheduled render for the configured FPS
        """
        if not self.debug_visualizer or not self.debug_envs_idx:
            return

        # Don't update for every step
        if not force and self.env.step_count % self._steps_per_debug_render != 0:
            return

        self._clear_debug_objects()

        goal_color = self._vis_cfg("goal_color")
        reached_color = self._vis_cfg("reached_color")
        envs_idx = self._debug_envs_tensor
        positions = self._debug_marker_positions()
        reached = self.goal_reached[envs_idx].cpu().numpy()
        headings = (
            self.goal_heading[envs_idx].cpu().numpy() if self._has_heading else None
        )

        for n in range(len(positions)):
            color = reached_color if reached[n] else goal_color
            try:
                if headings is not None:
                    node = self._draw_goal_arrow(positions[n], headings[n], color)
                else:
                    node = self.env.scene.draw_debug_sphere(
                        pos=positions[n],
                        radius=self._vis_cfg("ball_radius"),
                        color=color,
                    )
                self._debug_nodes.append(node)
            except Exception as e:  # noqa
                print(f"Error drawing debug visuals in Pose2dCommand: {e}")

    def _debug_marker_positions(self) -> np.ndarray:
        """
        The world position of the goal marker for each debug environment.
        shape (len(debug_envs_idx), 3)
        """
        envs_idx = self._debug_envs_tensor
        goal_xy = self.command[envs_idx, :2]

        height = torch.full(
            (len(envs_idx),), self._vis_cfg("marker_height"), device=gs.device
        )
        if self.terrain_manager is not None:
            height += self.terrain_manager.get_terrain_height(
                goal_xy[:, 0], goal_xy[:, 1]
            )

        pos = torch.empty(len(envs_idx), 3, device=gs.device)
        pos[:, :2] = goal_xy
        pos[:, 2] = height
        pos += self._scene_env_offset[envs_idx]
        return pos.cpu().numpy()

    def _draw_goal_arrow(
        self, pos: np.ndarray, heading: float, color: tuple[float, float, float, float]
    ):
        """Draw the arrow that shows the goal position and the way to face on arrival"""
        length = self._vis_cfg("arrow_length")
        vector = (math.cos(heading) * length, math.sin(heading) * length, 0.0)
        mesh = arrow_mesh(
            pos,
            vector,
            self._vis_cfg("arrow_radius"),
            color=color,
        )
        return self.env.scene.draw_debug_mesh(mesh)

    def _clear_debug_objects(self):
        """Remove all debug objects drawn by the previous render"""
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []
