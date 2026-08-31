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


class PositionCommandRange(TypedDict):
    x: CommandRangeValue
    y: CommandRangeValue


class PositionDebugVisualizerConfig(TypedDict):
    """Defines the configuration for the debug visualizer."""

    envs_idx: NotRequired[list[int]]
    """The indices of the environments to visualize. If None, all environments will be visualized."""

    fps: NotRequired[int]
    """The FPS of the debug visualization. Lower FPS means fewer frames are rendered, saving GPU memory."""

    marker_radius: NotRequired[float]
    """The radius of the goal marker sphere"""

    marker_height: NotRequired[float]
    """The height above the ground to draw the goal marker at"""

    goal_color: NotRequired[tuple[float, float, float, float]]
    """The color of the goal marker"""

    reached_color: NotRequired[tuple[float, float, float, float]]
    """The color of the goal marker when the goal has been reached"""


DEFAULT_VISUALIZER_CONFIG = {
    "envs_idx": [],
    "fps": 30,
    "marker_radius": 0.05,
    "marker_height": 0.15,
    "goal_color": (0.0, 0.5, 0.0, 1.0),
    "reached_color": (1.0, 0.0, 0.0, 1.0),
}


class PositionCommandManager(CommandManager):
    """
    Generates a goal position command, sampled uniformly from an X/Y range in the
    environment's local frame. Use this for navigation tasks where the robot should
    drive to a target point.

    The goal for each environment is resampled when the environment resets, when the
    goal is reached (if `resample_on_reached` is True), and on a timer
    (if `resample_time_sec` is set).

    !!! note "Debug Visualization"
        If you set `debug_visualizer` to True, a sphere is rendered at each goal
        position, changing color when the goal has been reached.

    Args:
        env: The environment to control
        range: The X/Y ranges to sample goal positions from, in the environment's local frame
        resample_time_sec: The time interval between changing the goal.
                           Defaults to None: the goal only changes on reset or when reached.
        goal_reached_threshold: The distance (in meters) at which the goal counts as reached.
        resample_on_reached: Sample a new goal for an environment when its goal is reached.
        entity: The entity that is navigating to the goal. Defaults to `env.robot`.
        avoid_entities: Entities whose current XY position a sampled goal must stay at least
                        `avoid_margin` away from (e.g. scattered obstacles). When a sampled goal
                        conflicts, it is redrawn, up to `avoid_max_attempts` times. Defaults to
                        None: goals are sampled from `range` with no avoidance.
        avoid_margin: The minimum XY distance a goal must keep from every `avoid_entities` position.
                      Has no effect unless `avoid_entities` is set.
        avoid_max_attempts: How many times to redraw a conflicting goal before giving up and
                            keeping the last-drawn (possibly still conflicting) position.
        debug_visualizer: Enable the debug visualization
        debug_visualizer_cfg: The configuration for the debug visualizer

    Example::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.position_command = PositionCommandManager(
                    self,
                    range={
                        "x": (-2.0, 2.0),
                        "y": (-2.0, 2.0),
                    },
                    debug_visualizer=True,
                )

                RewardManager(
                    self,
                    cfg={
                        "position_tracking": {
                            "weight": 1.0,
                            "fn": rewards.position_tracking(
                                position_cmd_manager=self.position_command,
                            ),
                        },
                        "reached_goal": {
                            "weight": 10.0,
                            "fn": rewards.reached_goal(
                                position_cmd_manager=self.position_command,
                            ),
                        },
                        # ... other rewards ...
                    },
                )

                # Observations
                ObservationManager(
                    self,
                    cfg={
                        # The goal position, as a vector in the robot's local frame
                        "goal_vec": {"fn": self.position_command.observation},
                        # ... other observations ...
                    },
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        range: PositionCommandRange,
        resample_time_sec: float | None = None,
        goal_reached_threshold: float = 0.15,
        resample_on_reached: bool = True,
        entity: "RigidEntity | None" = None,
        avoid_entities: "list[RigidEntity] | None" = None,
        avoid_margin: float = 0.0,
        avoid_max_attempts: int = 10,
        debug_visualizer: bool = False,
        debug_visualizer_cfg: PositionDebugVisualizerConfig | None = None,
    ):
        # The resampled mask is written by resample_command, which the base constructor
        # can indirectly trigger, so it must exist first
        self._resampled_last_step = torch.zeros(
            env.num_envs, dtype=torch.bool, device=gs.device
        )

        super().__init__(
            env,
            range=range,
            resample_time_sec=resample_time_sec,
        )

        self.goal_reached_threshold = goal_reached_threshold
        self.resample_on_reached = resample_on_reached
        self._entity = entity

        self.avoid_entities = avoid_entities
        self.avoid_margin = avoid_margin
        self.avoid_max_attempts = avoid_max_attempts

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
    def range(self) -> PositionCommandRange:
        """The goal position range dict."""
        return cast(PositionCommandRange, self._range)

    @range.setter
    def range(self, range: PositionCommandRange, *_args, **_kwargs):
        """Update the goal position ranges."""
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
    def distance_to_goal(self) -> torch.Tensor:
        """The XY distance from the entity to its goal position. Shape is (num_envs,)."""
        return torch.norm(self.command - self.entity.get_pos()[:, :2], dim=-1)

    @property
    def goal_reached(self) -> torch.Tensor:
        """Whether each environment's entity is within `goal_reached_threshold` of its goal. Shape is (num_envs,)."""
        return self.distance_to_goal < self.goal_reached_threshold

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
        """Build the position command manager"""
        super().build()
        self._obs_vec_buffer = torch.zeros(self.env.num_envs, 3, device=gs.device)
        self.build_debug()

    def resample_command(self, env_ids: torch.Tensor):
        """Sample new goal positions for the given environment ids."""
        super().resample_command(env_ids)
        if self.avoid_entities:
            self._resample_away_from_avoid_entities(env_ids)
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
        The vector from the entity to its goal position, in the entity's local frame.
        This encodes both the direction and the distance to the goal.

        Returns:
            torch.Tensor: Shape (num_envs, 2)
        """
        entity = self.entity
        self._obs_vec_buffer[:, :2] = self.command - entity.get_pos()[:, :2]
        self._obs_vec_buffer[:, 2] = 0.0
        vec_local = transform_by_quat(
            self._obs_vec_buffer, inv_quat(entity.get_quat())
        )
        return vec_local[:, :2]

    """
    Internal Implementation
    """

    def _resample_away_from_avoid_entities(self, env_ids: torch.Tensor):
        """
        Redraw any goal in `env_ids` that landed within `avoid_margin` of an
        `avoid_entities` position, retrying up to `avoid_max_attempts` times.
        Environments still in conflict after that keep their last-drawn position.
        """
        remaining = env_ids
        for _ in range(self.avoid_max_attempts):
            remaining = remaining[self._conflicts_with_avoid_entities(remaining)]
            if len(remaining) == 0:
                return
            super().resample_command(remaining)

    def _conflicts_with_avoid_entities(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Boolean mask (aligned with `env_ids`) of goals closer than `avoid_margin` to an avoided entity."""
        goal_xy = self._command[env_ids, :2]
        conflict = torch.zeros(len(env_ids), dtype=torch.bool, device=gs.device)
        for entity in self.avoid_entities:
            dist = torch.norm(goal_xy - entity.get_pos()[env_ids, :2], dim=-1)
            conflict |= dist < self.avoid_margin
        return conflict

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
        Draw a marker sphere at each debug environment's goal position.

        Args:
            force: Draw now, even if this step is not a scheduled render for the configured FPS
        """
        if not self.debug_visualizer or not self.debug_envs_idx:
            return

        # Don't update for every step
        if not force and self.env.step_count % self._steps_per_debug_render != 0:
            return

        self._clear_debug_objects()

        marker_height = self._vis_cfg("marker_height")
        marker_radius = self._vis_cfg("marker_radius")
        goal_color = self._vis_cfg("goal_color")
        reached_color = self._vis_cfg("reached_color")
        goal_reached = self.goal_reached

        for i in self.debug_envs_idx:
            pos = (
                self.command[i, 0].item() + self._scene_env_offset[i, 0].item(),
                self.command[i, 1].item() + self._scene_env_offset[i, 1].item(),
                marker_height + self._scene_env_offset[i, 2].item(),
            )
            self._add_debug_object(
                self.env.scene.draw_debug_sphere,
                pos=pos,
                radius=marker_radius,
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
            print(f"Error adding debug visualizing in PositionCommandManager: {e}")
            return
        if node:
            self._debug_nodes.append(node)

    def _clear_debug_objects(self):
        """Remove all debug objects drawn by the previous render"""
        for node in self._debug_nodes:
            self.env.scene.clear_debug_object(node)
        self._debug_nodes = []
