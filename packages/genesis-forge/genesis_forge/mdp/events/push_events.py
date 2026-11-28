from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, cast

import genesis as gs
import torch
from genesis.utils.geom import (
    inv_quat,
    transform_by_quat,
)

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import DurationEventMdpFn, EntityManager, EventMdpFn
from genesis_forge.meshes import arrow_meshes
from genesis_forge.utils import apply_external_wrench, links_by_name_pattern


class PushAxisRanges(TypedDict):
    """The (min, max) range to push along, or around, each axis. Axes that are left out aren't pushed."""

    x: NotRequired[tuple[float, float]]
    """The range for the x-axis"""

    y: NotRequired[tuple[float, float]]
    """The range for the y-axis"""

    z: NotRequired[tuple[float, float]]
    """The range for the z-axis"""


class PushDebugStyle(TypedDict):
    """Defines how a push is drawn by the debug visualizer."""

    arrow_offset: NotRequired[float]
    """The vertical offset of the arrow from the link being pushed"""

    arrow_radius: NotRequired[float]
    """The radius of the shaft of the arrow"""

    arrow_max_length: NotRequired[float]
    """The length of the arrow for the strongest possible push"""

    color: NotRequired[tuple[float, float, float, float]]
    """The color of the arrow"""


DEFAULT_PUSH_DEBUG_STYLE: PushDebugStyle = {
    "arrow_offset": 0.0,
    "arrow_radius": 0.01,
    "arrow_max_length": 0.5,
    "color": (1.0, 0.0, 0.0, 1.0),
}


@dataclass(kw_only=True, eq=False)
class PushEvent(EventMdpFn):
    """
    Base class for events that push an entity, and draws the push in the debug visualizer.

    For a duration push, also extend :class:`DurationEventMdpFn`::

        class my_push(PushEvent, DurationEventMdpFn):
            ...

    Args:
        entity_manager: The manager of the entity to push.
        debug_style: How the push is drawn by the :class:`EventManager` debug visualizer.
    """

    entity_manager: EntityManager
    debug_style: PushDebugStyle | None = None

    def build(self):
        # Keep the debug vector after a push ends, since it can still be drawn for a moment
        if not hasattr(self, "_debug_vec"):
            self._debug_vec = torch.zeros(
                (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
            )

    def _push_position(self, envs_idx: torch.Tensor) -> torch.Tensor:
        """Where the entity is pushed, in the environment's local frame. Shape is (len(envs_idx), 3)."""
        pos = self.entity_manager.entity.get_pos(envs_idx=envs_idx)
        return cast(torch.Tensor, pos)

    def _record_push(self, envs_idx: torch.Tensor, push: torch.Tensor, max_push: float):
        """
        Save the push for the debug visualizer.

        Args:
            envs_idx: The environment ids that were pushed.
            push: The vector of each push. Shape is (len(envs_idx), 3).
            max_push: The size of the strongest push that the event's params allow. A push of
                this size is drawn with the longest arrow.
        """
        self._debug_vec[envs_idx] = push / max(max_push, 1e-6)

    def debug_draw(self, envs_idx: torch.Tensor) -> list[Any]:
        """Draw an arrow in the direction of the push, with a length that shows its strength."""
        style: dict[str, Any] = {
            **DEFAULT_PUSH_DEBUG_STYLE,
            **(self.debug_style or {}),
        }
        scene = self.env.scene
        offset = scene.envs_offset[envs_idx.cpu().numpy()]
        position = self._push_position(envs_idx).cpu().numpy()
        meshes = arrow_meshes(
            pos=position + offset + (0.0, 0.0, style["arrow_offset"]),
            vec=self._debug_vec[envs_idx].cpu().numpy() * style["arrow_max_length"],
            radius=style["arrow_radius"],
            color=style["color"],
        )
        return [scene.draw_debug_mesh(mesh) for mesh in meshes]

    @staticmethod
    def _require_min_max(name: str, value: tuple[float, float] | None) -> None:
        """
        Raise an error if a param isn't a valid (min, max) pair. Called by ``build``, so the
        error is raised when the param is set, rather than when the event first fires.
        """
        if value is None:
            return
        try:
            low, high = value
            valid = math.isfinite(low) and math.isfinite(high) and low <= high
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"{name} must be a finite (min, max) pair with min <= max, got {value!r}"
            )


@dataclass(kw_only=True, eq=False)
class push_velocity(PushEvent):
    """
    Push the entity by instantly changing the velocity of its base.

    A random value is added to the current velocity of each axis that has a range.
    The axes are in the world frame. The entity's base cannot be fixed.

    Example::

        events.push_velocity(
            entity_manager=self.robot_manager,
            linear={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            angular={"z": (-0.3, 0.3)},
        )

    Args:
        entity_manager: The manager of the entity to push.
        linear: The (min, max) change in linear velocity along each axis, in m/s.
        angular: The (min, max) change in angular velocity around each axis, in rad/s.
        debug_style: How the push is drawn by the :class:`EventManager` debug visualizer.
    """

    linear: PushAxisRanges | None = None
    angular: PushAxisRanges | None = None

    def build(self):
        super().build()

        base_joint = self.entity_manager.entity.base_joint
        if base_joint.n_dofs != 6:
            raise ValueError(
                "push_velocity requires an entity with a free base. Use push_force instead."
            )
        self._dofs_idx = list(base_joint.dofs_idx_local)

        # In the order of the base's dofs
        self._ranges: list[tuple[float, float] | None] = []
        for name, axes in (("linear", self.linear), ("angular", self.angular)):
            axes = axes or {}
            for axis in ("x", "y", "z"):
                min_max = axes.get(axis)
                self._require_min_max(f"{name}['{axis}']", min_max)
                self._ranges.append(min_max)
        if all(min_max is None for min_max in self._ranges):
            raise ValueError("push_velocity needs a range for at least one axis")

        self._max_linear = math.hypot(
            *(max(abs(value) for value in axis) for axis in self._ranges[:3] if axis)
        )

    def __call__(self, env: GenesisEnv, envs_idx: torch.Tensor) -> None:
        entity = self.entity_manager.entity

        delta = torch.zeros((len(envs_idx), 6), device=gs.device, dtype=gs.tc_float)
        for axis, value_range in enumerate(self._ranges):
            if value_range is not None:
                delta[:, axis].uniform_(*value_range)

        self._record_push(envs_idx, delta[:, :3], self._max_linear)

        # A free joint's linear velocity is in the world frame, but its angular velocity
        # is in the base link's frame, as used by the solver (including any morph offset)
        base_quat = entity.get_quat(envs_idx=envs_idx, relative=False)
        ang_vel = transform_by_quat(delta[:, 3:], inv_quat(base_quat))
        delta[:, 3:] = cast(torch.Tensor, ang_vel)

        velocity = entity.get_dofs_velocity(self._dofs_idx, envs_idx=envs_idx)
        entity.set_dofs_velocity(velocity + delta, self._dofs_idx, envs_idx=envs_idx)


@dataclass(kw_only=True, eq=False)
class push_force(PushEvent, DurationEventMdpFn):
    """
    Push one of the entity's links with a force, in a random horizontal direction.

    The direction can be any angle in the world's x/y plane. The force never points up or down.

    The size and direction of the push are chosen when the event fires, and stay the same
    (in the world frame) until it ends. The force is applied to the link's center of mass.

    The approximate change in velocity is: ``force * duration / mass``

    Args:
        entity_manager: The manager of the entity to push.
        force: The (min, max) size of the force, in newtons.
        torque: The (min, max) size of an additional torque around a random axis, in newton-meters.
        link_name: The name, or regex pattern, of the link to push. Defaults to the base link.
        debug_style: How the push is drawn by the :class:`EventManager` debug visualizer.
    """

    force: tuple[float, float]
    torque: tuple[float, float] | None = None
    link_name: str | None = None

    def build(self):
        super().build()

        if self.force is None:
            raise ValueError("force is required")
        self._require_min_max("force", self.force)
        self._require_min_max("torque", self.torque)

        entity = self.entity_manager.entity
        link = entity.base_link
        if self.link_name is not None:
            links = links_by_name_pattern(entity, self.link_name)
            if len(links) != 1:
                raise ValueError(
                    f"Expected one link with name/pattern '{self.link_name}', "
                    f"found: {[link.name for link in links]}"
                )
            link = links[0]
        self._links_idx_local = [link.idx_local]

        # Keep the force and torque across rebuilds, so a param change doesn't interrupt a push
        if not hasattr(self, "_force_buf"):
            self._force_buf = torch.zeros(
                (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
            )
            self._torque_buf = torch.zeros_like(self._force_buf)
        # Without a torque range, none is applied, so the observation must not report one
        if self.torque is None:
            self._torque_buf.zero_()

    @property
    def current_force(self) -> torch.Tensor:
        """
        The force being applied, in the world frame. Zero when there is no push. Shape is (num_envs, 3).
        This can be used as a privileged observation for the critic.
        """
        return self._force_buf

    @property
    def current_torque(self) -> torch.Tensor:
        """The torque being applied, in the world frame. Zero when there is no push. Shape is (num_envs, 3)."""
        return self._torque_buf

    def start(self, envs_idx: torch.Tensor) -> None:
        n_envs = len(envs_idx)

        angle = torch.empty(n_envs, device=gs.device).uniform_(0, 2 * math.pi)
        size = torch.empty(n_envs, device=gs.device).uniform_(*self.force)
        self._force_buf[envs_idx, 0] = size * torch.cos(angle)
        self._force_buf[envs_idx, 1] = size * torch.sin(angle)
        max_push = max(abs(value) for value in self.force)
        self._record_push(envs_idx, self._force_buf[envs_idx], max_push)

        if self.torque is not None:
            axis = torch.randn((n_envs, 3), device=gs.device)
            axis = torch.nn.functional.normalize(axis, dim=-1)
            size = torch.empty((n_envs, 1), device=gs.device).uniform_(*self.torque)
            self._torque_buf[envs_idx] = size * axis

    def step(self, envs_idx: torch.Tensor) -> None:
        """
        Genesis clears external forces after each step, so we need to reapply them each step
        """
        torque = None
        force = self._force_buf[envs_idx].unsqueeze(1)
        if self.torque is not None:
            torque = self._torque_buf[envs_idx].unsqueeze(1)
        apply_external_wrench(
            self.entity_manager.entity,
            links_idx_local=self._links_idx_local,
            envs_idx=envs_idx,
            force=force,
            torque=torque,
        )

    def end(self, envs_idx: torch.Tensor) -> None:
        """
        Clear the applied forces and torques for the specified environments.
        """
        self._force_buf[envs_idx] = 0.0
        self._torque_buf[envs_idx] = 0.0

    def _push_position(self, envs_idx: torch.Tensor) -> torch.Tensor:
        entity = self.entity_manager.entity
        pos = entity.get_links_pos(self._links_idx_local, envs_idx=envs_idx)[:, 0]
        return cast(torch.Tensor, pos)
