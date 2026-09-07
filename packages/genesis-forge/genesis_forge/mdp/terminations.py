"""
Termination functions for the Genesis environment.
Each of these should return a boolean tensor indicating which environments should terminate, in the tensor shape (num_envs,).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    MdpFn,
    TerrainManager,
)
from genesis_forge.utils import entity_projected_gravity

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


@dataclass(kw_only=True, eq=False)
class timeout(MdpFn):
    """
    Terminate the environment if the episode length exceeds the maximum episode length.
    """

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if env.max_episode_length is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
        return env.episode_length > env.max_episode_length


@dataclass(kw_only=True, eq=False)
class bad_orientation(MdpFn):
    """
    Terminate the environment if the robot is tipping over too much.

    This function uses projected gravity to detect when the robot has tilted
    beyond a safe threshold. When the robot is perfectly upright, projected
    gravity should be [0, 0, -1] in the body frame. As the robot tilts,
    the x,y components increase, indicating roll and pitch angles.

    Args:
        limit_angle: Maximum allowed tilt angle in degrees (default: 40 degrees)
        entity_manager: The entity manager for the entity.
        entity: The entity to check. Defaults to `env.robot`.
                        This isn't necessary if `entity_manager` is provided.
        grace_steps: Number of steps at episode start to ignore tilt detection (default: 0)
                     This gives the robot a chance to stabilize before tilt detection is active.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """

    limit_angle: float = 40.0
    entity: RigidEntity = None
    entity_manager: EntityManager = None
    grace_steps: int = 0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        in_grace_period = env.episode_length <= self.grace_steps

        # Get the projected gravity vector in body frame
        if self.entity_manager is not None:
            projected_gravity = self.entity_manager.get_projected_gravity()
        else:
            entity = self.entity if self.entity is not None else env.robot
            projected_gravity = entity_projected_gravity(entity)

        # Calculate the magnitude of tilt (distance from perfectly upright)
        projected_gravity_xy = projected_gravity[:, :2]
        tilt_magnitude = torch.norm(projected_gravity_xy, dim=1)

        # Convert tilt magnitude to angle
        tilt_angle = torch.asin(torch.clamp(tilt_magnitude, max=0.99))

        # Terminate if tilt angle exceeds the limit
        return (~in_grace_period) & (tilt_angle > math.radians(self.limit_angle))


@dataclass(kw_only=True, eq=False)
class is_upsidedown(MdpFn):
    """
    Terminate when the robot is belly-up (inverted).

    Uses projected gravity in the body frame: upright is approximately [0, 0, -1],
    belly-up is approximately [0, 0, +1]. Side-lying poses keep z below threshold.

    Args:
        threshold: Terminate when projected_gravity[:, 2] exceeds this value
        entity_manager: The entity manager for the robot
        entity: The entity to check. Defaults to `env.robot`. Not necessary if entity_manager is provided
        grace_steps: Steps at episode start to ignore this check
    """

    threshold: float = 0.5
    entity: RigidEntity = None
    entity_manager: EntityManager = None
    grace_steps: int = 0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        in_grace_period = env.episode_length <= self.grace_steps

        if self.entity_manager is not None:
            projected_gravity = self.entity_manager.get_projected_gravity()
        else:
            entity = self.entity if self.entity is not None else env.robot
            projected_gravity = entity_projected_gravity(entity)

        return (~in_grace_period) & (projected_gravity[:, 2] > self.threshold)


@dataclass(kw_only=True, eq=False)
class base_height_below_minimum(MdpFn):
    """
    Terminate the environment if the robot's base height falls below a minimum threshold.

    Args:
        minimum_height: Minimum allowed base height in meters
        entity_manager: The entity manager for the entity.
        entity: The entity to check. Defaults to `env.robot`.
                        This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """

    minimum_height: float = 0.05
    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            base_pos = self.entity_manager.base_pos
        else:
            entity = self.entity if self.entity is not None else env.robot
            base_pos = entity.get_pos()
        return base_pos[:, 2] < self.minimum_height


@dataclass(kw_only=True, eq=False)
class out_of_bounds(MdpFn):
    """
    Terminate if the entity's base position is outside of the terrain.

    Args:
        terrain_manager: The terrain manager to check for out of bounds
        subterrain: The subterrain to keep the robot inside of
        border_margin: The margin (in meters) to add to the terrain bounds
                       This terminates the episode before the robot falls off the terrain.
        entity: The entity to check. Defaults to `env.robot`.
    """

    terrain_manager: TerrainManager
    subterrain: str | None = None
    border_margin: float = 0.5
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        # Get the entity's base position
        entity = self.entity if self.entity is not None else env.robot
        position = entity.get_pos()

        # Get terrain bounds
        (x_min, x_max, y_min, y_max) = self.terrain_manager.get_bounds(self.subterrain)
        x_min_bound, x_max_bound = x_min + self.border_margin, x_max - self.border_margin
        y_min_bound, y_max_bound = y_min + self.border_margin, y_max - self.border_margin

        # Check bounds
        x_pos, y_pos = position[:, 0], position[:, 1]
        return (
            (x_pos < x_min_bound)
            | (x_pos > x_max_bound)
            | (y_pos < y_min_bound)
            | (y_pos > y_max_bound)
        )


@dataclass(kw_only=True, eq=False)
class has_contact(MdpFn):
    """
    One or more links in the contact manager are in contact with something.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The force threshold, per contact, for contact detection (default: 1.0)
        min_contacts: The minimum number of contacts required to terminate (default: 1)

    Returns:
        True for each environment that has contact
    """

    contact_manager: ContactManager
    threshold: float = 1.0
    min_contacts: int = 1

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        in_contact = self.contact_manager.contacts[:, :].norm(dim=-1) > self.threshold
        return in_contact.sum(dim=1) >= self.min_contacts


@dataclass(kw_only=True, eq=False)
class contact_force(MdpFn):
    """
    Terminate if any link in the contact manager is in contact with something with a force greater than the threshold.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0 N)

    Returns:
        The total force for the contact manager for each environment
    """

    contact_manager: ContactManager
    threshold: float = 1.0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        return torch.any(
            torch.norm(self.contact_manager.contacts, dim=-1) > self.threshold, dim=-1
        )


@dataclass(kw_only=True, eq=False)
class contact_force_with_grace_period(MdpFn):
    """
    Terminate if contact force exceeds threshold, with a grace period at episode start.

    This is useful for quadrupeds that may start in slightly unstable positions
    and need a few steps to stabilize before fall detection becomes active.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 100.0 N)
        grace_steps: Number of steps at episode start to ignore contacts (default: 10)

    Returns:
        Boolean tensor indicating which environments should terminate
    """

    contact_manager: ContactManager
    threshold: float = 100.0
    grace_steps: int = 10

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        # Don't terminate during grace period (early in episode)
        in_grace_period = env.episode_length <= self.grace_steps

        # Check contact forces
        contact_exceeded = torch.any(
            torch.norm(self.contact_manager.contacts, dim=-1) > self.threshold, dim=-1
        )

        # Only terminate if past grace period AND contact exceeded
        return (~in_grace_period) & contact_exceeded.detach()


@dataclass(kw_only=True, eq=False)
class dof_control_force_limit(MdpFn):
    """
    Terminate if any joint's commanded actuator force exceeds a limit (+/-).

    Uses control/output force (what the actuator commands), not measured joint load.
    Suitable for teaching policies to stay within rated motor torque.

    Args:
        actuator_manager: Actuator manager for the controlled joints
        threshold: Force/torque limit (in simulator units).
                   If None, uses `max_force` value from actuator_manager.

    Returns:
        Boolean tensor indicating which environments should terminate
    """

    actuator_manager: ActuatorManager
    threshold: float | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        force = self.actuator_manager.get_dofs_control_force()
        threshold = self.threshold
        if threshold is None:
            threshold = self.actuator_manager.get_dofs_max_force()
        return torch.any(torch.abs(force) > threshold, dim=-1)


@dataclass(kw_only=True, eq=False)
class dof_velocity_limit(MdpFn):
    """
    Terminate if any of the actuator_manager's joints moves faster than a speed limit.

    Args:
        actuator_manager: Actuator manager for the controlled joints
        threshold: Speed limit in the units given by `unit`
        unit: The speed units
              - `"rad"` for radians per second (default)
              - `"rpm"` for revolutions per minute

    Returns:
        Boolean tensor indicating which environments should terminate
    """

    actuator_manager: ActuatorManager
    threshold: float
    unit: Literal["rpm", "rad"] = "rad"

    def build(self):
        assert self.unit in (
            "rad",
            "rpm",
        ), f"Unknown velocity unit '{self.unit}'. Use 'rad' or 'rpm'."

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        threshold = self.threshold
        if self.unit == "rpm":
            threshold = threshold * (2 * math.pi / 60)
        vel = self.actuator_manager.get_dofs_velocity()
        return torch.any(torch.abs(vel) > threshold, dim=-1)
