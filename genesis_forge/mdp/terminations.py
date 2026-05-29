from __future__ import annotations
"""
Termination functions for the Genesis environment.
Each of these should return a boolean tensor indicating which environments should terminate, in the tensor shape (num_envs,).
"""

import math
from typing import Literal

import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import entity_projected_gravity
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    TerrainManager,
)


def timeout(env: GenesisEnv) -> torch.Tensor:
    """
    Terminate the environment if the episode length exceeds the maximum episode length.
    """
    if env.max_episode_length is None:
        return torch.zeros(env.num_envs, dtype=torch.bool)
    return env.episode_length > env.max_episode_length


def bad_orientation(
    env: GenesisEnv,
    limit_angle: float = 40.0,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
    grace_steps: int = 0,
) -> torch.Tensor:
    """
    Terminate the environment if the robot is tipping over too much.

    This function uses projected gravity to detect when the robot has tilted
    beyond a safe threshold. When the robot is perfectly upright, projected
    gravity should be [0, 0, -1] in the body frame. As the robot tilts,
    the x,y components increase, indicating roll and pitch angles.

    Args:
        env: The Genesis environment containing the robot
        limit_angle: Maximum allowed tilt angle in degrees (default: 40 degrees)
        entity_manager: The entity manager for the entity.
        entity_attr: The attribute name of the entity in the environment.
                        This isn't necessary if `entity_manager` is provided.
        grace_steps: Number of steps at episode start to ignore tilt detection (default: 0)
                     This gives the robot a chance to stabilize before tilt detection is active.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """
    in_grace_period = env.episode_length <= grace_steps

    # Get the projected gravity vector in body frame
    projected_gravity = None
    if entity_manager is not None:
        projected_gravity = entity_manager.get_projected_gravity()
    else:
        entity = getattr(env, entity_attr)
        projected_gravity = entity_projected_gravity(entity)

    # Calculate the magnitude of tilt (distance from perfectly upright)
    projected_gravity_xy = projected_gravity[:, :2]
    tilt_magnitude = torch.norm(projected_gravity_xy, dim=1)

    # Convert tilt magnitude to angle
    tilt_angle = torch.asin(torch.clamp(tilt_magnitude, max=0.99))

    # Terminate if tilt angle exceeds the limit
    return (~in_grace_period) & (tilt_angle > math.radians(limit_angle))


def is_upsidedown(
    env: GenesisEnv,
    threshold: float = 0.5,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
    grace_steps: int = 0,
) -> torch.Tensor:
    """
    Terminate when the robot is belly-up (inverted).

    Uses projected gravity in the body frame: upright is approximately [0, 0, -1],
    belly-up is approximately [0, 0, +1]. Side-lying poses keep z below threshold.

    Args:
        env: The Genesis environment
        threshold: Terminate when projected_gravity[:, 2] exceeds this value
        entity_manager: The entity manager for the robot
        entity_attr: Entity attribute if entity_manager is not provided
        grace_steps: Steps at episode start to ignore this check
    """
    in_grace_period = env.episode_length <= grace_steps

    if entity_manager is not None:
        projected_gravity = entity_manager.get_projected_gravity()
    else:
        entity = getattr(env, entity_attr)
        projected_gravity = entity_projected_gravity(entity)

    return (~in_grace_period) & (projected_gravity[:, 2] > threshold)


def base_height_below_minimum(
    env: GenesisEnv,
    minimum_height: float = 0.05,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Terminate the environment if the robot's base height falls below a minimum threshold.

    Args:
        env: The Genesis environment containing the robot
        minimum_height: Minimum allowed base height in meters
        entity_manager: The entity manager for the entity.
        entity_attr: The attribute name of the entity in the environment.
                        This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate
    """
    base_pos = None
    if entity_manager is not None:
        base_pos = entity_manager.base_pos
    else:
        entity = getattr(env, entity_attr)
        base_pos = entity.get_pos()
    return base_pos[:, 2] < minimum_height


def out_of_bounds(
    env: GenesisEnv,
    terrain_manager: TerrainManager,
    subterrain: str | None = None,
    border_margin: float = 0.5,
    entity_attr: str = "robot",
) -> torch.Tensor:
    """
    Terminate if the entity's base position is outside of the terrain.

    Args:
        env: The Genesis environment containing the robot
        terrain_manager: The terrain manager to check for out of bounds
        subterrain: The subterrain to keep the robot inside of
        border_margin: The margin (in meters) to add to the terrain bounds
                       This terminates the episode before the robot falls off the terrain.
        entity_attr: The attribute name of the entity in the environment.
                        This isn't necessary if `entity_manager` is provided.
    """
    # Get the entity's base position
    entity = getattr(env, entity_attr)
    position = entity.get_pos()

    # Get terrain bounds
    (x_min, x_max, y_min, y_max) = terrain_manager.get_bounds(subterrain)
    x_min_bound, x_max_bound = x_min + border_margin, x_max - border_margin
    y_min_bound, y_max_bound = y_min + border_margin, y_max - border_margin

    # Check bounds
    x_pos, y_pos = position[:, 0], position[:, 1]
    return (
        (x_pos < x_min_bound)
        | (x_pos > x_max_bound)
        | (y_pos < y_min_bound)
        | (y_pos > y_max_bound)
    )


def has_contact(
    _env: GenesisEnv, contact_manager: ContactManager, threshold=1.0, min_contacts=1
) -> torch.Tensor:
    """
    One or more links in the contact manager are in contact with something.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold, per contact, for contact detection (default: 1.0)
        min_contacts: The minimum number of contacts required to terminate (default: 1)

    Returns:
        True for each environment that has contact
    """
    has_contact = contact_manager.contacts[:, :].norm(dim=-1) > threshold
    return has_contact.sum(dim=1) >= min_contacts


def contact_force(
    _env: GenesisEnv, contact_manager: ContactManager, threshold: float = 1.0
) -> torch.Tensor:
    """
    Terminate if any link in the contact manager is in contact with something with a force greater than the threshold.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0 N)

    Returns:
        The total force for the contact manager for each environment
    """
    return torch.any(torch.norm(contact_manager.contacts, dim=-1) > threshold, dim=-1)


def contact_force_with_grace_period(
    env: GenesisEnv,
    contact_manager: ContactManager,
    threshold: float = 100.0,
    grace_steps: int = 10,
) -> torch.Tensor:
    """
    Terminate if contact force exceeds threshold, with a grace period at episode start.

    This is useful for quadrupeds that may start in slightly unstable positions
    and need a few steps to stabilize before fall detection becomes active.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 100.0 N)
        grace_steps: Number of steps at episode start to ignore contacts (default: 10)

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    # Don't terminate during grace period (early in episode)
    in_grace_period = env.episode_length <= grace_steps

    # Check contact forces
    contact_exceeded = torch.any(
        torch.norm(contact_manager.contacts, dim=-1) > threshold, dim=-1
    )

    # Only terminate if past grace period AND contact exceeded
    return (~in_grace_period) & contact_exceeded.detach()


def dof_control_force_limit(
    _env: GenesisEnv,
    actuator_manager: ActuatorManager,
    threshold: float | None = None,
) -> torch.Tensor:
    """
    Terminate if any joint's commanded actuator force exceeds a limit (+/-).

    Uses control/output force (what the actuator commands), not measured joint load.
    Suitable for teaching policies to stay within rated motor torque.

    Args:
        env: The Genesis environment
        actuator_manager: Actuator manager for the controlled joints
        threshold: Force/torque limit (in simulator units). 
                   If None, uses `max_force` value from actuator_manager.

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    force = actuator_manager.get_dofs_control_force()
    if threshold is None:
        threshold = actuator_manager.get_dofs_max_force()
    return torch.any(torch.abs(force) > threshold, dim=-1)


def dof_velocity_limit(
    _env: GenesisEnv,
    actuator_manager: ActuatorManager,
    threshold: float,
    unit: Literal["rpm", "rad"] = "rad",
) -> torch.Tensor:
    """
    Terminate if any of the actuator_manager's joints moves faster than a speed limit.

    Args:
        env: The Genesis environment
        actuator_manager: Actuator manager for the controlled joints
        threshold: Speed limit in the units given by `unit`
        unit: The speed units
              - `"rad"` for radians per second (default)
              - `"rpm"` for revolutions per minute

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    assert unit in ["rad", "rpm"], f"Unknown velocity unit '{unit}'. Use 'rad' or 'rpm'."

    if unit == "rpm":
        threshold = threshold * (2 * math.pi / 60)
    vel = actuator_manager.get_dofs_velocity()
    return torch.any(torch.abs(vel) > threshold, dim=-1)
