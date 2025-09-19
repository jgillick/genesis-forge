"""
Termination functions for the Genesis environment.
Each of these should return a boolean tensor indicating which environments should terminate, in the tensor shape (num_envs,).
"""

import math
import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.utils import entity_projected_gravity
from genesis_forge.managers import (
    ContactManager,
    EntityManager,
)


def timeout(env: GenesisEnv) -> torch.Tensor:
    """
    Terminate the environment if the episode length exceeds the maximum episode length.
    """
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
