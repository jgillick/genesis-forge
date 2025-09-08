"""
Reward functions for the Genesis environment.
Each of these should return a float tensor with the reward value for each environment, in the shape (num_envs,).
"""

import torch
from typing import Union
import genesis as gs
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    CommandManager,
    VelocityCommandManager,
    PositionActionManager,
    ContactManager,
    TerminationManager,
    TerrainManager,
)
from genesis_forge.utils import entity_lin_vel, entity_ang_vel, entity_projected_gravity

"""
Aliveness
"""


def is_alive(env: GenesisEnv, term_manager: TerminationManager) -> torch.Tensor:
    """
    Reward for being alive and not terminating this step.
    """
    return (~term_manager.terminated).float()


def is_terminated(env: GenesisEnv, term_manager: TerminationManager) -> torch.Tensor:
    """
    Penalize terminated episodes that terminated.
    """
    return term_manager.terminated.float()


"""
Robot base position/state
"""


def base_height(
    env: GenesisEnv,
    target_height: Union[float, torch.Tensor] = None,
    height_command: CommandManager = None,
    terrain_manager: TerrainManager = None,
) -> torch.Tensor:
    """
    Penalize base height away from target

    Args:
        env: The Genesis environment containing the robot
        target_height: The target height to penalize the base height away from
        height_command: Get the target height from a height command manager. This expects the command to have a single range value.
        terrain_manager: The terrain manager will adjust the height based on the terrain height.

    Returns:
        torch.Tensor: Penalty for base height away from target
    """
    base_pos = env.robot.get_pos()
    height_offset = 0.0
    if terrain_manager is not None:
        height_offset = terrain_manager.get_terrain_height(
            base_pos[:, 0], base_pos[:, 1]
        )
    if height_command is not None:
        target_height = height_command.command.squeeze(-1)
    return torch.square(base_pos[:, 2] - height_offset - target_height)


def dof_similar_to_default(
    env: GenesisEnv,
    dof_action_manager: PositionActionManager,
):
    """
    Penalize joint poses far away from default pose

    Args:
        env: The Genesis environment containing the robot
        dof_action_manager: The DOF action manager

    Returns:
        torch.Tensor: Penalty for joint poses far away from default pose
    """
    dof_pos = dof_action_manager.get_dofs_position()
    default_pos = dof_action_manager.default_dofs_pos
    return torch.sum(torch.abs(dof_pos - default_pos), dim=1)


def lin_vel_z(env: GenesisEnv) -> torch.Tensor:
    """
    Penalize z axis base linear velocity

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Penalty for z axis base linear velocity
    """
    linear_vel = entity_lin_vel(env.robot)
    return torch.square(linear_vel[:, 2])


def flat_orientation_l2(env: GenesisEnv) -> torch.Tensor:
    """
    Penalize non-flat base orientation using L2 squared kernel.
    This is computed by penalizing the xy-components of the projected gravity vector.

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Penalty for non-flat base orientation
    """
    # Get the projected gravity vector in the robot's base frame
    # This represents how "tilted" the robot is from upright
    projected_gravity = entity_projected_gravity(env.robot)

    # Penalize the xy-components (horizontal tilt) using L2 squared kernel
    # A flat orientation means these components should be close to zero
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


"""
Action penalties.
"""


def action_rate(env: GenesisEnv) -> torch.Tensor:
    """
    Penalize changes in actions

    Args:
        env: The Genesis environment containing the robot

    Returns:
        torch.Tensor: Penalty for changes in actions
    """
    actions = env.actions
    last_actions = env.last_actions
    if last_actions is None:
        return torch.zeros_like(actions, device=gs.device)
    return torch.sum(torch.square(last_actions - actions), dim=1)


"""
Velocity Tracking
"""


def command_tracking_lin_vel(
    env: GenesisEnv,
    vel_cmd_manager: VelocityCommandManager,
    sensitivity: float = 0.25,
) -> torch.Tensor:
    """
    Penalize not tracking commanded linear velocity (xy axes)

    Args:
        env: The Genesis environment containing the robot
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error

    Returns:
        torch.Tensor: Penalty for tracking of linear velocity commands (xy axes)
    """
    command = vel_cmd_manager.command
    linear_vel_local = entity_lin_vel(env.robot)
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - linear_vel_local[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / sensitivity)


def command_tracking_ang_vel(
    env: GenesisEnv,
    vel_cmd_manager: VelocityCommandManager,
    sensitivity: float = 0.25,
) -> torch.Tensor:
    """
    Penalize not tracking commanded angular velocity (yaw)

    Args:
        env: The Genesis environment containing the robot
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error

    Returns:
        torch.Tensor: Penalty for tracking of angular velocity commands (yaw)
    """
    command = vel_cmd_manager.command
    angular_vel = entity_ang_vel(env.robot)
    ang_vel_error = torch.square(command[:, 2] - angular_vel[:, 2])
    return torch.exp(-ang_vel_error / sensitivity)


"""
Contacts
"""


def has_contact(
    _env: GenesisEnv, contact_manager: ContactManager, threshold=1.0, min_contacts=1
) -> torch.Tensor:
    """
    One or more links in the contact manager are in contact with something.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0)
        min_contacts: The minimum number of contacts required. (default: 1)

    Returns:
        1 for each contact meeting the threshold
    """
    has_contact = contact_manager.contacts[:, :].norm(dim=-1) > threshold
    result = has_contact.sum(dim=1) >= min_contacts
    return result.float()


def contact_force(
    _env: GenesisEnv, contact_manager: ContactManager, threshold: float = 1.0
) -> torch.Tensor:
    """
    Reward for the total contact force acting on all the target links in the contact manager over the threshold.

    Args:
        env: The Genesis environment containing the robot
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0 N)

    Returns:
        The total force for the contact manager for each environment
    """
    violation = torch.norm(contact_manager.contacts[:, :, :], dim=-1) - threshold
    return torch.sum(violation.clip(min=0.0), dim=1)


def feet_air_time(
    env: GenesisEnv,
    contact_manager: ContactManager,
    threshold: float,
    vel_cmd_manager: VelocityCommandManager,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    made_contact = contact_manager.has_made_contact(env.dt)
    last_air_time = contact_manager.last_air_time
    reward = torch.sum((last_air_time - threshold) * made_contact, dim=1)
    # no reward for zero velocity command
    if vel_cmd_manager is not None:
        reward *= torch.norm(vel_cmd_manager.command[:, :2], dim=1) > 0.1
    return reward
