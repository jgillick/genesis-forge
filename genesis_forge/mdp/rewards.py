"""
Reward functions for the Genesis Forge environment.
Each of these should return a float tensor with the reward value for each environment, in the shape (num_envs,).
"""

from __future__ import annotations

import torch
import genesis as gs
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    CommandManager,
    VelocityCommandManager,
    PositionActionManager,
    ContactManager,
    TerrainManager,
    EntityManager,
)
from genesis_forge.utils import entity_lin_vel, entity_ang_vel, entity_projected_gravity
from genesis_forge.managers import MdpFnClass
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


"""
Aliveness
"""


def is_alive(env: GenesisEnv) -> torch.Tensor:
    """
    Reward for being alive and not terminating this step.
    This assumes that `env.extras["terminations"]` is a boolean tensor with the termination signals for the environments.
    """
    terminations: torch.Tensor = env.extras["terminations"]
    return (~terminations).float().detach()


def terminated(env: GenesisEnv) -> torch.Tensor:
    """
    Penalize terminated episodes that terminated.
    This assumes that `env.extras["terminations"]` is a boolean tensor with the termination signals for the environments.
    """
    terminations: torch.Tensor = env.extras["terminations"]
    return terminations.float().detach()


"""
Robot base position/state
"""


def base_height(
    env: GenesisEnv,
    target_height: Union[float, torch.Tensor] = None,
    height_command: CommandManager = None,
    terrain_manager: TerrainManager = None,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Penalize base height away from target, using the L2 squared kernel.

    Args:
        env: The Genesis environment containing the robot
        target_height: The target height to penalize the base height away from
        height_command: Get the target height from a height command manager. This expects the command to have a single range value.
        terrain_manager: The terrain manager will adjust the height based on the terrain height.
        entity_attr: The attribute name of the entity in the environment.
        entity_manager: The entity manager for the entity.

    Returns:
        torch.Tensor: Penalty for base height away from target
    """
    robot = None
    if entity_manager is not None:
        robot = entity_manager.entity
    else:
        robot = getattr(env, entity_attr)

    base_pos = robot.get_pos()
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
    actuator_manager: ActuatorManager | list[ActuatorManager] | None = None,
    action_manager: PositionActionManager | None = None,
) -> torch.Tensor:
    """
    Penalize joint poses far away from default pose(s).

    Pass ``actuator_manager`` as one manager or a non-empty list/tuple (e.g. per-limb
    stacks); penalties are summed per environment across all included DOFs.

    Args:
        env: The Genesis environment containing the robot (unused today; accepted for MDP signature consistency).
        actuator_manager: One or more actuator managers.
        action_manager: (deprecated) One or more position-action managers. Use
            ``actuator_manager`` instead.

    Returns:
        torch.Tensor: Penalty summed over included DOFs, shape ``(num_envs,)``.
    """
    if actuator_manager is not None:
        if isinstance(actuator_manager, list):
            total = None
            for mgr in actuator_manager:
                dof_pos = mgr.get_dofs_position()
                part = torch.sum(torch.abs(dof_pos - mgr.default_dofs_pos), dim=1)
                total = part if total is None else total + part
            return total
        else:
            dof_pos = actuator_manager.get_dofs_position()
            default_pos = actuator_manager.default_dofs_pos
            return torch.sum(torch.abs(dof_pos - default_pos), dim=1)
    elif action_manager is not None:
        dof_pos = action_manager.get_dofs_position()
        default_pos = action_manager.default_dofs_pos
        return torch.sum(torch.abs(dof_pos - default_pos), dim=1)

    raise ValueError("dof_similar_to_default: Either actuator_manager or action_manager must be provided")



def lin_vel_z_l2(
    env: GenesisEnv,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Penalize z axis base linear velocity

    Args:
        env: The Genesis environment containing the entity
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Penalty for z axis base linear velocity
    """
    linear_vel = None
    if entity_manager is not None:
        linear_vel = entity_manager.get_linear_velocity()
    else:
        robot = getattr(env, entity_attr)
        linear_vel = entity_lin_vel(robot)
    return torch.square(linear_vel[:, 2])

def lin_vel_xy_l2(env: GenesisEnv, entity_manager) -> torch.Tensor:
    """Penalize horizontal base linear velocity."""
    lin_vel = entity_manager.get_linear_velocity()
    return torch.sum(torch.square(lin_vel[:, :2]), dim=1)

def ang_vel_xy_l2(
    env: GenesisEnv,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
):
    """
    Penalize xy-axis base angular velocity using L2 squared kernel.

    Args:
        env: The Genesis environment containing the entity
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor
    """
    angle_vel = None
    if entity_manager is not None:
        angle_vel = entity_manager.get_angular_velocity()
    else:
        robot = getattr(env, entity_attr)
        angle_vel = entity_ang_vel(robot)
    return torch.sum(torch.square(angle_vel[:, :2]), dim=1)


def flat_orientation_l2(
    env: GenesisEnv,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Penalize non-flat base orientation using L2 squared kernel.
    This is computed by penalizing the xy-components of the projected gravity vector.

    Args:
        env: The Genesis environment containing the robot
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Penalty for non-flat base orientation
    """
    # Get the projected gravity vector in the robot's base frame
    # This represents how "tilted" the robot is from upright
    projected_gravity = None
    if entity_manager is not None:
        projected_gravity = entity_manager.get_projected_gravity()
    else:
        robot = getattr(env, entity_attr)
        projected_gravity = entity_projected_gravity(robot)

    # Penalize the xy-components (horizontal tilt) using L2 squared kernel
    # A flat orientation means these components should be close to zero
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


class body_acceleration_exp(MdpFnClass):
    """
    Penalize jerky body acceleration to encourage smooth locomotion.

    Args:
        env: The Genesis environment containing the robot
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.
        sensitivity: The sensitivity of the exponential decay. A lower value means the reward is more sensitive to the error.
    """

    def __init__(
        self,
        env: GenesisEnv,
        entity_attr: str = "robot",
        entity_manager: EntityManager = None,
        sensitivity: float = 0.10,
    ):
        super().__init__(env)

    def __call__(
        self,
        env: GenesisEnv,
        entity_attr: str = "robot",
        entity_manager: EntityManager = None,
        sensitivity: float = 0.10,
    ):
        # Current velocities
        curr_lin_vel = None
        curr_ang_vel = None
        if entity_manager is not None:
            curr_lin_vel = entity_manager.get_linear_velocity()
            curr_ang_vel = entity_manager.get_angular_velocity()
        else:
            robot = getattr(env, entity_attr)
            curr_lin_vel = entity_lin_vel(robot)
            curr_ang_vel = entity_ang_vel(robot)

        # Calculate acceleration from previous step
        if hasattr(self, "prev_lin_vel"):
            lin_acc = (curr_lin_vel - self.prev_lin_vel) / env.dt
            ang_acc = (curr_ang_vel - self.prev_ang_vel) / env.dt
        else:
            lin_acc = torch.zeros_like(curr_lin_vel)
            ang_acc = torch.zeros_like(curr_ang_vel)

        # Store for next step
        self.prev_lin_vel = curr_lin_vel.clone()
        self.prev_ang_vel = curr_ang_vel.clone()

        # Calculate penalty using exponential kernel
        pelvis_motion = torch.norm(lin_acc, dim=-1) + torch.norm(ang_acc, dim=-1)
        return 1 - torch.exp(-sensitivity * pelvis_motion)


"""
Action penalties.
"""


def action_rate_l2(env: GenesisEnv) -> torch.Tensor:
    """
    Penalize the rate of change of the actions using L2 squared kernel.

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


class action_acceleration_l2(MdpFnClass):
    """
    Targets jittery oscillations (rather than smooth consistent movement), by penalize the second-order 
    finite difference of actions (discrete acceleration) using the L2 squared kernel. 
    
    This encourages a smooth consistent movement, where a smooth ramp has zero acceleration even at high velocity.

    A smooth action ramp looks like this: 0.5 → 0.6 → 0.7 → 0.8
     * Velocities: 0.1, 0.1, 0.1
     * Accelerations: 0.0, 0.0 (zero -- perfectly smooth)
     * Penalty: zero

    A jittery action ramp looks like this: 0.5 → 0.8 → 0.5 → 0.8
     * Velocities: 0.3, -0.3, 0.3
     * Accelerations: -0.6, 0.6 (large -- direction keeps reversing)
     * Penalty: very large

    The acceleration is computed as:

    .. math::

        \\text{acc}_t = a_t - 2 \\cdot a_{t-1} + a_{t-2}

    and the penalty is :math:`\\sum \\text{acc}_t^2` across all action dimensions.

    Args:
        env: The Genesis environment containing the robot
        action_manager: Optional action manager to source actions from.
                        If not provided, actions are read from ``env.actions``.
    """

    def __init__(
        self,
        env: GenesisEnv,
        action_manager: PositionActionManager = None,
    ):
        super().__init__(env)
        self.env = env
        self._prev_action: torch.Tensor | None = None
        self._prev_prev_action: torch.Tensor | None = None
        self._action_log_count: torch.Tensor | None = None

    def _init_buffers(self, actions: torch.Tensor):
        self._prev_action = torch.zeros_like(actions)
        self._prev_prev_action = torch.zeros_like(actions)
        self._action_log_count = torch.zeros((self.env.num_envs, ), dtype=torch.long, device=gs.device)

    def reset(self, envs_idx):
        """
        Clear the action history for the specified environments.
        """
        if self._prev_action is None:
            return
        self._prev_action[envs_idx] = 0.0
        self._prev_prev_action[envs_idx] = 0.0
        self._action_log_count[envs_idx] = 0

    def __call__(
        self,
        env: GenesisEnv,
        action_manager: PositionActionManager = None,
    ) -> torch.Tensor:
        # Get the current actions for this step
        actions = env.actions
        if action_manager is not None:
            actions = action_manager.get_actions()

        # Initialize the buffers, if necessary
        if self._prev_action is None:
            self._init_buffers(actions)

        # Calculate the acceleration
        acceleration = actions - 2.0 * self._prev_action + self._prev_prev_action
        penalty = torch.sum(torch.square(acceleration), dim=1)

        # Mask out envs that don't yet have two steps of valid history
        penalty = penalty * (self._action_log_count >= 2)

        # Shift the actions to the next step
        self._prev_prev_action = self._prev_action
        self._prev_action = actions.clone()
        self._action_log_count.add_(1).clamp_(max=2)

        return penalty


def dof_torque_l2(
    env: GenesisEnv,
    actuator_manager: ActuatorManager,
) -> torch.Tensor:
    """
    Penalize joint torque effort using the L2 squared kernel.

    Discourages the policy from applying unnecessary force, particularly when the
    robot is near equilibrium. This helps reduce actuator oscillation when the robot
    is stationary or moving slowly.

    Args:
        env: The Genesis environment containing the robot
        actuator_manager: The actuator manager to retrieve DOF forces from.

    Returns:
        torch.Tensor: Penalty for joint torque effort, shape (num_envs,)
    """
    torque = actuator_manager.get_dofs_control_force()
    return torch.sum(torch.square(torque), dim=1)

def dof_velocity_l2(env: GenesisEnv, action_manager: PositionActionManager) -> torch.Tensor:
    """Penalize joint angular velocities to encourage slow, deliberate motion."""
    dof_vel = action_manager.get_dofs_velocity()
    return torch.sum(torch.square(dof_vel), dim=1)


"""
Velocity Command Rewards
"""


def command_tracking_lin_vel(
    env: GenesisEnv,
    command: torch.Tensor = None,
    vel_cmd_manager: VelocityCommandManager = None,
    sensitivity: float = 0.25,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Reward for tracking commanded linear velocity (xy axes)

    Args:
        env: The Genesis environment containing the robot
        command: The commanded XY linear velocity in the shape (num_envs, 2)
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Reward for tracking of linear velocity commands (xy axes)
    """
    assert (
        command is not None or vel_cmd_manager is not None
    ), "Either command or vel_cmd_manager must be provided to command_tracking_lin_vel"

    linear_vel_local = None
    if entity_manager is not None:
        linear_vel_local = entity_manager.get_linear_velocity()
    else:
        robot = getattr(env, entity_attr)
        linear_vel_local = entity_lin_vel(robot)

    if vel_cmd_manager is not None:
        command = vel_cmd_manager.command[:, :2]

    lin_vel_error = torch.sum(torch.square(command - linear_vel_local[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / sensitivity)


def command_tracking_ang_vel(
    env: GenesisEnv,
    commanded_ang_vel: torch.Tensor = None,
    vel_cmd_manager: VelocityCommandManager = None,
    sensitivity: float = 0.25,
    entity_attr: str = "robot",
    entity_manager: EntityManager = None,
) -> torch.Tensor:
    """
    Reward for tracking commanded angular velocity (yaw)

    Args:
        env: The Genesis Forge environment
        commanded_ang_vel: The commanded angular velocity in the shape (num_envs, 1)
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity_attr` parameter.
        entity_attr: The attribute name of the entity in the environment. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Reward for tracking of angular velocity commands (yaw)
    """
    assert (
        commanded_ang_vel is not None or vel_cmd_manager is not None
    ), "Either commanded_ang_vel or vel_cmd_manager must be provided to command_tracking_ang_vel"

    angular_vel = None
    if entity_manager is not None:
        angular_vel = entity_manager.get_angular_velocity()
    else:
        robot = getattr(env, entity_attr)
        angular_vel = entity_ang_vel(robot)

    if vel_cmd_manager is not None:
        commanded_ang_vel = vel_cmd_manager.command[:, 2]

    ang_vel_error = torch.square(commanded_ang_vel - angular_vel[:, 2])
    return torch.exp(-ang_vel_error / sensitivity)


def stand_still_joint_deviation_l1(
    env,
    vel_cmd_manager: VelocityCommandManager,
    actuator_manager: ActuatorManager = None,
    command_threshold: float = 0.06,
    action_manager: PositionActionManager = None,
) -> torch.Tensor:
    """
    Penalize offsets from the default joint positions when the command is very small.

    Args:
        env: The Genesis Forge environment
        command_threshold: The threshold for the command to be considered small
        vel_cmd_manager: The velocity command manager
        actuator_manager: The actuator manager to get the joint positions and recent actions from.
        action_manager: The action manager to get the joint positions and recent actions from.

    Returns:
        torch.Tensor: Penalty for offsets from the default joint positions when the command is very small
    """
    assert (
        actuator_manager is not None or action_manager is not None
    ), "Either actuator_manager or action_manager must be provided to stand_still_joint_deviation_l1"

    if actuator_manager is not None:
        joint_pos = actuator_manager.get_dofs_position()
        default_pos = actuator_manager.default_dofs_pos
    elif action_manager is not None:
        joint_pos = action_manager.get_dofs_position()
        default_pos = action_manager.default_dofs_pos
    joint_deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)

    # Penalize motion when command is nearly zero.
    command = vel_cmd_manager.command
    return joint_deviation * (torch.norm(command[:, :2], dim=1) < command_threshold)


"""
Contacts
"""


def has_contact(
    env: GenesisEnv, contact_manager: ContactManager, threshold=1.0, min_contacts=1
) -> torch.Tensor:
    """
    One or more links in the contact manager are in contact with something.

    Args:
        env: The Genesis Forge environment
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
    env: GenesisEnv, contact_manager: ContactManager, threshold: float = 1.0
) -> torch.Tensor:
    """
    Reward for the total contact force acting on all the target links in the contact manager over the threshold.

    Args:
        env: The Genesis Forge environment
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
    time_threshold: float,
    time_threshold_max: float | None = None,
    vel_cmd_manager: VelocityCommandManager | None = None,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the velocity commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.

    Args:
        env: The Genesis Forge environment
        contact_manager: The contact manager to check for contact
        time_threshold: The minimum time (in seconds) the feet should be in the air
        time_threshold_max: (optional) The maximum time (in seconds) the feet should be in the air.
                            The reward will be capped at this value.
        vel_cmd_manager: The velocity command manager

    Returns:
        The reward for the feet air time
    """
    made_contact = contact_manager.has_made_contact(env.dt)
    last_air_time = contact_manager.last_air_time

    # Calculate the air time
    air_time = (last_air_time - time_threshold) * made_contact
    if time_threshold_max is not None:
        air_time = torch.clamp(air_time, max=time_threshold_max - time_threshold)
    reward = torch.sum(air_time, dim=1)

    # no reward for zero velocity command
    if vel_cmd_manager is not None:
        reward *= torch.norm(vel_cmd_manager.command[:, :2], dim=1) > 0.1
    return reward


def feet_ground_time(
    env: GenesisEnv,
    contact_manager: ContactManager,
    time_threshold: float,
) -> torch.Tensor:
    """Penalize brief ground contacts (foot tapping) using a linear kernel.

    Fires at the moment a foot lifts off. The penalty is proportional to how
    much the stance duration fell below time_threshold. A proper stance phase
    (contact_time >= time_threshold) produces zero penalty.

    Intended to be paired with feet_air_time (positive reward) to fully shape
    gait timing: feet_air_time rewards long swings while this penalizes taps.
    Use a negative weight in the RewardManager.

    Args:
        env: The Genesis Forge environment
        contact_manager: The contact manager to check for contact
        time_threshold: Contacts shorter than this (in seconds) are penalized.
                        Set independently from the feet_air_time threshold based
                        on the expected stance duration of your target gait.

    Returns:
        The penalty for brief ground contacts, shape (num_envs,)
    """
    just_lifted = contact_manager.has_broken_contact(env.dt)
    last_contact_time = contact_manager.last_contact_time
    short_contact = (time_threshold - last_contact_time).clamp(min=0.0) * just_lifted
    return torch.sum(short_contact, dim=1)


def feet_slide(
    env,
    contact_manager: ContactManager,
    entity_attr: str = "robot",
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.

    This penalty is less effective at longer foot-contact links (for example, long legs without dedicated foot links),
    because they might have some velocity while they're being used to move the robot. However, dedicated foot links
    will be stationary on the ground and not moving while pushing the robot forward.

    Args:
        env: The Genesis Forge environment
        contact_manager: The contact manager for the feet
        entity_attr: The attribute name of the robot entity that the feet are attached to.

    Returns:
        The penalty for the feet slide
    """
    # Get links in contact
    contacts = torch.norm(contact_manager.contacts[:, :, :], dim=-1) > 1.0

    # Get link velocities.
    # If the links aren't moving, then they're being used to move the robot and not sliding.
    link_ids = contact_manager.local_link_ids
    robot: RigidEntity = getattr(env, entity_attr)
    link_vel = robot.get_links_vel(links_idx_local=link_ids)

    return torch.sum(link_vel.norm(dim=-1) * contacts, dim=1)
