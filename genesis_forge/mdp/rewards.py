"""
Reward functions for the Genesis Forge environment.
Each of these should return a float tensor with the reward value for each environment, in the shape (num_envs,).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import genesis as gs
import torch
from deprecated import deprecated

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    CommandManager,
    ContactManager,
    EntityManager,
    MdpFn,
    PositionActionManager,
    TerrainManager,
    VelocityCommandManager,
)
from genesis_forge.utils import entity_ang_vel, entity_lin_vel, entity_projected_gravity

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity


"""
Aliveness
"""


@dataclass(kw_only=True, eq=False)
class is_alive(MdpFn):
    """
    Reward for being alive and not terminating this step.
    This assumes that `env.extras["terminations"]` is a boolean tensor with the termination signals for the environments.
    """

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        terminations: torch.Tensor = env.extras["terminations"]
        return (~terminations).float().detach()


@dataclass(kw_only=True, eq=False)
class terminated(MdpFn):
    """
    Penalize terminated episodes that terminated.
    This assumes that `env.extras["terminations"]` is a boolean tensor with the termination signals for the environments.
    """

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        terminations: torch.Tensor = env.extras["terminations"]
        return terminations.float().detach()


"""
Robot base position/state
"""


@dataclass(kw_only=True, eq=False)
class base_height(MdpFn):
    """
    Penalize base height away from target, using the L2 squared kernel.

    Args:
        target_height: The target height to penalize the base height away from
        height_command: Get the target height from a height command manager. This expects the command to have a single range value.
        terrain_manager: The terrain manager will adjust the height based on the terrain height.
        entity: The entity to compute the reward for. Defaults to `env.robot`. Not necessary if `entity_manager` is provided.
        entity_manager: The entity manager for the entity.

    Returns:
        torch.Tensor: Penalty for base height away from target
    """

    target_height: float | torch.Tensor = None
    height_command: CommandManager = None
    terrain_manager: TerrainManager = None
    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            robot = self.entity_manager.entity
        else:
            robot = self.entity if self.entity is not None else env.robot

        base_pos = robot.get_pos()
        height_offset = 0.0
        if self.terrain_manager is not None:
            height_offset = self.terrain_manager.get_terrain_height(
                base_pos[:, 0], base_pos[:, 1]
            )

        target_height = self.target_height
        if self.height_command is not None:
            target_height = self.height_command.command.squeeze(-1)
        return torch.square(base_pos[:, 2] - height_offset - target_height)


@dataclass(kw_only=True, eq=False)
class dof_similar_to_default(MdpFn):
    """
    Penalize joint poses far away from default pose(s).

    Pass ``actuator_manager`` as one manager or a non-empty list/tuple (e.g. per-limb
    stacks); penalties are summed per environment across all included DOFs.

    Args:
        actuator_manager: One or more actuator managers.

    Returns:
        torch.Tensor: Penalty summed over included DOFs, shape ``(num_envs,)``.
    """

    actuator_manager: ActuatorManager | list[ActuatorManager]

    def build(self):
        if self.actuator_manager is None:
            raise ValueError(
                "dof_similar_to_default: actuator_manager must be provided"
            )

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if isinstance(self.actuator_manager, list):
            total = None
            for mgr in self.actuator_manager:
                dof_pos = mgr.get_dofs_position()
                part = torch.sum(torch.abs(dof_pos - mgr.default_dofs_pos), dim=1)
                total = part if total is None else total + part
            return total
        dof_pos = self.actuator_manager.get_dofs_position()
        default_pos = self.actuator_manager.default_dofs_pos
        return torch.sum(torch.abs(dof_pos - default_pos), dim=1)


@dataclass(kw_only=True, eq=False)
class lin_vel_z_l2(MdpFn):
    """
    Penalize z axis base linear velocity

    Args:
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Penalty for z axis base linear velocity
    """

    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            linear_vel = self.entity_manager.get_linear_velocity()
        else:
            robot = self.entity if self.entity is not None else env.robot
            linear_vel = entity_lin_vel(robot)
        return torch.square(linear_vel[:, 2])


@dataclass(kw_only=True, eq=False)
class lin_vel_xy_l2(MdpFn):
    """
    Penalize horizontal base linear velocity.

    Args:
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Penalty for xy axis base linear velocity
    """

    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            lin_vel = self.entity_manager.get_linear_velocity()
        else:
            robot = self.entity if self.entity is not None else env.robot
            lin_vel = entity_lin_vel(robot)
        return torch.sum(torch.square(lin_vel[:, :2]), dim=1)


@dataclass(kw_only=True, eq=False)
class ang_vel_xy_l2(MdpFn):
    """
    Penalize xy-axis base angular velocity using L2 squared kernel.

    Args:
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor
    """

    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            angle_vel = self.entity_manager.get_angular_velocity()
        else:
            robot = self.entity if self.entity is not None else env.robot
            angle_vel = entity_ang_vel(robot)
        return torch.sum(torch.square(angle_vel[:, :2]), dim=1)


@dataclass(kw_only=True, eq=False)
class flat_orientation_l2(MdpFn):
    """
    Penalize non-flat base orientation using L2 squared kernel.
    This is computed by penalizing the xy-components of the projected gravity vector.

    Args:
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Penalty for non-flat base orientation
    """

    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        # Get the projected gravity vector in the robot's base frame.
        # This represents how "tilted" the robot is from upright.
        if self.entity_manager is not None:
            projected_gravity = self.entity_manager.get_projected_gravity()
        else:
            robot = self.entity if self.entity is not None else env.robot
            projected_gravity = entity_projected_gravity(robot)

        # Penalize the xy-components (horizontal tilt) using L2 squared kernel.
        # A flat orientation means these components should be close to zero.
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)


@dataclass(kw_only=True, eq=False)
class body_acceleration_exp(MdpFn):
    """
    Penalize jerky body acceleration to encourage smooth locomotion.

    Args:
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.
        sensitivity: The sensitivity of the exponential decay. A lower value means the reward is more sensitive to the error.
    """

    entity: RigidEntity = None
    entity_manager: EntityManager = None
    sensitivity: float = 0.10

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        sensitivity = self.sensitivity

        # Current velocities
        curr_lin_vel = None
        curr_ang_vel = None
        if self.entity_manager is not None:
            curr_lin_vel = self.entity_manager.get_linear_velocity()
            curr_ang_vel = self.entity_manager.get_angular_velocity()
        else:
            robot = self.entity if self.entity is not None else env.robot
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


@dataclass(kw_only=True, eq=False)
class action_rate_l2(MdpFn):
    """
    Penalize the rate of change of the actions using L2 squared kernel.

    Returns:
        torch.Tensor: Penalty for changes in actions
    """

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        actions = env.actions
        last_actions = env.last_actions
        if last_actions is None:
            return torch.zeros_like(actions, device=gs.device)
        return torch.sum(torch.square(last_actions - actions), dim=1)


@dataclass(kw_only=True, eq=False)
class action_acceleration_l2(MdpFn):
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
        action_manager: Optional action manager to source actions from.
                        If not provided, actions are read from ``env.actions``.
    """

    action_manager: PositionActionManager = None

    def build(self):
        # Buffers are sized to the action dimension, which is not known until the first
        # step supplies an action tensor -- so they stay lazily allocated in __call__.
        self._prev_action: torch.Tensor | None = None
        self._prev_prev_action: torch.Tensor | None = None
        self._action_log_count: torch.Tensor | None = None

    def _init_buffers(self, actions: torch.Tensor):
        self._prev_action = torch.zeros_like(actions)
        self._prev_prev_action = torch.zeros_like(actions)
        self._action_log_count = torch.zeros(
            (self.env.num_envs,), dtype=torch.long, device=gs.device
        )

    def reset(self, envs_idx: torch.Tensor):
        """
        Clear the action history for the specified environments.
        """
        if self._prev_action is None:
            return
        self._prev_action[envs_idx] = 0.0
        self._prev_prev_action[envs_idx] = 0.0
        self._action_log_count[envs_idx] = 0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        # Get the current actions for this step
        actions = env.actions
        if self.action_manager is not None:
            actions = self.action_manager.get_actions()

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


@dataclass(kw_only=True, eq=False)
class dof_torque_l2(MdpFn):
    """
    Penalize joint torque effort using the L2 squared kernel.

    Discourages the policy from applying unnecessary force, particularly when the
    robot is near equilibrium. This helps reduce actuator oscillation when the robot
    is stationary or moving slowly.

    Args:
        actuator_manager: The actuator manager to retrieve DOF forces from.

    Returns:
        torch.Tensor: Penalty for joint torque effort, shape (num_envs,)
    """

    actuator_manager: ActuatorManager

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        torque = self.actuator_manager.get_dofs_control_force()
        return torch.sum(torch.square(torque), dim=1)


@dataclass(kw_only=True, eq=False)
class dof_velocity_l2(MdpFn):
    """
    Penalize joint angular velocities to encourage slow, deliberate motion.

    Args:
        action_manager: The action manager to retrieve DOF velocities from.

    Returns:
        torch.Tensor: Penalty for joint angular velocity, shape (num_envs,)
    """

    action_manager: PositionActionManager

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        dof_vel = self.action_manager.get_dofs_velocity()
        return torch.sum(torch.square(dof_vel), dim=1)


"""
Velocity Command Rewards
"""

DEFAULT_TRACKING_SENSITIVITY = 0.25
"""The tracking reward sensitivity when there is no command range to derive one from"""


def _calculate_tracking_sensitivity(max_command: float) -> float:
    """
    Automatically calculate the tracking reward sensitivity for a command range: an error of half the maximum
    commanded speed decays the reward to 1/e. Falls back to the default for a zero range.
    """
    if max_command <= 0.0:
        return DEFAULT_TRACKING_SENSITIVITY
    return (0.5 * max_command) ** 2


@dataclass(kw_only=True, eq=False)
class command_tracking_lin_vel(MdpFn):
    """
    Reward for tracking commanded linear velocity (xy axes)

    Args:
        command: The commanded XY linear velocity in the shape (num_envs, 2)
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error
                     If not defined, the sensitivity will be derived from the command manager's range on every step
                     using the formulation: (0.5 * max_command) ** 2, where max_command is the largest commanded
                     speed the range allows (the norm of the largest absolute lin_vel_x and lin_vel_y values).
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Reward for tracking of linear velocity commands (xy axes)
    """

    command: torch.Tensor = None
    vel_cmd_manager: VelocityCommandManager = None
    sensitivity: float | None = None
    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def build(self):
        assert (
            self.command is not None or self.vel_cmd_manager is not None
        ), "Either command or vel_cmd_manager must be provided to command_tracking_lin_vel"

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            linear_vel_local = self.entity_manager.get_linear_velocity()
        else:
            robot = self.entity if self.entity is not None else env.robot
            linear_vel_local = entity_lin_vel(robot)

        command = self.command
        if self.vel_cmd_manager is not None:
            command = self.vel_cmd_manager.command[:, :2]

        lin_vel_error = torch.sum(
            torch.square(command - linear_vel_local[:, :2]), dim=1
        )
        sensitivity = self._get_sensitivity()
        return torch.exp(-lin_vel_error / sensitivity)

    def _get_sensitivity(self) -> float:
        """Get or calculate the sensitivity value"""
        if self.sensitivity is not None:
            return self.sensitivity

        # The largest linear speed in the command manager's current range. The error sums
        # the squares of both axes, so the fastest command is the diagonal one: the norm
        # of the largest speeds along each axis.
        max_speed = 0.0
        if self.vel_cmd_manager is not None:
            velocity_range = self.vel_cmd_manager.range
            max_speed = math.hypot(
                max(abs(v) for v in velocity_range["lin_vel_x"]),
                max(abs(v) for v in velocity_range["lin_vel_y"]),
            )
        return _calculate_tracking_sensitivity(max_speed)


@dataclass(kw_only=True, eq=False)
class command_tracking_ang_vel(MdpFn):
    """
    Reward for tracking commanded angular velocity (yaw)

    Args:
        commanded_ang_vel: The commanded angular velocity in the shape (num_envs, 1)
        vel_cmd_manager: The velocity command manager
        sensitivity: A lower value means the reward is more sensitive to the error
                     If not defined, the sensitivity will be derived from the command manager's range on every step
                     using the formulation: (0.5 * max_command) ** 2, where max_command is the largest absolute value of the command range.
        entity_manager: The entity manager for the robot/entity the reward is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the reward for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: Reward for tracking of angular velocity commands (yaw)
    """

    commanded_ang_vel: torch.Tensor = None
    vel_cmd_manager: VelocityCommandManager = None
    sensitivity: float | None = None
    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def build(self):
        assert (
            self.commanded_ang_vel is not None or self.vel_cmd_manager is not None
        ), "Either commanded_ang_vel or vel_cmd_manager must be provided to command_tracking_ang_vel"

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            angular_vel = self.entity_manager.get_angular_velocity()
        else:
            robot = self.entity if self.entity is not None else env.robot
            angular_vel = entity_ang_vel(robot)

        target = self.commanded_ang_vel
        if self.vel_cmd_manager is not None:
            target = self.vel_cmd_manager.command[:, 2]

        ang_vel_error = torch.square(target - angular_vel[:, 2])
        sensitivity = self._get_sensitivity()
        return torch.exp(-ang_vel_error / sensitivity)

    def _get_sensitivity(self) -> float:
        """Get or calculate the sensitivity value"""
        if self.sensitivity is not None:
            return self.sensitivity

        # The largest angular speed in the command manager's current range
        max_speed = 0.0
        if self.vel_cmd_manager is not None:
            values = [abs(v) for v in self.vel_cmd_manager.range["ang_vel_z"]]
            max_speed = max(values)
        return _calculate_tracking_sensitivity(max_speed)


@dataclass(kw_only=True, eq=False)
class stopped_joint_deviation_l1(MdpFn):
    """
    Penalize offsets from the default joint positions when the command is very small.

    Args:
        command_threshold: The threshold for the command to be considered small
        vel_cmd_manager: The velocity command manager
        actuator_manager: The actuator manager to get the joint positions and recent actions from.
        action_manager: The action manager to get the joint positions and recent actions from.

    Returns:
        torch.Tensor: Penalty for offsets from the default joint positions when the command is very small
    """

    vel_cmd_manager: VelocityCommandManager
    actuator_manager: ActuatorManager = None
    command_threshold: float = 0.06
    action_manager: PositionActionManager = None

    def build(self):
        assert (
            self.actuator_manager is not None or self.action_manager is not None
        ), "Either actuator_manager or action_manager must be provided to stopped_joint_deviation_l1"

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.actuator_manager is not None:
            joint_pos = self.actuator_manager.get_dofs_position()
            default_pos = self.actuator_manager.default_dofs_pos
        else:
            joint_pos = self.action_manager.get_dofs_position()
            default_pos = self.action_manager.default_dofs_pos
        joint_deviation = torch.sum(torch.abs(joint_pos - default_pos), dim=1)

        # Penalize motion when command is nearly zero.
        command = self.vel_cmd_manager.command
        return joint_deviation * (
            torch.norm(command[:, :2], dim=1) < self.command_threshold
        )


@deprecated(reason="Use 'stopped_joint_deviation_l1' instead")
@dataclass(kw_only=True, eq=False)
class stand_still_joint_deviation_l1(stopped_joint_deviation_l1):
    """Deprecated alias of :class:`stopped_joint_deviation_l1`."""


@dataclass(kw_only=True, eq=False)
class stopped_dof_velocity_l2(MdpFn):
    """
    Penalize joint velocities when the velocity command is stopped (no linear or angular velocity commanded),
    using the L2 squared kernel.

    Args:
        vel_cmd_manager: The velocity command manager
        actuator_manager: The actuator manager to get the DOF velocities from.
        command_threshold: The command is considered stopped when the norm of all its
                           components (linear xy and angular z) is below this value.

    Returns:
        torch.Tensor: Penalty for joint velocity while the command is stopped, shape (num_envs,)
    """

    vel_cmd_manager: VelocityCommandManager
    actuator_manager: ActuatorManager
    command_threshold: float = 0.01

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        dof_vel = self.actuator_manager.get_dofs_velocity()
        penalty = torch.sum(torch.square(dof_vel), dim=1)

        # Only penalize when nothing is commanded, linear or angular
        is_stopped = self.vel_cmd_manager.stopped_envs(self.command_threshold)
        return penalty * is_stopped


"""
Contacts
"""


@dataclass(kw_only=True, eq=False)
class has_contact(MdpFn):
    """
    One or more links in the contact manager are in contact with something.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0)
        min_contacts: The minimum number of contacts required. (default: 1)

    Returns:
        1 for each contact meeting the threshold
    """

    contact_manager: ContactManager
    threshold: float = 1.0
    min_contacts: int = 1

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        in_contact = self.contact_manager.contacts[:, :].norm(dim=-1) > self.threshold
        result = in_contact.sum(dim=1) >= self.min_contacts
        return result.float()


@dataclass(kw_only=True, eq=False)
class contact_force(MdpFn):
    """
    Reward for the total contact force acting on all the target links in the contact manager over the threshold.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The force threshold for contact detection (default: 1.0 N)

    Returns:
        The total force for the contact manager for each environment
    """

    contact_manager: ContactManager
    threshold: float = 1.0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        violation = (
            torch.norm(self.contact_manager.contacts[:, :, :], dim=-1) - self.threshold
        )
        return torch.sum(violation.clip(min=0.0), dim=1)


@dataclass(kw_only=True, eq=False)
class feet_air_time(MdpFn):
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the velocity commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.

    Args:
        contact_manager: The contact manager to check for contact
        time_threshold: The minimum time (in seconds) the feet should be in the air
        time_threshold_max: (optional) The maximum time (in seconds) the feet should be in the air.
                            The reward will be capped at this value.
        vel_cmd_manager: The velocity command manager

    Returns:
        The reward for the feet air time
    """

    contact_manager: ContactManager
    time_threshold: float
    time_threshold_max: float | None = None
    vel_cmd_manager: VelocityCommandManager | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        made_contact = self.contact_manager.has_made_contact(env.dt)
        last_air_time = self.contact_manager.last_air_time

        # Calculate the air time
        air_time = (last_air_time - self.time_threshold) * made_contact
        if self.time_threshold_max is not None:
            air_time = torch.clamp(
                air_time, max=self.time_threshold_max - self.time_threshold
            )
        reward = torch.sum(air_time, dim=1)

        # no reward for zero velocity command
        if self.vel_cmd_manager is not None:
            reward *= torch.norm(self.vel_cmd_manager.command[:, :2], dim=1) > 0.1
        return reward


@dataclass(kw_only=True, eq=False)
class feet_ground_time(MdpFn):
    """Penalize brief ground contacts (foot tapping) using a linear kernel.

    Fires at the moment a foot lifts off. The penalty is proportional to how
    much the stance duration fell below time_threshold. A proper stance phase
    (contact_time >= time_threshold) produces zero penalty.

    Intended to be paired with feet_air_time (positive reward) to fully shape
    gait timing: feet_air_time rewards long swings while this penalizes taps.
    Use a negative weight in the RewardManager.

    Args:
        contact_manager: The contact manager to check for contact
        time_threshold: Contacts shorter than this (in seconds) are penalized.
                        Set independently from the feet_air_time threshold based
                        on the expected stance duration of your target gait.

    Returns:
        The penalty for brief ground contacts, shape (num_envs,)
    """

    contact_manager: ContactManager
    time_threshold: float

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        just_lifted = self.contact_manager.has_broken_contact(env.dt)
        last_contact_time = self.contact_manager.last_contact_time
        short_contact = (self.time_threshold - last_contact_time).clamp(
            min=0.0
        ) * just_lifted
        return torch.sum(short_contact, dim=1)


@dataclass(kw_only=True, eq=False)
class feet_slide(MdpFn):
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.

    This penalty is less effective at longer foot-contact links (for example, long legs without dedicated foot links),
    because they might have some velocity while they're being used to move the robot. However, dedicated foot links
    will be stationary on the ground and not moving while pushing the robot forward.

    Args:
        contact_manager: The contact manager for the feet
        entity: The robot entity that the feet are attached to. Defaults to `env.robot`.

    Returns:
        The penalty for the feet slide
    """

    contact_manager: ContactManager
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        # Get links in contact
        contacts = torch.norm(self.contact_manager.contacts[:, :, :], dim=-1) > 1.0

        # Get link velocities.
        # If the links aren't moving, then they're being used to move the robot and not sliding.
        link_ids = self.contact_manager.local_link_ids
        robot: RigidEntity = self.entity if self.entity is not None else env.robot
        link_vel = robot.get_links_vel(links_idx_local=link_ids)

        return torch.sum(link_vel.norm(dim=-1) * contacts, dim=1)
