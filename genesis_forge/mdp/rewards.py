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
    BaseActionManager,
    CommandManager,
    ContactManager,
    EntityManager,
    MdpFn,
    Pose2dCommand,
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

    Args:
        action_manager: Only count the actions belonging to this action manager, instead of
                        every action the policy produces. Use this when part of the robot is
                        *meant* to keep moving -- a sensor being swept around to look for
                        obstacles, say -- where penalizing its changes works against the
                        behavior you are trying to get. Defaults to None: every action counts.

    Returns:
        torch.Tensor: Penalty for changes in actions, shape (num_envs,)
    """

    action_manager: BaseActionManager = None

    def build(self):
        self._action_slice = slice(None)
        if self.action_manager is not None:
            self._action_slice = self._find_action_slice()

    def _find_action_slice(self) -> slice:
        """
        Which part of the environment's action vector belongs to `action_manager`.

        The environment hands each action manager its own slice of the policy's output,
        in the order the managers were created, so the slice is found by counting past
        the managers created before this one.
        """
        start = 0
        for manager in self.env.managers["action"]:
            if manager is self.action_manager:
                return slice(start, start + manager.num_actions)
            start += manager.num_actions
        raise ValueError(
            "The action_manager passed to action_rate_l2 is not registered with this "
            "environment, so there is no way to tell which actions belong to it."
        )

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        last_actions = env.last_actions
        if last_actions is None:
            return torch.zeros(env.num_envs, device=gs.device)
        change = (
            last_actions[:, self._action_slice] - env.actions[:, self._action_slice]
        )
        return torch.sum(torch.square(change), dim=1)


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
Pose Command Rewards
"""


@dataclass(kw_only=True, eq=False)
class position_tracking(MdpFn):
    """
    Reward for being close to the commanded goal position.

    This is a dense reward that grows as the entity nears its goal. Pair it with
    :class:`position_progress` to also reward moving toward the goal, which gives the
    policy a signal even when it is still far away.

    !!! warning "Not for goals that are replaced on arrival"
        This pays every step for *being* near the goal, so an entity can earn it by
        parking just outside the reach threshold. If the command manager replaces the
        goal on arrival (`resample_on_reached`), that parking spot can easily be worth
        more than arriving -- the entity keeps the reward instead of trading it for one
        bonus and a goal that jumps out of reach. In that setup prefer
        :class:`position_progress`, which pays for closing the distance and so pays an
        entity that stands still exactly nothing.

    Args:
        pose_cmd_manager: The pose command manager holding the goal position.
        sensitivity: A lower value means the reward is more sensitive to the distance.
                     If not defined, the sensitivity will be derived from the command manager's range on every step
                     using the formulation: (0.5 * max_command) ** 2, where max_command is the
                     furthest goal distance in the command range.

    Returns:
        torch.Tensor: Reward for proximity to the goal position, shape (num_envs,)
    """

    pose_cmd_manager: Pose2dCommand
    sensitivity: float | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        distance = self.pose_cmd_manager.distance_to_goal
        return torch.exp(-torch.square(distance) / self._get_sensitivity())

    def _get_sensitivity(self) -> float:
        """Get or calculate the sensitivity value"""
        if self.sensitivity is not None:
            return self.sensitivity

        # The furthest goal that can be sampled from the command manager's current range
        position_range = self.pose_cmd_manager.range
        max_x = max(abs(v) for v in position_range["x"])
        max_y = max(abs(v) for v in position_range["y"])
        return _calculate_tracking_sensitivity(math.hypot(max_x, max_y))


@dataclass(kw_only=True, eq=False)
class heading_tracking(MdpFn):
    """
    Reward for facing the direction the goal pose asks for, as the entity comes in to land.

    The reward fades out with distance from the goal, so it is only really worth earning
    on the final approach. Without that fade, facing the right way pays just as well from
    across the map as it does at the goal, and the cheapest way to collect it is to stop
    and turn on the spot instead of driving on -- which fights the rewards for closing
    the distance. Fading it out means the only way to collect the heading reward is to
    get to the goal first, and the entity is encouraged to arrive already lined up.

    !!! warning "Not for goals that are replaced on arrival"
        The fade limits *where* this can be collected from, but it still pays every step
        for *being* lined up, so an entity can park near the goal and hold the reward
        rather than arrive. Where the goal is replaced on arrival
        (`resample_on_reached`), prefer :class:`heading_progress`, which pays for turning
        toward the goal heading and so pays an entity that stands still nothing at all.

    Args:
        pose_cmd_manager: The pose command manager holding the goal heading.
        sensitivity: A lower value means the reward is more sensitive to the heading error.
                     If not defined, it is derived the same way as :class:`position_tracking`,
                     from the furthest the entity could ever be from the goal heading.
        matters_within: The distance (in meters) from the goal over which the heading starts
                        to matter. Defaults to three times the command manager's
                        `goal_reached_threshold`: near enough to the goal that the entity is
                        lining up to arrive, rather than still travelling.

    Returns:
        torch.Tensor: Reward for facing the goal heading near the goal, shape (num_envs,)
    """

    pose_cmd_manager: Pose2dCommand
    sensitivity: float | None = None
    matters_within: float | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        heading_error = self.pose_cmd_manager.heading_error
        facing_the_right_way = torch.exp(
            -torch.square(heading_error) / self._get_sensitivity()
        )

        # Fade the reward out with distance, so heading only pays near the goal
        distance = self.pose_cmd_manager.distance_to_goal
        nearness = torch.exp(-torch.square(distance) / self._get_fade_distance() ** 2)

        return facing_the_right_way * nearness

    def _get_sensitivity(self) -> float:
        """Get or calculate the sensitivity value"""
        if self.sensitivity is not None:
            return self.sensitivity

        # However the heading is commanded, the entity is never more than half a turn
        # away from it, since it turns whichever way is closer
        return _calculate_tracking_sensitivity(math.pi)

    def _get_fade_distance(self) -> float:
        """Get or calculate the distance over which the heading reward fades out"""
        if self.matters_within is not None:
            return self.matters_within
        return 3.0 * self.pose_cmd_manager.goal_reached_threshold


@dataclass(kw_only=True, eq=False)
class position_progress(MdpFn):
    """
    Reward for closing the distance to the commanded goal position, measured as the
    speed (m/s) at which the entity is approaching its goal. Moving away is penalized.

    Steps that don't have a valid distance to compare against are skipped: the first
    step of an episode, and any step where the goal was resampled.

    Args:
        pose_cmd_manager: The pose command manager holding the goal position.

    Returns:
        torch.Tensor: Approach speed toward the goal, shape (num_envs,)
    """

    pose_cmd_manager: Pose2dCommand

    def build(self):
        self._prev_distance = torch.zeros(self.env.num_envs, device=gs.device)
        self._has_prev_distance = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=gs.device
        )

    def reset(self, envs_idx: torch.Tensor):
        self._has_prev_distance[envs_idx] = False

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        distance = self.pose_cmd_manager.distance_to_goal

        # The previous distance was measured against a different goal, so it can't be compared
        self._has_prev_distance &= ~self.pose_cmd_manager.resampled_last_step

        progress = (self._prev_distance - distance) / env.dt
        progress = progress * self._has_prev_distance

        self._prev_distance[:] = distance
        self._has_prev_distance[:] = True

        return progress


@dataclass(kw_only=True, eq=False)
class heading_progress(MdpFn):
    """
    Reward for turning the right way, measured as the speed (rad/s) at which the entity
    is closing the angle. Turning the wrong way is penalized.

    This is the heading counterpart to :class:`position_progress`, and like it, pays for
    *changing* rather than for *being*: an entity sitting still earns exactly nothing,
    however well it is lined up. Over a whole goal it can only ever add up to the angle
    it started with, so there is no way to farm it by turning back and forth.

    Which angle counts depends on `lines_up_within`. By default it is always the goal
    heading -- the way to face on arrival. That is the natural choice for something that
    can travel in one direction while facing another, like a legged or omnidirectional
    robot. Anything that has to point where it is going, like a car or a differential
    drive robot, cannot chase the goal heading from far away without driving sideways to
    reach the goal, which it physically cannot do. Setting `lines_up_within` makes the
    reward ask for the bearing while there is still ground to cover, and hand over to the
    goal heading on the final approach.

    Steps that don't have a previous angle to compare against are skipped: the first step
    of an episode, and any step where the goal was resampled.

    Args:
        pose_cmd_manager: The pose command manager holding the goal pose.
        lines_up_within: How close to the goal (in meters) the entity should stop steering
                         toward the goal and start lining up with the goal heading. The
                         changeover is gradual, so the reward doesn't jump as the entity
                         closes in. Defaults to None: the goal heading is asked for at
                         every distance.

    Returns:
        torch.Tensor: Turning speed toward the angle being asked for, shape (num_envs,)
    """

    pose_cmd_manager: Pose2dCommand
    lines_up_within: float | None = None

    def build(self):
        self._prev_error = torch.zeros(self.env.num_envs, device=gs.device)
        self._has_prev_error = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=gs.device
        )

    def reset(self, envs_idx: torch.Tensor):
        self._has_prev_error[envs_idx] = False

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        error = self._tracked_error()

        # The previous error was measured against a different goal, so it can't be compared
        self._has_prev_error &= ~self.pose_cmd_manager.resampled_last_step

        progress = (self._prev_error - error) / env.dt
        progress = progress * self._has_prev_error

        self._prev_error[:] = error
        self._has_prev_error[:] = True

        return progress

    def _tracked_error(self) -> torch.Tensor:
        """
        The angle the entity is being asked to close, in radians.

        With `lines_up_within` set, the entity is rewarded for turning to face the goal
        position while it is far away, and for turning into the goal heading as it
        closes in. The two are blended by distance rather than switched at a threshold,
        so the reward stays smooth on the approach.
        """
        heading_error = self.pose_cmd_manager.heading_error.abs()
        if self.lines_up_within is None:
            return heading_error

        bearing_error = self.pose_cmd_manager.bearing_error.abs()
        distance = self.pose_cmd_manager.distance_to_goal

        # 1 at the goal, fading to 0 well beyond `lines_up_within`
        lining_up = torch.exp(-torch.square(distance) / self.lines_up_within**2)
        return lining_up * heading_error + (1.0 - lining_up) * bearing_error


@dataclass(kw_only=True, eq=False)
class reached_goal(MdpFn):
    """
    Reward for reaching the commanded goal position.

    This is a sparse bonus, paid on each step the entity is within the threshold of its
    goal. When the command manager is configured to resample on reach, this is paid once
    per goal, since the goal is replaced immediately after the rewards are computed.

    Args:
        pose_cmd_manager: The pose command manager holding the goal pose.
        threshold: Pay the bonus within this distance (in meters) of the goal, ignoring the
                   goal heading. Defaults to None: the bonus is paid whenever the command
                   manager itself counts the goal as reached, which is also when it hands
                   out a new one.

    Returns:
        torch.Tensor: 1.0 for each environment that has reached its goal, shape (num_envs,)
    """

    pose_cmd_manager: Pose2dCommand
    threshold: float | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.threshold is None:
            return self.pose_cmd_manager.goal_reached.float()
        return (self.pose_cmd_manager.distance_to_goal < self.threshold).float()


@dataclass(kw_only=True, eq=False)
class keep_clear(MdpFn):
    """
    Penalty for crowding the things the entity is supposed to keep away from, or move around.
    The penalty growing from nothing at `clearance` to its full value on contact.

    A collision termination only tells the entity it got something wrong once it is too
    late to do anything about it. This gives it a gradient to follow on the way in, so it
    can learn to leave room rather than only to regret not having done so.

    Only the nearest obstacle counts. Threading a gap between two obstacles is no worse
    than passing one at the same distance -- what matters is the closest thing, not how
    many things are around.

    !!! note "This deliberately reads the true distance, not a sensor"

        The penalty uses the actual positions from the simulation, which a real robot
        could not know. That is fine and usual for a reward -- rewards are free to use
        information the observation withholds -- but it does mean the entity is being
        asked to avoid things it may not be able to see. If it is crashing into obstacles
        that never enter its sensor's view, the fix is the observation, not this.

    Args:
        entities: The entities to keep clear of.
        clearance: The distance (in meters, centre to centre) at which the penalty starts.
        entity: The entity being kept clear of them. Defaults to `env.robot`.
        entity_manager: The entity manager for the above, which is slightly faster than
                        passing `entity` since it reads a position cached once per step
                        instead of querying the simulator.

    Returns:
        torch.Tensor: 0.0 when further than `clearance` from everything, rising toward 1.0
                      as the nearest obstacle is approached, shape (num_envs,)
    """

    entities: list[RigidEntity]
    clearance: float = 0.5
    entity: RigidEntity = None
    entity_manager: EntityManager = None

    def build(self):
        if (
            self.entity is None
            and self.entity_manager is None
            and self.env.robot is None
        ):
            raise ValueError(
                "keep_clear: no entity to compute the reward for -- pass entity or "
                "entity_manager, or set env.robot"
            )

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if not self.entities:
            return torch.zeros(env.num_envs, device=gs.device)

        if self.entity_manager is not None:
            entity_xy = self.entity_manager.base_pos[:, :2]
        else:
            entity = self.entity if self.entity is not None else env.robot
            entity_xy = entity.get_pos()[:, :2]

        # Stack every obstacle's position into one tensor so the distance to all of
        # them, and the nearest one, are each a single reduction.
        obstacles_xy = torch.stack(
            [obstacle.get_pos()[:, :2] for obstacle in self.entities], dim=0
        )
        distance = torch.norm(obstacles_xy - entity_xy, dim=-1)
        nearest_distance = distance.min(dim=0).values

        # 0 at `clearance` and beyond, rising to 1 as the nearest obstacle closes to nothing
        return (1.0 - nearest_distance / self.clearance).clamp(min=0.0)


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
