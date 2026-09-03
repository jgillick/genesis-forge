"""
Numerical behavior of the reward functions in genesis_forge.mdp.rewards.

Training runs are stochastic, so these pin each function against hand-computed
values instead. Entities and managers are faked -- no Genesis scene is built.
"""

import math

import pytest
import torch

from genesis_forge.managers.config import ConfigItem
from genesis_forge.mdp import rewards


class FakeRobot:
    def __init__(self, pos):
        self._pos = pos

    def get_pos(self):
        return self._pos


class FakeEntityManager:
    def __init__(self, lin_vel=None, ang_vel=None, projected_gravity=None):
        self._lin = lin_vel
        self._ang = ang_vel
        self._gravity = projected_gravity

    def get_linear_velocity(self):
        return self._lin

    def get_angular_velocity(self):
        return self._ang

    def get_projected_gravity(self):
        return self._gravity


class FakeVelCmd:
    def __init__(self, command, range=None):
        self.command = command
        self.range = range or {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-1.0, 1.0),
        }

    def stopped_envs(self, threshold: float = 0.01) -> torch.Tensor:
        return torch.norm(self.command, dim=1) <= threshold


class FakeContactManager:
    def __init__(self, contacts):
        self.contacts = contacts


"""
Aliveness
"""


def test_is_alive_rewards_envs_that_did_not_terminate(env):
    env.extras["terminations"] = torch.tensor([False, True, False])
    fn = rewards.is_alive()
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([1.0, 0.0, 1.0]))


def test_terminated_penalizes_envs_that_did_terminate(env):
    env.extras["terminations"] = torch.tensor([False, True, False])
    fn = rewards.terminated()
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([0.0, 1.0, 0.0]))


"""
base_height
"""


def test_base_height_squares_the_offset(env):
    env.robot = FakeRobot(torch.tensor([[0.0, 0.0, 0.35], [0.0, 0.0, 0.25]]))
    fn = rewards.base_height(target_height=0.3)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.0025, 0.0025]), atol=1e-6)


def test_base_height_responds_to_a_param_change(env):
    env.robot = FakeRobot(torch.tensor([[0.0, 0.0, 0.35], [0.0, 0.0, 0.25]]))
    fn = rewards.base_height(target_height=0.3)
    fn.context(env)
    fn.safe_build()

    fn.target_height = 0.25

    assert torch.allclose(fn(env), torch.tensor([0.01, 0.0]), atol=1e-6)


def test_base_height_uses_entity_manager_when_given(env):
    class Mgr:
        entity = FakeRobot(torch.tensor([[0.0, 0.0, 0.5]]))

    fn = rewards.base_height(target_height=0.3, entity_manager=Mgr())
    fn.context(env)
    fn.safe_build()
    assert torch.allclose(fn(env), torch.tensor([0.04]), atol=1e-6)


"""
dof_similar_to_default
"""


class FakeActuatorManager:
    def __init__(self, pos, default_pos):
        self._pos = pos
        self.default_dofs_pos = default_pos

    def get_dofs_position(self):
        return self._pos


def test_dof_similar_to_default_sums_abs_offset_from_default(env):
    actuator = FakeActuatorManager(
        pos=torch.tensor([[0.5, -0.2]]), default_pos=torch.tensor([0.0, 0.0])
    )
    fn = rewards.dof_similar_to_default(actuator_manager=actuator)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.7]))


def test_dof_similar_to_default_sums_across_a_list_of_actuator_managers(env):
    a = FakeActuatorManager(pos=torch.tensor([[0.5]]), default_pos=torch.tensor([0.0]))
    b = FakeActuatorManager(pos=torch.tensor([[-0.3]]), default_pos=torch.tensor([0.0]))
    fn = rewards.dof_similar_to_default(actuator_manager=[a, b])
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.8]))


def test_dof_similar_to_default_requires_a_manager(env):
    with pytest.raises(TypeError, match="actuator_manager"):
        rewards.dof_similar_to_default()


def test_dof_similar_to_default_rejects_an_explicit_none_at_build_time(env):
    """actuator_manager is a required constructor arg, but nothing stops someone from
    explicitly passing None -- build() still catches that case."""
    fn = rewards.dof_similar_to_default(actuator_manager=None)
    fn.context(env)
    with pytest.raises(ValueError, match="actuator_manager must be provided"):
        fn.safe_build()


"""
lin_vel_z_l2 / lin_vel_xy_l2 / ang_vel_xy_l2 / flat_orientation_l2
"""


def test_lin_vel_z_l2_squares_the_z_component(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.0, 0.0, 0.4]]))
    fn = rewards.lin_vel_z_l2(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.16]))


def test_lin_vel_xy_l2_sums_the_squared_xy_components(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.3, 0.4, 9.0]]))
    fn = rewards.lin_vel_xy_l2(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.25]))


def test_ang_vel_xy_l2_sums_the_squared_xy_components(env):
    mgr = FakeEntityManager(ang_vel=torch.tensor([[0.3, 0.4, 9.0]]))
    fn = rewards.ang_vel_xy_l2(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.25]))


def test_flat_orientation_l2_sums_the_squared_xy_tilt(env):
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.3, 0.4, -0.9]]))
    fn = rewards.flat_orientation_l2(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.25]))


"""
body_acceleration_exp
"""


def test_body_acceleration_exp_is_zero_penalty_on_the_first_call(env):
    mgr = FakeEntityManager(
        lin_vel=torch.tensor([[1.0, 0.0, 0.0]]), ang_vel=torch.tensor([[0.0, 0.0, 0.0]])
    )
    fn = rewards.body_acceleration_exp(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.0]))


def test_body_acceleration_exp_penalizes_the_change_since_the_last_call(env):
    mgr = FakeEntityManager(
        lin_vel=torch.tensor([[0.0, 0.0, 0.0]]), ang_vel=torch.tensor([[0.0, 0.0, 0.0]])
    )
    fn = rewards.body_acceleration_exp(entity_manager=mgr, sensitivity=0.1)
    fn.context(env)
    fn.safe_build()
    fn(env)  # first call establishes the zero-velocity baseline

    mgr._lin = torch.tensor([[env.dt, 0.0, 0.0]])  # 1 m/s^2 over one step
    value = fn(env)

    # lin_acc norm = 1.0, ang_acc norm = 0.0 -> motion = 1.0
    assert torch.allclose(value, torch.tensor([1 - math.exp(-0.1)]), atol=1e-6)


"""
Action penalties
"""


def test_action_rate_l2_is_zero_with_no_last_actions(env):
    env.actions = torch.tensor([[0.1, 0.2]])
    env.last_actions = None
    fn = rewards.action_rate_l2()
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.zeros_like(env.actions))


def test_action_rate_l2_sums_the_squared_change_from_the_last_step(env):
    env.actions = torch.tensor([[0.5, 0.2]])
    env.last_actions = torch.tensor([[0.3, 0.2]])
    fn = rewards.action_rate_l2()
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.04]))


def test_action_acceleration_masks_until_two_steps_of_history(env):
    env.actions = torch.zeros((env.num_envs, 3))
    fn = rewards.action_acceleration_l2()
    fn.context(env)
    fn.safe_build()

    # A smooth ramp: zero acceleration, but the first two steps are masked anyway.
    assert torch.equal(fn(env), torch.zeros(env.num_envs))
    env.actions = torch.full((env.num_envs, 3), 0.1)
    assert torch.equal(fn(env), torch.zeros(env.num_envs))


def test_action_acceleration_penalises_direction_reversal(env):
    fn = rewards.action_acceleration_l2()
    fn.context(env)
    fn.safe_build()

    for actions in ([0.5] * 3, [0.8] * 3, [0.5] * 3):
        env.actions = torch.tensor([actions] * env.num_envs)
        value = fn(env)

    # acc = 0.5 - 2(0.8) + 0.5 = -0.6 per dim, summed over 3 dims
    assert torch.allclose(value, torch.full((env.num_envs,), 3 * 0.36), atol=1e-6)


def test_action_acceleration_reset_clears_history(env):
    fn = rewards.action_acceleration_l2()
    fn.context(env)
    fn.safe_build()

    for actions in ([0.5] * 3, [0.8] * 3):
        env.actions = torch.tensor([actions] * env.num_envs)
        fn(env)

    fn.reset(torch.tensor([0, 1]))

    env.actions = torch.tensor([[0.5] * 3] * env.num_envs)
    value = fn(env)
    assert torch.equal(value[:2], torch.zeros(2)), "reset envs are masked again"
    assert torch.all(value[2:] > 0), "untouched envs keep their history"


def test_dof_torque_l2_sums_the_squared_control_force(env):
    class FakeActuatorManagerTorque:
        def get_dofs_control_force(self):
            return torch.tensor([[3.0, 4.0]])

    fn = rewards.dof_torque_l2(actuator_manager=FakeActuatorManagerTorque())
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([25.0]))


def test_dof_velocity_l2_sums_the_squared_dof_velocity(env):
    class FakeActionManagerVel:
        def get_dofs_velocity(self):
            return torch.tensor([[1.0, 2.0]])

    fn = rewards.dof_velocity_l2(action_manager=FakeActionManagerVel())
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([5.0]))


"""
Velocity Command Rewards
"""


def test_lin_vel_tracking_is_one_when_command_is_met(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[1.0, 0.0, 0.0]]))
    fn = rewards.command_tracking_lin_vel(
        vel_cmd_manager=FakeVelCmd(torch.tensor([[1.0, 0.0, 0.0]])),
        entity_manager=mgr,
    )
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([1.0]), atol=1e-6)


def test_lin_vel_tracking_decays_with_error(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.5, 0.0, 0.0]]))
    fn = rewards.command_tracking_lin_vel(
        vel_cmd_manager=FakeVelCmd(torch.tensor([[1.0, 0.0, 0.0]])),
        entity_manager=mgr,
        sensitivity=0.25,
    )
    fn.context(env)
    fn.safe_build()

    # error = 0.25, exp(-0.25 / 0.25) = exp(-1)
    assert torch.allclose(fn(env), torch.tensor([0.36787944]), atol=1e-6)


def test_lin_vel_sensitivity_is_param_adjustable(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.5, 0.0, 0.0]]))
    fn = rewards.command_tracking_lin_vel(
        vel_cmd_manager=FakeVelCmd(torch.tensor([[1.0, 0.0, 0.0]])),
        entity_manager=mgr,
        sensitivity=0.25,
    )
    fn.context(env)
    fn.safe_build()

    fn.sensitivity = 0.5  # error 0.25 -> exp(-0.5)

    assert torch.allclose(fn(env), torch.tensor([0.60653066]), atol=1e-6)


def test_ang_vel_tracking_decays_with_error(env):
    mgr = FakeEntityManager(ang_vel=torch.tensor([[0.0, 0.0, 0.5]]))
    fn = rewards.command_tracking_ang_vel(
        vel_cmd_manager=FakeVelCmd(torch.tensor([[0.0, 0.0, 1.0]])),
        entity_manager=mgr,
        sensitivity=0.25,
    )
    fn.context(env)
    fn.safe_build()

    # error = 0.25, exp(-0.25 / 0.25) = exp(-1)
    assert torch.allclose(fn(env), torch.tensor([0.36787944]), atol=1e-6)


def test_lin_vel_sensitivity_is_derived_from_the_command_range(env):
    """Without `sensitivity`, half the max commanded speed is the 1/e error."""
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.05, 0.0, 0.0]]))
    vel_cmd = FakeVelCmd(
        torch.tensor([[0.1, 0.0, 0.0]]),
        range={"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
    )
    fn = rewards.command_tracking_lin_vel(vel_cmd_manager=vel_cmd, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    # sensitivity = (0.5 * 0.1)^2 = 0.0025; error = 0.05^2 = 0.0025 -> exp(-1)
    assert torch.allclose(fn(env), torch.tensor([0.36787944]), atol=1e-6)


def test_derived_sensitivity_follows_a_range_change(env):
    """A curriculum widening the command range loosens the reward on the next call."""
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.05, 0.0, 0.0]]))
    vel_cmd = FakeVelCmd(
        torch.tensor([[0.1, 0.0, 0.0]]),
        range={"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
    )
    fn = rewards.command_tracking_lin_vel(vel_cmd_manager=vel_cmd, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()
    before = fn(env)

    vel_cmd.range["lin_vel_x"] = (-0.2, 0.2)  # sensitivity -> 0.01, same error -> exp(-0.25)
    assert torch.allclose(fn(env), torch.tensor([0.77880078]), atol=1e-6)
    assert fn(env) > before


def test_ang_vel_sensitivity_is_derived_from_the_command_range(env):
    mgr = FakeEntityManager(ang_vel=torch.tensor([[0.0, 0.0, 0.25]]))
    vel_cmd = FakeVelCmd(
        torch.tensor([[0.0, 0.0, 0.5]]),
        range={"lin_vel_x": (-0.1, 0.1), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)},
    )
    fn = rewards.command_tracking_ang_vel(vel_cmd_manager=vel_cmd, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    # sensitivity = (0.5 * 0.5)^2 = 0.0625; error = 0.25^2 = 0.0625 -> exp(-1)
    assert torch.allclose(fn(env), torch.tensor([0.36787944]), atol=1e-6)


def test_sensitivity_falls_back_to_default_without_a_command_manager(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[0.5, 0.0, 0.0]]))
    fn = rewards.command_tracking_lin_vel(command=torch.tensor([[1.0, 0.0]]), entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    # error = 0.25, default sensitivity 0.25 -> exp(-1)
    assert torch.allclose(fn(env), torch.tensor([0.36787944]), atol=1e-6)


def test_sensitivity_falls_back_to_default_for_a_zero_range(env):
    mgr = FakeEntityManager(ang_vel=torch.tensor([[0.0, 0.0, 0.5]]))
    vel_cmd = FakeVelCmd(
        torch.tensor([[0.0, 0.0, 0.0]]),
        range={"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (0.0, 0.0)},
    )
    fn = rewards.command_tracking_ang_vel(vel_cmd_manager=vel_cmd, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    # error = 0.25, default sensitivity 0.25 -> exp(-1), no division by zero
    assert torch.allclose(fn(env), torch.tensor([0.36787944]), atol=1e-6)


@pytest.mark.parametrize(
    "factory",
    [rewards.command_tracking_lin_vel, rewards.command_tracking_ang_vel],
)
def test_command_tracking_requires_a_command_source(env, factory):
    """The missing-command-source assertion fires at build time, not on the first step."""
    fn = factory()
    fn.context(env)
    with pytest.raises(AssertionError):
        fn.safe_build()


def test_stopped_joint_deviation_penalizes_only_below_command_threshold(env):
    actuator = FakeActuatorManager(
        pos=torch.tensor([[0.5, -0.2], [0.5, -0.2]]),
        default_pos=torch.tensor([0.0, 0.0]),
    )
    vel_cmd = FakeVelCmd(torch.tensor([[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    fn = rewards.stopped_joint_deviation_l1(
        actuator_manager=actuator, vel_cmd_manager=vel_cmd, command_threshold=0.06
    )
    fn.context(env)
    fn.safe_build()

    # env 0's command is below threshold (penalized); env 1's is above (masked to zero)
    assert torch.allclose(fn(env), torch.tensor([0.7, 0.0]))


def test_stopped_joint_deviation_requires_a_manager_at_build_time(env):
    fn = rewards.stopped_joint_deviation_l1(vel_cmd_manager=FakeVelCmd(torch.zeros((1, 3))))
    fn.context(env)
    with pytest.raises(AssertionError, match="actuator_manager or action_manager"):
        fn.safe_build()


def test_stand_still_joint_deviation_is_a_deprecated_alias(env):
    actuator = FakeActuatorManager(
        pos=torch.tensor([[0.5, -0.2]]),
        default_pos=torch.tensor([0.0, 0.0]),
    )
    vel_cmd = FakeVelCmd(torch.tensor([[0.01, 0.0, 0.0]]))
    with pytest.deprecated_call():
        fn = rewards.stand_still_joint_deviation_l1(
            actuator_manager=actuator, vel_cmd_manager=vel_cmd
        )
    assert isinstance(fn, rewards.stopped_joint_deviation_l1)
    fn.context(env)
    fn.safe_build()
    assert torch.allclose(fn(env), torch.tensor([0.7]))


def test_stopped_dof_velocity_penalizes_only_when_the_command_is_stopped(env):
    class FakeActuatorManagerVel:
        def get_dofs_velocity(self):
            return torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])

    # env 0: fully stopped, env 1: linear command, env 2: angular-only command
    vel_cmd = FakeVelCmd(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.2]])
    )
    fn = rewards.stopped_dof_velocity_l2(
        vel_cmd_manager=vel_cmd, actuator_manager=FakeActuatorManagerVel()
    )
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([5.0, 0.0, 0.0]))


def test_stopped_dof_velocity_requires_an_actuator_manager(env):
    with pytest.raises(TypeError, match="actuator_manager"):
        rewards.stopped_dof_velocity_l2(vel_cmd_manager=FakeVelCmd(torch.zeros((1, 3))))


"""
Contacts
"""


def test_has_contact_rewards_envs_with_enough_contacts(env):
    contacts = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    fn = rewards.has_contact(
        contact_manager=FakeContactManager(contacts), threshold=1.0, min_contacts=2
    )
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([1.0, 0.0]))


def test_contact_force_sums_the_over_threshold_violation(env):
    contacts = torch.tensor([[[3.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    fn = rewards.contact_force(contact_manager=FakeContactManager(contacts), threshold=1.0)
    fn.context(env)
    fn.safe_build()

    # norms: 3.0, 0.1 -> violations (clipped at 0): 2.0, 0.0 -> sum 2.0
    assert torch.allclose(fn(env), torch.tensor([2.0]))


class FakeGaitContactManager:
    def __init__(
        self,
        made_contact=None,
        last_air_time=None,
        broke_contact=None,
        last_contact_time=None,
        contacts=None,
        local_link_ids=None,
    ):
        self._made_contact = made_contact
        self.last_air_time = last_air_time
        self._broke_contact = broke_contact
        self.last_contact_time = last_contact_time
        self.contacts = contacts
        self.local_link_ids = local_link_ids

    def has_made_contact(self, dt):
        return self._made_contact

    def has_broken_contact(self, dt):
        return self._broke_contact


def test_feet_air_time_rewards_long_swings_above_threshold(env):
    mgr = FakeGaitContactManager(
        made_contact=torch.tensor([[True, False]]),
        last_air_time=torch.tensor([[0.5, 0.5]]),
    )
    fn = rewards.feet_air_time(contact_manager=mgr, time_threshold=0.2)
    fn.context(env)
    fn.safe_build()

    # only the foot that just made contact counts: 0.5 - 0.2 = 0.3
    assert torch.allclose(fn(env), torch.tensor([0.3]))


def test_feet_air_time_clamps_to_the_max_threshold(env):
    mgr = FakeGaitContactManager(
        made_contact=torch.tensor([[True]]),
        last_air_time=torch.tensor([[2.0]]),
    )
    fn = rewards.feet_air_time(
        contact_manager=mgr, time_threshold=0.2, time_threshold_max=0.5
    )
    fn.context(env)
    fn.safe_build()

    # (2.0 - 0.2) clamped to (0.5 - 0.2) = 0.3
    assert torch.allclose(fn(env), torch.tensor([0.3]))


def test_feet_air_time_is_zeroed_when_command_is_near_zero(env):
    mgr = FakeGaitContactManager(
        made_contact=torch.tensor([[True]]),
        last_air_time=torch.tensor([[0.5]]),
    )
    vel_cmd = FakeVelCmd(torch.tensor([[0.01, 0.0, 0.0]]))
    fn = rewards.feet_air_time(
        contact_manager=mgr, time_threshold=0.2, vel_cmd_manager=vel_cmd
    )
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.0]))


def test_feet_ground_time_penalizes_stances_shorter_than_threshold(env):
    mgr = FakeGaitContactManager(
        broke_contact=torch.tensor([[True, True]]),
        last_contact_time=torch.tensor([[0.1, 0.5]]),
    )
    fn = rewards.feet_ground_time(contact_manager=mgr, time_threshold=0.3)
    fn.context(env)
    fn.safe_build()

    # foot 1: (0.3 - 0.1) = 0.2 penalty; foot 2's stance was long enough -> 0
    assert torch.allclose(fn(env), torch.tensor([0.2]))


def test_feet_ground_time_ignores_feet_that_have_not_lifted(env):
    mgr = FakeGaitContactManager(
        broke_contact=torch.tensor([[False]]),
        last_contact_time=torch.tensor([[0.05]]),
    )
    fn = rewards.feet_ground_time(contact_manager=mgr, time_threshold=0.3)
    fn.context(env)
    fn.safe_build()

    assert torch.allclose(fn(env), torch.tensor([0.0]))


def test_feet_slide_penalizes_moving_feet_that_are_in_contact(env):
    contacts = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # foot0 in contact, foot1 not
    mgr = FakeGaitContactManager(contacts=contacts, local_link_ids=[0, 1])

    class FakeRobotLinks:
        def get_links_vel(self, links_idx_local):
            assert links_idx_local == [0, 1]
            return torch.tensor([[[0.3, 0.4, 0.0], [9.0, 9.0, 0.0]]])

    env.robot = FakeRobotLinks()
    fn = rewards.feet_slide(contact_manager=mgr)
    fn.context(env)
    fn.safe_build()

    # only foot0 is in contact: norm([0.3, 0.4, 0.0]) = 0.5
    assert torch.allclose(fn(env), torch.tensor([0.5]))


"""
Passing an MdpFn subclass uninstantiated
"""


def test_mdp_fn_class_passed_uninstantiated_raises_a_clear_error(env):
    with pytest.raises(TypeError, match="base_height must be constructed, not passed as a class"):
        ConfigItem({"fn": rewards.base_height, "params": {"target_height": 0.3}}, env)
