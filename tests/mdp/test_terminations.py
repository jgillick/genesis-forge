"""Numerical behavior of the termination functions in genesis_forge.mdp.terminations."""

import pytest
import torch

from genesis_forge.mdp import terminations


class FakeEntityManager:
    def __init__(self, projected_gravity=None, base_pos=None):
        self._gravity = projected_gravity
        self.base_pos = base_pos

    def get_projected_gravity(self):
        return self._gravity


class FakeContactManager:
    def __init__(self, contacts):
        self.contacts = contacts


class FakeTerrainManager:
    def __init__(self, bounds):
        self._bounds = bounds

    def get_bounds(self, subterrain=None):
        return self._bounds


class FakeActuatorManager:
    def __init__(self, control_force=None, max_force=None, velocity=None):
        self._force, self._max_force, self._vel = control_force, max_force, velocity

    def get_dofs_control_force(self):
        return self._force

    def get_dofs_max_force(self):
        return self._max_force

    def get_dofs_velocity(self):
        return self._vel


"""
timeout
"""


def test_timeout_terminates_past_max_episode_length(env):
    env.episode_length = torch.tensor([15, 5])
    env.max_episode_length = 10
    fn = terminations.timeout()
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([True, False]))


def test_timeout_never_terminates_without_a_max_episode_length(env):
    env.max_episode_length = None
    fn = terminations.timeout()
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.zeros(env.num_envs, dtype=torch.bool))


"""
bad_orientation / is_upsidedown -- grace period gating
"""


def test_bad_orientation_terminates_past_the_tilt_limit(env):
    env.episode_length = torch.tensor([10])
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.9, 0.0, -0.436]]))
    fn = terminations.bad_orientation(limit_angle=40.0, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    # tilt magnitude 0.9 -> asin clamped to 0.99 -> ~81.9 degrees > 40
    assert torch.equal(fn(env), torch.tensor([True]))


def test_bad_orientation_respects_grace_period(env):
    env.episode_length = torch.tensor([2])
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.9, 0.0, -0.436]]))
    fn = terminations.bad_orientation(limit_angle=40.0, entity_manager=mgr, grace_steps=5)
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([False]))


def test_is_upsidedown_uses_z_component_only(env):
    env.episode_length = torch.tensor([10])
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.0, 0.0, 0.8]]))
    fn = terminations.is_upsidedown(threshold=0.5, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([True]))


"""
base_height_below_minimum
"""


def test_base_height_below_minimum_uses_entity_manager_base_pos(env):
    mgr = FakeEntityManager(base_pos=torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.5]]))
    fn = terminations.base_height_below_minimum(minimum_height=0.05, entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([True, False]))


"""
out_of_bounds
"""


def test_out_of_bounds_checks_terrain_bounds_with_margin(env):
    class FakeRobot:
        def get_pos(self):
            return torch.tensor([[9.8, 0.0, 0.0]])  # just outside x_max - margin

    env.robot = FakeRobot()
    terrain = FakeTerrainManager((0.0, 10.0, 0.0, 10.0))
    fn = terminations.out_of_bounds(terrain_manager=terrain, border_margin=0.5)
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([True]))


"""
Contact-based terminations
"""


def test_has_contact_requires_min_contacts(env):
    contacts = torch.tensor([[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    fn = terminations.has_contact(
        contact_manager=FakeContactManager(contacts), threshold=1.0, min_contacts=2
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([True]))


def test_contact_force_terminates_on_any_link_over_threshold(env):
    contacts = torch.tensor([[[3.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    fn = terminations.contact_force(
        contact_manager=FakeContactManager(contacts), threshold=1.0
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([True]))


def test_contact_force_with_grace_period_gates_by_episode_length(env):
    env.episode_length = torch.tensor([2])
    contacts = torch.tensor([[[200.0, 0.0, 0.0]]])
    fn = terminations.contact_force_with_grace_period(
        contact_manager=FakeContactManager(contacts), threshold=100.0, grace_steps=10
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([False]))


"""
Actuator limit terminations
"""


def test_dof_control_force_limit_falls_back_to_actuator_max(env):
    actuator = FakeActuatorManager(
        control_force=torch.tensor([[15.0, 5.0]]), max_force=torch.tensor([10.0, 10.0])
    )
    fn = terminations.dof_control_force_limit(actuator_manager=actuator)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([True]))


def test_dof_control_force_limit_uses_explicit_threshold_over_actuator_max(env):
    actuator = FakeActuatorManager(
        control_force=torch.tensor([[8.0]]), max_force=torch.tensor([5.0])
    )
    fn = terminations.dof_control_force_limit(actuator_manager=actuator, threshold=20.0)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([False]))


def test_dof_velocity_limit_converts_rpm_to_rad(env):
    # 300 rpm -> ~31.4 rad/s; a measured velocity of 35 rad/s should exceed it
    actuator = FakeActuatorManager(velocity=torch.tensor([[35.0]]))
    fn = terminations.dof_velocity_limit(
        actuator_manager=actuator, threshold=300.0, unit="rpm"
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([True]))


def test_dof_velocity_limit_rejects_unknown_unit_at_build_time(env):
    """The unit is validated once in build() (and again on any param change),
    not on every __call__."""
    fn = terminations.dof_velocity_limit(actuator_manager=object(), threshold=1.0, unit="bogus")
    fn.context(env)
    with pytest.raises(AssertionError, match="Unknown velocity unit"):
        fn.safe_build()
