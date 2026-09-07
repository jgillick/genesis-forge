"""Numerical behavior of the observation functions in genesis_forge.mdp.observations."""

import torch

from genesis_forge.mdp import observations


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


class FakeEntity:
    def get_quat(self):
        return torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # identity: upright


class FakeActuatorManager:
    def __init__(self, pos=None, vel=None, force=None):
        self._pos, self._vel, self._force = pos, vel, force

    def get_dofs_position(self):
        return self._pos

    def get_dofs_velocity(self):
        return self._vel

    def get_dofs_force(self, clip_to_max_force=False):
        return self._force * (0.5 if clip_to_max_force else 1.0)


"""
entity_projected_gravity
"""

def test_entity_projected_gravity_prefers_entity_manager(env):
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.1, 0.2, -0.9]]))
    fn = observations.entity_projected_gravity(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()

    assert torch.equal(fn(env), torch.tensor([[0.1, 0.2, -0.9]]))


"""
entity_linear_velocity / entity_angular_velocity
"""


def test_entity_linear_velocity_uses_entity_manager(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[1.0, 0.0, 0.0]]))
    fn = observations.entity_linear_velocity(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[1.0, 0.0, 0.0]]))


def test_entity_angular_velocity_uses_entity_manager(env):
    mgr = FakeEntityManager(ang_vel=torch.tensor([[0.0, 0.0, 0.5]]))
    fn = observations.entity_angular_velocity(entity_manager=mgr)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[0.0, 0.0, 0.5]]))


"""
entity_dofs_position / velocity / force -- actuator_manager (or action_manager, for
velocity) takes priority over the raw entity path
"""


def test_dofs_position_prefers_actuator_manager_over_entity(env):
    actuator = FakeActuatorManager(pos=torch.tensor([[1.0, 2.0]]))

    class FakeEntityWithDofs:
        def get_dofs_position(self, dofs_idx):
            return torch.tensor([[9.0, 9.0]])

    env.robot = FakeEntityWithDofs()
    fn = observations.entity_dofs_position(actuator_manager=actuator)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[1.0, 2.0]]))


def test_dofs_force_clip_to_max_force_is_passed_through(env):
    actuator = FakeActuatorManager(force=torch.tensor([[10.0, 10.0]]))
    fn = observations.entity_dofs_force(actuator_manager=actuator, clip_to_max_force=True)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[5.0, 5.0]]))


def test_dofs_velocity_prefers_action_manager_over_entity(env):
    class FakeActionManager:
        def get_dofs_velocity(self):
            return torch.tensor([[1.0, 2.0]])

    fn = observations.entity_dofs_velocity(action_manager=FakeActionManager())
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[1.0, 2.0]]))


def test_dofs_velocity_falls_back_to_entity(env):
    class FakeEntityWithDofs:
        def get_dofs_velocity(self, dofs_idx):
            assert dofs_idx == [0, 1]
            return torch.tensor([[3.0, 4.0]])

    env.robot = FakeEntityWithDofs()
    fn = observations.entity_dofs_velocity(dofs_idx=[0, 1])
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[3.0, 4.0]]))


def test_dofs_velocity_prefers_explicit_entity_over_env_robot(env):
    class FakeEntityWithDofs:
        def __init__(self, value):
            self._value = value

        def get_dofs_velocity(self, dofs_idx):
            return self._value

    env.robot = FakeEntityWithDofs(torch.tensor([[9.0, 9.0]]))
    other_entity = FakeEntityWithDofs(torch.tensor([[3.0, 4.0]]))
    fn = observations.entity_dofs_velocity(entity=other_entity)
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[3.0, 4.0]]))


"""
read_imu
"""


def test_read_imu_concatenates_lin_acc_and_ang_vel(env):
    class Reading:
        lin_acc = torch.tensor([[1.0, 2.0, 3.0]])
        ang_vel = torch.tensor([[4.0, 5.0, 6.0]])

    class FakeImu:
        def read(self):
            return Reading()

    fn = observations.read_imu(imu=FakeImu())
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))


"""
raycaster_distance
"""


class FakeRaycasterSensor:
    def __init__(self, distances, max_range=None):
        self._distances = distances
        if max_range is not None:
            self._options = type("Options", (), {"max_range": max_range})()

    def read(self):
        return type("Reading", (), {"distances": self._distances})()


def test_raycaster_distance_min_reduces_to_nearest_reading(env):
    distances = torch.tensor([[[2.0, 3.0], [0.5, 4.0]], [[1.0, 1.5], [2.5, 3.5]]])
    fn = observations.raycaster_distance(
        sensor=FakeRaycasterSensor(distances), reduce="min"
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[0.5], [1.0]]))


def test_raycaster_distance_defaults_to_keeping_every_ray(env):
    """The lossy reduction is opt-in: a default that quietly discards which direction a
    reading came from is very hard to notice from the outside."""
    distances = torch.tensor([[[2.0, 3.0], [0.5, 4.0]]])
    fn = observations.raycaster_distance(sensor=FakeRaycasterSensor(distances))
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[2.0, 3.0, 0.5, 4.0]]))


def test_raycaster_distance_flatten_returns_all_rays(env):
    distances = torch.tensor([[[2.0, 3.0], [0.5, 4.0]]])
    fn = observations.raycaster_distance(
        sensor=FakeRaycasterSensor(distances), reduce="flatten"
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[2.0, 3.0, 0.5, 4.0]]))


def test_raycaster_distance_normalizes_by_sensor_max_range(env):
    distances = torch.tensor([[[2.0, 4.0]]])
    fn = observations.raycaster_distance(
        sensor=FakeRaycasterSensor(distances, max_range=4.0),
        reduce="min",
        normalize=True,
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[0.5]]))


def test_raycaster_distance_explicit_max_range_overrides_sensor(env):
    distances = torch.tensor([[[1.0]]])
    fn = observations.raycaster_distance(
        sensor=FakeRaycasterSensor(distances, max_range=4.0),
        normalize=True,
        max_range=2.0,
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[0.5]]))


def test_raycaster_distance_normalize_without_max_range_raises(env):
    import pytest

    fn = observations.raycaster_distance(
        sensor=FakeRaycasterSensor(torch.zeros((1, 1))), normalize=True
    )
    fn.context(env)
    with pytest.raises(ValueError, match="max range"):
        fn.safe_build()


"""
current_actions
"""


def test_current_actions_falls_back_to_env_actions(env):
    env.actions = torch.tensor([[0.1, 0.2]])
    fn = observations.current_actions()
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[0.1, 0.2]]))


def test_current_actions_falls_back_to_zeros_before_first_step(env):
    env.num_envs = 4
    env.num_actions = 3
    env.actions = None
    fn = observations.current_actions()
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.zeros((4, 3)))


def test_current_actions_prefers_action_manager(env):
    class FakeActionManager:
        raw_actions = torch.tensor([[9.0]])

    env.actions = torch.tensor([[0.1]])
    fn = observations.current_actions(action_manager=FakeActionManager())
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[9.0]]))


"""
Contacts
"""


class FakeContactManager:
    def __init__(self, contacts):
        self.contacts = contacts


def test_contact_force_is_the_norm_of_contacts(env):
    contacts = torch.tensor([[[3.0, 4.0, 0.0]]])  # norm = 5
    fn = observations.contact_force(contact_manager=FakeContactManager(contacts))
    fn.context(env)
    fn.safe_build()
    assert torch.allclose(fn(env), torch.tensor([[5.0]]))


def test_has_contact_thresholds_per_link(env):
    contacts = torch.tensor([[[3.0, 4.0, 0.0], [0.1, 0.0, 0.0]]])  # norms: 5.0, 0.1
    fn = observations.has_contact(
        contact_manager=FakeContactManager(contacts), threshold=1.0
    )
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[1.0, 0.0]]))
