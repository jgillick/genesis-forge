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
velocity) takes priority over the raw entity_attr path
"""


def test_dofs_position_prefers_actuator_manager_over_entity_attr(env):
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


def test_dofs_velocity_prefers_action_manager_over_entity_attr(env):
    class FakeActionManager:
        def get_dofs_velocity(self):
            return torch.tensor([[1.0, 2.0]])

    fn = observations.entity_dofs_velocity(action_manager=FakeActionManager())
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[1.0, 2.0]]))


def test_dofs_velocity_falls_back_to_entity_attr(env):
    class FakeEntityWithDofs:
        def get_dofs_velocity(self, dofs_idx):
            assert dofs_idx == [0, 1]
            return torch.tensor([[3.0, 4.0]])

    env.robot = FakeEntityWithDofs()
    fn = observations.entity_dofs_velocity(dofs_idx=[0, 1])
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
current_actions
"""


def test_current_actions_falls_back_to_env_actions(env):
    env.actions = torch.tensor([[0.1, 0.2]])
    fn = observations.current_actions()
    fn.context(env)
    fn.safe_build()
    assert torch.equal(fn(env), torch.tensor([[0.1, 0.2]]))


def test_current_actions_prefers_action_manager(env):
    class FakeActionManager:
        def get_actions(self):
            return torch.tensor([[9.0]])

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
