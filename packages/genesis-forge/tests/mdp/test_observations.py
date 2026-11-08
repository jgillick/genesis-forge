"""Numerical behavior of the observation functions in genesis_forge.mdp.observations."""

import math

import pytest
import torch
from helpers import built

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
    env.robot = FakeEntity()  # upright: its own gravity would be (0, 0, -1)
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.1, 0.2, -0.9]]))
    fn = built(observations.entity_projected_gravity(entity_manager=mgr), env)

    assert torch.equal(fn(env), torch.tensor([[0.1, 0.2, -0.9]]))


"""
entity_linear_velocity / entity_angular_velocity
"""


def test_entity_linear_velocity_uses_entity_manager(env):
    mgr = FakeEntityManager(lin_vel=torch.tensor([[1.0, 0.0, 0.0]]))
    fn = built(observations.entity_linear_velocity(entity_manager=mgr), env)
    assert torch.equal(fn(env), torch.tensor([[1.0, 0.0, 0.0]]))


def test_entity_angular_velocity_uses_entity_manager(env):
    mgr = FakeEntityManager(ang_vel=torch.tensor([[0.0, 0.0, 0.5]]))
    fn = built(observations.entity_angular_velocity(entity_manager=mgr), env)
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
    fn = built(observations.entity_dofs_position(actuator_manager=actuator), env)
    assert torch.equal(fn(env), torch.tensor([[1.0, 2.0]]))


def test_dofs_force_clip_to_max_force_is_passed_through(env):
    actuator = FakeActuatorManager(force=torch.tensor([[10.0, 10.0]]))
    fn = built(
        observations.entity_dofs_force(
            actuator_manager=actuator, clip_to_max_force=True
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([[5.0, 5.0]]))


def test_dofs_velocity_prefers_actuator_manager_over_entity(env):
    actuator = FakeActuatorManager(vel=torch.tensor([[1.0, 2.0]]))

    class FakeEntityWithDofs:
        def get_dofs_velocity(self, dofs_idx):
            return torch.tensor([[9.0, 9.0]])

    env.robot = FakeEntityWithDofs()
    fn = built(observations.entity_dofs_velocity(actuator_manager=actuator), env)
    assert torch.equal(fn(env), torch.tensor([[1.0, 2.0]]))


def test_dofs_velocity_falls_back_to_entity(env):
    class FakeEntityWithDofs:
        def get_dofs_velocity(self, dofs_idx):
            assert dofs_idx == [0, 1]
            return torch.tensor([[3.0, 4.0]])

    env.robot = FakeEntityWithDofs()
    fn = built(observations.entity_dofs_velocity(dofs_idx=[0, 1]), env)
    assert torch.equal(fn(env), torch.tensor([[3.0, 4.0]]))


def test_dofs_velocity_prefers_explicit_entity_over_env_robot(env):
    class FakeEntityWithDofs:
        def __init__(self, value):
            self._value = value

        def get_dofs_velocity(self, dofs_idx):
            return self._value

    env.robot = FakeEntityWithDofs(torch.tensor([[9.0, 9.0]]))
    other_entity = FakeEntityWithDofs(torch.tensor([[3.0, 4.0]]))
    fn = built(observations.entity_dofs_velocity(entity=other_entity), env)
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

    fn = built(observations.read_imu(imu=FakeImu()), env)
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
    fn = built(
        observations.raycaster_distance(
            sensor=FakeRaycasterSensor(distances), reduce="min"
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([[0.5], [1.0]]))


def test_raycaster_distance_defaults_to_keeping_every_ray(env):
    """The lossy reduction is opt-in: a default that quietly discards which direction a
    reading came from is very hard to notice from the outside."""
    distances = torch.tensor([[[2.0, 3.0], [0.5, 4.0]]])
    fn = built(
        observations.raycaster_distance(sensor=FakeRaycasterSensor(distances)), env
    )
    assert torch.equal(fn(env), torch.tensor([[2.0, 3.0, 0.5, 4.0]]))


def test_raycaster_distance_normalizes_by_sensor_max_range(env):
    distances = torch.tensor([[[2.0, 4.0]]])
    fn = built(
        observations.raycaster_distance(
            sensor=FakeRaycasterSensor(distances, max_range=4.0),
            reduce="min",
            normalize=True,
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([[0.5]]))


def test_raycaster_distance_explicit_max_range_overrides_sensor(env):
    distances = torch.tensor([[[1.0]]])
    fn = built(
        observations.raycaster_distance(
            sensor=FakeRaycasterSensor(distances, max_range=4.0),
            normalize=True,
            max_range=2.0,
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([[0.5]]))


def test_raycaster_distance_normalize_without_max_range_raises(env):
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
    fn = built(observations.current_actions(), env)
    assert torch.equal(fn(env), torch.tensor([[0.1, 0.2]]))


def test_current_actions_falls_back_to_zeros_before_first_step(env):
    env.num_envs = 4
    env.num_actions = 3
    env.actions = None
    fn = built(observations.current_actions(), env)
    assert torch.equal(fn(env), torch.zeros((4, 3)))


def test_current_actions_prefers_action_manager(env):
    class FakeActionManager:
        raw_actions = torch.tensor([[9.0]])

    env.actions = torch.tensor([[0.1]])
    fn = built(observations.current_actions(action_manager=FakeActionManager()), env)
    assert torch.equal(fn(env), torch.tensor([[9.0]]))


"""
Contacts
"""


class FakeContactManager:
    def __init__(self, contacts):
        self.contacts = contacts


def test_contact_force_is_the_norm_of_contacts(env):
    contacts = torch.tensor([[[3.0, 4.0, 0.0]]])  # norm = 5
    fn = built(
        observations.contact_force(contact_manager=FakeContactManager(contacts)), env
    )
    assert torch.allclose(fn(env), torch.tensor([[5.0]]))


def test_has_contact_thresholds_per_link(env):
    contacts = torch.tensor([[[3.0, 4.0, 0.0], [0.1, 0.0, 0.0]]])  # norms: 5.0, 0.1
    fn = built(
        observations.has_contact(
            contact_manager=FakeContactManager(contacts), threshold=1.0
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([[1.0, 0.0]]))


"""
height_scan
"""


class FakeBaseEntityManager:
    def __init__(self, base_pos, base_quat):
        """An entity manager with the given base positions and quaternions."""
        self.base_pos = base_pos
        self.base_quat = base_quat


class FakeTerrainManager:
    """Ground height is ``slope * x``, so a point's height shows where it was sampled."""

    def __init__(self, slope=0.0, height=0.0):
        """Ground at ``height + slope * x``."""
        self.slope = slope
        self.height = height

    def get_terrain_height(self, x, y):
        """The ground height at each point."""
        return self.height + self.slope * x


def yaw_quat(yaw):
    """A quaternion turned ``yaw`` radians around z, for one environment."""
    return torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]])


def scan(env, terrain, pos=(0.0, 0.0, 0.0), quat=None, **params):
    """Build a height_scan for one base at ``pos``, and return it with one scan's values."""
    manager = FakeBaseEntityManager(
        torch.tensor([pos]), quat if quat is not None else yaw_quat(0.0)
    )
    fn = built(
        observations.height_scan(
            terrain_manager=terrain, entity_manager=manager, **params
        ),
        env,
    )
    return fn, fn(env)


def test_height_scan_default_grid_has_187_points(env):
    """The default 1.6m x 1.0m grid at 0.1m gives 17 x 11 points."""
    _, values = scan(env, FakeTerrainManager())

    assert values.shape == (1, 187)


def test_height_scan_is_base_height_minus_ground_minus_offset(env):
    """A base at 0.5m over ground at 0.1m, with a 0.3m offset, reads 0.1 everywhere."""
    _, values = scan(
        env, FakeTerrainManager(height=0.1), pos=(0.0, 0.0, 0.5), offset=0.3
    )

    assert torch.allclose(values, torch.full((1, 187), 0.1))


def test_height_scan_grid_spans_the_size_around_the_base(env):
    """A 2m x 1m grid at 0.5m samples 5 x 3 points, from 1m behind the base to 1m ahead."""
    # Ground height equals x, so the values reveal where each point was sampled
    _, values = scan(
        env,
        FakeTerrainManager(slope=1.0),
        pos=(10.0, 0.0, 0.0),
        size=(2.0, 1.0),
        resolution=0.5,
    )

    sampled_x = -values[0]
    assert values.shape == (1, 15)  # 5 x 3
    assert torch.isclose(sampled_x.min(), torch.tensor(9.0))
    assert torch.isclose(sampled_x.max(), torch.tensor(11.0))


def test_height_scan_grid_stays_centered_when_the_size_is_not_a_multiple(env):
    """A 1m side at 0.3m fits 4 points, trimmed evenly so the grid stays centered."""
    _, values = scan(
        env,
        FakeTerrainManager(slope=1.0),
        pos=(10.0, 0.0, 0.0),
        size=(1.0, 1.0),
        resolution=0.3,
    )

    sampled_x = -values[0]
    assert values.shape == (1, 16)  # 4 x 4
    assert torch.isclose(sampled_x.min(), torch.tensor(9.55))
    assert torch.isclose(sampled_x.max(), torch.tensor(10.45))


def test_height_scan_steps_along_x_first(env):
    """The first points step along x, then the next row starts back at the lowest x."""
    _, values = scan(
        env, FakeTerrainManager(slope=1.0), size=(1.0, 1.0), resolution=0.5
    )

    assert torch.allclose(-values[0, :4], torch.tensor([-0.5, 0.0, 0.5, -0.5]))


def test_height_scan_offset_defaults_to_zero(env):
    """Without an offset, the values are the base's height above the ground."""
    fn = observations.height_scan(terrain_manager=FakeTerrainManager())

    assert fn.offset == 0.0


def test_height_scan_turns_with_the_heading(env):
    """Turned 90 degrees, a grid that is long in x samples along world y instead."""
    # Turned 90 degrees, the grid's length runs along world y, so x barely varies
    _, values = scan(
        env,
        FakeTerrainManager(slope=1.0),
        quat=yaw_quat(math.pi / 2),
        size=(2.0, 0.0001),
        resolution=0.5,
    )

    assert torch.allclose(values, torch.zeros_like(values), atol=1e-4)


def test_height_scan_ignores_pitch(env):
    """A pitched base still scans level ground ahead of it."""
    pitch = math.radians(20)
    quat = torch.tensor([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])
    _, level = scan(env, FakeTerrainManager(slope=1.0))
    _, pitched = scan(env, FakeTerrainManager(slope=1.0), quat=quat)

    assert torch.allclose(level, pitched, atol=1e-5)


def test_height_scan_scans_each_env_around_its_own_base(env):
    """Each environment's grid is centered on its own base."""
    manager = FakeBaseEntityManager(
        torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        torch.cat([yaw_quat(0.0), yaw_quat(0.0)]),
    )
    fn = built(
        observations.height_scan(
            terrain_manager=FakeTerrainManager(slope=1.0),
            entity_manager=manager,
        ),
        env,
    )

    values = fn(env)

    assert torch.isclose(-values[1].mean() - -values[0].mean(), torch.tensor(5.0))


@pytest.mark.parametrize(
    "params",
    [
        {"resolution": 0.0},
        {"size": (1.0, -1.0)},
    ],
)
def test_height_scan_rejects_bad_params(env, params):
    """A zero resolution or a negative size fails at build."""
    with pytest.raises(ValueError):
        scan(env, FakeTerrainManager(), **params)


def test_height_scan_rebuilds_the_grid_when_the_size_changes(env):
    """Assigning a new size rebuilds the grid, so a curriculum can change it."""
    fn, _ = scan(env, FakeTerrainManager())

    fn.size = (1.0, 1.0)

    assert fn(env).shape == (1, 121)
