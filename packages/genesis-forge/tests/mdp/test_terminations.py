"""Numerical behavior of the termination functions in genesis_forge.mdp.terminations."""

import pytest
import torch
from helpers import built

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
    def __init__(self, bounds, env_subterrain=None, subterrain_bounds=None):
        """
        A terrain with whole-terrain bounds, and optional per-env subterrains and their
        bounds.
        """
        self._bounds = bounds
        self.env_subterrain = env_subterrain
        self._subterrain_bounds = subterrain_bounds
        self.asked_for = []

    def indices(self, names):
        """``known`` is one subterrain and ``group`` is two; anything else is unknown."""
        lookup = {"known": [0], "group": [0, 1]}
        names = [names] if isinstance(names, str) else names
        return torch.tensor([i for name in names for i in lookup[name]])

    def get_subterrain_bounds(self, subterrains):
        """The bounds of each index, of every subterrain a selection names, or the whole terrain."""
        if isinstance(subterrains, torch.Tensor):
            return self._subterrain_bounds[subterrains]
        self.asked_for.append(subterrains)
        if subterrains is None:
            return torch.tensor([self._bounds])
        return self._subterrain_bounds[self.indices(subterrains)]


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
    fn = built(terminations.timeout(), env)

    assert torch.equal(fn(env), torch.tensor([True, False]))


def test_timeout_never_terminates_without_a_max_episode_length(env):
    env.max_episode_length = None
    fn = built(terminations.timeout(), env)

    assert torch.equal(fn(env), torch.zeros(env.num_envs, dtype=torch.bool))


"""
bad_orientation / is_upsidedown -- grace period gating
"""


def test_bad_orientation_terminates_past_the_tilt_limit(env):
    env.episode_length = torch.tensor([10])
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.9, 0.0, -0.436]]))
    fn = built(terminations.bad_orientation(limit_angle=40.0, entity_manager=mgr), env)

    # tilt magnitude 0.9 -> asin(0.9) = ~64 degrees > 40
    assert torch.equal(fn(env), torch.tensor([True]))


def test_bad_orientation_respects_grace_period(env):
    env.episode_length = torch.tensor([2])
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.9, 0.0, -0.436]]))
    fn = built(
        terminations.bad_orientation(
            limit_angle=40.0, entity_manager=mgr, grace_steps=5
        ),
        env,
    )

    assert torch.equal(fn(env), torch.tensor([False]))


def test_is_upsidedown_uses_z_component_only(env):
    env.episode_length = torch.tensor([10])
    mgr = FakeEntityManager(projected_gravity=torch.tensor([[0.0, 0.0, 0.8]]))
    fn = built(terminations.is_upsidedown(threshold=0.5, entity_manager=mgr), env)

    assert torch.equal(fn(env), torch.tensor([True]))


"""
base_height_below_minimum
"""


def test_base_height_below_minimum_uses_entity_manager_base_pos(env):
    mgr = FakeEntityManager(base_pos=torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.5]]))
    fn = built(
        terminations.base_height_below_minimum(minimum_height=0.05, entity_manager=mgr),
        env,
    )

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
    fn = built(
        terminations.out_of_bounds(terrain_manager=terrain, border_margin=0.5), env
    )

    assert torch.equal(fn(env), torch.tensor([True]))


class RobotsAt:
    """A fake robot for each (x, y) position."""

    def __init__(self, *positions):
        self._pos = torch.tensor([[x, y, 0.0] for x, y in positions])

    def get_pos(self):
        return self._pos


TWO_SUBTERRAINS = torch.tensor([[0.0, 10.0, 0.0, 10.0], [10.0, 20.0, 0.0, 10.0]])


def test_out_of_bounds_looks_up_a_named_subterrain_once_at_build(env):
    """A name's bounds come from the terrain manager at build, not on every step."""
    env.robot = RobotsAt((5.0, 5.0))
    terrain = FakeTerrainManager(
        (0.0, 20.0, 0.0, 10.0), subterrain_bounds=TWO_SUBTERRAINS
    )
    fn = built(
        terminations.out_of_bounds(terrain_manager=terrain, subterrain="known"), env
    )
    fn(env)
    fn(env)

    assert terrain.asked_for == ["known"]


def test_out_of_bounds_with_subterrain_false_uses_the_whole_terrain(env):
    """subterrain=False means the same as None: the whole terrain."""
    env.robot = RobotsAt((15.0, 5.0))
    terrain = FakeTerrainManager((0.0, 20.0, 0.0, 10.0))
    fn = built(
        terminations.out_of_bounds(terrain_manager=terrain, subterrain=False), env
    )

    assert torch.equal(fn(env), torch.tensor([False]))
    assert terrain.asked_for == [None]


@pytest.mark.parametrize("subterrain", ["group", ["known", "group"]])
def test_out_of_bounds_keeps_robots_inside_any_selected_subterrain(env, subterrain):
    """
    A shared name, or a list of names, is satisfied by any of its subterrains: robots on
    either one stay in, and one past both is out.
    """
    env.robot = RobotsAt((5.0, 5.0), (15.0, 5.0), (25.0, 5.0))
    terrain = FakeTerrainManager(
        (0.0, 20.0, 0.0, 10.0), subterrain_bounds=TWO_SUBTERRAINS
    )
    fn = built(
        terminations.out_of_bounds(terrain_manager=terrain, subterrain=subterrain),
        env,
    )

    assert torch.equal(fn(env), torch.tensor([False, False, True]))


def test_out_of_bounds_applies_the_margin_to_each_subterrain(env):
    """The margin applies inside each subterrain, so near the edge of one is out."""
    env.robot = RobotsAt((9.8, 5.0))
    terrain = FakeTerrainManager(
        (0.0, 20.0, 0.0, 10.0), subterrain_bounds=TWO_SUBTERRAINS
    )
    fn = built(
        terminations.out_of_bounds(
            terrain_manager=terrain, subterrain="known", border_margin=0.5
        ),
        env,
    )

    assert torch.equal(fn(env), torch.tensor([True]))


def test_out_of_bounds_checks_each_env_against_its_assigned_subterrain(env):
    """
    Two robots at the same spot: inside the first one's subterrain, outside the second
    one's.
    """

    class FakeRobot:
        def get_pos(self):
            """Both robots at x=5, y=5."""
            # Both robots sit at x=5: inside subterrain 0, outside subterrain 1
            return torch.tensor([[5.0, 5.0, 0.0], [5.0, 5.0, 0.0]])

    env.robot = FakeRobot()
    terrain = FakeTerrainManager(
        (0.0, 20.0, 0.0, 10.0),
        env_subterrain=torch.tensor([0, 1]),
        subterrain_bounds=torch.tensor(
            [[0.0, 10.0, 0.0, 10.0], [10.0, 20.0, 0.0, 10.0]]
        ),
    )
    fn = built(
        terminations.out_of_bounds(
            terrain_manager=terrain, subterrain=True, border_margin=0.5
        ),
        env,
    )

    assert torch.equal(fn(env), torch.tensor([False, True]))


"""
Contact-based terminations
"""


def test_has_contact_requires_min_contacts(env):
    contacts = torch.tensor([[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    fn = built(
        terminations.has_contact(
            contact_manager=FakeContactManager(contacts), threshold=1.0, min_contacts=2
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([True]))


def test_contact_force_terminates_on_any_link_over_threshold(env):
    contacts = torch.tensor([[[3.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    fn = built(
        terminations.contact_force(
            contact_manager=FakeContactManager(contacts), threshold=1.0
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([True]))


def test_contact_force_with_grace_period_gates_by_episode_length(env):
    env.episode_length = torch.tensor([2])
    contacts = torch.tensor([[[200.0, 0.0, 0.0]]])
    fn = built(
        terminations.contact_force_with_grace_period(
            contact_manager=FakeContactManager(contacts),
            threshold=100.0,
            grace_steps=10,
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([False]))


"""
Actuator limit terminations
"""


def test_dof_control_force_limit_falls_back_to_actuator_max(env):
    actuator = FakeActuatorManager(
        control_force=torch.tensor([[15.0, 5.0]]), max_force=torch.tensor([10.0, 10.0])
    )
    fn = built(terminations.dof_control_force_limit(actuator_manager=actuator), env)
    assert torch.equal(fn(env), torch.tensor([True]))


def test_dof_control_force_limit_uses_explicit_threshold_over_actuator_max(env):
    actuator = FakeActuatorManager(
        control_force=torch.tensor([[8.0]]), max_force=torch.tensor([5.0])
    )
    fn = built(
        terminations.dof_control_force_limit(actuator_manager=actuator, threshold=20.0),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([False]))


def test_dof_velocity_limit_converts_rpm_to_rad(env):
    # 300 rpm -> ~31.4 rad/s; a measured velocity of 35 rad/s should exceed it
    actuator = FakeActuatorManager(velocity=torch.tensor([[35.0]]))
    fn = built(
        terminations.dof_velocity_limit(
            actuator_manager=actuator, threshold=300.0, unit="rpm"
        ),
        env,
    )
    assert torch.equal(fn(env), torch.tensor([True]))


def test_dof_velocity_limit_rejects_unknown_unit_at_build_time(env):
    """The unit is validated once in build() (and again on any param change),
    not on every __call__."""
    fn = terminations.dof_velocity_limit(
        actuator_manager=object(), threshold=1.0, unit="bogus"
    )
    fn.context(env)
    with pytest.raises(AssertionError, match="Unknown velocity unit"):
        fn.safe_build()
