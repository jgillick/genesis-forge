"""Behavior of EntityManager: cached base-frame values (position, quaternion,
local-frame projected gravity/velocities) and dispatching its on_reset config items.

Uses a FakeEntity -- no Genesis scene is built. transform_by_quat/inv_quat are the
real genesis.utils.geom functions (pure quaternion math, no gs.init() needed) --
used directly in assertions to compute the expected value independently of the
manager, rather than re-deriving the trig by hand.
"""

from dataclasses import dataclass

import pytest
import torch
from genesis.utils.geom import transform_by_quat, inv_quat

from genesis_forge.managers import EntityManager
from genesis_forge.managers.config import ResetMdpFn


class FakeEntity:
    def __init__(self, pos, quat, vel=None, ang=None):
        self._pos = pos
        self._quat = quat
        self._vel = vel
        self._ang = ang

    def get_pos(self):
        return self._pos

    def get_quat(self):
        return self._quat

    def get_vel(self):
        return self._vel

    def get_ang(self):
        return self._ang


def make_entity(num_envs, quat=None, vel=None, ang=None):
    if quat is None:
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs)
    return FakeEntity(
        pos=torch.zeros((num_envs, 3)),
        quat=quat,
        vel=vel if vel is not None else torch.zeros((num_envs, 3)),
        ang=ang if ang is not None else torch.zeros((num_envs, 3)),
    )


"""
build()
"""


def test_build_sets_the_entity_from_the_configured_attr(env):
    entity = make_entity(env.num_envs)
    env.robot = entity
    mgr = EntityManager(env, entity_attr="robot")
    mgr.build()

    assert mgr.entity is entity


def test_build_populates_the_cached_base_pos_and_quat(env):
    pos = torch.tensor([[1.0, 2.0, 3.0]] * env.num_envs)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * env.num_envs)
    env.robot = FakeEntity(pos=pos, quat=quat)
    mgr = EntityManager(env, entity_attr="robot")
    mgr.build()

    assert torch.equal(mgr.base_pos, pos)
    assert torch.equal(mgr.base_quat, quat)
    assert torch.equal(mgr.inv_base_quat, inv_quat(quat))


def test_build_builds_each_reset_config_item(env):
    entity = make_entity(env.num_envs)
    env.robot = entity

    @dataclass(kw_only=True, eq=False)
    class Counting(ResetMdpFn):
        def build(self):
            self.builds = getattr(self, "builds", 0) + 1

        def __call__(self, env, entity, envs_idx):
            pass

    fn = Counting()
    mgr = EntityManager(env, entity_attr="robot", on_reset={"pos": {"fn": fn}})
    assert getattr(fn, "builds", 0) == 0

    mgr.build()

    assert fn.builds == 1
    assert fn.entity is entity


def test_step_recomputes_cached_values_after_the_entity_moves(env):
    entity = make_entity(env.num_envs)
    env.robot = entity
    mgr = EntityManager(env, entity_attr="robot")
    mgr.build()

    entity._pos = torch.full((env.num_envs, 3), 5.0)
    mgr.step()

    assert torch.equal(mgr.base_pos, entity._pos)


"""
Local-frame helpers
"""


def test_get_projected_gravity_transforms_by_the_inverse_base_quat(env):
    quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * env.num_envs)  # 180 deg about x
    env.robot = make_entity(env.num_envs, quat=quat)
    mgr = EntityManager(env, entity_attr="robot")
    mgr.build()

    expected = transform_by_quat(
        torch.tensor([[0.0, 0.0, -1.0]] * env.num_envs), inv_quat(quat)
    )
    assert torch.allclose(mgr.get_projected_gravity(), expected)


def test_get_linear_velocity_transforms_the_entity_velocity(env):
    quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * env.num_envs)
    vel = torch.tensor([[1.0, 2.0, 3.0]] * env.num_envs)
    env.robot = make_entity(env.num_envs, quat=quat, vel=vel)
    mgr = EntityManager(env, entity_attr="robot")
    mgr.build()

    expected = transform_by_quat(vel, inv_quat(quat))
    assert torch.allclose(mgr.get_linear_velocity(), expected)


def test_get_angular_velocity_transforms_the_entity_angular_velocity(env):
    quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * env.num_envs)
    ang = torch.tensor([[0.1, 0.2, 0.3]] * env.num_envs)
    env.robot = make_entity(env.num_envs, quat=quat, ang=ang)
    mgr = EntityManager(env, entity_attr="robot")
    mgr.build()

    expected = transform_by_quat(ang, inv_quat(quat))
    assert torch.allclose(mgr.get_angular_velocity(), expected)


"""
reset() -- dispatches to each on_reset config item
"""


def test_reset_forwards_to_a_plain_reset_function(env):
    entity = make_entity(env.num_envs)
    env.robot = entity
    calls = []

    def my_reset(env, entity, envs_idx, offset=0.0):
        calls.append((env, entity, list(envs_idx), offset))

    mgr = EntityManager(
        env, entity_attr="robot", on_reset={"pos": {"fn": my_reset, "params": {"offset": 1.0}}}
    )
    mgr.build()

    mgr.reset([0, 2])

    assert calls == [(env, entity, [0, 2], 1.0)]


def test_reset_forwards_to_a_resetmdpfn_instance(env):
    entity = make_entity(env.num_envs)
    env.robot = entity

    @dataclass(kw_only=True, eq=False)
    class Recorder(ResetMdpFn):
        def __call__(self, env, entity, envs_idx):
            self.calls = getattr(self, "calls", []) + [(entity, list(envs_idx))]

    fn = Recorder()
    mgr = EntityManager(env, entity_attr="robot", on_reset={"pos": {"fn": fn}})
    mgr.build()

    mgr.reset([1, 3])

    assert fn.entity is entity
    assert fn.calls == [(entity, [1, 3])]


def test_reset_defaults_to_every_env(env):
    entity = make_entity(env.num_envs)
    env.robot = entity
    calls = []

    def my_reset(env, entity, envs_idx):
        calls.append([int(i) for i in envs_idx])

    mgr = EntityManager(env, entity_attr="robot", on_reset={"pos": {"fn": my_reset}})
    mgr.build()

    mgr.reset()

    assert calls == [list(range(env.num_envs))]


def test_reset_is_a_noop_when_disabled(env):
    entity = make_entity(env.num_envs)
    env.robot = entity
    calls = []

    def my_reset(env, entity, envs_idx):
        calls.append(1)

    mgr = EntityManager(env, entity_attr="robot", on_reset={"pos": {"fn": my_reset}})
    mgr.build()
    mgr.enabled = False

    mgr.reset([0])

    assert calls == []


def test_reset_reraises_the_original_exception(env):
    entity = make_entity(env.num_envs)
    env.robot = entity

    def failing_reset(env, entity, envs_idx):
        raise ValueError("boom")

    mgr = EntityManager(env, entity_attr="robot", on_reset={"pos": {"fn": failing_reset}})
    mgr.build()

    with pytest.raises(ValueError, match="boom"):
        mgr.reset([0])
