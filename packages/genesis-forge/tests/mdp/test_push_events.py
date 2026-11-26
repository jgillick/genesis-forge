"""Behavior of the built-in event functions, against a fake entity."""

import math
from types import SimpleNamespace
from typing import Any

import genesis as gs
import numpy as np
import pytest
import torch

from genesis_forge.mdp import events

# A 90 degree rotation around the z-axis
QUAT_Z90 = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]


class FakeSolver:
    def __init__(self):
        self.calls = []

    def apply_links_external_force(self, force, links_idx, envs_idx, ref):
        self.calls.append(("force", force.clone(), links_idx, envs_idx.tolist(), ref))

    def apply_links_external_torque(self, torque, links_idx, envs_idx):
        self.calls.append(("torque", torque.clone(), links_idx, envs_idx.tolist()))


class FakeEntity:
    def __init__(self, num_envs, n_dofs=6):
        self.links = [
            SimpleNamespace(name="base", idx=10, idx_local=0),
            SimpleNamespace(name="FL_foot", idx=11, idx_local=1),
            SimpleNamespace(name="FR_foot", idx=12, idx_local=2),
        ]
        self.base_link = self.links[0]
        self.base_joint = SimpleNamespace(
            n_dofs=n_dofs, dofs_idx_local=list(range(n_dofs))
        )
        self.solver = FakeSolver()
        self.vel = torch.ones((num_envs, 6))

    def get_quat(self, envs_idx, *, relative=True):
        # The free joint's velocity is in the solver's frame, not the user's
        assert not relative
        return torch.tensor([QUAT_Z90] * len(envs_idx))

    def get_pos(self, envs_idx):
        return torch.ones((len(envs_idx), 3))

    def get_links_pos(self, links_idx_local, envs_idx):
        return torch.full((len(envs_idx), 1, 3), float(links_idx_local[0]))

    def get_dofs_velocity(self, dofs_idx, envs_idx):
        return self.vel[envs_idx][:, dofs_idx]

    def set_dofs_velocity(self, velocity, dofs_idx, envs_idx):
        self.vel[envs_idx] = velocity


def bind(fn, env):
    fn.context(env)
    fn.safe_build()
    return fn


@pytest.fixture
def mgr(env):
    return SimpleNamespace(entity=FakeEntity(env.num_envs))


def test_push_velocity_adds_to_the_current_velocity(env, mgr):
    fn = bind(
        events.push_velocity(
            entity_manager=mgr, linear={"x": (0.5, 0.5)}, angular={"x": (1.0, 1.0)}
        ),
        env,
    )
    fn(env, torch.tensor([1, 3]))
    # World x is the base's -y, after a 90 degree rotation around z
    expected = torch.tensor([1.5, 1.0, 1.0, 1.0, 0.0, 1.0])
    assert torch.allclose(mgr.entity.vel[[1, 3]], expected, atol=1e-6)
    assert torch.equal(mgr.entity.vel[[0, 2]], torch.ones((2, 6)))


def test_push_velocity_requires_a_free_base(env):
    mgr: Any = SimpleNamespace(entity=FakeEntity(env.num_envs, n_dofs=0))
    with pytest.raises(ValueError, match="free base"):
        bind(events.push_velocity(entity_manager=mgr, linear={"x": (0, 1)}), env)


def test_push_velocity_requires_an_axis(env, mgr):
    with pytest.raises(ValueError, match="at least one axis"):
        bind(events.push_velocity(entity_manager=mgr), env)
    with pytest.raises(ValueError, match="at least one axis"):
        bind(events.push_velocity(entity_manager=mgr, linear={}, angular={}), env)


def test_push_velocity_uses_changed_ranges(env, mgr):
    fn = bind(events.push_velocity(entity_manager=mgr, linear={"x": (1, 1)}), env)
    fn.linear = {"y": (2, 2), "z": (3, 3)}
    fn(env, torch.tensor([0]))
    expected = torch.tensor([1.0, 3.0, 4.0, 1.0, 1.0, 1.0])
    assert torch.allclose(mgr.entity.vel[0], expected, atol=1e-6)


def test_push_force_is_horizontal_constant_and_cleared(env, mgr):
    fn = bind(events.push_force(entity_manager=mgr, force=(10.0, 40.0)), env)
    envs_idx = torch.tensor([0, 2])
    fn.start(envs_idx)
    fn.step(envs_idx)
    fn.step(envs_idx)

    first, second = mgr.entity.solver.calls
    assert first[0] == "force" and first[2:] == ([10], [0, 2], "link_com")
    assert first[1].shape == (2, 1, 3)
    assert torch.equal(first[1], second[1])
    size = first[1].norm(dim=-1)
    assert ((size >= 10.0 - 1e-4) & (size <= 40.0 + 1e-4)).all()
    assert (first[1][..., 2] == 0).all()
    assert (fn.current_force[[1, 3]] == 0).all()

    fn.end(envs_idx)
    assert (fn.current_force == 0).all()


def test_push_force_with_torque(env, mgr):
    fn = bind(
        events.push_force(
            entity_manager=mgr, force=(1.0, 1.0), torque=(2.0, 2.0), link_name="FL_.*"
        ),
        env,
    )
    envs_idx = torch.tensor([0, 2])
    fn.start(envs_idx)
    fn.step(envs_idx)
    _, torque = mgr.entity.solver.calls
    assert torque[0] == "torque" and torque[2] == [11]
    assert torch.allclose(torque[1].norm(dim=-1), torch.tensor(2.0))


def test_push_force_survives_a_param_change(env, mgr):
    fn = bind(events.push_force(entity_manager=mgr, force=(10.0, 10.0)), env)
    fn.start(torch.tensor([0]))
    before = fn.current_force.clone()
    fn.force = (20.0, 20.0)
    assert torch.equal(fn.current_force, before)


@pytest.mark.parametrize("name", ["nope", ".*_foot"])
def test_push_force_requires_exactly_one_link(env, mgr, name):
    with pytest.raises(ValueError, match="Expected one link"):
        bind(events.push_force(entity_manager=mgr, force=(1, 1), link_name=name), env)


def test_push_force_on_genesis_1_4(env, mgr, monkeypatch):
    calls = []
    frames = SimpleNamespace(link_COM="com")
    monkeypatch.setattr(gs, "link_ref_frame", frames, raising=False)
    mgr.entity.solver = SimpleNamespace(
        apply_links_external_wrench=lambda **kwargs: calls.append(kwargs)
    )
    fn = bind(events.push_force(entity_manager=mgr, force=(5.0, 5.0)), env)
    envs_idx = torch.tensor([1])
    fn.start(envs_idx)
    fn.step(envs_idx)
    assert len(calls) == 1
    assert calls[0]["links_idx"] == [10]
    assert calls[0]["ref"] == "com"
    assert calls[0]["torque"] is None
    assert torch.allclose(calls[0]["force"].norm(), torch.tensor(5.0))


class FakeScene:
    """Records the meshes that were drawn."""

    def __init__(self, num_envs):
        self.envs_offset = np.zeros((num_envs, 3))
        self.envs_offset[:, 0] = np.arange(num_envs) * 10
        self.meshes = []

    def draw_debug_mesh(self, mesh):
        self.meshes.append(mesh)
        return mesh


def test_push_velocity_draws_an_arrow_relative_to_the_strongest_push(env, mgr):
    env.scene = FakeScene(env.num_envs)
    fn = bind(
        events.push_velocity(
            entity_manager=mgr,
            linear={"x": (0.5, 0.5)},
            debug_style={"arrow_max_length": 2.0},
        ),
        env,
    )
    envs_idx = torch.tensor([1, 3])
    fn(env, envs_idx)
    nodes = fn.debug_draw(envs_idx)
    assert len(nodes) == 2
    # The entity is at x=1, and env 1 is offset by 10
    assert nodes[0].bounds[:, 0] == pytest.approx([11.0, 13.0], abs=1e-4)
    assert nodes[1].bounds[:, 0] == pytest.approx([31.0, 33.0], abs=1e-4)


def test_push_force_is_drawn_after_it_ends(env, mgr):
    env.scene = FakeScene(env.num_envs)
    fn = bind(events.push_force(entity_manager=mgr, force=(10.0, 40.0)), env)
    envs_idx = torch.tensor([0])
    fn.start(envs_idx)
    fn.end(envs_idx)
    assert len(fn.debug_draw(envs_idx)) == 1


def test_an_angular_push_draws_nothing(env, mgr):
    env.scene = FakeScene(env.num_envs)
    fn = bind(events.push_velocity(entity_manager=mgr, angular={"z": (1.0, 1.0)}), env)
    fn(env, torch.tensor([0]))
    assert fn.debug_draw(torch.tensor([0])) == []


INVALID_RANGES = [(2.0, 1.0), 3.0, (1.0,), (0.0, float("inf")), ("a", "b")]


@pytest.mark.parametrize("value", INVALID_RANGES)
def test_invalid_ranges_are_rejected_when_built(env, mgr, value):
    with pytest.raises(ValueError, match="force must be"):
        bind(events.push_force(entity_manager=mgr, force=value), env)
    with pytest.raises(ValueError, match="torque must be"):
        bind(events.push_force(entity_manager=mgr, force=(1, 1), torque=value), env)
    with pytest.raises(ValueError, match=r"angular\['y'\] must be"):
        bind(events.push_velocity(entity_manager=mgr, angular={"y": value}), env)


def test_an_invalid_range_is_rejected_when_changed(env, mgr):
    fn = bind(events.push_force(entity_manager=mgr, force=(1.0, 2.0)), env)
    with pytest.raises(ValueError, match="force must be"):
        fn.force = (5.0, 2.0)
