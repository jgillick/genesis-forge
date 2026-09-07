"""Numerical behavior of the reset functions in genesis_forge.mdp.reset.
"""

import math

import pytest
import torch

from genesis_forge.mdp import reset


class FakeEntity:
    def __init__(self):
        self.links = []  # links_by_name_pattern iterates this
        self.pos_calls = []
        self.quat_calls = []
        self.mass_shift_calls = []

    def set_pos(self, pos, envs_idx, zero_velocity=None):
        self.pos_calls.append((pos.clone(), list(envs_idx), zero_velocity))

    def set_quat(self, quat, envs_idx, zero_velocity=None):
        self.quat_calls.append((quat.clone(), list(envs_idx), zero_velocity))

    def set_mass_shift(self, mass, links_idx_local, envs_idx):
        self.mass_shift_calls.append((mass.clone(), list(links_idx_local), list(envs_idx)))


class FakeTerrainManager:
    def __init__(self, pos):
        self._pos = pos

    def generate_random_env_pos(self, envs_idx, subterrain, height_offset):
        self.last_call = (list(envs_idx), subterrain, height_offset)
        return self._pos[envs_idx]


"""
zero_all_dofs_velocity
"""


def test_zero_all_dofs_velocity_calls_the_entity_method(env):
    class FakeEntityWithVelocity:
        def __init__(self):
            self.calls = []

        def zero_all_dofs_velocity(self, envs_idx):
            self.calls.append(list(envs_idx))

    entity = FakeEntityWithVelocity()
    fn = reset.zero_all_dofs_velocity()
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0, 2])

    assert entity.calls == [[0, 2]]


"""
set_rotation
"""


def test_set_rotation_only_randomizes_tuple_axes(env, monkeypatch):
    fn = reset.set_rotation(x=0, y=(0.0, 0.0), z=0)
    fn.context(env, entity=FakeEntity())
    fn.safe_build()
    entity = fn.entity

    fn(env, entity, [0, 1])

    quat, envs_idx, _ = entity.quat_calls[0]
    assert envs_idx == [0, 1]
    assert quat.shape == (2, 4)


def test_set_rotation_fixed_angles_produce_identity_quat(env):
    entity = FakeEntity()
    fn = reset.set_rotation(x=0, y=0, z=0)
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0])

    quat, _, _ = entity.quat_calls[0]
    assert torch.allclose(quat, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-5)


"""
position
"""


def test_position_sets_pos_and_quat_for_the_given_envs(env):
    entity = FakeEntity()
    fn = reset.position(position=(1.0, 2.0, 3.0), quat=(1.0, 0.0, 0.0, 0.0))
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [1, 3])

    pos, envs_idx, zero_vel = entity.pos_calls[0]
    assert envs_idx == [1, 3]
    assert zero_vel is True
    assert torch.allclose(pos, torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))

    quat, quat_envs_idx, _ = entity.quat_calls[0]
    assert quat_envs_idx == [1, 3]
    assert torch.allclose(quat, torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))


def test_position_skips_quat_when_not_provided(env):
    entity = FakeEntity()
    fn = reset.position(position=(0.0, 0.0, 0.4))
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0])

    assert len(entity.pos_calls) == 1
    assert len(entity.quat_calls) == 0


def test_position_param_change_rebuilds_the_target(env):
    entity = FakeEntity()
    fn = reset.position(position=(0.0, 0.0, 0.4))
    fn.context(env, entity=entity)
    fn.safe_build()

    fn.position = (0.0, 0.0, 0.6)
    fn(env, entity, [0])

    pos, _, _ = entity.pos_calls[0]
    assert torch.allclose(pos, torch.tensor([[0.0, 0.0, 0.6]]))


"""
randomize_terrain_position
"""


def test_randomize_terrain_position_default_rotation_is_z_only(env):
    """Each instance gets its own rotation dict -- a shared mutable default across
    instances would be a real bug (the field(default_factory=...) fix for it)."""
    entity = FakeEntity()
    terrain = FakeTerrainManager(torch.tensor([[1.0, 2.0, 3.0]]))
    fn = reset.randomize_terrain_position(terrain_manager=terrain)
    fn.context(env, entity=entity)
    fn.safe_build()

    assert fn.rotation == {"z": (0, 2 * math.pi)}

    fn(env, entity, [0])

    assert entity.pos_calls[0][0].tolist() == [[1.0, 2.0, 3.0]]
    assert len(entity.quat_calls) == 1  # z rotation was applied


def test_randomize_terrain_position_instances_do_not_share_the_default_dict(env):
    terrain = FakeTerrainManager(torch.zeros((1, 3)))
    a = reset.randomize_terrain_position(terrain_manager=terrain)
    b = reset.randomize_terrain_position(terrain_manager=terrain)
    a.context(env, entity=FakeEntity())
    a.safe_build()
    b.context(env, entity=FakeEntity())
    b.safe_build()

    a.rotation["z"] = (1.0, 1.0)

    assert b.rotation == {"z": (0, 2 * math.pi)}


def test_randomize_terrain_position_none_rotation_skips_quat(env):
    entity = FakeEntity()
    terrain = FakeTerrainManager(torch.zeros((1, 3)))
    fn = reset.randomize_terrain_position(terrain_manager=terrain, rotation=None)
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0])

    assert len(entity.quat_calls) == 0


def test_randomize_terrain_position_calls_subterrain_when_callable(env):
    entity = FakeEntity()
    terrain = FakeTerrainManager(torch.zeros((1, 3)))
    fn = reset.randomize_terrain_position(
        terrain_manager=terrain, subterrain=lambda: "rocky"
    )
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0])

    assert terrain.last_call[1] == "rocky"


"""
randomize_annulus_position
"""


def test_randomize_annulus_position_respects_radius_range(env):
    env.num_envs = 64
    entity = FakeEntity()
    fn = reset.randomize_annulus_position(radius_range=(0.5, 2.0), center=(1.0, -1.0), z=0.1)
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, list(range(64)))

    pos, envs_idx, zero_vel = entity.pos_calls[0]
    assert envs_idx == list(range(64))
    assert zero_vel is True
    radii = torch.sqrt((pos[:, 0] - 1.0) ** 2 + (pos[:, 1] + 1.0) ** 2)
    assert torch.all(radii >= 0.5 - 1e-5)
    assert torch.all(radii <= 2.0 + 1e-5)
    assert torch.all(pos[:, 2] == 0.1)


def test_randomize_annulus_position_default_rotation_is_z_only(env):
    entity = FakeEntity()
    fn = reset.randomize_annulus_position(radius_range=(0.5, 2.0))
    fn.context(env, entity=entity)
    fn.safe_build()

    assert fn.rotation == {"z": (0, 2 * math.pi)}

    fn(env, entity, [0])

    assert len(entity.quat_calls) == 1


def test_randomize_annulus_position_none_rotation_skips_quat(env):
    entity = FakeEntity()
    fn = reset.randomize_annulus_position(radius_range=(0.5, 2.0), rotation=None)
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0])

    assert len(entity.quat_calls) == 0


def test_randomize_annulus_position_draws_fresh_positions_each_call(env):
    entity = FakeEntity()
    fn = reset.randomize_annulus_position(radius_range=(0.5, 2.0))
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0, 1, 2, 3])
    fn(env, entity, [0, 1, 2, 3])

    first, _, _ = entity.pos_calls[0]
    second, _, _ = entity.pos_calls[1]
    assert not torch.equal(first, second)


"""
randomize_link_mass_shift
"""


def test_randomize_link_mass_shift_raises_when_no_links_found(env):
    entity = FakeEntity()
    fn = reset.randomize_link_mass_shift(link_name="nonexistent_.*", mass_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="No links found"):
        fn.context(env, entity=entity)
        fn.safe_build()


def test_randomize_link_mass_shift_applies_to_matched_links(env, monkeypatch):
    class FakeLink:
        def __init__(self, idx_local):
            self.idx_local = idx_local

    monkeypatch.setattr(
        "genesis_forge.mdp.reset.links_by_name_pattern",
        lambda entity, pattern: [FakeLink(2), FakeLink(5)],
    )

    entity = FakeEntity()
    fn = reset.randomize_link_mass_shift(link_name="base", mass_range=(-1.0, 1.0))
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0, 1])

    mass, links_idx, envs_idx = entity.mass_shift_calls[0]
    assert links_idx == [2, 5]
    assert envs_idx == [0, 1]
    assert mass.shape == (2, 2)
    assert torch.all(mass >= -1.0) and torch.all(mass <= 1.0)


def test_randomize_link_mass_shift_accepts_a_list_of_patterns(env, monkeypatch):
    class FakeLink:
        def __init__(self, idx_local):
            self.idx_local = idx_local

    def fake_links_by_name_pattern(entity, pattern):
        return {
            "front": [FakeLink(2), FakeLink(3)],
            "back": [FakeLink(7)],
        }[pattern]

    monkeypatch.setattr(
        "genesis_forge.mdp.reset.links_by_name_pattern", fake_links_by_name_pattern
    )

    entity = FakeEntity()
    fn = reset.randomize_link_mass_shift(
        link_name=["front", "back"], mass_range=(-1.0, 1.0)
    )
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0, 1])

    _, links_idx, _ = entity.mass_shift_calls[0]
    assert links_idx == [2, 3, 7]


def test_randomize_link_mass_shift_raises_when_one_of_several_patterns_matches_nothing(
    env, monkeypatch
):
    class FakeLink:
        def __init__(self, idx_local):
            self.idx_local = idx_local

    def fake_links_by_name_pattern(entity, pattern):
        return [FakeLink(2)] if pattern == "front" else []

    monkeypatch.setattr(
        "genesis_forge.mdp.reset.links_by_name_pattern", fake_links_by_name_pattern
    )

    entity = FakeEntity()
    fn = reset.randomize_link_mass_shift(
        link_name=["front", "nonexistent"], mass_range=(-1.0, 1.0)
    )
    with pytest.raises(ValueError, match="No links found with name/pattern 'nonexistent'"):
        fn.context(env, entity=entity)
        fn.safe_build()


def test_randomize_link_mass_shift_draws_fresh_values_on_each_call(env, monkeypatch):
    """Regression guard: mass_shift used to be sliced out of a persistent buffer via
    advanced indexing, which writes to a copy and silently no-ops the randomization.
    It's now a freshly allocated tensor per call -- confirm two calls actually vary."""
    class FakeLink:
        def __init__(self, idx_local):
            self.idx_local = idx_local

    monkeypatch.setattr(
        "genesis_forge.mdp.reset.links_by_name_pattern",
        lambda entity, pattern: [FakeLink(2)],
    )

    entity = FakeEntity()
    fn = reset.randomize_link_mass_shift(link_name="base", mass_range=(-1.0, 1.0))
    fn.context(env, entity=entity)
    fn.safe_build()

    fn(env, entity, [0, 1, 2, 3])
    fn(env, entity, [0, 1, 2, 3])

    first, _, _ = entity.mass_shift_calls[0]
    second, _, _ = entity.mass_shift_calls[1]
    assert not torch.equal(first, second)
