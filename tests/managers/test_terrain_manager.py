"""Behavior of TerrainManager: mapping terrain/subterrain bounds from the morph
(or falling back to the AABB for non-terrain morphs), height-field interpolation,
and random position generation within the usable area.

Uses a FakeTerrain/FakeGeom/FakeMorph -- no Genesis scene is built. F.grid_sample's
corner semantics (align_corners=True) were verified directly against torch before
writing the interpolation test, rather than assumed.
"""

import pytest
import torch

from genesis_forge.managers import TerrainManager


class FakeGeom:
    def __init__(self, aabb, pos, metadata=None):
        self._aabb = aabb
        self._pos = pos
        self.metadata = metadata or {}

    def get_AABB(self):
        return self._aabb

    def get_pos(self):
        return self._pos


class FakeMorph:
    def __init__(
        self,
        pos=None,
        n_subterrains=None,
        subterrain_size=None,
        subterrain_types=None,
        vertical_scale=1.0,
    ):
        self.pos = pos
        self.n_subterrains = n_subterrains
        self.subterrain_size = subterrain_size
        self.subterrain_types = subterrain_types
        self.vertical_scale = vertical_scale


class FakeTerrain:
    def __init__(self, geoms, morph):
        self.geoms = geoms
        self.morph = morph


def make_flat_terrain(z=0.0):
    """A single terrain block spanning x:[0,10], y:[0,10] -- the AABB fallback path."""
    aabb = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    pos = torch.tensor([0.0, 0.0, z])
    geom = FakeGeom(aabb=aabb, pos=pos)
    return FakeTerrain(geoms=[geom], morph=FakeMorph())


def make_subterrain_setup():
    """A 2x2 grid of 10x10 subterrains -- the terrain-morph path."""
    aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])  # unused in this branch
    pos = torch.tensor([0.0, 0.0, 0.0])
    geom = FakeGeom(aabb=aabb, pos=pos)
    morph = FakeMorph(
        pos=(0.0, 0.0, 0.0),
        n_subterrains=(2, 2),
        subterrain_size=(10.0, 10.0),
        subterrain_types=[["flat", "rough"], ["stairs", "pyramid"]],
    )
    return FakeTerrain(geoms=[geom], morph=morph)


"""
build() -- bounds mapping
"""


def test_build_falls_back_to_the_aabb_for_a_non_terrain_morph(env):
    env.terrain = make_flat_terrain(z=0.7)
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    assert mgr.get_bounds() == (0.0, 10.0, 0.0, 10.0)


def test_build_maps_bounds_from_the_terrain_morph(env):
    env.terrain = make_subterrain_setup()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    assert mgr.get_bounds() == (0.0, 20.0, 0.0, 20.0)


def test_build_maps_each_subterrains_bounds(env):
    env.terrain = make_subterrain_setup()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    assert mgr.get_bounds("flat") == (0.0, 10.0, 0.0, 10.0)
    assert mgr.get_bounds("rough") == (0.0, 10.0, 10.0, 20.0)
    assert mgr.get_bounds("stairs") == (10.0, 20.0, 0.0, 10.0)
    assert mgr.get_bounds("pyramid") == (10.0, 20.0, 10.0, 20.0)


def test_get_bounds_falls_back_to_full_bounds_for_an_unknown_subterrain(env):
    env.terrain = make_subterrain_setup()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    assert mgr.get_bounds("nonexistent") == mgr.get_bounds()


"""
get_terrain_height()
"""


def test_get_terrain_height_without_a_height_field_returns_the_origin_z(env):
    env.terrain = make_flat_terrain(z=0.7)
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    heights = mgr.get_terrain_height(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    assert torch.equal(heights, torch.tensor([0.7, 0.7]))


def test_get_terrain_height_interpolates_the_height_field(env):
    aabb = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    pos = torch.tensor([0.0, 0.0, 0.0])
    # Stored as (width, height); TerrainManager transposes it to (height, width) =
    # [[0, 10], [20, 30]] -- i.e. corners (x_min,y_min)=0, (x_max,y_min)=10,
    # (x_min,y_max)=20, (x_max,y_max)=30, center=mean of all four=15.
    height_field = torch.tensor([[0.0, 20.0], [10.0, 30.0]])
    geom = FakeGeom(aabb=aabb, pos=pos, metadata={"height_field": height_field})
    env.terrain = FakeTerrain(geoms=[geom], morph=FakeMorph(vertical_scale=1.0))
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    x = torch.tensor([0.0, 10.0, 0.0, 10.0])
    y = torch.tensor([0.0, 0.0, 10.0, 10.0])
    heights = mgr.get_terrain_height(x, y)

    assert torch.allclose(heights, torch.tensor([0.0, 10.0, 20.0, 30.0]), atol=1e-4)


def test_get_terrain_height_is_independent_of_num_envs(env):
    """get_terrain_height's query batch size (x.shape[0]) has no relationship to
    env.num_envs (4 here) -- it must work for any number of query points."""
    aabb = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    pos = torch.tensor([0.0, 0.0, 0.0])
    height_field = torch.tensor([[0.0, 20.0], [10.0, 30.0]])
    geom = FakeGeom(aabb=aabb, pos=pos, metadata={"height_field": height_field})
    env.terrain = FakeTerrain(geoms=[geom], morph=FakeMorph(vertical_scale=1.0))
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    x = torch.tensor([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    y = torch.tensor([0.0, 0.0, 10.0, 10.0, 0.0, 0.0])
    heights = mgr.get_terrain_height(x, y)

    assert torch.allclose(
        heights, torch.tensor([0.0, 10.0, 20.0, 30.0, 0.0, 10.0]), atol=1e-4
    )


def test_get_terrain_height_applies_the_vertical_scale(env):
    aabb = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    pos = torch.tensor([0.0, 0.0, 0.0])
    height_field = torch.full((2, 2), 5.0)
    geom = FakeGeom(aabb=aabb, pos=pos, metadata={"height_field": height_field})
    env.terrain = FakeTerrain(geoms=[geom], morph=FakeMorph(vertical_scale=2.0))
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    heights = mgr.get_terrain_height(torch.tensor([5.0]), torch.tensor([5.0]))
    assert torch.allclose(heights, torch.tensor([10.0]))  # 5.0 * vertical_scale(2.0)


"""
generate_random_positions()
"""


def test_generate_random_positions_stays_within_the_usable_ratio_bounds(env):
    env.terrain = make_flat_terrain()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(num=env.num_envs, usable_ratio=0.5, height_offset=0.0)

    assert torch.all(output[:, 0] >= 2.5) and torch.all(output[:, 0] <= 7.5)
    assert torch.all(output[:, 1] >= 2.5) and torch.all(output[:, 1] <= 7.5)
    assert torch.all(output[:, 2] == 0.0)  # flat terrain, no offset


def test_generate_random_positions_applies_the_height_offset(env):
    env.terrain = make_flat_terrain(z=1.0)
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(num=env.num_envs, height_offset=0.05)

    assert torch.allclose(output[:, 2], torch.full((env.num_envs,), 1.05))


def test_generate_random_positions_requires_output_or_num(env):
    env.terrain = make_flat_terrain()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    with pytest.raises(AssertionError):
        mgr.generate_random_positions()


def test_generate_random_positions_writes_only_to_the_given_out_idx(env):
    env.terrain = make_flat_terrain()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = torch.full((4, 3), -1.0)
    mgr.generate_random_positions(output=output, out_idx=[1, 3], height_offset=0.0)

    assert torch.all(output[0] == -1.0)
    assert torch.all(output[2] == -1.0)
    assert not torch.any(output[1] == -1.0)
    assert not torch.any(output[3] == -1.0)


def test_generate_random_positions_supports_num_greater_than_num_envs(env):
    """num is independent of env.num_envs -- generate_random_positions is a
    general-purpose "N random positions" utility, not limited to one per env."""
    env.terrain = make_flat_terrain()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(
        num=50, usable_ratio=0.5, height_offset=0.0
    )

    assert output.shape == (50, 3)
    assert torch.all(output[:, 0] >= 2.5) and torch.all(output[:, 0] <= 7.5)
    assert torch.all(output[:, 1] >= 2.5) and torch.all(output[:, 1] <= 7.5)
    assert torch.all(output[:, 2] == 0.0)  # flat terrain, no offset


def test_generate_random_positions_uses_the_subterrain_bounds(env):
    env.terrain = make_subterrain_setup()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(
        num=env.num_envs, subterrain="stairs", usable_ratio=1.0, height_offset=0.0
    )

    assert torch.all(output[:, 0] >= 10.0) and torch.all(output[:, 0] <= 20.0)
    assert torch.all(output[:, 1] >= 0.0) and torch.all(output[:, 1] <= 10.0)


"""
generate_random_env_pos()
"""


def test_generate_random_env_pos_updates_only_the_requested_envs(env):
    env.terrain = make_flat_terrain()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()
    mgr._env_pos_buffer[:] = -1.0

    result = mgr.generate_random_env_pos(envs_idx=[0, 2], height_offset=0.0)

    assert result.shape == (2, 3)
    assert torch.all(mgr._env_pos_buffer[1] == -1.0)
    assert torch.all(mgr._env_pos_buffer[3] == -1.0)
    assert not torch.any(mgr._env_pos_buffer[0] == -1.0)
    assert not torch.any(mgr._env_pos_buffer[2] == -1.0)


def test_generate_random_env_pos_defaults_to_every_env(env):
    env.terrain = make_flat_terrain()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    result = mgr.generate_random_env_pos()

    assert result.shape == (env.num_envs, 3)
