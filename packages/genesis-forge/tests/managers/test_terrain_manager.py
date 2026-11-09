"""Behavior of TerrainManager: mapping terrain/subterrain bounds from the morph
(or falling back to the AABB for non-terrain morphs) or from a TerrainLayout,
subterrain naming and selection, height queries, and random position generation
within the usable area.

Uses a FakeTerrain/FakeGeom/FakeMorph -- no Genesis scene is built. A FakeTerrain
with a height field stands in for a built Genesis terrain entity, with its own
height query.
"""

import genesis as gs
import pytest
import torch

from genesis_forge import terrain
from genesis_forge.managers import TerrainManager


class FakeGeom:
    def __init__(self, aabb, pos):
        """Store the AABB and position the manager reads for non-terrain ground."""
        self._aabb = aabb
        self._pos = pos

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
        height_field=None,
        horizontal_scale=1.0,
    ):
        self.pos = pos
        self.n_subterrains = n_subterrains
        self.subterrain_size = subterrain_size
        self.subterrain_types = subterrain_types
        self.vertical_scale = vertical_scale
        self.height_field = height_field
        self.horizontal_scale = horizontal_scale


class FakeScene:
    def __init__(self, n_envs):
        """A scene with ``n_envs`` parallel environments (0 when not parallel)."""
        self.n_envs = n_envs


class FakeTerrain:
    """
    A terrain entity. Given a height field, it is a Genesis terrain, with a
    height query that returns ``height_fn(x, y)`` for one environment.
    """

    def __init__(self, geoms, morph, height_field=None, height_fn=None, n_envs=4):
        """A terrain entity; with a height field, it acts as a built Genesis terrain."""
        self.geoms = geoms
        self.morph = morph
        self.terrain_hf = height_field
        self.scene = FakeScene(n_envs)
        self._height_fn = height_fn or (lambda x, y: torch.zeros_like(x))
        self.height_calls = []

    def get_terrain_height(self, positions, envs_idx=None):
        """Record the query, and return ``height_fn`` shaped like Genesis' result."""
        self.height_calls.append((positions.clone(), envs_idx))
        heights = self._height_fn(positions[:, 0], positions[:, 1])
        return heights.unsqueeze(0) if envs_idx is not None else heights


def make_plane(z=0.0):
    """A ground plane spanning x:[0,10], y:[0,10] -- the AABB path."""
    aabb = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    pos = torch.tensor([0.0, 0.0, z])
    geom = FakeGeom(aabb=aabb, pos=pos)
    return FakeTerrain(geoms=[geom], morph=None)


def make_genesis_terrain(height_fn=None, n_envs=4):
    """A single 10x10 Genesis terrain subterrain at the origin."""
    geom = FakeGeom(aabb=None, pos=None)
    morph = FakeMorph(
        pos=(0.0, 0.0, 0.0),
        n_subterrains=(1, 1),
        subterrain_size=(10.0, 10.0),
        subterrain_types=[["flat"]],
    )
    return FakeTerrain(
        geoms=[geom],
        morph=morph,
        height_field=[[0.0]],
        height_fn=height_fn,
        n_envs=n_envs,
    )


def make_layout_terrain():
    """A Genesis terrain entity for a layout, which measures nothing from its morph."""
    return FakeTerrain(geoms=[], morph=None, height_field=[[0.0]])


def make_subterrain_setup(subterrain_types=None):
    """A 2x2 grid of 10x10 subterrains -- the terrain-morph path."""
    aabb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])  # unused in this branch
    pos = torch.tensor([0.0, 0.0, 0.0])
    geom = FakeGeom(aabb=aabb, pos=pos)
    morph = FakeMorph(
        pos=(0.0, 0.0, 0.0),
        n_subterrains=(2, 2),
        subterrain_size=(10.0, 10.0),
        subterrain_types=subterrain_types or [["flat", "rough"], ["stairs", "pyramid"]],
    )
    return FakeTerrain(geoms=[geom], morph=morph, height_field=[[0.0]])


def make_manager(env, terrain_entity):
    """A built TerrainManager for the given terrain entity, which is also the env's terrain."""
    env.terrain = terrain_entity
    mgr = TerrainManager(env, terrain=terrain_entity)
    mgr.build()
    return mgr


class LayoutScene:
    """A scene that a TerrainLayout adds its terrain to."""

    def add_entity(self, morph, **kwargs):
        """Return a terrain entity for the layout."""
        return make_layout_terrain()


@pytest.fixture(autouse=True)
def _fake_terrain_morph(monkeypatch):
    """Take the gs.morphs.Terrain options as they are, instead of validating a real morph."""
    monkeypatch.setattr(gs.morphs, "Terrain", lambda **kwargs: kwargs)


def make_layout(**kwargs):
    """A 2x2 layout of 2x2m subterrains, with flat ground at different heights."""
    return terrain.TerrainLayout(
        LayoutScene(),
        subterrain_size=(2.0, 2.0),
        horizontal_scale=0.5,
        vertical_scale=0.01,
        grid=[
            [
                terrain.Flat(name="low", height=0.0),
                terrain.Flat(name="mid", height=0.1),
            ],
            [
                terrain.Flat(name="high", height=0.2),
                terrain.Flat(name="top", height=0.3),
            ],
        ],
        **kwargs,
    )


def in_bounds(pos, bounds):
    """Whether every position lies within ``(x_min, x_max, y_min, y_max)``."""
    x_min, x_max, y_min, y_max = bounds
    return bool(
        torch.all((pos[:, 0] >= x_min) & (pos[:, 0] <= x_max))
        and torch.all((pos[:, 1] >= y_min) & (pos[:, 1] <= y_max))
    )


"""
build() -- bounds mapping
"""


def test_build_maps_a_plane_from_its_aabb(env):
    """A plane has no subterrains, so its bounds come from its AABB."""
    env.terrain = make_plane(z=0.7)
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


def test_get_bounds_of_an_unknown_subterrain_is_the_whole_terrain(env):
    """As in 1.0, an unknown name gives the whole terrain, now with a warning."""
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.warns(UserWarning, match="Valid names"):
        assert mgr.get_bounds("nonexistent") == (0.0, 20.0, 0.0, 20.0)


def test_build_maps_a_height_field_morph_to_its_own_size(env):
    """A height-field terrain ignores n_subterrains, which defaults to (3, 3) in Genesis."""
    geom = FakeGeom(aabb=None, pos=None)
    morph = FakeMorph(
        pos=(1.0, 2.0, 0.0),
        n_subterrains=(3, 3),
        subterrain_size=(12.0, 12.0),
        height_field=[[0.0] * 5] * 9,
        horizontal_scale=0.5,
    )
    mgr = make_manager(
        env, FakeTerrain(geoms=[geom], morph=morph, height_field=[[0.0]])
    )

    assert mgr.get_bounds() == (1.0, 5.0, 2.0, 4.0)
    assert mgr.subterrain_names == []


"""
Subterrain naming and lookup
"""


def test_unique_types_are_named_after_their_type(env):
    """In a gs.morphs.Terrain grid, a type used once is the subterrain's name."""
    mgr = make_manager(env, make_subterrain_setup())

    assert mgr.subterrain_names == ["flat", "rough", "stairs", "pyramid"]
    assert mgr.index("stairs") == 2


def test_repeated_types_are_numbered_in_grid_order(env):
    """A type used three times is numbered flat_0 to flat_2, in grid order."""
    mgr = make_manager(
        env,
        make_subterrain_setup([["flat", "rough"], ["flat", "flat"]]),
    )

    assert mgr.subterrain_names == ["flat_0", "rough", "flat_1", "flat_2"]
    assert mgr.terrain_map.types == ("flat", "rough", "flat", "flat")
    assert mgr.get_bounds("flat_1") == (10.0, 20.0, 0.0, 10.0)


def test_get_bounds_of_a_repeated_type_name_is_the_last_one(env):
    """As in 1.0, a shared type name gives the last subterrain of that type, with a warning."""
    mgr = make_manager(env, make_subterrain_setup([["flat", "rough"], ["flat", "x"]]))

    with pytest.warns(UserWarning, match="flat_1"):
        assert mgr.get_bounds("flat") == (10.0, 20.0, 0.0, 10.0)


def test_a_repeated_type_name_places_on_any_of_its_subterrains(env):
    """Placing robots on a shared type name samples from every subterrain of that type."""
    mgr = make_manager(env, make_subterrain_setup([["flat", "rough"], ["flat", "x"]]))

    idx = mgr.sample_subterrains(200, "flat")

    assert set(idx.tolist()) == {0, 2}


def test_indices_rejects_an_unknown_name(env):
    """One unknown name in a list fails the whole selection."""
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.raises(KeyError, match="'typo'"):
        mgr.indices(["flat", "typo"])


def test_indices_rejects_an_empty_selection(env):
    """An empty list selects nothing, so it's rejected."""
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.raises(ValueError):
        mgr.indices([])


def test_get_subterrain_bounds_gathers_by_index_and_minus_one_is_the_whole_terrain(env):
    """Bounds are looked up per index, and -1 gives the whole terrain."""
    mgr = make_manager(env, make_subterrain_setup())

    bounds = mgr.get_subterrain_bounds(torch.tensor([3, -1]))

    assert bounds.tolist() == [[10.0, 20.0, 10.0, 20.0], [0.0, 20.0, 0.0, 20.0]]


def test_get_subterrain_bounds_by_name_gives_every_subterrain_it_selects(env):
    """A shared name, or a list of names, gives one row of bounds per subterrain."""
    mgr = make_manager(env, make_subterrain_setup([["flat", "rough"], ["flat", "x"]]))

    assert mgr.get_subterrain_bounds("flat").tolist() == [
        [0.0, 10.0, 0.0, 10.0],
        [10.0, 20.0, 0.0, 10.0],
    ]
    assert mgr.get_subterrain_bounds(["rough", "x"]).tolist() == [
        [0.0, 10.0, 10.0, 20.0],
        [10.0, 20.0, 10.0, 20.0],
    ]


def test_get_subterrain_bounds_of_none_or_an_unknown_name_is_the_whole_terrain(env):
    """None gives the whole terrain, and so does an unknown name, with a warning."""
    mgr = make_manager(env, make_subterrain_setup())

    assert mgr.get_subterrain_bounds(None).tolist() == [[0.0, 20.0, 0.0, 20.0]]
    with pytest.warns(UserWarning, match="'typo'"):
        assert mgr.get_subterrain_bounds("typo").tolist() == [[0.0, 20.0, 0.0, 20.0]]


"""
validate_selection()
"""


def test_validate_selection_accepts_every_form(env):
    """None, a name, a list of names, and an index tensor all pass."""
    mgr = make_manager(env, make_subterrain_setup())

    mgr.validate_selection(None)
    mgr.validate_selection("flat")
    mgr.validate_selection(["flat", "rough"])
    mgr.validate_selection(torch.tensor([0, 3, -1, 1]), num=4)


@pytest.mark.parametrize(
    "selection, error",
    [
        (["flat", "typo"], KeyError),
        (["flat", 3], TypeError),
        ({"flat": 1.0}, TypeError),
        (torch.tensor([0.0, 1.0, 0.0, 1.0]), TypeError),
        (torch.tensor([0, 1]), ValueError),
        (torch.tensor([0, 1, 4, 0]), ValueError),
        (torch.tensor([0, 1, -2, 0]), ValueError),
    ],
)
def test_validate_selection_rejects_bad_selections(env, selection, error):
    """
    An unknown name in a list, wrong types, a dict, a float tensor, a wrong shape, and
    out-of-range indices all fail.
    """
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.raises(error):
        mgr.validate_selection(selection, num=4)


"""
get_terrain_height()
"""


def test_get_terrain_height_on_a_plane_is_the_plane_height(env):
    """A flat plane's height is its z position everywhere."""
    mgr = make_manager(env, make_plane(z=0.7))

    heights = mgr.get_terrain_height(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
    assert torch.allclose(heights, torch.tensor([0.7, 0.7]))


def test_validate_selection_accepts_an_unknown_single_name_with_a_warning(env):
    """As in 1.0, an unknown name stands for the whole terrain, so it only warns."""
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.warns(UserWarning, match="'typo'"):
        mgr.validate_selection("typo")


def test_get_terrain_height_on_other_ground_is_its_z_position(env):
    """As in 1.0, ground with no height query, like a box or a URDF plane, is flat at its z."""
    aabb = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 1.0]])
    geom = FakeGeom(aabb=aabb, pos=torch.tensor([[5.0, 5.0, 0.5], [5.0, 5.0, 0.9]]))
    mgr = make_manager(env, FakeTerrain(geoms=[geom], morph=FakeMorph()))

    assert mgr.get_bounds() == (0.0, 10.0, 0.0, 10.0)
    heights = mgr.get_terrain_height(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert torch.allclose(
        heights, torch.tensor([0.5, 0.5])
    )  # <~ the first environment's


def test_get_terrain_height_uses_the_genesis_query_on_a_terrain(env):
    """On a Genesis terrain, heights come from Genesis, querying only the first environment."""
    entity = make_genesis_terrain(height_fn=lambda x, y: x + y)
    mgr = make_manager(env, entity)

    heights = mgr.get_terrain_height(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

    assert heights.tolist() == [4.0, 6.0]
    ((_, envs_idx),) = entity.height_calls
    assert envs_idx == 0  # only the first env is queried


def test_get_terrain_height_clamps_positions_to_the_terrain_for_genesis(env):
    """Genesis returns NaN far outside the terrain; the edge height is used instead."""
    entity = make_genesis_terrain()
    mgr = make_manager(env, entity)

    mgr.get_terrain_height(torch.tensor([-5.0, 50.0]), torch.tensor([5.0, -1.0]))

    ((positions, _),) = entity.height_calls
    assert positions.tolist() == [[0.0, 5.0], [10.0, 0.0]]


def test_get_terrain_height_handles_a_non_parallel_scene(env):
    """Without parallel environments, Genesis is queried with no envs_idx."""
    entity = make_genesis_terrain(height_fn=lambda x, y: x + y, n_envs=0)
    mgr = make_manager(env, entity)

    heights = mgr.get_terrain_height(torch.tensor([1.0]), torch.tensor([1.0]))

    assert heights.tolist() == [2.0]
    assert entity.height_calls[0][1] is None


def test_get_terrain_height_skips_genesis_for_no_positions(env):
    """Genesis rejects an empty query, as from a reset of no environments."""
    entity = make_genesis_terrain()
    mgr = make_manager(env, entity)

    heights = mgr.get_terrain_height(torch.tensor([]), torch.tensor([]))

    assert heights.shape == (0,)
    assert entity.height_calls == []


"""
generate_random_positions()
"""


def test_generate_random_positions_stays_within_the_usable_ratio_bounds(env):
    env.terrain = make_plane()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(
        num=env.num_envs, usable_ratio=0.5, height_offset=0.0
    )

    assert torch.all(output[:, 0] >= 2.5) and torch.all(output[:, 0] <= 7.5)
    assert torch.all(output[:, 1] >= 2.5) and torch.all(output[:, 1] <= 7.5)
    assert torch.all(output[:, 2] == 0.0)  # flat terrain, no offset


def test_generate_random_positions_applies_the_height_offset(env):
    env.terrain = make_plane(z=1.0)
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(num=env.num_envs, height_offset=0.05)

    assert torch.allclose(output[:, 2], torch.full((env.num_envs,), 1.05))


def test_generate_random_positions_requires_output_or_num(env):
    env.terrain = make_plane()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    with pytest.raises(AssertionError):
        mgr.generate_random_positions()


def test_generate_random_positions_writes_only_to_the_given_out_idx(env):
    env.terrain = make_plane()
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
    env.terrain = make_plane()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    output = mgr.generate_random_positions(num=50, usable_ratio=0.5, height_offset=0.0)

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
    env.terrain = make_plane()
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
    env.terrain = make_plane()
    mgr = TerrainManager(env, terrain=env.terrain)
    mgr.build()

    result = mgr.generate_random_env_pos()

    assert result.shape == (env.num_envs, 3)


def test_generate_random_env_pos_places_each_env_on_its_tensor_subterrain(env):
    """A per-env index tensor is read at envs_idx, placing each env on its own subterrain."""
    mgr = make_manager(env, make_subterrain_setup())
    per_env = torch.tensor([0, 1, 2, 3])

    result = mgr.generate_random_env_pos(
        envs_idx=[1, 3], subterrain=per_env, usable_ratio=1.0
    )

    assert in_bounds(result[[0]], mgr.get_bounds("rough"))
    assert in_bounds(result[[1]], mgr.get_bounds("pyramid"))


def test_generate_random_env_pos_records_each_envs_subterrain(env):
    """
    Each placement updates env_subterrain for those envs only, and -1 means the whole
    terrain.
    """
    mgr = make_manager(env, make_subterrain_setup())

    mgr.generate_random_env_pos(envs_idx=[0, 2], subterrain="stairs")

    assert mgr.env_subterrain.tolist() == [2, -1, 2, -1]

    mgr.generate_random_env_pos(envs_idx=[2], subterrain=None)

    assert mgr.env_subterrain.tolist() == [2, -1, -1, -1]


def test_generate_random_env_pos_requires_a_tensor_with_one_entry_per_env(env):
    """A per-env tensor needs one entry for every env, not one per placed env."""
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.raises(ValueError):
        mgr.generate_random_env_pos(envs_idx=[0, 1], subterrain=torch.tensor([0, 1]))


"""
TerrainLayout-backed manager
"""


def test_a_layout_names_the_subterrains(env):
    """With a layout, names, types and bounds come from the layout."""
    layout = make_layout()
    mgr = TerrainManager(env, layout=layout)
    mgr.build()

    assert mgr.subterrain_names == ["low", "mid", "high", "top"]
    assert mgr.terrain_map.types == ("flat_terrain",) * 4
    assert mgr.get_bounds() == (0.0, 4.0, 0.0, 4.0)
    assert mgr.get_bounds("high") == (2.0, 4.0, 0.0, 2.0)


def test_a_repeated_layout_name_selects_any_of_its_subterrains(env):
    """
    A name used twice in a layout is numbered, and the shared name places robots on either
    one.
    """
    rough = terrain.Flat(name="rough")
    layout = terrain.TerrainLayout(
        LayoutScene(),
        subterrain_size=(2.0, 2.0),
        horizontal_scale=0.5,
        grid=[[rough, terrain.Flat(name="start")], [terrain.Flat(name="end"), rough]],
    )
    mgr = TerrainManager(env, layout=layout)
    mgr.build()

    assert mgr.subterrain_names == ["rough_0", "start", "end", "rough_1"]
    assert mgr.indices("rough").tolist() == [0, 3]
    assert mgr.get_bounds("rough_1") == (2.0, 4.0, 2.0, 4.0)
    with pytest.raises(KeyError, match="rough_0"):
        mgr.index("rough")

    mgr.generate_random_env_pos(subterrain="rough", height_offset=0.0)
    assert set(mgr.env_subterrain.tolist()) <= {0, 3}


def test_a_layout_terrain_is_found_on_the_layout(env):
    """With only layout=, the manager uses the terrain the layout added to the scene."""
    layout = make_layout()

    mgr = TerrainManager(env, layout=layout)
    assert mgr.terrain is layout.entity


def test_terrain_and_layout_cannot_both_be_passed(env):
    """The manager takes its terrain from one place: the entity, or the layout."""
    with pytest.raises(ValueError, match="not both"):
        TerrainManager(env, terrain=make_layout_terrain(), layout=make_layout())


def test_a_layout_places_envs_on_named_subterrains(env):
    """A list of layout names places each env on one of them, and records which."""
    mgr = TerrainManager(env, layout=make_layout())
    mgr.build()

    result = mgr.generate_random_env_pos(
        subterrain=["mid", "top"], usable_ratio=0.5, height_offset=0.0
    )

    on_mid = (result[:, 0] < 2.0) & (result[:, 1] > 2.0)
    on_top = (result[:, 0] > 2.0) & (result[:, 1] > 2.0)
    assert torch.all(on_mid | on_top)
    assert set(mgr.env_subterrain.tolist()) <= {1, 3}


def test_generate_random_positions_spreads_a_list_across_its_subterrains(env):
    """A list of names spreads positions across every listed subterrain."""
    mgr = make_manager(env, make_subterrain_setup())

    output = mgr.generate_random_positions(
        num=400, subterrain=["flat", "pyramid"], usable_ratio=1.0, height_offset=0.0
    )

    on_flat = (output[:, 0] <= 10.0) & (output[:, 1] <= 10.0)
    on_pyramid = (output[:, 0] >= 10.0) & (output[:, 1] >= 10.0)
    assert torch.all(on_flat | on_pyramid)
    assert torch.any(on_flat) and torch.any(on_pyramid)


def test_generate_random_positions_on_an_unknown_subterrain_uses_the_whole_terrain(env):
    """As in 1.0, an unknown name places anywhere on the terrain, now with a warning."""
    mgr = make_manager(env, make_subterrain_setup())

    with pytest.warns(UserWarning, match="nonexistent"):
        output = mgr.generate_random_positions(
            num=400, subterrain="nonexistent", usable_ratio=1.0
        )

    assert in_bounds(output, mgr.get_bounds())
    assert torch.any(output[:, 0] > 10.0) and torch.any(output[:, 0] < 10.0)


def test_generate_random_positions_uses_a_tensor_index_per_position(env):
    """A tensor gives one subterrain index for each generated position."""
    mgr = make_manager(env, make_subterrain_setup())

    output = mgr.generate_random_positions(
        num=3, subterrain=torch.tensor([3, 0, 3]), usable_ratio=1.0
    )

    assert in_bounds(output[[0, 2]], mgr.get_bounds("pyramid"))
    assert in_bounds(output[[1]], mgr.get_bounds("flat"))


def test_indices_resolves_each_selection_once(env):
    """A repeated selection returns the cached tensor instead of resolving the names again."""
    mgr = make_manager(env, make_subterrain_setup())

    assert mgr.indices(["flat", "rough"]) is mgr.indices(["flat", "rough"])
    assert mgr.indices(["flat", "rough"]).tolist() == [0, 1]
