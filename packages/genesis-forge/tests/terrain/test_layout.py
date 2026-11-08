"""Behavior of TerrainLayout: config validation, grid naming and bounds, per-subterrain
seeding, height field stitching, and adding the terrain to a scene.

Height fields are generated with the real Genesis generators, but no scene is built:
a FakeScene records what the layout adds to it.
"""

from dataclasses import dataclass
from typing import ClassVar

import genesis as gs
import numpy as np
import pytest

from genesis_forge import terrain
from genesis_forge.terrain.layout import GRID_DARK, GRID_LIGHT, GRID_LINE


class FakeScene:
    def __init__(self):
        """Start with no recorded entities."""
        self.calls = []

    def add_entity(self, morph, **kwargs):
        """Record the morph and options, and return a stand-in entity."""
        self.calls.append((morph, kwargs))
        return "terrain-entity"


@pytest.fixture(autouse=True)
def _fake_terrain_morph(monkeypatch):
    """Capture the gs.morphs.Terrain options instead of validating a real morph."""
    monkeypatch.setattr(gs.morphs, "Terrain", lambda **kwargs: kwargs)


def make_layout(grid=None, scene=None, **kwargs):
    """
    A layout of 2m subterrains at a 0.5m scale, added to a FakeScene unless a scene is
    given.
    """
    kwargs.setdefault("subterrain_size", (2.0, 2.0))
    kwargs.setdefault("horizontal_scale", 0.5)
    if grid is None:
        grid = [[terrain.Flat(name="a"), terrain.Flat(name="b")]]
    return terrain.TerrainLayout(
        scene if scene is not None else FakeScene(), grid=grid, **kwargs
    )


def row(**subterrains):
    """A one-row grid, naming each subterrain after its keyword."""
    for name, subterrain in subterrains.items():
        subterrain.name = name
    return [list(subterrains.values())]


def rough(**kwargs):
    """A RandomUniform subterrain with fine steps, so its heights vary within a small layout."""
    kwargs.setdefault("min_height", 0.0)
    kwargs.setdefault("max_height", 0.1)
    kwargs.setdefault("step", 0.01)
    return terrain.RandomUniform(**kwargs)


def heights_of(layout, name):
    """The height field points of one subterrain, edges included."""
    i, j = layout.grid_position(name)
    rows, cols = layout._subterrain_shape()
    return layout.height_field[
        i * (rows - 1) : (i + 1) * (rows - 1) + 1,
        j * (cols - 1) : (j + 1) * (cols - 1) + 1,
    ]


"""
Validation
"""


def test_grid_rows_must_have_the_same_length():
    """A row shorter than the others is rejected."""
    with pytest.raises(ValueError, match="same"):
        make_layout(grid=[[terrain.Flat(), terrain.Flat()], [terrain.Flat()]])


def test_a_flat_list_is_one_row_along_y():
    """A flat list becomes a single row, laid out along the y axis."""
    layout = make_layout(grid=[terrain.Flat(name="a"), terrain.Flat(name="b")])

    assert layout.n_subterrains == (1, 2)
    assert layout.get_bounds("b") == (0.0, 2.0, 2.0, 4.0)


def test_a_grid_mixing_rows_and_subterrains_is_rejected():
    """A list mixing rows with single subterrains is ambiguous, so it's rejected with a hint."""
    with pytest.raises(ValueError, match="wrap it in a list"):
        make_layout(grid=[[terrain.Flat()], terrain.Flat()])


def test_a_class_in_a_flat_list_still_gets_a_hint():
    """A subterrain class in a flat list is still caught, with the hint to call it."""
    with pytest.raises(TypeError, match=r"Flat\(\)"):
        make_layout(grid=[terrain.Flat(), terrain.Flat])


def test_the_grid_must_not_be_empty():
    """An empty grid, or an empty row, is rejected."""
    with pytest.raises(ValueError):
        make_layout(grid=[])
    with pytest.raises(ValueError):
        make_layout(grid=[[]])


def test_a_dict_grid_is_rejected():
    """A dict of subterrains (the old API) is rejected as not a 2D list."""
    with pytest.raises(ValueError, match="2D list"):
        make_layout(grid={"a": terrain.Flat()})


def test_a_subterrain_class_is_rejected_with_a_hint():
    """Passing terrain.Flat instead of terrain.Flat() suggests calling it."""
    with pytest.raises(TypeError, match=r"Flat\(\)"):
        make_layout(grid=[[terrain.Flat]])


def test_a_non_subterrain_value_is_rejected():
    """A type name string is rejected, and the error points at its grid position."""
    with pytest.raises(TypeError, match=r"grid\[0\]\[1\]"):
        make_layout(grid=[[terrain.Flat(), "flat_terrain"]])


def test_names_must_be_non_empty_strings():
    """An empty name is rejected."""
    with pytest.raises(TypeError, match="non-empty"):
        make_layout(grid=[[terrain.Flat(name="")]])


def test_subterrain_size_must_be_a_multiple_of_the_horizontal_scale():
    """A 2.1m side doesn't fit a 0.5m scale, so it's rejected."""
    with pytest.raises(ValueError, match="multiple"):
        make_layout(subterrain_size=(2.0, 2.1))


def test_scales_must_be_positive():
    """A zero horizontal scale or a negative vertical scale is rejected."""
    with pytest.raises(ValueError):
        make_layout(horizontal_scale=0.0)
    with pytest.raises(ValueError):
        make_layout(vertical_scale=-1.0)


"""
Names and bounds
"""


def test_names_follow_grid_order():
    """Names are listed x row by x row, so index order matches grid order."""
    layout = make_layout(
        grid=[
            [terrain.Flat(name="d"), terrain.Flat(name="c")],
            [terrain.Flat(name="b"), terrain.Flat(name="a")],
        ]
    )

    assert layout.names == ("d", "c", "b", "a")
    assert layout.grid_position("b") == (1, 0)
    assert layout.groups == {}


def test_an_unnamed_subterrain_is_named_after_its_type():
    """Without a name, a subterrain uses its type, like gs.morphs.Terrain does."""
    layout = make_layout(grid=[[terrain.Flat(), terrain.Wave()]])

    assert layout.names == ("flat_terrain", "wave_terrain")


def test_a_repeated_name_is_numbered_in_grid_order():
    """
    Two objects sharing a name, one used twice, become three numbered subterrains in one
    group.
    """
    rough_a, rough_b = rough(name="rough"), rough(name="rough")
    layout = make_layout(
        grid=[[rough_a, terrain.Flat(name="start")], [rough_b, rough_a]]
    )

    assert layout.names == ("rough_0", "start", "rough_1", "rough_2")
    assert layout.groups == {"rough": ("rough_0", "rough_1", "rough_2")}
    assert layout.subterrains["rough_2"] is layout.subterrains["rough_0"]
    assert layout.get_bounds("rough_1") == (2.0, 4.0, 0.0, 2.0)


def test_repeated_unnamed_subterrains_are_numbered_by_type():
    """
    Two unnamed Flat subterrains are numbered under their type, which becomes the group
    name.
    """
    layout = make_layout(grid=[[terrain.Flat(), terrain.Wave(), terrain.Flat()]])

    assert layout.names == ("flat_terrain_0", "wave_terrain", "flat_terrain_1")
    assert layout.groups == {"flat_terrain": ("flat_terrain_0", "flat_terrain_1")}


def test_a_group_name_does_not_pick_one_subterrain():
    """get_bounds needs one subterrain, so a group name raises and lists the numbered names."""
    flat = terrain.Flat(name="flat")
    layout = make_layout(grid=[[flat, flat]])

    with pytest.raises(KeyError, match="flat_0"):
        layout.get_bounds("flat")


def test_a_name_that_clashes_with_a_numbered_name_is_rejected():
    """A subterrain named rough_0 next to two named rough would be ambiguous."""
    shared = terrain.Flat(name="rough")
    with pytest.raises(ValueError, match="'rough_0'"):
        make_layout(grid=[[shared, shared, terrain.Flat(name="rough_0")]])


def test_a_name_that_clashes_with_a_group_is_rejected():
    """
    Repeated a_0 subterrains form a group that clashes with the numbered name of a repeated
    a.
    """
    a, a_0 = terrain.Flat(name="a"), terrain.Flat(name="a_0")
    with pytest.raises(ValueError, match="'a_0'"):
        make_layout(grid=[[a, a, a_0, a_0]])


def test_changing_a_subterrain_afterwards_does_not_change_the_layout():
    """The layout keeps its own copy, so a later rename or param change has no effect."""
    patch = terrain.Flat(name="patch", height=0.1)
    layout = make_layout(grid=[[patch]], vertical_scale=0.01)

    patch.name = "renamed"
    patch.height = 0.5

    assert layout.names == ("patch",)
    assert np.allclose(layout.height_field, 10.0)


def test_bounds_are_offset_by_the_position():
    """The layout's pos moves the whole terrain and every subterrain's bounds."""
    layout = make_layout(
        grid=[[terrain.Flat(name="a")], [terrain.Flat(name="b")]],
        pos=(10.0, 20.0, 0.0),
    )

    assert layout.get_bounds() == (10.0, 14.0, 20.0, 22.0)
    assert layout.get_bounds("b") == (12.0, 14.0, 20.0, 22.0)


def test_unknown_names_raise():
    """A typo raises a KeyError that lists the valid names."""
    with pytest.raises(KeyError, match="Valid names"):
        make_layout().get_bounds("typo")


"""
Height field generation
"""


def test_the_height_field_covers_every_subterrain_with_shared_edges():
    """Two 4-interval subterrains along x share their edge row, so 9 rows rather than 10."""
    layout = make_layout(grid=[[terrain.Flat()], [terrain.Flat()]])

    # 2m / 0.5m = 4 intervals per subterrain, 2 subterrains along x share one edge
    assert layout.height_field.shape == (9, 5)


def test_heights_are_in_vertical_steps():
    """A 0.2m flat at a 0.01m vertical scale is stored as 20 steps."""
    layout = make_layout(grid=[[terrain.Flat(height=0.2)]], vertical_scale=0.01)

    assert np.allclose(layout.height_field, 20.0)


def test_shared_edges_keep_the_higher_neighbor():
    """Where a 0m and a 0.5m flat meet, the shared edge takes the higher one."""
    layout = make_layout(
        grid=row(a=terrain.Flat(height=0.0), b=terrain.Flat(height=0.5)),
        vertical_scale=0.01,
    )

    edge = layout.height_field[:, 4]  # the last column of "a", the first of "b"
    assert np.allclose(edge, 50.0)
    assert np.allclose(layout.height_field[:, 3], 0.0)


def test_one_subterrain_in_several_places_still_varies():
    """The same object in two places is seeded separately, so the patches differ."""
    patch = rough(name="rough")
    layout = make_layout(grid=[[patch, patch]])

    assert not np.array_equal(
        heights_of(layout, "rough_0"), heights_of(layout, "rough_1")
    )


def test_same_type_subterrains_get_different_seeds():
    """Two identical rough subterrains get different seeds, so their heights differ."""
    layout = make_layout(grid=row(a=rough(), b=rough()))

    assert not np.array_equal(heights_of(layout, "a"), heights_of(layout, "b"))


def test_the_same_seed_generates_the_same_terrain():
    """A seed reproduces the terrain exactly, and a different seed changes it."""
    first = make_layout(grid=row(a=rough(), b=rough()), seed=7)
    second = make_layout(grid=row(a=rough(), b=rough()), seed=7)
    other = make_layout(grid=row(a=rough(), b=rough()), seed=8)

    assert np.array_equal(first.height_field, second.height_field)
    assert not np.array_equal(first.height_field, other.height_field)


def test_seeding_leaves_the_global_random_state_alone():
    """Generating a layout restores numpy's global random state afterwards."""
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)

    make_layout(grid=row(a=rough(), b=rough()))

    assert np.random.rand() == expected


def test_same_type_subterrains_can_have_different_params():
    """Unlike gs.morphs.Terrain, two subterrains of one type can have different settings."""
    layout = make_layout(
        grid=row(a=rough(max_height=0.02), b=rough(max_height=0.2)),
        vertical_scale=0.005,
    )

    assert heights_of(layout, "a").max() < heights_of(layout, "b").max()


def test_a_height_field_subterrain_must_match_the_subterrain_shape():
    """A HeightField of the wrong shape fails, and the error gives the shape it needs."""
    with pytest.raises(ValueError, match=r"\(5, 5\)"):
        make_layout(grid=[[terrain.HeightField(heights=np.zeros((3, 3)))]])


def test_a_custom_subterrain_generates_its_own_heights():
    """A Subterrain subclass's heights, in meters, end up in the height field."""

    @dataclass(kw_only=True)
    class Step(terrain.Subterrain):
        type: ClassVar[str] = "step"
        height: float = 0.1

        def generate(self, shape, horizontal_scale, vertical_scale):
            """Raise the far half of the subterrain to ``height``."""
            heights = np.zeros(shape)
            heights[shape[0] // 2 :, :] = self.height
            return heights

    layout = make_layout(grid=[[Step(height=0.5)]], vertical_scale=0.01)

    assert np.allclose(layout.height_field[:2], 0.0)
    assert np.allclose(layout.height_field[2:], 50.0)


def test_a_custom_subterrain_can_run_genesis_generators():
    """A custom subterrain can run a Genesis generator through the Subterrain helpers."""
    from genesis.ext.isaacgym import terrain_utils

    @dataclass(kw_only=True)
    class Ramp(terrain.Subterrain):
        def generate(self, shape, horizontal_scale, vertical_scale):
            """A 0.5 slope from Genesis' sloped terrain generator."""
            genesis_terrain = self.empty_genesis_terrain(
                shape, horizontal_scale, vertical_scale
            )
            terrain_utils.sloped_terrain(genesis_terrain, slope=0.5)
            return self.genesis_heights(genesis_terrain)

    layout = make_layout(grid=[[Ramp()]], vertical_scale=0.01)

    # 0.5 m rise per meter over the 2 m subterrain, along x
    heights = layout.height_field * layout.vertical_scale
    assert np.allclose(heights[0], 0.0)
    assert np.allclose(heights[-1], 1.0, atol=0.01)


def test_a_wrong_shape_from_a_custom_subterrain_is_reported_by_name():
    """A custom subterrain that returns the wrong shape is named in the error."""

    @dataclass(kw_only=True)
    class Broken(terrain.Subterrain):
        def generate(self, shape, horizontal_scale, vertical_scale):
            """Return a height field of the wrong shape."""
            return np.zeros((1, 1))

    with pytest.raises(ValueError, match="'broken'"):
        make_layout(grid=[[Broken(name="broken")]])


@pytest.mark.parametrize(
    "subterrain",
    [
        terrain.Flat(),
        terrain.Fractal(),
        terrain.RandomUniform(),
        terrain.Sloped(),
        terrain.PyramidSloped(),
        terrain.DiscreteObstacles(),
        terrain.Wave(),
        terrain.Stairs(),
        terrain.PyramidStairs(),
        terrain.SteppingStones(),
        terrain.RandomGrid(grid_width=1.0),  # <~ at least the 0.5m scale used here
    ],
    ids=lambda s: type(s).__name__,
)
def test_every_built_in_subterrain_generates(subterrain):
    """Each built-in, with its defaults, generates finite heights of the right shape."""
    layout = make_layout(grid=[[subterrain]], subterrain_size=(4.0, 4.0))

    assert layout.height_field.shape == (9, 9)
    assert np.all(np.isfinite(layout.height_field))


def border_layout(subterrain, seed=0):
    """A layout of one 12m subterrain at a 0.25m scale, for checking its edges."""
    return make_layout(
        grid=[[subterrain]],
        subterrain_size=(12.0, 12.0),
        horizontal_scale=0.25,
        seed=seed,
    )


def test_fractal_edges_meet_the_ground():
    """The border eases a fractal's edges to 0, whichever way it drifted."""
    for seed in range(4):
        heights = border_layout(terrain.Fractal(), seed=seed).height_field

        assert np.allclose(heights[[0, -1], :], 0.0)
        assert np.allclose(heights[:, [0, -1]], 0.0)
        # Still rolling inside the border
        assert np.abs(heights[8:-8, 8:-8]).max() > 0.01


def test_fractal_without_a_border_keeps_its_edges():
    """A border_width of 0 leaves the heights as Genesis generated them."""
    np.random.seed(1)  # <~ a seed whose edges drift well below 0
    heights = terrain.Fractal(border_width=0.0).generate((49, 49), 0.25, 0.001)

    assert not np.allclose(heights[0], 0.0)


def test_other_subterrains_have_no_border_by_default():
    """Only Fractal eases its edges unless asked, so the others generate as before."""
    eased = border_layout(terrain.Wave()).height_field

    assert terrain.Wave().border_width == 0.0
    assert not np.allclose(eased[0], 0.0)


def test_any_subterrain_can_ease_its_edges():
    """A border_width on a built-in, or on your own subterrain, eases its edges to 0."""

    @dataclass(kw_only=True)
    class Raised(terrain.Subterrain):
        type: ClassVar[str] = "raised"

        def generate(self, shape, horizontal_scale, vertical_scale):
            return np.full(shape, 0.5)

    for subterrain in (terrain.Wave(border_width=1.0), Raised(border_width=1.0)):
        heights = border_layout(subterrain).height_field

        assert np.allclose(heights[[0, -1], :], 0.0)
        assert np.allclose(heights[:, [0, -1]], 0.0)
        assert np.abs(heights[8:-8, 8:-8]).max() > 0.01


def test_easing_is_smooth():
    """The ease rises from 0 at the edge to 1 at border_width from the edge, with no step along the way."""
    subterrain = terrain.Flat(border_width=1.0)
    eased = subterrain.ease_edges(np.ones((49, 49)), horizontal_scale=0.25)
    center_row = eased[:, 24]

    assert center_row[0] == 0.0
    assert np.allclose(center_row[4:-4], 1.0)  # <~ 1m in, at a 0.25m scale
    assert np.all(np.diff(center_row[:5]) > 0)


def test_random_uniform_rejects_inverted_heights():
    """A max_height below min_height is rejected while generating."""
    with pytest.raises(ValueError, match="min_height"):
        make_layout(grid=[[terrain.RandomUniform(min_height=0.1, max_height=0.0)]])


"""
Adding to a scene
"""


def test_the_layout_adds_its_terrain_to_the_scene():
    """
    The morph gets the height field, scale and position, plus any extra options like
    uv_scale.
    """
    scene = FakeScene()
    layout = make_layout(scene=scene, pos=(1.0, 2.0, 3.0), uv_scale=2.0)

    assert layout.entity == "terrain-entity"
    ((morph, kwargs),) = scene.calls
    assert morph["height_field"] is layout.height_field
    assert morph["horizontal_scale"] == 0.5
    assert morph["pos"] == (1.0, 2.0, 3.0)
    assert morph["uv_scale"] == 2.0
    assert kwargs["surface"].diffuse_texture is not None


def test_the_default_surface_is_a_half_meter_grid():
    """
    On a 4m x 2m terrain, the default texture's columns follow x and rows follow y, with two
    squares per meter.
    """
    scene = FakeScene()
    # 2 subterrains of 2m along x, 1 along y: a 4m x 2m terrain
    make_layout(scene=scene, grid=[[terrain.Flat()], [terrain.Flat()]])

    image = np.asarray(scene.calls[0][1]["surface"].diffuse_texture.image_array)
    # Columns follow x and rows follow y, with two squares per meter
    assert image.shape[:2] == (4 * 32, 8 * 32)
    # Inside the first squares: dark, then light along x and along y, then dark again
    assert tuple(image[1, 1]) == GRID_DARK
    assert tuple(image[1, 33]) == GRID_LIGHT
    assert tuple(image[33, 1]) == GRID_LIGHT
    assert tuple(image[33, 33]) == GRID_DARK
    # A one-pixel line where the squares meet
    assert tuple(image[32, 5]) == GRID_LINE
    assert tuple(image[5, 32]) == GRID_LINE


def test_the_default_grid_texture_stays_small_on_a_large_terrain():
    """On a 1km terrain, the default texture is capped at 4096 pixels per side."""
    scene = FakeScene()
    make_layout(
        scene=scene,
        grid=[[terrain.Flat()] * 20] * 20,
        subterrain_size=(50.0, 50.0),
        horizontal_scale=5.0,
    )

    image = np.asarray(scene.calls[0][1]["surface"].diffuse_texture.image_array)
    assert max(image.shape[:2]) <= 4096


def test_the_layout_passes_the_surface():
    """A surface you pass replaces the default grid."""
    scene = FakeScene()

    make_layout(scene=scene, surface="surface")

    assert scene.calls[0][1] == {"surface": "surface"}


def test_a_height_field_option_is_rejected():
    """The layout makes the height field, so passing one as a morph option is rejected."""
    with pytest.raises(TypeError, match="HeightField"):
        make_layout(height_field=np.zeros((5, 9)))


def test_nothing_is_added_to_the_scene_when_the_layout_is_invalid():
    """An invalid layout fails before it touches the scene."""
    scene = FakeScene()

    with pytest.raises(ValueError):
        make_layout(scene=scene, grid=[[terrain.Flat()], []])

    assert scene.calls == []


def test_neighboring_seeds_do_not_share_subterrains():
    """
    Seed 7's second subterrain doesn't match seed 8's first, so seed sequences never
    overlap.
    """
    first = make_layout(grid=row(a=rough(), b=rough()), seed=7)
    second = make_layout(grid=row(a=rough(), b=rough()), seed=8)

    assert not np.array_equal(heights_of(first, "b"), heights_of(second, "a"))


def test_a_one_subterrain_layout_seeds_like_a_genesis_terrain():
    """gs.morphs.Terrain seeds every subterrain with np.random.seed(0)."""
    layout = make_layout(grid=[[rough()]])

    np.random.seed(0)
    expected = rough().generate(
        layout._subterrain_shape(), layout.horizontal_scale, layout.vertical_scale
    )

    assert np.allclose(layout.height_field * layout.vertical_scale, expected)


"""
RandomGrid
"""


def random_grid_heights(**params):
    """Generate a RandomGrid on an 8m subterrain at a 0.1m scale, in meters."""
    np.random.seed(0)
    return terrain.RandomGrid(**params).generate((81, 81), 0.1, 0.005)


def test_random_grid_blocks_stay_within_their_height():
    """Blocks are within +/- grid_height, the border is flat, and the platform is on top."""
    heights = random_grid_heights(grid_height=0.2)

    assert heights.min() >= -0.2 and heights.max() <= 0.2
    assert np.allclose(heights[0], 0.0) and np.allclose(heights[:, -1], 0.0)
    assert np.allclose(heights[36:45, 36:45], 0.2)  # the 1m platform around the center


def test_random_grid_blocks_are_flat_squares():
    """Every point on one block shares its height."""
    heights = random_grid_heights(grid_width=0.4, border_width=0.0)

    # With no border, the first 0.4m block spans the first five points along each axis
    block = heights[0:4, 0:4]
    assert np.allclose(block, block[0, 0])
    assert not np.isclose(heights[0, 0], heights[0, 5])


def test_random_grid_holes_leave_a_cross():
    """With holes, only blocks in line with the platform remain."""
    heights = random_grid_heights(holes=True, hole_depth=-5.0)

    assert heights[10, 10] == -5.0  # a corner block
    assert heights[10, 40] > -5.0  # in line with the platform along y
    assert heights[40, 10] > -5.0  # in line with the platform along x


def test_random_grid_rejects_blocks_smaller_than_the_scale():
    """A block narrower than the height field spacing couldn't be drawn."""
    with pytest.raises(ValueError, match="horizontal_scale"):
        terrain.RandomGrid(grid_width=0.05).generate((81, 81), 0.1, 0.005)


def test_random_grid_needs_room_for_one_block():
    """A border that leaves no room for a block is rejected."""
    with pytest.raises(ValueError, match="No"):
        terrain.RandomGrid(border_width=5.0).generate((81, 81), 0.1, 0.005)
