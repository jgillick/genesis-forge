"""Behavior of TerrainMap: naming and bounds for a grid, lookups by name, and mapping
terrain entities that weren't built from a TerrainLayout.

Entities are stand-ins with only the attributes the map reads -- no scene is built.
"""

import pytest
import torch

from genesis_forge.terrain.map import TerrainMap


def grid_map(labels, pos=(0.0, 0.0), size=(2.0, 3.0)):
    """A map of a grid whose types are its labels."""
    return TerrainMap.from_grid(pos, size, labels=labels, types=labels)


class FakeMorph:
    def __init__(self, **attributes):
        """A morph with only the given attributes."""
        self.__dict__.update(attributes)


class FakeGeom:
    def __init__(self, aabb):
        """A geom with the given bounding box."""
        self._aabb = aabb

    def get_AABB(self):
        """The bounding box, as ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``."""
        return self._aabb


class FakeEntity:
    def __init__(self, morph, aabb=None):
        """An entity with one geom."""
        self.morph = morph
        self.geoms = [FakeGeom(aabb)]


"""
from_grid()
"""


def test_a_grid_is_mapped_in_grid_order():
    """Subterrains are listed x row by x row, with their bounds offset by the position."""
    terrain_map = grid_map([["a", "b"], ["c", "d"]], pos=(10.0, 20.0))

    assert terrain_map.names == ("a", "b", "c", "d")
    assert terrain_map.bounds == (10.0, 14.0, 20.0, 26.0)
    assert terrain_map.get_bounds("b") == (10.0, 12.0, 23.0, 26.0)
    assert terrain_map.get_bounds("c") == (12.0, 14.0, 20.0, 23.0)


def test_a_repeated_label_is_numbered_and_grouped():
    """A label used twice gets numbered names, and becomes a group of both."""
    terrain_map = grid_map([["rough", "flat", "rough"]])

    assert terrain_map.names == ("rough_0", "flat", "rough_1")
    assert terrain_map.groups == {"rough": (0, 2)}
    assert terrain_map.indices("rough") == (0, 2)
    assert terrain_map.indices("rough_1") == (2,)


def test_labels_and_types_are_separate():
    """A layout names subterrains its own way, but still records their types."""
    terrain_map = TerrainMap.from_grid(
        (0.0, 0.0),
        (1.0, 1.0),
        labels=[["start", "rough"]],
        types=[["flat_terrain", "random_uniform_terrain"]],
    )

    assert terrain_map.names == ("start", "rough")
    assert terrain_map.types == ("flat_terrain", "random_uniform_terrain")


"""
Lookups
"""


def test_index_needs_one_subterrain():
    """index() and get_bounds() refuse a shared name, and list the numbered names to use."""
    terrain_map = grid_map([["rough", "rough"]])

    with pytest.raises(KeyError, match="rough_0"):
        terrain_map.index("rough")
    with pytest.raises(KeyError, match="more than one"):
        terrain_map.get_bounds("rough")


def test_an_unknown_name_lists_the_valid_names():
    """A typo raises a KeyError listing every subterrain and group name."""
    terrain_map = grid_map([["rough", "rough", "flat"]])

    with pytest.raises(KeyError, match=r"\['rough_0', 'rough_1', 'flat', 'rough'\]"):
        terrain_map.indices("typo")


def test_no_name_is_the_whole_terrain():
    """get_bounds() without a name gives the whole terrain."""
    terrain_map = grid_map([["a"], ["b"]])

    assert terrain_map.get_bounds() == (0.0, 4.0, 0.0, 3.0)


"""
from_entity()
"""


def test_a_genesis_terrain_grid_is_named_after_its_types():
    """A gs.morphs.Terrain grid is mapped like a layout, using its types as names."""
    morph = FakeMorph(
        pos=(0.0, 0.0, 0.0),
        n_subterrains=(1, 3),
        subterrain_size=(5.0, 5.0),
        subterrain_types=[["flat_terrain", "wave_terrain", "flat_terrain"]],
        height_field=None,
    )

    terrain_map = TerrainMap.from_entity(FakeEntity(morph))

    assert terrain_map.names == ("flat_terrain_0", "wave_terrain", "flat_terrain_1")
    assert terrain_map.types == ("flat_terrain", "wave_terrain", "flat_terrain")
    assert terrain_map.get_bounds("wave_terrain") == (0.0, 5.0, 5.0, 10.0)


def test_a_height_field_terrain_has_no_subterrains():
    """A terrain built from a height field ignores its subterrain settings."""
    morph = FakeMorph(
        pos=(1.0, 2.0, 0.0),
        n_subterrains=(3, 3),
        subterrain_size=(12.0, 12.0),
        height_field=[[0.0] * 5] * 9,
        horizontal_scale=0.5,
    )

    terrain_map = TerrainMap.from_entity(FakeEntity(morph))

    assert terrain_map.bounds == (1.0, 5.0, 2.0, 4.0)
    assert terrain_map.names == ()


def test_other_ground_is_measured_by_its_bounding_box():
    """A plane or mesh has no subterrains, so its bounding box is the whole terrain."""
    aabb = torch.tensor([[[-5.0, -6.0, 0.0], [5.0, 6.0, 0.1]]])  # one per environment

    terrain_map = TerrainMap.from_entity(FakeEntity(FakeMorph(), aabb=aabb))

    assert terrain_map.bounds == (-5.0, 5.0, -6.0, 6.0)
    assert terrain_map.names == ()
