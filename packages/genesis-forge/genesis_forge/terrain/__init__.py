"""
Terrain building for Genesis Forge environments.

A :class:`TerrainLayout` is a grid of subterrains, each with its own params, seed, and an
optional name. Create it with the scene before building, then pass it to a
:class:`~genesis_forge.managers.TerrainManager` so reset and termination functions can refer to
subterrains by name::

    from genesis_forge import terrain

    self.terrain_layout = terrain.TerrainLayout(
        self.scene,
        subterrain_size=(12, 12),
        grid=[
            terrain.Flat(name="flat"),
            terrain.RandomUniform(name="rough", max_height=0.08),
        ],
    )

Subterrain types:
- :class:`Flat`, :class:`HeightField` (your own heights)
- :class:`Fractal`, :class:`RandomUniform`, :class:`Wave`
- :class:`Sloped`, :class:`PyramidSloped`
- :class:`Stairs`, :class:`PyramidStairs`, :class:`SteppingStones`, :class:`DiscreteObstacles`
- :class:`RandomGrid`

Subclass :class:`Subterrain` to make your own.
"""

from .layout import TerrainLayout
from .map import TerrainMap
from .subterrains import (
    DiscreteObstacles,
    Flat,
    Fractal,
    HeightField,
    PyramidSloped,
    PyramidStairs,
    RandomGrid,
    RandomUniform,
    Sloped,
    Stairs,
    SteppingStones,
    Subterrain,
    Wave,
)

__all__ = [
    "DiscreteObstacles",
    "Flat",
    "Fractal",
    "HeightField",
    "PyramidSloped",
    "PyramidStairs",
    "RandomGrid",
    "RandomUniform",
    "Sloped",
    "Stairs",
    "SteppingStones",
    "Subterrain",
    "TerrainLayout",
    "TerrainMap",
    "Wave",
]
