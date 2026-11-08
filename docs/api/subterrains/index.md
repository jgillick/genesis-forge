# Subterrains

A subterrain generates the ground for one place in a [`TerrainLayout`](../other/terrain.md) grid. Put them in the layout's `grid`, and give them a `name` to place robots on them.

| Class | Terrain |
|-------|---------|
| [`Flat`](flat.md) | Flat ground |
| [`Fractal`](fractal.md) | Rolling, natural-looking ground |
| [`RandomUniform`](random_uniform.md) | Random bumps |
| [`Wave`](wave.md) | Smooth waves |
| [`Sloped`](sloped.md) | A ramp |
| [`PyramidSloped`](pyramid_sloped.md) | Slopes up to a center platform |
| [`Stairs`](stairs.md) | A staircase |
| [`PyramidStairs`](pyramid_stairs.md) | Stairs up to a center platform |
| [`SteppingStones`](stepping_stones.md) | Stones with gaps between them |
| [`DiscreteObstacles`](discrete_obstacles.md) | Flat ground with random blocks |
| [`RandomGrid`](random_grid.md) | Square blocks at random heights around a raised platform |
| [`HeightField`](height_field.md) | Your own heights, in meters |

To make your own, subclass [`Subterrain`](subterrain.md) and return the heights in its `generate()` method. The [terrain guide](../../guide/managers/terrain.md) has examples.
