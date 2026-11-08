"""
Typed subterrain definitions for a :class:`~genesis_forge.terrain.TerrainLayout`.

Each class generates one subterrain of the terrain grid. Most built-in types wrap the
generators Genesis uses for ``gs.morphs.Terrain``. Unlike ``gs.morphs.Terrain``, every 
subterrain has its own params and its own random seed.

Subclass :class:`Subterrain` to make your own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from genesis.ext.isaacgym import terrain_utils
from genesis.utils.terrain import fractal_terrain


@dataclass(kw_only=True)
class Subterrain:
    """
    Base class for one subterrain of a terrain layout.

    Subclasses declare their params as dataclass fields and implement :meth:`generate`.

    Example - A terrain with a single step in the middle::

        @dataclass(kw_only=True)
        class Step(Subterrain):
            type: ClassVar[str] = "step"
            height: float = 0.1

            def generate(self, shape, horizontal_scale, vertical_scale):
                heights = np.zeros(shape)
                heights[shape[0] // 2 :, :] = self.height
                return heights
    """

    type: ClassVar[str] = "subterrain"
    """A short name for the kind of terrain, e.g. ``"flat_terrain"``."""

    name: str | None = None
    """
    The name to select this subterrain by, e.g. ``"rough"``. Defaults to its ``type``.
    Every subterrain that shares a name is selected by it.
    """

    border_width: float = 0.0
    """
    How far in from the edges, in meters, the heights ease down to 0, so the subterrain
    meets its neighbors without a step. 0 leaves the edges wherever they land.

    Best for ground that varies around 0, like :class:`Fractal` or :class:`RandomUniform`.
    A ramp or staircase that ends well above or below 0 gets a steep wall at its edge
    instead.
    """

    horizontal_scale: float | None = None
    """
    The distance between two of this subterrain's height field points, in meters.
    None uses the layout's ``horizontal_scale``.

    The layout's height field uses the finest spacing any subterrain asks for, and joins up
    a coarser subterrain's points with straight lines. So one subterrain can have fine detail,
    like crisp block edges, without changing how the others generate. Every spacing in the
    layout must be a whole multiple of the finest one, and the subterrain size a whole
    multiple of each.

    :meth:`generate` receives the resolved spacing as its ``horizontal_scale`` argument; use
    that, since this field may be None.
    """

    def generate(
        self,
        shape: tuple[int, int],
        horizontal_scale: float,
        vertical_scale: float,
    ) -> np.ndarray:
        """
        Generate the subterrain's height field.

        Called once for each place the subterrain appears in the grid, with numpy's global
        random state already seeded for that place.

        Args:
            shape: The ``(x, y)`` number of height field points. The first and last points
                   on each axis lie on the subterrain's edges, and are shared with its neighbor.
            horizontal_scale: The distance between two height field points, in meters: this
                              subterrain's own :attr:`horizontal_scale`, or the layout's.
            vertical_scale: The height resolution of the terrain, in meters.

        Returns:
            The heights, in meters, in an array of the given shape.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement generate(shape, horizontal_scale, vertical_scale)"
        )

    def ease_edges(self, heights: np.ndarray, horizontal_scale: float) -> np.ndarray:
        """
        Ease the heights down to 0 over :attr:`border_width` from each edge, to ease the
        transition from one subterrain to another. The :class:`~genesis_forge.terrain.TerrainLayout`
        calls this on whatever :meth:`generate` returns, once it's filled in to the layout's
        height field spacing.

        Args:
            heights: The heights from :meth:`generate`, in meters.
            horizontal_scale: The distance between two height field points, in meters.
        """
        if self.border_width <= 0:
            return heights
        x_ease = self._edge_ease(heights.shape[0], horizontal_scale)
        y_ease = self._edge_ease(heights.shape[1], horizontal_scale)
        # Combine them into one multiplier per point, so all four edges are eased
        return heights * np.outer(x_ease, y_ease)

    def _edge_ease(self, n_points: int, horizontal_scale: float) -> np.ndarray:
        """
        Create a border at each point along the edges of one axis, to ease the transition 
        from one subterrain to another. 
        """
        # Each point's index along the axis
        points = np.arange(n_points)
        # Its distance from the nearer of the two edges, in meters
        from_edge = np.minimum(points, n_points - 1 - points) * horizontal_scale
        # How far through the border it is: 0 at the edge, 1 at border_width in and beyond
        ramp = np.clip(from_edge / self.border_width, 0.0, 1.0)
        # Bend the straight ramp into an S-curve (smoothstep), so there's no crease at either end
        return ramp * ramp * (3 - 2 * ramp)

    @staticmethod
    def empty_genesis_terrain(
        shape: tuple[int, int], horizontal_scale: float, vertical_scale: float
    ) -> terrain_utils.SubTerrain:
        """
        An empty Genesis terrain of one subterrain, for Genesis' terrain generators to write into.

        Pass the arguments :meth:`generate` received, run a generator from
        ``genesis.ext.isaacgym.terrain_utils`` on it, and return :meth:`genesis_heights`::

            def generate(self, shape, horizontal_scale, vertical_scale):
                genesis_terrain = self.empty_genesis_terrain(shape, horizontal_scale, vertical_scale)
                terrain_utils.wave_terrain(genesis_terrain, num_waves=3, amplitude=0.1)
                return self.genesis_heights(genesis_terrain)
        """
        return terrain_utils.SubTerrain(
            width=shape[0],
            length=shape[1],
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
        )

    @staticmethod
    def genesis_heights(genesis_terrain: terrain_utils.SubTerrain) -> np.ndarray:
        """
        The heights a Genesis terrain generator wrote, in meters, ready to return from
        :meth:`generate`. See :meth:`empty_genesis_terrain`.
        """
        return (
            np.asarray(genesis_terrain.height_field_raw, dtype=np.float64)
            * genesis_terrain.vertical_scale
        )


@dataclass(kw_only=True)
class Flat(Subterrain):
    """Flat ground."""

    type: ClassVar[str] = "flat_terrain"

    height: float = 0.0
    """The ground height, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate flat ground at ``height``."""
        return np.full(shape, self.height)


@dataclass(kw_only=True)
class HeightField(Subterrain):
    """
    A height field you provide, in meters.

    Its shape must be ``subterrain_size / horizontal_scale + 1`` on each axis, using its own
    ``horizontal_scale`` if it has one. 
    """

    type: ClassVar[str] = "height_field"

    heights: np.ndarray

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Return the given heights, after checking they fit the subterrain's shape."""
        heights = np.asarray(self.heights, dtype=np.float64)
        if heights.shape != tuple(shape):
            raise ValueError(
                f"HeightField heights must have the shape {tuple(shape)} for this layout, "
                f"got {heights.shape}."
            )
        return heights


@dataclass(kw_only=True)
class Fractal(Subterrain):
    """Rolling, natural-looking ground."""

    type: ClassVar[str] = "fractal_terrain"

    levels: int = 8
    """How fine the detail is."""
    scale: float = 5.0
    """How much the height varies."""
    border_width: float = 1.0
    """
    How far in from the edges, in meters, the ground eases down to 0. Genesis' fractal
    drifts up or down at random, so without it the edges can sit well above or below its
    neighbors. 0 leaves the edges wherever they land.
    """

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate rolling ground with Genesis' fractal generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        fractal_terrain(genesis_terrain, levels=self.levels, scale=self.scale)
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class RandomUniform(Subterrain):
    """
    Bumpy ground with random heights.

    Give it a ``horizontal_scale`` equal to ``downsampled_scale`` for straight slopes between
    the random heights, that stay between ``min_height`` and ``max_height``. At a finer
    spacing, Genesis curves between them, and the curve might overshoot the max_height by a little.
    """

    type: ClassVar[str] = "random_uniform_terrain"

    min_height: float = -0.1
    """The lowest height, in meters."""
    max_height: float = 0.1
    """The highest height, in meters."""
    step: float = 0.1
    """The smallest height difference between two bumps, in meters."""
    downsampled_scale: float = 0.5
    """The distance between two random heights, in meters. At least the subterrain's horizontal scale."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Check the heights and scale, then generate random bumps with Genesis' generator."""
        if self.max_height < self.min_height:
            raise ValueError(
                f"RandomUniform max_height ({self.max_height}) is below min_height ({self.min_height})."
            )
        if self.downsampled_scale < horizontal_scale:
            raise ValueError(
                f"RandomUniform downsampled_scale ({self.downsampled_scale}) must be at least "
                f"its horizontal_scale ({horizontal_scale})."
            )
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.random_uniform_terrain(
            genesis_terrain,
            min_height=self.min_height,
            max_height=self.max_height,
            step=self.step,
            downsampled_scale=self.downsampled_scale,
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class Sloped(Subterrain):
    """A ramp that rises along the x axis."""

    type: ClassVar[str] = "sloped_terrain"

    slope: float = -0.5
    """Height change per meter."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate a ramp along the x axis with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.sloped_terrain(genesis_terrain, slope=self.slope)
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class PyramidSloped(Subterrain):
    """Slopes rising (or falling) to a flat platform in the center."""

    type: ClassVar[str] = "pyramid_sloped_terrain"

    slope: float = 0.1
    """Height change per meter. Negative makes a pit."""
    platform_size: float = 1.0
    """The size of the flat center platform, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate slopes up (or down) to a center platform with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.pyramid_sloped_terrain(
            genesis_terrain, slope=self.slope, platform_size=self.platform_size
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class DiscreteObstacles(Subterrain):
    """Flat ground scattered with random rectangular blocks."""

    type: ClassVar[str] = "discrete_obstacles_terrain"

    max_height: float = 0.25
    """The tallest block, in meters."""
    min_size: float = 0.5
    """The smallest block side, in meters."""
    max_size: float = 2.0
    """The largest block side, in meters."""
    num_rects: int = 10
    """How many blocks."""
    platform_size: float = 1.0
    """The size of the clear center platform, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate flat ground with random blocks with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.discrete_obstacles_terrain(
            genesis_terrain,
            max_height=self.max_height,
            min_size=self.min_size,
            max_size=self.max_size,
            num_rects=self.num_rects,
            platform_size=self.platform_size,
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class Wave(Subterrain):
    """Smooth waves."""

    type: ClassVar[str] = "wave_terrain"

    num_waves: float = 2.0
    """How many waves fit across the subterrain."""
    amplitude: float = 0.1
    """The wave height, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate smooth waves with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.wave_terrain(
            genesis_terrain, num_waves=self.num_waves, amplitude=self.amplitude
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class Stairs(Subterrain):
    """A straight staircase along the x axis."""

    type: ClassVar[str] = "stairs_terrain"

    step_width: float = 0.75
    """The depth of each step, in meters."""
    step_height: float = -0.1
    """The height of each step, in meters. Negative goes down."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate a staircase along the x axis with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.stairs_terrain(
            genesis_terrain, step_width=self.step_width, step_height=self.step_height
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class PyramidStairs(Subterrain):
    """Stairs on all four sides, up (or down) to a center platform."""

    type: ClassVar[str] = "pyramid_stairs_terrain"

    step_width: float = 0.75
    """The depth of each step, in meters."""
    step_height: float = 0.1
    """The height of each step, in meters. Negative makes a pit."""
    platform_size: float = 1.0
    """The size of the center platform, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate stairs up (or down) to a center platform with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.pyramid_stairs_terrain(
            genesis_terrain,
            step_width=self.step_width,
            step_height=self.step_height,
            platform_size=self.platform_size,
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class SteppingStones(Subterrain):
    """Square stones separated by deep gaps."""

    type: ClassVar[str] = "stepping_stones_terrain"

    stone_size: float = 1.0
    """The side of each stone, in meters."""
    stone_distance: float = 0.25
    """The gap between stones, in meters."""
    max_height: float = 0.2
    """The largest random height of a stone, in meters."""
    platform_size: float = 0.0
    """The size of the solid center platform, in meters."""
    depth: float = -10.0
    """The height of the gaps between stones, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate stones separated by deep gaps with Genesis' generator."""
        genesis_terrain = self.empty_genesis_terrain(
            shape, horizontal_scale, vertical_scale
        )
        terrain_utils.stepping_stones_terrain(
            genesis_terrain,
            stone_size=self.stone_size,
            stone_distance=self.stone_distance,
            max_height=self.max_height,
            platform_size=self.platform_size,
            depth=self.depth,
        )
        return self.genesis_heights(genesis_terrain)


@dataclass(kw_only=True)
class RandomGrid(Subterrain):
    """
    A grid of square blocks, each raised or lowered by a random height, around a raised
    platform in the center, with a flat border.
    
    Give it a ``horizontal_scale`` of a quarter of ``grid_width`` or finer for crisp blocks.
    The rest of the layout keeps its own spacing.

    Based on mjlab's ``BoxRandomGridTerrainCfg``
    """

    type: ClassVar[str] = "random_grid_terrain"

    grid_width: float = 0.4
    """The side of each square block, in meters."""
    grid_height: float = 0.1
    """Each block's height is random, between ``-grid_height`` and ``grid_height``, in meters.
    The center platform sits at ``grid_height``."""
    platform_width: float = 1.0
    """The side of the square platform in the center, in meters."""
    border_width: float = 0.25
    """
    The least width of flat ground at height 0 along each edge, in meters. The grid stops
    at the border, so there's nothing for the easing to change.
    """
    holes: bool = False
    """
    Turn most of the grid into pits, leaving a narrow walkway, only as wide as the platform, running from it out to the border. 
    Everything else drops to ``hole_depth``, so a robot has to stay on the walkway.
    """
    hole_depth: float = -10.0
    """How far down the pits go when ``holes`` is on, in meters."""

    def generate(self, shape, horizontal_scale, vertical_scale):
        """Generate the blocks, platform and border, in meters."""
        if self.grid_width < horizontal_scale:
            raise ValueError(
                f"RandomGrid grid_width ({self.grid_width}) must be at least its "
                f"horizontal_scale ({horizontal_scale})."
            )
        if self.grid_height < 0:
            raise ValueError(
                f"RandomGrid grid_height must not be negative, got {self.grid_height}."
            )

        # The position of each height field point along x and y, in meters
        x = np.arange(shape[0]) * horizontal_scale
        y = np.arange(shape[1]) * horizontal_scale
        x_size, y_size = x[-1], y[-1]

        # How many blocks fit inside the border, centered on the subterrain
        x_blocks = int((x_size - 2 * self.border_width) / self.grid_width + 1e-9)
        y_blocks = int((y_size - 2 * self.border_width) / self.grid_width + 1e-9)
        if x_blocks <= 0 or y_blocks <= 0:
            raise ValueError(
                f"No {self.grid_width}m blocks fit inside a {self.border_width}m border on a "
                f"{x_size}m x {y_size}m subterrain."
            )
        x_start = (x_size - x_blocks * self.grid_width) / 2
        y_start = (y_size - y_blocks * self.grid_width) / 2

        # Which block each point is on, or -1 for the border
        x_block = np.floor((x - x_start) / self.grid_width).astype(int)
        y_block = np.floor((y - y_start) / self.grid_width).astype(int)
        x_block[(x_block < 0) | (x_block >= x_blocks)] = -1
        y_block[(y_block < 0) | (y_block >= y_blocks)] = -1

        block_heights = np.random.uniform(
            -self.grid_height, self.grid_height, (x_blocks, y_blocks)
        )
        if self.holes:
            # Only the rows and columns of blocks whose centers line up with the platform remain
            half_platform = self.platform_width / 2
            x_centers = x_start + (np.arange(x_blocks) + 0.5) * self.grid_width
            y_centers = y_start + (np.arange(y_blocks) + 0.5) * self.grid_width
            in_x_strip = np.abs(x_centers - x_size / 2) <= half_platform
            in_y_strip = np.abs(y_centers - y_size / 2) <= half_platform
            in_cross = in_x_strip[:, None] | in_y_strip[None, :]
            block_heights[~in_cross] = self.hole_depth

        heights = np.zeros(shape)
        on_x_block, on_y_block = x_block >= 0, y_block >= 0
        heights[np.ix_(on_x_block, on_y_block)] = block_heights[
            np.ix_(x_block[on_x_block], y_block[on_y_block])
        ]

        # The platform sits on top of the blocks in the center
        on_x_platform = np.abs(x - x_size / 2) <= self.platform_width / 2
        on_y_platform = np.abs(y - y_size / 2) <= self.platform_width / 2
        heights[np.ix_(on_x_platform, on_y_platform)] = self.grid_height
        return heights
