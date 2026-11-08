from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import BaseManager
from genesis_forge.terrain.map import TerrainMap

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

    from genesis_forge.terrain import TerrainLayout

SubterrainSelection = str | Sequence[str] | torch.Tensor | None
"""
Which subterrain(s) to place something on:

- ``None``: anywhere on the whole terrain.
- ``"name"``: that subterrain.
- ``["a", "b"]``: a uniformly random choice among them, made per position.
- A long tensor of subterrain indices (see :meth:`TerrainManager.index`), one per position.
  The index ``-1`` means the whole terrain.
"""

WHOLE_TERRAIN = -1
"""The subterrain index that stands for the whole terrain."""


class TerrainManager(BaseManager):
    """
    Provides useful functions for the environment terrain.
    The manager maps out the sizes and heights of the terrain and subterrain.
    This allows your environment to calculate the robot's height above rough terrain.
    You can also generate random positions on the terrain or subterrain to place your robots on reset.

    Every subterrain has a unique name and an integer index. With a
    :class:`~genesis_forge.terrain.TerrainLayout`, the names are the layout's subterrain names. With a
    plain ``gs.morphs.Terrain``, a subterrain is named after its type. Either way, a name that
    appears more than once is numbered in grid order (``flat_terrain_0``, ``flat_terrain_1``, ...),
    and the name itself selects any one of them at random.

    Args:
        env: The environment instance.
        terrain: The terrain entity to manage, if it isn't from a layout. Defaults to
                 ``env.terrain``.
        layout: The layout the terrain was built from, instead of ``terrain``. Recommended --
                it lets you name your subterrains.

    Example::

        from genesis_forge import terrain

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.scene = gs.Scene()

                # Add terrain
                self.terrain_layout = terrain.TerrainLayout(
                    self.scene,
                    subterrain_size=(25, 25),
                    grid=[
                        terrain.Flat(name="flat"),
                        terrain.RandomUniform(name="rough", max_height=0.08),
                    ],
                )

            def config(self):
                self.terrain_manager = TerrainManager(self, layout=self.terrain_layout)

            def reset(self, envs_idx: torch.Tensor | Sequence[int] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
                # Randomize positions on the terrain
                pos = self.terrain_manager.generate_random_env_pos(
                    envs_idx=envs_idx,
                    subterrain="flat",
                    height_offset=0.15,
                )
                self.robot.set_pos(pos, envs_idx=envs_idx)
    """

    def __init__(
        self,
        env: GenesisEnv,
        terrain: RigidEntity = None,
        layout: TerrainLayout | None = None,
    ):
        super().__init__(env, type="terrain")

        if layout is not None:
            if terrain is not None:
                raise ValueError(
                    "Pass TerrainManager either terrain= or layout=, not both."
                )
            terrain = layout.entity
        elif terrain is None:
            terrain = env.terrain

        self._terrain: RigidEntity = terrain
        self._layout = layout
        self._map = TerrainMap(bounds=(0.0, 0.0, 0.0, 0.0))
        self._bounds_table = torch.zeros((1, 4), device=gs.device, dtype=gs.tc_float)
        self._indices_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._has_height_field = False
        self._ground_height = 0.0
        self._env_pos_buffer = torch.zeros(
            (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self._env_subterrain = torch.full(
            (self.env.num_envs,), WHOLE_TERRAIN, device=gs.device, dtype=torch.long
        )

    def build(self):
        """Map out the terrain, its subterrains, and where heights come from."""
        if self._layout is not None:
            self._map = self._layout.map
        else:
            self._map = TerrainMap.from_entity(self._terrain)
        self._indices_cache = {}

        self._has_height_field = getattr(self._terrain, "terrain_hf", None) is not None

        # Any other ground, like a plane, is taken as flat at its z position, for the first
        # environment
        if not self._has_height_field:
            pos = self._terrain.geoms[0].get_pos()
            if pos.ndim == 2:
                pos = pos[0]
            self._ground_height = float(pos[2])

        rows = [*self._map.subterrain_bounds, self._map.bounds]
        self._bounds_table = torch.tensor(rows, device=gs.device, dtype=gs.tc_float)

    """
    Properties
    """

    @property
    def terrain(self) -> RigidEntity:
        """The terrain entity this manager describes."""
        return self._terrain

    @property
    def layout(self) -> TerrainLayout | None:
        """The layout the terrain was built from, if one was given."""
        return self._layout

    @property
    def terrain_map(self) -> TerrainMap:
        """The terrain's lookup table of subterrain names, types and areas."""
        return self._map

    @property
    def subterrain_names(self) -> list[str]:
        """The subterrain names, in index order."""
        return list(self._map.names)

    @property
    def env_subterrain(self) -> torch.Tensor:
        """
        The index of the subterrain each environment was last placed on by
        :meth:`generate_random_env_pos`, in the shape ``(num_envs,)``.
        ``-1`` means the whole terrain (or not placed yet).
        """
        return self._env_subterrain

    """
    Subterrain lookup
    """

    def index(self, name: str) -> int:
        """
        Get the index of a subterrain by name.

        Raises:
            KeyError: If there is no subterrain by that name, or it's shared by several.
        """
        return self._map.index(name)

    def indices(self, selection: str | Sequence[str]) -> torch.Tensor:
        """
        Get the indices of one or more subterrains, by name.
        A type name that is shared by several subterrains expands to all of them.

        Returns:
            A long tensor of subterrain indices.
        """
        if isinstance(selection, Mapping):
            raise TypeError(
                "Subterrains are selected by a name or a list of names, not a dict."
            )
        names = (selection,) if isinstance(selection, str) else tuple(selection)
        if not names:
            raise ValueError("The subterrain selection is empty.")
        cached = self._indices_cache.get(names)
        if cached is not None:
            return cached
        idx: list[int] = []
        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"Subterrain names must be strings, got {type(name).__name__}: {name!r}"
                )
            idx.extend(self._map.indices(name))
        tensor = torch.tensor(idx, device=gs.device, dtype=torch.long)
        self._indices_cache[names] = tensor
        return tensor

    def validate_selection(
        self, selection: SubterrainSelection, num: int | None = None
    ) -> None:
        """
        Check that a subterrain selection refers to real subterrains.
        Useful at build time, to catch a typo before the first reset.

        Args:
            selection: The selection to check.
            num: The number of entries a tensor selection must have.

        Raises:
            KeyError: If a name is unknown.
            TypeError, ValueError: If the selection is malformed.
        """
        if selection is None or self._is_unknown_name(selection):
            return
        if isinstance(selection, torch.Tensor):
            self._check_index_tensor(selection, num)
            if selection.numel() > 0:
                n = len(self._map.names)
                low, high = int(selection.min()), int(selection.max())
                if low < WHOLE_TERRAIN or high >= n:
                    raise ValueError(
                        f"Subterrain indices must be between -1 and {n - 1}, "
                        f"got values from {low} to {high}."
                    )
            return
        self.indices(selection)

    def sample_subterrains(
        self, num: int, selection: SubterrainSelection
    ) -> torch.Tensor:
        """
        Choose a subterrain index for each of ``num`` positions.

        Args:
            num: How many indices to choose.
            selection: Which subterrains to choose from. See :data:`SubterrainSelection`.

        Returns:
            A long tensor of subterrain indices in the shape ``(num,)``.
        """
        if selection is None:
            return torch.full((num,), WHOLE_TERRAIN, device=gs.device, dtype=torch.long)
        if isinstance(selection, torch.Tensor):
            self._check_index_tensor(selection, num)
            return selection.to(device=gs.device, dtype=torch.long)
        if self._is_unknown_name(selection):
            return torch.full((num,), WHOLE_TERRAIN, device=gs.device, dtype=torch.long)
        # Pick one of the selected subterrains at random for each position
        candidates = self.indices(selection)
        picks = torch.randint(len(candidates), (num,), device=gs.device)
        return candidates[picks]

    """
    Bounds and heights
    """

    def get_bounds(
        self, subterrain: str | None = None
    ) -> tuple[float, float, float, float]:
        """
        Get the bounds of the terrain, or subterrain.

        A name shared by several subterrains gives the last of them, and an unknown name gives
        the whole terrain, each with a warning.

        Args:
            subterrain: The subterrain name. If None, the bounds of the whole terrain are returned.

        Returns:
            The bounds as ``(x_min, x_max, y_min, y_max)``.
        """
        if subterrain is None or self._is_unknown_name(subterrain):
            return self._map.bounds
        if subterrain in self._map.groups:
            last = self._map.groups[subterrain][-1]
            last_name = self._map.names[last]
            warnings.warn(
                f"'{subterrain}' names more than one subterrain, so its bounds are those of the "
                f"last one, '{last_name}'. Use one of the numbered names instead, or, in "
                "out_of_bounds, subterrain=True to keep each robot inside its own subterrain.",
                stacklevel=2,
            )
            return self._map.subterrain_bounds[last]
        return self._map.get_bounds(subterrain)

    def get_subterrain_bounds(self, subterrain_idx: torch.Tensor) -> torch.Tensor:
        """
        Get the bounds of many subterrains at once, by index.

        Args:
            subterrain_idx: A long tensor of subterrain indices. ``-1`` is the whole terrain.

        Returns:
            The bounds in the shape ``(*subterrain_idx.shape, 4)``, as ``(x_min, x_max, y_min, y_max)``.
        """
        return self._bounds_table[subterrain_idx]

    def get_terrain_height(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Get the terrain height at world coordinates (x, y).

        On a ``gs.morphs.Terrain`` (including a :class:`~genesis_forge.terrain.TerrainLayout`), this is
        the exact surface rigid bodies rest on, and positions outside the terrain get the height at its
        nearest edge. On anything else, like a plane, the ground is taken as flat at its z position.

        Args:
            x: Tensor of shape (n_envs,)
            y: Tensor of shape (n_envs,)

        Returns:
            Heights in the torch.Tensor of shape (n_envs,)
        """
        if self._has_height_field:
            # Genesis returns NaN past one height field step outside the terrain, so clamp to the edge
            x_min, x_max, y_min, y_max = self._map.bounds
            positions = torch.stack(
                [x.clamp(x_min, x_max), y.clamp(y_min, y_max)], dim=-1
            )
            # Every environment shares the terrain, so only measure the first one. Without
            # parallel environments, Genesis takes no environment index at all.
            first_env = 0 if self._terrain.scene.n_envs > 0 else None
            heights = self._terrain.get_terrain_height(positions, envs_idx=first_env)
            return heights.reshape(-1).to(dtype=gs.tc_float)

        return torch.full(
            (x.shape[0],), self._ground_height, device=gs.device, dtype=gs.tc_float
        )

    """
    Random positions
    """

    def generate_random_positions(
        self,
        num: int | None = None,
        usable_ratio: float = 0.5,
        subterrain: SubterrainSelection = None,
        height_offset: float = 0.1e-3,
        output: torch.Tensor | None = None,
        out_idx: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        """
        Distribute X/Y/Z positions across the terrain or subterrain.
        The X & Y positions will be random points and the Z position will be at the terrain height at that point.

        Args:
            num: The number of positions to generate. Not necessary if output is provided
            output: The position tensor to update in-place.
            out_idx: The indices of the output position tensor to update.
                     Can be a list of ints or a torch.Tensor for better performance.
            usable_ratio: How much of the terrain/subterrain area should be used for random positions.
                          For example, 0.25 will only generate positions within the center 25% of the area of the terrain/subterrain.
                          This helps avoid placing things right on the edge of the terrain/subterrain.
            subterrain: The subterrain(s) to generate positions on. See :data:`SubterrainSelection`.
                        A tensor selection has one index per generated position.
                        If None, positions will be generated for the entire terrain.
            height_offset: The offset to add to the terrain height.
                           A small offset keeps items from starting in contact with the ground.

        Returns:
            The positions tensor of shape (num, 3)
        """
        # Prep output buffer
        assert output is not None or num is not None, (
            "Either output or num must be provided"
        )
        if output is None:
            output = torch.zeros(num, 3, device=gs.device)
        if out_idx is None:
            out_idx = torch.arange(output.shape[0], device=gs.device)
        elif isinstance(out_idx, list):
            out_idx = torch.tensor(out_idx, device=gs.device, dtype=torch.long)
        if num is None:
            num = out_idx.shape[0]

        # A random point inside the center of each position's subterrain
        subterrain_idx = self.sample_subterrains(num, subterrain)
        x_min, x_max, y_min, y_max = self._bounds_table[subterrain_idx].unbind(dim=-1)
        x_size = (x_max - x_min) * usable_ratio
        y_size = (y_max - y_min) * usable_ratio
        x_start = x_min + (x_max - x_min - x_size) / 2
        y_start = y_min + (y_max - y_min - y_size) / 2
        x = x_start + x_size * torch.rand(num, device=gs.device, dtype=gs.tc_float)
        y = y_start + y_size * torch.rand(num, device=gs.device, dtype=gs.tc_float)

        output[out_idx, 0] = x
        output[out_idx, 1] = y
        output[out_idx, 2] = self.get_terrain_height(x, y) + height_offset
        return output

    def generate_random_env_pos(
        self,
        envs_idx: torch.Tensor | Sequence[int] | None = None,
        usable_ratio: float = 0.5,
        subterrain: SubterrainSelection = None,
        height_offset: float = 0.1e-3,
    ) -> torch.Tensor:
        """
        Generate one X/Y/Z position on the terrain for each environment.
        The X & Y positions will be random points and the Z position will be at the terrain height at that point.
        The chosen subterrain is recorded in :attr:`env_subterrain`.

        Args:
            envs_idx: The indices of the environments to generate positions for.
                      If None, positions will be generated for all environments.
                      Can be a list of ints or a torch.Tensor for better performance.
            usable_ratio: How much of the terrain/subterrain area should be used for random positions.
                          For example, 0.25 will only generate positions within the center 25% of the area of the terrain/subterrain.
                          This helps avoid placing things right on the edge of the terrain/subterrain.
            subterrain: The subterrain(s) to place the environments on. See :data:`SubterrainSelection`.
                        A tensor selection has the shape ``(num_envs,)`` -- one index for every
                        environment, of which only the ``envs_idx`` entries are read.
                        If None, positions will be generated for the entire terrain.
            height_offset: The offset to add to the terrain height.

        Returns:
            The position tensor of shape (len(envs_idx), 3)
        """
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx
        elif isinstance(envs_idx, list):
            envs_idx = torch.tensor(envs_idx, device=gs.device, dtype=torch.long)

        # Pick each environment's subterrain here, so it can be recorded
        if isinstance(subterrain, torch.Tensor):
            self._check_index_tensor(subterrain, self.env.num_envs)
            subterrain_idx = subterrain.to(device=gs.device, dtype=torch.long)[envs_idx]
        else:
            subterrain_idx = self.sample_subterrains(envs_idx.shape[0], subterrain)
        self._env_subterrain[envs_idx] = subterrain_idx

        self.generate_random_positions(
            output=self._env_pos_buffer,
            out_idx=envs_idx,
            subterrain=subterrain_idx,
            usable_ratio=usable_ratio,
            height_offset=height_offset,
        )
        return self._env_pos_buffer[envs_idx]

    """
    Implementation
    """

    def _is_unknown_name(self, selection: SubterrainSelection) -> bool:
        """
        Whether the selection is a single name that matches no subterrain, and warn if so.
        Such a name stands for the whole terrain.
        """
        if not isinstance(selection, str):
            return False
        if selection in self._map.names or selection in self._map.groups:
            return False
        valid = list(self._map.names) + list(self._map.groups)
        warnings.warn(
            f"No subterrain named '{selection}', so the whole terrain is used instead. "
            f"Valid names: {valid}",
            stacklevel=3,
        )
        return True

    def _check_index_tensor(self, selection: torch.Tensor, num: int | None):
        """Check the dtype and shape of a tensor of subterrain indices."""
        if selection.dtype.is_floating_point or selection.dtype == torch.bool:
            raise TypeError(
                f"A subterrain index tensor must have an integer dtype, got {selection.dtype}."
            )
        if num is not None and selection.shape != (num,):
            raise ValueError(
                f"A subterrain index tensor must have the shape ({num},), got {tuple(selection.shape)}."
            )
