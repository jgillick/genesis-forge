from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

Bounds = tuple[float, float, float, float]
"""An area as ``(x_min, x_max, y_min, y_max)``, in meters."""


@dataclass(frozen=True)
class TerrainMap:
    """
    A lookup table of a terrain's subterrains: each one's name, type, and the x/y area it
    covers, plus the area of the whole terrain.

    It answers questions like "where is the subterrain called ``stairs``?" and "which
    subterrains does the name ``rough`` select?". The
    :class:`~genesis_forge.managers.TerrainManager` uses it to place robots and check bounds.
    It describes the terrain's layout only: it doesn't hold heights, and doesn't generate
    anything.

    Every terrain gets the same kind of map, whatever it was built from:

    - A :class:`~genesis_forge.terrain.TerrainLayout`: the names you gave its subterrains.
      Get it with ``layout.map``.
    - A ``gs.morphs.Terrain`` grid: each subterrain is named after its type. Use
      :meth:`from_entity`.
    - Anything else, like a plane: no subterrains, only the area of the whole terrain. Also
      :meth:`from_entity`.

    Subterrains are listed in grid order -- ``grid[0][0], grid[0][1], ...`` -- and their
    position in that order is their index. A name used by several subterrains is numbered
    (``rough_0``, ``rough_1``, ...), and the shared name selects all of them.

    Example::

        terrain_map = self.terrain_manager.terrain_map
        terrain_map.names                # ("start", "rough_0", "rough_1", "stairs")
        terrain_map.get_bounds("stairs") # (x_min, x_max, y_min, y_max)
        terrain_map.indices("rough")     # (1, 2)
    """

    bounds: Bounds
    """The whole terrain."""
    names: tuple[str, ...] = ()
    """The subterrain names, in index order."""
    types: tuple[str, ...] = ()
    """The subterrain types, like ``"flat_terrain"``, in index order."""
    subterrain_bounds: tuple[Bounds, ...] = ()
    """Each subterrain's bounds, in index order."""
    groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """The names shared by more than one subterrain, and the indices of the subterrains sharing each one."""

    """
    Making a map
    """

    @classmethod
    def from_grid(
        cls,
        pos: Sequence[float],
        subterrain_size: tuple[float, float],
        labels: Sequence[Sequence[str]],
        types: Sequence[Sequence[str]],
    ) -> TerrainMap:
        """
        Map a grid of same-size subterrains.

        Args:
            pos: The position of the terrain's ``(x_min, y_min)`` corner.
            subterrain_size: The ``(x, y)`` size of each subterrain, in meters.
            labels: What to call each subterrain, as ``labels[x][y]``. A label used more than
                    once is numbered (``rough_0``, ``rough_1``, ...), and becomes a group.
            types: Each subterrain's type, as ``types[x][y]``.
        """
        n_x, n_y = len(labels), len(labels[0])
        x_size, y_size = subterrain_size
        x_min, y_min = float(pos[0]), float(pos[1])

        names, groups = cls._name_subterrains(
            [label for row in labels for label in row]
        )
        subterrain_bounds = []
        for grid_x in range(n_x):
            for grid_y in range(n_y):
                x_start = x_min + grid_x * x_size
                y_start = y_min + grid_y * y_size
                subterrain_bounds.append(
                    (
                        x_start,
                        x_start + x_size,
                        y_start,
                        y_start + y_size,
                    )
                )

        return cls(
            bounds=(x_min, x_min + n_x * x_size, y_min, y_min + n_y * y_size),
            names=names,
            types=tuple(t for row in types for t in row),
            subterrain_bounds=tuple(subterrain_bounds),
            groups=groups,
        )

    @classmethod
    def from_entity(cls, entity: RigidEntity) -> TerrainMap:
        """
        Map a terrain entity in the scene.

        - A ``gs.morphs.Terrain`` grid: each subterrain is named after its type, and a type
          used more than once is numbered in grid order (``flat_terrain_0``, ...).
        - A ``gs.morphs.Terrain`` built from a height field: the height field's area, with no
          subterrains.
        - Anything else, like a plane or a mesh: its bounding box, with no subterrains.
        """
        morph = getattr(entity, "morph", None)
        n_subterrains = getattr(morph, "n_subterrains", None)
        height_field = getattr(morph, "height_field", None)

        if n_subterrains is not None and height_field is not None:
            # A terrain built from a height field ignores its subterrain settings
            x_size = (len(height_field) - 1) * morph.horizontal_scale
            y_size = (len(height_field[0]) - 1) * morph.horizontal_scale
            x_min, y_min = float(morph.pos[0]), float(morph.pos[1])
            return cls(bounds=(x_min, x_min + x_size, y_min, y_min + y_size))

        if n_subterrains is not None:
            n_x, n_y = n_subterrains
            types = [
                [morph.subterrain_types[x][y] for y in range(n_y)] for x in range(n_x)
            ]
            return cls.from_grid(morph.pos, morph.subterrain_size, types, types)

        # Anything else is measured by its bounding box, for the first environment
        (geom,) = entity.geoms
        aabb = geom.get_AABB()
        if aabb.ndim == 3:
            aabb = aabb[0]
        (x_min, y_min, _), (x_max, y_max, _) = aabb.tolist()
        return cls(bounds=(x_min, x_max, y_min, y_max))

    """
    Lookup
    """

    def index(self, name: str) -> int:
        """
        The index of one subterrain, by name.

        Raises:
            KeyError: If the name is unknown, or shared by more than one subterrain.
        """
        if name in self.groups:
            sharing = [self.names[i] for i in self.groups[name]]
            raise KeyError(
                f"'{name}' names more than one subterrain: {sharing}. "
                "Use one of those names instead."
            )
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(self._unknown_name_message(name)) from None

    def indices(self, name: str) -> tuple[int, ...]:
        """
        The indices of every subterrain a name selects: one for a subterrain's own name, or
        all of them for a shared name.

        Raises:
            KeyError: If the name is unknown.
        """
        if name in self.groups:
            return self.groups[name]
        try:
            return (self.names.index(name),)
        except ValueError:
            raise KeyError(self._unknown_name_message(name)) from None

    def get_bounds(self, name: str | None = None) -> Bounds:
        """
        The bounds of one subterrain, or of the whole terrain if ``name`` is None.

        Raises:
            KeyError: If the name is unknown, or shared by more than one subterrain.
        """
        if name is None:
            return self.bounds
        return self.subterrain_bounds[self.index(name)]

    def _unknown_name_message(self, name: str) -> str:
        """The error for an unknown name, listing every name that would work."""
        valid = list(self.names) + list(self.groups)
        return f"No subterrain named '{name}'. Valid names: {valid}"

    @staticmethod
    def _name_subterrains(
        labels: Sequence[str],
    ) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
        """
        Give each subterrain in a grid a unique name from its label, in grid order.

        A label used once is that subterrain's name. A label used by several subterrains is
        numbered (``rough_0``, ``rough_1``, ...), and becomes a group of them.

        Returns:
            The subterrain names, and the subterrain indices of each group.

        Raises:
            ValueError: If a numbered name would clash with a name used elsewhere.
        """
        counts = Counter(labels)
        names: list[str] = []
        groups: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            if counts[label] > 1:
                group = groups.setdefault(label, [])
                names.append(f"{label}_{len(group)}")
                group.append(i)
            else:
                names.append(label)

        taken = Counter(names) + Counter(list(groups))
        clashes = sorted(name for name, count in taken.items() if count > 1)
        if clashes:
            raise ValueError(
                f"The subterrain name '{clashes[0]}' would refer to two different things: "
                "a subterrain's own name, and the number given to a repeated name. "
                "Rename one of them."
            )
        return tuple(names), {label: tuple(idx) for label, idx in groups.items()}
