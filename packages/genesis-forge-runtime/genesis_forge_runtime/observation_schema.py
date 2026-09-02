"""How the policy's observation vector is laid out.

The half of the manifest that :mod:`genesis_forge_runtime.observations` consumes:
what each slot holds, how wide it is, and what it is scaled by.

Every slot is an input you supply each tick. Most come from sensors; some echo the
policy's own previous output, which you read off the decoder. The bundle does not
distinguish them -- from the assembler's side they are passed in the same way, and
the deployment guide covers which of your entries is which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import HISTORY_NEWEST_FIRST
from .errors import MalformedBundleError
from .serialization import require


@dataclass(frozen=True)
class ObservationEntry:
    """One named slot in the policy's observation vector."""

    name: str
    size: int
    scale: float = 1.0
    description: str | None = None
    units: str | None = None

    def describe(self) -> str:
        """One-line human summary, used by the listings and the wiring stub."""
        parts = [f"{self.name} ({self.size} value{'s' if self.size != 1 else ''})"]
        if self.units:
            parts.append(f"in {self.units}")
        if self.scale != 1.0:
            parts.append(f"scaled by {self.scale}")
        summary = ", ".join(parts)
        if self.description:
            summary = f"{summary} -- {self.description}"
        return summary

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, where: str) -> ObservationEntry:
        name = require(data, "name", where=where)
        entry = cls(
            name=name,
            size=int(require(data, "size", where=f"{where}.{name}")),
            scale=float(data.get("scale", 1.0)),
            description=data.get("description"),
            units=data.get("units"),
        )
        return entry

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "size": self.size, "scale": self.scale}
        if self.description is not None:
            data["description"] = self.description
        if self.units is not None:
            data["units"] = self.units
        return data


@dataclass(frozen=True)
class ObservationLayout:
    """Ordering, scaling, and history configuration of the policy's input vector."""

    entries: tuple[ObservationEntry, ...]
    history_length: int = 1
    history_order: str = HISTORY_NEWEST_FIRST

    @property
    def single_size(self) -> int:
        """Width of one observation tick, before history stacking."""
        return sum(entry.size for entry in self.entries)

    @property
    def total_size(self) -> int:
        """Width of the full vector handed to the policy."""
        return self.single_size * self.history_length

    def entry(self, name: str) -> ObservationEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        known = ", ".join(item.name for item in self.entries)
        raise KeyError(f"No observation entry named '{name}'. Known entries: {known}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationLayout:
        where = "observations"
        raw_entries = require(data, "entries", where=where)
        if not isinstance(raw_entries, list) or not raw_entries:
            raise MalformedBundleError(
                "'observations.entries' must be a non-empty list of observation entries."
            )
        entries = tuple(
            ObservationEntry.from_dict(item, where=f"{where}.entries")
            for item in raw_entries
        )
        names = [entry.name for entry in entries]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise MalformedBundleError(
                f"Duplicate observation entry names: {', '.join(sorted(duplicates))}."
            )
        history_length = int(data.get("history_length", 1))
        if history_length < 1:
            raise MalformedBundleError(
                f"'observations.history_length' must be at least 1, got {history_length}."
            )
        history_order = data.get("history_order", HISTORY_NEWEST_FIRST)
        if history_order != HISTORY_NEWEST_FIRST:
            raise MalformedBundleError(
                f"Unsupported observation history order '{history_order}'. This runtime "
                f"only implements '{HISTORY_NEWEST_FIRST}'."
            )
        return cls(
            entries=entries,
            history_length=history_length,
            history_order=history_order,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_length": self.history_length,
            "history_order": self.history_order,
            "single_size": self.single_size,
            "total_size": self.total_size,
            "entries": [entry.to_dict() for entry in self.entries],
        }
