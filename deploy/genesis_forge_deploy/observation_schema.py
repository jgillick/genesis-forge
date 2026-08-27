"""How the policy's observation vector is laid out.

The half of the manifest that :mod:`genesis_forge_deploy.observations` consumes:
what each slot holds, how wide it is, what it is scaled by, and -- for values that
echo the pipeline's own output -- where on the decoder to read it from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import (
    HISTORY_NEWEST_FIRST,
    SOURCE_PIPELINE_STATE,
    SOURCE_SENSOR,
    STAGE_RAW_ACTIONS,
    STAGE_TARGET_ACTIONS,
)
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
    source: str = SOURCE_SENSOR
    pipeline_stage: str | None = None
    action_manager: str | None = None

    @property
    def is_pipeline_state(self) -> bool:
        """True when this entry echoes the pipeline's own output, not a sensor."""
        return self.source == SOURCE_PIPELINE_STATE

    @property
    def decoder_source(self) -> str:
        """The decoder expression to read this entry's value from, verbatim.

        Empty for sensor inputs. When the entry belongs to a specific action
        manager the per-manager form is used, because the flat properties hold the
        whole policy vector -- which only coincides with one manager's slice when
        there is exactly one manager.
        """
        if not self.is_pipeline_state:
            return ""
        attribute = f"last_{self.pipeline_stage}"
        if self.action_manager:
            return f'action_decoder.{attribute}_by_manager["{self.action_manager}"]'
        return f"action_decoder.{attribute}"

    def describe(self) -> str:
        """One-line human summary, used by the listings and the wiring stub."""
        parts = [f"{self.name} ({self.size} value{'s' if self.size != 1 else ''})"]
        if self.units:
            parts.append(f"in {self.units}")
        if self.scale != 1.0:
            parts.append(f"scaled by {self.scale}")
        if self.is_pipeline_state:
            parts.append(f"from {self.decoder_source}")
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
            source=data.get("source", SOURCE_SENSOR),
            pipeline_stage=data.get("pipeline_stage"),
            action_manager=data.get("action_manager"),
        )
        if entry.source not in (SOURCE_SENSOR, SOURCE_PIPELINE_STATE):
            raise MalformedBundleError(
                f"Observation entry '{name}' has unknown source '{entry.source}'. "
                f"Expected '{SOURCE_SENSOR}' or '{SOURCE_PIPELINE_STATE}'."
            )
        if entry.is_pipeline_state and entry.pipeline_stage not in (
            STAGE_RAW_ACTIONS,
            STAGE_TARGET_ACTIONS,
        ):
            raise MalformedBundleError(
                f"Observation entry '{name}' is marked pipeline state but its stage is "
                f"'{entry.pipeline_stage}'. Expected '{STAGE_RAW_ACTIONS}' or "
                f"'{STAGE_TARGET_ACTIONS}'."
            )
        if entry.pipeline_stage == STAGE_TARGET_ACTIONS and not entry.action_manager:
            raise MalformedBundleError(
                f"Observation entry '{name}' echoes target actions but does not say "
                f"which action manager they come from, so the runtime cannot tell you "
                f"where to read them. Re-export with a current version of Genesis Forge."
            )
        return entry

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "size": self.size, "scale": self.scale}
        if self.description is not None:
            data["description"] = self.description
        if self.units is not None:
            data["units"] = self.units
        if self.source != SOURCE_SENSOR:
            data["source"] = self.source
        if self.pipeline_stage is not None:
            data["pipeline_stage"] = self.pipeline_stage
        if self.action_manager is not None:
            data["action_manager"] = self.action_manager
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

    @property
    def sensor_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries that come from real sensors."""
        return tuple(entry for entry in self.entries if not entry.is_pipeline_state)

    @property
    def pipeline_state_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries fed back from the decoder's previous output."""
        return tuple(entry for entry in self.entries if entry.is_pipeline_state)

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
