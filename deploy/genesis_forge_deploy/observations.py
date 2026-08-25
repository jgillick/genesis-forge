"""Rebuild the policy's observation vector from real sensor readings.

This mirrors ``genesis_forge.managers.ObservationManager`` exactly -- same entry
order, same per-entry scaling, same newest-first history stacking -- minus the
simulator lookups and minus the training noise. The parity gate on the export
side proves the two agree before a bundle is ever written.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .bundle import (
    STAGE_PROCESSED_ACTIONS,
    STAGE_RAW_ACTIONS,
    ObservationEntry,
    ObservationLayout,
)


class ObservationError(Exception):
    """A value handed to the assembler was missing, mis-sized, or unexpected."""


class ObservationAssembler:
    """Assembles the policy input vector from named sensor values.

    Args:
        layout: The observation layout from a loaded bundle's manifest.
        dtype: Output dtype. Defaults to float32, what the policy expects.
        strict_inputs: Raise when handed a name the layout does not define.
            On by default: a typo'd or stale sensor key is exactly the silent
            mis-wiring this package exists to prevent.

    Example::

        assembler = bundle.observation_assembler()
        for entry in assembler.required_inputs:
            print(entry.describe())

        obs = assembler.assemble({
            "robot_ang_vel": imu.gyro,
            "dof_pos": joints.positions,
        })
    """

    def __init__(
        self,
        layout: ObservationLayout,
        *,
        dtype: Any = np.float32,
        strict_inputs: bool = True,
    ) -> None:
        self._layout = layout
        self._dtype = np.dtype(dtype)
        self._strict_inputs = strict_inputs

        self._offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for entry in layout.entries:
            self._offsets[entry.name] = (cursor, cursor + entry.size)
            cursor += entry.size

        self._last_raw_actions: np.ndarray | None = None
        self._last_decoded_actions: dict[str, np.ndarray] = {}
        self._history: list[np.ndarray] = []
        self.reset()

    """
    Properties
    """

    @property
    def layout(self) -> ObservationLayout:
        return self._layout

    @property
    def required_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries the caller must supply each tick. Pipeline state is excluded."""
        return self._layout.required_inputs

    @property
    def auto_filled_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries the runtime fills from the policy's own previous output."""
        return self._layout.pipeline_state_inputs

    @property
    def output_size(self) -> int:
        """Length of the vector :meth:`assemble` returns."""
        return self._layout.total_size

    """
    Public methods
    """

    def reset(self) -> None:
        """Clear history and remembered actions back to a fresh-start state.

        History fills with zeros, exactly as training's ``ObservationManager.reset()``
        does at the start of every episode. Call this whenever you (re)start control
        so the robot begins from the same state an episode began from in training.
        """
        self._history = [
            np.zeros(self._layout.single_size, dtype=self._dtype)
            for _ in range(self._layout.history_length)
        ]
        self._last_raw_actions = None
        self._last_decoded_actions = {}

    def record_actions(
        self,
        raw_actions: Any = None,
        *,
        decoded: dict[str, Any] | None = None,
    ) -> None:
        """Feed the policy's output back in, for auto-filled observation entries.

        Call this once per tick after running the policy when the layout contains
        pipeline-state entries (:attr:`auto_filled_inputs`). ``raw_actions`` is the
        policy's raw output; ``decoded`` maps an action manager's name to its
        post-decode joint targets.
        """
        if raw_actions is not None:
            self._last_raw_actions = self._to_array(raw_actions, name="raw_actions")
        if decoded:
            for manager_name, values in decoded.items():
                self._last_decoded_actions[manager_name] = self._to_array(
                    values, name=f"decoded[{manager_name}]"
                )

    def assemble(self, values: dict[str, Any] | None = None) -> np.ndarray:
        """Build one observation vector.

        Args:
            values: One entry per name in :attr:`required_inputs`. Values may be
                scalars, sequences, or numpy arrays; each is flattened and must
                match the entry's declared size.

        Returns:
            The policy input vector, shape ``(output_size,)``. Add a batch
            dimension (``obs[None, :]``) before handing it to onnxruntime.

        Raises:
            ObservationError: A required entry is missing, mis-sized, or unknown.
        """
        values = values or {}
        self._check_for_unknown_names(values)

        current = np.empty(self._layout.single_size, dtype=self._dtype)
        for entry in self._layout.entries:
            start, end = self._offsets[entry.name]
            current[start:end] = self._value_for(entry, values)

        # Rotate newest-first, reusing the oldest buffer -- the same rotation
        # ObservationManager.get_observations performs on the training side.
        buffer = self._history.pop()
        buffer[:] = current
        self._history.insert(0, buffer)

        if self._layout.history_length == 1:
            return self._history[0].copy()
        return np.concatenate(self._history)

    def describe_inputs(self) -> str:
        """Human-readable listing of everything the caller must supply."""
        lines = ["Observation inputs required each tick:"]
        lines.extend(f"  - {entry.describe()}" for entry in self.required_inputs)
        auto = self.auto_filled_inputs
        if auto:
            lines.append("Filled automatically from the policy's own output:")
            lines.extend(f"  - {entry.describe()}" for entry in auto)
        return "\n".join(lines)

    """
    Internal methods
    """

    def _value_for(self, entry: ObservationEntry, values: dict[str, Any]) -> np.ndarray:
        if entry.is_pipeline_state:
            raw = self._pipeline_state_value(entry)
        else:
            if entry.name not in values:
                raise ObservationError(
                    f"Missing observation value '{entry.name}'. This layout requires: "
                    f"{', '.join(item.name for item in self.required_inputs)}."
                )
            raw = self._to_array(values[entry.name], name=entry.name)

        if raw.size != entry.size:
            raise ObservationError(
                f"Observation '{entry.name}' expects {entry.size} value(s), got "
                f"{raw.size}."
            )

        if entry.scale != 1.0:
            # Multiply out-of-place: the caller's array must not be mutated, and
            # the training-side override path has an in-place-scaling quirk we
            # deliberately do not reproduce.
            return raw * np.asarray(entry.scale, dtype=self._dtype)
        return raw

    def _pipeline_state_value(self, entry: ObservationEntry) -> np.ndarray:
        """Resolve an auto-filled entry from the runtime's own last output."""
        if entry.pipeline_stage == STAGE_RAW_ACTIONS:
            source = self._last_raw_actions
        elif entry.pipeline_stage == STAGE_PROCESSED_ACTIONS:
            source = self._last_decoded_actions.get(entry.action_manager)
        else:  # pragma: no cover - the manifest loader rejects other stages
            raise ObservationError(
                f"Observation '{entry.name}' has unsupported pipeline stage "
                f"'{entry.pipeline_stage}'."
            )

        if source is None:
            # First tick: nothing has run yet. Training starts from zeros too.
            return np.zeros(entry.size, dtype=self._dtype)
        return source

    def _check_for_unknown_names(self, values: dict[str, Any]) -> None:
        if not self._strict_inputs:
            return
        known = {entry.name for entry in self._layout.entries}
        unknown = sorted(set(values) - known)
        auto_filled = {entry.name for entry in self.auto_filled_inputs} & set(values)
        if auto_filled:
            raise ObservationError(
                f"Observation(s) {', '.join(sorted(auto_filled))} are filled "
                f"automatically from the policy's output; pass them via "
                f"record_actions() instead of assemble()."
            )
        if unknown:
            raise ObservationError(
                f"Unknown observation name(s): {', '.join(unknown)}. Expected one of: "
                f"{', '.join(entry.name for entry in self.required_inputs)}."
            )

    def _to_array(self, value: Any, *, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=self._dtype)
        except (TypeError, ValueError) as error:
            raise ObservationError(
                f"Observation '{name}' could not be read as numbers: {error}"
            ) from error
        return np.atleast_1d(array).ravel()


def iter_input_names(entries: Iterable[ObservationEntry]) -> list[str]:
    """Convenience for stub generation and diagnostics."""
    return [entry.name for entry in entries]


__all__ = ["ObservationAssembler", "ObservationError", "iter_input_names"]
