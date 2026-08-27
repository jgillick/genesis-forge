"""Rebuild the policy's observation vector from real sensor readings.

This mirrors ``genesis_forge.managers.ObservationManager`` exactly -- same entry
order, same per-entry scaling, same newest-first history stacking -- minus the
simulator lookups and minus the training noise. The parity gate on the export
side proves the two agree before a bundle is ever written.

Every entry is supplied by the caller, including the ones that echo the pipeline's
own previous output (read those off the decoder). Nothing is filled in silently:
a feedback wire you forget raises, rather than quietly reading zeros forever.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ObservationError
from .observation_schema import ObservationEntry, ObservationLayout


class ObservationAssembler:
    """Assembles the policy input vector from named values.

    Args:
        layout: The observation layout from a loaded bundle's manifest.
        dtype: Output dtype. Defaults to float32, what the policy expects.

    A name the layout does not define is always rejected. It could not corrupt
    the vector -- assembly reads entries by name, so a stray key is inert -- but
    it means the control loop and the bundle disagree about what this policy
    consumes, and that is worth hearing about on the bench rather than later.

    Example::

        observation_assembler = bundle.create_observation_assembler()
        action_decoder = bundle.create_action_decoder()
        print(observation_assembler.describe_inputs())

        obs = observation_assembler.assemble({
            "robot_ang_vel": imu.gyro,
            "dof_pos": joints.positions,
            "actions": action_decoder.last_target_actions,
        })
    """

    def __init__(
        self,
        layout: ObservationLayout,
        *,
        dtype: Any = np.float32,
    ) -> None:
        self._layout = layout
        self._dtype = np.dtype(dtype)

        # Determine where each entry lives in the flat vector
        self._offsets: dict[str, tuple[int, int]] = {}
        cursor = 0
        for entry in layout.entries:
            self._offsets[entry.name] = (cursor, cursor + entry.size)
            cursor += entry.size

        self._history: list[np.ndarray] = []
        self.reset()

    """
    Properties
    """

    @property
    def inputs(self) -> tuple[ObservationEntry, ...]:
        """Everything you pass to :meth:`assemble` each tick.

        The union of :attr:`sensor_inputs` and :attr:`pipeline_state_inputs` --
        the two differ only in where you read the value from, not in whether you
        have to supply it.
        """
        return self._layout.entries

    @property
    def sensor_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries that come from real sensors."""
        return self._layout.sensor_inputs

    @property
    def pipeline_state_inputs(self) -> tuple[ObservationEntry, ...]:
        """Entries fed back from the decoder's previous output."""
        return self._layout.pipeline_state_inputs

    @property
    def output_size(self) -> int:
        """Length of the vector :meth:`assemble` returns."""
        return self._layout.total_size

    """
    Public methods
    """

    def reset(self) -> None:
        """Clear the stacked history back to zeros.

        This is exactly what training's ``ObservationManager.reset()`` does at the
        start of every episode. Call it whenever you (re)start control so the robot
        begins from the same state an episode began from in training.
        """
        self._history = [
            np.zeros(self._layout.single_size, dtype=self._dtype)
            for _ in range(self._layout.history_length)
        ]

    def assemble(self, values: dict[str, Any] | None = None) -> np.ndarray:
        """Build one observation vector.

        Args:
            values: One entry per name in :attr:`inputs`. Sensor readings
                come from your hardware; entries that echo the pipeline's own output
                come from the decoder (``action_decoder.last_target_actions`` or
                ``action_decoder.last_raw_actions``). Values may be scalars, sequences, or
                numpy arrays; each is flattened and must match the declared size.

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
        buffer = self._history.pop()
        buffer[:] = current
        self._history.insert(0, buffer)

        if self._layout.history_length == 1:
            return self._history[0].copy()
        return np.concatenate(self._history)

    def describe_inputs(self) -> str:
        """Human-readable listing of everything the caller must supply."""
        lines = ["Sensor values to supply each tick:"]
        lines.extend(f"  - {entry.describe()}" for entry in self.sensor_inputs)
        fed_back = self.pipeline_state_inputs
        if fed_back:
            lines.append("Values to feed back from the decoder:")
            lines.extend(f"  - {entry.describe()}" for entry in fed_back)
        return "\n".join(lines)

    """
    Internal methods
    """

    def _value_for(self, entry: ObservationEntry, values: dict[str, Any]) -> np.ndarray:
        if entry.name not in values:
            raise ObservationError(self._missing_value_message(entry))

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

    def _missing_value_message(self, entry: ObservationEntry) -> str:
        """Say what is missing -- and for fed-back entries, where to get it."""
        if entry.is_pipeline_state:
            return (
                f"Missing observation value '{entry.name}'. It echoes the policy's "
                f"own previous output rather than a sensor -- pass "
                f"{entry.decoder_source}."
            )
        return (
            f"Missing observation value '{entry.name}'. This layout requires: "
            f"{', '.join(item.name for item in self.inputs)}."
        )

    def _check_for_unknown_names(self, values: dict[str, Any]) -> None:
        known = {entry.name for entry in self._layout.entries}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ObservationError(
                f"Unknown observation name(s): {', '.join(unknown)}. Expected one of: "
                f"{', '.join(entry.name for entry in self.inputs)}."
            )

    def _to_array(self, value: Any, *, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=self._dtype)
        except (TypeError, ValueError) as error:
            raise ObservationError(
                f"Observation '{name}' could not be read as numbers: {error}"
            ) from error
        return np.atleast_1d(array).ravel()
