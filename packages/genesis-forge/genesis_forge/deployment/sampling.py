"""Generating the inputs the parity gate compares on.

Random values alone would only ever exercise the linear middle of each transform,
so the samples deliberately include clip-boundary values -- clipping is where two
implementations most easily disagree -- and run as a multi-tick sequence so history
stacking and per-step decoder state are exercised too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import genesis as gs
import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge_runtime import Manifest

    from .capture import Capture


class preserved_observation_history:
    """Zero an ObservationManager's history for the comparison, then restore it.

    Export normally happens *after* training, so the manager's history buffer holds
    whatever the last step left there. The deployment assembler starts from a reset,
    so both sides must start from the same zeroed state or every comparison would
    diverge on the older history slots for reasons that have nothing to do with
    correctness. The original contents go back afterwards -- exporting must not
    disturb an environment someone may still be using.
    """

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._snapshot: list[torch.Tensor] | None = None

    def __enter__(self) -> None:
        history = getattr(self._manager, "_history", None)
        if history:
            self._snapshot = [tensor.clone() for tensor in history]
            for tensor in history:
                tensor.zero_()

    def __exit__(self, *_exc: object) -> None:
        if self._snapshot is None:
            return
        history = getattr(self._manager, "_history", None)
        if history and len(history) == len(self._snapshot):
            for live, saved in zip(history, self._snapshot, strict=True):
                live[:] = saved


def sample_observation_values(
    manifest: Manifest, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Random values at realistic magnitudes, one per entry the caller supplies."""
    return {
        entry.name: rng.uniform(-2.0, 2.0, size=entry.size).astype(np.float32)
        for entry in manifest.observations.entries
    }


def sample_actions(
    manifest: Manifest, rng: np.random.Generator, *, tick: int
) -> np.ndarray:
    """Random policy output, with clip-boundary values mixed in.

    Every third tick pushes values well outside the usual range, so the clipping
    branches -- where two implementations most easily disagree -- are exercised
    rather than only the linear region.
    """
    size = manifest.num_actions
    if tick % 3 == 1:
        return np.full(size, 25.0, dtype=np.float32)
    if tick % 3 == 2:
        return np.full(size, -25.0, dtype=np.float32)
    return rng.uniform(-1.5, 1.5, size=size).astype(np.float32)


def torch_observations(
    capture: Capture,
    observation_values: dict[str, np.ndarray],
) -> torch.Tensor:
    """Run the training-side observation pipeline on the same inputs."""
    overrides: dict[str, torch.Tensor] = {}

    for name in capture.observation_entry_names:
        if name in observation_values:
            values = observation_values[name]
        else:
            # Zero-width entries are skipped by the training pipeline but must
            # still be present in the override dict.
            values = np.zeros(0, dtype=np.float32)
        # On gs.device, where the manager's own buffers live. The copy into the
        # output buffer would convert a CPU tensor anyway, but relying on that
        # would leave the one path that does not convert -- action decoding --
        # as the only place a device mismatch shows up.
        overrides[name] = torch.as_tensor(
            np.tile(values, (capture.num_envs, 1)),
            dtype=torch.float32,
            device=gs.device,
        )

    observations = capture.observation_manager.get_observations(values=overrides)
    return observations.detach()[0]
