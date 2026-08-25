"""Prove the deployment runtime reproduces the training pipeline, before export.

Manager-owned deployment counterparts are written by hand, so nothing structural
guarantees the numpy decode matches the torch ``process_actions`` it mirrors. This
harness closes that gap: it runs the *actual* ``genesis_forge_deploy`` classes --
the same code the robot imports -- against the live torch pipeline on identical
inputs, and refuses the export if they disagree.

Tolerances are tiered by what can legitimately differ:

* numpy vs torch pipeline math -- same operations, same dtype, so near-bit-exact.
  Ordering and scale bugs produce large errors, so a tight bound costs nothing.
* onnxruntime vs torch -- graph rewrites change accumulation order, so looser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge_deploy import Manifest

    from .capture import Capture

#: numpy-vs-torch: same math in two libraries. Matches torch.testing float32 defaults.
PIPELINE_RTOL = 1.3e-6
PIPELINE_ATOL = 1e-5

#: onnxruntime-vs-torch: graph rewrites reorder accumulation.
POLICY_ATOL = 1e-5


class ParityError(Exception):
    """The deployment pipeline disagreed with the training pipeline."""


@dataclass
class ParityReport:
    """What the parity run compared, and the worst disagreement it saw."""

    ticks: int
    max_observation_error: float = 0.0
    max_action_error: dict[str, float] = field(default_factory=dict)
    golden: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> str:
        actions = ", ".join(
            f"{name}: {error:.2e}" for name, error in sorted(self.max_action_error.items())
        )
        return (
            f"parity over {self.ticks} tick(s) -- observations within "
            f"{self.max_observation_error:.2e}"
            + (f"; actions {actions}" if actions else "")
        )


def check_parity(
    capture: Capture,
    *,
    ticks: int = 6,
    seed: int = 0,
    rtol: float = PIPELINE_RTOL,
    atol: float = PIPELINE_ATOL,
) -> ParityReport:
    """Compare the numpy deployment pipeline against the live torch pipeline.

    Runs a multi-tick sequence so history stacking and any per-step decoder state
    are exercised, not just a single snapshot. Inputs mix seeded random values with
    clip-boundary values, since clipping is where the two implementations are most
    likely to diverge.

    Args:
        capture: What the exporter read out of the built environment.
        ticks: How many sequential ticks to compare.
        seed: Seed for the sampled inputs, so a failure is reproducible.
        rtol: Relative tolerance for the numpy-vs-torch comparison.
        atol: Absolute tolerance for the numpy-vs-torch comparison.

    Returns:
        A :class:`ParityReport`, including golden samples to ship in the bundle.

    Raises:
        ParityError: The two pipelines disagreed. The message names the component.
    """
    from genesis_forge_deploy import ObservationAssembler
    from genesis_forge_deploy.actions import ActionDecoder

    manifest: Manifest = capture.manifest
    assembler = ObservationAssembler(manifest.observations)
    decoder = ActionDecoder(manifest.actions)

    rng = np.random.default_rng(seed)
    report = ParityReport(ticks=ticks)

    golden_observations: list[np.ndarray] = []
    golden_policy_inputs: list[np.ndarray] = []
    golden_actions: list[np.ndarray] = []
    golden_targets: list[np.ndarray] = []

    # Training mutates its own history buffer as a side effect of observing, so
    # snapshot it and put it back -- exporting must not disturb the environment.
    with _preserved_observation_history(capture.observation_manager):
        assembler.reset()
        decoder.reset()

        for tick in range(ticks):
            sensor_values = _sample_observation_values(manifest, rng)
            auto_filled = _sample_auto_filled_values(manifest, rng)

            # Feed the numpy side the auto-fill sources it would normally get from
            # its own previous output, and hand the torch side those same numbers.
            if auto_filled.raw is not None or auto_filled.decoded:
                assembler.record_actions(auto_filled.raw, decoded=auto_filled.decoded)

            numpy_obs = assembler.assemble(sensor_values)
            torch_obs = _torch_observations(capture, manifest, sensor_values, auto_filled)

            error = _max_abs_error(numpy_obs, torch_obs)
            report.max_observation_error = max(report.max_observation_error, error)
            _require_close(
                numpy_obs,
                torch_obs,
                rtol=rtol,
                atol=atol,
                component="observation assembly",
                detail=(
                    f"tick {tick}: the deployment assembler and the training "
                    f"ObservationManager produced different vectors"
                ),
            )

            raw_actions = _sample_actions(manifest, rng, tick=tick)
            decoded = decoder.decode(raw_actions)

            for spec in manifest.actions:
                manager = capture.action_managers[spec.name]
                chunk = raw_actions[spec.slice_start : spec.slice_end]
                torch_chunk = torch.as_tensor(
                    np.tile(chunk, (capture.num_envs, 1)), dtype=torch.float32
                )
                torch_targets = manager.process_actions(torch_chunk).detach()[0]
                numpy_targets = decoded.by_manager[spec.name]

                error = _max_abs_error(numpy_targets, torch_targets)
                report.max_action_error[spec.name] = max(
                    report.max_action_error.get(spec.name, 0.0), error
                )
                _require_close(
                    numpy_targets,
                    torch_targets,
                    rtol=rtol,
                    atol=atol,
                    component=f"action manager '{spec.name}' ({spec.deploy_type})",
                    detail=(
                        f"tick {tick}: the deployment decoder and the manager's "
                        f"process_actions produced different joint targets"
                    ),
                )

            golden_observations.append(numpy_obs)
            golden_policy_inputs.append(
                np.concatenate([sensor_values[name] for name in sorted(sensor_values)])
                if sensor_values
                else np.zeros(0, dtype=np.float32)
            )
            golden_actions.append(raw_actions)
            golden_targets.append(decoded.targets)

    report.golden = {
        "observations": np.asarray(golden_observations, dtype=np.float32),
        "raw_actions": np.asarray(golden_actions, dtype=np.float32),
        "joint_targets": np.asarray(golden_targets, dtype=np.float32),
    }
    if golden_policy_inputs and golden_policy_inputs[0].size:
        report.golden["sensor_inputs"] = np.asarray(
            golden_policy_inputs, dtype=np.float32
        )
    return report


def check_policy_parity(
    policy_path: str,
    torch_policy: Any,
    observations: np.ndarray,
    *,
    input_name: str = "obs",
    atol: float = POLICY_ATOL,
) -> float:
    """Compare an exported ONNX policy against the live torch policy.

    Closes the seam the pipeline check cannot see: a policy graph that silently
    dropped its observation normalizer passes every other check and then misbehaves
    on hardware.

    Args:
        policy_path: The exported ``.onnx`` file.
        torch_policy: A callable taking a float32 tensor and returning actions.
        observations: Observation vectors to compare on, shaped ``(ticks, size)``.
        input_name: The ONNX graph's input name.
        atol: Absolute tolerance.

    Returns:
        The largest absolute difference observed.

    Raises:
        ParityError: The two disagreed, or onnxruntime is not installed.
    """
    try:
        import onnxruntime
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ParityError(
            "onnxruntime is required to verify an exported policy. Install it with "
            "`pip install onnxruntime`, or export without a policy file."
        ) from error

    session = onnxruntime.InferenceSession(
        str(policy_path), providers=["CPUExecutionProvider"]
    )
    worst = 0.0
    for observation in np.atleast_2d(observations).astype(np.float32):
        batched = observation[None, :]
        onnx_actions = np.asarray(session.run(None, {input_name: batched})[0]).ravel()
        with torch.no_grad():
            torch_actions = (
                torch_policy(torch.as_tensor(batched, dtype=torch.float32))
                .detach()
                .cpu()
                .numpy()
                .ravel()
            )
        worst = max(worst, float(np.max(np.abs(onnx_actions - torch_actions))))
        if worst > atol:
            raise ParityError(
                f"The exported ONNX policy disagrees with the trained torch policy "
                f"(largest difference {worst:.3e}, tolerance {atol:.1e}). The usual "
                f"cause is an observation normalizer that did not make it into the "
                f"exported graph."
            )
    return worst


"""
Internal helpers
"""


class _preserved_observation_history:
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


def _sample_observation_values(
    manifest: Manifest, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Random sensor readings at realistic magnitudes, one per required entry."""
    return {
        entry.name: rng.uniform(-2.0, 2.0, size=entry.size).astype(np.float32)
        for entry in manifest.observations.required_inputs
    }


@dataclass
class _AutoFillSamples:
    """Sampled values for auto-filled entries, addressed the way each side needs.

    The numpy assembler reads them through ``record_actions`` (by stage), while the
    torch pipeline needs them keyed by observation name, so the same samples are
    carried in both shapes.
    """

    by_name: dict[str, np.ndarray] = field(default_factory=dict)
    raw: np.ndarray | None = None
    decoded: dict[str, np.ndarray] = field(default_factory=dict)


def _sample_auto_filled_values(
    manifest: Manifest, rng: np.random.Generator
) -> _AutoFillSamples:
    """Values for pipeline-state entries, fed identically to both pipelines.

    This validates the *layout* of auto-filled entries -- their position, width and
    scale -- but not the semantic choice of which pipeline stage they echo, since
    both sides receive the same numbers by construction. That choice is verified on
    the export side, where ``current_actions`` is inspected directly.
    """
    from genesis_forge_deploy import STAGE_PROCESSED_ACTIONS

    samples = _AutoFillSamples()
    for entry in manifest.observations.pipeline_state_inputs:
        value = rng.uniform(-1.0, 1.0, size=entry.size).astype(np.float32)
        samples.by_name[entry.name] = value
        if entry.pipeline_stage == STAGE_PROCESSED_ACTIONS and entry.action_manager:
            samples.decoded[entry.action_manager] = value
        else:
            samples.raw = value
    return samples


def _sample_actions(
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


def _torch_observations(
    capture: Capture,
    manifest: Manifest,
    sensor_values: dict[str, np.ndarray],
    auto_filled: _AutoFillSamples,
) -> torch.Tensor:
    """Run the training-side observation pipeline on the same inputs."""
    overrides: dict[str, torch.Tensor] = {}

    for name in capture.observation_entry_names:
        if name in sensor_values:
            values = sensor_values[name]
        elif name in auto_filled.by_name:
            values = auto_filled.by_name[name]
        else:
            # Zero-width entries are skipped by the training pipeline but must
            # still be present in the override dict.
            values = np.zeros(0, dtype=np.float32)
        # A fresh tensor per call: the training override path scales in place.
        overrides[name] = torch.as_tensor(
            np.tile(values, (capture.num_envs, 1)), dtype=torch.float32
        )

    observations = capture.observation_manager.get_observations(values=overrides)
    return observations.detach()[0]


def _max_abs_error(numpy_values: np.ndarray, torch_values: torch.Tensor) -> float:
    expected = torch_values.detach().cpu().numpy().ravel()
    actual = np.asarray(numpy_values).ravel()
    if expected.shape != actual.shape:
        return float("inf")
    if expected.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def _require_close(
    numpy_values: np.ndarray,
    torch_values: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    component: str,
    detail: str,
) -> None:
    expected = torch_values.detach().cpu().numpy().ravel()
    actual = np.asarray(numpy_values).ravel()

    if expected.shape != actual.shape:
        raise ParityError(
            f"Parity failed in {component}. {detail}: the deployment pipeline "
            f"produced {actual.shape[0]} value(s) where training produced "
            f"{expected.shape[0]}."
        )

    if np.allclose(actual, expected, rtol=rtol, atol=atol):
        return

    difference = np.abs(actual - expected)
    worst = int(np.argmax(difference))
    raise ParityError(
        f"Parity failed in {component}. {detail}. Largest difference "
        f"{difference[worst]:.3e} at index {worst}: deployment produced "
        f"{actual[worst]:.6g}, training produced {expected[worst]:.6g} "
        f"(tolerance rtol={rtol:g}, atol={atol:g}). The bundle was not written."
    )


__all__ = [
    "PIPELINE_ATOL",
    "PIPELINE_RTOL",
    "POLICY_ATOL",
    "ParityError",
    "ParityReport",
    "check_parity",
    "check_policy_parity",
]
