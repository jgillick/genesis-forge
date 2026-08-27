"""Export a built environment into a deployment bundle.

The whole flow is: read the contract out of the managers, prove the numpy runtime
reproduces the training pipeline, and only then write anything to disk. A bundle
that exists is a bundle that passed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from genesis_forge_deploy import POLICY_STEM, save_bundle

from .capture import Capture, capture_environment
from .errors import ExportError
from .parity import ParityReport, check_parity
from .policy_parity import infer_policy_format, validate_policy


def export(
    env: Any,
    path: str | Path,
    *,
    policy_path: str | Path | None = None,
    reference_policy: Any = None,
    checkpoint: str | Path | None = None,
    parity_ticks: int = 6,
    seed: int = 0,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Capture a built environment into a deployment bundle.

    Writes a directory holding ``manifest.json`` (the readable deployment
    contract), ``golden.npz`` (recorded input/output pairs that double as an
    on-robot smoke test), and the exported policy when one is supplied.

    The parity gate is not optional. Before anything is written, the numpy
    deployment classes are run against the live torch pipeline; if they disagree,
    the export aborts and names the component that diverged.

    Args:
        env: A built :class:`~genesis_forge.ManagedEnvironment`.
        path: Directory to write the bundle to.
        policy_path: An exported policy to copy into the bundle. ONNX is the
            documented path, but TorchScript (or anything else you load
            yourself) works too -- the bundle records the format rather than
            requiring one.
        reference_policy: The trained policy still live in memory -- a torch
            callable -- for the exported file to be checked against. That check
            is what catches a dropped observation normalizer, or a stale file
            from a previous run. ONNX and TorchScript are both supported.
        checkpoint: Path to the trained checkpoint, recorded as provenance.
        parity_ticks: How many sequential ticks the parity gate compares.
        seed: Seed for the parity inputs, so a failure reproduces.
        overwrite: Replace an existing bundle directory.
        verbose: Print a short summary of what was written.

    Returns:
        The bundle directory.

    Raises:
        ExportError: The environment cannot be exported as configured, or the
            destination already exists and ``overwrite`` is False.
        ParityError: The deployment pipeline disagreed with training. Nothing is
            written.

    Example::

        from genesis_forge.deployment import export

        env = MyEnv(num_envs=1)
        env.build()
        export(env, "./my_policy", policy_path="policy.onnx")
    """
    destination = Path(path)
    if destination.exists():
        if not overwrite:
            raise ExportError(
                f"'{destination}' already exists. Pass overwrite=True to replace it."
            )
        if not destination.is_dir():
            raise ExportError(f"'{destination}' exists and is not a directory.")

    policy_source = Path(policy_path) if policy_path else None
    policy_file = None
    policy_format = None
    if policy_source is not None:
        if not policy_source.is_file():
            raise ExportError(f"No policy file at '{policy_source}'.")
        # Keep the file's own extension: a bundle must not claim to hold an ONNX
        # graph when it holds a TorchScript module.
        policy_file = f"{POLICY_STEM}{policy_source.suffix}"
        policy_format = infer_policy_format(policy_source)

    capture = capture_environment(
        env,
        checkpoint=str(checkpoint) if checkpoint else None,
        policy_file=policy_file,
        policy_format=policy_format,
        reference_policy=reference_policy,
    )

    # The gate. Raises ParityError before anything reaches disk.
    report = check_parity(capture, ticks=parity_ticks, seed=seed)

    if policy_source is not None and reference_policy is not None:
        worst = validate_policy(
            policy_source,
            reference_policy,
            report.golden["observations"],
            policy_format=policy_format,
            input_name=capture.manifest.policy.input_name,
        )
        report.golden["policy_max_error"] = _as_array(worst)

    # Build the bundle somewhere temporary, then move it into place, so a failure
    # part-way through cannot leave a half-written bundle looking usable.
    with tempfile.TemporaryDirectory(prefix="genesis-forge-export-") as staging:
        staged = save_bundle(
            Path(staging) / "bundle", capture.manifest, golden=report.golden
        )
        if policy_source is not None:
            shutil.copy2(policy_source, staged / policy_file)

        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(destination))

    if verbose:
        _report(destination, capture, report)
        if policy_source is not None and reference_policy is None:
            print(
                f"  note: {policy_file} was packaged but not verified against the "
                f"trained policy -- pass reference_policy= to check it"
            )
    return destination


def _as_array(value: float):
    return np.asarray([value], dtype=np.float32)


def _report(destination: Path, capture: Capture, report: ParityReport) -> None:
    manifest = capture.manifest
    layout = manifest.observations
    print(f"Deployment bundle written to {destination}")
    print(f"  {report.summary()}")
    fed_back = len(layout.pipeline_state_inputs)
    feedback_note = f", {fed_back} fed back from the decoder" if fed_back else ""
    print(
        f"  observations: {layout.total_size} values "
        f"({len(layout.sensor_inputs)} sensor input(s) to wire up{feedback_note})"
    )
    print(f"  actions: {manifest.num_actions} joint target(s)")
    print(f"  control rate: {manifest.control_hz:.1f} Hz")
    print("  install the runtime on the robot with: pip install genesis-forge-deploy")
