"""Export a built environment into a deployment bundle.

The whole flow is: read the contract out of the managers, prove the numpy runtime
reproduces the training pipeline, and only then write anything to disk. A bundle
that exists is a bundle that passed.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from genesis_forge_runtime import (
    ARCHIVE_SUFFIX,
    MANIFEST_FILENAME,
    POLICY_FORMAT_ONNX,
    POLICY_FORMAT_TORCHSCRIPT,
    POLICY_STEM,
    Bundle,
    save_bundle,
)

from .capture import Capture, capture_environment
from .errors import ExportError
from .parity import ParityReport, check_parity
from .provenance import clean_additional

#: What a policy file's extension says it holds. Recorded in the manifest so an
#: operator can see what the bundle carries; nothing here loads or checks it.
_POLICY_FORMATS = {
    ".onnx": POLICY_FORMAT_ONNX,
    ".pt": POLICY_FORMAT_TORCHSCRIPT,
    ".pth": POLICY_FORMAT_TORCHSCRIPT,
    ".jit": POLICY_FORMAT_TORCHSCRIPT,
    ".ts": POLICY_FORMAT_TORCHSCRIPT,
}


def export(
    env: Any,
    path: str | Path,
    *,
    policy_path: str | Path | Sequence[str | Path] | None = None,
    additional_provenance: dict[str, Any] | None = None,
    archive: bool = True,
    parity_ticks: int = 6,
    seed: int = 0,
    overwrite: bool = True,
    verbose: bool = True,
) -> Bundle:
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
        policy_path: The exported policy to copy into the bundle. ONNX is the
            documented path, but TorchScript (or anything else you load
            yourself) works too -- the bundle records the format rather than
            requiring one.

            Pass a list when the export produced more than one file, as ONNX
            does once the weights exceed its inline threshold and as OpenVINO
            always does. The first entry is the one the runtime loads; the rest
            are copied beside it under their own names, since a graph refers to
            its companions by filename::

                policy_path=["policy.onnx", "policy.onnx.data"]

            Checking that what landed in the bundle still matches the policy it
            came from is yours to do, in the script that exports it; the
            deployment guide shows how, and it is what catches a companion file
            left behind.
        additional_provenance: Anything you want recorded about where this bundle
            came from, written to ``provenance.additional`` in the manifest. The
            exporter stamps what it can measure itself -- the export time and the
            Genesis Forge and torch versions -- but everything else depends on how
            you train, so it is yours to state rather than the library's to guess.
            The conventional keys are ``checkpoint``, ``framework`` and
            ``framework_version``; any JSON-friendly key is accepted::

                additional_provenance={
                    "checkpoint": "logs/my_run/model_500.pt",
                    "framework": "rsl_rl",
                    "framework_version": "5.4.2",
                }
        archive: Write the bundle as a single ``.gfb`` file, the default, since
            that is what you copy to a robot -- one artifact, and a transfer that
            either arrives whole or not at all. Pass False for a plain directory,
            which is easier to poke at while you are working. ``load_bundle``
            reads either.
        parity_ticks: How many sequential ticks the parity gate compares.
        seed: Seed for the parity inputs, so a failure reproduces.
        overwrite: Replace a bundle already at this path, which is the default:
            re-exporting after every training run is the normal thing to do.
            Whatever is there must itself be a bundle -- a path holding anything
            else is refused however this is set, so a mistyped destination cannot
            cost you a file.
        verbose: Print a short summary of what was written.

    Returns:
        The :class:`~genesis_forge_runtime.Bundle` that was written. Its manifest
        and golden samples are already in memory, so describing or checking what
        you just exported does not read the bundle back -- ``bundle.path`` is
        where it landed.

    Raises:
        ExportError: The environment cannot be exported as configured, or the
            destination already exists and ``overwrite`` is False.
        ParityError: The deployment pipeline disagreed with training. Nothing is
            written.

    Example::

        from genesis_forge.deployment import export

        env = MyEnv(num_envs=1)
        env.build()

        bundle = export(env, "./my_policy", policy_path="policy.onnx")
        print(bundle.describe())
    """
    # Checked first: a value that cannot be written should fail now, not after
    # the parity gate has run.
    additional_provenance = clean_additional(additional_provenance)

    destination = Path(path)
    if archive and not destination.suffix:
        destination = destination.with_suffix(ARCHIVE_SUFFIX)
    if destination.exists():
        if not overwrite:
            raise ExportError(
                f"'{destination}' already exists. Pass overwrite=True to replace it."
            )
        if archive and destination.is_dir():
            raise ExportError(
                f"'{destination}' exists and is a directory, so it will not be "
                f"replaced with an archive. Remove it, or export to another path."
            )
        if not archive and not destination.is_dir():
            raise ExportError(f"'{destination}' exists and is not a directory.")
        _refuse_unless_a_bundle(destination)

    policy_sources = _policy_sources(policy_path)
    policy_file = None
    policy_format = None
    if policy_sources:
        # Keep the entry point's own extension: a bundle must not claim to hold
        # an ONNX graph when it holds a TorchScript module.
        entry_point = policy_sources[0]
        policy_file = f"{POLICY_STEM}{entry_point.suffix}"
        policy_format = _POLICY_FORMATS.get(entry_point.suffix.lower())
        _check_names_do_not_collide(policy_file, policy_sources)

    capture = capture_environment(
        env,
        additional_provenance=additional_provenance,
        policy_file=policy_file,
        policy_format=policy_format,
    )

    # The gate. Raises ParityError before anything reaches disk.
    report = check_parity(capture, ticks=parity_ticks, seed=seed)

    # Build the bundle somewhere temporary, then move it into place, so a failure
    # part-way through cannot leave a half-written bundle looking usable.
    with tempfile.TemporaryDirectory(prefix="genesis-forge-export-") as staging:
        staged = save_bundle(
            Path(staging) / "bundle", capture.manifest, golden=report.golden
        )
        if policy_sources:
            shutil.copy2(policy_sources[0], staged / policy_file)
            # Companions keep their own names: a graph refers to them by the
            # filename recorded inside it, so renaming one breaks the reference.
            for companion in policy_sources[1:]:
                shutil.copy2(companion, staged / companion.name)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if archive:
            # Built somewhere temporary and moved into place, so an interrupted
            # export cannot leave a half-written archive looking loadable.
            packed = Path(staging) / f"bundle{ARCHIVE_SUFFIX}"
            _write_archive(staged, packed)
            if destination.exists():
                destination.unlink()
            shutil.move(str(packed), str(destination))
        else:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(staged), str(destination))

    bundle = Bundle(manifest=capture.manifest, path=destination, golden=report.golden)
    if verbose:
        _report(destination, capture, report)
    return bundle


def _refuse_unless_a_bundle(destination: Path) -> None:
    """Only ever replace something that is itself a bundle.

    Overwriting is the default because re-exporting is routine, but that must not
    turn a mistyped destination into lost work. Whatever is already there has to
    look like a bundle before it is replaced.
    """
    if destination.is_dir():
        recognised = (destination / MANIFEST_FILENAME).is_file()
    else:
        try:
            with zipfile.ZipFile(destination) as existing:
                recognised = MANIFEST_FILENAME in existing.namelist()
        except (zipfile.BadZipFile, OSError):
            recognised = False

    if not recognised:
        raise ExportError(
            f"'{destination}' already exists and is not a deployment bundle, so it "
            f"will not be overwritten. Export somewhere else, or move it aside."
        )


def _write_archive(staged: Path, destination: Path) -> None:
    """Zip a staged bundle directory, entries at the archive root."""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(staged.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(staged).as_posix())


def _policy_sources(policy_path: Any) -> list[Path]:
    """Resolve what the caller said the policy is made of.

    One path, or several when the export produced more than one file. Nothing here
    knows or asks what format they are -- the caller just ran the export that wrote
    them, so which files belong together is theirs to state rather than ours to
    infer from naming conventions that differ per format.
    """
    if policy_path is None:
        return []
    if isinstance(policy_path, (str, Path)):
        candidates = [policy_path]
    else:
        candidates = list(policy_path)
        if not candidates:
            raise ExportError(
                "policy_path is an empty list. Pass the policy's file(s), or omit "
                "it to export the pipeline contract without a policy."
            )

    sources = []
    for candidate in candidates:
        source = Path(candidate)
        if not source.is_file():
            raise ExportError(f"No policy file at '{source}'.")
        sources.append(source)
    return sources


def _check_names_do_not_collide(policy_file: str, sources: list[Path]) -> None:
    """Every file must land in the bundle under a distinct name."""
    names = [policy_file] + [companion.name for companion in sources[1:]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ExportError(
            f"More than one policy file would be written as "
            f"{', '.join(repr(name) for name in duplicates)} in the bundle. The "
            f"first path becomes '{policy_file}'; the rest keep their own names, "
            f"so they must all differ."
        )


def _report(destination: Path, capture: Capture, report: ParityReport) -> None:
    manifest = capture.manifest
    layout = manifest.observations
    print(f"Deployment bundle written to {destination}")
    print(f"  {report.summary()}")
    print(
        f"  observations: {layout.total_size} values "
        f"({len(layout.entries)} input(s) to wire up)"
    )
    print(f"  actions: {manifest.num_actions} joint target(s)")
    print(f"  control rate: {manifest.control_hz:.1f} Hz")
    print("  install the runtime on the robot with: pip install genesis-forge-runtime")
