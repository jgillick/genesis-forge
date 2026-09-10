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
    POLICY_DIRNAME,
    Bundle,
    save_bundle,
)

from .capture import Capture, capture_environment
from .errors import ExportError
from .parity import ParityReport, check_parity


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

    The bundle holds ``manifest.json`` (the readable deployment contract),
    ``golden.npz`` (recorded input/output pairs that double as an on-robot smoke
    test), and the exported policy when one is supplied.

    The parity gate is not optional. Before anything is written, the numpy
    deployment classes are run against the live torch pipeline; if they disagree,
    the export aborts and names the component that diverged.

    Args:
        env: A built :class:`~genesis_forge.ManagedEnvironment`.
        path: Where to write the bundle. Gains a ``.gfb`` suffix if it has none,
            unless ``archive`` is False, in which case it names a directory.
        policy_path: The exported policy file, or files, to export with the bundle.
        additional_provenance: Extra entries recorded under
            ``provenance.additional`` in the manifest, conventionally
            ``checkpoint``, ``framework`` and ``framework_version``. Must be
            JSON serializable.
        archive: Write one ``.gfb`` file rather than a directory. A directory is
            easier to inspect while working; ``load_bundle`` reads either.
        parity_ticks: How many sequential ticks the parity gate compares.
        seed: Seed for the parity inputs, so a failure reproduces.
        overwrite: Replace a bundle already at this path. A path holding anything
            that is not a bundle is refused either way.
        verbose: Print a short summary of what was written.

    Returns:
        The :class:`~genesis_forge_runtime.Bundle` written, with its manifest and
        golden samples already in memory. ``bundle.path`` is where it landed.

    Raises:
        ExportError: The environment cannot be exported as configured, or the
            destination already exists and ``overwrite`` is False.
        ParityError: The deployment pipeline disagreed with training. Nothing is
            written.

    Example::

        from genesis_forge.deployment import export

        env = MyEnv(num_envs=1)
        env.build()

        bundle = export(env, "./go2_walk", policy_path="policy.onnx")
        print(bundle.describe())
    """
    destination = _resolve_destination(path, archive=archive, overwrite=overwrite)
    policy_sources = _policy_sources(policy_path)
    capture = capture_environment(
        env,
        additional_provenance=additional_provenance,
        policy_files=[source.name for source in policy_sources],
    )

    # The gate. Raises ParityError before anything reaches disk.
    report = check_parity(capture, ticks=parity_ticks, seed=seed)

    # Build the bundle somewhere temporary, then move it into place, so a failure
    # part-way through leaves the previous bundle where it was.
    with tempfile.TemporaryDirectory(prefix="genesis-forge-export-") as staging:
        staged = save_bundle(
            Path(staging) / "bundle", capture.manifest, golden=report.golden
        )
        if policy_sources:
            policy_dir = staged / POLICY_DIRNAME
            policy_dir.mkdir()
            for source in policy_sources:
                shutil.copy2(source, policy_dir / source.name)

        if archive:
            packed = Path(staging) / f"bundle{ARCHIVE_SUFFIX}"
            _write_archive(staged, packed)
            staged = packed

        # A partial bundle is refused on load rather than run: a truncated archive
        # has no readable central directory, and a directory missing files is
        # caught against the manifest. So the move need not be atomic.
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
        shutil.move(str(staged), str(destination))

    bundle = Bundle(manifest=capture.manifest, path=destination, golden=report.golden)
    if verbose:
        _report(destination, capture, report)
    return bundle


def _resolve_destination(path: str | Path, *, archive: bool, overwrite: bool) -> Path:
    """Where the bundle will be written, once it is safe to write there.

    Raises:
        ExportError: Something is already at that path that this export must not
            replace.
    """
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

    return destination


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
    """The policy files to copy, in the order they were given.

    Accepts one path or several. None, or an empty list, means no policy.

    Raises:
        ExportError: A file is missing, or two of them share a name -- they are
            copied in under the names they were given, so those must differ.
    """
    if policy_path is None:
        return []
    if isinstance(policy_path, (str, Path)):
        candidates = [policy_path]
    else:
        candidates = list(policy_path)

    sources = []
    for candidate in candidates:
        source = Path(candidate)
        if not source.is_file():
            raise ExportError(f"No policy file at '{source}'.")
        sources.append(source)

    names = [source.name for source in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ExportError(
            f"More than one policy file is named "
            f"{', '.join(repr(name) for name in duplicates)}. They keep the names "
            f"you gave, so those must differ."
        )
    return sources


def _report(destination: Path, capture: Capture, report: ParityReport) -> None:
    manifest = capture.manifest
    layout = manifest.observations
    print(f"Deployment bundle written to {destination}")
    print(f"  {report.summary()}")
    print(
        f"  observations: {layout.total_size} values "
        f"({len(layout.entries)} input(s) to wire up)"
    )
    joints = len(manifest.joint_names)
    if joints == manifest.num_actions:
        print(f"  actions: {joints} joint target(s)")
    else:
        print(
            f"  actions: {manifest.num_actions} policy output(s) -> "
            f"{joints} joint target(s)"
        )
    print(f"  control rate: {manifest.control_hz:.1f} Hz")
    print("  install the runtime on the robot with: pip install genesis-forge-runtime")
