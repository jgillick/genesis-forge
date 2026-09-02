"""Bundles stored as a single file.

The reuse rules matter more than the packing: a robot loads the same archive on
every restart, and must never keep running a bundle that has since been replaced.
"""

import json
import zipfile

import pytest
from test_bundle import make_manifest, write_bundle

from genesis_forge_runtime import (
    ARCHIVE_SUFFIX,
    EXTRACT_MARKER,
    MalformedBundleError,
    load_bundle,
)
from genesis_forge_runtime.archive import extract_dir_for, fingerprint, is_archive


def pack(directory, destination):
    with zipfile.ZipFile(destination, "w") as archive:
        for item in sorted(directory.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(directory).as_posix())
    return destination


def an_archive(tmp_path, name="my_policy"):
    return pack(write_bundle(tmp_path / "src"), tmp_path / f"{name}{ARCHIVE_SUFFIX}")


"""
Recognising one
"""


def test_an_archive_is_recognised_by_content_not_by_name(tmp_path):
    """A bundle someone renamed to .zip still loads."""
    renamed = pack(write_bundle(tmp_path / "src"), tmp_path / "my_policy.zip")

    assert is_archive(renamed)
    assert load_bundle(renamed).manifest.dt == pytest.approx(0.02)


def test_a_directory_is_not_an_archive(tmp_path):
    assert not is_archive(write_bundle(tmp_path / "src"))


def test_a_zip_without_a_manifest_is_refused(tmp_path):
    not_a_bundle = tmp_path / "photos.gfb"
    with zipfile.ZipFile(not_a_bundle, "w") as archive:
        archive.writestr("holiday.jpg", "not a bundle")

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(not_a_bundle)

    assert "manifest.json" in str(error.value)


"""
Where it unpacks, and when it unpacks again
"""


def test_it_unpacks_beside_itself_into_a_dot_directory(tmp_path):
    archive = an_archive(tmp_path)

    bundle = load_bundle(archive)

    assert bundle.path == tmp_path / ".my_policy"
    assert (bundle.path / EXTRACT_MARKER).is_file()
    assert (bundle.path / "manifest.json").is_file()


def test_a_second_load_reuses_the_extraction(tmp_path):
    archive = an_archive(tmp_path)
    first = load_bundle(archive)
    marker = first.path / EXTRACT_MARKER
    stamp = marker.stat().st_mtime_ns
    (first.path / "scratch.txt").write_text("left by the operator")

    second = load_bundle(archive)

    assert second.path == first.path
    assert marker.stat().st_mtime_ns == stamp  # not rewritten
    assert (second.path / "scratch.txt").is_file()  # not re-extracted


def test_replacing_the_archive_unpacks_it_again(tmp_path):
    """The deploy loop: a new bundle must never be masked by the previous one."""
    archive = an_archive(tmp_path)
    extracted = load_bundle(archive).path
    before = json.loads((extracted / "manifest.json").read_text())["control"]["dt"]

    changed = write_bundle(tmp_path / "src2", make_manifest(dt=0.01))
    pack(changed, archive)

    after = load_bundle(archive)

    assert before == pytest.approx(0.02)
    assert after.manifest.dt == pytest.approx(0.01)
    on_disk = json.loads((after.path / "manifest.json").read_text())
    assert on_disk["control"]["dt"] == 0.01


def test_a_stale_extraction_is_cleared_not_merged(tmp_path):
    """Files from the previous bundle must not survive into the new one."""
    archive = an_archive(tmp_path)
    extracted = load_bundle(archive).path
    (extracted / "policy_from_before.onnx").write_text("stale")

    pack(write_bundle(tmp_path / "src2", make_manifest(dt=0.01)), archive)
    reloaded = load_bundle(archive)

    assert not (reloaded.path / "policy_from_before.onnx").exists()


"""
Refusing to trample things
"""


def test_a_foreign_directory_in_the_way_is_not_overwritten(tmp_path):
    archive = an_archive(tmp_path)
    squatter = extract_dir_for(archive)
    squatter.mkdir()
    (squatter / "notes.txt").write_text("someone else's work")

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(archive)

    assert "will not be overwritten" in str(error.value)
    assert (squatter / "notes.txt").read_text() == "someone else's work"


def test_a_file_where_the_extraction_goes_is_reported(tmp_path):
    archive = an_archive(tmp_path)
    extract_dir_for(archive).write_text("in the way")

    with pytest.raises(MalformedBundleError) as error:
        load_bundle(archive)

    assert "not a directory" in str(error.value)


"""
The fingerprint
"""


def test_the_fingerprint_is_read_from_the_central_directory(tmp_path):
    """Costs the same whatever the bundle weighs -- no payload is read."""
    archive = an_archive(tmp_path)
    padded = tmp_path / f"padded{ARCHIVE_SUFFIX}"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(padded, "w") as target:
        for item in source.infolist():
            target.writestr(item.filename, source.read(item.filename))
        target.writestr("extra.bin", b"x" * 100_000)

    assert fingerprint(archive) != fingerprint(padded)
    assert fingerprint(archive) == fingerprint(archive)
