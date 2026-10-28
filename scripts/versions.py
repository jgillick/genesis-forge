"""Release rules for genesis-forge and genesis-forge-runtime, called from the Makefile.

The two packages always share a major.minor version, and genesis-forge pins the
runtime to that minor (``>=X.Y.Z,<X.(Y+1)``), so either can ship a patch on its
own. Both released together are tagged ``vX.Y.Z``; one released alone is tagged
``<package>-vX.Y.Z``.

    versions.py check                   the packages share a minor, and the pin matches it
    versions.py pin                     the runtime requirement genesis-forge should declare
    versions.py bump <target> <V>       V is a valid next release of <target>
    versions.py find-tag <target>       the existing tag that releases <target>
    versions.py check-tag <target>      HEAD is tagged as a release of <target>

<target> is ``all``, ``genesis-forge``, or ``genesis-forge-runtime``.
"""

import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, where pytest brings in the backport
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
FORGE = "genesis-forge"
RUNTIME = "genesis-forge-runtime"
TARGETS = ("all", FORGE, RUNTIME)


class ReleaseError(Exception):
    """A release rule is broken. The message says which, and how to fix it."""


def read_pyproject(package: str) -> dict:
    """
    Read a package's ``[project]`` table.

    Args:
        package: The package name, which is also its directory under ``packages/``.

    Returns:
        The ``[project]`` table, as a dict.
    """
    with (ROOT / "packages" / package / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def current_version(package: str) -> Version:
    """
    The version a package's pyproject declares.

    Args:
        package: The package name.
    """
    return Version(read_pyproject(package)["version"])


def git(*args: str) -> str:
    """
    Run a git command in the repo root.

    Args:
        *args: The git arguments, e.g. ``"tag", "--points-at", "HEAD"``.

    Returns:
        The command's stdout.

    Raises:
        subprocess.CalledProcessError: git exited with an error.
    """
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def release_tag(target: str, version: Version | str) -> str:
    """
    The git tag that marks a release.

    Args:
        target: ``"all"`` for both packages together, or a package name.
        version: The version being released.

    Returns:
        ``vX.Y.Z`` for both packages, or ``<package>-vX.Y.Z`` for one.
    """
    return f"v{version}" if target == "all" else f"{target}-v{version}"


"""
Commands
"""


def check() -> None:
    """
    Check that the repo follows the version rules, and print the versions.

    Raises:
        ReleaseError: The packages don't share a major.minor version, or
            genesis-forge's runtime pin isn't ``>=X.Y.N,<X.(Y+1)`` with a lower
            bound the workspace runtime satisfies.
    """
    forge = current_version(FORGE)
    runtime = current_version(RUNTIME)
    major, minor = runtime.major, runtime.minor
    if (forge.major, forge.minor) != (major, minor):
        raise ReleaseError(
            f"{FORGE} is {forge} and {RUNTIME} is {runtime}, but they must share a "
            f"major.minor version. Release a new minor of both with 'make version'."
        )

    dependencies = read_pyproject(FORGE)["dependencies"]
    pin = next((d for d in dependencies if d.startswith(RUNTIME)), "(none)")
    # uv doesn't hold a workspace package to the pin, so the lower bound is checked here.
    match = re.fullmatch(rf"{RUNTIME}>=([^,]+),<{major}\.{minor + 1}", pin)
    lower = Version(match[1]) if match else None
    if lower is None or (lower.major, lower.minor) != (major, minor) or lower > runtime:
        raise ReleaseError(
            f"{FORGE} requires {pin}, but it should be "
            f"'{RUNTIME}>={major}.{minor}.N,<{major}.{minor + 1}' with N <= "
            f"{runtime.micro}. Run 'make version' or 'make version-forge' to set it."
        )
    print(f"{FORGE} {forge}, {RUNTIME} {runtime}, pinned {pin}")


def pin() -> None:
    """
    Print the runtime requirement genesis-forge should declare, e.g.
    ``genesis-forge-runtime>=1.0.1,<1.1``: the current runtime version, up to
    the next minor. The Makefile passes it to ``uv add``.
    """
    runtime = current_version(RUNTIME)
    print(f"{RUNTIME}>={runtime},<{runtime.major}.{runtime.minor + 1}")


def bump(target: str, value: str) -> None:
    """
    Check that ``value`` can be released for ``target``, before the Makefile
    changes anything.

    Args:
        target: ``"all"`` to release both packages, or one package's name.
        value: The new version, from the Makefile's ``V``.

    Raises:
        ReleaseError: No version was given, the working tree isn't clean, the
            release tag already exists, the version isn't higher than the
            current one, or a single-package release would leave the two
            packages on different major.minor versions.
        InvalidVersion: ``value`` is not a valid version.
    """
    if not value:
        raise ReleaseError("Give the new version, e.g. 'make version V=1.2.3'.")
    new = Version(value)
    if git("status", "--porcelain"):
        raise ReleaseError("Working tree is not clean; commit or stash first.")
    if git("tag", "--list", release_tag(target, new)):
        raise ReleaseError(f"Tag {release_tag(target, new)} already exists.")

    for package in (FORGE, RUNTIME) if target == "all" else (target,):
        if new <= current_version(package):
            raise ReleaseError(f"{package} is already {current_version(package)}.")

    other = {FORGE: RUNTIME, RUNTIME: FORGE}.get(target)
    if other:
        other_version = current_version(other)
        if (new.major, new.minor) != (other_version.major, other_version.minor):
            raise ReleaseError(
                f"{target} can only ship a patch on its own, and {new} is a new "
                f"major or minor. Release both with 'make version V={new}'."
            )


def accepted_tags(target: str) -> set[str]:
    """
    The tags that would release ``target`` at its current version.

    A package is released by its own ``<package>-vX.Y.Z`` tag, or by a
    ``vX.Y.Z`` tag when both packages are at X.Y.Z. Both packages together
    need ``vX.Y.Z``.

    Args:
        target: ``"all"`` for both packages, or one package's name.

    Raises:
        ReleaseError: ``target`` is ``"all"`` and the packages are at
            different versions.
    """
    forge = current_version(FORGE)
    runtime = current_version(RUNTIME)
    # A together tag releases each package; a package's own tag releases only it.
    accepted = {release_tag("all", forge)} if forge == runtime else set()
    if target != "all":
        accepted.add(release_tag(target, current_version(target)))
    if not accepted:
        raise ReleaseError(
            f"Releasing both needs them at one version, but {FORGE} is {forge} "
            f"and {RUNTIME} is {runtime}."
        )
    return accepted


def find_tag(target: str) -> None:
    """
    Print the existing tag that releases ``target`` at its current version, so
    CI can deploy from it when HEAD has moved past the release commit. A
    package's own tag wins over a together tag.

    Args:
        target: ``"all"`` for both packages, or one package's name.

    Raises:
        ReleaseError: No accepted tag exists yet.
    """
    accepted = accepted_tags(target)
    existing = accepted & set(git("tag", "--list").split())
    if not existing:
        raise ReleaseError(
            f"No {' or '.join(sorted(accepted))} tag exists. Run 'make version' "
            f"and push the tag first."
        )
    if target != "all" and release_tag(target, current_version(target)) in existing:
        print(release_tag(target, current_version(target)))
    else:
        print(release_tag("all", current_version(FORGE)))


def check_tag(target: str) -> None:
    """
    Check that HEAD is tagged as a release of ``target`` at its current version,
    so only tagged commits get published.

    Args:
        target: ``"all"`` for both packages, or one package's name.

    Raises:
        ReleaseError: HEAD doesn't carry an accepted tag, or ``target`` is
            ``"all"`` and the packages are at different versions.
    """
    accepted = accepted_tags(target)
    tags = set(git("tag", "--points-at", "HEAD").split())
    if not accepted & tags:
        raise ReleaseError(
            f"HEAD must be tagged {' or '.join(sorted(accepted))} to release "
            f"(it has: {', '.join(sorted(tags)) or 'no tags'}). Run 'make version' first."
        )


COMMANDS = {
    "check": check,
    "pin": pin,
    "bump": bump,
    "find-tag": find_tag,
    "check-tag": check_tag,
}


def main(argv: list[str]) -> None:
    """
    Run the command named in ``argv``.

    Args:
        argv: The command name and its arguments, without the script name.

    Raises:
        ReleaseError: The command or target is unknown; the message is the usage.
    """
    command, *args = argv or [""]
    if command not in COMMANDS or (args and args[0] not in TARGETS):
        raise ReleaseError(__doc__)
    COMMANDS[command](*args)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (ReleaseError, InvalidVersion) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
