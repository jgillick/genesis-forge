"""Recording where a bundle came from.

The first question anyone asks when a robot misbehaves is which export produced
the bundle it is running, so every bundle carries the versions and the checkpoint
it was built from.
"""

from __future__ import annotations

import datetime as _datetime
from typing import Any

from genesis_forge_deploy import Provenance


def build_provenance(*, checkpoint: str | None, reference_policy: Any) -> Any:
    return Provenance(
        exported_at=_datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        genesis_forge_version=_package_version("genesis-forge"),
        torch_version=_torch_version(),
        policy_framework=_policy_framework(reference_policy),
        policy_framework_version=_policy_framework_version(reference_policy),
        checkpoint=str(checkpoint) if checkpoint else None,
    )


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except (ImportError, PackageNotFoundError):  # pragma: no cover
        return None


def _torch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:  # pragma: no cover
        return None


def _policy_framework(reference_policy: Any) -> str | None:
    if reference_policy is None:
        return None
    module = type(reference_policy).__module__ or ""
    return module.split(".")[0] or None


def _policy_framework_version(reference_policy: Any) -> str | None:
    framework = _policy_framework(reference_policy)
    if not framework:
        return None
    return (
        _package_version(_distribution_for(framework))
        or _package_version(framework.replace("_", "-"))
        or _package_version(framework)
    )


def _distribution_for(package: str) -> str:
    """Find the distribution that installed a top-level package.

    The two names often differ -- rsl_rl ships as rsl-rl-lib -- so guessing from
    the import name alone loses the version, which is exactly what you want when
    working out why a robot is misbehaving.
    """
    try:
        from importlib.metadata import packages_distributions
    except ImportError:  # pragma: no cover - Python < 3.10
        return package
    distributions = packages_distributions().get(package) or []
    return distributions[0] if distributions else package
