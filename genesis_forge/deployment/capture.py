"""Read a built environment's deployment contract out of its managers.

Everything here asks managers to describe themselves -- ``get_deployment_config``,
``get_deployment_layout``, ``get_deployment_values`` -- rather than reaching into
their internals. That is what lets a custom action manager participate without the
exporter knowing anything about it.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge_deploy import Manifest

    from genesis_forge.managers.action.base import BaseActionManager


class ExportError(Exception):
    """The environment cannot be exported as it is currently configured."""


@dataclass
class Capture:
    """Everything the exporter read out of a built environment."""

    manifest: Manifest
    observation_manager: Any
    observation_entry_names: list[str]
    action_managers: dict[str, BaseActionManager] = field(default_factory=dict)
    num_envs: int = 1


def capture_environment(
    env: Any,
    *,
    checkpoint: str | None = None,
    policy_file: str | None = None,
    policy: Any = None,
) -> Capture:
    """Read the full deployment contract from a built environment.

    Args:
        env: A built :class:`~genesis_forge.ManagedEnvironment`.
        checkpoint: Optional path to the trained checkpoint, recorded as provenance.
        policy_file: Filename of an exported policy shipped inside the bundle.
        policy: The framework's policy object, used only to record provenance.

    Returns:
        The :class:`Capture` the exporter and parity harness both work from.

    Raises:
        ExportError: The environment is not built, or is configured in a way the
            bundle schema cannot express.
    """
    from genesis_forge_deploy import (
        ActionManagerSpec,
        ActuatorSpec,
        Manifest,
        ObservationLayout,
        PolicySpec,
    )

    managers = getattr(env, "managers", None)
    if not isinstance(managers, dict):
        raise ExportError(
            "Deployment export needs a ManagedEnvironment (one that registers "
            f"managers); got {type(env).__name__}. A plain GenesisEnv subclass has "
            "no managers to describe, so there is nothing to capture."
        )

    names = _attribute_names(env)
    num_envs = int(getattr(env, "num_envs", 1))

    action_specs, action_managers = _capture_actions(env, managers, names)
    observation_manager, layout_data, entry_names = _capture_observations(
        managers, names, action_managers
    )
    actuators = _capture_actuators(managers, names)

    dt = float(getattr(env, "dt", 0.0))
    if dt <= 0:
        raise ExportError(
            f"The environment reports dt={dt!r}, so the control rate the policy was "
            f"trained at is unknown. Set a positive dt on the environment."
        )

    manifest = Manifest(
        dt=dt,
        observations=ObservationLayout.from_dict(layout_data),
        actions=tuple(
            ActionManagerSpec.from_dict(spec, where="actions") for spec in action_specs
        ),
        actuators=tuple(
            ActuatorSpec.from_dict(spec, where="actuators") for spec in actuators
        ),
        policy=PolicySpec(file=policy_file) if policy_file else None,
        provenance=_provenance(checkpoint=checkpoint, policy=policy),
    )

    return Capture(
        manifest=manifest,
        observation_manager=observation_manager,
        observation_entry_names=entry_names,
        action_managers=action_managers,
        num_envs=num_envs,
    )


"""
Per-manager capture
"""


def _capture_actions(
    env: Any, managers: dict[str, Any], names: dict[int, str]
) -> tuple[list[dict[str, Any]], dict[str, BaseActionManager]]:
    """Ask each action manager to describe its own decode."""
    action_managers = managers.get("action", [])
    if not action_managers:
        raise ExportError(
            "The environment has no action managers, so there is no policy output to "
            "decode. Deployment export needs at least one."
        )

    ranges = getattr(env, "action_ranges", None)
    if not ranges or len(ranges) != len(action_managers):
        raise ExportError(
            "The environment's action slices are unavailable, which means it has not "
            "been built. Call env.build() before exporting."
        )

    specs: list[dict[str, Any]] = []
    resolved: dict[str, BaseActionManager] = {}

    for index, (manager, (start, end)) in enumerate(
        zip(action_managers, ranges, strict=True)
    ):
        name = names.get(id(manager)) or f"action_manager_{index}"
        try:
            contract = manager.get_deployment_config()
        except NotImplementedError as error:
            raise ExportError(
                f"Action manager '{name}' ({type(manager).__name__}) does not support "
                f"deployment export. {error}"
            ) from error

        specs.append(
            {
                "name": name,
                "deploy_type": contract.deploy_type,
                "slice": [int(start), int(end)],
                "joint_names": list(manager.dofs.keys()),
                "config": contract.config,
                "decoder_import_path": contract.decoder_import_path,
                "delay_step": int(getattr(manager, "_delay_step", 0) or 0),
            }
        )
        resolved[name] = manager

    return specs, resolved


def _capture_observations(
    managers: dict[str, Any],
    names: dict[int, str],
    action_managers: dict[str, BaseActionManager],
) -> tuple[Any, dict[str, Any], list[str]]:
    """Capture the layout of the pipeline that actually feeds the policy."""
    observation_managers = managers.get("observation", [])
    if not observation_managers:
        raise ExportError(
            "The environment has no observation managers, so the policy's input "
            "layout is unknown."
        )

    policy_managers = [
        manager for manager in observation_managers if manager.name == "policy"
    ]
    if policy_managers:
        chosen = policy_managers[0]
    elif len(observation_managers) == 1:
        chosen = observation_managers[0]
    else:
        available = ", ".join(manager.name for manager in observation_managers)
        raise ExportError(
            f"This environment has several observation managers ({available}) and none "
            f"is named 'policy', so which one feeds the deployed policy is ambiguous. "
            f"Name the policy's manager 'policy' and re-export. (Managers that read "
            f"privileged simulator state, such as a critic's, cannot be deployed "
            f"anyway -- a robot has no way to supply those values.)"
        )

    layout = chosen.get_deployment_layout()
    _detect_pipeline_state_entries(chosen, layout, names, action_managers)
    return chosen, layout, list(chosen.cfg.keys())


def _detect_pipeline_state_entries(
    manager: Any,
    layout: dict[str, Any],
    names: dict[int, str],
    action_managers: dict[str, BaseActionManager],
) -> None:
    """Mark observations that echo the policy's own output, so the robot auto-fills them.

    ``current_actions`` is the built-in way to feed previous actions back in, and it
    returns *processed* actions when given an action manager but the *raw* policy
    output otherwise. Getting that backwards on hardware is silent and expensive, so
    detect it here rather than relying on the user to annotate it. An explicit
    ``pipeline_state`` marker in the config always wins.
    """
    from genesis_forge.mdp.observations import current_actions

    entries = {entry["name"]: entry for entry in layout["entries"]}

    for name, config_item in manager.cfg.items():
        entry = entries.get(name)
        if entry is None:
            continue  # zero-width entries are not deployed

        if entry.get("source") == "pipeline_state":
            # Explicitly marked in the config. A processed-actions entry still needs
            # to say which manager it came from; with a single action manager that is
            # unambiguous, so fill it in rather than making the user repeat it.
            _resolve_marked_action_manager(name, entry, action_managers)
            continue

        function = getattr(config_item, "fn", None)
        if not isinstance(function, current_actions):
            continue

        source_manager = getattr(function, "action_manager", None)
        entry["source"] = "pipeline_state"
        if source_manager is None:
            entry["pipeline_stage"] = "raw_actions"
            continue

        manager_name = names.get(id(source_manager))
        if manager_name is None or manager_name not in action_managers:
            raise ExportError(
                f"Observation '{name}' reads processed actions from an action manager "
                f"that is not registered with this environment, so the deployment "
                f"runtime could not reproduce it."
            )
        entry["pipeline_stage"] = "processed_actions"
        entry["action_manager"] = manager_name


def _resolve_marked_action_manager(
    name: str, entry: dict[str, Any], action_managers: dict[str, BaseActionManager]
) -> None:
    """Attach the source manager to an explicitly-marked processed-actions entry."""
    if entry.get("pipeline_stage") != "processed_actions" or entry.get("action_manager"):
        return

    if len(action_managers) == 1:
        entry["action_manager"] = next(iter(action_managers))
        return

    available = ", ".join(sorted(action_managers))
    raise ExportError(
        f"Observation '{name}' is marked as echoing processed actions, but this "
        f"environment has several action managers ({available}), so which one it "
        f"reads from is ambiguous. Use "
        f"genesis_forge.mdp.observations.current_actions(action_manager=...) instead "
        f"of the marker, so the source is unambiguous."
    )


def _capture_actuators(
    managers: dict[str, Any], names: dict[int, str]
) -> list[dict[str, Any]]:
    """Record nominal gains and defaults, so the robot's controllers can match."""
    specs: list[dict[str, Any]] = []
    for index, manager in enumerate(managers.get("actuator", [])):
        if not hasattr(manager, "get_deployment_values"):
            continue
        values = manager.get_deployment_values()
        values["name"] = names.get(id(manager)) or f"actuator_manager_{index}"
        specs.append(values)
    return specs


"""
Internal helpers
"""


def _attribute_names(env: Any) -> dict[int, str]:
    """Map manager objects to the attribute they were assigned to on the environment.

    Gives the manifest names a reader recognizes (``action_manager`` rather than
    ``action_manager_0``), matching how they were written in ``config()``.
    """
    return {id(value): name for name, value in vars(env).items()}


def _provenance(*, checkpoint: str | None, policy: Any) -> Any:
    from genesis_forge_deploy import Provenance

    return Provenance(
        exported_at=_datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        genesis_forge_version=_package_version("genesis-forge"),
        torch_version=_torch_version(),
        policy_framework=_policy_framework(policy),
        policy_framework_version=_policy_framework_version(policy),
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


def _policy_framework(policy: Any) -> str | None:
    if policy is None:
        return None
    module = type(policy).__module__ or ""
    return module.split(".")[0] or None


def _policy_framework_version(policy: Any) -> str | None:
    framework = _policy_framework(policy)
    if not framework:
        return None
    return _package_version(framework.replace("_", "-")) or _package_version(framework)


__all__ = ["Capture", "ExportError", "capture_environment"]
