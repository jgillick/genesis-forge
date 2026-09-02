"""Read a built environment's deployment contract out of its managers.

Everything here asks managers to describe themselves -- ``get_deployment_config``,
``get_deployment_layout``, ``get_deployment_values`` -- rather than reaching into
their internals. That is what lets a custom action manager participate without the
exporter knowing anything about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from genesis_forge_runtime import (
    MIN_SUPPORTED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ActionManagerSpec,
    ActuatorSpec,
    Manifest,
    ObservationLayout,
    PolicySpec,
)

from .errors import ExportError
from .provenance import build_provenance

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge.managers.action.base import BaseActionManager


#: The bundle schema this exporter knows how to build.
#:
#: Deliberately stated here rather than taken from whatever ``genesis_forge_runtime``
#: happens to be installed. Those are two distributions on their own release
#: cadences, so a newer runtime can arrive beside an older exporter -- and inheriting
#: its number would stamp bundles as a format this code has never produced, with the
#: old shape inside. Writing an older number is honest, and the runtime reads it
#: happily: that is what MIN_SUPPORTED_SCHEMA_VERSION is for.
#:
#: Bump this together with SCHEMA_VERSION in genesis_forge_runtime, and only when the
#: manifest changes in a way an older reader would get wrong. Adding an optional
#: field is not such a change -- readers ignore keys they do not know.
WRITES_SCHEMA_VERSION = 1


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
    additional_provenance: dict[str, Any] | None = None,
    policy_file: str | None = None,
    policy_format: str | None = None,
) -> Capture:
    """Read the full deployment contract from a built environment.

    Args:
        env: A built :class:`~genesis_forge.ManagedEnvironment`.
        additional_provenance: Extra provenance entries supplied by the caller.
        policy_file: Filename of an exported policy shipped inside the bundle.
        policy_format: What kind of policy file it is (``"onnx"``, ``"torchscript"``).

    Returns:
        The :class:`Capture` the exporter and parity harness both work from.

    Raises:
        ExportError: The environment is not built, or is configured in a way the
            bundle schema cannot express.
    """
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
    observation_manager, layout_data, entry_names = _capture_observations(managers)
    actuators = _capture_actuators(managers, names)

    dt = float(getattr(env, "dt", 0.0))
    if dt <= 0:
        raise ExportError(
            f"The environment reports dt={dt!r}, so the control rate the policy was "
            f"trained at is unknown. Set a positive dt on the environment."
        )

    _check_runtime_can_read_what_we_write()

    manifest = Manifest(
        schema_version=WRITES_SCHEMA_VERSION,
        dt=dt,
        observations=ObservationLayout.from_dict(layout_data),
        actions=tuple(
            ActionManagerSpec.from_dict(spec, where="actions") for spec in action_specs
        ),
        actuators=tuple(
            ActuatorSpec.from_dict(spec, where="actuators") for spec in actuators
        ),
        policy=PolicySpec(file=policy_file, format=policy_format)
        if policy_file
        else None,
        provenance=build_provenance(additional=additional_provenance),
    )

    return Capture(
        manifest=manifest,
        observation_manager=observation_manager,
        observation_entry_names=entry_names,
        action_managers=action_managers,
        num_envs=num_envs,
    )


def _check_runtime_can_read_what_we_write() -> None:
    """Refuse to build a bundle the installed runtime could not load.

    The two packages version independently, so the pair on this machine may not
    agree. Better to say so here than to hand someone a bundle their robot rejects.
    """
    if WRITES_SCHEMA_VERSION > SCHEMA_VERSION:
        raise ExportError(
            f"This version of Genesis Forge writes bundle schema "
            f"{WRITES_SCHEMA_VERSION}, but the installed genesis-forge-runtime only "
            f"understands up to {SCHEMA_VERSION}. Upgrade genesis-forge-runtime."
        )
    if WRITES_SCHEMA_VERSION < MIN_SUPPORTED_SCHEMA_VERSION:
        raise ExportError(
            f"This version of Genesis Forge writes bundle schema "
            f"{WRITES_SCHEMA_VERSION}, which the installed genesis-forge-runtime no "
            f"longer reads (it requires at least {MIN_SUPPORTED_SCHEMA_VERSION}). "
            f"Upgrade genesis-forge."
        )


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
                "delay_step": manager.delay_step,
            }
        )
        resolved[name] = manager

    return specs, resolved


def _capture_observations(
    managers: dict[str, Any],
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

    if not getattr(chosen, "enabled", True):
        raise ExportError(
            f"Observation manager '{chosen.name}' is disabled, so training feeds "
            f"the policy zeros rather than the layout it describes. Exporting it "
            f"would produce a bundle the robot fills in for real. Enable it and "
            f"re-export."
        )

    try:
        layout = chosen.get_deployment_layout()
    except ValueError as error:
        raise ExportError(
            f"Observation manager '{chosen.name}' cannot be described in a bundle. "
            f"{error}"
        ) from error
    return chosen, layout, list(chosen.cfg.keys())


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


def _attribute_names(env: Any) -> dict[int, str]:
    """Map manager objects to the attribute they were assigned to on the environment.

    Gives the manifest names a reader recognizes (``action_manager`` rather than
    ``action_manager_0``), matching how they were written in ``config()``.
    """
    return {id(value): name for name, value in vars(env).items()}
