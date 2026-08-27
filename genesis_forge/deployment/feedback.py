"""Working out which observations echo the pipeline's own output.

An observation built from :class:`~genesis_forge.mdp.observations.current_actions`
reads the policy's previous output rather than a sensor. On a robot that value
comes off the action decoder, so the bundle records where to read it and the
robot-side listing says so.

Detection is by inspection: ``current_actions`` is an ``MdpFn`` instance, so its
source is right there on the object. Anything else -- notably a lambda, whose
body cannot be inspected -- is treated as an ordinary sensor input, which is the
safe default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genesis_forge.mdp.observations import current_actions

from .errors import ExportError

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge.managers.action.base import BaseActionManager


def classify_feedback_entries(
    manager: Any,
    names: dict[int, str],
    action_managers: dict[str, BaseActionManager],
) -> dict[str, dict[str, Any]]:
    """Find the observations that echo the pipeline's own output.

    Args:
        manager: The observation manager feeding the policy.
        names: Manager objects to the attribute they are assigned to on the env.
        action_managers: The environment's action managers, by name.

    Returns:
        A mapping of observation name to the manifest fields that describe where
        its value comes from. Entries not present are ordinary sensor inputs.

    Raises:
        ExportError: An observation reads from an action manager that is not
            registered with this environment.
    """
    classified: dict[str, dict[str, Any]] = {}

    for name, config_item in manager.cfg.items():
        function = getattr(config_item, "fn", None)
        if not isinstance(function, current_actions):
            continue

        # current_actions returns raw policy output either way: the whole vector
        # when given no manager, that manager's slice when given one.
        fields: dict[str, Any] = {
            "source": "pipeline_state",
            "pipeline_stage": "raw_actions",
        }

        source_manager = getattr(function, "action_manager", None)
        if source_manager is not None:
            manager_name = names.get(id(source_manager))
            if manager_name is None or manager_name not in action_managers:
                raise ExportError(
                    f"Observation '{name}' reads actions from an action manager that "
                    f"is not registered with this environment, so the deployment "
                    f"runtime could not reproduce it."
                )
            fields["action_manager"] = manager_name

        classified[name] = fields

    return classified
