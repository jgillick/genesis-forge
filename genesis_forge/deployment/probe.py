"""Working out which observations echo the pipeline's own output.

An entry written as ``lambda env: self.action_manager.get_actions()`` reads the
policy's previous output rather than a sensor, and on a robot that value comes off
the action decoder. We cannot read intent out of a lambda, so we determine it by
observation: write a distinct sentinel into each action source, run every
observation function once, and see which sentinel comes back.

Anything matching none of them is an ordinary sensor input -- the safe default,
and exactly what would have happened without probing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import ExportError

if TYPE_CHECKING:  # pragma: no cover
    from genesis_forge.managers.action.base import BaseActionManager


def detect_pipeline_state_entries(
    manager: Any,
    layout: dict[str, Any],
    env: Any,
    action_managers: dict[str, BaseActionManager],
) -> None:
    """Work out which observation entries echo the pipeline's own output.

    An entry like ``lambda env: self.action_manager.get_actions()`` reads the
    policy's own previous output rather than a sensor. On a robot that value comes
    off the action decoder, and feeding back the wrong one -- raw policy output
    where the policy trained on decoded targets -- is silent and expensive.

    We cannot read the intent out of a lambda, so we determine it by observation:
    write a distinct sentinel into each action source, run every observation
    function once, and see which sentinel comes back. Anything that matches none of
    them is an ordinary sensor input, which is the safe default.

    An explicit ``pipeline_state`` marker in the config always wins over the probe.
    """
    entries = {entry["name"]: entry for entry in layout["entries"]}

    # Explicit markers first -- they take precedence and skip probing.
    unmarked: dict[str, Any] = {}
    for name, config_item in manager.cfg.items():
        entry = entries.get(name)
        if entry is None:
            continue  # zero-width entries are not deployed
        if entry.get("source") == "pipeline_state":
            _resolve_marked_action_manager(name, entry, action_managers)
        else:
            unmarked[name] = config_item

    if unmarked:
        _probe_pipeline_state_entries(unmarked, entries, env, action_managers)


_RAW_SENTINEL = -13579.25


_TARGET_SENTINEL_BASE = -24680.0


_MANAGER_RAW_SENTINEL_BASE = -35791.0


def _probe_pipeline_state_entries(
    config_items: dict[str, Any],
    entries: dict[str, dict[str, Any]],
    env: Any,
    action_managers: dict[str, BaseActionManager],
) -> None:
    """Identify action-echoing entries by writing sentinels and seeing what returns."""
    import torch

    num_envs = int(getattr(env, "num_envs", 1))
    manager_list = list(action_managers.items())

    saved = _save_action_buffers(env, manager_list)
    try:
        # env.actions is the full raw policy vector -- what current_actions() reads
        # when it is not given a manager.
        total_actions = sum(item.num_actions for _, item in manager_list)
        _set_env_actions(
            env,
            torch.full((num_envs, total_actions), _RAW_SENTINEL, dtype=torch.float32),
        )

        # Each source gets its own sentinel, so a match also says which manager.
        target_sentinels: dict[float, str] = {}
        manager_raw_sentinels: dict[float, str] = {}
        for index, (manager_name, action_manager) in enumerate(manager_list):
            width = action_manager.num_actions

            target = _TARGET_SENTINEL_BASE - index
            target_sentinels[target] = manager_name
            action_manager._actions = torch.full(
                (num_envs, width), target, dtype=torch.float32
            )

            # A manager's raw slice is a third, distinct source -- what
            # current_actions(action_manager=...) reads.
            manager_raw = _MANAGER_RAW_SENTINEL_BASE - index
            manager_raw_sentinels[manager_raw] = manager_name
            action_manager._raw_actions = torch.full(
                (num_envs, width), manager_raw, dtype=torch.float32
            )

        for name, config_item in config_items.items():
            try:
                value = config_item.execute()
            except Exception:  # noqa: BLE001, S112
                # Probing is best-effort and must never be the reason an export
                # fails. Every function already ran successfully during build(), so
                # a failure here is unusual -- and the fallback is the safe one:
                # the entry stays an ordinary sensor input, which is what it would
                # have been without probing at all.
                continue
            entry = entries[name]

            # The whole raw policy vector, read straight off the environment.
            if _all_equal(value, _RAW_SENTINEL):
                entry["source"] = "pipeline_state"
                entry["pipeline_stage"] = "raw_actions"
                continue

            stage_by_sentinel = [
                (target_sentinels, "target_actions"),
                (manager_raw_sentinels, "raw_actions"),
            ]
            for sentinels, stage in stage_by_sentinel:
                match = next(
                    (
                        manager_name
                        for sentinel, manager_name in sentinels.items()
                        if _all_equal(value, sentinel)
                    ),
                    None,
                )
                if match is not None:
                    entry["source"] = "pipeline_state"
                    entry["pipeline_stage"] = stage
                    entry["action_manager"] = match
                    break
    finally:
        _restore_action_buffers(saved)


def _all_equal(value: Any, sentinel: float) -> bool:
    """True when every element of an observation is exactly the sentinel."""
    import torch

    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return False
    return bool(torch.all(value == sentinel))


def _env_actions_attribute(env: Any) -> str:
    """Which attribute actually holds the environment's raw action vector.

    ``GenesisEnv`` exposes ``actions`` as a read-only property backed by
    ``_actions``; a lightweight test double may just use a plain attribute.
    """
    return "_actions" if hasattr(type(env), "actions") else "actions"


def _set_env_actions(env: Any, value: Any) -> None:
    setattr(env, _env_actions_attribute(env), value)


def _save_action_buffers(
    env: Any, manager_list: list[tuple[str, BaseActionManager]]
) -> list[tuple[Any, str, Any]]:
    """Snapshot every buffer the probe overwrites, so it can be put back exactly."""
    attribute = _env_actions_attribute(env)
    saved: list[tuple[Any, str, Any]] = [
        (env, attribute, getattr(env, attribute, None))
    ]
    for _name, action_manager in manager_list:
        saved.append((action_manager, "_actions", action_manager._actions))
        saved.append((action_manager, "_raw_actions", action_manager._raw_actions))
    return saved


def _restore_action_buffers(saved: list[tuple[Any, str, Any]]) -> None:
    """Put the original buffer objects back. Exporting must not disturb the env."""
    for owner, attribute, value in saved:
        setattr(owner, attribute, value)


def _resolve_marked_action_manager(
    name: str, entry: dict[str, Any], action_managers: dict[str, BaseActionManager]
) -> None:
    """Attach the source manager to an explicitly-marked target-actions entry."""
    if entry.get("pipeline_stage") != "target_actions" or entry.get("action_manager"):
        return

    if len(action_managers) == 1:
        entry["action_manager"] = next(iter(action_managers))
        return

    available = ", ".join(sorted(action_managers))
    raise ExportError(
        f"Observation '{name}' is marked as echoing target actions, but this "
        f"environment has several action managers ({available}), so which one it "
        f"reads from is ambiguous. Use "
        f"genesis_forge.mdp.observations.current_actions(action_manager=...) instead "
        f"of the marker, so the source is unambiguous."
    )
