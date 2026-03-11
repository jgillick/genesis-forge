from __future__ import annotations

from typing import Callable

import torch

from genesis_forge.deploy.config import DeploymentConfig
from genesis_forge.managers.action.transformers import (
    position_decode,
    position_within_limits_decode,
)


# ---------------------------------------------------------------------------
# Decoder registry
# ---------------------------------------------------------------------------

_decoder_registry: dict[str, Callable[[torch.Tensor, dict], torch.Tensor]] = {}


def register_action_decoder(
    action_type: str,
    fn: Callable[[torch.Tensor, dict], torch.Tensor] | None = None,
) -> Callable:
    """
    Register a decoder function for a custom action manager type.

    Decoder functions must have the signature::

        def decoder(actions: torch.Tensor, params: dict) -> torch.Tensor:
            ...

    where ``actions`` is a 1-D float tensor of shape
    ``(num_actions_for_this_manager,)`` and ``params`` is the dict returned by
    the action manager's :meth:`~genesis_forge.managers.action.BaseActionManager.get_deploy_config`
    method.  The function should return a 1-D tensor of decoded position (or
    command) values with the same number of elements as ``actions``.

    Can be used as a decorator::

        @register_action_decoder("velocity")
        def decode_velocity(actions: torch.Tensor, params: dict) -> torch.Tensor:
            max_vel = torch.tensor(params["max_velocity"])
            return torch.clamp(actions, -max_vel, max_vel)

    Or called directly::

        register_action_decoder("velocity", decode_velocity)

    Args:
        action_type: The :attr:`~genesis_forge.managers.action.BaseActionManager.deploy_type`
                     string that identifies the action manager class.
        fn: The decoder function to register.  If ``None``, the call returns a
            decorator.

    Returns:
        The registered function (useful when used as a decorator).
    """
    if fn is not None:
        _decoder_registry[action_type] = fn
        return fn

    def decorator(f: Callable) -> Callable:
        _decoder_registry[action_type] = f
        return f

    return decorator


# ---------------------------------------------------------------------------
# Built-in decoders
# ---------------------------------------------------------------------------


@register_action_decoder("position")
def _decode_position(actions: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Decoder for :class:`~genesis_forge.managers.PositionActionManager`.

    Delegates to :func:`~genesis_forge.utils.transforms.position_decode`, the same
    function called by ``process_actions`` during training.
    """
    return position_decode(
        actions,
        params["scale"],
        params["offset"],
        params["clip_low"],
        params["clip_high"],
    )


@register_action_decoder("position_within_limits")
def _decode_position_within_limits(actions: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Decoder for :class:`~genesis_forge.managers.PositionWithinLimitsActionManager`.

    Delegates to :func:`~genesis_forge.utils.transforms.position_within_limits_decode`,
    the same function called by ``process_actions`` during training.
    """
    return position_within_limits_decode(actions, params["scale"], params["offset"])


# ---------------------------------------------------------------------------
# ActionDecoder
# ---------------------------------------------------------------------------


class ActionDecoder:
    """
    Standalone action decoder for deployment on a real robot.

    Mirrors the behavior of the action managers but without any Genesis or
    simulation dependency.  Accepts a raw policy action tensor and decodes it
    into a ``dict`` of ``joint_name -> target_position``.

    Built-in support for ``"position"`` and ``"position_within_limits"`` manager
    types is pre-registered.  For custom action manager types, register a
    decoder with :func:`register_action_decoder` **before** creating an
    ``ActionDecoder`` instance -- or catch the :class:`ValueError` raised at
    construction time.

    Example::

        from genesis_forge.deploy import DeploymentConfig, ActionDecoder

        config = DeploymentConfig.from_json("deploy_config.json")
        decoder = ActionDecoder(config)

        raw_actions = policy(obs)           # shape: (num_actions,)
        joint_commands = decoder.decode(raw_actions)
        # {"FL_hip_joint": 0.12, "FL_thigh_joint": 0.85, ...}

    For custom action manager types::

        from genesis_forge.deploy import register_action_decoder

        @register_action_decoder("velocity")
        def decode_velocity(actions, params):
            max_vel = torch.tensor(params["max_velocity"])
            return torch.clamp(actions, -max_vel, max_vel)

        decoder = ActionDecoder(config)   # now succeeds for "velocity" managers
    """

    def __init__(self, config: DeploymentConfig):
        self._config = config

        # Validate all types are registered at init time for an early, clear error
        missing = [
            am.type for am in config.action_managers if am.type not in _decoder_registry
        ]
        if missing:
            types_str = ", ".join(f'"{t}"' for t in missing)
            example_type = missing[0]
            raise ValueError(
                f"No decoder registered for action manager type(s): {types_str}.\n\n"
                f"Register a decoder before creating ActionDecoder, for example:\n\n"
                f"    from genesis_forge.deploy import register_action_decoder\n\n"
                f'    @register_action_decoder("{example_type}")\n'
                f"    def decode_{example_type}(actions, params):\n"
                f"        ...\n\n"
                f"Registered types: {list(_decoder_registry.keys())}"
            )

    def decode(self, actions: torch.Tensor) -> dict[str, float]:
        """
        Decode a raw policy action tensor into named joint position commands.

        Args:
            actions: 1-D float tensor of shape ``(num_actions,)`` from the
                     policy.  If the policy outputs shape ``(1, num_actions)``,
                     call ``actions.squeeze(0)`` first.

        Returns:
            ``dict`` mapping each joint name to its decoded target position
            value (in radians for revolute joints).

        Raises:
            ValueError: If ``actions`` is not 1-D.
        """
        if actions.ndim != 1:
            raise ValueError(
                f"Expected a 1-D action tensor, got shape {tuple(actions.shape)}. "
                "If your policy outputs shape (1, num_actions), use "
                "actions.squeeze(0) before calling decode()."
            )

        result: dict[str, float] = {}
        for am_config in self._config.action_managers:
            start, end = am_config.action_range
            am_actions = actions[start:end].float()

            decoder_fn = _decoder_registry[am_config.type]
            decoded = decoder_fn(am_actions, am_config.params)

            for name, val in zip(am_config.params["joint_names"], decoded):
                result[name] = val.item()

        return result
