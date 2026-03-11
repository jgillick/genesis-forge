from __future__ import annotations

from typing import TYPE_CHECKING

from genesis_forge.deploy.config import (
    ActuatorConfig,
    ActionManagerConfig,
    DeploymentConfig,
    ObservationItemConfig,
)

if TYPE_CHECKING:
    from genesis_forge.managed_env import ManagedEnvironment


def export(env, path: str | None = None) -> DeploymentConfig:
    """
    Export the deployment configuration from a built ``ManagedEnvironment``.

    Captures the observation pipeline (ordering, scales, history length) and
    the action decoding pipeline (per-manager type and transform parameters) so
    they can be reproduced on a real robot without any Genesis or simulation
    dependency.

    Must be called **after** ``env.build()``.

    Args:
        env: A built :class:`~genesis_forge.ManagedEnvironment` instance, or a
             :class:`~genesis_forge.wrappers.Wrapper` wrapping one.
        path: Optional file path to write the JSON config.  Parent directories
              are created automatically.

    Returns:
        A :class:`~genesis_forge.deploy.DeploymentConfig` instance.

    Example::

        from genesis_forge.deploy import export

        env = Go2SimpleEnv(num_envs=1)
        env = RslRlWrapper(env)
        env.build()

        config = export(env, path="./deploy/go2_config.json")

    Raises:
        RuntimeError: If the environment has not been built yet, or if an action
                      manager does not implement ``get_deploy_config()``.
        ValueError: If no ``"policy"`` observation manager is found.
    """
    # Unwrap any Wrapper layers to reach the ManagedEnvironment
    actual_env = env
    while hasattr(actual_env, "env"):
        actual_env = actual_env.env

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if not actual_env.managers.get("action"):
        raise RuntimeError(
            "No action managers found. Make sure the environment has been built "
            "by calling env.build() before exporting."
        )

    # ------------------------------------------------------------------
    # Observations -- find the "policy" observation manager
    # ------------------------------------------------------------------
    obs_managers = actual_env.managers.get("observation", [])
    policy_obs_manager = None
    for om in obs_managers:
        if om.name == "policy":
            policy_obs_manager = om
            break

    if policy_obs_manager is None:
        raise ValueError(
            "No 'policy' observation manager found. "
            "Make sure an ObservationManager with name='policy' is configured."
        )

    if not policy_obs_manager._observation_dims:
        raise RuntimeError(
            "The 'policy' observation manager has no cached observation dims. "
            "Make sure the environment has been built by calling env.build()."
        )

    observations = [
        ObservationItemConfig(
            name=name,
            dim=dim,
            scale=policy_obs_manager.cfg[name].scale,
        )
        for name, dim in policy_obs_manager._observation_dims
    ]

    # ------------------------------------------------------------------
    # Action managers
    # ------------------------------------------------------------------
    action_managers_config: list[ActionManagerConfig] = []
    for i, am in enumerate(actual_env.managers["action"]):
        if not hasattr(am, "export"):
            raise RuntimeError(
                f"Action manager '{type(am).__name__}' does not implement "
                f"export(). To support deployment, add:\n\n"
                f'    deploy_type = "my_type"\n\n'
                f"    def export(self) -> dict:\n"
                f"        config = super().export()\n"
                f"        config.update({{...}})\n"
                f"        return config\n"
            )

        start, end = actual_env._action_ranges[i]
        action_managers_config.append(
            ActionManagerConfig(
                type=am.deploy_type,
                action_range=(start, end),
                params=am.export(),
            )
        )

    # ------------------------------------------------------------------
    # Actuators
    # ------------------------------------------------------------------
    actuators_config: list[ActuatorConfig] = []
    for actuator_manager in actual_env.managers.get("actuator", []):
        joint_names = list(actuator_manager.dofs.keys())

        def _get_base_values(name: str) -> list[float]:
            """Extract the base buffer values (before any noise) as a Python list."""
            entry = actuator_manager._values.get(name)
            if entry is None:
                return [0.0] * len(joint_names)
            buf = entry["buffer"]
            # buffer may be (num_envs, n_dofs) or (n_dofs,); take first row if batched
            if buf.ndim == 2:
                buf = buf[0]
            return buf.tolist()

        lower, upper = actuator_manager.get_dofs_limits()
        # Limits may be (num_envs, n_dofs) when batch_dofs_info is enabled
        if lower.ndim == 2:
            lower = lower[0]
            upper = upper[0]

        actuators_config.append(
            ActuatorConfig(
                joint_names=joint_names,
                kp=_get_base_values("kp"),
                kv=_get_base_values("kv"),
                default_pos=_get_base_values("default_pos"),
                position_limits_low=lower.tolist(),
                position_limits_high=upper.tolist(),
            )
        )

    # ------------------------------------------------------------------
    # Assemble config
    # ------------------------------------------------------------------
    config = DeploymentConfig(
        dt=actual_env.dt,
        num_observations=actual_env.num_observations,
        num_actions=actual_env.num_actions,
        observation_history_len=policy_obs_manager._history_len,
        observations=observations,
        action_managers=action_managers_config,
        actuators=actuators_config,
    )

    if path is not None:
        config.to_json(path)

    return config
