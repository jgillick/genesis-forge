from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class ObservationItemConfig:
    """Configuration for a single observation slot."""

    name: str
    """The observation key name, matching the training config."""

    dim: int
    """Number of values this observation contributes to the flat observation vector."""

    scale: float | None
    """Scale applied to raw values before concatenation. None or 1.0 means no scaling."""


@dataclass
class ActionManagerConfig:
    """Configuration for a single action manager."""

    type: str
    """
    The deploy_type string identifying the action manager class.
    Built-in values: "position", "position_within_limits".
    Custom managers define their own unique string.
    """

    action_range: tuple[int, int]
    """(start, end) slice into the full flattened action vector for this manager."""

    params: dict
    """
    Opaque, type-specific configuration returned by the manager's
    ``get_deploy_config()`` method. The matching decoder function receives
    this dict verbatim.
    """


@dataclass
class ActuatorConfig:
    """Configuration for a single actuator manager."""

    joint_names: list[str]
    """Ordered list of joint names managed by this actuator manager."""

    kp: list[float]
    """Proportional gains, one per joint (base value, before any domain randomization noise)."""

    kv: list[float]
    """Derivative gains, one per joint (base value, before any domain randomization noise)."""

    default_pos: list[float]
    """Default joint positions used at reset, one per joint."""

    position_limits_low: list[float]
    """Lower position limits from the URDF, one per joint (radians)."""

    position_limits_high: list[float]
    """Upper position limits from the URDF, one per joint (radians)."""


@dataclass
class DeploymentConfig:
    """
    Portable deployment configuration exported from a built ``ManagedEnvironment``.

    Contains everything needed to reconstruct the observation assembly and action
    decoding pipelines on a real robot without any Genesis or simulation dependency.

    Use :func:`genesis_forge.deploy.export` to create an instance from a trained
    environment, and :meth:`to_json` / :meth:`from_json` to persist it.

    Example::

        # Export after training
        from genesis_forge.deploy import export
        config = export(env, path="./deploy/go2_config.json")

        # Load on the robot (no Genesis required)
        from genesis_forge.deploy import DeploymentConfig
        config = DeploymentConfig.from_json("./deploy/go2_config.json")
    """

    dt: float
    """Simulation / control timestep in seconds."""

    num_observations: int
    """Total size of the flat observation vector (including history)."""

    num_actions: int
    """Total number of actions across all action managers."""

    observation_history_len: int
    """Number of consecutive observation frames concatenated into one obs vector."""

    observations: list[ObservationItemConfig]
    """Ordered list of observation slots, matching the training ObservationManager config."""

    action_managers: list[ActionManagerConfig]
    """One entry per action manager, in the same order as training."""

    actuators: list[ActuatorConfig]
    """One entry per actuator manager, in the same order as training."""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, path: str | None = None) -> str:
        """
        Serialize to a JSON string. Optionally write to a file.

        Args:
            path: If provided, the JSON is written to this file path.
                  Parent directories are created automatically.

        Returns:
            The JSON string.
        """
        data = asdict(self)
        text = json.dumps(data, indent=2)
        if path is not None:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_json(cls, path_or_str: str) -> DeploymentConfig:
        """
        Deserialize from a JSON file path or a raw JSON string.

        Args:
            path_or_str: A file path (e.g. ``"./deploy/config.json"``) or a
                         raw JSON string.

        Returns:
            A :class:`DeploymentConfig` instance.
        """
        if os.path.exists(path_or_str) or path_or_str.endswith(".json"):
            with open(path_or_str, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(path_or_str)

        return cls(
            dt=data["dt"],
            num_observations=data["num_observations"],
            num_actions=data["num_actions"],
            observation_history_len=data["observation_history_len"],
            observations=[ObservationItemConfig(**o) for o in data["observations"]],
            action_managers=[
                ActionManagerConfig(
                    type=am["type"],
                    action_range=tuple(am["action_range"]),
                    params=am["params"],
                )
                for am in data["action_managers"]
            ],
            actuators=[ActuatorConfig(**a) for a in data["actuators"]],
        )
