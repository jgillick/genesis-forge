from __future__ import annotations

from collections.abc import Mapping
from itertools import compress
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.utils import assign_by_pattern, inf_to_none, name_matches
from genesis_forge.values import ensure_dof_pattern

from .noisy_value import NoisyValue

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

ValueName = Literal[
    "kp",
    "kv",
    "default_pos",
    "force_min",
    "force_max",
    "damping",
    "stiffness",
    "frictionloss",
    "armature",
]


class BufferItem(TypedDict):
    buffer: torch.Tensor
    """A buffer of values for each DOF index."""

    noise: torch.Tensor
    """The noise scale for each index of the buffer."""

    noise_buffer: torch.Tensor
    """A pre-allocated buffer that will be filled with random noise values at each reset."""

    output_buffer: torch.Tensor
    """A pre-allocated buffer that will be filled with the values at each step."""

    has_noise: bool
    """Does this value have noise associated with it?"""

    has_been_set: bool
    """Has this value been set to the actuator at least once?"""

    defined: torch.Tensor
    """Which DOFs the config defines a value for. The others are never set, so they
    keep the simulator's value."""

    defined_dofs_idx: list[int]
    """The DOF indices of the defined DOFs, for setting the value on the robot."""

    all_defined: bool
    """Does the config define a value for every DOF?"""


ValueBuffers = dict[ValueName, BufferItem]


class ActuatorManager(BaseManager):
    """
    Configures and manages the actuators of your robot.
    You can define values for all actuators, or target specific joints by name or name pattern (see example).
    To add some domain randomization, you can define the values as `NoisyValue` objects, which will apply random noise at each reset.

    Args:
        env: The environment to manage the DOF actuators for.
        joint_names: The joint names to manage.
        default_pos: The default DOF positions. The DOF joints will be set to these positions on reset.
        kp: The positional gain values.
        kv: The velocity gains values.
        max_force: Define the maximum actuator force. Either as a single value or a tuple range.
        damping: The damping values.
        stiffness: The stiffness values.
        frictionloss: The frictionloss values.
        armature: The armature values.
        entity: The robot entity to manage. Defaults to `env.robot`.

    A joint a value does not cover (for example, ``kp={"hip": 50}`` leaves the knee out)
    is never set by this manager, so it keeps the simulator's value -- likely from the
    robot model file. ``default_pos`` is the exception: an uncovered joint resets to 0.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=".*",
                    default_pos={
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        "Leg[1-4]_Femur": 0.5,
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    kp={
                        ".*_Hip": NoisyValue(50, 0.02),
                        "*__Femur": NoisyValue(30, 0.01),
                        "*__Tibia": NoisyValue(30, 0.01),
                    },
                    kv=0.5,
                    max_force=8.0,
                )

    """

    _dofs_idx_tensor: torch.Tensor
    """The configured DOF indices, as a tensor for passing to Genesis."""

    _sim_accessors: ClassVar[dict[str, tuple[str, str]]] = {
        "kp": ("set_dofs_kp", "get_dofs_kp"),
        "kv": ("set_dofs_kv", "get_dofs_kv"),
        "damping": ("set_dofs_damping", "get_dofs_damping"),
        "stiffness": ("set_dofs_stiffness", "get_dofs_stiffness"),
        "frictionloss": ("set_dofs_frictionloss", "get_dofs_frictionloss"),
        "armature": ("set_dofs_armature", "get_dofs_armature"),
    }
    """The robot's setter and getter for each per-DOF value. Force limits are a pair,
    so they are handled on their own."""

    def __init__(
        self,
        env: GenesisEnv,
        joint_names: list[str] | str = ".*",
        default_pos: float | NoisyValue | dict | None = None,
        kp: float | NoisyValue | dict = None,
        kv: float | NoisyValue | dict = None,
        max_force: float | NoisyValue | tuple[Any, Any] | dict = None,
        damping: float | NoisyValue | dict = None,
        stiffness: float | NoisyValue | dict = None,
        frictionloss: float | NoisyValue | dict = None,
        armature: float | NoisyValue | dict = None,
        entity: RigidEntity = None,
    ):
        super().__init__(env, type="actuator")
        default_pos_val = default_pos if default_pos is not None else {".*": 0.0}

        self._dofs: dict[str, int] = {}
        self._robot: RigidEntity = entity if entity is not None else env.robot
        self._default_pos_cfg = ensure_dof_pattern(default_pos_val)
        self._kp_cfg = ensure_dof_pattern(kp)
        self._kv_cfg = ensure_dof_pattern(kv)
        self._max_force_cfg = ensure_dof_pattern(max_force)
        self._damping_cfg = ensure_dof_pattern(damping)
        self._stiffness_cfg = ensure_dof_pattern(stiffness)
        self._frictionloss_cfg = ensure_dof_pattern(frictionloss)
        self._armature_cfg = ensure_dof_pattern(armature)
        self._batch_dofs_enabled = False  # determined at build
        # The simulator's force range, for DOFs max_force leaves out. Read again after
        # each reset, rather than on every step.
        self._sim_force_range: tuple[torch.Tensor, torch.Tensor] | None = None

        self._values: ValueBuffers = {
            "default_pos": None,
            "force_min": None,
            "force_max": None,
            "kp": None,
            "kv": None,
            "damping": None,
            "stiffness": None,
            "frictionloss": None,
            "armature": None,
        }

        if isinstance(joint_names, str):
            self._joint_name_cfg = [joint_names]
        elif isinstance(joint_names, list):
            self._joint_name_cfg = joint_names

    """
    Properties
    """

    @property
    def dofs(self) -> dict[str, int]:
        """
        Get the names of the joints that are enabled, in the order of the DOF indices.
        """
        return self._dofs

    @property
    def num_dofs(self) -> int:
        """
        Get the number of configured DOFs.
        """
        return len(self._dofs)

    @property
    def dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return list[int](self._dofs.values())

    @property
    def dofs_names(self) -> list[str]:
        """
        Get the names of the configured DOFs.
        """
        return list[str](self._dofs.keys())

    @property
    def default_dofs_pos(self) -> torch.Tensor:
        """
        Return the default DOF positions.
        """
        return self._values.get("default_pos", {}).get("buffer", None)

    """
    DOF groups
    """

    def resolve_dof_groups(
        self, dof_groups: Mapping[str, str | list[str]]
    ) -> dict[str, torch.Tensor]:
        """
        Resolve named groups of joint name patterns into robot DOF indices.

        Args:
            dof_groups: Maps a group name to a joint name pattern, or a list of them.

        Returns:
            Maps each group name to the DOF indices of its joints.

        Raises:
            ValueError: A group matched none of this manager's joints.

        Example::

            groups = actuator_manager.resolve_dof_groups({
                "hip": ".*_Hip",
                "legs": [".*_Femur", ".*_Tibia"],
            })
            hip_torque = actuator_manager.get_dofs_control_force(dofs_idx=groups["hip"])
        """
        resolved: dict[str, torch.Tensor] = {}
        for group, patterns in dof_groups.items():
            if isinstance(patterns, str):
                patterns = [patterns]
            dofs_idx = [
                idx
                for name, idx in self._dofs.items()
                if any(name_matches(name, pattern) for pattern in patterns)
            ]
            if not dofs_idx:
                raise ValueError(
                    f"DOF group '{group}' ({patterns}) matched none of the "
                    f"actuator manager's joints: {self.dofs_names}"
                )
            resolved[group] = torch.tensor(dofs_idx, dtype=gs.tc_int, device=gs.device)
        return resolved

    """
    Deployment
    """

    def get_deployment_values(self) -> dict[str, Any]:
        """Nominal actuator gains and defaults, for a deployment bundle.

        Reads the *nominal* buffers -- the values configured at build time, before
        any per-reset domain randomization, which is only ever written into the
        separate output buffer. A robot's motor controllers should be configured
        with these, matching what the policy trained against.

        Returns:
            Plain data: the joint names, one value per joint for each configured
            actuator parameter, and the names of any parameters that were
            randomized during training (recorded so the difference is visible
            rather than silently flattened). A joint a value does not cover gets
            the simulator's value for it, and ``None`` when that is unbounded.

        Raises:
            ValueError: A configured parameter is not one value per DOF, so the
                bundle cannot record it.
        """
        values: dict[str, list[float | None]] = {}
        randomized: list[str] = []

        for value_name, item in self._values.items():
            # Every parameter is pre-seeded as None; only configured ones hold a buffer.
            if not item:
                continue
            buffer = item.get("buffer", None)
            if buffer is None:
                continue
            tensor = buffer.detach()
            # Per-DOF buffers may carry a leading environment dimension; the
            # nominal row is identical across environments by construction.
            if tensor.ndim == 2:
                tensor = tensor[0]
            if tensor.ndim != 1 or tensor.shape[0] != self.num_dofs:
                raise ValueError(
                    f"Actuator value '{value_name}' has shape "
                    f"{tuple(buffer.shape)}, which is not one value per DOF for "
                    f"{self.num_dofs} DOF(s). A bundle records one nominal value "
                    f"per joint, and dropping this one would leave the robot "
                    f"configuring its controllers from a partial set of gains."
                )

            if value_name in ("force_min", "force_max"):
                # A pair, so both come from the range getter
                lower, upper = self.get_dofs_nominal_force_range()
                tensor = lower if value_name == "force_min" else upper
            elif value_name in self._sim_accessors and not item.get(
                "all_defined", True
            ):
                # A DOF the config leaves out keeps the simulator's value, so export that
                _, getter = self._sim_accessors[value_name]
                sim_value = self._get_sim_value(getter)
                tensor = torch.where(item["defined"], tensor, sim_value)

            values[value_name] = inf_to_none(tensor)
            if item.get("has_noise", False):
                randomized.append(value_name)

        return {
            "joint_names": list(self.dofs_names),
            "values": values,
            "randomized": sorted(randomized),
        }

    """
    Actuator handlers
    """

    def get_dofs_position(
        self, noise: float = 0.0, dofs_idx: list[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Return the current position of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_position`.

        Args:
            noise: The maximum amount of random noise to add to the position values returned.
            dofs_idx: The indices of the DOFs to get the position for. If None, all the DOFs of this actuator manager are used.

        Returns:
            position: torch.Tensor, shape (n_envs, n_dofs)
            The position of the enabled DOFs.
        """
        dofs_idx = dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        pos = self._robot.get_dofs_position(dofs_idx)
        if noise > 0.0:
            pos = self._add_random_noise(pos, noise)
        return pos

    def get_dofs_velocity(
        self,
        noise: float = 0.0,
        clip: tuple[float, float] | None = None,
        dofs_idx: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the current velocity of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_velocity`.

        Args:
            noise: The maximum amount of random noise to add to the velocity values returned.
            clip: Clip the velocity returned.
            dofs_idx: The indices of the DOFs to get the velocity for. If None, all the DOFs of this actuator manager are used.

        Returns:
            velocity:torch.Tensor, shape (n_envs, n_dofs)
            The velocity of the enabled DOFs.
        """
        dofs_idx = dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        vel = self._robot.get_dofs_velocity(dofs_idx)
        if noise > 0.0:
            vel = self._add_random_noise(vel, noise)
        if clip is not None:
            vel = vel.clamp(**clip)
        return vel

    def get_dofs_force(
        self,
        noise: float = 0.0,
        clip_to_max_force: bool = False,
        dofs_idx: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the force experienced by the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_force`.

        Args:
            noise: The maximum amount of random noise to add to the force values returned.
            clip_to_max_force: Clip the force returned to the maximum force of the actuators.
            dofs_idx: The indices of the DOFs to get the force for. If None, all the DOFs of this actuator manager are used.

        Returns:
            force: torch.Tensor, shape (n_envs, n_dofs)
            The force experienced by the enabled DOFs.
        """
        dofs_idx = dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        force = self._robot.get_dofs_force(dofs_idx)
        if noise > 0.0:
            force = self._add_random_noise(force, noise)
        if clip_to_max_force:
            [lower, upper] = self._robot.get_dofs_force_range(dofs_idx)
            force = force.clamp(lower, upper)
        return force

    def get_dofs_control_force(
        self,
        noise: float = 0.0,
        clip_to_max_force: bool = False,
        dofs_idx: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return the force output by the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_control_force`.

        Args:
            noise: The maximum amount of random noise to add to the force values returned.
            clip_to_max_force: Clip the force returned to the maximum force of the actuators.
            dofs_idx: The indices of the DOFs to get the force for. If None, all the DOFs of this actuator manager are used.

        Returns:
            force: torch.Tensor, shape (n_envs, n_dofs)
            The force experienced by the enabled DOFs.
        """
        dofs_idx = dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        force = self._robot.get_dofs_control_force(dofs_idx)
        if noise > 0.0:
            force = self._add_random_noise(force, noise)
        if clip_to_max_force:
            [lower, upper] = self._robot.get_dofs_force_range(dofs_idx)
            force = force.clamp(lower, upper)
        return force

    def get_dofs_force_range(
        self, dofs_idx: list[int] | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the force range the simulator currently holds for the configured DOFs,
        including any randomized ``max_force`` applied at the last reset.

        Args:
            dofs_idx: The indices of the DOFs to get the range for. If None, all the DOFs of this actuator manager are used.

        Returns:
            lower: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            upper: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
        """
        dofs_idx = dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        return self._robot.get_dofs_force_range(dofs_idx)

    def get_dofs_max_force(self) -> torch.Tensor:
        """
        Get the positive force/torque limit per configured DOF from ``max_force`` config,
        including any randomized value applied at the last reset. A DOF with a missing
        ``max_force`` config value reports the limit the simulator currently holds
        for it -- likely from the robot model file.

        Returns:
            limits: torch.Tensor, shape (n_envs, n_dofs) or (n_dofs,)
                Upper force limit per DOF (symmetric limits use the same magnitude).

        Raises:
            ValueError: If ``max_force`` was not configured on this actuator manager.
        """
        if self._max_force_cfg is None or self._values.get("force_max") is None:
            raise ValueError(
                "ActuatorManager max_force is not configured. "
                "Set max_force on the actuator manager or pass an explicit threshold."
            )
        force_max = self._values["force_max"]
        limits = torch.abs(force_max["output_buffer"])
        if limits.ndim == 1:
            limits = limits.unsqueeze(0).expand(self.env.num_envs, -1)
        if not force_max["all_defined"]:
            if self._sim_force_range is None:
                self._sim_force_range = self._get_sim_force_range()
            _, sim_upper = self._sim_force_range
            limits = torch.where(force_max["defined"], limits, sim_upper.abs())
        return limits

    def get_dofs_nominal_force_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the force/torque range per configured DOF from the ``max_force`` config,
        without any randomization noise. A DOF that does not have a ``max_force``
        defined reports the limit the simulator currently holds for it -- likely
        from the robot model file.

        Returns:
            lower: torch.Tensor, shape (n_dofs,)
                   The lower force limit per DOF.
            upper: torch.Tensor, shape (n_dofs,)
                   The upper force limit per DOF.
        """
        if self._values["force_max"] is None:
            return self._get_sim_force_range()

        lower = self._first_env_row(self._values["force_min"]["buffer"])
        upper = self._first_env_row(self._values["force_max"]["buffer"])
        force_max = self._values["force_max"]
        if not force_max["all_defined"]:
            sim_lower, sim_upper = self._get_sim_force_range()
            lower = torch.where(force_max["defined"], lower, sim_lower)
            upper = torch.where(force_max["defined"], upper, sim_upper)
        return lower.clone(), upper.clone()

    def get_dofs_limits(
        self, dofs_idx: list[int] | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the position limits of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_limit`.

        Args:
            dofs_idx: The indices of the DOFs to get the limits for. If None, all the DOFs of this actuator manager are used.

        Returns:
            lower_limit: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
                         The lower limit of the positional limits for the entity's dofs.
            upper_limit: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
                         The upper limit of the positional limits for the entity's dofs.
        """
        dofs_idx = dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        return self._robot.get_dofs_limit(dofs_idx)

    def set_dofs_position(
        self, position: torch.Tensor, dofs_idx: list[int] | torch.Tensor | None = None
    ):
        """
        Set the position of the configured DOFs.
        This is a wrapper for `RigidEntity.set_dofs_position`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
            dofs_idx: The indices of the DOFs to control. If None, all the DOFs of this actuator manager are used.
        """
        self._robot.set_dofs_position(
            position, dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        )

    def control_dofs_position(
        self, position: torch.Tensor, dofs_idx: list[int] | torch.Tensor | None = None
    ):
        """
        Control the position of the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_position`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
            dofs_idx: The indices of the DOFs to control. If None, all the DOFs of this actuator manager are used.
        """
        self._robot.control_dofs_position(
            position, dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        )

    def control_dofs_velocity(
        self, velocity: torch.Tensor, dofs_idx: list[int] | torch.Tensor | None = None
    ):
        """
        Control the velocity of the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_velocity`.

        Args:
            velocity: The velocity to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
            dofs_idx: The indices of the DOFs to control. If None, all the DOFs of this actuator manager are used.
        """
        self._robot.control_dofs_velocity(
            velocity, dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        )

    def control_dofs_force(
        self, force: torch.Tensor, dofs_idx: list[int] | torch.Tensor | None = None
    ):
        """
        Apply a torque to the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_force`.

        Args:
            force: The torque to apply to the DOFs. The indices of this tensor should match the configured DOFs
                   (see: `dofs_names` and `dofs_idx` properties).
            dofs_idx: The indices of the DOFs to control. If None, all the DOFs of this actuator manager are used.
        """
        self._robot.control_dofs_force(
            force, dofs_idx if dofs_idx is not None else self._dofs_idx_tensor
        )

    """
    Lifecycle operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        # Whether DOF batching is enabled, which allows some DOF
        # properties to vary per environment. Every DOF setter this manager
        # uses gates per-env values on batch_dofs_info alone; batch_links_info
        # only affects link properties (e.g. link mass), which are not set here.
        solver = getattr(self._robot, "solver", None)
        if hasattr(solver, "is_links_info_batched"):
            # genesis-world >= 1.4
            self._batch_dofs_enabled = self._robot.get_dofs_armature().dim() == 2
        else:
            self._batch_dofs_enabled = self.env.scene.rigid_options.batch_dofs_info

        # Find all configured joints by names/patterns
        for joint in self._robot.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            name = joint.name
            for pattern in self._joint_name_cfg:
                if name_matches(name, pattern):
                    self._dofs[name] = joint.dof_start
                    break
        self._dofs_idx_tensor = torch.tensor(
            self.dofs_idx, dtype=gs.tc_int, device=gs.device
        )

        # Default DOF positions
        # If no configuration is provided, use zero positions for all DOFs.
        if self._default_pos_cfg is not None:
            self._fill_value_buffer("default_pos", self._default_pos_cfg)
        else:
            self._fill_value_buffer("default_pos", {".*": 0.0})

        # Max force
        # The value can either be a single float or a tuple range
        # First normalize them into two dicts: min and max
        if self._max_force_cfg is not None:
            # Normalize the max_force values into a min & max dict
            force_min = {}
            force_max = {}
            for pattern, value in self._max_force_cfg.items():
                if isinstance(value, list):
                    force_min[pattern] = value[0]
                    force_max[pattern] = value[1]
                elif isinstance(value, NoisyValue):
                    force_min[pattern] = NoisyValue(-value.value, value.noise)
                    force_max[pattern] = value
                else:
                    force_min[pattern] = -value
                    force_max[pattern] = value

            self._fill_value_buffer("force_min", force_min)
            self._fill_value_buffer("force_max", force_max)

        # Armature
        # If DOF batching is not enabled, print a warning and set the armature values for all environments without noise.
        if self._armature_cfg is not None:
            self._fill_value_buffer("armature", self._armature_cfg)
            if not self._batch_dofs_enabled:
                armature = self._values["armature"]
                if torch.any(armature["noise"] != 0.0):
                    print(
                        "WARNING: Armature randomization settings are only supported when 'batch_dofs_info' is True in RigidOptions."
                    )
                self._robot.set_dofs_armature(
                    self._defined_only("armature", armature["buffer"]),
                    armature["defined_dofs_idx"],
                )

        # Other actuator values
        self._fill_value_buffer("kp", self._kp_cfg)
        self._fill_value_buffer("kv", self._kv_cfg)
        self._fill_value_buffer("damping", self._damping_cfg)
        self._fill_value_buffer("stiffness", self._stiffness_cfg)
        self._fill_value_buffer("frictionloss", self._frictionloss_cfg)

    def reset(
        self,
        envs_idx: torch.Tensor | None = None,
    ):
        """Reset the DOF positions."""
        if not self.enabled:
            return
        dofs_idx = self.dofs_idx

        # Set actuator controller values, on the DOFs each one defines
        for name, (setter, _) in self._sim_accessors.items():
            if name == "armature" and not self._batch_dofs_enabled:
                continue  # Set once, at build
            if not self._should_set_value(name):
                continue
            value = self._get_value_buffer(name, envs_idx)
            getattr(self._robot, setter)(
                self._defined_only(name, value),
                self._values[name]["defined_dofs_idx"],
                envs_idx,
            )
            self._mark_set(name, envs_idx)

        if self._should_set_value("force_min") or self._should_set_value("force_max"):
            force_min = self._get_value_buffer("force_min", envs_idx)
            force_max = self._get_value_buffer("force_max", envs_idx)
            self._robot.set_dofs_force_range(
                self._defined_only("force_min", force_min),
                self._defined_only("force_max", force_max),
                self._values["force_max"]["defined_dofs_idx"],
                envs_idx,
            )
            self._mark_set("force_min", envs_idx)
            self._mark_set("force_max", envs_idx)
        self._sim_force_range = None

        # Reset DOF positions
        default_position = self._get_value_buffer("default_pos", envs_idx)
        self._robot.set_dofs_position(
            position=default_position,
            dofs_idx_local=dofs_idx,
            envs_idx=envs_idx,
        )

    """
    Internal methods
    """

    def _should_set_value(self, value_name: ValueName) -> bool:
        """
        Check if the actuator value should be set.
        We don't want to set the value if we've already set it and there is no noise associated with it.
        """
        cfg = self._values[value_name]
        if cfg is None:
            return False
        if cfg["has_been_set"] and not cfg["has_noise"]:  # noqa SIM103
            return False
        return True

    def _fill_value_buffer(
        self,
        value_name: ValueName,
        config: dict | None,
    ):
        """
        Given a ActuatorValue dict, loop over the entries, and set them to the appropriate value buffer DOF indices that match
        the DOF pattern.

        Args:
            value_name: The name of the value buffer to fill
            config: The DOF value to convert (for example: `{".*": 50}`).

        """
        num_dofs = len(self._dofs)
        dof_names = list(self._dofs.keys())
        has_noise = False

        # Nothing to be done if the config is None
        if config is None:
            return

        # Initialize the buffers
        value_buffer = torch.zeros((num_dofs,), device=gs.device, dtype=gs.tc_float)
        noise = torch.zeros((num_dofs,), device=gs.device, dtype=gs.tc_float)
        noise_buffer = torch.zeros_like(value_buffer, device=gs.device)

        dof_values = assign_by_pattern(dof_names, config)
        defined = [value is not None for value in dof_values]
        for i, cfg_value in enumerate(dof_values):
            if cfg_value is None:
                continue
            if isinstance(cfg_value, NoisyValue):
                noise[i] = cfg_value.noise
                value_buffer[i] = cfg_value.value
                has_noise = True
            else:
                value_buffer[i] = cfg_value

        # Expand the default position buffer to the number of environments
        if value_name == "default_pos" or self._batch_dofs_enabled:
            value_buffer = value_buffer.unsqueeze(0).repeat(self.env.num_envs, 1)
            noise = noise.unsqueeze(0).repeat(self.env.num_envs, 1)
            noise_buffer = noise_buffer.unsqueeze(0).repeat(self.env.num_envs, 1)

        self._values[value_name] = {
            "buffer": value_buffer,
            "noise": noise,
            "noise_buffer": noise_buffer,
            # Nominal until a reset applies noise, so it never reads as 0
            "output_buffer": value_buffer.clone(),
            "has_noise": has_noise,
            "has_been_set": False,
            "defined": torch.tensor(defined, device=gs.device),
            "defined_dofs_idx": list(compress(self.dofs_idx, defined)),
            "all_defined": all(defined),
        }

    def _get_value_buffer(
        self, name: ValueName, envs_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Sample new noise for the value, and get the result to apply to the robot.

        Only the rows of the environments being reset are resampled, so the output
        buffer always holds the values that were last applied to each environment.
        """
        output = self._values[name]["output_buffer"]
        noise = self._values[name]["noise"]
        noise_buffer = self._values[name]["noise_buffer"]
        buffer = self._values[name]["buffer"]

        if envs_idx is not None and output.ndim == 2:
            sample = torch.empty_like(output[envs_idx]).uniform_(-1, 1)
            output[envs_idx] = buffer[envs_idx] + sample * noise[envs_idx]
            return output[envs_idx]

        output[:] = buffer
        output += noise_buffer.uniform_(-1, 1) * noise
        return output

    def _defined_only(self, name: ValueName, value: torch.Tensor) -> torch.Tensor:
        """
        Return the value for only the DOFs its config defines.
        """
        item = self._values[name]
        if item["all_defined"]:
            return value
        return value[..., item["defined"]]

    def _mark_set(self, name: ValueName, envs_idx: torch.Tensor | None):
        """
        Set the value's ``has_been_set`` flag if this reset covered every environment,
        or if DOF batching is off, so every environment shares one value.
        """
        reset_count = self.env.num_envs if envs_idx is None else len(envs_idx)
        if reset_count == self.env.num_envs or not self._batch_dofs_enabled:
            self._values[name]["has_been_set"] = True

    def _add_random_noise(
        self, values: torch.Tensor, noise_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Add random noise to the tensor values
        """
        if noise_scale == 0.0:
            return values
        noise_value = torch.empty_like(values).uniform_(-1, 1) * noise_scale
        return values + noise_value

    def _get_sim_value(self, getter: str) -> torch.Tensor:
        """
        Read a per-DOF value the simulator currently holds for the configured DOFs.

        Args:
            getter: The name of the robot's ``get_dofs_*`` method for the value.

        Returns:
            value: torch.Tensor, shape (n_dofs,)
        """
        read_value = getattr(self._robot, getter)
        value = read_value(self.dofs_idx)
        return self._first_env_row(value)

    def _get_sim_force_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The force range the simulator currently holds for the configured DOFs.

        Returns:
            lower: torch.Tensor, shape (n_dofs,)
            upper: torch.Tensor, shape (n_dofs,)
        """
        lower, upper = self._robot.get_dofs_force_range(self.dofs_idx)
        return self._first_env_row(lower), self._first_env_row(upper)

    @staticmethod
    def _first_env_row(value: torch.Tensor) -> torch.Tensor:
        """
        Convert a per-DOF value to a tensor of shape (n_dofs,).

        A value with a row per environment (from DOF batching) is reduced to
        environment 0's row, which matches the others unless something randomized it.
        """
        value = torch.as_tensor(value, device=gs.device, dtype=gs.tc_float)
        return value[0] if value.ndim == 2 else value
