from __future__ import annotations
import re
import torch
import genesis as gs
from typing import Literal, TypedDict, TYPE_CHECKING, Any
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
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
    """The noise for each index of the buffer."""

    has_noise: bool
    """Does this value have noise associated with it?"""

    has_been_set: bool
    """Has this value been set to the actuator at least once?"""


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
        entity_attr: The attribute of the environment to get the robot from.
        default_noise_scale: (deprecated) This noise scale will be applied to all actuator values. Use `NoisyValue` instead.

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

    def __init__(
        self,
        env: GenesisEnv,
        joint_names: list[str] | str = ".*",
        default_pos: float | NoisyValue | dict = {".*": 0.0},
        kp: float | NoisyValue | dict = None,
        kv: float | NoisyValue | dict = None,
        max_force: float | NoisyValue | tuple[Any, Any] | dict = None,
        damping: float | NoisyValue | dict = None,
        stiffness: float | NoisyValue | dict = None,
        frictionloss: float | NoisyValue | dict = None,
        armature: float | NoisyValue | dict = None,
        default_noise_scale: float = 0.0,
        entity_attr: str = "robot",
    ):
        super().__init__(env, type="actuator")
        self._dofs: dict[str, int] = {}
        self._robot: RigidEntity = getattr(env, entity_attr)
        self._default_pos_cfg = ensure_dof_pattern(default_pos)
        self._kp_cfg = ensure_dof_pattern(kp)
        self._kv_cfg = ensure_dof_pattern(kv)
        self._max_force_cfg = ensure_dof_pattern(max_force)
        self._damping_cfg = ensure_dof_pattern(damping)
        self._stiffness_cfg = ensure_dof_pattern(stiffness)
        self._frictionloss_cfg = ensure_dof_pattern(frictionloss)
        self._armature_cfg = ensure_dof_pattern(armature)
        self._default_noise_scale = default_noise_scale

        self._batch_dofs_enabled = (
            env.scene.rigid_options.batch_dofs_info
            and env.scene.rigid_options.batch_links_info
        )

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

    @property
    def join_names(self) -> list[str]:
        """
        Get the names of the joints that are enabled, in the order of the DOF indices.
        """
        return list[str](self._dofs.keys())

    """
    Actuator handlers
    """

    def get_dofs_position(self, noise: float = 0.0):
        """
        Return the current position of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_position`.

        Args:
            noise: The maximum amount of random noise to add to the position values returned.
        """
        pos = self._robot.get_dofs_position(self.dofs_idx)
        if noise > 0.0:
            pos = self._add_random_noise(pos, noise)
        return pos

    def get_dofs_velocity(self, noise: float = 0.0, clip: tuple[float, float] = None):
        """
        Return the current velocity of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_velocity`.

        Args:
            noise: The maximum amount of random noise to add to the velocity values returned.
            clip: Clip the velocity returned.
        """
        vel = self._robot.get_dofs_velocity(self.dofs_idx)
        if noise > 0.0:
            vel = self._add_random_noise(vel, noise)
        if clip is not None:
            vel = vel.clamp(**clip)
        return vel

    def get_dofs_force(self, noise: float = 0.0, clip_to_max_force: bool = False):
        """
        Return the force experienced by the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_force`.

        Args:
            noise: The maximum amount of random noise to add to the force values returned.
            clip_to_max_force: Clip the force returned to the maximum force of the actuators.

        Returns:
            The force experienced by the enabled DOFs.
        """
        force = self._robot.get_dofs_force(self.dofs_idx)
        if noise > 0.0:
            force = self._add_random_noise(force, noise)
        if clip_to_max_force:
            [lower, upper] = self._robot.get_dofs_force_range(self.dofs_idx)
            force = force.clamp(lower, upper)
        return force

    def get_dofs_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the limits of the configured DOFs.
        This is a wrapper for `RigidEntity.get_dofs_limit`.

        Returns:
            A tuple of two tensors, the first is the lower limits and the second is the upper limits.
            Each tensor is of shape (num_envs, num_dofs).
        """
        return self._robot.get_dofs_limit(self.dofs_idx)

    def set_dofs_position(self, position: torch.Tensor):
        """
        Set the position of the configured DOFs.
        This is a wrapper for `RigidEntity.set_dofs_position`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        self._robot.set_dofs_position(position, self.dofs_idx)

    def control_dofs_position(self, position: torch.Tensor):
        """
        Control the position of the configured DOFs.
        This is a wrapper for `RigidEntity.control_dofs_position`.

        Args:
            position: The position to set the DOFs to. The indices of this tensor should match the configured DOFs
                      (see: `dofs_names` and `dofs_idx` properties).
        """
        self._robot.control_dofs_position(position, self.dofs_idx)

    """
    Lifecycle operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        # Find all configured joints by names/patterns
        for joint in self._robot.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            name = joint.name
            for pattern in self._joint_name_cfg:
                if pattern == name or re.match(f"^{pattern}$", name):
                    self._dofs[name] = joint.dof_start
                    break

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
                        "WARNING: Armature randomization settings are only supported when 'batch_dofs_info' and 'batch_links_info' are True in RigidOptions."
                    )
                self._robot.set_dofs_armature(armature["buffer"], self.dofs_idx)

        # Other actuator values
        self._fill_value_buffer("kp", self._kp_cfg)
        self._fill_value_buffer("kv", self._kv_cfg)
        self._fill_value_buffer("damping", self._damping_cfg)
        self._fill_value_buffer("stiffness", self._stiffness_cfg)
        self._fill_value_buffer("frictionloss", self._frictionloss_cfg)

    def reset(
        self,
        envs_idx: list[int] = None,
    ):
        """Reset the DOF positions."""
        if not self.enabled:
            return
        dofs_idx = self.dofs_idx

        # Set actuator controller values
        if self._should_set_value("kp"):
            kp = self._get_value_buffer("kp", envs_idx)
            self._robot.set_dofs_kp(kp, dofs_idx, envs_idx)
            self._values["kp"]["has_been_set"] = True

        if self._should_set_value("kv"):
            kv = self._get_value_buffer("kv", envs_idx)
            self._robot.set_dofs_kv(kv, dofs_idx, envs_idx)
            self._values["kv"]["has_been_set"] = True

        if self._should_set_value("damping"):
            damping = self._get_value_buffer("damping", envs_idx)
            self._robot.set_dofs_damping(damping, dofs_idx, envs_idx)
            self._values["damping"]["has_been_set"] = True

        if self._should_set_value("stiffness"):
            stiffness = self._get_value_buffer("stiffness", envs_idx)
            self._robot.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)
            self._values["stiffness"]["has_been_set"] = True

        if self._should_set_value("frictionloss"):
            frictionloss = self._get_value_buffer("frictionloss", envs_idx)
            self._robot.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)
            self._values["frictionloss"]["has_been_set"] = True

        if self._should_set_value("armature") and self._batch_dofs_enabled:
            armature = self._get_value_buffer("armature", envs_idx)
            self._robot.set_dofs_armature(armature, dofs_idx, envs_idx)
            self._values["armature"]["has_been_set"] = True

        if self._should_set_value("force_min") or self._should_set_value("force_max"):
            force_min = self._get_value_buffer("force_min", envs_idx)
            force_max = self._get_value_buffer("force_max", envs_idx)
            self._robot.set_dofs_force_range(force_min, force_max, dofs_idx, envs_idx)
            self._values["force_min"]["has_been_set"] = True
            self._values["force_max"]["has_been_set"] = True

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
        Check if the actuator control value should.
        We don't want to set the value if we've already set it and there is no noise associated with it.
        """
        cfg = self._values[value_name]
        if cfg is None:
            return False
        if cfg["has_been_set"] and not cfg["has_noise"]:
            return False
        return True

    def _fill_value_buffer(
        self, value_name: ValueName, config: NoisyValue[float] | None
    ):
        """
        Given a ActuatorValue dict, loop over the entries, and set them to the appropriate value buffer DOF indices that match
        the DOF pattern.

        Args:
            name: The name of the value buffer to fill
            values: The DOF value to convert (for example: `{".*": 50}`).

        """
        num_dofs = len(self._dofs)
        is_idx_set = [False] * num_dofs
        dof_names = self._dofs.keys()
        has_noise = False

        # Nothing to be done if the config is None
        if config is None:
            return

        # Initialize the buffers
        buffer = torch.zeros((num_dofs,), device=gs.device, dtype=gs.tc_float)
        noise = torch.zeros((num_dofs,), device=gs.device, dtype=gs.tc_float).fill_(
            self._default_noise_scale
        )

        for pattern, value in config.items():
            found = False
            for i, name in enumerate[str](dof_names):
                if is_idx_set[i]:
                    continue
                if pattern == name or re.match(f"^{pattern}$", name):
                    found = True
                    is_idx_set[i] = True

                    if isinstance(value, NoisyValue):
                        noise[i] = value.noise
                        buffer[i] = value.value
                        has_noise = True
                    else:
                        buffer[i] = value
            if not found:
                raise RuntimeError(f"Joint DOF '{pattern}' not found.")

        # Expand the default postion buffer to the number of environments
        if value_name == "default_pos" or self._batch_dofs_enabled:
            buffer = buffer.unsqueeze(0).expand(self.env.num_envs, -1)
            noise = noise.unsqueeze(0).expand(self.env.num_envs, -1)

        # Expand the buffer to the number of environments
        self._values[value_name] = {
            "buffer": buffer,
            "noise": noise,
            "has_noise": has_noise,
            "has_been_set": False,
        }

    def _get_value_buffer(
        self, name: ValueName, envs_idx: list[int] | None = None
    ) -> torch.Tensor:
        """
        Get the value buffer tensor, with noise applied
        """
        values = self._values[name]["buffer"].clone()
        noise = self._values[name]["noise"]
        values += torch.empty_like(values).uniform_(-1, 1) * noise
        if envs_idx is not None and values.ndim == 2:
            values = values[envs_idx]
        return values

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
