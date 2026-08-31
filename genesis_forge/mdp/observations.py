from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from genesis import gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    MdpFn,
    PositionActionManager,
)
from genesis_forge.managers.action.base import BaseActionManager
from genesis_forge.utils import (
    entity_ang_vel,
    entity_lin_vel,
)
from genesis_forge.utils import (
    entity_projected_gravity as get_entity_projected_gravity,
)

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity
    from genesis.engine.sensors.imu import IMUSensor

"""
Entity Observations
"""


@dataclass(kw_only=True, eq=False)
class entity_linear_velocity(MdpFn):
    """
    The linear velocity of the entity's base link, in the entity's local frame.

    Args:
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the observation for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The linear velocity of the entity's base link, in the entity's local frame.
    """

    entity_manager: EntityManager = None
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            return self.entity_manager.get_linear_velocity()
        entity = self.entity if self.entity is not None else env.robot
        return entity_lin_vel(entity)


@dataclass(kw_only=True, eq=False)
class entity_angular_velocity(MdpFn):
    """
    The angular velocity of the entity's base link, in the entity's local frame.

    Args:
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the observation for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The angular velocity of the entity's base link, in the entity's local frame.
    """

    entity_manager: EntityManager = None
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            return self.entity_manager.get_angular_velocity()
        entity = self.entity if self.entity is not None else env.robot
        return entity_ang_vel(entity)


@dataclass(kw_only=True, eq=False)
class entity_projected_gravity(MdpFn):
    """
    The projected gravity of the entity's base link, in the entity's local frame.

    Args:
        entity_manager: The entity manager for the robot/entity the observation is being computed for.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to compute the observation for. Defaults to `env.robot`. This isn't necessary if `entity_manager` is provided.

    Returns:
        torch.Tensor: The projected gravity of the entity's base link, in the entity's local frame.
    """

    entity_manager: EntityManager = None
    entity: RigidEntity = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.entity_manager is not None:
            return self.entity_manager.get_projected_gravity()
        entity = self.entity if self.entity is not None else env.robot
        return get_entity_projected_gravity(entity)


"""
Sensor observations
"""


@dataclass(kw_only=True, eq=False)
class read_imu(MdpFn):
    """
    Makes an IMU reading and returns the concatenated linear acceleration and angular velocity readings.

    Args:
        imu: The IMU sensor to read from.

    Example::

        self.imu = gs.sensors.IMU(
            entity_idx=self.robot.idx,
            pos_offset=(0.24, 0.0, 0.0),
            euler_offset=(0.0, 0.0, 0.0),
        )

        ...

        ObservationManager(
            self,
            cfg={
                "imu_sensor": {
                    "fn": observations.read_imu(imu=self.imu),
                },
            }
        )

    Returns:
        torch.Tensor: Shape `(n_envs, 6)` — `[lin_acc_xyz, ang_vel_xyz]` per env.
    """

    imu: gs.sensors.IMUSensor

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        value = self.imu.read()
        return torch.cat([value.lin_acc, value.ang_vel], dim=-1)


@dataclass(kw_only=True, eq=False)
class imu_projected_gravity(MdpFn):
    """
    Estimates the projected gravity vector from an IMU sensor's accelerometer and
    gyroscope readings, using a complementary filter.

    Each step, the previous estimate is propagated by the gyro reading (a first-order
    approximation of rotating the body-frame gravity vector opposite to the measured
    body rotation) and then pulled towards the accelerometer reading, negated and
    normalized -- under quasi-static conditions (no significant linear acceleration)
    an accelerometer measures specific force pointing opposite to gravity. The two are
    blended by ``correction_gain`` and re-normalized.

    Because the estimate is derived entirely from the IMU's `lin_acc`/`ang_vel`
    readings, any noise, bias, delay or drift configured on the sensor carries through
    into the resulting value -- unlike `entity_projected_gravity`, which reads the
    entity's true orientation directly.

    Args:
        imu_sensor: The IMU sensor to read from.
        correction_gain: How strongly the accelerometer corrects the gyro-propagated
            estimate each step, in (0, 1]. Higher values track the accelerometer more
            closely (less drift, but more sensitive to non-gravity acceleration);
            lower values rely more on the gyro propagation (smoother, but drifts
            without correction).

    Example::

        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx,
                pos_offset=(0.24, 0.0, 0.0),
                euler_offset=(0.0, 0.0, 0.0),
                acc_noise=(0.01, 0.01, 0.01),
                gyro_noise=(0.01, 0.01, 0.01),
                acc_random_walk=(0.001, 0.001, 0.001),
                gyro_random_walk=(0.001, 0.001, 0.001),
                delay=self.dt,
                jitter=self.dt,
            )
        )

        ...

        ObservationManager(
            self,
            cfg={
                "projected_gravity": {
                    "fn": observations.imu_projected_gravity(
                        imu_sensor=self.imu,
                    ),
                },
            }
        )

    Returns:
        torch.Tensor: Shape `(num_envs, 3)` -- the estimated gravity direction, in the
            IMU's local frame.
    """

    imu_sensor: IMUSensor
    correction_gain: float = 0.02

    def build(self):
        self._estimate = torch.zeros(
            (self.env.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self._estimate[:, 2] = -1.0

    def reset(self, envs_idx):
        self._estimate[envs_idx] = 0.0
        self._estimate[envs_idx, 2] = -1.0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        reading = self.imu_sensor.read()

        gyro_estimate = self._estimate - env.dt * torch.cross(
            reading.ang_vel, self._estimate, dim=-1
        )
        accel_estimate = -F.normalize(reading.lin_acc, dim=-1)

        blended = (
            1.0 - self.correction_gain
        ) * gyro_estimate + self.correction_gain * accel_estimate
        self._estimate = F.normalize(blended, dim=-1)

        return self._estimate


@dataclass(kw_only=True, eq=False)
class raycaster_distance(MdpFn):
    """
    Distance reading(s) from a raycaster-based sensor (`gs.sensors.Raycaster`, `gs.sensors.Lidar`,
    or `gs.sensors.DepthCamera`).

    With ``reduce="min"``, returns the smallest distance across all of the sensor's rays,
    which approximates a sensor that reports its nearest echo, like an ultrasonic range sensor.
    With ``reduce="flatten"``, returns all ray distances as a flat vector — for example, a
    depth camera image as an observation.

    Args:
        sensor: The raycaster sensor to read from, as returned by `scene.add_sensor()`.
        reduce: How to reduce the ray distances: "min" for the nearest reading (default),
                "flatten" for all readings as a flat vector.
        normalize: Divide the distances by the sensor's max range, scaling them to [0, 1].
        max_range: The range to normalize by. If not set, it is read from the sensor's options.

    Example::

        # In the environment's __init__ (sensors must be added before the scene is built):
        self.ultrasonic = self.scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.SphericalPattern(fov=(15.0, 15.0), n_points=(5, 5)),
                entity_idx=self.robot.idx,
                link_idx_local=self.robot.get_link("head").idx_local,
                max_range=4.0,
                return_points=False,
            )
        )

        # In config():
        ObservationManager(
            self,
            cfg={
                "ultrasonic": {
                    "fn": observations.raycaster_distance(sensor=self.ultrasonic, normalize=True),
                },
            },
        )

    Returns:
        torch.Tensor: Shape `(num_envs, 1)` for "min", or `(num_envs, num_rays)` for "flatten".
    """

    sensor: gs.sensors.RaycasterSensor
    reduce: Literal["min", "flatten"] = "min"
    normalize: bool = False
    max_range: float | None = None

    def build(self):
        self._max_range = self.max_range
        if self._max_range is None:
            options = getattr(self.sensor, "_options", None)
            self._max_range = getattr(options, "max_range", None)
        if self.normalize and self._max_range is None:
            raise ValueError(
                "Could not determine the sensor's max range for normalization. "
                "Set the max_range parameter explicitly."
            )

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        distances = self.sensor.read().distances.flatten(start_dim=1)
        if self.reduce == "min":
            distances = distances.amin(dim=1, keepdim=True)
        if self.normalize:
            distances = distances / self._max_range
        return distances


"""
DOF/Join observations
"""


@dataclass(kw_only=True, eq=False)
class entity_dofs_position(MdpFn):
    """
    The position of the entity's DOFs.

    Args:
        actuator_manager: The actuator manager for the robot/entity.
                          This bypasses the need for dofs_idx and entity parameters.
        entity: The entity to read DOFs from. Defaults to `env.robot`. This isn't necessary if `actuator_manager` is provided.
        dofs_idx: The indices of the DOFs to get the position of. This isn't necessary if `actuator_manager` is provided.

    Returns:
        torch.Tensor: The position of the entity's DOFs.
    """

    actuator_manager: ActuatorManager = None
    entity: RigidEntity = None
    dofs_idx: list[int] = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.actuator_manager is not None:
            return self.actuator_manager.get_dofs_position()
        entity: RigidEntity = self.entity if self.entity is not None else env.robot
        return entity.get_dofs_position(self.dofs_idx)


@dataclass(kw_only=True, eq=False)
class entity_dofs_velocity(MdpFn):
    """
    The velocity of the entity's DOFs.

    Args:
        action_manager: The action manager for the robot/entity.
                        This is slightly more performant than using the `entity` parameter.
        entity: The entity to read DOFs from. Defaults to `env.robot`. This isn't necessary if `action_manager` is provided.
        dofs_idx: The indices of the DOFs to get the velocity of. This isn't necessary if `action_manager` is provided.

    Returns:
        torch.Tensor: The velocity of the entity's DOFs.
    """

    action_manager: PositionActionManager = None
    entity: RigidEntity = None
    dofs_idx: list[int] = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.action_manager is not None:
            return self.action_manager.get_dofs_velocity()
        entity: RigidEntity = self.entity if self.entity is not None else env.robot
        return entity.get_dofs_velocity(self.dofs_idx)


@dataclass(kw_only=True, eq=False)
class entity_dofs_force(MdpFn):
    """
    The DOF's force being experienced.

    Args:
        actuator_manager: The actuator manager for the robot/entity.
                          This bypasses the need for dofs_idx and entity parameters.
        entity: The entity to read DOFs from. Defaults to `env.robot`. This isn't necessary if `actuator_manager` is provided.
        dofs_idx: The indices of the DOFs to get the force of. This isn't necessary if `actuator_manager` is provided.
        clip_to_max_force: Clip the force to the maximum force defined in the `actuator_manager`.

    Returns:
        torch.Tensor: The force of the entity's DOFs.
    """

    actuator_manager: ActuatorManager = None
    entity: RigidEntity = None
    dofs_idx: list[int] = None
    clip_to_max_force: bool = False

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.actuator_manager is not None:
            return self.actuator_manager.get_dofs_force(
                clip_to_max_force=self.clip_to_max_force
            )
        entity: RigidEntity = self.entity if self.entity is not None else env.robot
        return entity.get_dofs_force(self.dofs_idx)


"""
Actions
"""


@dataclass(kw_only=True, eq=False)
class current_actions(MdpFn):
    """
    The most current step's raw actions.
    This should be the actions before they've been processed and converted into their target values.

    Args:
        action_manager: The action manager to source actions from. If not provided,
                        all actions are read from `env.actions`.
    """

    action_manager: BaseActionManager | None = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        if self.action_manager is not None:
            return self.action_manager.raw_actions
        if env.actions is None:
            return torch.zeros((env.num_envs, env.num_actions), device=gs.device)
        return env.actions


"""
Contacts
"""


@dataclass(kw_only=True, eq=False)
class contact_force(MdpFn):
    """
    Returns the vector norm contact force at each contact point.

    Args:
        contact_manager: The contact manager to check for contact

    Returns:
        torch.Tensor: Shape `(num_envs, num_contacts)`.
    """

    contact_manager: ContactManager

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        return torch.norm(self.contact_manager.contacts[:, :, :], dim=-1)


@dataclass(kw_only=True, eq=False)
class has_contact(MdpFn):
    """
    Return 1 (true) or 0 (false) for each link in the contact manager that meets the contact threshold.

    Args:
        contact_manager: The contact manager to check for contact
        threshold: The minimum force necessary for contact detection (default: 1.0)

    Returns:
        1 for each link meeting the contact threshold
    """

    contact_manager: ContactManager
    threshold: float = 1.0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        in_contact = self.contact_manager.contacts.norm(dim=-1) > self.threshold
        return in_contact.float()
