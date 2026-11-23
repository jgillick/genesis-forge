from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.utils import assign_by_pattern

from .dof_monitor import DofMonitor
from .monitor_config import MonitorConfig


@dataclass(frozen=True)
class MotorSpec:
    """
    The electrical values of an electric motor, from its datasheet or controller config.

    Example::

        GIM8108_8 = MotorSpec(torque_constant=0.43, phase_resistance=0.436)
    """

    torque_constant: float
    """
    Torque per amp of phase current, in N·m/A, at the joint (after any gearbox).
    If the datasheet gives it for the motor shaft, multiply it by the gear ratio.
    """

    phase_resistance: float
    """The resistance of one motor phase, in ohms."""


class PowerMonitor(DofMonitor):
    """
    Estimates the electrical power your robot's motors would draw.

    Each motor's power is the heat in its windings (copper loss) plus the positive
    mechanical work it does. Energy returned while braking is assumed lost, so this
    slightly overestimates the draw. Power is calculated from the torque actually
    applied, clipped to the actuator force limits.

    Monitor items:

    - `copper_loss_w`: Winding heat for the whole robot, in watts.
    - `mechanical_w`: Positive mechanical power for the whole robot, in watts.
    - `total_w`: Copper loss plus mechanical power for the whole robot, in watts.
    - `motor_w`: The average power per motor, in watts, for each DOF group.
    - `bus_current_a`: The battery current, in amps. Only when `supply_voltage` is set.

    Args:
        env: The environment to monitor.
        actuator_manager: The actuator manager whose motors are monitored.
        motors: The motor spec for every DOF, or a `{<joint name pattern>: MotorSpec}` dict.
        supply_voltage: The battery voltage, in volts, used to estimate the bus current.
        dof_groups: Named joint name patterns to report on, like `{"hip": ".*_Hip"}`.
        cfg: Additional monitor items.
        logging_enabled: Whether to add the values to the training logs.
        logging_tag: The section name the values are logged under.

    Example::

        GIM8108_8 = MotorSpec(torque_constant=0.43, phase_resistance=0.436)

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.actuator_manager = ActuatorManager(self, ...)

                self.power_monitor = PowerMonitor(
                    self,
                    actuator_manager=self.actuator_manager,
                    motors=GIM8108_8,
                    supply_voltage=44.4,
                    dof_groups={"hip": ".*_Hip", "knee": ".*_Knee"},
                )

                RewardManager(
                    self,
                    cfg={
                        "energy": {
                            "weight": -1e-4,
                            "fn": rewards.electrical_power(power_monitor=self.power_monitor),
                        },
                    },
                )
    """

    _torque_constant: torch.Tensor
    _phase_resistance: torch.Tensor
    _dof_copper_loss: torch.Tensor
    _dof_mechanical_power: torch.Tensor
    _dof_power: torch.Tensor

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager,
        motors: MotorSpec | dict[str, MotorSpec],
        supply_voltage: float | None = None,
        dof_groups: Mapping[str, str | list[str]] | None = None,
        cfg: dict[str, MonitorConfig] | None = None,
        logging_enabled: bool = True,
        logging_tag: str = "Power",
    ):
        self._motors_cfg = motors if isinstance(motors, dict) else {".*": motors}
        self.supply_voltage = supply_voltage

        items: dict[str, MonitorConfig] = {
            "copper_loss_w": {"fn": self._copper_loss_w},
            "mechanical_w": {"fn": self._mechanical_w},
            "total_w": {"fn": self._total_w, "reduce": ["mean", "p99"]},
            "motor_w": {"fn": self._motor_w, "labels": self.group_labels},
        }
        if supply_voltage is not None:
            items["bus_current_a"] = {"fn": self._bus_current_a}

        super().__init__(
            env,
            actuator_manager=actuator_manager,
            dof_groups=dof_groups,
            cfg={**items, **(cfg or {})},
            logging_enabled=logging_enabled,
            logging_tag=logging_tag,
        )

    """
    Properties
    """

    @property
    def dof_copper_loss(self) -> torch.Tensor:
        """Winding heat per motor, in watts. Shape (num_envs, num_dofs)."""
        return self._dof_copper_loss

    @property
    def dof_mechanical_power(self) -> torch.Tensor:
        """Positive mechanical power per motor, in watts. Shape (num_envs, num_dofs)."""
        return self._dof_mechanical_power

    @property
    def dof_power(self) -> torch.Tensor:
        """Estimated electrical power per motor, in watts. Shape (num_envs, num_dofs)."""
        return self._dof_power

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Assign a motor spec to each DOF, and initialize the power buffers.
        """
        names = self.actuator_manager.dofs_names
        specs = assign_by_pattern(names, self._motors_cfg)
        missing = [
            name for name, spec in zip(names, specs, strict=True) if spec is None
        ]
        if missing:
            raise ValueError(f"PowerMonitor has no motor spec for the DOFs: {missing}")
        motors = [spec for spec in specs if spec is not None]

        self._torque_constant = torch.tensor(
            [motor.torque_constant for motor in motors],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self._phase_resistance = torch.tensor(
            [motor.phase_resistance for motor in motors],
            device=gs.device,
            dtype=gs.tc_float,
        )
        per_dof_shape = (self.env.num_envs, len(names))
        self._dof_copper_loss = torch.zeros(
            per_dof_shape, device=gs.device, dtype=gs.tc_float
        )
        self._dof_mechanical_power = torch.zeros(
            per_dof_shape, device=gs.device, dtype=gs.tc_float
        )
        self._dof_power = torch.zeros(
            per_dof_shape, device=gs.device, dtype=gs.tc_float
        )
        super().build()

    def update(self):
        """
        Estimate each motor's power, then calculate the monitor items.
        """
        torque = self._read_dof_torque().float()
        velocity = self.actuator_manager.get_dofs_velocity().float()

        # Copper loss for the phase current, with amplitude-invariant dq currents
        current = torque / self._torque_constant
        self._dof_copper_loss[:] = 1.5 * self._phase_resistance * current.square()
        self._dof_mechanical_power[:] = (torque * velocity).clamp(min=0.0)
        torch.add(
            self._dof_copper_loss, self._dof_mechanical_power, out=self._dof_power
        )
        super().update()

    """
    Monitor items
    """

    def _copper_loss_w(self, env: GenesisEnv) -> torch.Tensor:
        """Winding heat for the whole robot, in watts."""
        return self.dof_copper_loss.sum(dim=-1)

    def _mechanical_w(self, env: GenesisEnv) -> torch.Tensor:
        """Positive mechanical power for the whole robot, in watts."""
        return self.dof_mechanical_power.sum(dim=-1)

    def _total_w(self, env: GenesisEnv) -> torch.Tensor:
        """Copper loss plus mechanical power for the whole robot, in watts."""
        return self.dof_power.sum(dim=-1)

    def _motor_w(self, env: GenesisEnv) -> torch.Tensor:
        """The average power per motor in each DOF group, in watts."""
        return self.by_group(self.dof_power, "mean")

    def _bus_current_a(self, env: GenesisEnv) -> torch.Tensor:
        """The battery current, in amps."""
        assert self.supply_voltage is not None
        return self["total_w"] / self.supply_voltage
