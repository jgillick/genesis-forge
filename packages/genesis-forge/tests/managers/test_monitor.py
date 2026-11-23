"""
Behavior of Monitor and its subclasses: calculating values each step, logging them
combined across environments, and making them available by name.

Uses a FakeEnv and a fake actuator manager -- no Genesis scene is built.
"""

from dataclasses import dataclass

import pytest
import torch

from genesis_forge.managers import (
    ActuatorManager,
    ActuatorMonitor,
    MdpFn,
    Monitor,
    MotorSpec,
    PowerMonitor,
)


def logged(env):
    return env.extras[env.extras_logging_key]


"""
Monitor -- values
"""


def test_step_calculates_each_value_and_makes_it_available_by_name(env):
    mon = Monitor(env, cfg={"height": {"fn": lambda env: torch.arange(4.0)}})
    mon.build()
    mon.step()

    assert torch.equal(mon["height"], torch.arange(4.0))
    assert "height" in mon
    assert list(mon) == ["height"]


def test_items_can_read_earlier_items_in_the_same_step(env):
    mon = Monitor(
        env,
        cfg={
            "a": {"fn": lambda env: torch.full((4,), 2.0)},
            "b": {"fn": lambda env: mon["a"] * 3},
        },
    )
    mon.build()
    mon.step()

    assert torch.equal(mon["b"], torch.full((4,), 6.0))


def test_reading_a_value_before_it_is_calculated_raises(env):
    mon = Monitor(env, cfg={"height": {"fn": lambda env: torch.zeros(4)}})
    mon.build()

    with pytest.raises(RuntimeError, match="not been calculated yet"):
        mon["height"]


def test_reading_an_unknown_item_raises(env):
    mon = Monitor(env, cfg={})
    with pytest.raises(KeyError, match="no item 'nope'"):
        mon["nope"]


def test_get_returns_a_labeled_column(env):
    mon = Monitor(
        env,
        cfg={
            "vel": {
                "fn": lambda env: torch.tensor([[1.0, 2.0]] * 4),
                "labels": ["x", "y"],
            }
        },
    )
    mon.build()
    mon.step()

    assert torch.equal(mon.get("vel", "y"), torch.full((4,), 2.0))
    with pytest.raises(KeyError, match="no column 'z'"):
        mon.get("vel", "z")


def test_values_are_calculated_even_with_logging_disabled(env):
    mon = Monitor(
        env, cfg={"a": {"fn": lambda env: torch.ones(4)}}, logging_enabled=False
    )
    mon.build()
    mon.step()

    assert torch.equal(mon["a"], torch.ones(4))
    assert logged(env) == {}


def test_a_disabled_monitor_does_not_calculate(env):
    calls = []
    mon = Monitor(env, cfg={"a": {"fn": lambda env: calls.append(1) or torch.ones(4)}})
    mon.build()
    mon.enabled = False
    mon.step()

    assert calls == []


"""
Monitor -- shapes and labels
"""


def test_a_2d_value_without_labels_raises(env):
    mon = Monitor(env, cfg={"vel": {"fn": lambda env: torch.zeros(4, 3)}})
    mon.build()

    with pytest.raises(ValueError, match="needs a label"):
        mon.step()


def test_a_1d_value_with_labels_raises(env):
    mon = Monitor(
        env, cfg={"height": {"fn": lambda env: torch.zeros(4), "labels": ["x"]}}
    )
    mon.build()

    with pytest.raises(ValueError, match="can't have labels"):
        mon.step()


def test_a_2d_value_with_the_wrong_number_of_labels_raises(env):
    mon = Monitor(
        env, cfg={"vel": {"fn": lambda env: torch.zeros(4, 3), "labels": ["x"]}}
    )
    mon.build()

    with pytest.raises(ValueError, match="3 columns"):
        mon.step()


def test_a_value_with_the_wrong_number_of_envs_raises(env):
    mon = Monitor(env, cfg={"a": {"fn": lambda env: torch.zeros(2)}})
    mon.build()

    with pytest.raises(ValueError, match=r"Expected a single number, \(4,\)"):
        mon.step()


def test_callable_labels_are_resolved_at_build(env):
    names = []
    mon = Monitor(
        env,
        cfg={"vel": {"fn": lambda env: torch.zeros(4, 2), "labels": lambda: names}},
    )
    names.extend(["x", "y"])
    mon.build()

    assert mon.labels("vel") == ["x", "y"]


def test_an_mdp_fn_can_provide_its_own_labels(env):
    @dataclass(kw_only=True, eq=False)
    class per_foot(MdpFn):
        def build(self):
            self.labels = ["left", "right"]

        def __call__(self, env):
            return torch.zeros(env.num_envs, 2)

    mon = Monitor(env, cfg={"air_time": {"fn": per_foot()}})
    mon.build()
    mon.step()

    assert mon.labels("air_time") == ["left", "right"]


def test_mdp_fn_labels_stay_current_when_the_fn_rebuilds(env):
    @dataclass(kw_only=True, eq=False)
    class per_foot(MdpFn):
        feet: int = 2

        def build(self):
            self.labels = [f"foot{i}" for i in range(self.feet)]

        def __call__(self, env):
            return torch.zeros(env.num_envs, self.feet)

    fn = per_foot()
    mon = Monitor(env, cfg={"air_time": {"fn": fn}})
    mon.build()
    fn.feet = 3  # <~ a curriculum step rebuilds the fn
    mon.step()

    assert mon.labels("air_time") == ["foot0", "foot1", "foot2"]


"""
Monitor -- logging
"""


def test_logs_the_mean_by_default(env):
    mon = Monitor(env, cfg={"a": {"fn": lambda env: torch.arange(4.0)}})
    mon.build()
    mon.step()

    assert logged(env) == {"Monitor / a": pytest.approx(1.5)}


def test_the_mean_is_unsuffixed_and_other_reductions_are_suffixed(env):
    mon = Monitor(
        env,
        logging_tag="Body",
        cfg={
            "a": {
                "fn": lambda env: torch.tensor([3.0, 4.0, 0.0, 0.0]),
                "reduce": ["rms", "mean", "p25"],
            }
        },
    )
    mon.build()
    mon.step()

    assert logged(env) == {
        "Body / a_rms": pytest.approx(2.5),  # sqrt((9 + 16) / 4)
        "Body / a": pytest.approx(1.75),
        "Body / a_p25": pytest.approx(0.0),
    }


def test_labeled_columns_are_logged_with_their_label(env):
    mon = Monitor(
        env,
        cfg={
            "vel": {
                "fn": lambda env: torch.tensor([[1.0, -1.0]] * 4),
                "labels": ["x", "y"],
                "reduce": ["mean", "min"],
            }
        },
    )
    mon.build()
    mon.step()

    assert logged(env) == {
        "Monitor / vel_x": pytest.approx(1.0),
        "Monitor / vel_y": pytest.approx(-1.0),
        "Monitor / vel_x_min": pytest.approx(1.0),
        "Monitor / vel_y_min": pytest.approx(-1.0),
    }


def test_log_false_calculates_without_logging(env):
    mon = Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4), "log": False}})
    mon.build()
    mon.step()

    assert torch.equal(mon["a"], torch.ones(4))
    assert logged(env) == {}


@pytest.mark.parametrize("value", [3, 3.0, torch.tensor(3.0)])
def test_a_single_number_is_logged_as_is(env, value):
    mon = Monitor(
        env,
        logging_tag="Curriculum",
        cfg={"level": {"fn": lambda env: value, "reduce": ["mean", "p99"]}},
    )
    mon.build()
    mon.step()

    assert isinstance(mon["level"], torch.Tensor)
    assert mon["level"] == 3
    assert list(logged(env)) == ["Curriculum / level"]
    assert logged(env)["Curriculum / level"] == 3


def test_a_single_number_with_log_false_is_not_logged(env):
    mon = Monitor(env, cfg={"level": {"fn": lambda env: 3, "log": False}})
    mon.build()
    mon.step()

    assert mon["level"] == 3
    assert logged(env) == {}


def test_an_empty_reduce_list_raises(env):
    with pytest.raises(ValueError, match="'log': False"):
        Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4), "reduce": []}})


def test_a_single_number_with_labels_raises(env):
    mon = Monitor(env, cfg={"level": {"fn": lambda env: 3, "labels": ["a"]}})
    mon.build()

    with pytest.raises(ValueError, match="single number"):
        mon.step()


@pytest.mark.parametrize("reduction", ["median", "p101", "p"])
def test_an_unknown_reduction_raises(env, reduction):
    with pytest.raises(ValueError, match="Unknown monitor reduction"):
        Monitor(
            env, cfg={"a": {"fn": lambda env: torch.ones(4), "reduce": [reduction]}}
        )


def test_a_single_reduction_can_be_given_without_a_list(env):
    mon = Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4), "reduce": "max"}})
    mon.build()
    mon.step()

    assert logged(env) == {"Monitor / a_max": pytest.approx(1.0)}


def test_update_calculates_values_without_logging_them(env):
    mon = Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4)}})
    mon.build()

    mon.update()

    assert torch.equal(mon["a"], torch.ones(4))
    assert logged(env) == {}


def test_two_monitors_logging_the_same_name_raise(env):
    first = Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4)}})
    second = Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4)}})
    env.managers = {"monitor": [first, second]}
    first.build()

    with pytest.raises(ValueError, match="different logging_tag"):
        second.build()


def test_two_monitors_with_different_tags_can_share_item_names(env):
    first = Monitor(
        env, logging_tag="Left", cfg={"a": {"fn": lambda env: torch.ones(4)}}
    )
    second = Monitor(
        env, logging_tag="Right", cfg={"a": {"fn": lambda env: torch.ones(4)}}
    )
    env.managers = {"monitor": [first, second]}
    first.build()
    second.build()


"""
Monitor -- reset
"""


def test_reset_resets_mdp_fn_items(env):
    @dataclass(kw_only=True, eq=False)
    class counter(MdpFn):
        def build(self):
            self.count = torch.zeros(self.env.num_envs)

        def reset(self, envs_idx):
            self.count[envs_idx] = 0.0

        def __call__(self, env):
            self.count += 1
            return self.count.clone()

    mon = Monitor(env, cfg={"steps": {"fn": counter()}})
    mon.build()
    mon.step()
    mon.step()
    mon.reset(torch.tensor([1]))

    assert mon["steps"].tolist() == [3.0, 1.0, 3.0, 3.0]


def test_reset_recalculates_values_without_logging_them(env):
    mon = Monitor(env, cfg={"a": {"fn": lambda env: torch.ones(4)}})
    mon.build()

    mon.reset()

    assert torch.equal(mon["a"], torch.ones(4))
    assert logged(env) == {}


def test_a_disabled_monitor_does_not_calculate_on_reset(env):
    calls = []
    mon = Monitor(env, cfg={"a": {"fn": lambda env: calls.append(1) or torch.ones(4)}})
    mon.build()
    mon.enabled = False
    mon.reset()

    assert calls == []


"""
PowerMonitor and ActuatorMonitor
"""


class FakeActuatorManager:
    """The DOF accessors the monitors read, with resolve_dof_groups borrowed from the
    real ActuatorManager."""

    resolve_dof_groups = ActuatorManager.resolve_dof_groups

    def __init__(self, names, torque, velocity=None):
        # Robot DOF indices start at 6, after a free base joint, so they differ from columns
        self._dofs = {name: 6 + i for i, name in enumerate(names)}
        self.dofs_idx = list(self._dofs.values())
        self.dofs_names = names
        self.num_dofs = len(names)
        self.torque = torque
        self.velocity = velocity if velocity is not None else torch.zeros_like(torque)
        self.force_upper = torch.full((len(names),), 10.0)
        self.force_lower = -self.force_upper

    def get_dofs_control_force(self, clip_to_max_force=False):
        return self.torque.clone()

    def get_dofs_velocity(self):
        return self.velocity

    def get_dofs_force_range(self):
        return self.force_lower, self.force_upper


NAMES = ["L_Hip", "L_Knee", "R_Hip", "R_Knee"]
GROUPS = {"hip": ".*_Hip", "knee": ".*_Knee"}
MOTOR = MotorSpec(torque_constant=0.5, phase_resistance=0.4)


def test_power_monitor_estimates_copper_loss_and_positive_mechanical_power(env):
    torque = torch.tensor([[1.0, -2.0, 0.0, 1.0]] * 4)
    velocity = torch.tensor([[3.0, 1.0, 5.0, -1.0]] * 4)
    mon = PowerMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torque, velocity),
        motors=MOTOR,
    )
    mon.build()
    mon.step()

    # copper = 1.5 * 0.4 * (torque / 0.5)^2 = 2.4 * torque^2
    assert torch.allclose(mon.dof_copper_loss[0], torch.tensor([2.4, 9.6, 0.0, 2.4]))
    # Only positive work counts: [3, -2, 0, -1] -> [3, 0, 0, 0]
    assert torch.allclose(mon.dof_mechanical_power[0], torch.tensor([3.0, 0, 0, 0]))
    assert torch.allclose(mon["total_w"], torch.full((4,), 14.4 + 3.0))


def test_reset_environments_read_zero_torque_until_their_next_step(env):
    torque = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4)
    mon = ActuatorMonitor(env, actuator_manager=FakeActuatorManager(NAMES, torque))
    mon.build()
    mon.step()

    mon.reset(torch.tensor([1]))
    # The reset environment hasn't applied any force yet in its new episode
    assert torch.equal(mon.dof_torque[1], torch.zeros(4))
    assert torch.equal(mon.dof_torque[0], torque[0])

    mon.step()
    assert torch.equal(mon.dof_torque[1], torque[1])


def test_a_second_reset_in_a_step_keeps_earlier_reset_environments_at_zero(env):
    torque = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 4)
    mon = ActuatorMonitor(env, actuator_manager=FakeActuatorManager(NAMES, torque))
    mon.build()
    mon.step()

    mon.reset(torch.tensor([1]))
    mon.reset(torch.tensor([2]))

    assert torch.equal(mon.dof_torque[1], torch.zeros(4))
    assert torch.equal(mon.dof_torque[2], torch.zeros(4))


def test_power_monitor_reports_bus_current_only_with_a_supply_voltage(env):
    torque = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4)
    actuator = FakeActuatorManager(NAMES, torque)

    without = PowerMonitor(env, actuator_manager=actuator, motors=MOTOR)
    assert "bus_current_a" not in without

    mon = PowerMonitor(
        env, actuator_manager=actuator, motors=MOTOR, supply_voltage=24.0
    )
    mon.build()
    mon.step()
    assert torch.allclose(mon["bus_current_a"], torch.full((4,), 2.4 / 24.0))


def test_power_monitor_reports_average_motor_power_per_group(env):
    torque = torch.tensor([[1.0, 0.0, 1.0, 0.0]] * 4)
    mon = PowerMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torque),
        motors=MOTOR,
        dof_groups=GROUPS,
    )
    mon.build()
    mon.step()

    assert mon.labels("motor_w") == ["hip", "knee"]
    assert torch.allclose(mon.get("motor_w", "hip"), torch.full((4,), 2.4))
    assert torch.allclose(mon.get("motor_w", "knee"), torch.zeros(4))
    assert "Power / motor_w_hip" in logged(env)
    assert "Power / total_w_p99" in logged(env)


def test_power_monitor_assigns_motor_specs_by_joint_pattern(env):
    torque = torch.ones(4, 4)
    big = MotorSpec(torque_constant=1.0, phase_resistance=0.4)
    mon = PowerMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torque),
        motors={".*_Hip": big, ".*_Knee": MOTOR},
    )
    mon.build()
    mon.step()

    assert torch.allclose(mon.dof_copper_loss[0], torch.tensor([0.6, 2.4, 0.6, 2.4]))


def test_power_monitor_raises_for_dofs_without_a_motor_spec(env):
    mon = PowerMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torch.zeros(4, 4)),
        motors={".*_Hip": MOTOR},
    )
    with pytest.raises(ValueError, match="L_Knee"):
        mon.build()


def test_power_monitor_accepts_extra_items(env):
    mon = PowerMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torch.ones(4, 4)),
        motors=MOTOR,
        cfg={"double_w": {"fn": lambda env: mon["total_w"] * 2}},
    )
    mon.build()
    mon.step()

    assert torch.allclose(mon["double_w"], mon["total_w"] * 2)


def test_actuator_monitor_reports_torque_per_group(env):
    torque = torch.tensor([[3.0, 1.0, -4.0, 1.0]] * 4)
    mon = ActuatorMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torque),
        dof_groups=GROUPS,
    )
    mon.build()
    mon.step()

    # hip torques are [3, -4]
    assert torch.allclose(mon.get("torque", "hip"), torch.full((4,), 12.5**0.5))
    assert torch.allclose(mon.get("torque_peak", "hip"), torch.full((4,), 4.0))
    assert logged(env)["Actuators / torque_hip_rms"] == pytest.approx(12.5**0.5)


def test_actuator_monitor_without_groups_reports_all_dofs_together(env):
    torque = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 4)
    mon = ActuatorMonitor(env, actuator_manager=FakeActuatorManager(NAMES, torque))
    mon.build()
    mon.step()

    assert mon.labels("torque") is None
    assert torch.allclose(mon["torque"], torch.ones(4))
    assert "Actuators / torque_rms" in logged(env)


def test_actuator_monitor_saturation_uses_each_envs_force_limit(env):
    torque = torch.tensor([[9.6, 9.0, 5.0, 5.0]] * 4)
    actuator_manager = FakeActuatorManager(NAMES, torque)
    actuator_manager.force_upper = torch.full((4, 4), 10.0)
    actuator_manager.force_upper[0, 2] = (
        5.0  # this env's max force was randomized lower
    )
    mon = ActuatorMonitor(env, actuator_manager=actuator_manager)
    mon.build()
    mon.step()

    assert mon.dof_saturated[0].tolist() == [True, False, True, False]
    assert mon.dof_saturated[1].tolist() == [True, False, False, False]


def test_actuator_monitor_reads_new_force_limits_on_reset(env):
    torque = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * 4)
    actuator_manager = FakeActuatorManager(NAMES, torque)
    mon = ActuatorMonitor(env, actuator_manager=actuator_manager)
    mon.build()
    actuator_manager.force_upper = torch.full((4,), 5.0)  # <~ randomized at reset
    mon.reset(torch.tensor([1]))
    mon.step()

    assert mon.dof_saturated[0, 0]


def test_actuator_monitor_saturation_checks_each_direction_of_the_force_range(env):
    torque = torch.tensor([[-3.9, -3.9, 0.0, 0.0]] * 4)
    actuator_manager = FakeActuatorManager(NAMES, torque)
    actuator_manager.force_lower = torch.tensor([-4.0, -10.0, -10.0, -10.0])
    mon = ActuatorMonitor(env, actuator_manager=actuator_manager)
    mon.build()
    mon.step()

    assert mon.dof_saturated[0].tolist() == [True, False, False, False]


def test_actuator_monitor_reports_the_saturated_fraction_per_group(env):
    torque = torch.tensor([[6.0, 0.0, 0.0, 0.0]] * 4)
    mon = ActuatorMonitor(
        env,
        actuator_manager=FakeActuatorManager(NAMES, torque),
        dof_groups=GROUPS,
        saturation_threshold=0.5,
    )
    mon.build()
    mon.step()

    # One of the two hip joints is at 50% of its 10 N·m limit
    assert torch.allclose(mon.get("saturation_frac", "hip"), torch.full((4,), 0.5))
    assert torch.allclose(mon.get("saturation_frac", "knee"), torch.zeros(4))
