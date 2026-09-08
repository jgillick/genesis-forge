"""Action decoding: affine transforms, multi-manager composition, and safety guards.

These pin the observable decode behavior. U5's parity harness proves the numbers
match the live torch `process_actions` path.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis_forge_runtime import (
    ActionDecoder,
    ActionManagerSpec,
    AffineDecoder,
    DecoderError,
    ManagerDecoder,
)
from genesis_forge_runtime.decoders import resolve_decoder_class


def position_spec(
    name: str = "action_manager",
    *,
    start: int = 0,
    joints: tuple[str, ...] = ("hip", "knee", "ankle"),
    scale=(0.5, 0.5, 0.5),
    offset=(0.0, 1.0, -1.0),
    clip_low=(-10.0, -10.0, -10.0),
    clip_high=(10.0, 10.0, 10.0),
) -> ActionManagerSpec:
    """A `position`-style manager: affine, then clipped to joint limits."""
    return ActionManagerSpec(
        name=name,
        deploy_type="position",
        joint_names=joints,
        slice_start=start,
        slice_end=start + len(joints),
        config={
            "scale": np.asarray(scale, dtype=np.float32),
            "offset": np.asarray(offset, dtype=np.float32),
            "clip_low": np.asarray(clip_low, dtype=np.float32),
            "clip_high": np.asarray(clip_high, dtype=np.float32),
        },
    )


def within_limits_spec(
    name: str = "action_manager",
    *,
    start: int = 0,
    joints: tuple[str, ...] = ("hip", "knee"),
) -> ActionManagerSpec:
    """A `position_within_limits` manager: pre-clip to [-1, 1], then map to limits."""
    return ActionManagerSpec(
        name=name,
        deploy_type="position_within_limits",
        joint_names=joints,
        slice_start=start,
        slice_end=start + len(joints),
        config={
            "raw_action_clip": [-1.0, 1.0],
            "scale": np.asarray([2.0, 4.0], dtype=np.float32),
            "offset": np.asarray([0.0, 1.0], dtype=np.float32),
        },
    )


def velocity_spec(
    name: str = "wheels",
    *,
    start: int = 0,
    joints: tuple[str, ...] = ("left_wheel", "right_wheel"),
    scale=(8.0, 8.0),
) -> ActionManagerSpec:
    """A `velocity` manager: affine, and unbounded unless a clip was configured."""
    return ActionManagerSpec(
        name=name,
        deploy_type="velocity",
        joint_names=joints,
        slice_start=start,
        slice_end=start + len(joints),
        config={
            "scale": np.asarray(scale, dtype=np.float32),
            "offset": np.zeros(len(joints), dtype=np.float32),
        },
    )


"""Affine decoding (position)"""


def test_affine_decode_matches_scale_offset_and_clip():
    decoder = ActionDecoder((position_spec(),))

    result = decoder.decode([2.0, 2.0, 2.0])

    # 2 * 0.5 + offset -> [1, 2, 0], all inside the clip range.
    np.testing.assert_allclose(result.targets, [1.0, 2.0, 0.0])


def test_decoded_targets_are_named_by_joint():
    decoder = ActionDecoder((position_spec(),))

    result = decoder.decode([2.0, 2.0, 2.0])

    assert result.by_joint == {"hip": 1.0, "knee": 2.0, "ankle": 0.0}
    assert result.joint_names == ("hip", "knee", "ankle")


def test_clip_bounds_are_honored():
    decoder = ActionDecoder(
        (
            position_spec(
                scale=(1.0, 1.0, 1.0),
                offset=(0.0, 0.0, 0.0),
                clip_low=(-1.0, -1.0, -1.0),
                clip_high=(1.0, 1.0, 1.0),
            ),
        )
    )

    result = decoder.decode([5.0, -5.0, 0.25])

    np.testing.assert_allclose(result.targets, [1.0, -1.0, 0.25])


def test_values_exactly_on_the_clip_boundary_pass_through():
    decoder = ActionDecoder(
        (
            position_spec(
                scale=(1.0, 1.0, 1.0),
                offset=(0.0, 0.0, 0.0),
                clip_low=(-1.0, -1.0, -1.0),
                clip_high=(1.0, 1.0, 1.0),
            ),
        )
    )

    result = decoder.decode([1.0, -1.0, 0.0])

    np.testing.assert_allclose(result.targets, [1.0, -1.0, 0.0])


def test_scalar_decode_parameters_broadcast_across_joints():
    spec = ActionManagerSpec(
        name="broadcast",
        deploy_type="position",
        joint_names=("a", "b", "c"),
        slice_start=0,
        slice_end=3,
        config={"scale": 2.0, "offset": 1.0},
    )

    result = ActionDecoder((spec,)).decode([1.0, 2.0, 3.0])

    np.testing.assert_allclose(result.targets, [3.0, 5.0, 7.0])


def test_output_is_float32():
    result = ActionDecoder((position_spec(),)).decode([1, 1, 1])

    assert result.targets.dtype == np.float32


"""Within-limits decoding"""


def test_within_limits_clips_the_raw_action_before_scaling():
    decoder = ActionDecoder((within_limits_spec(),))

    # Inputs beyond +/-1 are clamped first, so both joints hit their limit value.
    result = decoder.decode([5.0, -5.0])

    # hip: clip(5)=1 -> 1*2 + 0 = 2 ; knee: clip(-5)=-1 -> -1*4 + 1 = -3
    np.testing.assert_allclose(result.targets, [2.0, -3.0])


def test_within_limits_maps_midpoint_to_the_offset():
    decoder = ActionDecoder((within_limits_spec(),))

    result = decoder.decode([0.0, 0.0])

    np.testing.assert_allclose(result.targets, [0.0, 1.0])


def test_within_limits_has_no_output_clip():
    """The training-side manager applies no post-clip, so neither may the decoder."""
    spec = ActionManagerSpec(
        name="unbounded",
        deploy_type="position_within_limits",
        joint_names=("a",),
        slice_start=0,
        slice_end=1,
        config={"raw_action_clip": [-1.0, 1.0], "scale": 100.0, "offset": 50.0},
    )

    result = ActionDecoder((spec,)).decode([1.0])

    np.testing.assert_allclose(result.targets, [150.0])


"""Multi-manager composition"""


def test_each_manager_decodes_only_its_own_slice():
    legs = position_spec("legs", start=0, joints=("hip", "knee", "ankle"))
    arm = within_limits_spec("arm", start=3)
    decoder = ActionDecoder((legs, arm))

    result = decoder.decode([2.0, 2.0, 2.0, 0.0, 0.0])

    assert decoder.num_actions == 5
    np.testing.assert_allclose(result.by_manager["legs"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(result.by_manager["arm"], [0.0, 1.0])


def test_joint_names_concatenate_in_slice_order():
    legs = position_spec("legs", start=0, joints=("hip", "knee", "ankle"))
    arm = within_limits_spec("arm", start=3)

    # Registered out of order -- the decoder must sort by slice, not argument order.
    decoder = ActionDecoder((arm, legs))

    assert decoder.joint_names == ("hip", "knee", "ankle", "hip", "knee")


def test_composed_targets_concatenate_in_slice_order():
    legs = position_spec("legs", start=0, joints=("hip", "knee", "ankle"))
    arm = within_limits_spec("arm", start=3)

    result = ActionDecoder((legs, arm)).decode([2.0, 2.0, 2.0, 0.0, 0.0])

    np.testing.assert_allclose(result.targets, [1.0, 2.0, 0.0, 0.0, 1.0])


"""Safety guards"""


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_policy_output_is_refused(bad):
    decoder = ActionDecoder((position_spec(),))

    with pytest.raises(DecoderError) as error:
        decoder.decode([1.0, bad, 1.0])

    message = str(error.value)
    assert "non-finite" in message
    assert "1" in message  # names the offending index


def test_non_finite_output_can_be_allowed_for_debugging():
    decoder = ActionDecoder((position_spec(),), check_finite=False)

    result = decoder.decode([1.0, np.nan, 1.0])

    assert np.isnan(result.targets[1])


def test_wrong_action_count_is_reported():
    decoder = ActionDecoder((position_spec(),))

    with pytest.raises(DecoderError) as error:
        decoder.decode([1.0, 2.0])

    assert "3" in str(error.value)


def test_a_leading_batch_dimension_is_accepted():
    decoder = ActionDecoder((position_spec(),))

    result = decoder.decode(np.array([[2.0, 2.0, 2.0]]))

    np.testing.assert_allclose(result.targets, [1.0, 2.0, 0.0])


"""Decoder resolution"""


def test_builtin_types_resolve_without_an_import_path():
    for spec in (position_spec(), within_limits_spec(), velocity_spec()):
        assert resolve_decoder_class(spec) is AffineDecoder
        assert spec.decoder_import_path is None


def test_every_builtin_action_manager_type_has_a_decoder():
    """Guard: a new affine manager type upstream must not ship undeployable."""
    from genesis_forge_runtime.decoders import BUILTIN_DECODERS

    assert {"affine_dof", "position", "position_within_limits", "velocity"} <= set(
        BUILTIN_DECODERS
    )


def test_unknown_type_without_an_import_path_names_the_type_and_the_fix():
    spec = ActionManagerSpec(
        name="impedance_manager",
        deploy_type="cartesian_impedance",
        joint_names=("a",),
        slice_start=0,
        slice_end=1,
        config={},
    )

    with pytest.raises(DecoderError) as error:
        resolve_decoder_class(spec)

    message = str(error.value)
    assert "cartesian_impedance" in message
    assert "impedance_manager" in message
    assert "import path" in message
    # The message lists what *is* available, so the reader knows their options.
    assert "position" in message and "velocity" in message


def test_malformed_import_path_is_reported():
    spec = ActionManagerSpec(
        name="custom",
        deploy_type="custom",
        joint_names=("a",),
        slice_start=0,
        slice_end=1,
        config={},
        decoder_import_path="no_colon_here",
    )

    with pytest.raises(DecoderError) as error:
        resolve_decoder_class(spec)

    assert "module.path:ClassName" in str(error.value)


def test_unimportable_module_is_reported_with_the_module_name():
    spec = ActionManagerSpec(
        name="custom",
        deploy_type="custom",
        joint_names=("a",),
        slice_start=0,
        slice_end=1,
        config={},
        decoder_import_path="definitely_not_installed_pkg:Decoder",
    )

    with pytest.raises(DecoderError) as error:
        resolve_decoder_class(spec)

    assert "definitely_not_installed_pkg" in str(error.value)


def test_a_class_that_is_not_a_manager_decoder_is_rejected(tmp_path, monkeypatch):
    module = tmp_path / "bogus_decoder_pkg.py"
    module.write_text("class NotADecoder:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    spec = ActionManagerSpec(
        name="custom",
        deploy_type="custom",
        joint_names=("a",),
        slice_start=0,
        slice_end=1,
        config={},
        decoder_import_path="bogus_decoder_pkg:NotADecoder",
    )

    with pytest.raises(DecoderError) as error:
        resolve_decoder_class(spec)

    assert "ManagerDecoder" in str(error.value)


"""Custom decoders (F3) and stateful decode (R12)"""


CUSTOM_DECODER_MODULE = '''
import numpy as np
from genesis_forge_runtime import ManagerDecoder


class DoublingDecoder(ManagerDecoder):
    """Stand-in for a third-party action manager's own decoder."""

    def decode(self, actions):
        return np.asarray(actions, dtype=np.float32) * 2.0


class RunningSumDecoder(ManagerDecoder):
    """Carries per-step state, which the contract must support (R12)."""

    def reset(self):
        self.total = None

    def decode(self, actions):
        values = np.asarray(actions, dtype=np.float32)
        self.total = values if self.total is None else self.total + values
        return self.total
'''


@pytest.fixture
def custom_decoders(tmp_path, monkeypatch):
    module = tmp_path / "third_party_decoders.py"
    module.write_text(CUSTOM_DECODER_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "third_party_decoders"


def custom_spec(
    module: str, class_name: str, name: str = "custom"
) -> ActionManagerSpec:
    return ActionManagerSpec(
        name=name,
        deploy_type="third_party_type",
        joint_names=("a", "b"),
        slice_start=0,
        slice_end=2,
        config={},
        decoder_import_path=f"{module}:{class_name}",
    )


def test_a_custom_decoder_loads_through_its_import_path(custom_decoders):
    decoder = ActionDecoder((custom_spec(custom_decoders, "DoublingDecoder"),))

    result = decoder.decode([1.5, -2.0])

    np.testing.assert_allclose(result.targets, [3.0, -4.0])
    assert result.by_joint == {"a": 3.0, "b": -4.0}


def test_a_stateful_custom_decoder_keeps_state_across_ticks(custom_decoders):
    """R12: a decoder may carry state across ticks."""
    decoder = ActionDecoder((custom_spec(custom_decoders, "RunningSumDecoder"),))

    np.testing.assert_allclose(decoder.decode([1.0, 1.0]).targets, [1.0, 1.0])
    np.testing.assert_allclose(decoder.decode([2.0, 2.0]).targets, [3.0, 3.0])
    np.testing.assert_allclose(decoder.decode([3.0, 3.0]).targets, [6.0, 6.0])


def test_resetting_clears_custom_decoder_state(custom_decoders):
    decoder = ActionDecoder((custom_spec(custom_decoders, "RunningSumDecoder"),))
    decoder.decode([5.0, 5.0])

    decoder.reset()

    np.testing.assert_allclose(decoder.decode([1.0, 1.0]).targets, [1.0, 1.0])


def test_a_decoder_without_decode_raises_a_clear_error():
    class Incomplete(ManagerDecoder):
        pass

    spec = position_spec()

    with pytest.raises(NotImplementedError) as error:
        Incomplete(spec).decode(np.zeros(3, dtype=np.float32))

    assert "Incomplete" in str(error.value)


"""Velocity decoding"""


def test_velocity_decode_is_unbounded_by_default():
    """VelocityActionManager defaults to no clip, so the decoder must not invent one."""
    decoder = ActionDecoder((velocity_spec(),))

    result = decoder.decode([100.0, -100.0])

    np.testing.assert_allclose(result.targets, [800.0, -800.0])


def test_velocity_targets_are_named_by_wheel():
    result = ActionDecoder((velocity_spec(),)).decode([1.0, 0.5])

    assert result.by_joint == {"left_wheel": 8.0, "right_wheel": 4.0}


def test_a_configured_velocity_clip_is_honored():
    spec = velocity_spec()
    spec.config["clip_low"] = np.asarray([-16.0, -16.0], dtype=np.float32)
    spec.config["clip_high"] = np.asarray([16.0, 16.0], dtype=np.float32)

    result = ActionDecoder((spec,)).decode([100.0, -100.0])

    np.testing.assert_allclose(result.targets, [16.0, -16.0])


"""Feeding the previous output back (R15)

The decoder remembers its last output so the caller can hand it to the next
observation, replacing the older implicit auto-fill.
"""


def test_remembered_outputs_start_at_zero():
    """Before the first decode, matching how training starts an episode."""
    decoder = ActionDecoder((position_spec(),))

    np.testing.assert_allclose(decoder.last_raw_actions, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(decoder.last_target_actions, [0.0, 0.0, 0.0])


def test_the_decoder_remembers_raw_and_target_actions_separately():
    """These differ, and feeding back the wrong one is the classic silent bug."""
    decoder = ActionDecoder((position_spec(),))

    decoder.decode([2.0, 2.0, 2.0])

    np.testing.assert_allclose(decoder.last_raw_actions, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(decoder.last_target_actions, [1.0, 2.0, 0.0])


def test_remembered_outputs_update_every_tick():
    decoder = ActionDecoder((position_spec(),))

    decoder.decode([2.0, 2.0, 2.0])
    decoder.decode([0.0, 0.0, 0.0])

    np.testing.assert_allclose(decoder.last_raw_actions, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(decoder.last_target_actions, [0.0, 1.0, -1.0])


def test_reset_clears_the_remembered_outputs():
    decoder = ActionDecoder((position_spec(),))
    decoder.decode([2.0, 2.0, 2.0])

    decoder.reset()

    np.testing.assert_allclose(decoder.last_raw_actions, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(decoder.last_target_actions, [0.0, 0.0, 0.0])


def test_remembered_outputs_are_copies_not_live_buffers():
    decoder = ActionDecoder((position_spec(),))
    decoder.decode([2.0, 2.0, 2.0])

    snapshot = decoder.last_target_actions
    decoder.decode([0.0, 0.0, 0.0])

    np.testing.assert_allclose(snapshot, [1.0, 2.0, 0.0])


def test_target_actions_are_also_available_per_manager():
    legs = position_spec("legs", start=0, joints=("hip", "knee", "ankle"))
    arm = within_limits_spec("arm", start=3)
    decoder = ActionDecoder((legs, arm))

    decoder.decode([2.0, 2.0, 2.0, 0.0, 0.0])

    by_manager = decoder.last_target_actions_by_manager
    np.testing.assert_allclose(by_manager["legs"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(by_manager["arm"], [0.0, 1.0])


def test_the_feedback_loop_reads_off_the_decoder():
    """The documented control-loop shape, end to end."""
    from genesis_forge_runtime import (
        ObservationAssembler,
        ObservationEntry,
        ObservationLayout,
    )

    layout = ObservationLayout(
        entries=(
            ObservationEntry(name="gyro", size=3),
            ObservationEntry(name="actions", size=3),
        )
    )
    assembler = ObservationAssembler(layout)
    decoder = ActionDecoder((position_spec(),))

    # Tick one: nothing decoded yet, so the feedback is zeros.
    first = assembler.assemble(
        {"gyro": [0.0, 0.0, 0.0], "actions": decoder.last_target_actions}
    )
    np.testing.assert_allclose(first, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    decoder.decode([2.0, 2.0, 2.0])

    # Tick two: the feedback carries the previous decode's targets.
    second = assembler.assemble(
        {"gyro": [0.0, 0.0, 0.0], "actions": decoder.last_target_actions}
    )
    np.testing.assert_allclose(second, [0.0, 0.0, 0.0, 1.0, 2.0, 0.0])


def test_the_decoder_exposes_every_property_the_docs_tell_users_to_read():
    """Guard the coupling between this class and the wiring instructions.

    The deployment guide and the examples tell people to pass
    ``action_decoder.last_raw_actions`` and friends into the assembler. Those are
    plain strings in prose now, so renaming a property here would send someone
    looking for an attribute that does not exist -- on a robot, while they are
    already unsure what to wire.
    """
    for attribute in (
        "last_raw_actions",
        "last_target_actions",
        "last_raw_actions_by_manager",
        "last_target_actions_by_manager",
    ):
        assert hasattr(ActionDecoder, attribute), (
            f"ActionDecoder.{attribute} is documented as the value to feed back, "
            f"but no longer exists."
        )


"""Reading a joint's clip range back off the decoder"""


def test_clip_range_by_joint_reports_the_bounds_the_decoder_applies():
    decoder = ActionDecoder(
        (position_spec(clip_low=(-1.0, -2.0, -3.0), clip_high=(1.0, 2.0, 3.0)),)
    )

    assert decoder.clip_range_by_joint == {
        "hip": (-1.0, 1.0),
        "knee": (-2.0, 2.0),
        "ankle": (-3.0, 3.0),
    }


def test_an_unbounded_side_reads_as_infinite():
    """The exporter omits a bound of infinity, so the decoder restores it."""
    spec = position_spec()
    del spec.config[
        "post_clip_high" if "post_clip_high" in spec.config else "clip_high"
    ]

    decoder = ActionDecoder((spec,))

    assert decoder.clip_range_by_joint["hip"] == (-10.0, float("inf"))


def test_a_joint_with_no_clip_at_all_is_absent():
    spec = position_spec()
    del spec.config["clip_low"]
    del spec.config["clip_high"]

    decoder = ActionDecoder((spec,))

    assert decoder.clip_range_by_joint == {}


def test_clip_ranges_merge_across_managers():
    decoder = ActionDecoder(
        (
            position_spec(
                name="front",
                joints=("hip",),
                scale=(1.0,),
                offset=(0.0,),
                clip_low=(-5.0,),
                clip_high=(5.0,),
                start=0,
            ),
            position_spec(
                name="rear",
                joints=("tail",),
                scale=(1.0,),
                offset=(0.0,),
                clip_low=(-9.0,),
                clip_high=(9.0,),
                start=1,
            ),
        )
    )

    assert decoder.clip_range_by_joint == {"hip": (-5.0, 5.0), "tail": (-9.0, 9.0)}
