"""Observation assembly: ordering, scaling, history, and auto-filled entries.

The assembler must reproduce ObservationManager's output exactly. These tests pin
the observable behavior; U5's parity harness proves the equivalence against the
live torch pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from genesis_forge_deploy import (
    SOURCE_PIPELINE_STATE,
    STAGE_PROCESSED_ACTIONS,
    STAGE_RAW_ACTIONS,
    ObservationEntry,
    ObservationLayout,
)
from genesis_forge_deploy.observations import ObservationAssembler, ObservationError


def layout(*entries: ObservationEntry, history_length: int = 1) -> ObservationLayout:
    return ObservationLayout(entries=tuple(entries), history_length=history_length)


def simple_layout(history_length: int = 1) -> ObservationLayout:
    return layout(
        ObservationEntry(name="gyro", size=3, scale=0.25),
        ObservationEntry(name="dof_pos", size=2, scale=2.0),
        history_length=history_length,
    )


"""Ordering and scaling"""


def test_entries_are_ordered_and_scaled_per_the_layout():
    assembler = ObservationAssembler(simple_layout())

    obs = assembler.assemble({"gyro": [4.0, 8.0, 12.0], "dof_pos": [1.0, 3.0]})

    # gyro scaled by 0.25, dof_pos by 2.0, concatenated in layout order.
    np.testing.assert_allclose(obs, [1.0, 2.0, 3.0, 2.0, 6.0])


def test_output_is_float32_and_correctly_sized():
    assembler = ObservationAssembler(simple_layout())

    obs = assembler.assemble({"gyro": [0, 0, 0], "dof_pos": [0, 0]})

    assert obs.dtype == np.float32
    assert obs.shape == (5,)
    assert assembler.output_size == 5


def test_a_scale_of_one_passes_values_through():
    assembler = ObservationAssembler(layout(ObservationEntry(name="raw", size=2)))

    np.testing.assert_allclose(assembler.assemble({"raw": [7.0, -7.0]}), [7.0, -7.0])


def test_assemble_does_not_mutate_the_callers_array():
    assembler = ObservationAssembler(layout(ObservationEntry(name="gyro", size=2, scale=10.0)))
    supplied = np.array([1.0, 2.0], dtype=np.float32)

    assembler.assemble({"gyro": supplied})

    np.testing.assert_allclose(supplied, [1.0, 2.0])


def test_accepts_scalars_lists_and_arrays():
    assembler = ObservationAssembler(
        layout(
            ObservationEntry(name="scalar", size=1),
            ObservationEntry(name="listy", size=2),
            ObservationEntry(name="arrayish", size=2),
        )
    )

    obs = assembler.assemble(
        {
            "scalar": 5.0,
            "listy": [1.0, 2.0],
            "arrayish": np.array([[3.0, 4.0]]),  # flattened
        }
    )

    np.testing.assert_allclose(obs, [5.0, 1.0, 2.0, 3.0, 4.0])


"""History stacking"""


def test_history_is_concatenated_newest_first():
    assembler = ObservationAssembler(
        layout(ObservationEntry(name="value", size=1), history_length=2)
    )

    first = assembler.assemble({"value": 1.0})
    # Nothing has been observed before, so the older slot is still zero.
    np.testing.assert_allclose(first, [1.0, 0.0])

    second = assembler.assemble({"value": 2.0})
    np.testing.assert_allclose(second, [2.0, 1.0])

    third = assembler.assemble({"value": 3.0})
    np.testing.assert_allclose(third, [3.0, 2.0])


def test_history_length_three_drops_the_oldest_tick():
    assembler = ObservationAssembler(
        layout(ObservationEntry(name="value", size=1), history_length=3)
    )

    for tick in (1.0, 2.0, 3.0):
        assembler.assemble({"value": tick})
    fourth = assembler.assemble({"value": 4.0})

    np.testing.assert_allclose(fourth, [4.0, 3.0, 2.0])


def test_returned_vectors_are_independent_snapshots():
    """The internal buffer is reused, so a returned vector must not alias it."""
    assembler = ObservationAssembler(
        layout(ObservationEntry(name="value", size=1), history_length=2)
    )

    first = assembler.assemble({"value": 1.0})
    assembler.assemble({"value": 2.0})

    np.testing.assert_allclose(first, [1.0, 0.0])


def test_single_tick_history_returns_an_independent_copy():
    assembler = ObservationAssembler(layout(ObservationEntry(name="value", size=1)))

    first = assembler.assemble({"value": 1.0})
    assembler.assemble({"value": 2.0})

    np.testing.assert_allclose(first, [1.0])


def test_reset_clears_history_back_to_zeros():
    assembler = ObservationAssembler(
        layout(ObservationEntry(name="value", size=1), history_length=2)
    )
    assembler.assemble({"value": 1.0})
    assembler.assemble({"value": 2.0})

    assembler.reset()

    np.testing.assert_allclose(assembler.assemble({"value": 9.0}), [9.0, 0.0])


"""Auto-filled pipeline-state entries (R15)"""


def raw_actions_layout() -> ObservationLayout:
    return layout(
        ObservationEntry(name="gyro", size=2),
        ObservationEntry(
            name="actions",
            size=2,
            source=SOURCE_PIPELINE_STATE,
            pipeline_stage=STAGE_RAW_ACTIONS,
        ),
    )


def processed_actions_layout() -> ObservationLayout:
    return layout(
        ObservationEntry(name="gyro", size=2),
        ObservationEntry(
            name="joint_targets",
            size=2,
            source=SOURCE_PIPELINE_STATE,
            pipeline_stage=STAGE_PROCESSED_ACTIONS,
            action_manager="action_manager",
        ),
    )


def test_pipeline_state_entries_are_excluded_from_required_inputs():
    assembler = ObservationAssembler(raw_actions_layout())

    assert [entry.name for entry in assembler.required_inputs] == ["gyro"]
    assert [entry.name for entry in assembler.auto_filled_inputs] == ["actions"]


def test_raw_action_entry_is_filled_from_the_last_policy_output():
    assembler = ObservationAssembler(raw_actions_layout())

    assembler.assemble({"gyro": [0.0, 0.0]})
    assembler.record_actions([0.5, -0.5])
    obs = assembler.assemble({"gyro": [1.0, 2.0]})

    np.testing.assert_allclose(obs, [1.0, 2.0, 0.5, -0.5])


def test_processed_action_entry_is_filled_from_the_decoded_output():
    """`current_actions` echoes *processed* actions when given a manager, so a
    processed-marked entry must read decoded targets, not the raw policy output."""
    assembler = ObservationAssembler(processed_actions_layout())

    assembler.record_actions(
        [0.1, 0.2],  # raw policy output -- deliberately different
        decoded={"action_manager": [1.5, -1.5]},
    )
    obs = assembler.assemble({"gyro": [0.0, 0.0]})

    np.testing.assert_allclose(obs, [0.0, 0.0, 1.5, -1.5])


def test_auto_filled_entries_start_at_zero_on_the_first_tick():
    assembler = ObservationAssembler(raw_actions_layout())

    obs = assembler.assemble({"gyro": [1.0, 2.0]})

    np.testing.assert_allclose(obs, [1.0, 2.0, 0.0, 0.0])


def test_reset_clears_remembered_actions():
    assembler = ObservationAssembler(raw_actions_layout())
    assembler.record_actions([9.0, 9.0])

    assembler.reset()

    np.testing.assert_allclose(assembler.assemble({"gyro": [0.0, 0.0]}), [0.0, 0.0, 0.0, 0.0])


def test_scaling_applies_to_auto_filled_entries_too():
    assembler = ObservationAssembler(
        layout(
            ObservationEntry(
                name="actions",
                size=2,
                scale=0.5,
                source=SOURCE_PIPELINE_STATE,
                pipeline_stage=STAGE_RAW_ACTIONS,
            )
        )
    )
    assembler.record_actions([2.0, 4.0])

    np.testing.assert_allclose(assembler.assemble(), [1.0, 2.0])


"""Error paths"""


def test_missing_entry_names_the_entry(tmp_path=None):
    """Covers AE2."""
    assembler = ObservationAssembler(simple_layout())

    with pytest.raises(ObservationError) as error:
        assembler.assemble({"gyro": [1.0, 2.0, 3.0]})

    assert "dof_pos" in str(error.value)


def test_wrong_size_names_the_entry_and_both_sizes():
    assembler = ObservationAssembler(simple_layout())

    with pytest.raises(ObservationError) as error:
        assembler.assemble({"gyro": [1.0, 2.0], "dof_pos": [1.0, 2.0]})

    message = str(error.value)
    assert "gyro" in message
    assert "3" in message and "2" in message


def test_unknown_name_is_rejected_with_the_valid_names():
    assembler = ObservationAssembler(simple_layout())

    with pytest.raises(ObservationError) as error:
        assembler.assemble(
            {"gyro": [1.0, 2.0, 3.0], "dof_pos": [1.0, 2.0], "gyroo": [0.0]}
        )

    message = str(error.value)
    assert "gyroo" in message
    assert "gyro" in message


def test_supplying_an_auto_filled_entry_points_at_record_actions():
    assembler = ObservationAssembler(raw_actions_layout())

    with pytest.raises(ObservationError) as error:
        assembler.assemble({"gyro": [0.0, 0.0], "actions": [1.0, 1.0]})

    assert "record_actions" in str(error.value)


def test_unknown_names_are_allowed_when_strict_inputs_is_off():
    assembler = ObservationAssembler(simple_layout(), strict_inputs=False)

    obs = assembler.assemble(
        {"gyro": [4.0, 8.0, 12.0], "dof_pos": [1.0, 3.0], "extra_sensor": [0.0]}
    )

    np.testing.assert_allclose(obs, [1.0, 2.0, 3.0, 2.0, 6.0])


def test_non_numeric_value_is_reported_with_the_entry_name():
    assembler = ObservationAssembler(layout(ObservationEntry(name="gyro", size=1)))

    with pytest.raises(ObservationError) as error:
        assembler.assemble({"gyro": "not-a-number"})

    assert "gyro" in str(error.value)


"""Listings (R5)"""


def test_describe_inputs_separates_supplied_from_auto_filled():
    assembler = ObservationAssembler(raw_actions_layout())

    text = assembler.describe_inputs()

    assert "gyro" in text
    assert "automatically" in text
    assert "actions" in text
