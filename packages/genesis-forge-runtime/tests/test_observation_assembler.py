"""Observation assembly: ordering, scaling, history, and auto-filled entries.

The assembler must reproduce ObservationManager's output exactly. These tests pin
the observable behavior; U5's parity harness proves the equivalence against the
live torch pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis_forge_runtime import (
    ObservationAssembler,
    ObservationEntry,
    ObservationError,
    ObservationLayout,
)


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


"""Entries that echo the policy's own previous output (R15)

The bundle does not mark these out -- they are ordinary inputs, passed the same
way as a sensor reading, just read off the decoder instead. What matters is that
the caller passes them at all, so a forgotten feedback wire raises instead of
silently feeding zeros forever.
"""


def fed_back_layout() -> ObservationLayout:
    return layout(
        ObservationEntry(name="gyro", size=2),
        ObservationEntry(name="actions", size=2),
    )


def test_a_fed_back_value_is_placed_like_any_other_entry():
    assembler = ObservationAssembler(fed_back_layout())

    obs = assembler.assemble({"gyro": [1.0, 2.0], "actions": [0.5, -0.5]})

    np.testing.assert_allclose(obs, [1.0, 2.0, 0.5, -0.5])


def test_forgetting_the_feedback_wire_raises_rather_than_reading_zeros():
    """The whole point of passing it explicitly: omitting it is loud, not silent."""
    assembler = ObservationAssembler(fed_back_layout())

    with pytest.raises(ObservationError) as error:
        assembler.assemble({"gyro": [1.0, 2.0]})

    message = str(error.value)
    assert "actions" in message
    assert "gyro, actions" in message  # the full set is listed, so nothing is guessed


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


def test_a_superfluous_name_is_rejected_even_though_it_is_harmless():
    """A stray key cannot corrupt the vector, but it means the loop has drifted
    from the bundle -- so it is reported rather than quietly ignored."""
    assembler = ObservationAssembler(simple_layout())

    with pytest.raises(ObservationError) as error:
        assembler.assemble(
            {"gyro": [4.0, 8.0, 12.0], "dof_pos": [1.0, 3.0], "extra_sensor": [0.0]}
        )

    assert "extra_sensor" in str(error.value)


def test_non_numeric_value_is_reported_with_the_entry_name():
    assembler = ObservationAssembler(layout(ObservationEntry(name="gyro", size=1)))

    with pytest.raises(ObservationError) as error:
        assembler.assemble({"gyro": "not-a-number"})

    assert "gyro" in str(error.value)


"""Listings (R5)"""


def test_describe_inputs_lists_every_value_to_supply():
    assembler = ObservationAssembler(fed_back_layout())

    text = assembler.describe_inputs()

    assert "Values to supply each tick" in text
    assert "gyro" in text
    assert "actions" in text
