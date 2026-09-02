"""The parity gate: does the numpy runtime reproduce the torch pipeline?

This is the load-bearing check of the whole feature. If it can be made to pass
while the two pipelines actually differ, every other guarantee is hollow -- so
several of these tests deliberately break one side and assert the gate notices.
"""

import numpy as np
import pytest
import torch

from genesis_forge.deployment import ParityError, capture_environment, check_parity
from genesis_forge.managers import ObservationManager

"""
Agreement on a correct pipeline
"""


def test_parity_passes_for_an_unmodified_environment(deployable_env):
    capture = capture_environment(deployable_env)

    report = check_parity(capture)

    assert report.max_observation_error < 1e-6
    assert all(error < 1e-6 for error in report.max_action_error.values())


def test_parity_passes_with_history_stacking(make_env):
    capture = capture_environment(make_env(history_len=3))

    report = check_parity(capture, ticks=6)

    assert report.max_observation_error < 1e-6


def test_parity_ignores_training_noise(make_env):
    """Noise hardens the policy during training; it is never replayed on a robot."""
    cfg = {
        "gyro": {
            "fn": lambda env: torch.ones((env.num_envs, 3)) * 0.5,
            "scale": 0.25,
            "noise": 0.5,
        }
    }
    capture = capture_environment(make_env(cfg=cfg))

    report = check_parity(capture)

    assert report.max_observation_error < 1e-6


def test_parity_covers_every_action_manager(deployable_env):
    capture = capture_environment(deployable_env)

    report = check_parity(capture)

    assert set(report.max_action_error) == {"action_manager"}


def test_parity_produces_golden_samples(deployable_env):
    capture = capture_environment(deployable_env)

    report = check_parity(capture, ticks=4)

    assert report.golden["observations"].shape[0] == 4
    assert report.golden["observations"].shape[1] == capture.manifest.observations.total_size
    assert report.golden["raw_actions"].shape == (4, capture.manifest.num_actions)
    assert report.golden["joint_targets"].shape == (4, capture.manifest.num_actions)


def test_parity_leaves_the_environment_history_untouched(make_env):
    """Exporting must not disturb an environment someone may still be using."""
    env = make_env(history_len=2)
    manager = env.observation_manager
    manager.get_observations()
    before = [tensor.clone() for tensor in manager._history]

    check_parity(capture_environment(env))

    for saved, live in zip(before, manager._history, strict=True):
        torch.testing.assert_close(saved, live)


"""
Catching real divergence
"""


def test_a_wrong_decode_parameter_is_caught_and_named(deployable_env, monkeypatch):
    """Covers AE1: the gate names the component that diverged."""
    capture = capture_environment(deployable_env)
    # Simulate a hand-written counterpart drifting from process_actions.
    capture.manifest.actions[0].config["scale"] = np.array(
        [0.9, 0.9, 0.9], dtype=np.float32
    )

    with pytest.raises(ParityError) as error:
        check_parity(capture)

    message = str(error.value)
    assert "action_manager" in message
    assert "position" in message
    assert "not written" in message


def test_a_wrong_observation_scale_is_caught(deployable_env):
    capture = capture_environment(deployable_env)
    entries = list(capture.manifest.observations.entries)
    entries[0] = type(entries[0])(**{**entries[0].__dict__, "scale": 99.0})
    object.__setattr__(capture.manifest.observations, "entries", tuple(entries))

    with pytest.raises(ParityError) as error:
        check_parity(capture)

    assert "observation assembly" in str(error.value)


def test_a_reversed_history_order_is_caught(make_env, monkeypatch):
    """Multi-tick comparison is what catches ordering bugs a single tick cannot."""
    from genesis_forge.deployment import parity
    from genesis_forge_runtime import ObservationAssembler

    class ReversedHistoryAssembler(ObservationAssembler):
        def assemble(self, values=None):
            vector = super().assemble(values)
            width = self._layout.single_size
            chunks = [vector[i : i + width] for i in range(0, vector.size, width)]
            return np.concatenate(list(reversed(chunks)))

    # Patch where it is looked up, not where it is defined.
    monkeypatch.setattr(parity, "ObservationAssembler", ReversedHistoryAssembler)
    capture = capture_environment(make_env(history_len=2))

    with pytest.raises(ParityError) as error:
        check_parity(capture, ticks=4)

    assert "observation assembly" in str(error.value)


def test_a_swapped_entry_order_is_caught(deployable_env):
    capture = capture_environment(deployable_env)
    reversed_entries = tuple(reversed(capture.manifest.observations.entries))
    object.__setattr__(capture.manifest.observations, "entries", reversed_entries)

    with pytest.raises(ParityError):
        check_parity(capture)


def test_a_missing_post_clip_is_caught(deployable_env):
    """Clip-boundary samples exist precisely to catch this."""
    capture = capture_environment(deployable_env)
    del capture.manifest.actions[0].config["post_clip_high"]

    with pytest.raises(ParityError) as error:
        check_parity(capture)

    assert "action_manager" in str(error.value)


def test_the_error_reports_the_worst_index_and_both_values(deployable_env):
    capture = capture_environment(deployable_env)
    capture.manifest.actions[0].config["offset"] = np.array(
        [0.1, 5.0, 0.3], dtype=np.float32
    )

    with pytest.raises(ParityError) as error:
        check_parity(capture)

    message = str(error.value)
    assert "index" in message
    assert "tolerance" in message


"""
Reproducibility
"""


def test_the_same_seed_produces_the_same_samples(deployable_env):
    capture = capture_environment(deployable_env)

    first = check_parity(capture, seed=7).golden["raw_actions"]
    second = check_parity(capture, seed=7).golden["raw_actions"]

    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_samples(deployable_env):
    capture = capture_environment(deployable_env)

    first = check_parity(capture, seed=1).golden["raw_actions"]
    second = check_parity(capture, seed=2).golden["raw_actions"]

    assert not np.array_equal(first, second)


"""
Auto-filled entries
"""


def test_parity_handles_auto_filled_observations(make_env):
    from genesis_forge.mdp.observations import current_actions

    env = make_env()
    manager = ObservationManager(
        env,
        cfg={
            "gyro": {"fn": lambda env: torch.ones((env.num_envs, 3)) * 0.5},
            "actions": {"fn": current_actions()},
        },
    )
    env.managers["observation"] = [manager]
    env.observation_manager = manager
    env.actions = torch.zeros((env.num_envs, 3))
    manager.build()

    report = check_parity(capture_environment(env))

    assert report.max_observation_error < 1e-6
