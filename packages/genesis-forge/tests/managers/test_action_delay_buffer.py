"""Behavior of ActionDelayBuffer: fixed and randomized per-env action latency.

Uses plain tensors -- no manager, no Genesis scene.
"""

import pytest
import torch

from genesis_forge.managers import ActionDelayBuffer


def make_buffer(delay_step, num_envs=4, num_actions=2):
    buf = ActionDelayBuffer(delay_step)
    buf.build(num_envs=num_envs, num_actions=num_actions)
    return buf


def test_no_delay_returns_the_input_untouched():
    buf = make_buffer(0)
    actions = torch.full((4, 2), 0.1)
    assert not buf.enabled
    assert buf.range == (0, 0)
    assert buf.push(actions) is actions
    assert torch.equal(buf.delay_steps, torch.zeros(4, dtype=torch.long))
    buf.reset(torch.tensor([1]))  # must not raise


def test_fixed_delay_delivers_each_action_delay_steps_later():
    buf = make_buffer(2)
    assert buf.enabled
    assert torch.equal(buf.delay_steps, torch.full((4,), 2))

    a = torch.full((4, 2), 0.1)
    b = torch.full((4, 2), 0.2)
    assert torch.equal(buf.push(a), torch.zeros((4, 2)))
    assert torch.equal(buf.push(b), torch.zeros((4, 2)))
    assert torch.equal(buf.push(torch.full((4, 2), 0.3)), a)
    assert torch.equal(buf.push(torch.full((4, 2), 0.4)), b)


def test_delayed_output_is_a_copy_of_the_pushed_actions():
    """Mutating the caller's tensor after a push doesn't change what is queued."""
    buf = make_buffer(1)
    actions = torch.full((4, 2), 0.1)
    buf.push(actions)
    actions.fill_(0.9)
    assert torch.equal(buf.push(torch.zeros((4, 2))), torch.full((4, 2), 0.1))


def test_range_draws_a_delay_per_env_within_the_range():
    torch.manual_seed(0)
    buf = make_buffer((1, 3), num_envs=64)

    assert buf.range == (1, 3)
    assert buf.delay_steps.shape == (64,)
    assert buf.delay_steps.min() >= 1
    assert buf.delay_steps.max() <= 3
    # With 64 draws from three values, a single repeated value means no randomization.
    assert len(torch.unique(buf.delay_steps)) > 1


def test_range_delivers_each_env_its_own_delayed_action():
    """Env i receives the action from `delay_steps[i]` steps ago, zeros before that."""
    torch.manual_seed(0)
    num_envs = 16
    buf = make_buffer((0, 2), num_envs=num_envs)
    delays = buf.delay_steps.clone()

    # Step t pushes the value (t + 1) / 10 to every env, so the origin of what a push
    # returns is readable off its value.
    sent = []
    for t in range(4):
        actions = torch.full((num_envs, 2), (t + 1) / 10)
        sent.append(actions)
        delivered = buf.push(actions)

        expected = torch.zeros((num_envs, 2))
        for i in range(num_envs):
            origin = t - int(delays[i])
            if origin >= 0:
                expected[i] = sent[origin][i]
        assert torch.equal(delivered, expected), f"step {t}"


def test_reset_clears_queued_actions_for_reset_envs_only():
    buf = make_buffer(1)
    queued = torch.full((4, 2), 0.1)
    buf.push(queued)
    buf.reset(torch.tensor([1]))

    expected = queued.clone()
    expected[1] = 0.0
    assert torch.equal(buf.push(torch.full((4, 2), 0.2)), expected)


def test_reset_redraws_delay_for_reset_envs_only():
    torch.manual_seed(0)
    buf = make_buffer((1, 3), num_envs=64)
    before = buf.delay_steps.clone()
    buf.push(torch.full((64, 2), 0.5))

    reset_idx = torch.arange(0, 64, 2)
    kept_idx = torch.arange(1, 64, 2)
    buf.reset(reset_idx)

    assert torch.equal(buf.delay_steps[kept_idx], before[kept_idx])
    assert buf.delay_steps[reset_idx].min() >= 1
    assert buf.delay_steps[reset_idx].max() <= 3
    assert not torch.equal(buf.delay_steps[reset_idx], before[reset_idx])

    # Nothing queued before the reset reaches a reset env, whatever its new delay.
    for _ in range(3):
        delivered = buf.push(torch.zeros((64, 2)))
        assert torch.equal(delivered[reset_idx], torch.zeros((len(reset_idx), 2)))


@pytest.mark.parametrize(
    "delay_step",
    [-1, (-1, 2), (2, 1), 1.5, (0, 1.5), (0, 1, 2), True],
)
def test_invalid_delay_step_is_rejected(delay_step):
    with pytest.raises(ValueError):
        ActionDelayBuffer(delay_step)
