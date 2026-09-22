"""Behavior of ObservationManager: observation-space sizing, the history ring
buffer, scale/noise application, and the values= override path used for manual
deployment/debugging.

Uses plain functions and MdpFn instances as observation functions -- no Genesis
scene is built.
"""

from dataclasses import dataclass

import pytest
import torch

from genesis_forge.managers import ObservationManager
from genesis_forge.managers.config import MdpFn


def const(env, value=1.0, size=1):
    return torch.full((env.num_envs, size), value)


"""
Construction
"""


def test_history_len_of_zero_raises(env):
    with pytest.raises(ValueError, match="history_len"):
        ObservationManager(env, cfg={}, history_len=0)


def test_name_defaults_to_policy(env):
    mgr = ObservationManager(env, cfg={})
    assert mgr.name == "policy"


"""
build() -- observation space sizing
"""


def test_build_sizes_the_space_from_the_sum_of_each_functions_last_dim(env):
    mgr = ObservationManager(
        env,
        cfg={
            "a": {"fn": const, "params": {"size": 2}},
            "b": {"fn": const, "params": {"size": 3}},
        },
    )
    mgr.build()
    assert mgr.observation_space.shape == (5,)


def test_build_multiplies_the_space_by_history_len(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"size": 2}}}, history_len=3
    )
    mgr.build()
    assert mgr.observation_space.shape == (6,)


def test_build_when_disabled_uses_a_size_of_one(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "params": {"size": 5}}})
    mgr.enabled = False
    mgr.build()
    assert mgr.observation_space.shape == (1,)


def test_build_builds_each_mdp_fn_config_item(env):
    @dataclass(kw_only=True, eq=False)
    class Counting(MdpFn):
        def build(self):
            self.builds = getattr(self, "builds", 0) + 1

        def __call__(self, env):
            return torch.zeros((env.num_envs, 1))

    fn = Counting()
    mgr = ObservationManager(env, cfg={"a": {"fn": fn}})
    assert getattr(fn, "builds", 0) == 0

    mgr.build()

    assert fn.builds == 1


def test_build_sizes_the_space_from_an_mdp_fn(env):
    """__call__ must be valid the instant build() returns, not on the first step."""

    @dataclass(kw_only=True, eq=False)
    class Buffered(MdpFn):
        width: int = 2

        def build(self):
            self.buf = torch.zeros((self.env.num_envs, self.width))

        def __call__(self, env):
            return self.buf

    mgr = ObservationManager(env, cfg={"history": {"fn": Buffered(width=3)}})
    mgr.build()

    assert mgr.observation_space.shape == (3,)


def test_build_raises_a_clear_error_for_a_noncallable_fn(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": 42}})
    with pytest.raises(AssertionError, match="not callable"):
        mgr.build()


def test_build_reraises_the_original_exception(env):
    def failing(env):
        raise ValueError("boom")

    mgr = ObservationManager(env, cfg={"a": {"fn": failing}})
    with pytest.raises(ValueError, match="boom"):
        mgr.build()


"""
get_observations() -- history ring buffer ordering: newest observation first
"""


def test_get_observations_orders_newest_first(env):
    values = iter([0.0, 1.0, 2.0, 3.0])  # 0.0 is consumed by build()'s sizing probe

    def sequential(env):
        return torch.full((env.num_envs, 1), next(values))

    mgr = ObservationManager(env, cfg={"a": {"fn": sequential}}, history_len=2)
    mgr.build()

    first = mgr.get_observations()
    assert torch.equal(first, torch.tensor([[1.0, 0.0]] * env.num_envs))

    second = mgr.get_observations()
    assert torch.equal(second, torch.tensor([[2.0, 1.0]] * env.num_envs))

    third = mgr.get_observations()
    assert torch.equal(third, torch.tensor([[3.0, 2.0]] * env.num_envs))


def test_get_observations_returns_zeros_when_disabled(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "params": {"size": 2}}})
    mgr.build()
    mgr.enabled = False

    result = mgr.get_observations()
    assert torch.equal(result, torch.zeros((env.num_envs, 2)))


"""
Scale and noise
"""


def test_scale_is_applied_to_the_value(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 2.0}, "scale": 3.0}}
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.equal(result, torch.full((env.num_envs, 1), 6.0))


def test_scale_of_one_is_a_noop(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 2.0}, "scale": 1.0}}
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.equal(result, torch.full((env.num_envs, 1), 2.0))


def test_per_item_noise_overrides_the_manager_default(env):
    mgr = ObservationManager(
        env,
        cfg={"a": {"fn": const, "params": {"value": 0.0}, "noise": 5.0}},
        noise=0.01,
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.all(result.abs() <= 5.0)
    assert torch.any(result != 0.0)  # noise was actually applied, not silently skipped


def test_manager_level_noise_applies_when_item_has_none(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 0.0}}}, noise=2.0
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.all(result.abs() <= 2.0)


def test_zero_noise_is_a_noop(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 5.0}, "noise": 0.0}}
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.equal(result, torch.full((env.num_envs, 1), 5.0))


def test_noise_is_applied_before_scale(env):
    # Noise models the real sensor's own measurement uncertainty, in the raw units `fn`
    # returns -- it must land before `scale`, which is only a downstream convenience for
    # the policy. `(value + noise) * scale`, never `value * scale + noise`.
    torch.manual_seed(0)
    expected_noise = torch.empty((env.num_envs, 1)).uniform_(-1, 1) * 5.0

    torch.manual_seed(0)
    mgr = ObservationManager(
        env,
        cfg={"a": {"fn": const, "params": {"value": 2.0}, "noise": 5.0, "scale": 10.0}},
    )
    mgr.build()
    result = mgr.get_observations()

    assert torch.allclose(result, (2.0 + expected_noise) * 10.0)


"""
Clip -- a wide safety bound in the raw units, between noise and scale
"""


def test_clip_bounds_the_value(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 50.0}, "clip": (-10.0, 10.0)}}
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.equal(result, torch.full((env.num_envs, 1), 10.0))


def test_clip_applies_before_scale(env):
    """The bound is written in the units `fn` returns, so `clip` then `scale`."""
    mgr = ObservationManager(
        env,
        cfg={
            "a": {
                "fn": const,
                "params": {"value": 50.0},
                "clip": (-10.0, 10.0),
                "scale": 0.1,
            }
        },
    )
    mgr.build()
    result = mgr.get_observations()
    # clip first: min(50, 10) * 0.1 = 1.0. Scale first would give min(5, 10) = 5.0.
    assert torch.allclose(result, torch.full((env.num_envs, 1), 1.0))


def test_clip_applies_after_noise(env):
    """Noise is the sensor's own error; the bound must catch the noisy reading too."""
    mgr = ObservationManager(
        env,
        cfg={
            "a": {
                "fn": const,
                "params": {"value": 10.0},
                "noise": 5.0,
                "clip": (-10.0, 10.0),
            }
        },
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.all(result <= 10.0)


def test_clip_removes_inf_but_not_nan(env):
    """inf is the value that poisons a running normalizer; clip stops it. NaN is not
    a magnitude and still surfaces through the non-finite guard."""

    def with_inf(env):
        value = torch.ones((env.num_envs, 2))
        value[0, 0] = float("inf")
        return value

    mgr = ObservationManager(env, cfg={"a": {"fn": with_inf, "clip": (-100.0, 100.0)}})
    mgr.build()
    result = mgr.get_observations()
    assert result[0, 0] == 100.0

    def with_nan(env):
        value = torch.ones((env.num_envs, 2))
        value[0, 0] = float("nan")
        return value

    mgr = ObservationManager(env, cfg={"a": {"fn": with_nan, "clip": (-100.0, 100.0)}})
    mgr.build()
    with pytest.raises(ValueError, match="non-finite"):
        mgr.get_observations()


def test_clip_does_not_mutate_the_buffer_a_function_returned(env):
    buffer = torch.full((env.num_envs, 1), 50.0)
    mgr = ObservationManager(
        env, cfg={"a": {"fn": returns_a_kept_buffer(buffer), "clip": (-10.0, 10.0)}}
    )
    mgr.build()

    mgr.get_observations()

    assert torch.equal(buffer, torch.full((env.num_envs, 1), 50.0)), (
        "clipping reached back into the function's own buffer"
    )


def test_clip_applies_to_override_values(env):
    """The deployment parity harness feeds values in through this path, and the
    robot-side assembler clips, so the training side must clip here too."""
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "clip": (-1.0, 1.0)}})
    mgr.build()
    supplied = torch.full((env.num_envs, 1), 7.0)

    result = mgr.get_observations(values={"a": supplied})

    assert torch.equal(result, torch.ones((env.num_envs, 1)))
    assert torch.equal(supplied, torch.full((env.num_envs, 1), 7.0))


@pytest.mark.parametrize(
    "clip",
    [100.0, (1.0,), (1.0, 2.0, 3.0), ("lo", "hi"), (10.0, -10.0), (5.0, 5.0)],
)
def test_a_malformed_clip_is_refused_when_the_config_is_read(env, clip):
    with pytest.raises(ValueError, match="Observation 'a'.*clip"):
        ObservationManager(env, cfg={"a": {"fn": const, "clip": clip}})


def test_an_infinite_clip_bound_is_refused(env):
    """A bundle is plain JSON, which cannot carry inf; spell an open side with a
    generous finite number instead."""
    with pytest.raises(ValueError, match="finite"):
        ObservationManager(env, cfg={"a": {"fn": const, "clip": (-1.0, float("inf"))}})


def test_no_clip_is_the_default(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "params": {"value": 1e6}}})
    mgr.build()
    result = mgr.get_observations()
    assert torch.equal(result, torch.full((env.num_envs, 1), 1e6))


"""
Non-finite guard -- a blow-up is reported the step it happens, naming the culprit

Passed on, one inf from one env silently corrupts an RL library's running
observation normalizer, and every env's actions turn to NaN a step later. The
guard lives in get_observations(); build()'s sizing probe only measures widths.

The message counts environments rather than listing them: with thousands of envs a
list of ids is noise, while "1 of 4096" versus "4096 of 4096" separates a lone
physics blow-up from something systematic.
"""


def non_finite_in(env_idx, col, value=float("inf"), size=3):
    def fn(env):
        out = torch.ones((env.num_envs, size))
        out[env_idx, col] = value
        return out

    return fn


def test_a_non_finite_observation_raises_naming_the_entry_and_a_count(env):
    mgr = ObservationManager(
        env,
        cfg={
            "fine": {"fn": const, "params": {"size": 2}},
            "gyro": {"fn": non_finite_in(env_idx=2, col=1)},
            "also_fine": {"fn": const},
        },
    )
    mgr.build()

    with pytest.raises(ValueError) as error:
        mgr.get_observations()

    message = str(error.value)
    assert "'gyro' (inf in 1 of 4 envs)" in message
    assert "'fine'" not in message and "'also_fine'" not in message
    assert "policy" in message  # the manager is named, for asymmetric setups


def test_nan_and_inf_are_counted_separately(env):
    """Clipping cures inf but not NaN, so the reader needs to know which it was."""

    def mixed(env):
        out = torch.ones((env.num_envs, 2))
        out[0, 0] = float("nan")
        out[1, 1] = float("inf")
        out[2, 0] = float("-inf")
        return out

    mgr = ObservationManager(env, cfg={"gyro": {"fn": mixed}})
    mgr.build()
    with pytest.raises(ValueError) as error:
        mgr.get_observations()

    assert "'gyro' (inf in 2 of 4 envs, NaN in 1 of 4 envs)" in str(error.value)


def test_every_affected_entry_is_listed(env):
    def two_bad(env):
        out = torch.ones((env.num_envs, 1))
        out[1, 0] = float("-inf")
        out[3, 0] = float("nan")
        return out

    mgr = ObservationManager(
        env,
        cfg={
            "first": {"fn": two_bad},
            "second": {"fn": non_finite_in(env_idx=0, col=2)},
        },
    )
    mgr.build()
    with pytest.raises(ValueError) as error:
        mgr.get_observations()

    message = str(error.value)
    assert "'first' (inf in 1 of 4 envs, NaN in 1 of 4 envs)" in message
    assert "'second' (inf in 1 of 4 envs)" in message


def test_the_guard_runs_every_step_not_just_at_build(env):
    values = iter([1.0, 2.0, float("inf")])  # 1.0 is consumed by build()'s sizing probe

    def sequential(env):
        return torch.full((env.num_envs, 1), next(values))

    mgr = ObservationManager(env, cfg={"a": {"fn": sequential}})
    mgr.build()
    mgr.get_observations()  # 2.0, fine

    with pytest.raises(ValueError, match="non-finite"):
        mgr.get_observations()


def test_finite_observations_pass_the_guard_untouched(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 3.0e38, "size": 2}}}
    )
    mgr.build()
    result = mgr.get_observations()
    assert torch.equal(result, torch.full((env.num_envs, 2), 3.0e38))


"""
Scaling and noise must not reach back into the values they were given

Both are applied to a value the manager did not create. An observation function may
hand back a buffer it keeps rather than a fresh tensor -- ``current_actions`` returns
``env.actions`` itself -- and the override path is handed the caller's own tensor. In
place, either would corrupt its source: silently, every step, compounding.
"""


def returns_a_kept_buffer(buffer):
    """An observation function that hands back a buffer it holds on to."""

    def fn(env):
        return buffer

    return fn


def test_scale_does_not_mutate_the_buffer_a_function_returned(env):
    buffer = torch.ones((env.num_envs, 1))
    mgr = ObservationManager(
        env, cfg={"a": {"fn": returns_a_kept_buffer(buffer), "scale": 0.5}}
    )
    mgr.build()

    result = mgr.get_observations()

    assert torch.equal(result, torch.full((env.num_envs, 1), 0.5))
    assert torch.equal(buffer, torch.ones((env.num_envs, 1))), (
        "scaling reached back into the function's own buffer"
    )


def test_noise_does_not_mutate_the_buffer_a_function_returned(env):
    buffer = torch.zeros((env.num_envs, 1))
    mgr = ObservationManager(
        env, cfg={"a": {"fn": returns_a_kept_buffer(buffer), "noise": 5.0}}
    )
    mgr.build()

    mgr.get_observations()

    assert torch.equal(buffer, torch.zeros((env.num_envs, 1))), (
        "noise reached back into the function's own buffer"
    )


def test_scaling_a_kept_buffer_does_not_compound_across_steps(env):
    """The failure this prevents: a scale re-applied to its own output every step."""
    buffer = torch.ones((env.num_envs, 1))
    mgr = ObservationManager(
        env, cfg={"a": {"fn": returns_a_kept_buffer(buffer), "scale": 0.5}}
    )
    mgr.build()

    for _ in range(5):
        result = mgr.get_observations()

    assert torch.equal(result, torch.full((env.num_envs, 1), 0.5)), (
        "the observation decayed over five steps instead of staying constant"
    )


def test_scale_does_not_mutate_a_supplied_override_tensor(env):
    """The deployment parity harness feeds values in through this path."""
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "scale": 0.25}})
    mgr.build()
    supplied = torch.ones((env.num_envs, 1))

    result = mgr.get_observations(values={"a": supplied})

    assert torch.equal(result, torch.full((env.num_envs, 1), 0.25))
    assert torch.equal(supplied, torch.ones((env.num_envs, 1))), (
        "scaling reached back into the caller's tensor"
    )


"""
get_observations(values=...) -- override path for manual deployment/debugging
"""


def test_override_values_are_used_instead_of_calling_the_function(env):
    calls = []

    def spy(env):
        calls.append(1)
        return torch.zeros((env.num_envs, 1))

    mgr = ObservationManager(env, cfg={"a": {"fn": spy, "scale": 2.0}})
    mgr.build()
    calls.clear()  # drop the build-time sizing probe call

    override = torch.full((env.num_envs, 1), 3.0)
    result = mgr.get_observations(values={"a": override})

    assert calls == []  # the function itself was never called for this round
    assert torch.equal(
        result, torch.full((env.num_envs, 1), 6.0)
    )  # scale still applies


def test_override_values_accepts_a_plain_scalar(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "scale": 2.0}})
    mgr.build()

    result = mgr.get_observations(values={"a": 0.3})

    assert torch.allclose(
        result, torch.full((env.num_envs, 1), 0.6)
    )  # scale still applies


def test_override_values_skip_noise(env):
    mgr = ObservationManager(
        env, cfg={"a": {"fn": const, "params": {"value": 0.0}, "noise": 5.0}}
    )
    mgr.build()

    override = torch.zeros((env.num_envs, 1))
    result = mgr.get_observations(values={"a": override})

    assert torch.equal(result, override)


def test_override_values_raises_for_a_missing_key(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const}})
    mgr.build()

    with pytest.raises(ValueError, match="not found in override values"):
        mgr.get_observations(values={})


"""
reset()
"""


def test_reset_forwards_to_each_config_items_reset(env):
    @dataclass(kw_only=True, eq=False)
    class Stateful(MdpFn):
        def build(self):
            self.reset_calls = []

        def reset(self, envs_idx):
            self.reset_calls.append(list(envs_idx))

        def __call__(self, env):
            return torch.zeros((env.num_envs, 1))

    fn = Stateful()
    mgr = ObservationManager(env, cfg={"a": {"fn": fn}})
    mgr.build()

    mgr.reset(torch.tensor([0, 2]))

    assert fn.reset_calls == [[0, 2]]


def test_reset_defaults_to_every_env(env):
    @dataclass(kw_only=True, eq=False)
    class Stateful(MdpFn):
        def build(self):
            self.reset_calls = []

        def reset(self, envs_idx):
            self.reset_calls.append([int(i) for i in envs_idx])

        def __call__(self, env):
            return torch.zeros((env.num_envs, 1))

    fn = Stateful()
    mgr = ObservationManager(env, cfg={"a": {"fn": fn}})
    mgr.build()

    mgr.reset()

    assert fn.reset_calls == [list(range(env.num_envs))]


def test_reset_tolerates_plain_functions(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const}})
    mgr.build()
    mgr.reset(torch.tensor([0]))  # must not raise


def test_reset_clears_history_only_for_the_reset_envs(env):
    values = iter([1.0, 2.0, 3.0, 4.0])  # 1.0 is consumed by build()'s sizing probe

    def sequential(env):
        return torch.full((env.num_envs, 1), next(values))

    mgr = ObservationManager(env, cfg={"a": {"fn": sequential}}, history_len=3)
    mgr.build()
    mgr.get_observations()  # 2.0
    mgr.get_observations()  # 3.0

    mgr.reset(torch.tensor([0, 2]))
    obs = mgr.get_observations()  # 4.0

    # Reset envs observe only the fresh value; their history slots are zero
    assert torch.equal(obs[[0, 2]], torch.tensor([[4.0, 0.0, 0.0]] * 2))
    # Non-reset envs keep their history, newest first
    assert torch.equal(obs[[1, 3]], torch.tensor([[4.0, 3.0, 2.0]] * 2))


def test_reset_with_no_envs_idx_clears_history_for_every_env(env):
    values = iter([1.0, 2.0, 3.0])  # 1.0 is consumed by build()'s sizing probe

    def sequential(env):
        return torch.full((env.num_envs, 1), next(values))

    mgr = ObservationManager(env, cfg={"a": {"fn": sequential}}, history_len=2)
    mgr.build()
    mgr.get_observations()  # 2.0

    mgr.reset()
    obs = mgr.get_observations()  # 3.0

    assert torch.equal(obs, torch.tensor([[3.0, 0.0]] * env.num_envs))


def test_reset_before_build_tolerates_the_empty_history(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const}}, history_len=2)
    mgr.reset(torch.tensor([0]))  # must not raise
