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
    assert torch.equal(result, torch.full((env.num_envs, 1), 6.0))  # scale still applies


def test_override_values_accepts_a_plain_scalar(env):
    mgr = ObservationManager(env, cfg={"a": {"fn": const, "scale": 2.0}})
    mgr.build()

    result = mgr.get_observations(values={"a": 0.3})

    assert torch.allclose(result, torch.full((env.num_envs, 1), 0.6))  # scale still applies


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

    mgr.reset([0, 2])

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
    mgr.reset([0])  # must not raise


def test_reset_clears_history_only_for_the_reset_envs(env):
    values = iter([1.0, 2.0, 3.0, 4.0])  # 1.0 is consumed by build()'s sizing probe

    def sequential(env):
        return torch.full((env.num_envs, 1), next(values))

    mgr = ObservationManager(env, cfg={"a": {"fn": sequential}}, history_len=3)
    mgr.build()
    mgr.get_observations()  # 2.0
    mgr.get_observations()  # 3.0

    mgr.reset([0, 2])
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
    mgr.reset([0])  # must not raise
