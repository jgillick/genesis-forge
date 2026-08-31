"""
Behavior of RewardManager: weighted aggregation, episode logging, and dict-like access.

Uses a FakeEnv and plain/MdpFn reward functions -- no Genesis scene is built. Tests
call through the manager's real `build()`/`step()`/`reset()`, not a hand-replicated
dispatch, so they'd catch a change to how the manager calls a config item's function.
"""

from dataclasses import dataclass

import pytest
import torch

from genesis_forge.managers import RewardManager
from genesis_forge.managers.config import MdpFn


def echo_reward(env, value=1.0):
    return torch.full((env.num_envs,), value)


"""
step() -- weighted aggregation
"""


def test_step_sums_weighted_dt_scaled_values(env):
    mgr = RewardManager(
        env,
        cfg={
            "a": {"fn": echo_reward, "params": {"value": 2.0}, "weight": 3.0},
            "b": {"fn": echo_reward, "params": {"value": 1.0}, "weight": -1.0},
        },
    )
    mgr.build()

    result = mgr.step()

    # (2.0 * 3.0 + 1.0 * -1.0) * dt = 5.0 * dt
    expected = torch.full((env.num_envs,), 5.0 * env.dt)
    assert torch.allclose(result, expected)
    assert torch.equal(mgr.rewards, result)


def test_step_skips_functions_with_zero_weight(env):
    calls = []

    def spy(env):
        calls.append(1)
        return torch.zeros(env.num_envs)

    mgr = RewardManager(env, cfg={"never": {"fn": spy, "weight": 0.0}})
    mgr.build()
    mgr.step()

    assert calls == []


def test_step_returns_the_unchanged_buffer_when_disabled(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 1.0}})
    mgr.build()
    first = mgr.step().clone()
    mgr.enabled = False

    result = mgr.step()

    assert torch.equal(result, first)


"""
step() -- episode data accumulation for logging
"""


def test_step_accumulates_episode_data_when_logging_enabled(env):
    mgr = RewardManager(
        env,
        cfg={"a": {"fn": echo_reward, "params": {"value": 2.0}, "weight": 3.0}},
        logging_enabled=True,
    )
    mgr.build()
    mgr.step()
    mgr.step()

    expected = torch.full((env.num_envs,), 2 * (2.0 * 3.0 * env.dt))
    assert torch.allclose(mgr.episode_data["a"], expected)


def test_step_does_not_accumulate_episode_data_when_logging_disabled(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 1.0}}, logging_enabled=False)
    mgr.build()
    mgr.step()

    assert torch.equal(mgr.episode_data["a"], torch.zeros(env.num_envs))


"""
reset() -- episode mean logging
"""


def test_reset_logs_the_mean_reward_before_and_after_weight(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "params": {"value": 2.0}, "weight": 3.0}})
    mgr.build()
    mgr.step()  # one step: episode_data = 2.0 * 3.0 * dt, episode_seconds = dt

    mgr.reset()

    assert mgr.last_episode_mean_reward("a", before_weight=False) == pytest.approx(6.0)
    assert mgr.last_episode_mean_reward("a", before_weight=True) == pytest.approx(2.0)


def test_last_episode_mean_reward_defaults_to_zero_before_any_reset(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 2.0}})
    mgr.build()

    assert mgr.last_episode_mean_reward("a") == 0.0


def test_reset_writes_to_the_env_extras_logging_dict(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 2.0}})
    mgr.build()
    mgr.step()

    mgr.reset()

    logging_dict = env.extras[env.extras_logging_key]
    assert "Rewards / a" in logging_dict


def test_reset_skips_logging_zero_weight_items(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 0.0}})
    mgr.build()
    mgr.step()

    mgr.reset()

    logging_dict = env.extras[env.extras_logging_key]
    assert "Rewards / a" not in logging_dict


def test_reset_without_indices_resets_every_env(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 1.0}})
    mgr.build()
    mgr.step()

    mgr.reset()

    assert torch.equal(mgr.episode_data["a"], torch.zeros(env.num_envs))


def test_reset_only_clears_the_given_envs_idx(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 1.0}})
    mgr.build()
    mgr.step()

    mgr.reset([0])

    assert mgr.episode_data["a"][0] == 0.0
    assert torch.all(mgr.episode_data["a"][1:] != 0.0)


"""
reset() -- forwards to stateful reward functions
"""


def test_reset_forwards_to_the_config_items_reset(env):
    @dataclass(kw_only=True, eq=False)
    class Stateful(MdpFn):
        def build(self):
            self.has_reset = torch.zeros(self.env.num_envs)

        def reset(self, envs_idx):
            self.has_reset[envs_idx] = 1.0

        def __call__(self, env):
            return torch.zeros(env.num_envs)

    fn = Stateful()
    mgr = RewardManager(env, cfg={"a": {"fn": fn, "weight": 1.0}})
    mgr.build()

    mgr.reset([0, 2])

    assert torch.equal(fn.has_reset, torch.tensor([1.0, 0.0, 1.0, 0.0]))


"""
build() -- builds each config item's function
"""


def test_build_builds_each_mdp_fn_config_item(env):
    @dataclass(kw_only=True, eq=False)
    class Counting(MdpFn):
        def build(self):
            self.builds = getattr(self, "builds", 0) + 1

        def __call__(self, env):
            return torch.zeros(env.num_envs)

    fn = Counting()
    mgr = RewardManager(env, cfg={"a": {"fn": fn, "weight": 1.0}})
    assert getattr(fn, "builds", 0) == 0

    mgr.build()

    assert fn.builds == 1


"""
Dict-like access
"""


def test_dict_like_access(env):
    mgr = RewardManager(env, cfg={"a": {"fn": echo_reward, "weight": 1.0}})

    assert "a" in mgr
    assert len(mgr) == 1
    assert list(mgr) == ["a"]
    assert mgr["a"].weight == 1.0

    mgr["b"] = mgr["a"]
    assert len(mgr) == 2
    assert "b" in mgr

    del mgr["a"]
    assert "a" not in mgr
    assert len(mgr) == 1
