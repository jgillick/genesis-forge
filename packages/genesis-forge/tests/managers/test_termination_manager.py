"""Behavior of TerminationManager: OR-combining termination/truncation signals
across config items, writing to env.extras, episode logging, and the enabled gate.

Uses plain functions and MdpFn instances as termination functions -- no Genesis
scene is built.
"""

from dataclasses import dataclass

import pytest
import torch

from genesis_forge.managers import TerminationManager
from genesis_forge.managers.config import MdpFn


def const(env, value):
    return torch.tensor(value, dtype=torch.bool)


"""
Properties -- dones is the OR of terminated and truncated
"""


def test_dones_is_the_or_of_terminated_and_truncated(env):
    mgr = TerminationManager(
        env,
        term_cfg={
            "a": {"fn": const, "params": {"value": [True, False, False, False]}},
            "b": {
                "fn": const,
                "params": {"value": [False, True, False, False]},
                "time_out": True,
            },
        },
    )
    mgr.build()
    mgr.step()

    assert mgr.terminated.tolist() == [True, False, False, False]
    assert mgr.truncated.tolist() == [False, True, False, False]
    assert mgr.dones.tolist() == [True, True, False, False]


"""
step() -- aggregation
"""


def test_step_ors_multiple_termination_items_together(env):
    mgr = TerminationManager(
        env,
        term_cfg={
            "a": {"fn": const, "params": {"value": [True, False, False, False]}},
            "b": {"fn": const, "params": {"value": [False, False, True, False]}},
        },
    )
    mgr.build()
    mgr.step()

    assert mgr.terminated.tolist() == [True, False, True, False]


def test_step_clears_previous_results_before_recomputing(env):
    flag = {"value": [True, True, True, True]}

    def dynamic(env):
        return torch.tensor(flag["value"], dtype=torch.bool)

    mgr = TerminationManager(env, term_cfg={"a": {"fn": dynamic}})
    mgr.build()
    mgr.step()
    assert mgr.terminated.tolist() == [True, True, True, True]

    flag["value"] = [False, False, False, False]
    mgr.step()
    assert mgr.terminated.tolist() == [False, False, False, False]


def test_step_is_a_noop_when_disabled(env):
    mgr = TerminationManager(
        env, term_cfg={"a": {"fn": const, "params": {"value": [True] * env.num_envs}}}
    )
    mgr.build()
    mgr.enabled = False

    terminated, truncated = mgr.step()

    assert not torch.any(terminated)
    assert not torch.any(truncated)


def test_step_reraises_the_original_exception(env):
    def failing(env):
        raise ValueError("boom")

    mgr = TerminationManager(env, term_cfg={"a": {"fn": failing}})
    mgr.build()

    with pytest.raises(ValueError, match="boom"):
        mgr.step()


"""
step() -- env.extras
"""


def test_step_writes_terminations_and_time_outs_to_env_extras(env):
    mgr = TerminationManager(
        env, term_cfg={"a": {"fn": const, "params": {"value": [True, False, False, False]}}}
    )
    mgr.build()
    mgr.step()

    assert torch.equal(env.extras["terminations"], mgr.terminated)
    assert torch.equal(env.extras["time_outs"], mgr.truncated)


"""
step() -- episode logging
"""


def test_step_logs_the_mean_value_when_any_env_is_done(env):
    mgr = TerminationManager(
        env, term_cfg={"a": {"fn": const, "params": {"value": [True, False, False, False]}}}
    )
    mgr.build()
    mgr.step()

    logging_dict = env.extras[env.extras_logging_key]
    assert logging_dict["Terminations / a"].item() == pytest.approx(0.25)


def test_step_skips_logging_when_nothing_is_done(env):
    mgr = TerminationManager(
        env, term_cfg={"a": {"fn": const, "params": {"value": [False] * env.num_envs}}}
    )
    mgr.build()
    mgr.step()

    logging_dict = env.extras[env.extras_logging_key]
    assert "Terminations / a" not in logging_dict


def test_step_skips_logging_when_logging_disabled(env):
    mgr = TerminationManager(
        env,
        term_cfg={"a": {"fn": const, "params": {"value": [True] * env.num_envs}}},
        logging_enabled=False,
    )
    mgr.build()
    mgr.step()

    logging_dict = env.extras[env.extras_logging_key]
    assert "Terminations / a" not in logging_dict


"""
build() / reset()
"""


def test_build_builds_each_mdp_fn_config_item(env):
    @dataclass(kw_only=True, eq=False)
    class Counting(MdpFn):
        def build(self):
            self.builds = getattr(self, "builds", 0) + 1

        def __call__(self, env):
            return torch.zeros(env.num_envs, dtype=torch.bool)

    fn = Counting()
    mgr = TerminationManager(env, term_cfg={"a": {"fn": fn}})
    assert getattr(fn, "builds", 0) == 0

    mgr.build()

    assert fn.builds == 1


def test_reset_forwards_to_each_config_items_reset(env):
    @dataclass(kw_only=True, eq=False)
    class Stateful(MdpFn):
        def build(self):
            self.reset_calls = []

        def reset(self, envs_idx):
            self.reset_calls.append(list(envs_idx))

        def __call__(self, env):
            return torch.zeros(env.num_envs, dtype=torch.bool)

    fn = Stateful()
    mgr = TerminationManager(env, term_cfg={"a": {"fn": fn}})
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
            return torch.zeros(env.num_envs, dtype=torch.bool)

    fn = Stateful()
    mgr = TerminationManager(env, term_cfg={"a": {"fn": fn}})
    mgr.build()

    mgr.reset()

    assert fn.reset_calls == [list(range(env.num_envs))]


def test_reset_tolerates_plain_functions(env):
    mgr = TerminationManager(
        env, term_cfg={"a": {"fn": const, "params": {"value": [False] * env.num_envs}}}
    )
    mgr.build()
    mgr.reset(torch.tensor([0]))  # must not raise
