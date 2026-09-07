"""ConfigItem dispatch across the two kinds of ``fn`` a manager config may carry.

A config item's ``fn`` is either a plain function paired with a params dict, or a
constructed :class:`MdpFn` instance. Each has a different lifecycle, and the managers
must be able to drive both through the same interface.
"""

from dataclasses import dataclass

import pytest
import torch

from genesis_forge.managers.config import ConfigItem, MdpFn, ResetMdpFn


def plain_reward(env, threshold=1.0, scale=2.0):
    return ("plain", env, threshold, scale)


@dataclass(kw_only=True, eq=False)
class DataclassReward(MdpFn):
    threshold: float = 1.0
    scale: float = 2.0

    def build(self):
        self.builds = getattr(self, "builds", 0) + 1
        self.env_at_build = self._env
        self.derived = self.threshold * self.scale

    def reset(self, envs_idx):
        self.reset_calls = getattr(self, "reset_calls", []) + [list(envs_idx)]

    def __call__(self, env, *args, **kwargs):
        return ("mdp_fn", env, args, kwargs, self.derived)


"""
Plain function
"""


def test_plain_function_params_are_splatted(env):
    item = ConfigItem({
        "fn": plain_reward,
        "params": {"threshold": 5.0}
    }, env)
    item.build()
    assert item.execute() == ("plain", env, 5.0, 2.0)


def test_plain_function_reset_is_a_noop(env):
    item = ConfigItem({"fn": plain_reward, "params": {}}, env)
    item.build()
    item.reset(torch.tensor([0, 1]))  # must not raise


def test_plain_function_without_params_key(env):
    item = ConfigItem({"fn": plain_reward}, env)
    item.build()
    assert item.execute() == ("plain", env, 1.0, 2.0)


def test_plain_function_params_setter_replaces_the_dict(env):
    item = ConfigItem({"fn": plain_reward, "params": {"threshold": 1.0}}, env)
    item.build()
    item.params = {"threshold": 9.0}
    assert item.execute() == ("plain", env, 9.0, 2.0)


"""
MdpFn instance
"""


def test_mdp_fn_is_bound_before_build_and_built_once(env):
    fn = DataclassReward(threshold=3.0)
    item = ConfigItem({"fn": fn}, env)
    assert getattr(fn, "builds", 0) == 0

    item.build()

    assert fn.builds == 1
    assert fn.env_at_build is env


def test_mdp_fn_params_read_empty_so_managers_do_not_splat(env):
    fn = DataclassReward(threshold=3.0)
    item = ConfigItem({"fn": fn}, env)
    item.build()

    assert dict(item.params) == {}
    # execute() is what every manager calls; confirm an empty params dict still splats cleanly.
    kind, called_env, args, kwargs, derived = item.execute()
    assert (kind, called_env, args, kwargs, derived) == ("mdp_fn", env, (), {}, 6.0)


def test_mdp_fn_reset_is_forwarded(env):
    """Regression: instance-style functions were skipped entirely by reset()."""
    fn = DataclassReward()
    item = ConfigItem({"fn": fn}, env)
    item.build()

    item.reset(torch.tensor([0, 2]))

    assert fn.reset_calls == [[0, 2]]


def test_mdp_fn_with_empty_params_key_is_allowed(env):
    fn = DataclassReward()
    item = ConfigItem({"fn": fn, "params": {}}, env)
    item.build()
    assert fn.builds == 1


def test_mdp_fn_with_populated_params_raises(env):
    """Params belong in the constructor now; a dict alongside is a silent-drop trap."""
    fn = DataclassReward()
    with pytest.raises(ValueError, match="params"):
        ConfigItem({"fn": fn, "params": {"threshold": 5.0}}, env)


"""
MdpFn on the entity-reset path
"""


def test_reset_mdp_fn_receives_entity_and_envs_idx(env):
    entity = object()

    @dataclass(kw_only=True, eq=False)
    class Placer(ResetMdpFn):
        height: float = 0.5

        def __call__(self, env, entity, envs_idx):
            self.calls = getattr(self, "calls", []) + [(env, entity, list(envs_idx))]

    fn = Placer()
    item = ConfigItem({"fn": fn}, env)
    item.build(entity=entity)

    item.execute(envs_idx=[1, 3])

    assert fn.entity is entity
    assert fn.calls == [(env, entity, [1, 3])]


"""
Passing a class that isn't an MdpFn (e.g. an old MdpFnClass/ResetMdpFnClass subclass)
"""


def test_class_based_fn_raises_a_clear_not_callable_error(env):
    class OldStyleReward:
        def __init__(self, env, threshold=1.0):
            self.env = env

        def __call__(self, env, envs_idx, threshold=1.0):
            return threshold

    with pytest.raises(TypeError, match="not a callable instance"):
        ConfigItem({"fn": OldStyleReward, "params": {"threshold": 1.0}}, env)


"""
Classification
"""


def test_each_kind_is_classified_distinctly(env):
    plain = ConfigItem({"fn": plain_reward}, env)
    instance = ConfigItem({"fn": DataclassReward()}, env)

    assert plain.is_mdp_class_fn is False
    assert instance.is_mdp_class_fn is True
