"""Runtime mutation of reward params, after the config item has already been built.

``increment_weight`` and ``increment_param`` let an environment adjust a reward's
weight or parameters mid-training -- a curriculum is one reason to do this, but not
the only one. They must behave identically whether the reward is a plain function
with a params dict or an :class:`MdpFn` with typed fields -- and in the MdpFn case
must inherit the single-rebuild guarantee.
"""

from dataclasses import dataclass

import pytest

from genesis_forge.managers.config import MdpFn, RewardConfigItem


@dataclass(kw_only=True, eq=False)
class Tracked(MdpFn):
    threshold: float = 1.0
    sensitivity: float = 0.25

    def build(self):
        self.builds = getattr(self, "builds", 0) + 1
        self.seen = (self.threshold, self.sensitivity)

    def __call__(self, env):
        return self.threshold * self.sensitivity


def plain(env, threshold=1.0, sensitivity=0.25):
    return threshold * sensitivity


@pytest.fixture
def mdp_item(env):
    item = RewardConfigItem({"fn": Tracked(threshold=1.0), "weight": -2.0}, env)
    item.build()
    return item


@pytest.fixture
def dict_item(env):
    item = RewardConfigItem(
        {"fn": plain, "params": {"threshold": 1.0, "sensitivity": 0.25}, "weight": -2.0},
        env,
    )
    item.build()
    return item


"""
increment_param — MdpFn fields
"""


def test_increment_param_rebuilds_once_and_takes_effect(mdp_item, env):
    returned = mdp_item.increment_param("threshold", 0.5)

    assert returned == 1.5
    assert mdp_item.fn.threshold == 1.5
    assert mdp_item.fn.builds == 2  # one at build, one at the increment
    assert mdp_item.fn.seen == (1.5, 0.25)
    assert mdp_item.fn(env) == 1.5 * 0.25


def test_repeated_increments_accumulate(mdp_item):
    for _ in range(4):
        mdp_item.increment_param("threshold", 0.25)
    assert mdp_item.fn.threshold == 2.0
    assert mdp_item.fn.builds == 5


def test_increment_param_clamps_upward(mdp_item):
    assert mdp_item.increment_param("threshold", 5.0, limit=2.0) == 2.0
    assert mdp_item.fn.threshold == 2.0


def test_increment_param_clamps_downward(mdp_item):
    assert mdp_item.increment_param("threshold", -5.0, limit=0.1) == 0.1
    assert mdp_item.fn.threshold == 0.1


def test_increment_param_below_limit_is_untouched(mdp_item):
    assert mdp_item.increment_param("threshold", 0.2, limit=10.0) == 1.2


def test_increment_unknown_param_raises(mdp_item):
    with pytest.raises(AttributeError, match="nonexistent"):
        mdp_item.increment_param("nonexistent", 1.0)
    assert mdp_item.fn.builds == 1


"""
increment_param — plain function params dict, unchanged behavior
"""


def test_dict_increment_param_takes_effect(dict_item):
    assert dict_item.increment_param("threshold", 0.5) == 1.5
    assert dict_item.params["threshold"] == 1.5


def test_dict_increment_param_clamps_both_directions(dict_item):
    assert dict_item.increment_param("threshold", 5.0, limit=2.0) == 2.0
    assert dict_item.increment_param("threshold", -5.0, limit=0.5) == 0.5


def test_dict_increment_unknown_param_raises(dict_item):
    with pytest.raises(KeyError):
        dict_item.increment_param("nonexistent", 1.0)


"""
Bulk replacement
"""


def test_bulk_params_replacement_rebuilds_once(mdp_item):
    mdp_item.params = {"threshold": 4.0, "sensitivity": 0.5}

    assert mdp_item.fn.builds == 2
    # Both values were in place for that single build — never a partial set.
    assert mdp_item.fn.seen == (4.0, 0.5)


def test_bulk_params_replacement_on_dict_item(dict_item):
    dict_item.params = {"threshold": 4.0, "sensitivity": 0.5}
    assert dict_item.params["threshold"] == 4.0
    assert dict_item.params["sensitivity"] == 0.5


def test_params_dict_is_read_only_for_mdp_fn(mdp_item):
    """Writing through .params would silently do nothing — fail loudly instead."""
    with pytest.raises(TypeError):
        mdp_item.params["threshold"] = 5.0


"""
increment_weight — independent of fn kind
"""


@pytest.mark.parametrize("item_name", ["mdp_item", "dict_item"])
def test_increment_weight_is_independent_of_fn_kind(item_name, request):
    item = request.getfixturevalue(item_name)
    assert item.increment_weight(0.5) == -1.5
    assert item.weight == -1.5


def test_increment_weight_clamps(mdp_item):
    assert mdp_item.increment_weight(5.0, limit=-1.0) == -1.0
    assert mdp_item.increment_weight(-5.0, limit=-3.0) == -3.0


def test_increment_weight_does_not_rebuild(mdp_item):
    mdp_item.increment_weight(0.5)
    assert mdp_item.fn.builds == 1
