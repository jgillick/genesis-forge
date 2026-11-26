"""Behavior of EventManager: per-environment timers, durations, manual triggers, and
the start/step/end lifecycle of its functions. No Genesis scene is built.

The env's dt is 0.02, so 0.1 seconds is 5 steps.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from genesis_forge.managers import DurationEventMdpFn, EventManager, EventMdpFn, MdpFn


@dataclass(kw_only=True, eq=False)
class recorder(DurationEventMdpFn):
    """Records which envs each hook was called for."""

    log: list = field(default_factory=list)

    def start(self, envs_idx):
        self.log.append(("start", envs_idx.tolist()))

    def step(self, envs_idx):
        self.log.append(("step", envs_idx.tolist()))

    def end(self, envs_idx):
        self.log.append(("end", envs_idx.tolist()))

    def reset(self, envs_idx):
        self.log.append(("reset", envs_idx.tolist()))


@dataclass(kw_only=True, eq=False)
class instant(EventMdpFn):
    def __call__(self, env, envs_idx):
        pass


@dataclass(kw_only=True, eq=False)
class not_an_event(MdpFn):
    def __call__(self, env):
        return torch.zeros(env.num_envs)


def make(env, fn=None, interval_range_s=None, **cfg):
    """A recorder event lasts for a single step, unless a duration is passed."""
    if fn is None:
        fn = recorder()
        cfg.setdefault("duration_range_s", 0.02)
    event: Any = {"fn": fn, "interval_range_s": interval_range_s, **cfg}
    mgr = EventManager(env, cfg={"e": event})
    mgr.build()
    return mgr, fn


def steps(mgr, count):
    for _ in range(count):
        mgr.step()


ALL = [0, 1, 2, 3]


"""
Timers
"""


def test_fires_after_the_interval(env):
    mgr, fn = make(env, interval_range_s=0.1)
    steps(mgr, 4)
    assert fn.log == []
    mgr.step()
    assert fn.log == [("start", ALL), ("step", ALL)]


def test_the_interval_is_counted_from_the_end_of_the_event(env):
    mgr, fn = make(env, interval_range_s=0.1, duration_range_s=0.06)
    active = []
    for _ in range(14):
        mgr.step()
        active.append(int(mgr.get_active("e")[0]))
    # 5 steps of waiting, 3 active, then 5 idle
    assert active == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]


def test_intervals_are_sampled_per_environment_within_the_range(env):
    env.num_envs = 500
    fired = []
    mgr, _ = make(
        env,
        fn=lambda env, envs_idx: fired.append(envs_idx),
        interval_range_s=(0.1, 0.2),
    )
    steps(mgr, 4)
    assert fired == []
    steps(mgr, 6)
    first = torch.cat(fired)
    assert len(fired) > 1
    assert first.unique().numel() == 500


def test_durations_are_sampled_per_environment_within_the_range(env):
    env.num_envs = 500
    mgr, _ = make(env, duration_range_s=(0.1, 0.2))
    mgr.trigger("e")
    active_steps = torch.zeros(500)
    for _ in range(12):
        mgr.step()
        active_steps += mgr.get_active("e")
    assert active_steps.min() >= 5
    assert active_steps.max() <= 10
    assert active_steps.unique().numel() > 1


def test_a_changed_interval_is_used_for_the_next_wait(env):
    mgr, fn = make(env, interval_range_s=0.1)
    mgr["e"].interval_range_s = 0.04
    steps(mgr, 4)
    assert fn.log == []
    steps(mgr, 1)  # fires
    fn.log.clear()
    steps(mgr, 3)  # ends, then 2 steps of waiting
    assert fn.log == [("end", ALL), ("start", ALL), ("step", ALL)]


def test_an_interval_can_be_turned_off_and_on(env):
    mgr, fn = make(env, interval_range_s=0.04)
    mgr["e"].interval_range_s = None
    steps(mgr, 10)
    assert fn.log == []
    mgr["e"].interval_range_s = 0.04
    steps(mgr, 2)
    assert fn.log[0] == ("start", ALL)


"""
Function lifecycle
"""


def test_start_is_called_once_and_step_every_active_step(env):
    mgr, fn = make(env, duration_range_s=0.06)
    mgr.trigger("e", [1, 2])
    steps(mgr, 5)
    assert fn.log == [
        ("start", [1, 2]),
        ("step", [1, 2]),
        ("step", [1, 2]),
        ("step", [1, 2]),
        ("end", [1, 2]),
    ]


def test_plain_functions_receive_their_params(env):
    calls = []

    def fn(env, envs_idx, size):
        calls.append((envs_idx.tolist(), size))

    mgr = EventManager(
        env, cfg={"e": {"fn": fn, "params": {"size": 3}, "interval_range_s": None}}
    )
    mgr.build()
    mgr.trigger("e", [2])
    mgr.step()
    assert calls == [([2], 3)]


"""
Reset and disable
"""


def test_reset_ends_active_events_only_in_the_reset_envs(env):
    mgr, fn = make(env, duration_range_s=1.0)
    mgr.trigger("e", [0, 1])
    mgr.step()
    fn.log.clear()
    mgr.reset(torch.tensor([1, 2]))
    assert fn.log == [("end", [1]), ("reset", [1, 2])]
    assert mgr.get_active("e").tolist() == [True, False, False, False]


def test_reset_restarts_the_wait(env):
    mgr, fn = make(env, interval_range_s=0.1)
    steps(mgr, 4)
    mgr.reset(torch.tensor([0]))
    mgr.step()
    assert fn.log[1] == ("start", [1, 2, 3])


def test_reset_discards_a_pending_trigger(env):
    mgr, fn = make(env)
    mgr.trigger("e")
    mgr.reset(torch.tensor([0, 1, 2]))
    mgr.step()
    assert ("start", [3]) in fn.log


def test_disabling_ends_events_pauses_timers_and_ignores_triggers(env):
    mgr, fn = make(env, interval_range_s=0.1, duration_range_s=1.0)
    steps(mgr, 3)
    mgr.trigger("e", [0])
    mgr.step()
    mgr.enabled = False
    fn.log.clear()
    mgr.trigger("e")
    steps(mgr, 20)
    assert fn.log == [("end", [0])]
    mgr.enabled = True
    mgr.step()
    # The rest had one step left to wait
    assert fn.log[1] == ("start", [1, 2, 3])


"""
Triggers
"""


def test_trigger_restarts_an_active_event(env):
    mgr, fn = make(env, duration_range_s=0.06)
    mgr.trigger("e", [0])
    steps(mgr, 2)
    fn.log.clear()
    mgr.trigger("e", [0])
    mgr.step()
    assert fn.log == [("end", [0]), ("start", [0]), ("step", [0])]
    steps(mgr, 2)
    assert mgr.get_active("e")[0]
    mgr.step()
    assert not mgr.get_active("e")[0]


def test_trigger_an_unknown_event(env):
    mgr, _ = make(env)
    with pytest.raises(KeyError):
        mgr.trigger("nope")


"""
Config validation
"""


@pytest.mark.parametrize("fn", [instant(), lambda env, envs_idx: None])
def test_instant_functions_reject_a_duration(env, fn):
    with pytest.raises(ValueError, match="DurationEventMdpFn"):
        make(env, fn=fn, duration_range_s=0.1)


def test_duration_functions_require_a_duration(env):
    with pytest.raises(ValueError, match="duration_range_s is required"):
        make(env, fn=recorder())


def test_a_duration_cannot_be_removed_at_runtime(env):
    mgr, _ = make(env)
    with pytest.raises(ValueError, match="duration_range_s is required"):
        mgr["e"].duration_range_s = None


def test_a_duration_function_with_call_is_rejected(env):
    class bad(DurationEventMdpFn):
        def __call__(self, env, envs_idx):
            pass

    with pytest.raises(TypeError, match=r"Implement start\(\)"):
        make(env, fn=bad(), duration_range_s=0.1)


def test_a_duration_function_must_implement_something(env):
    class bad(DurationEventMdpFn):
        pass

    with pytest.raises(TypeError, match="at least one"):
        make(env, fn=bad(), duration_range_s=0.1)


def test_an_instant_function_must_implement_call(env):
    class bad(EventMdpFn):
        def step(self, envs_idx):
            pass

    with pytest.raises(TypeError, match=r"must implement __call__\(\)"):
        make(env, fn=bad())


def test_an_instant_function_is_called_once_and_active_for_one_step(env):
    calls = []

    class kick(EventMdpFn):
        def __call__(self, env, envs_idx):
            calls.append(envs_idx.tolist())

    mgr, _ = make(env, fn=kick())
    mgr.trigger("e", [3])
    mgr.step()
    assert mgr.get_active("e").tolist() == [False, False, False, True]
    mgr.step()
    assert not mgr.get_active("e").any()
    assert calls == [[3]]


def test_without_an_interval_it_only_fires_when_triggered(env):
    fired = []
    mgr = EventManager(
        env, cfg={"e": {"fn": lambda env, envs_idx: fired.append(envs_idx.tolist())}}
    )
    mgr.build()
    steps(mgr, 100)
    assert fired == []
    mgr.trigger("e", [2])
    mgr.step()
    assert fired == [[2]]
    mgr.reset()
    steps(mgr, 100)
    assert fired == [[2]]


def test_other_mdp_functions_are_rejected(env):
    with pytest.raises(TypeError, match="EventMdpFn"):
        make(env, fn=not_an_event())


@pytest.mark.parametrize(
    "value", ["fast", (1.0,), (2.0, 1.0), (-1.0, 1.0), 0, (0, float("inf"))]
)
def test_invalid_times_are_rejected(env, value):
    with pytest.raises(ValueError, match="interval_range_s must be"):
        make(env, interval_range_s=value)


"""
Debug visualizer
"""


class FakeScene:
    """Records the debug objects that are currently drawn."""

    def __init__(self, rendered_envs_idx=None):
        self.vis_options = SimpleNamespace(rendered_envs_idx=rendered_envs_idx)
        self.nodes = {}
        self.count = 0

    def draw_debug_mesh(self, mesh):
        self.count += 1
        self.nodes[self.count] = mesh
        return self.count

    def clear_debug_object(self, node):
        del self.nodes[node]


@dataclass(kw_only=True, eq=False)
class drawn_event(EventMdpFn):
    """Draws a plain object for each env."""

    def __call__(self, env, envs_idx):
        pass

    def debug_draw(self, envs_idx):
        return [self.env.scene.draw_debug_mesh(int(i)) for i in envs_idx] + [None]


def make_debug(env, fn=None, enabled=True, rendered_envs_idx=None, **debug_cfg: Any):
    env.scene = FakeScene(rendered_envs_idx)
    visualizer_cfg: Any = {"fps": 50, **debug_cfg}
    mgr = EventManager(
        env,
        cfg={"e": {"fn": fn or drawn_event(), "interval_range_s": None}},
        debug_visualizer=enabled,
        debug_visualizer_cfg=visualizer_cfg,
    )
    mgr.build()
    return mgr


def test_an_event_is_drawn_for_at_least_the_minimum_time(env):
    mgr = make_debug(env, envs_idx=[1], min_display_s=0.06)
    mgr.trigger("e")
    drawn = []
    for _ in range(5):
        mgr.step()
        drawn.append(list(env.scene.nodes.values()))
    assert drawn == [[1], [1], [1], [], []]


def test_events_are_drawn_in_every_rendered_env_by_default(env):
    mgr = make_debug(env)
    mgr.trigger("e")
    mgr.step()
    assert len(env.scene.nodes) == env.num_envs


def test_reset_removes_the_drawings_in_the_reset_envs(env):
    mgr = make_debug(env)
    mgr.trigger("e")
    mgr.step()
    mgr.reset(torch.tensor([0, 2]))
    assert len(env.scene.nodes) == 2


def test_debug_visualizer_can_be_turned_off_and_on(env):
    mgr = make_debug(env, enabled=False)
    mgr.trigger("e")
    mgr.step()
    assert env.scene.nodes == {}
    mgr.debug_visualizer = True
    mgr.trigger("e")
    mgr.step()
    assert len(env.scene.nodes) == env.num_envs
    mgr.debug_visualizer = False
    mgr.step()
    assert env.scene.nodes == {}


def test_events_without_debug_draw_are_not_drawn(env):
    mgr = make_debug(env, fn=instant())
    mgr.trigger("e")
    mgr.step()
    assert env.scene.nodes == {}


def test_debug_envs_must_be_rendered(env):
    mgr = make_debug(env, envs_idx=[1, 2], rendered_envs_idx=[0, 1])
    mgr.trigger("e")
    with pytest.raises(ValueError, match=r"not rendered: \[2\]"):
        mgr.step()


def test_reset_of_environments_that_are_not_visualized_does_not_redraw(env):
    mgr = make_debug(env, envs_idx=[1])
    mgr.trigger("e")
    mgr.step()
    count = env.scene.count
    mgr.reset(torch.tensor([0, 2]))
    assert env.scene.count == count


def test_debug_rendering_is_throttled_to_the_fps(env):
    mgr = make_debug(env, envs_idx=[1], fps=25)  # every 2 steps
    mgr.trigger("e")
    drawn = []
    for step in range(4):
        env.step_count = step
        mgr.step()
        drawn.append(env.scene.count)
    assert drawn == [1, 1, 2, 2]
