"""Lifecycle and rebuild semantics for the MdpFn dataclass base.

The failure modes covered here are all silent at runtime: a rebuild firing during
construction (before the environment is attached), a build() that recurses into
itself, and a rebuild firing on every step because a function stashes per-step
state on ``self``. Build counts are asserted exactly, never "at least once".
"""

import dataclasses
from dataclasses import dataclass

import pytest
import torch

from genesis_forge.managers.config import MdpFn, ResetMdpFn


@dataclass(kw_only=True, eq=False)
class Counting(MdpFn):
    """Records every build so the tests can assert exact counts."""

    threshold: float = 1.0
    scale: float = 2.0
    label: str = "a"

    def build(self):
        self.builds = getattr(self, "builds", 0) + 1
        self.seen = (self.threshold, self.scale, self.label)
        self._derived = self.threshold * self.scale

    def __call__(self, env):
        return self._derived


"""
Construction
"""


def test_fields_are_keyword_only():
    with pytest.raises(TypeError):
        Counting(1.0)


def test_unknown_field_rejected():
    with pytest.raises(TypeError):
        Counting(nonexistent=1.0)


def test_construction_does_not_build(env):
    fn = Counting(threshold=3.0)
    assert getattr(fn, "builds", 0) == 0


def test_assignment_before_context_does_not_build(env):
    fn = Counting()
    fn.threshold = 9.0
    assert getattr(fn, "builds", 0) == 0
    assert fn.threshold == 9.0


def test_env_access_before_context_raises():
    fn = Counting()
    with pytest.raises(RuntimeError, match="not bound"):
        fn.env # noqa: B018


"""
context() and safe_build() -- the two steps the manager drives separately
"""


def test_context_alone_does_not_build(env):
    fn = Counting(threshold=3.0)
    fn.context(env)
    assert getattr(fn, "builds", 0) == 0
    assert fn.env is env


def test_safe_build_after_context_builds_once(env):
    """build() assigns self.builds, self.seen and self._derived every time; if any
    of those re-entered the hook this would already have blown the stack."""
    fn = Counting(threshold=3.0)
    fn.context(env)
    fn.safe_build()
    assert fn.builds == 1
    assert fn(env) == 6.0


def test_context_rejects_unexpected_kwarg(env):
    fn = Counting()
    with pytest.raises(TypeError):
        fn.context(env, entity=object())


def test_build_can_read_env(env):
    @dataclass(kw_only=True, eq=False)
    class Buffered(MdpFn):
        width: int = 2

        def build(self):
            self.buf = torch.zeros((self.env.num_envs, self.width))

        def __call__(self, env):
            return self.buf

    fn = Buffered()
    fn.context(env)
    fn.safe_build()
    assert fn.buf.shape == (env.num_envs, 2)


"""
Rebuild on param change
"""


def test_field_assignment_rebuilds_once_with_new_value(env):
    fn = Counting(threshold=3.0)
    fn.context(env)
    fn.safe_build()
    fn.threshold = 5.0
    assert fn.builds == 2
    assert fn.seen == (5.0, 2.0, "a")
    assert fn(env) == 10.0


@pytest.mark.parametrize("name", ["_scratch", "scratch"])
def test_non_field_assignment_never_rebuilds(env, name):
    """The guard must key on declared fields, not on an underscore convention."""
    fn = Counting()
    fn.context(env)
    fn.safe_build()
    setattr(fn, name, torch.zeros(3))
    assert fn.builds == 1


def test_build_assigning_a_declared_field_does_not_recurse(env):
    @dataclass(kw_only=True, eq=False)
    class Normalizing(MdpFn):
        angle: float = 0.0

        def build(self):
            self.builds = getattr(self, "builds", 0) + 1
            self.angle = self.angle % 360

        def __call__(self, env):
            return self.angle

    fn = Normalizing(angle=400.0)
    fn.context(env)
    fn.safe_build()
    assert fn.builds == 1
    assert fn.angle == 40.0

    fn.angle = 725.0
    assert fn.builds == 2
    assert fn.angle == 5.0


def test_rebuild_preserves_instance_identity_and_untouched_buffers(env):
    @dataclass(kw_only=True, eq=False)
    class History(MdpFn):
        gain: float = 1.0

        def build(self):
            # Only allocate once — a param change must not wipe history.
            if not hasattr(self, "history"):
                self.history = torch.zeros(self.env.num_envs)
            self.doubled = self.gain * 2

        def __call__(self, env):
            return self.history * self.doubled

    fn = History()
    fn.context(env)
    fn.safe_build()
    before = fn.history
    fn.history[:] = 7.0

    fn.gain = 3.0

    assert fn.history is before
    assert torch.equal(fn.history, torch.full((env.num_envs,), 7.0))
    assert fn.doubled == 6.0


"""
Batched updates
"""


def test_update_rebuilds_once_for_multiple_fields(env):
    fn = Counting(threshold=1.0, scale=1.0, label="a")
    fn.context(env)
    fn.safe_build()
    fn.update(threshold=4.0, scale=5.0, label="z")
    assert fn.builds == 2
    # All three values are visible within that single build — never a partial set.
    assert fn.seen == (4.0, 5.0, "z")


def test_update_before_first_build_sets_without_building(env):
    fn = Counting()
    fn.update(threshold=8.0)
    assert getattr(fn, "builds", 0) == 0
    assert fn.threshold == 8.0


def test_update_rejects_unknown_field(env):
    fn = Counting()
    fn.context(env)
    fn.safe_build()
    with pytest.raises(AttributeError, match="nonexistent"):
        fn.update(nonexistent=1.0)
    assert fn.builds == 1


def test_update_with_no_arguments_does_not_rebuild(env):
    fn = Counting()
    fn.context(env)
    fn.safe_build()
    fn.update()
    assert fn.builds == 1


"""
Reset
"""


def test_reset_defaults_to_noop(env):
    fn = Counting()
    fn.context(env)
    fn.safe_build()
    fn.reset(torch.tensor([0, 1]))


def test_reset_receives_indices(env):
    @dataclass(kw_only=True, eq=False)
    class Resettable(MdpFn):
        def build(self):
            self.buf = torch.ones(self.env.num_envs)

        def reset(self, envs_idx):
            self.buf[envs_idx] = 0.0

        def __call__(self, env):
            return self.buf

    fn = Resettable()
    fn.context(env)
    fn.safe_build()
    fn.reset(torch.tensor([0, 2]))
    assert torch.equal(fn.buf, torch.tensor([0.0, 1.0, 0.0, 1.0]))


"""
Defaults and identity
"""


def test_default_build_and_call_are_safe_to_inherit(env):
    @dataclass(kw_only=True, eq=False)
    class Stateless(MdpFn):
        weight: float = 1.0

    fn = Stateless()
    fn.context(env)
    fn.safe_build()  # inherited no-op build must not raise
    with pytest.raises(NotImplementedError):
        fn(env)


def test_init_subclass_is_a_backstop_for_a_forgotten_decorator():
    """Always write @dataclass(kw_only=True, eq=False) yourself. griffe (which builds
    the published API reference) only recognizes the literal decorator token in source,
    so an undecorated class renders an empty constructor signature there even though it
    still behaves correctly at runtime -- fields are registered, kw_only is enforced,
    and eq=False still applies, all via __init_subclass__.
    """

    class Forgotten(MdpFn):
        threshold: float = 1.0

    assert [f.name for f in dataclasses.fields(Forgotten)] == ["threshold"]
    assert Forgotten(threshold=5.0).threshold == 5.0
    with pytest.raises(TypeError):
        Forgotten(5.0)  # kw_only still enforced

    a, b = Forgotten(threshold=1.0), Forgotten(threshold=1.0)
    assert a != b and len({a, b}) == 2  # eq=False still enforced


def test_explicit_dataclass_can_override_the_automatic_eq_false():
    """Documents why eq=False must be written explicitly, not assumed: value equality
    and unhashability are one wrong decorator away, even with the backstop in place.

    A decorator on the class statement runs after __init_subclass__ and wins,
    regenerating __eq__ and setting __hash__ = None regardless of what the automatic
    pass chose. Nothing in the framework hashes or compares a config fn today, so this
    is a footgun to avoid, not a constraint.
    """

    @dataclass(kw_only=True)
    class Defaulted(MdpFn):
        threshold: float = 1.0

    assert Defaulted(threshold=1.0) == Defaulted(threshold=1.0)
    with pytest.raises(TypeError, match="unhashable"):
        {Defaulted()}


"""
ResetMdpFn
"""


def test_reset_fn_context_stores_entity_before_build_runs(env):
    entity = object()

    @dataclass(kw_only=True, eq=False)
    class Placer(ResetMdpFn):
        height: float = 0.5

        def build(self):
            self.builds = getattr(self, "builds", 0) + 1
            self.target = self.entity

        def __call__(self, env, entity, envs_idx):
            return self.height

    fn = Placer()
    fn.context(env, entity=entity)
    assert getattr(fn, "builds", 0) == 0  # context() alone still does not build
    fn.safe_build()
    assert fn.entity is entity
    assert fn.target is entity  # build() could already read self.entity
    assert fn.builds == 1


def test_reset_fn_rebuilds_on_param_change(env):
    @dataclass(kw_only=True, eq=False)
    class Placer(ResetMdpFn):
        height: float = 0.5

        def build(self):
            self.builds = getattr(self, "builds", 0) + 1

        def __call__(self, env, entity, envs_idx):
            return self.height

    fn = Placer()
    fn.context(env, entity=object())
    fn.safe_build()
    fn.height = 1.5
    assert fn.builds == 2


def test_reset_fn_entity_access_before_context_raises():
    @dataclass(kw_only=True, eq=False)
    class Placer(ResetMdpFn):
        def __call__(self, env, entity, envs_idx):
            return None

    with pytest.raises(RuntimeError, match="not bound"):
        Placer().entity # noqa: B018
