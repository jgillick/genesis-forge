"""Behavior of ContactManager: link resolution, the with-filter, air-time state
tracking, and the buffer lifecycle.

Uses a FakeEntity/FakeScene -- no Genesis scene is built. `_calculate_contact_forces`
(the actual per-step contact-force kernel call) needs a real rigid_solver/collider
and GPU kernel, so it isn't covered here -- see CLAUDE.md's convention of verifying
anything that touches the simulator by running an example (e.g. examples/contacts)
instead. `_calculate_air_time`, the pure-tensor state machine built on top of
`.contacts`, is covered directly since it needs nothing but a contacts tensor.
"""

import pytest
import torch

from genesis_forge.managers import ContactManager


class FakeLink:
    def __init__(self, name, idx, idx_local):
        self.name = name
        self.idx = idx
        self.idx_local = idx_local


class FakeEntity:
    def __init__(self, links):
        self.links = links


class FakeVisOptions:
    def __init__(self, rendered_envs_idx=None):
        self.rendered_envs_idx = rendered_envs_idx


class FakeScene:
    def __init__(self, dt=0.02, vis_options=None):
        self.dt = dt
        self.vis_options = vis_options


def make_robot():
    return FakeEntity(
        [
            FakeLink("FL_foot", idx=100, idx_local=0),
            FakeLink("FR_foot", idx=101, idx_local=1),
            FakeLink("RL_foot", idx=102, idx_local=2),
            FakeLink("base", idx=103, idx_local=3),
        ]
    )


"""
build() -- link resolution
"""


def test_build_resolves_links_in_pattern_list_order_not_entity_order(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=["RL_foot", "FL_foot"])
    mgr.build()

    # "RL_foot" is listed first, even though FL_foot comes first on the entity.
    assert mgr.link_ids.tolist() == [102, 100]
    assert mgr.local_link_ids.tolist() == [2, 0]


def test_build_resolves_a_single_regex_matching_several_links(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    assert mgr.link_ids.tolist() == [100, 101, 102]


def test_build_raises_when_a_pattern_matches_no_links(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=["nonexistent"])
    with pytest.raises(RuntimeError, match="not found"):
        mgr.build()


"""
build() -- the with-filter (contacts against a specific other entity/links)
"""


def test_build_with_entity_alone_selects_every_link_of_that_entity(env):
    env.robot = make_robot()
    env.terrain = FakeEntity([FakeLink("ground", idx=200, idx_local=0)])
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], with_entity=env.terrain)
    mgr.build()

    assert mgr._with_link_ids.tolist() == [200]


def test_build_with_links_names_filters_the_default_robot_entity(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], with_links_names=["base"])
    mgr.build()

    assert mgr._with_link_ids.tolist() == [103]


def test_build_without_a_with_filter_leaves_with_link_ids_empty(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    assert mgr._with_link_ids.numel() == 0


"""
build() -- buffers
"""


def test_build_allocates_contact_buffers_with_the_right_shape(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    assert mgr.contacts.shape == (env.num_envs, 3, 3)
    assert mgr.contact_positions.shape == (env.num_envs, 3, 3)


def test_build_allocates_air_time_buffers_only_when_tracking(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()

    assert mgr.last_air_time.shape == (env.num_envs, 3)
    assert mgr.current_air_time.shape == (env.num_envs, 3)
    assert mgr.last_contact_time.shape == (env.num_envs, 3)
    assert mgr.current_contact_time.shape == (env.num_envs, 3)


def test_build_leaves_air_time_buffers_none_when_not_tracking(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    assert mgr.last_air_time is None
    assert mgr.current_air_time is None


"""
build() -- debug envs_idx resolution
"""


def test_build_debug_envs_idx_falls_back_to_vis_options(env):
    env.robot = make_robot()
    env.scene = FakeScene(vis_options=FakeVisOptions(rendered_envs_idx=[0, 1]))
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    assert mgr.debug_envs_idx == [0, 1]


def test_build_debug_envs_idx_falls_back_to_every_env(env):
    env.robot = make_robot()
    env.scene = FakeScene(vis_options=None)
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    assert mgr.debug_envs_idx == list(range(env.num_envs))


def test_build_debug_envs_idx_prefers_the_explicit_cfg_value(env):
    env.robot = make_robot()
    env.scene = FakeScene(vis_options=FakeVisOptions(rendered_envs_idx=[0, 1]))
    mgr = ContactManager(
        env, link_names=[".*_foot"], debug_visualizer_cfg={"envs_idx": [3]}
    )
    mgr.build()

    assert mgr.debug_envs_idx == [3]


"""
has_made_contact / has_broken_contact
"""


def test_has_made_contact_requires_track_air_time(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    with pytest.raises(RuntimeError, match="track air time"):
        mgr.has_made_contact(dt=0.02)


def test_has_made_contact_true_within_the_time_window(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()
    mgr.current_contact_time = torch.tensor([[0.01, 0.0, 0.05]])

    result = mgr.has_made_contact(dt=0.02)

    assert result.tolist() == [[True, False, False]]


def test_has_broken_contact_requires_track_air_time(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    with pytest.raises(RuntimeError, match="track air time"):
        mgr.has_broken_contact(dt=0.02)


def test_has_broken_contact_true_within_the_time_window(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()
    mgr.current_air_time = torch.tensor([[0.01, 0.0, 0.05]])

    result = mgr.has_broken_contact(dt=0.02)

    assert result.tolist() == [[True, False, False]]


"""
get_contact_forces
"""


def test_get_contact_forces_for_a_single_link(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()
    mgr.contacts = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    assert torch.equal(mgr.get_contact_forces(101), torch.tensor([[[2.0, 0.0, 0.0]]]))


def test_get_contact_forces_for_a_list_of_links(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()
    mgr.contacts = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    result = mgr.get_contact_forces([102, 100])
    assert torch.equal(result, torch.tensor([[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]))


"""
reset()
"""


def test_reset_zeros_air_time_buffers_for_the_given_envs(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()
    mgr.current_air_time[:] = 1.0
    mgr.current_contact_time[:] = 1.0
    mgr.last_air_time[:] = 1.0
    mgr.last_contact_time[:] = 1.0

    mgr.reset(torch.tensor([0]))

    assert torch.all(mgr.current_air_time[0] == 0.0)
    assert torch.all(mgr.current_air_time[1:] == 1.0)


def test_reset_without_track_air_time_is_a_noop(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    mgr.reset()  # must not raise -- no air-time buffers were allocated


def test_reset_is_a_noop_when_disabled(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()
    mgr.current_air_time[:] = 1.0
    mgr.enabled = False

    mgr.reset()

    assert torch.all(mgr.current_air_time == 1.0)


def test_reset_defaults_to_every_env(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()
    mgr.current_air_time[:] = 1.0

    mgr.reset()

    assert torch.all(mgr.current_air_time == 0.0)


"""
step() -- only the enabled/disabled gate is tested here; the contact-force
calculation itself needs a real physics scene and GPU
"""


def test_step_is_a_noop_when_disabled(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    mgr.build()
    mgr.contacts[:] = 1.0
    mgr.enabled = False

    mgr.step()  # must not raise and must not touch any buffers

    assert torch.all(mgr.contacts == 1.0)


"""
_calculate_air_time -- pure tensor state machine over .contacts
"""


def test_calculate_air_time_is_a_noop_when_not_tracking(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"])
    mgr.build()

    mgr._calculate_air_time()  # must not raise, even though the buffers are None


def test_calculate_air_time_records_last_air_time_on_new_contact(env):
    env.robot = make_robot()
    env.scene = FakeScene(dt=0.02)
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True, air_time_contact_threshold=1.0)
    mgr.build()
    mgr.current_air_time[:] = 0.5  # was in the air
    mgr.contacts[0, 0] = torch.tensor([2.0, 0.0, 0.0])  # now above the threshold

    mgr._calculate_air_time()

    assert mgr.last_air_time[0, 0].item() == pytest.approx(0.52)  # 0.5 + dt
    assert mgr.current_air_time[0, 0].item() == 0.0  # reset now that it's in contact


def test_calculate_air_time_increments_while_not_in_contact(env):
    env.robot = make_robot()
    env.scene = FakeScene(dt=0.02)
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True, air_time_contact_threshold=1.0)
    mgr.build()
    mgr.current_air_time[:] = 0.5
    # contacts stay at zero -- below the threshold

    mgr._calculate_air_time()

    assert mgr.current_air_time[0, 0].item() == pytest.approx(0.52)
    assert mgr.current_contact_time[0, 0].item() == 0.0


def test_calculate_air_time_records_last_contact_time_on_detach(env):
    env.robot = make_robot()
    env.scene = FakeScene(dt=0.02)
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True, air_time_contact_threshold=1.0)
    mgr.build()
    mgr.current_contact_time[:] = 0.3  # was in contact
    # contacts stay at zero this step -- now below the threshold (detached)

    mgr._calculate_air_time()

    assert mgr.last_contact_time[0, 0].item() == pytest.approx(0.32)  # 0.3 + dt
    assert mgr.current_contact_time[0, 0].item() == 0.0


def test_calculate_air_time_increments_while_in_contact(env):
    env.robot = make_robot()
    env.scene = FakeScene(dt=0.02)
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True, air_time_contact_threshold=1.0)
    mgr.build()
    mgr.current_contact_time[:] = 0.3
    mgr.contacts[0, 0] = torch.tensor([2.0, 0.0, 0.0])  # stays above the threshold

    mgr._calculate_air_time()

    assert mgr.current_contact_time[0, 0].item() == pytest.approx(0.32)
    assert mgr.current_air_time[0, 0].item() == 0.0


"""
__repr__
"""


def test_repr_includes_the_configured_link_names(env):
    env.robot = make_robot()
    env.scene = FakeScene()
    mgr = ContactManager(env, link_names=[".*_foot"], track_air_time=True)
    assert "link_names=['.*_foot']" in repr(mgr)
    assert "track_air_time=True" in repr(mgr)
