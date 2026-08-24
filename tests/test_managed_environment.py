"""Behavior of ManagedEnvironment: manager registration rules, the build/step/reset
orchestration order, action/observation space merging, and the auto-reset-on-done
step() behavior.

Uses "recording" BaseManager subclasses as test doubles for every manager type --
each just appends to a shared log and returns the minimum a real manager of that
type would. This isolates ManagedEnvironment's own orchestration logic from any
particular manager's internals (already covered by their own test files). No
Genesis scene is built: `env.scene` is a bare fake exposing only build()/step().
"""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from genesis_forge import ManagedEnvironment
from genesis_forge.managers.base import BaseManager


class FakeScene:
    def build(self, n_envs):
        pass

    def step(self):
        pass


class ConfigurableEnv(ManagedEnvironment):
    """A ManagedEnvironment whose manager registration is supplied per-test."""

    def __init__(self, configure=None, **kwargs):
        self._configure = configure or (lambda env: None)
        super().__init__(**kwargs)
        self.scene = FakeScene()

    def config(self):
        self._configure(self)


class RecordingManager(BaseManager):
    """A generic manager double for the types build()/step()/reset() just loop over
    without reading any extra properties (terrain, actuator, contact, termination-
    adjacent bookkeeping aside, command, entity)."""

    def __init__(self, env, type, log):
        super().__init__(env, type=type)
        self.log = log

    def build(self):
        self.log.append(("build", self.type))

    def step(self):
        self.log.append(("step", self.type))

    def reset(self, envs_idx=None):
        self.log.append(("reset", self.type, envs_idx))


class RecordingActionManager(BaseManager):
    """`_build_action_managers` also reads `.action_space` to merge action spaces."""

    def __init__(self, env, log, low=-1.0, high=1.0, size=1):
        super().__init__(env, type="action")
        self.log = log
        self.action_space = spaces.Box(
            low=np.full(size, low, dtype=np.float32),
            high=np.full(size, high, dtype=np.float32),
            dtype=np.float32,
        )
        self.received_actions = None

    def build(self):
        self.log.append(("build", "action"))

    def step(self, actions):
        self.log.append(("step", "action"))
        self.received_actions = actions
        return actions

    def send_actions_to_simulation(self, actions):
        self.log.append(("send_actions", "action"))

    def reset(self, envs_idx=None):
        self.log.append(("reset", "action", envs_idx))


class RecordingObservationManager(BaseManager):
    """`_build_observation_managers`/`get_observations` also read `.name` and
    `.observation_space`."""

    def __init__(self, env, log, name="policy", size=2, value=0.0):
        super().__init__(env, type="observation")
        self.log = log
        self.name = name
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32
        )
        self._value = value

    def build(self):
        self.log.append(("build", "observation"))

    def get_observations(self):
        self.log.append(("get_observations", self.name))
        return torch.full((self.env.num_envs, self.observation_space.shape[0]), self._value)

    def reset(self, envs_idx=None):
        self.log.append(("reset", "observation", envs_idx))


class RecordingTerminationManager(BaseManager):
    def __init__(self, env, log, terminated=None, truncated=None):
        super().__init__(env, type="termination")
        self.log = log
        self._terminated = (
            terminated if terminated is not None else torch.zeros(env.num_envs, dtype=torch.bool)
        )
        self._truncated = (
            truncated if truncated is not None else torch.zeros(env.num_envs, dtype=torch.bool)
        )

    def build(self):
        self.log.append(("build", "termination"))

    def step(self):
        self.log.append(("step", "termination"))
        return self._terminated, self._truncated

    def reset(self, envs_idx=None):
        self.log.append(("reset", "termination", envs_idx))


class RecordingRewardManager(BaseManager):
    def __init__(self, env, log, reward=None):
        super().__init__(env, type="reward")
        self.log = log
        self._reward = reward if reward is not None else torch.zeros(env.num_envs)

    def build(self):
        self.log.append(("build", "reward"))

    def step(self):
        self.log.append(("step", "reward"))
        return self._reward

    def reset(self, envs_idx=None):
        self.log.append(("reset", "reward", envs_idx))


"""
add_manager()
"""


def test_add_manager_rejects_an_unknown_type():
    env = ConfigurableEnv(num_envs=4)
    with pytest.raises(ValueError, match="not a valid manager type"):
        env.add_manager("nonexistent", object())


def test_add_manager_appends_to_list_type_slots():
    log = []

    def configure(env):
        RecordingManager(env, "contact", log)
        RecordingManager(env, "contact", log)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    assert len(env.managers["contact"]) == 2


def test_add_manager_rejects_a_second_singleton_manager():
    log = []

    def configure(env):
        RecordingRewardManager(env, log)
        RecordingRewardManager(env, log)  # second reward manager

    env = ConfigurableEnv(configure=configure, num_envs=4)
    with pytest.raises(ValueError, match="already has a manager"):
        env.build()


"""
build() -- orchestration order
"""


def test_build_calls_managers_in_the_documented_order():
    log = []

    def configure(env):
        RecordingManager(env, "terrain", log)
        RecordingManager(env, "actuator", log)
        RecordingActionManager(env, log)
        RecordingManager(env, "contact", log)
        RecordingTerminationManager(env, log)
        RecordingRewardManager(env, log)
        RecordingManager(env, "command", log)
        RecordingManager(env, "entity", log)
        RecordingObservationManager(env, log)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    order = [entry[1] for entry in log if entry[0] == "build"]
    assert order == [
        "terrain",
        "actuator",
        "action",
        "contact",
        "termination",
        "reward",
        "command",
        "entity",
        "observation",
    ]


"""
build() -- action space merging
"""


def test_build_merges_multiple_action_manager_spaces():
    log = []

    def configure(env):
        RecordingActionManager(env, log, low=-1.0, high=1.0, size=2)
        RecordingActionManager(env, log, low=-2.0, high=2.0, size=3)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    assert env.action_space.shape == (5,)
    assert np.allclose(env.action_space.low, [-1.0, -1.0, -2.0, -2.0, -2.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0, 2.0, 2.0, 2.0])
    assert env._action_ranges == [(0, 2), (2, 5)]


def test_build_without_any_action_manager_leaves_the_action_space_unset():
    env = ConfigurableEnv(num_envs=4)
    env.build()
    assert env.action_space is None


"""
build() -- observation space merging
"""


def test_build_uses_only_the_policy_managers_space_when_registered_last():
    log = []

    def configure(env):
        RecordingObservationManager(env, log, name="critic", size=5)
        RecordingObservationManager(env, log, name="policy", size=2)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    assert env.observation_space.shape == (2,)


def test_build_uses_the_policy_space_even_when_registered_first():
    log = []

    def configure(env):
        RecordingObservationManager(env, log, name="policy", size=2)
        RecordingObservationManager(env, log, name="critic", size=5)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    assert env.observation_space.shape == (2,)


def test_build_concatenates_spaces_when_no_manager_is_named_policy():
    log = []

    def configure(env):
        RecordingObservationManager(env, log, name="a", size=2)
        RecordingObservationManager(env, log, name="b", size=3)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    assert env.observation_space.shape == (5,)


def test_build_without_any_observation_manager_leaves_the_space_none():
    env = ConfigurableEnv(num_envs=4)
    env.build()
    assert env.observation_space is None


"""
get_observations()
"""


def test_get_observations_returns_only_the_policy_tensor():
    log = []

    def configure(env):
        RecordingObservationManager(env, log, name="policy", size=2, value=1.0)
        RecordingObservationManager(env, log, name="critic", size=5, value=2.0)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    obs = env.get_observations()

    assert obs.shape == (4, 2)
    assert torch.all(obs == 1.0)
    assert env.extras["observations"]["policy"].shape == (4, 2)
    assert env.extras["observations"]["critic"].shape == (4, 5)


def test_get_observations_concatenates_when_no_manager_is_named_policy():
    log = []

    def configure(env):
        RecordingObservationManager(env, log, name="a", size=2, value=1.0)
        RecordingObservationManager(env, log, name="b", size=3, value=2.0)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    obs = env.get_observations()

    assert obs.shape == (4, 5)
    assert torch.all(obs[:, :2] == 1.0)
    assert torch.all(obs[:, 2:] == 2.0)


"""
step() -- action dispatch and auto-reset on done
"""


def test_step_dispatches_the_right_action_slice_to_each_manager():
    log = []

    def configure(env):
        RecordingActionManager(env, log, size=2)
        RecordingActionManager(env, log, size=3)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    env.reset()

    actions = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    env.step(actions)

    first_mgr, second_mgr = env.managers["action"]
    assert torch.equal(first_mgr.received_actions, actions[:, 0:2])
    assert torch.equal(second_mgr.received_actions, actions[:, 2:5])


def test_step_triggers_reset_for_envs_that_terminated_or_truncated():
    log = []

    def configure(env):
        terminated = torch.tensor([False, True, False, False])
        truncated = torch.tensor([False, False, False, True])
        RecordingTerminationManager(env, log, terminated=terminated, truncated=truncated)
        RecordingManager(env, "entity", log)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    env.reset()
    log.clear()

    env.step(torch.zeros((4, 0)))

    reset_calls = [entry for entry in log if entry[0] == "reset"]
    assert len(reset_calls) == 2  # one per registered manager type: entity, termination
    for _, _, envs_idx in reset_calls:
        assert sorted(envs_idx.tolist()) == [1, 3]


def test_step_does_not_reset_when_nothing_is_done():
    log = []

    def configure(env):
        RecordingTerminationManager(env, log)
        RecordingManager(env, "entity", log)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    env.reset()
    log.clear()

    env.step(torch.zeros((4, 0)))

    assert [entry for entry in log if entry[0] == "reset"] == []


def test_step_returns_the_reward_and_termination_manager_outputs():
    log = []

    def configure(env):
        RecordingRewardManager(env, log, reward=torch.full((4,), 3.0))
        RecordingTerminationManager(
            env, log, terminated=torch.tensor([True, False, False, False])
        )

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    env.reset()

    obs, rewards, terminated, truncated, extras = env.step(torch.zeros((4, 0)))

    assert torch.equal(rewards, torch.full((4,), 3.0))
    assert terminated.tolist() == [True, False, False, False]


def test_step_without_a_reward_or_termination_manager_uses_the_zeroed_defaults():
    env = ConfigurableEnv(num_envs=4)
    env.build()
    env.reset()

    obs, rewards, terminated, truncated, extras = env.step(torch.zeros((4, 0)))

    assert torch.equal(rewards, torch.zeros(4))
    assert torch.equal(terminated, torch.zeros(4, dtype=torch.bool))
    assert torch.equal(truncated, torch.zeros(4, dtype=torch.bool))


"""
reset()
"""


def test_reset_calls_every_manager_type_with_the_given_env_ids():
    log = []

    def configure(env):
        RecordingManager(env, "actuator", log)
        RecordingActionManager(env, log)
        RecordingManager(env, "entity", log)
        RecordingManager(env, "contact", log)
        RecordingTerminationManager(env, log)
        RecordingRewardManager(env, log)
        RecordingManager(env, "command", log)
        RecordingObservationManager(env, log)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    log.clear()

    env.reset([0, 2])

    reset_types = {entry[1] for entry in log if entry[0] == "reset"}
    assert reset_types == {
        "actuator",
        "action",
        "entity",
        "contact",
        "termination",
        "reward",
        "command",
        "observation",
    }
    for entry in log:
        if entry[0] == "reset":
            assert entry[2].tolist() == [0, 2]


def test_reset_with_specific_env_ids_does_not_return_real_observations():
    """Only a full reset (env_ids=None) recomputes observations -- a partial reset's
    returned obs is ignored by callers, per ManagedEnvironment.reset's own comment."""
    log = []

    def configure(env):
        RecordingObservationManager(env, log)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()
    env.reset()  # full reset first, to establish buffers

    obs, extras = env.reset([0])

    assert obs is None


def test_reset_with_no_env_ids_returns_real_observations():
    log = []

    def configure(env):
        RecordingObservationManager(env, log, value=5.0)

    env = ConfigurableEnv(configure=configure, num_envs=4)
    env.build()

    obs, extras = env.reset()

    assert obs.shape == (4, 2)
    assert torch.all(obs == 5.0)
