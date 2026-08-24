"""Reset propagation from the managers down to their config items.

``ManagedEnvironment.reset`` already calls ``reset(env_ids)`` on the termination and
observation managers, but both inherited the no-op ``BaseManager.reset`` -- so a stateful
termination or observation function never had its state cleared between episodes. No such
function exists in the library yet; these tests keep the gap closed before one does.
"""

from dataclasses import dataclass

import torch

from genesis_forge.managers import ObservationManager, TerminationManager
from genesis_forge.managers.config import MdpFn


@dataclass(kw_only=True, eq=False)
class Stateful(MdpFn):
    width: int = 2

    def build(self):
        self.has_reset = torch.zeros((self.env.num_envs, self.width))

    def reset(self, envs_idx):
        self.has_reset[envs_idx] = 1.0

    def __call__(self, env):
        return self.has_reset


def plain_obs(env):
    return torch.zeros((env.num_envs, 2))


"""
TerminationManager
"""


def test_termination_manager_forwards_reset(env):
    fn = Stateful()
    mgr = TerminationManager(env, term_cfg={"stuck": {"fn": fn}})
    mgr.build()

    mgr.reset([0, 2])

    assert torch.equal(fn.has_reset[:, 0], torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_termination_manager_reset_without_indices_resets_all(env):
    fn = Stateful()
    mgr = TerminationManager(env, term_cfg={"stuck": {"fn": fn}})
    mgr.build()

    mgr.reset()

    assert torch.equal(fn.has_reset, torch.ones((env.num_envs, 2)))


def test_termination_manager_reset_tolerates_plain_functions(env):
    mgr = TerminationManager(env, term_cfg={"plain": {"fn": plain_obs}})
    mgr.build()
    mgr.reset([0])  # must not raise


"""
ObservationManager
"""


def test_observation_manager_forwards_reset(env):
    fn = Stateful()
    mgr = ObservationManager(env, cfg={"history": {"fn": fn}})
    mgr.build()

    mgr.reset([1])

    assert torch.equal(fn.has_reset[:, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_observation_manager_reset_without_indices_resets_all(env):
    fn = Stateful()
    mgr = ObservationManager(env, cfg={"history": {"fn": fn}})
    mgr.build()

    mgr.reset()

    assert torch.equal(fn.has_reset, torch.ones((env.num_envs, 2)))


def test_observation_manager_reset_tolerates_plain_functions(env):
    mgr = ObservationManager(env, cfg={"plain": {"fn": plain_obs}})
    mgr.build()
    mgr.reset([0])  # must not raise


"""
The observation manager probe-calls each function at build time to size the space
"""


def test_observation_manager_sizes_the_space_from_an_mdp_fn(env):
    """__call__ must be valid the instant build() returns, not on the first step."""
    mgr = ObservationManager(env, cfg={"history": {"fn": Stateful(width=3)}})
    mgr.build()

    assert mgr.observation_space.shape == (3,)
