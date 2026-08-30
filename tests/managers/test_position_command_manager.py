"""Behavior of PositionCommandManager: goal sampling, reach detection, the
robot-frame goal observation, and the resampling rules (on reset, on reach, and on
the optional timer).

Uses a FakeEntity -- no Genesis scene is built. Ranges are chosen with equal min/max
(e.g. (2.0, 2.0)) wherever a sampled goal needs to be deterministic, since
resample_command draws from `torch.uniform_(min, max)`.
"""

import math

import torch

from genesis_forge.managers import PositionCommandManager


class FakeEntity:
    """A robot at a given XY position and yaw."""

    def __init__(self, pos=None, yaw=0.0, num_envs=4):
        if pos is None:
            pos = torch.zeros((num_envs, 3))
        self._pos = pos
        half = yaw / 2.0
        self._quat = torch.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]]
        ).repeat(pos.shape[0], 1)

    def get_pos(self):
        return self._pos

    def get_quat(self):
        return self._quat


def make_manager(env, entity=None, **kwargs):
    """A built manager over a fixed goal range, so sampled goals are deterministic."""
    kwargs.setdefault("range", {"x": (2.0, 2.0), "y": (0.0, 0.0)})
    if entity is None:
        entity = FakeEntity(num_envs=env.num_envs)
    env.robot = entity
    env.episode_length = torch.zeros(env.num_envs, dtype=torch.long)
    env.step_count = 0
    mgr = PositionCommandManager(env, **kwargs)
    mgr.build()
    return mgr


"""
Construction
"""


def test_range_allocates_an_xy_command(env):
    mgr = make_manager(env)
    assert mgr.command.shape == (env.num_envs, 2)


def test_resample_time_sec_defaults_to_no_timer(env):
    mgr = make_manager(env)
    assert mgr.resample_time_sec is None
    assert mgr._resample_steps == 0


def test_resample_time_sec_computes_resample_steps_when_set(env):
    mgr = make_manager(env, resample_time_sec=0.1)
    assert mgr._resample_steps == int(0.1 / env.dt)


"""
distance_to_goal / goal_reached
"""


def test_distance_to_goal_is_the_xy_distance(env):
    entity = FakeEntity(pos=torch.tensor([[0.0, 0.0, 0.3], [2.0, 0.0, 0.3]]), num_envs=2)
    env.num_envs = 2
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    # Both goals are (2.0, 0.0): the first robot is 2m away, the second is on top of it
    assert torch.allclose(mgr.distance_to_goal, torch.tensor([2.0, 0.0]))


def test_distance_to_goal_ignores_height(env):
    """A robot directly under its goal has arrived, however tall it is."""
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 5.0]]), num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    assert torch.allclose(mgr.distance_to_goal, torch.tensor([0.0]))


def test_goal_reached_uses_the_threshold(env):
    entity = FakeEntity(
        pos=torch.tensor([[1.9, 0.0, 0.0], [1.0, 0.0, 0.0]]), num_envs=2
    )
    env.num_envs = 2
    mgr = make_manager(env, entity=entity, goal_reached_threshold=0.15)
    mgr.reset()

    # Distances to the (2.0, 0.0) goal are 0.1 and 1.0
    assert mgr.goal_reached.tolist() == [True, False]


"""
observation() -- the goal vector in the robot's local frame
"""


def test_observation_of_a_forward_goal_for_an_unrotated_robot(env):
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=0.0, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    # Goal is 2m along world +X, and the robot faces +X, so it's 2m straight ahead
    assert torch.allclose(mgr.observation(env), torch.tensor([[2.0, 0.0]]), atol=1e-5)


def test_observation_rotates_the_goal_into_the_robot_frame(env):
    """A robot turned 90 degrees left sees a goal on the world +X axis to its right."""
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=math.pi / 2, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    assert torch.allclose(mgr.observation(env), torch.tensor([[0.0, -2.0]]), atol=1e-5)


def test_observation_is_relative_to_the_robot_position(env):
    entity = FakeEntity(pos=torch.tensor([[1.0, 0.0, 0.0]]), num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    assert torch.allclose(mgr.observation(env), torch.tensor([[1.0, 0.0]]), atol=1e-5)


"""
Resampling
"""


def test_reset_samples_goals_within_the_range(env):
    mgr = make_manager(env, range={"x": (-2.0, 2.0), "y": (-1.0, 1.0)})
    mgr.reset()

    assert torch.all(mgr.command[:, 0] >= -2.0) and torch.all(mgr.command[:, 0] <= 2.0)
    assert torch.all(mgr.command[:, 1] >= -1.0) and torch.all(mgr.command[:, 1] <= 1.0)


def test_step_resamples_only_the_environments_that_reached_their_goal(env):
    env.num_envs = 2
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), num_envs=2)
    mgr = make_manager(env, entity=entity, range={"x": (-5.0, 5.0), "y": (0.0, 0.0)})
    mgr.command[:] = torch.tensor([[2.0, 0.0], [2.0, 0.0]])

    mgr.step()

    # Env 0 is on its goal so it gets a new one; env 1 is 2m away and keeps its goal
    assert mgr.resampled_last_step.tolist() == [True, False]
    assert mgr.command[1].tolist() == [2.0, 0.0]


def test_step_does_not_resample_on_reach_when_disabled(env):
    env.num_envs = 1
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0]]), num_envs=1)
    mgr = make_manager(env, entity=entity, resample_on_reached=False)
    mgr.command[:] = torch.tensor([[2.0, 0.0]])

    mgr.step()

    assert mgr.resampled_last_step.tolist() == [False]


def test_step_without_a_timer_does_not_resample_unreached_goals(env):
    """With resample_time_sec=None, an episode keeps its goal until it is reached."""
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1)
    mgr = make_manager(env, entity=entity, range={"x": (-5.0, 5.0), "y": (0.0, 0.0)})
    mgr.command[:] = torch.tensor([[3.0, 0.0]])

    for _ in range(10):
        mgr.step()
        env.step_count += 1
        env.episode_length += 1

    assert mgr.command.tolist() == [[3.0, 0.0]]


def test_step_resamples_on_the_timer_when_configured(env):
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1)
    mgr = make_manager(
        env,
        entity=entity,
        resample_time_sec=env.dt * 2,
        range={"x": (7.0, 7.0), "y": (0.0, 0.0)},
    )
    mgr.command[:] = torch.tensor([[3.0, 0.0]])

    env.episode_length += 2  # a multiple of the 2-step resample interval
    mgr.step()

    assert mgr.command.tolist() == [[7.0, 0.0]]


def test_resampled_last_step_is_cleared_at_the_start_of_the_next_step(env):
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1)
    mgr = make_manager(env, entity=entity)
    mgr.reset()
    assert mgr.resampled_last_step.tolist() == [True]

    mgr.step()

    assert mgr.resampled_last_step.tolist() == [False]
