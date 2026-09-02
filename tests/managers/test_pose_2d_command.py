"""Behavior of Pose2dCommand: goal sampling, reach detection, the robot-frame goal
observation, the resampling rules (on reset, on reach, and on the optional timer), and
keeping goals clear of everything else in the scene.

Uses a FakeEntity -- no Genesis scene is built. Ranges are chosen with equal min/max
(e.g. (2.0, 2.0)) wherever a sampled goal needs to be deterministic, since
resample_command draws from `torch.uniform_(min, max)`.
"""

import math

import torch

from genesis_forge.managers import Pose2dCommand


class FakeEntity:
    """A robot at a given XY position and yaw, occupying a square footprint."""

    def __init__(self, pos=None, yaw=0.0, num_envs=4, size=0.0):
        if pos is None:
            pos = torch.zeros((num_envs, 3))
        self._pos = pos
        self._size = size
        half = yaw / 2.0
        self._quat = torch.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]]
        ).repeat(pos.shape[0], 1)

    def get_pos(self):
        return self._pos

    def get_quat(self):
        return self._quat

    def get_AABB(self):
        """The bounding box of the footprint, centered on the entity, per environment."""
        half = self._size / 2
        corners = torch.tensor([[-half, -half, 0.0], [half, half, 0.0]])
        return corners.unsqueeze(0).repeat(self._pos.shape[0], 1, 1) + self._pos.unsqueeze(1)


class FakeScene:
    """Just enough of a Genesis scene for the manager to look up what to avoid."""

    def __init__(self, entities=None):
        self.entities = entities if entities is not None else []


def make_manager(env, entity=None, scene_entities=None, **kwargs):
    """A built manager over a fixed goal range, so sampled goals are deterministic."""
    kwargs.setdefault("range", {"x": (2.0, 2.0), "y": (0.0, 0.0), "heading": (0.0, 0.0)})
    if entity is None:
        entity = FakeEntity(num_envs=env.num_envs)
    env.robot = entity
    env.terrain = None
    env.scene = FakeScene(scene_entities)
    env.episode_length = torch.zeros(env.num_envs, dtype=torch.long)
    env.step_count = 0
    mgr = Pose2dCommand(env, **kwargs)
    mgr.build()
    return mgr


"""
Construction
"""


def test_range_allocates_an_xy_and_heading_command(env):
    mgr = make_manager(env)
    assert mgr.command.shape == (env.num_envs, 3)


def test_resample_time_sec_defaults_to_no_timer(env):
    mgr = make_manager(env)
    assert mgr.resample_time_sec is None
    assert mgr._resample_steps == 0


def test_resample_time_sec_computes_resample_steps_when_set(env):
    mgr = make_manager(env, resample_time_sec=0.1)
    assert mgr._resample_steps == int(0.1 / env.dt)


"""
shortest_turn_to
"""


def test_shortest_turn_takes_the_short_way_around():
    """Turning from 179 to -179 degrees is a small right turn, not an almost-full circle."""
    from genesis_forge.managers.command.pose_2d_command import shortest_turn_to

    target = torch.tensor([math.radians(-179)])
    current = torch.tensor([math.radians(179)])

    turn = shortest_turn_to(target, current)

    assert math.isclose(turn.item(), math.radians(2), abs_tol=1e-5)


def test_shortest_turn_is_positive_to_the_left():
    from genesis_forge.managers.command.pose_2d_command import shortest_turn_to

    turn = shortest_turn_to(torch.tensor([math.pi / 2]), torch.tensor([0.0]))

    assert math.isclose(turn.item(), math.pi / 2, abs_tol=1e-5)


"""
distance_to_goal / heading_error / goal_reached
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


def test_heading_error_is_how_far_the_robot_still_has_to_turn(env):
    """A robot facing +X, told to face +Y, has a quarter turn left to make."""
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=0.0, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(
        env,
        entity=entity,
        range={"x": (2.0, 2.0), "y": (0.0, 0.0), "heading": (math.pi / 2, math.pi / 2)},
    )
    mgr.reset()

    assert torch.allclose(mgr.heading_error, torch.tensor([math.pi / 2]), atol=1e-5)


def test_heading_error_is_zero_when_already_facing_the_goal_heading(env):
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=math.pi / 2, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(
        env,
        entity=entity,
        range={"x": (2.0, 2.0), "y": (0.0, 0.0), "heading": (math.pi / 2, math.pi / 2)},
    )
    mgr.reset()

    assert torch.allclose(mgr.heading_error, torch.tensor([0.0]), atol=1e-5)


def test_goal_reached_uses_the_threshold(env):
    entity = FakeEntity(
        pos=torch.tensor([[1.9, 0.0, 0.0], [1.0, 0.0, 0.0]]), num_envs=2
    )
    env.num_envs = 2
    mgr = make_manager(env, entity=entity, goal_reached_threshold=0.15)
    mgr.reset()

    # Distances to the (2.0, 0.0) goal are 0.1 and 1.0
    assert mgr.goal_reached.tolist() == [True, False]


def test_goal_reached_ignores_heading_by_default(env):
    """Arriving at the position is enough unless a heading threshold is configured."""
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0]]), yaw=0.0, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(
        env,
        entity=entity,
        range={"x": (2.0, 2.0), "y": (0.0, 0.0), "heading": (math.pi, math.pi)},
    )
    mgr.reset()

    # The robot is on its goal, but facing exactly the wrong way
    assert mgr.goal_reached.tolist() == [True]


def test_goal_reached_also_requires_heading_when_a_threshold_is_set(env):
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0]]), yaw=0.0, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(
        env,
        entity=entity,
        heading_reached_threshold=0.1,
        range={"x": (2.0, 2.0), "y": (0.0, 0.0), "heading": (math.pi, math.pi)},
    )
    mgr.reset()

    assert mgr.goal_reached.tolist() == [False]


"""
observation() -- the goal pose in the robot's local frame
"""


def test_observation_of_a_forward_goal_for_an_unrotated_robot(env):
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=0.0, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    # Goal is 2m along world +X, the robot faces +X, and the goal heading matches its own:
    # 2m straight ahead, no turn to drive there, no turn to face the goal heading
    assert torch.allclose(
        mgr.observation(env),
        #        ahead  left  dist  cos/sin bearing  cos/sin heading err
        torch.tensor([[2.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0]]),
        atol=1e-5,
    )


def test_observation_rotates_the_goal_into_the_robot_frame(env):
    """A robot turned 90 degrees left sees a goal on the world +X axis to its right."""
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=math.pi / 2, num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    obs = mgr.observation(env)

    # The goal is 2m off to the robot's right, so it has a quarter turn right to drive
    # there, and the same quarter turn right to face the commanded heading of 0
    assert torch.allclose(obs[0, :3], torch.tensor([0.0, -2.0, 2.0]), atol=1e-5)
    assert torch.allclose(obs[0, 3:5], torch.tensor([0.0, -1.0]), atol=1e-5)
    assert torch.allclose(obs[0, 5:7], torch.tensor([0.0, -1.0]), atol=1e-5)


def test_observation_is_relative_to_the_robot_position(env):
    entity = FakeEntity(pos=torch.tensor([[1.0, 0.0, 0.0]]), num_envs=1)
    env.num_envs = 1
    mgr = make_manager(env, entity=entity)
    mgr.reset()

    assert torch.allclose(
        mgr.observation(env),
        torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]]),
        atol=1e-5,
    )


def test_observation_keeps_the_bearing_at_full_strength_when_close(env):
    """
    The whole reason the bearing is reported separately: the goal vector shrinks as the
    robot closes in, but the direction to steer must not fade with it.
    """
    env.num_envs = 1
    far = FakeEntity(pos=torch.tensor([[0.0, 0.0, 0.0]]), num_envs=1)
    mgr = make_manager(
        env, entity=far, range={"x": (0.0, 0.0), "y": (2.0, 2.0), "heading": (0.0, 0.0)}
    )
    mgr.reset()
    bearing_far = mgr.observation(env)[0, 3:5].clone()

    # Same direction, a hundredth of the distance away
    far._pos = torch.tensor([[0.0, 1.98, 0.0]])
    obs_near = mgr.observation(env)

    assert obs_near[0, 2] < 0.03  # the goal vector has all but vanished
    assert torch.allclose(obs_near[0, 3:5], bearing_far, atol=1e-4)  # the bearing has not


def test_bearing_error_is_the_turn_needed_to_point_at_the_goal(env):
    """Not to be confused with heading_error, which is the way to face on arrival."""
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), yaw=0.0, num_envs=1)
    mgr = make_manager(
        env,
        entity=entity,
        # Goal is off to the robot's left, but asks it to arrive facing straight ahead
        range={"x": (0.0, 0.0), "y": (2.0, 2.0), "heading": (0.0, 0.0)},
    )
    mgr.reset()

    assert torch.allclose(mgr.bearing_error, torch.tensor([math.pi / 2]), atol=1e-5)
    assert torch.allclose(mgr.heading_error, torch.tensor([0.0]), atol=1e-5)


"""
Resampling
"""


def test_reset_samples_goals_within_the_range(env):
    mgr = make_manager(
        env,
        range={"x": (-2.0, 2.0), "y": (-1.0, 1.0), "heading": (-math.pi, math.pi)},
    )
    mgr.reset()

    assert torch.all(mgr.command[:, 0] >= -2.0) and torch.all(mgr.command[:, 0] <= 2.0)
    assert torch.all(mgr.command[:, 1] >= -1.0) and torch.all(mgr.command[:, 1] <= 1.0)
    assert torch.all(mgr.command[:, 2] >= -math.pi) and torch.all(
        mgr.command[:, 2] <= math.pi
    )


def test_step_resamples_only_the_environments_that_reached_their_goal(env):
    env.num_envs = 2
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), num_envs=2)
    mgr = make_manager(
        env,
        entity=entity,
        range={"x": (-5.0, 5.0), "y": (0.0, 0.0), "heading": (0.0, 0.0)},
    )
    mgr.command[:] = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    mgr.step()

    # Env 0 is on its goal so it gets a new one; env 1 is 2m away and keeps its goal
    assert mgr.resampled_last_step.tolist() == [True, False]
    assert mgr.command[1].tolist() == [2.0, 0.0, 0.0]


def test_step_does_not_resample_on_reach_when_disabled(env):
    env.num_envs = 1
    entity = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0]]), num_envs=1)
    mgr = make_manager(env, entity=entity, resample_on_reached=False)
    mgr.command[:] = torch.tensor([[2.0, 0.0, 0.0]])

    mgr.step()

    assert mgr.resampled_last_step.tolist() == [False]


def test_step_without_a_timer_does_not_resample_unreached_goals(env):
    """With resample_time_sec=None, an episode keeps its goal until it is reached."""
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1)
    mgr = make_manager(
        env,
        entity=entity,
        range={"x": (-5.0, 5.0), "y": (0.0, 0.0), "heading": (0.0, 0.0)},
    )
    mgr.command[:] = torch.tensor([[3.0, 0.0, 0.0]])

    for _ in range(10):
        mgr.step()
        env.step_count += 1
        env.episode_length += 1

    assert mgr.command.tolist() == [[3.0, 0.0, 0.0]]


def test_step_resamples_on_the_timer_when_configured(env):
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1)
    mgr = make_manager(
        env,
        entity=entity,
        resample_time_sec=env.dt * 2,
        range={"x": (7.0, 7.0), "y": (0.0, 0.0), "heading": (0.0, 0.0)},
    )
    mgr.command[:] = torch.tensor([[3.0, 0.0, 0.0]])

    env.episode_length += 2  # a multiple of the 2-step resample interval
    mgr.step()

    assert mgr.command.tolist() == [[7.0, 0.0, 0.0]]


def test_resampled_last_step_is_cleared_at_the_start_of_the_next_step(env):
    env.num_envs = 1
    entity = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1)
    mgr = make_manager(env, entity=entity)
    mgr.reset()
    assert mgr.resampled_last_step.tolist() == [True]

    mgr.step()

    assert mgr.resampled_last_step.tolist() == [False]


"""
Keeping goals clear of the scene
"""


def test_goals_are_redrawn_away_from_scene_entities(env):
    torch.manual_seed(0)
    env.num_envs = 20
    robot = FakeEntity(pos=torch.full((20, 3), 10.0), num_envs=20)
    obstacle = FakeEntity(pos=torch.zeros((20, 3)), num_envs=20, size=0.6)
    mgr = make_manager(
        env,
        entity=robot,
        scene_entities=[robot, obstacle],
        goal_reached_threshold=0.1,
        range={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "heading": (0.0, 0.0)},
    )

    mgr.reset()

    # Half the 0.6m square's diagonal, plus the 0.1m reach threshold
    margin = math.hypot(0.6, 0.6) / 2 + 0.1
    distance = torch.norm(mgr.command[:, :2] - obstacle.get_pos()[:, :2], dim=-1)
    assert torch.all(distance >= margin)


def test_goals_are_kept_clear_of_the_robot_itself(env):
    """A goal must never spawn on top of the robot, or it would start out reached."""
    torch.manual_seed(0)
    env.num_envs = 20
    robot = FakeEntity(pos=torch.zeros((20, 3)), num_envs=20, size=0.2)
    mgr = make_manager(
        env,
        entity=robot,
        scene_entities=[robot],
        goal_reached_threshold=0.1,
        range={"x": (-1.0, 1.0), "y": (-1.0, 1.0), "heading": (0.0, 0.0)},
    )

    mgr.reset()

    assert not torch.any(mgr.goal_reached)


def test_terrain_is_not_avoided(env):
    """Goals sit on the ground, so the terrain is not something to keep away from."""
    env.num_envs = 1
    robot = FakeEntity(pos=torch.tensor([[10.0, 10.0, 0.0]]), num_envs=1)
    terrain = FakeEntity(pos=torch.zeros((1, 3)), num_envs=1, size=100.0)
    mgr = make_manager(env, entity=robot, scene_entities=[robot, terrain])
    mgr.env.terrain = terrain
    mgr._find_entities_to_avoid()

    mgr.reset()

    # The terrain covers the whole world; if it were avoided, no goal could ever be placed
    assert mgr.command[:, :2].tolist() == [[2.0, 0.0]]


def test_resampling_gives_up_rather_than_looping_forever(env):
    """A degenerate range that always redraws the same blocked point must still return."""
    env.num_envs = 1
    robot = FakeEntity(pos=torch.tensor([[10.0, 10.0, 0.0]]), num_envs=1)
    obstacle = FakeEntity(pos=torch.tensor([[2.0, 0.0, 0.0]]), num_envs=1, size=0.5)
    mgr = make_manager(
        env,
        entity=robot,
        scene_entities=[robot, obstacle],
        range={"x": (2.0, 2.0), "y": (0.0, 0.0), "heading": (0.0, 0.0)},
    )

    mgr.reset()

    # Every draw lands on the obstacle, so the manager keeps the last (still blocked) one
    assert mgr.command[:, :2].tolist() == [[2.0, 0.0]]
