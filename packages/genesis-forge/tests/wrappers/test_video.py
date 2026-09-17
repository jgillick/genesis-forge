"""VideoWrapper trigger scheduling, exercised without a Genesis scene or a real camera."""

import pytest
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers import VideoWrapper


class FakeCamera:
    """Records the filenames it was asked to record to, and how many frames were rendered."""

    def __init__(self):
        self.recordings: list[str] = []
        self.frames = 0

    def start_recording(self, save_to_filename, fps):
        self.recordings.append(save_to_filename)

    def render(self):
        self.frames += 1

    def stop_recording(self):
        pass


class FakeEnv(GenesisEnv):
    def __init__(self, num_envs=2, dt=0.02):
        super().__init__(num_envs=num_envs, dt=dt)
        self.camera = FakeCamera()

    def build(self):
        pass

    def step(self, actions):
        zeros = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        return zeros, zeros, done, done, {}

    def close(self):
        pass


def run_steps(env, n):
    for _ in range(n):
        env.step(None)


def test_iteration_trigger_requires_steps_per_iteration(tmp_path):
    with pytest.raises(AssertionError):
        VideoWrapper(
            FakeEnv(), out_dir=str(tmp_path), iteration_trigger=lambda it: True
        )


def test_only_one_trigger_allowed(tmp_path):
    with pytest.raises(AssertionError):
        VideoWrapper(
            FakeEnv(),
            out_dir=str(tmp_path),
            step_trigger=lambda s: True,
            iteration_trigger=lambda it: True,
            steps_per_iteration=4,
        )


def test_iteration_trigger_called_once_per_iteration(tmp_path):
    seen = []
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: seen.append(it) or False,
        steps_per_iteration=4,
    )
    env.build()
    run_steps(env, 12)
    assert seen == [0, 1, 2]


def test_video_named_for_iteration_recording_started_on(tmp_path):
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: it % 2 == 0,
        steps_per_iteration=4,
    )
    env.build()
    run_steps(env, 20)
    names = [r.split("/")[-1] for r in env.unwrapped.camera.recordings]
    assert names == ["0.mp4", "2.mp4", "4.mp4"]


def test_iteration_offset_shifts_iteration_and_filename(tmp_path):
    seen = []
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: seen.append(it) or it % 2 == 0,
        steps_per_iteration=4,
        iteration_offset=1000,
    )
    env.build()
    run_steps(env, 12)
    assert seen == [1000, 1001, 1002]
    names = [r.split("/")[-1] for r in env.unwrapped.camera.recordings]
    assert names == ["1000.mp4", "1002.mp4"]


def test_iteration_offset_can_be_set_after_construction(tmp_path):
    # The runner (and so the checkpoint's iteration) usually exists only after the wrapper does.
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: True,
        steps_per_iteration=4,
    )
    env.build()
    env.iteration_offset = 250
    assert env.iteration_offset == 250
    run_steps(env, 4)
    names = [r.split("/")[-1] for r in env.unwrapped.camera.recordings]
    assert names == ["250.mp4"]


def test_recording_spanning_iterations_does_not_restart_mid_iteration(tmp_path):
    # A 6-step video started at iteration 0 finishes at step 5, part way through
    # iteration 1. The trigger fires for every iteration but must wait for the
    # start of iteration 2 rather than restarting with a stale index.
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.12,  # six steps
        iteration_trigger=lambda it: True,
        steps_per_iteration=4,
    )
    env.build()
    run_steps(env, 9)
    names = [r.split("/")[-1] for r in env.unwrapped.camera.recordings]
    assert names == ["0.mp4", "2.mp4"]


def test_recording_ending_on_iteration_boundary_restarts_on_that_boundary(tmp_path):
    # A 4-step video started at iteration 0 ends exactly where iteration 1 begins,
    # so back-to-back recordings cover every iteration with no gaps.
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.08,  # four steps
        iteration_trigger=lambda it: True,
        steps_per_iteration=4,
    )
    env.build()
    run_steps(env, 12)
    names = [r.split("/")[-1] for r in env.unwrapped.camera.recordings]
    assert names == ["0.mp4", "1.mp4", "2.mp4"]


def test_video_records_exactly_video_length_steps_frames(tmp_path):
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.1,  # five steps
        fps=50,  # one frame per step
        step_trigger=lambda step: step == 0,
    )
    env.build()
    run_steps(env, 20)
    assert env.video_length_steps == 5
    assert env.unwrapped.camera.frames == 5
