"""VideoWrapper trigger scheduling, exercised without a Genesis scene or a real camera."""

import pytest
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers import VideoFilename, VideoWrapper


class FakeCamera:
    """Records the filenames it was asked to record to, and which env steps were rendered."""

    def __init__(self, env: "FakeEnv"):
        self.env = env
        self.recordings: list[str] = []
        self.frame_steps: list[int] = []

    @property
    def frames(self) -> int:
        return len(self.frame_steps)

    def start_recording(self, save_to_filename, fps):
        self.recordings.append(save_to_filename)

    def render(self):
        self.frame_steps.append(self.env.step_num)

    def stop_recording(self):
        pass


class FakeEnv(GenesisEnv):
    """Counts its own steps, and optionally terminates every ``episode_length`` steps."""

    def __init__(self, num_envs=2, dt=0.02, episode_length: int | None = None):
        super().__init__(num_envs=num_envs, dt=dt)
        self.camera = FakeCamera(self)
        self.step_num = 0
        self._episode_length = episode_length

    def build(self):
        pass

    def step(self, actions):
        self.step_num += 1
        zeros = torch.zeros(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._episode_length and self.step_num % self._episode_length == 0:
            done[:] = True
        return zeros, zeros, done, done, {}

    def close(self):
        pass


def run_steps(env, n):
    for _ in range(n):
        env.step(None)


def recording_names(env) -> list[str]:
    return [r.split("/")[-1] for r in env.unwrapped.camera.recordings]


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


def test_step_count_starts_at_one(tmp_path):
    seen = []
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        step_trigger=lambda step: seen.append(step) or False,
    )
    env.build()
    run_steps(env, 3)
    assert seen == [1, 2, 3]


def test_episode_count_starts_at_one_and_is_checked_once_per_episode(tmp_path):
    seen = []
    env = VideoWrapper(
        FakeEnv(episode_length=4),
        out_dir=str(tmp_path),
        episode_trigger=lambda episode: seen.append(episode) or False,
    )
    env.build()
    run_steps(env, 12)
    assert seen == [1, 2, 3]


def test_episode_recording_starts_on_first_step_of_the_episode(tmp_path):
    # Episode 2 covers steps 5-8, so those are exactly the frames the video holds.
    env = VideoWrapper(
        FakeEnv(episode_length=4),
        out_dir=str(tmp_path),
        video_length_sec=0.08,  # four steps
        fps=50,  # one frame per step
        episode_trigger=lambda episode: episode == 2,
    )
    env.build()
    run_steps(env, 12)
    assert recording_names(env) == ["2.mp4"]
    assert env.unwrapped.camera.frame_steps == [5, 6, 7, 8]


def test_episode_recording_does_not_start_mid_episode(tmp_path):
    # A 6-step video started at the top of a 10-step episode finishes at step 7,
    # part way through that episode. The trigger fires for every episode, but the
    # next recording must wait for the start of episode 2 (step 11).
    env = VideoWrapper(
        FakeEnv(episode_length=10),
        out_dir=str(tmp_path),
        video_length_sec=0.12,  # six steps
        fps=50,  # one frame per step
        episode_trigger=lambda episode: True,
    )
    env.build()
    run_steps(env, 22)
    assert recording_names(env) == ["1.mp4", "2.mp4", "3.mp4"]
    assert env.unwrapped.camera.frame_steps == [
        *range(1, 7),
        *range(11, 17),
        *range(21, 23),
    ]


def test_episode_recording_ending_on_episode_boundary_restarts_immediately(tmp_path):
    # A video as long as the episode ends exactly where the next episode begins,
    # so back-to-back recordings cover every episode with no gaps.
    env = VideoWrapper(
        FakeEnv(episode_length=6),
        out_dir=str(tmp_path),
        video_length_sec=0.12,  # six steps
        fps=50,  # one frame per step
        episode_trigger=lambda episode: True,
    )
    env.build()
    run_steps(env, 12)
    assert recording_names(env) == ["1.mp4", "2.mp4"]
    assert env.unwrapped.camera.frame_steps == list(range(1, 13))


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


def test_iteration_trigger_fires_on_the_first_step_of_the_iteration(tmp_path):
    # Iteration 1 covers steps 5-8, so those are exactly the frames the video holds.
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.08,  # four steps
        fps=50,  # one frame per step
        iteration_trigger=lambda it: it == 1,
        steps_per_iteration=4,
    )
    env.build()
    run_steps(env, 12)
    assert recording_names(env) == ["1.mp4"]
    assert env.unwrapped.camera.frame_steps == [5, 6, 7, 8]


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
    assert recording_names(env) == ["0.mp4", "2.mp4", "4.mp4"]


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
    assert recording_names(env) == ["1000.mp4", "1002.mp4"]


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
    assert recording_names(env) == ["250.mp4"]


def test_recording_spanning_iterations_does_not_restart_mid_iteration(tmp_path):
    # A 6-step video started at iteration 0 finishes at step 7, part way through
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
    assert recording_names(env) == ["0.mp4", "2.mp4"]


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
    assert recording_names(env) == ["0.mp4", "1.mp4", "2.mp4"]


def test_video_records_exactly_video_length_steps_frames(tmp_path):
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.1,  # five steps
        fps=50,  # one frame per step
        step_trigger=lambda step: step == 1,
    )
    env.build()
    run_steps(env, 20)
    assert env.video_length_steps == 5
    assert env.unwrapped.camera.frame_steps == [1, 2, 3, 4, 5]


def test_wait_for_episode_start_delays_recording_to_next_episode(tmp_path):
    # Iteration 1 triggers at step 5, part way through episode 1 (steps 1-6),
    # so the video waits for episode 2 to begin at step 7.
    env = VideoWrapper(
        FakeEnv(episode_length=6),
        out_dir=str(tmp_path),
        video_length_sec=0.08,  # four steps
        fps=50,  # one frame per step
        iteration_trigger=lambda it: it == 1,
        steps_per_iteration=4,
        wait_for_episode_start=True,
    )
    env.build()
    run_steps(env, 16)
    assert env.unwrapped.camera.frame_steps == [7, 8, 9, 10]


def test_wait_for_episode_start_names_video_for_iteration_it_started_on(tmp_path):
    # Iteration 11 triggers at step 5, but episode 2 begins at step 11, in iteration 12.
    env = VideoWrapper(
        FakeEnv(episode_length=10),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: it == 11,
        steps_per_iteration=4,
        iteration_offset=10,
        wait_for_episode_start=True,
    )
    env.build()
    run_steps(env, 20)
    assert recording_names(env) == ["12.mp4"]


def test_wait_for_episode_start_starts_immediately_on_an_episode_boundary(tmp_path):
    # Iteration 0 starts on step 1, which is also the first step of episode 1.
    env = VideoWrapper(
        FakeEnv(episode_length=10),
        out_dir=str(tmp_path),
        video_length_sec=0.04,  # two steps
        fps=50,  # one frame per step
        iteration_trigger=lambda it: it == 0,
        steps_per_iteration=4,
        wait_for_episode_start=True,
    )
    env.build()
    run_steps(env, 10)
    assert recording_names(env) == ["0.mp4"]
    assert env.unwrapped.camera.frame_steps == [1, 2]


def test_wait_for_episode_start_ignores_triggers_while_waiting(tmp_path):
    # Every iteration triggers, but only one recording can be waiting at a time:
    # iterations 0-2 (steps 1-12) all fall in episode 1 and collapse into its video,
    # iterations 3-4 wait for episode 2 at step 13.
    env = VideoWrapper(
        FakeEnv(episode_length=12),
        out_dir=str(tmp_path),
        video_length_sec=0.04,  # two steps
        iteration_trigger=lambda it: True,
        steps_per_iteration=4,
        wait_for_episode_start=True,
    )
    env.build()
    run_steps(env, 20)
    assert recording_names(env) == ["0.mp4", "3.mp4"]


def test_wait_for_episode_start_with_step_trigger(tmp_path):
    env = VideoWrapper(
        FakeEnv(episode_length=6),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        step_trigger=lambda step: step == 3,
        wait_for_episode_start=True,
    )
    env.build()
    run_steps(env, 12)
    assert recording_names(env) == ["7.mp4"]


def test_filename_string_names_every_video(tmp_path):
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        step_trigger=lambda step: step in (1, 5),
        filename="latest.mp4",
    )
    env.build()
    run_steps(env, 8)
    assert recording_names(env) == ["latest.mp4", "latest.mp4"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (VideoFilename.STEP, "7.mp4"),
        (VideoFilename.EPISODE, "3.mp4"),
        (VideoFilename.ITERATION, "101.mp4"),
    ],
)
def test_filename_enum_names_video_for_that_count(tmp_path, filename, expected):
    # Step 7 is the first step of episode 3, in iteration 1 (steps 5-8), offset by 100.
    env = VideoWrapper(
        FakeEnv(episode_length=3),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        step_trigger=lambda step: step == 7,
        steps_per_iteration=4,
        iteration_offset=100,
        filename=filename,
    )
    env.build()
    run_steps(env, 8)
    assert recording_names(env) == [expected]


def test_filename_iteration_requires_steps_per_iteration(tmp_path):
    with pytest.raises(AssertionError):
        VideoWrapper(
            FakeEnv(),
            out_dir=str(tmp_path),
            step_trigger=lambda step: True,
            filename=VideoFilename.ITERATION,
        )


def test_filename_function_receives_step_episode_and_iteration(tmp_path):
    env = VideoWrapper(
        FakeEnv(episode_length=3),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: it == 1,
        steps_per_iteration=4,
        filename=lambda step, episode, iteration: (
            f"s{step}_e{episode}_i{iteration}.mp4"
        ),
    )
    env.build()
    run_steps(env, 8)
    assert recording_names(env) == ["s5_e2_i1.mp4"]


def test_filename_function_iteration_is_none_without_steps_per_iteration(tmp_path):
    env = VideoWrapper(
        FakeEnv(),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        step_trigger=lambda step: step == 2,
        filename=lambda step, episode, iteration: f"{step}_{episode}_{iteration}.mp4",
    )
    env.build()
    run_steps(env, 4)
    assert recording_names(env) == ["2_1_None.mp4"]


def test_filename_function_called_when_waiting_recording_starts(tmp_path):
    # Iteration 1 triggers at step 5, but the recording waits for episode 2 at step 7.
    env = VideoWrapper(
        FakeEnv(episode_length=6),
        out_dir=str(tmp_path),
        video_length_sec=0.02,  # one step
        iteration_trigger=lambda it: it == 1,
        steps_per_iteration=4,
        wait_for_episode_start=True,
        filename=lambda step, episode, iteration: (
            f"s{step}_e{episode}_i{iteration}.mp4"
        ),
    )
    env.build()
    run_steps(env, 12)
    assert recording_names(env) == ["s7_e2_i1.mp4"]
