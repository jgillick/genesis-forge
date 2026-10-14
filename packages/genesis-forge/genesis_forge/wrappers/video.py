from __future__ import annotations

import math
import os
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers.wrapper import Wrapper

if TYPE_CHECKING:
    from genesis.vis.camera import Camera


class VideoFilename(Enum):
    """How the videos files are named"""

    STEP = "step"
    """Name the video for the step number."""
    EPISODE = "episode"
    """Name the video for the episode number of the watched environment."""
    ITERATION = "iteration"
    """Name the video for the training iteration. Requires ``steps_per_iteration``."""


def capped_cubic_episode_trigger(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return round(episode_id ** (1.0 / 3)) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class VideoWrapper(Wrapper):
    """
    Automatically record videos during training at a regular step or episode intervals.

    Based on the RecordVideo wrapper from Gymnasium: https://gymnasium.farama.org/main/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo

    Recordings will be made from a dedicated camera, which you need to add to your environment (see the example below).

    To control how frequently recordings are made specify **one** of ``episode_trigger``, ``step_trigger``, or
    ``iteration_trigger``. They should be functions returning a boolean that indicates whether a recording should
    be started at the current episode, step, or training iteration, respectively. Step and episode counts start at 1,
    so the first environment step is step 1 of episode 1. If no trigger is passed, a default ``episode_trigger``
    will be used, which records at the episode indices 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000,.

    The training iteration is derived from the step count: each learning iteration steps the environment a fixed
    number of times (``num_steps_per_env`` in RSL-RL). Pass that value as ``steps_per_iteration`` and ``iteration_trigger``
    will be called once at the start of every iteration. The iteration count starts at ``iteration_offset`` (default 0).
    When resuming from a checkpoint, set ``iteration_offset`` to the checkpoint's iteration so the count, and the video filenames,
    line up with the framework's (see the example below).

    Set ``wait_for_episode_start`` to have every video begin at the start of an episode. A trigger then no longer
    starts the recording immediately; it waits for the next episode of the watched environment to begin. The video
    is named for the step or iteration it actually started on, which can be later than the one that triggered it.
    For example,if iteration 10 triggers a recording and the next episode begins in iteration 11, the video
    is named ``11.mp4``.

    Args:
        env: GenesisEnv
        camera_attr: The attribute of the base environment that contains the camera to use for recording.
        episode_trigger: Function that accepts an episode count integer (starting at 1) and returns ``True`` if a recording
                         should be started at this episode. Only called on the first step of an episode.
        step_trigger: Function that accepts a step count integer (starting at 1) and returns ``True`` if a recording should be started at this step
        iteration_trigger: Function that accepts a training iteration integer and returns ``True`` if a recording should be started at this iteration.
                           Requires ``steps_per_iteration``.
        steps_per_iteration: The number of environment steps per training iteration (RSL-RL's ``num_steps_per_env``).
                             Required by ``iteration_trigger`` and ``VideoFilename.ITERATION``, and when set, the
                             iteration is also passed to a ``filename`` function.
        iteration_offset: The training iteration the first step belongs to. Set this when resuming from a checkpoint.
                          Only used when ``steps_per_iteration`` is set. Can also be assigned after construction via the
                          :attr:`iteration_offset` property, as long as it is set before the first step.
        wait_for_episode_start: Delay each triggered recording until the next episode of the watched environment begins.
                                Has no effect with ``episode_trigger``, which always starts on the first step of an episode.
        video_length_sec: Length of each video, in seconds.
        out_dir: Directory to save the videos to.
        fps: Frames per second for the video.
        env_idx: If triggering on episode, this is the index of the environment to be counting episodes for.
        filename: How to name each video, from the step, episode, and iteration the recording started on.
                  - A ``VideoFilename`` names the video for that count, e.g. ``VideoFilename.STEP`` gives ``1500.mp4``.
                  - A function is called as ``filename(step, episode, iteration)`` and returns the filename, including
                    its extension. ``iteration`` is ``None`` unless ``steps_per_iteration`` is set.
                  - A string is used as the filename for every video, so each one overwrites the last.
                  - ``None`` (the default) names the video for whichever count the trigger uses.

    Example::

        class MyEnv(GenesisEnv):
            __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Construct the scene
                self.scene = gs.Scene()

                # Assign a camera to the `camera` env attribute
                self.camera = scene.add_camera(pos=(-2.5, -1.5, 1.0))


        def train():
            env = MyEnv()
            env = VideoWrapper(
                env,
                camera_attr="camera",
                out_dir="./videos"
            )
            env.build()
            ...training code...

    Record every 1500 steps::

        env = MyEnv()
        env = VideoWrapper(
            env,
            camera_attr="camera",
            out_dir="./videos",
            step_trigger=lambda step: step % 1500 == 0
        )

    Create a custom video filename::

        env = VideoWrapper(
            env,
            camera_attr="camera",
            out_dir="./videos",
            iteration_trigger=lambda it: it % 50 == 0,
            filename=lambda step, episode, iteration: f"ep{episode}_step{step}.mp4",
        )

    Record every 50 RSL-RL training iterations::

        train_cfg = {"num_steps_per_env": 24, ...}
        env = MyEnv()
        env = VideoWrapper(
            env,
            camera_attr="camera",
            out_dir="./videos",
            iteration_trigger=lambda it: it % 50 == 0,
            steps_per_iteration=train_cfg["num_steps_per_env"],
        )

    Resuming from a checkpoint with and iteration offset from RSL-RL::

        video_env = VideoWrapper(
            env,
            camera_attr="camera",
            out_dir="./videos",
            iteration_trigger=lambda it: it % 50 == 0,
            steps_per_iteration=train_cfg["num_steps_per_env"],
        )
        env = RslRlWrapper(video_env)
        env.build()

        runner = OnPolicyRunner(env, train_cfg, log_dir)
        runner.load("./logs/model_1000.pt")
        video_env.iteration_offset = runner.current_learning_iteration
        runner.learn(num_learning_iterations=500)
    """

    def __init__(
        self,
        env: GenesisEnv | Wrapper,
        camera_attr: str = "camera",
        video_length_sec: int = 8,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        iteration_trigger: Callable[[int], bool] | None = None,
        steps_per_iteration: int | None = None,
        iteration_offset: int = 0,
        wait_for_episode_start: bool = False,
        out_dir: str = "./videos",
        fps: int = 60,
        env_idx: int = 0,
        filename: (
            str | VideoFilename | Callable[[int, int, int | None], str] | None
        ) = None,
        logging: bool = True,
    ):
        super().__init__(env)
        self._is_recording: bool = False
        self._logging: bool = logging
        self._current_step: int = 1
        self._current_episode: int = 1
        self._episode_step: int = 1
        self._recording_start_step: int = 0
        self._recording_stop_step: int = 0
        self._recording_name: str = "recording"
        self._wait_for_episode_start = wait_for_episode_start
        self._recording_pending: bool = False

        self._cam: Camera | None = None
        self._camera_attr = camera_attr
        self._out_dir = out_dir
        self._video_length_steps = math.ceil(video_length_sec / self.dt)
        self._steps_per_frame = max(
            1, round(1.0 / fps / self.dt)
        )  # max prevents division by zero
        self._actual_fps = round(1.0 / self.dt / self._steps_per_frame)
        self._env_idx = env_idx

        if (
            episode_trigger is None
            and step_trigger is None
            and iteration_trigger is None
        ):
            episode_trigger = capped_cubic_episode_trigger

        trigger_count = sum(
            x is not None for x in [episode_trigger, step_trigger, iteration_trigger]
        )
        assert trigger_count == 1, "Must specify only one trigger"
        if iteration_trigger is not None:
            assert (
                steps_per_iteration is not None and steps_per_iteration > 0
            ), "steps_per_iteration is required with iteration_trigger"

        # Determine how to name the file
        if filename is None:
            if iteration_trigger is not None:
                filename = VideoFilename.ITERATION
            elif step_trigger is not None:
                filename = VideoFilename.STEP
            else:
                filename = VideoFilename.EPISODE
        if filename == VideoFilename.ITERATION:
            assert (
                steps_per_iteration is not None and steps_per_iteration > 0
            ), "steps_per_iteration is required with VideoFilename.ITERATION"
        self._filename = filename

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.iteration_trigger = iteration_trigger
        self._steps_per_iteration = steps_per_iteration
        self._iteration_offset = iteration_offset

        # Videos are encoded into a scratch directory and moved into place once complete.
        self._tmp_dir = os.path.join(self._out_dir, ".tmp")

        os.makedirs(self._out_dir, exist_ok=True)

    @property
    def video_length_steps(self) -> int:
        """
        The number of steps that will be recorded for each video.
        """
        return self._video_length_steps

    @property
    def iteration_offset(self) -> int:
        """
        The training iteration the first step belongs to.
        Set this when resuming from a checkpoint so the iteration count, and the video filenames, match the framework's.
        """
        return self._iteration_offset

    @iteration_offset.setter
    def iteration_offset(self, value: int) -> None:
        self._iteration_offset = value

    def build(self) -> None:
        """Load the camera from the environment."""
        super().build()
        self._cam = self.unwrapped.__getattribute__(self._camera_attr)
        assert (
            self._cam is not None
        ), f"Camera not found at attribute: {self.unwrapped.__class__.__name__}.{self._camera_attr}"

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Record a video image at each step."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            extras,
        ) = super().step(actions)

        # Stop recording if the recording stop step is reached
        if self._is_recording and self._recording_stop_step <= self._current_step:
            self.finish_recording()

        self._check_recording_trigger()
        if self._is_recording:
            if self._current_step % self._steps_per_frame == 0 and self._cam:
                self._cam.render()

        # Increment episode count if the watched environment has terminated or truncated
        if self._is_done(terminateds) or self._is_done(truncateds):
            self._current_episode += 1
            self._episode_step = 1
        else:
            self._episode_step += 1
        self._current_step += 1

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            extras,
        )

    def close(self):
        """Finish recording on close"""
        if self._is_recording:
            self.finish_recording()
        super().close()

    def start_recording(self):
        """Start recording a video."""
        if self._cam is None:
            return

        self._is_recording = True
        self._recording_start_step = self._current_step
        self._recording_name = self._video_filename()
        self._recording_stop_step = self._current_step + self._video_length_steps

        filepath = os.path.join(self._tmp_dir, self._recording_name)
        self._cam.start_recording(
            save_to_filename=filepath,
            fps=self._actual_fps,  # pyright: ignore[reportCallIssue]
        )

    def finish_recording(self):
        """
        Stop recording and save the video.
        """
        if not self._is_recording or self._cam is None:
            return

        # Save recording
        filepath = os.path.join(self._out_dir, self._recording_name)
        tmp_filepath = os.path.join(self._tmp_dir, self._recording_name)
        if self._logging:
            print(f"Saving recording to {filepath}")
        self._cam.stop_recording()
        if os.path.exists(tmp_filepath):
            os.replace(tmp_filepath, filepath)

        # Reset recording state
        self._is_recording = False
        self._recording_stop_step = 0

    def _check_recording_trigger(self) -> bool:
        """Check if a recording should be started"""
        if self._is_recording:
            return False

        # Check the recording triggers
        if not self._recording_pending:
            self._recording_pending = self._is_triggered()
        if not self._recording_pending:
            return False

        # Hold a triggered recording until the watched env starts a new episode
        if self._wait_for_episode_start and self._episode_step > 1:
            return False

        self._recording_pending = False
        self.start_recording()
        return True

    def _is_triggered(self) -> bool:
        """Check if the active trigger fires on the current step"""
        if self.episode_trigger is not None:
            # Only start on the first step of an episode, never part way through one
            if self._episode_step > 1:
                return False
            return self.episode_trigger(self._current_episode)
        if self.step_trigger is not None:
            return self.step_trigger(self._current_step)
        if self.iteration_trigger is not None and self._steps_per_iteration is not None:
            # Only trigger on the first step of each iteration
            step_idx = self._current_step - 1  # make step 0-based count
            if step_idx % self._steps_per_iteration != 0:
                return False
            return self.iteration_trigger(self._current_iteration())
        return False

    def _current_iteration(self) -> int | None:
        """The training iteration the current step belongs to, or None without ``steps_per_iteration``"""
        if self._steps_per_iteration is None:
            return None
        # Iterations are counted from `iteration_offset`
        iteration = (self._current_step - 1) // self._steps_per_iteration
        return self._iteration_offset + iteration

    def _video_filename(self) -> str:
        """The filename for a video starting on the current step"""
        filename = self._filename
        if isinstance(filename, str):
            return filename
        if isinstance(filename, VideoFilename):
            index = {
                VideoFilename.STEP: self._current_step,
                VideoFilename.EPISODE: self._current_episode,
                VideoFilename.ITERATION: self._current_iteration(),
            }[filename]
            return f"{index}.mp4"
        return filename(
            self._current_step, self._current_episode, self._current_iteration()
        )

    def _is_done(self, term_buffer: torch.Tensor | None) -> bool:
        """
        Check if the watched environment has terminated or truncated.

        Args:
            term_buffer: The termination buffer to check.

        Returns:
            True if the watched environment has terminated or truncated, False otherwise.
        """
        if term_buffer is None:
            return False
        value = term_buffer[self._env_idx]
        return bool(value.item()) if isinstance(value, torch.Tensor) else bool(value)
