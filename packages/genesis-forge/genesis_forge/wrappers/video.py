from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers.wrapper import Wrapper

if TYPE_CHECKING:
    from genesis.vis.camera import Camera


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
    be started at the current episode, step, or training iteration, respectively. If no trigger is passed,
    a default ``episode_trigger`` will be used, which records at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000,.

    The training iteration is derived from the step count: each learning iteration steps the environment a fixed
    number of times (``num_steps_per_env`` in RSL-RL). Pass that value as ``steps_per_iteration`` and ``iteration_trigger``
    will be called once at the start of every iteration. The iteration count starts at ``iteration_offset`` (default 0).
    When resuming from a checkpoint, set ``iteration_offset`` to the checkpoint's iteration so the count, and the video filenames,
    line up with the framework's (see the example below).

    Args:
        env: GenesisEnv
        camera_attr: The attribute of the base environment that contains the camera to use for recording.
        episode_trigger: Function that accepts an episode count integer and returns ``True`` if a recording should be started at this episode
        step_trigger: Function that accepts a step count integer and returns ``True`` if a recording should be started at this step
        iteration_trigger: Function that accepts a training iteration integer and returns ``True`` if a recording should be started at this iteration.
                           Requires ``steps_per_iteration``.
        steps_per_iteration: The number of environment steps per training iteration (RSL-RL's ``num_steps_per_env``).
                             Only used with ``iteration_trigger``.
        iteration_offset: The training iteration the first step belongs to. Set this when resuming from a checkpoint.
                          Only used with ``iteration_trigger``. Can also be assigned after construction via the
                          :attr:`iteration_offset` property, as long as it is set before the first step.
        video_length_sec: Length of each video, in seconds.
        out_dir: Directory to save the videos to.
        fps: Frames per second for the video.
        env_idx: If triggering on episode, this is the index of the environment to be counting episodes for.
        filename: The filename for the video.
                  If None, the video will automatically be named for the episode, step, or iteration the recording started on.
                  If defined, each video will overwrite the previous video with this name.

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
        out_dir: str = "./videos",
        fps: int = 60,
        env_idx: int = 0,
        filename: str | None = None,
        logging: bool = True,
    ):
        super().__init__(env)
        self._is_recording: bool = False
        self._logging: bool = logging
        self._current_step: int = 0
        self._current_episode: int = 0
        self._recording_start_step: int = 0
        self._recording_stop_step: int = 0
        self._recording_name: str = "recording"

        self._cam: Camera | None = None
        self._camera_attr = camera_attr
        self._out_dir = out_dir
        self._filename = filename
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

    def start_recording(self, index: int):
        """Start recording a video."""
        if self._cam is None:
            return

        self._is_recording = True
        self._recording_start_step = self._current_step
        self._recording_name = self._filename or f"{index}.mp4"
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
        record = False
        index = None
        if not self._is_recording:
            if self.episode_trigger is not None:
                record = self.episode_trigger(self._current_episode)
                index = self._current_episode
            elif self.step_trigger is not None:
                record = self.step_trigger(self._current_step)
                index = self._current_step
            elif (
                self.iteration_trigger is not None
                and self._steps_per_iteration is not None
                and self._current_step % self._steps_per_iteration == 0
            ):
                iteration = self._current_step // self._steps_per_iteration
                index = self._iteration_offset + iteration
                record = self.iteration_trigger(index)

        if record and index is not None:
            self.start_recording(index)
        return record

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
