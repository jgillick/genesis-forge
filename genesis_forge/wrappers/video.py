from __future__ import annotations

import os
import math
import inspect
import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers.wrapper import Wrapper
from typing import Tuple, Any, Callable, TYPE_CHECKING

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
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class VideoWrapper(Wrapper):
    """
    Automatically record videos during training at a regular step or episode intervals.

    Based on the RecordVideo wrapper from Gymnasium: https://gymnasium.farama.org/main/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo

    Recordings will be made from a dedicated camera, which you need to add to your environment (see the example below).

    To control how frequently recordings are made specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively. If neither :attr:`episode_trigger` nor ``step_trigger`` is passed,
    a default ``episode_trigger`` will be used, which records at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000,.

    Args:
        env: GenesisEnv
        camera_attr: The attribute of the base environment that contains the camera to use for recording.
        episode_trigger: Function that accepts an episode count integer and returns ``True`` if a recording should be started at this episode
        step_trigger: Function that accepts a step count integer and returns ``True`` if a recording should be started at this step
        video_length_sec: Length of each video, in seconds.
        out_dir: Directory to save the videos to.
        fps: Frames per second for the video.
        env_idx: If triggering on episode, this is the index of the environment to be counting episodes for.
        filename: The filename for the video.
                  If None, the video will automatically be named for the current step.
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
    """

    def __init__(
        self,
        env: GenesisEnv | Wrapper,
        camera_attr: str = "camera",
        video_length_sec: int = 8,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
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

        self._cam: Camera | None = None
        self._start_recording_has_args: bool = False
        self._camera_attr = camera_attr
        self._out_dir = out_dir
        self._filename = filename
        self._video_length_steps = math.ceil(video_length_sec / self.dt)
        self._steps_per_frame = max(1, round(1.0 / fps / self.dt)) # max prevents division by zero
        self._actual_fps = round(1.0 / self.dt / self._steps_per_frame)
        self._env_idx = env_idx

        self._recording_filepath = os.path.join(self._out_dir, ".recording.mp4")

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_episode_trigger

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify only one trigger"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger

        os.makedirs(self._out_dir, exist_ok=True)

    @property
    def video_length_steps(self) -> int:
        """
        The number of steps that will be recorded for each video.
        """
        return self._video_length_steps

    def build(self) -> None:
        """Load the camera from the environment."""
        super().build()
        self._cam = self.unwrapped.__getattribute__(self._camera_attr)
        assert (
            self._cam is not None
        ), f"Camera not found at attribute: {self.unwrapped.__class__.__name__}.{self._camera_attr}"

        # Genesis >= 1.3 moved the filename/fps args from stop_recording to start_recording
        start_recording_params = inspect.signature(self._cam.start_recording).parameters
        self._start_recording_has_args = "save_to_filename" in start_recording_params

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Record a video image at each step."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            extras,
        ) = super().step(actions)

        self._check_recording_trigger()
        if self._is_recording:
            if self._current_step % self._steps_per_frame == 0 and self._cam:
                self._cam.render()

            # Stop recording if the recording stop step is reached
            if self._recording_stop_step <= self._current_step:
                self.finish_recording()

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

    def start_recording(self):
        """Start recording a video."""
        if self._cam is None:
            return

        self._is_recording = True
        self._recording_start_step = self._current_step
        self._recording_stop_step = self._current_step + self._video_length_steps

        # Genesis >= 1.3: filename/fps are passed to start_recording
        if self._start_recording_has_args:
            self._cam.start_recording(
                save_to_filename=self._recording_filepath, fps=self._actual_fps # pyright: ignore[reportCallIssue]
            )
        else:
            self._cam.start_recording()

    def finish_recording(self):
        """
        Stop recording and save the video.
        """
        if not self._is_recording or self._cam is None:
            return

        filename = self._filename or f"{self._recording_start_step}.mp4"
        filepath = os.path.join(self._out_dir, filename)

        # Save recording
        if self._logging:
            print(f"Saving recording to {filepath}")
        if self._start_recording_has_args:
            self._cam.stop_recording()
            os.rename(self._recording_filepath, filepath)
        else:
            self._cam.stop_recording(filepath, fps=self._actual_fps)

        # Reset recording state
        self._is_recording = False
        self._recording_stop_step = 0

    def _check_recording_trigger(self) -> bool:
        """Check if a recording should be started"""
        record = False
        if not self._is_recording:
            if self.episode_trigger is not None:
                record = self.episode_trigger(self._current_episode)
            if self.step_trigger is not None:
                record = self.step_trigger(self._current_step)

        if record:
            self.start_recording()
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
