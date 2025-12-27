from __future__ import annotations

import os
import math
import torch
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.wrappers.wrapper import Wrapper
from typing import Tuple, Any, Callable, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from genesis.vis.camera import Camera

RecordingType = Literal["active", "background"]


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


def capped_cubic_iteration_trigger(iteration: int) -> bool:
    """The default iteration trigger.

    This function will trigger recordings at the iteration indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        iteration: The iteration number

    Returns:
        If to apply a video schedule number
    """
    if iteration < 1000:
        return int(round(iteration ** (1.0 / 3))) ** 3 == iteration
    else:
        return iteration % 1000 == 0


class VideoWrapper(Wrapper):
    """
    Automatically record videos during training at a regular step, episode, or iteration intervals.

    Based on the RecordVideo wrapper from Gymnasium: https://gymnasium.farama.org/main/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo

    Recordings will be made from a dedicated camera, which you need to add to your environment (see the example below).

    To control how frequently recordings are made specify **one of** ``episode_trigger``, ``step_trigger``, or ``iteration_trigger``.
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode, step, or iteration, respectively. If none are passed,
    a default ``episode_trigger`` will be used, which records at the episode indices 0, 1, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000,.

    Args:
        env: GenesisEnv
        camera_attr: The attribute of the base environment that contains the camera to use for recording.
        episode_trigger: Function that accepts an episode count integer and returns ``True`` if a recording should be started at this episode
        step_trigger: Function that accepts a step count integer and returns ``True`` if a recording should be started at this step
        iteration_trigger: Function that accepts an iteration count integer and returns ``True`` if a recording should be started at this iteration.
                          Requires ``num_steps_per_env`` to be set to calculate iteration from step count.
        num_steps_per_env: Number of steps per environment per iteration. Required when using ``iteration_trigger``.
        initial_iteration: Initial iteration offset (useful when resuming training).
        video_length_sec: Length of each video, in seconds.
        out_dir: Directory to save the videos to.
        fps: Frames per second for the video.
        env_idx: If triggering on episode, this is the index of the environment to be counting episodes for.
        filename: The filename for the video.
                  If None, the video will automatically be named for the current step (or iteration if using iteration_trigger).
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

    Record every 50 iterations (useful for RL training)::

        env = MyEnv()
        env = VideoWrapper(
            env,
            camera_attr="camera",
            out_dir="./videos",
            iteration_trigger=lambda it: it % 50 == 0,
            num_steps_per_env=24,  # Should match your training config
        )
    """

    def __init__(
        self,
        env: GenesisEnv,
        camera_attr: str = "camera",
        video_length_sec: int = 8,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        iteration_trigger: Callable[[int], bool] | None = None,
        num_steps_per_env: Optional[int] = None,
        initial_iteration: int = 0,
        out_dir: str = "./videos",
        fps: int = 60,
        env_idx: int = 0,
        filename: str = None,
        record_final_episode: bool = True,
        logging: bool = True,
    ):
        super().__init__(env)
        self._is_recording: bool = False
        self._logging: bool = logging
        self._current_step: int = 0
        self._current_episode: int = 0
        self._current_iteration: int = initial_iteration
        self._last_triggered_iteration: int = initial_iteration - 1
        self._recording_start_step: int = 0
        self._recording_start_iteration: int = 0
        self._recording_stop_step: int = 0
        self._record_final_episode = record_final_episode
        self._has_recording_buffer = False

        # active: a triggered recording that will save to file
        # background: a recording that will only be saved if the environment is closed.
        #             This is so you get a video of the final episode, even if it was not triggered.
        self._recording_type: RecordingType = "background"

        self._cam: Camera = None
        self._camera_attr = camera_attr
        self._out_dir = out_dir
        self._filename = filename
        self._video_length_steps = math.ceil(video_length_sec / self.dt)
        self._steps_per_frame = round(1.0 / fps / self.dt)
        self._actual_fps = round(1.0 / self.dt / self._steps_per_frame)
        self._env_idx = env_idx

        # Iteration tracking
        self._num_steps_per_env = num_steps_per_env
        self._initial_iteration = initial_iteration

        # Validate trigger configuration
        if iteration_trigger is not None and num_steps_per_env is None:
            raise ValueError("num_steps_per_env must be specified when using iteration_trigger")

        if episode_trigger is None and step_trigger is None and iteration_trigger is None:
            episode_trigger = capped_cubic_episode_trigger

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger, iteration_trigger])
        assert trigger_count == 1, "Must specify only one trigger (episode_trigger, step_trigger, or iteration_trigger)"

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.iteration_trigger = iteration_trigger

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

        # Update iteration count if using iteration trigger
        if self._num_steps_per_env is not None:
            self._current_iteration = self._initial_iteration + (self._current_step // self._num_steps_per_env)

        self._check_recording_trigger()
        if self._current_step % self._steps_per_frame == 0:
            self._cam.render()

        # Stop recording if the recording stop step is reached
        if self._is_recording and self._recording_stop_step <= self._current_step:
            self.finish_recording()

        # Increment episode count
        terminated = False if terminateds is None else terminateds[self._env_idx]
        truncated = False if truncateds is None else truncateds[self._env_idx]
        if torch.any(terminated or truncated):
            self._current_episode += 1
            # If we're not recording, start a background recording at the beginning of the episode
            # The last one of these will be saved when the environment is closed as the final training episode
            if not self._is_recording and self._record_final_episode:
                self.start_recording("background")
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
        if self._is_recording or self._has_recording_buffer:
            self.finish_recording()
        super().close()

    def start_recording(self, type: RecordingType = "active"):
        """Start recording a video."""
        # Clear any existing frames
        self._cam._recorded_imgs.clear()

        self._is_recording = True
        self._has_recording_buffer = False
        self._recording_type = type
        self._recording_start_step = self._current_step
        self._recording_start_iteration = self._current_iteration
        self._recording_stop_step = self._current_step + self._video_length_steps
        self._cam.start_recording()

    def finish_recording(self):
        """
        Stop recording and save the video, if the recording type is 'active'.
        """
        if not self._is_recording and self._cam is not None:
            return

        # Save recording
        if self._recording_type == "active":
            # Use iteration-based filename if iteration_trigger is active
            if self._filename:
                filename = self._filename
            elif self.iteration_trigger is not None:
                filename = f"iter_{self._recording_start_iteration}.mp4"
            else:
                filename = f"{self._recording_start_step}.mp4"
            filepath = os.path.join(self._out_dir, filename)
            if self._logging:
                print(f"Saving recording to {filepath}")
            self._cam.stop_recording(filepath, fps=self._actual_fps)
            self._has_recording_buffer = False
        else:
            self._cam.pause_recording()
            self._has_recording_buffer = True

        # Reset recording state
        self._is_recording = False
        self._recording_type = None
        self._recording_stop_step = 0

    def _check_recording_trigger(self) -> bool:
        """Check if a recording should be started"""
        record = False
        if self._is_recording and self._recording_type == "active":
            record = False
        elif self.episode_trigger is not None:
            record = self.episode_trigger(self._current_episode)
        elif self.step_trigger is not None:
            record = self.step_trigger(self._current_step)
        elif self.iteration_trigger is not None:
            # Only trigger once per iteration (at the start of each new iteration)
            if self._current_iteration > self._last_triggered_iteration:
                if self.iteration_trigger(self._current_iteration):
                    record = True
                    self._last_triggered_iteration = self._current_iteration

        if record:
            self.start_recording()
        return record
