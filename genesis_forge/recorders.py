from __future__ import annotations

import numpy as np
import imageio

from genesis.options.recorders import RecorderOptions
from genesis.recorders.base_recorder import Recorder
from genesis.recorders.recorder_manager import register_recording


class AsyncVideoFile(RecorderOptions):
    """
    Records sensor frames to an MP4 video file asynchronously using a background thread.

    Unlike the built-in ``VideoFile`` recorder, this recorder uses ``run_in_thread = True``
    so encoding never blocks the simulation step. Frames are streamed into a queue and
    consumed by a background thread that encodes them incrementally via imageio/ffmpeg.

    Use :meth:`start_clip` and :meth:`stop_clip` on the returned recorder handle to control
    which portion of the simulation is saved to each file.

    Parameters
    ----------
    filename : str
        Output ``.mp4`` file path. Acts as a default; can be overridden per clip via
        :meth:`AsyncVideoRecorder.start_clip`.
    fps : int | None
        Frames per second for the output video. If ``None``, inferred from the ``hz``
        option (or simulation dt if ``hz`` is also ``None``).
    codec : str
        FFmpeg codec for encoding. Defaults to ``"libx264"``.
    env_idx : int
        Index of the environment whose frame to extract when the sensor returns
        multi-environment data. Defaults to ``0``.
    """

    filename: str = "output.mp4"
    fps: int | None = None
    codec: str = "libx264"
    env_idx: int = 0


@register_recording(AsyncVideoFile)
class AsyncVideoRecorder(Recorder):
    """
    Video recorder that streams frames to an MP4 in a background thread.

    Registered automatically via :func:`register_recording` and used by :class:`AsyncVideoFile`.
    Not intended to be instantiated directly; use ``sensor.start_recording(AsyncVideoFile(...))``
    from your wrapper's ``build()`` method instead.

    Call :meth:`start_clip` to begin capturing a new video segment and :meth:`stop_clip`
    to finalize it. The recorder remains active (background thread alive) between clips,
    with zero overhead when no clip is active.
    """

    def build(self):
        super().build()
        self._writer = None
        self._clip_active = False
        self._clip_filepath = self._options.filename
        self._env_idx = self._options.env_idx
        self._fps = self._options.fps or int(
            round(1.0 / (self._steps_per_sample * self._manager._step_dt))
        )

    def step(self, global_step: int):
        """Skip data capture entirely when no clip is active."""
        if not self._clip_active:
            return
        super().step(global_step)

    def process(self, data, cur_time: float):
        """Encode one frame. Called in the background thread."""
        # data is the return value of sensor.read() -- a CameraData namedtuple
        # with a .rgb field shaped (n_envs, H, W, 3) or (H, W, 3) for single env
        rgb = data.rgb if hasattr(data, "rgb") else data

        if isinstance(rgb, np.ndarray):
            frame = rgb
        else:
            frame = rgb.detach().cpu().numpy()

        # Extract the target environment's frame
        if frame.ndim == 4:
            frame = frame[self._env_idx]

        frame = frame.astype(np.uint8)

        if self._writer is None:
            self._writer = imageio.get_writer(
                self._clip_filepath,
                fps=self._fps,
                codec=self._options.codec,
                format="FFMPEG",
            )
        self._writer.append_data(frame)

    def start_clip(self, filepath: str):
        """
        Begin recording a new video clip to the given file path.

        Should be called from the main thread in response to a trigger firing.
        Any previously open writer is finalized before opening the new file.
        """
        self._close_writer()
        self._clip_filepath = filepath
        self._clip_active = True

    def stop_clip(self):
        """
        Stop the current clip and finalize the video file.

        Drains the frame queue so every queued frame is encoded before the writer is
        closed. The brief block here is bounded to the few frames still in-flight, not
        the full video length.
        """
        self._clip_active = False
        # Wait for background thread to process all frames already in the queue
        if self._data_queue is not None:
            self._data_queue.join()
        self._close_writer()

    def cleanup(self):
        """Called by RecorderManager when the scene is torn down."""
        self._clip_active = False
        if self._data_queue is not None:
            self._data_queue.join()
        self._close_writer()

    def _close_writer(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def run_in_thread(self) -> bool:
        return True
