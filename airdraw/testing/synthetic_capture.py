"""Synthetic frame source used for automated soak-testing without a real webcam."""
import numpy as np


class SyntheticCapture:
    """Generates textured frames that drift/jitter over time, mimicking Capture's interface."""

    def __init__(self, width=640, height=480, num_frames=300, seed=0):
        self._width = width
        self._height = height
        self._num_frames = num_frames
        self._frame_idx = 0
        rng = np.random.default_rng(seed)
        self._noise = rng.integers(0, 255, (height + 40, width + 40, 3), dtype=np.uint8)

    def get_frame(self):
        if self._frame_idx >= self._num_frames:
            return None
        t = self._frame_idx
        dx = int(10 * np.sin(t * 0.05))
        dy = int(10 * np.cos(t * 0.05))
        frame = self._noise[20 + dy:20 + dy + self._height, 20 + dx:20 + dx + self._width].copy()
        self._frame_idx += 1
        return frame

    def release(self):
        pass
