import cv2
import numpy as np

from airdraw.motion import CONFIDENCE_GREEN, CONFIDENCE_RED, CONFIDENCE_YELLOW

_LEVEL_COLORS = {
    CONFIDENCE_GREEN: (0, 200, 0),
    CONFIDENCE_YELLOW: (0, 200, 255),
    CONFIDENCE_RED: (0, 0, 255),
}


class Canvas:
    """Holds the persistent drawing surface and composites it onto camera frames."""

    def __init__(self, width, height, color=(255, 255, 255), thickness=3):
        self._surface = np.zeros((height, width, 3), dtype=np.uint8)
        self._color = color
        self._thickness = thickness

    def draw_segment(self, start, end):
        cv2.line(self._surface, _to_int(start), _to_int(end), self._color, self._thickness, cv2.LINE_AA)

    def clear(self):
        self._surface[:] = 0

    def composite(self, frame_bgr):
        mask = self._surface.any(axis=2)
        out = frame_bgr.copy()
        out[mask] = self._surface[mask]
        return out

    @staticmethod
    def draw_confidence_indicator(frame_bgr, level, confidence, matched, tracked):
        """Green/yellow/red dot + label showing this frame's optical-flow match rate."""
        color = _LEVEL_COLORS[level]
        cv2.circle(frame_bgr, (30, 30), 12, color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (30, 30), 12, (0, 0, 0), 1, cv2.LINE_AA)
        label = f"tracking: {level} ({matched}/{tracked})"
        cv2.putText(frame_bgr, label, (52, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, label, (52, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr


def _to_int(point):
    return (int(round(point[0])), int(round(point[1])))
