import time

import cv2

from airdraw.canvas import Canvas
from airdraw.capture import Capture
from airdraw.hands import HandTracker
from airdraw.motion import DRAW, MotionTracker, TrackingConfidence

# Normalized thumb-tip/index-tip distance below which a pinch is registered.
# See CALIBRATION.md for how to tune this per hand size / camera distance.
PINCH_THRESHOLD = 0.45


class App:
    """Wires capture, hand tracking, motion, and canvas into the main loop."""

    def __init__(self, pinch_threshold=PINCH_THRESHOLD):
        self._pinch_threshold = pinch_threshold
        self._capture = Capture()
        self._hands = HandTracker()
        self._gestures = MotionTracker()
        self._flow_confidence = TrackingConfidence()
        self._canvas = None
        self._fps_smoothed = 0.0
        self._last_tick = time.perf_counter()

    def run(self):
        try:
            while True:
                frame = self._capture.get_frame()
                if frame is None:
                    break
                if not self._process_frame(frame):
                    break
        finally:
            self._hands.close()
            self._capture.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame):
        """Runs one frame through the pipeline and displays it. Returns False to stop."""
        if self._canvas is None:
            h, w = frame.shape[:2]
            self._canvas = Canvas(w, h)

        self._flow_confidence.update(frame)

        hand = self._hands.process(frame)
        point = hand.index_tip if hand else None
        is_pinching = hand.is_pinching(self._pinch_threshold) if hand else False

        action, segment = self._gestures.update(point, is_pinching)
        if action == DRAW and segment is not None:
            self._canvas.draw_segment(*segment)

        display = self._canvas.composite(frame)
        Canvas.draw_confidence_indicator(
            display,
            self._flow_confidence.level(),
            self._flow_confidence.confidence,
            self._flow_confidence.matched_count,
            self._flow_confidence.tracked_count,
        )
        self._draw_fps(display)

        cv2.imshow("AirDraw", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("c"):
            self._canvas.clear()
        return True

    def _draw_fps(self, frame):
        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            fps = 1.0 / dt
            self._fps_smoothed = fps if not self._fps_smoothed else self._fps_smoothed * 0.9 + fps * 0.1
        label = f"{self._fps_smoothed:.1f} fps"
        x = frame.shape[1] - 120
        cv2.putText(frame, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
