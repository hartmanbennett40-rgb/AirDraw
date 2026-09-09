import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# MediaPipe hand-landmark indices used below (see
# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "hand_landmarker.task",
)


def _ensure_model(path=_MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


class HandResult:
    """21 hand landmarks (pixel coordinates) for a single detected hand."""

    def __init__(self, points):
        self.points = points

    @property
    def index_tip(self):
        return self.points[INDEX_TIP]

    @property
    def hand_scale(self):
        # wrist -> middle-finger-MCP distance, used to normalize pinch distance
        # so the pinch threshold doesn't depend on hand size or distance from camera.
        wx, wy = self.points[WRIST]
        mx, my = self.points[MIDDLE_MCP]
        return max(((wx - mx) ** 2 + (wy - my) ** 2) ** 0.5, 1e-6)

    def pinch_distance(self):
        tx, ty = self.points[THUMB_TIP]
        ix, iy = self.points[INDEX_TIP]
        return ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5 / self.hand_scale

    def is_pinching(self, threshold):
        return self.pinch_distance() < threshold


class HandTracker:
    """Detects a single hand's landmarks in a frame using MediaPipe's HandLandmarker task."""

    def __init__(
        self,
        max_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_path=None,
    ):
        resolved_path = _ensure_model(model_path or _MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=resolved_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start = time.perf_counter()

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self._start) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        points = [(lm.x * w, lm.y * h) for lm in result.hand_landmarks[0]]
        return HandResult(points)

    def close(self):
        self._landmarker.close()
