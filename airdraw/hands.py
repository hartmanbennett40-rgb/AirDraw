import math

import mediapipe as mp

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
MIDDLE_MCP = 9


class HandTracker:
    """Detects a single hand's landmarks using MediaPipe Hands and derives a
    pinch gesture from the thumb-tip/index-tip distance."""

    def __init__(self, pinch_threshold=0.12, debounce_frames=3):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._pinch_threshold = pinch_threshold
        self._debounce_frames = debounce_frames

        self._is_pinching = False
        self._pending_state = None
        self._pending_count = 0

    def update(self, frame):
        """Process one BGR frame. Returns (found, fingertip_px, is_pinching).

        fingertip_px is an (x, y) pixel tuple for the index fingertip, or
        None if no hand was found. is_pinching is debounced across frames
        to avoid flicker at the pinch threshold.
        """
        height, width = frame.shape[:2]
        rgb_frame = frame[:, :, ::-1]
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self._pending_state = None
            self._pending_count = 0
            return False, None, self._is_pinching

        landmarks = results.multi_hand_landmarks[0].landmark

        index_tip = landmarks[INDEX_TIP]
        fingertip_px = (int(index_tip.x * width), int(index_tip.y * height))

        raw_pinch = self._compute_pinch(landmarks)
        self._debounce(raw_pinch)

        return True, fingertip_px, self._is_pinching

    def _compute_pinch(self, landmarks):
        thumb_tip = landmarks[THUMB_TIP]
        index_tip = landmarks[INDEX_TIP]
        wrist = landmarks[WRIST]
        middle_mcp = landmarks[MIDDLE_MCP]

        pinch_distance = math.hypot(
            thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y
        )
        scale = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
        if scale < 1e-6:
            return self._is_pinching

        normalized_distance = pinch_distance / scale
        return normalized_distance < self._pinch_threshold

    def _debounce(self, raw_pinch):
        if raw_pinch == self._is_pinching:
            self._pending_state = None
            self._pending_count = 0
            return

        if raw_pinch == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_pinch
            self._pending_count = 1

        if self._pending_count >= self._debounce_frames:
            self._is_pinching = raw_pinch
            self._pending_state = None
            self._pending_count = 0

    def close(self):
        self._hands.close()
