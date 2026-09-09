import cv2

DRAW = "draw"
MOVE = "move"
IDLE = "idle"

CONFIDENCE_GREEN = "green"
CONFIDENCE_YELLOW = "yellow"
CONFIDENCE_RED = "red"

_FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class MotionTracker:
    """Converts fingertip position + pinch state over time into draw/move gestures."""

    def __init__(self):
        self._last_point = None
        self._was_pinching = False

    def update(self, point, is_pinching):
        """
        point: (x, y) fingertip position this frame, or None if no hand was detected.
        Returns (action, segment): action is one of DRAW/MOVE/IDLE; segment is the
        (start, end) tuple to draw when action is DRAW, else None.
        """
        if point is None:
            self._last_point = None
            self._was_pinching = False
            return IDLE, None

        if is_pinching and self._was_pinching and self._last_point is not None:
            action, segment = DRAW, (self._last_point, point)
        else:
            action, segment = MOVE, None

        self._last_point = point
        self._was_pinching = is_pinching
        return action, segment


class TrackingConfidence:
    """
    Tracks a set of background optical-flow features (Lucas-Kanade pyramid) frame to
    frame and reports what fraction were successfully matched. This is a proxy for
    overall tracking reliability: when the scene has little texture, lighting changes
    abruptly, or the camera/head moves fast, fewer features survive and confidence
    drops - which correlates with fingertip position drift. See CALIBRATION.md.
    """

    def __init__(self, green_threshold=0.7, yellow_threshold=0.4, reseed_every=30, min_features=10):
        self._green_threshold = green_threshold
        self._yellow_threshold = yellow_threshold
        self._reseed_every = reseed_every
        self._min_features = min_features
        self._prev_gray = None
        self._prev_points = None
        self._frames_since_reseed = 0
        self.tracked_count = 0
        self.matched_count = 0
        self.confidence = 1.0

    def _seed(self, gray):
        self._prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **_FEATURE_PARAMS)
        self._frames_since_reseed = 0

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if (
            self._prev_gray is None
            or self._prev_points is None
            or len(self._prev_points) < self._min_features
        ):
            self._seed(gray)
            self._prev_gray = gray
            self.tracked_count = 0 if self._prev_points is None else len(self._prev_points)
            self.matched_count = self.tracked_count
            self.confidence = 1.0 if self.tracked_count >= self._min_features else 0.0
            return self.confidence

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **_LK_PARAMS
        )

        self.tracked_count = len(self._prev_points)
        if new_points is None or status is None:
            self.matched_count = 0
            self.confidence = 0.0
            self._prev_points = None
        else:
            matched_mask = status.flatten() == 1
            self.matched_count = int(matched_mask.sum())
            self.confidence = self.matched_count / max(self.tracked_count, 1)
            self._prev_points = new_points[matched_mask].reshape(-1, 1, 2)

        self._frames_since_reseed += 1
        if self._frames_since_reseed >= self._reseed_every or self.matched_count < self._min_features:
            self._seed(gray)

        self._prev_gray = gray
        return self.confidence

    def level(self):
        if self.confidence >= self._green_threshold:
            return CONFIDENCE_GREEN
        if self.confidence >= self._yellow_threshold:
            return CONFIDENCE_YELLOW
        return CONFIDENCE_RED
