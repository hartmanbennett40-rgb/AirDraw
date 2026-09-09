import cv2
import numpy as np


class MotionTracker:
    """Rough per-frame camera pan estimate from sparse optical flow.

    This is NOT 6DOF pose estimation - there's no camera model and no
    rotation/translation decomposition, no depth. It tracks a handful of
    corner features (cv2.goodFeaturesToTrack) with Lucas-Kanade optical
    flow (cv2.calcOpticalFlowPyrLK) between consecutive grayscale frames
    and reports the median 2D displacement of those points as a proxy for
    "how far did the whole scene appear to shift in pixels". It cannot
    tell camera rotation apart from camera translation, or from something
    physically moving in front of the lens (like the tracked hand) - they
    all just look like "points moved". The result is smoothed with an
    exponential moving average to cut down frame-to-frame jitter.
    """

    def __init__(self, alpha=0.5, max_corners=200, quality_level=0.01,
                 min_distance=7, block_size=7):
        self.alpha = alpha  # EMA weight on the new sample; higher = less smoothing
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
        )

        self._prev_gray = None
        self._prev_points = None
        self.dx = 0.0
        self.dy = 0.0

    def update(self, frame):
        """Feed the next BGR frame, return the smoothed (dx, dy) pan estimate in pixels."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_points is None or len(self._prev_points) < 4:
            self._prev_gray = gray
            self._prev_points = self._find_features(gray)
            return self.dx, self.dy

        new_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None, **self._lk_params
        )

        good_new = None
        if new_points is not None and status is not None:
            mask = status.flatten() == 1
            good_new = new_points[mask]
            good_old = self._prev_points[mask]

            if len(good_new) >= 4:
                # Median, not mean, so a handful of features stuck on a
                # moving hand/object can't drag the whole estimate around.
                deltas = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
                raw_dx, raw_dy = np.median(deltas, axis=0)
                self.dx = self.alpha * raw_dx + (1 - self.alpha) * self.dx
                self.dy = self.alpha * raw_dy + (1 - self.alpha) * self.dy

        self._prev_gray = gray
        if good_new is not None and len(good_new) >= max(4, self.max_corners // 4):
            self._prev_points = good_new.reshape(-1, 1, 2).astype(np.float32)
        else:
            # Lost too many tracks (fast motion, low light, or a blank
            # wall with no corners to find) - start over from fresh features.
            self._prev_points = self._find_features(gray)

        return self.dx, self.dy

    def _find_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )
