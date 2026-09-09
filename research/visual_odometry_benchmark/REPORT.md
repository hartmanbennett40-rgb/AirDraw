# Visual odometry vs. optical-flow drift — findings

## Context / caveat

`airdraw/motion.py` is currently an empty stub (`class MotionTracker: pass`) —
there is no existing optical-flow implementation in this repo to swap out, and
nothing in the current app tracks camera ego-motion at all (it's a fixed-webcam
fingertip tracker). This research treats the question in isolation: *if* we
later need to compensate for webcam pan while drawing, is essential-matrix-based
VO worth the complexity over a naive 2D-flow accumulator? The prototype and
benchmark below don't touch `motion.py` or any other app code.

## Method

No physical camera is available in this environment, so drift was measured
against **known ground truth** instead of a real recording — the only way to
get an actual drift *number* rather than a guess:

- Synthetic 3D scene: 250 points scattered at 3–12 units depth.
- Synthetic camera trajectory over 900 frames (30 fps × 30 s): yaw pans
  0° → 25° → 0° (smooth sine), plus small handheld pitch/roll wobble and a
  few-mm translational sway — realistic for a laptop webcam being nudged/panned,
  not a tripod.
- Each frame is actually rendered (points projected with a calibrated `K`,
  pixel noise added) and run through **real OpenCV**: `goodFeaturesToTrack` +
  `calcOpticalFlowPyrLK` for tracking, `findEssentialMat` + `recoverPose` for
  VO, `findHomography` + `decomposeHomographyMat` for the rotation-only
  variant. Both estimators consume the *same* tracked correspondences per
  frame, so the comparison isolates the estimator, not the tracker.
- Metric: yaw drift vs. ground truth (which returns to exactly 0° at t=30s),
  reported as final error, RMS error, and error percentiles across the run.

Run `python3 benchmark.py` / `python3 benchmark_homography.py` to reproduce
(needs `opencv-python`, `numpy`).

## Results (900 frames, 30s pan-and-return)

| Approach | Final drift | RMS error | Median \|error\| | p90 \|error\| | Cost/frame |
|---|---|---|---|---|---|
| Naive optical-flow accumulation | **+0.04°** | 1.60° | 1.30° | 2.82° | 0.08 ms |
| Essential matrix (`recoverPose`) | +28.7° | 108.6° | 96.8° | 163.1° | 1.76 ms |
| Homography decomposition (rotation-only model) | +1.24° | 36.5° | 37.4° | 50.1° | ~similar to above |

**The naive approach won on every metric, including with zero pixel noise.**

## Why VO loses here

This isn't noise or a tuning issue — it's structural. Frame-to-frame
translation baseline in this trajectory is ~0.001 units against a scene depth
of ~7.5 units (ratio ≈ 0.0001, confirmed by direct measurement of the
synthetic trajectory). Two-view geometry (`E = [t]_x R`) needs a meaningful
translation baseline to disambiguate rotation from translation; a webcam pan
is close to pure rotation about the camera center, which is the textbook
degenerate case for essential-matrix estimation. I confirmed this by re-running
with **zero pixel noise** (perfect correspondences) — `recoverPose` still blew
up to -147° final drift, so it's not a tracking-quality problem, it's the
motion itself.

Swapping to homography decomposition (the standard fix real SLAM stacks use
for exactly this degenerate case, since a homography is well-conditioned under
pure rotation) recovers a plausible *final* number (1.24°) but is still noisy
frame-to-frame (median 37° error) — it has large excursions mid-trajectory
that happen to cancel out by t=30s. For a live drawing app you need the
per-frame estimate to be stable, not just correct in hindsight, so this
doesn't actually fix the usability problem.

There's also a fundamental limitation this benchmark doesn't even get to:
monocular VO's translation is scale-ambiguous without an IMU or a known
reference length. Even a perfect rotation estimate wouldn't give you metric
position drift for the translational component of hand/camera motion — you'd
still need an external scale reference, which a single RGB webcam can't
provide. `pyslam` and similar full-SLAM stacks solve this with loop closure,
bundle adjustment, and (usually) stereo/RGB-D/IMU input, none of which apply
to a single laptop webcam; that dependency and complexity weren't prototyped
here for that reason — the degenerate-motion result above already answers the
core question without needing to.

## Recommendation

**Don't switch.** For this motion regime (webcam pan, near-zero translation
baseline), essential-matrix VO is measurably *worse* than a naive optical-flow
accumulator — about 25x higher RMS error, 20x higher per-frame cost, and it
fails even with perfect correspondences. The homography variant is better but
still noisier than naive, moment-to-moment, and adds real implementation
complexity (RANSAC homography fit, 4-way decomposition ambiguity, solution
disambiguation) for no measured benefit. If `motion.py` ever needs actual
camera-pan compensation, a naive 2D-flow accumulator is not just simpler, it's
the better-performing choice given this hardware constraint.
