import time
import numpy as np
import cv2

from sim import K, FX, camera_trajectory, make_scene, render_frame, N_FRAMES

MAX_CORNERS = 250
MIN_TRACKED = 80
FEATURE_PARAMS = dict(maxCorners=MAX_CORNERS, qualityLevel=0.01, minDistance=8, blockSize=7)
LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def yaw_from_R(R):
    return np.degrees(np.arctan2(R[0, 2], R[2, 2]))


def run():
    scene = make_scene()
    poses = camera_trajectory(N_FRAMES)

    frames = [render_frame(scene, R, C) for (R, C, _) in poses]
    gt_yaw = np.array([p[2] for p in poses])

    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, mask=None, **FEATURE_PARAMS)

    naive_yaw = np.zeros(N_FRAMES)
    vo_yaw = np.zeros(N_FRAMES)
    R_global = np.eye(3)

    reseed_count = 0
    track_loss_events = 0
    t_naive_total = 0.0
    t_vo_total = 0.0

    for i in range(1, N_FRAMES):
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        if pts_prev is None or len(pts_prev) < MIN_TRACKED:
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, mask=None, **FEATURE_PARAMS)
            reseed_count += 1
            if pts_prev is None:
                naive_yaw[i] = naive_yaw[i - 1]
                vo_yaw[i] = vo_yaw[i - 1]
                gray_prev = gray_curr
                continue

        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts_prev, None, **LK_PARAMS)
        status = status.reshape(-1).astype(bool)
        good_prev = pts_prev[status]
        good_curr = pts_curr[status]

        if len(good_prev) < 8:
            track_loss_events += 1
            naive_yaw[i] = naive_yaw[i - 1]
            vo_yaw[i] = vo_yaw[i - 1]
            pts_prev = cv2.goodFeaturesToTrack(gray_curr, mask=None, **FEATURE_PARAMS)
            gray_prev = gray_curr
            continue

        # --- naive optical-flow approximation: treat mean pixel flow as a
        # pure-rotation small-angle signal, integrate directly ---
        t0 = time.perf_counter()
        dx = np.median(good_curr[:, 0, 0] - good_prev[:, 0, 0])
        dyaw_naive = np.degrees(np.arctan2(-dx, FX))
        naive_yaw[i] = naive_yaw[i - 1] + dyaw_naive
        t_naive_total += time.perf_counter() - t0

        # --- proper VO: essential matrix + recoverPose with calibrated K ---
        t0 = time.perf_counter()
        E, mask_e = cv2.findEssentialMat(
            good_prev.reshape(-1, 2), good_curr.reshape(-1, 2), K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if E is not None and E.shape == (3, 3):
            _, R_incr, t_incr, mask_p = cv2.recoverPose(
                E, good_prev.reshape(-1, 2), good_curr.reshape(-1, 2), K
            )
            R_global = R_incr @ R_global
        vo_yaw[i] = yaw_from_R(R_global)
        t_vo_total += time.perf_counter() - t0

        pts_prev = good_curr
        gray_prev = gray_curr

    return gt_yaw, naive_yaw, vo_yaw, {
        "reseed_count": reseed_count,
        "track_loss_events": track_loss_events,
        "t_naive_ms_per_frame": 1000 * t_naive_total / N_FRAMES,
        "t_vo_ms_per_frame": 1000 * t_vo_total / N_FRAMES,
    }


if __name__ == "__main__":
    gt_yaw, naive_yaw, vo_yaw, stats = run()

    naive_err = naive_yaw - gt_yaw
    vo_err = vo_yaw - gt_yaw

    print(f"frames: {N_FRAMES}, duration: {N_FRAMES/30:.1f}s")
    print(f"reseed events: {stats['reseed_count']}, track-loss events: {stats['track_loss_events']}")
    print(f"naive est. cost/frame: {stats['t_naive_ms_per_frame']:.4f} ms")
    print(f"VO est. cost/frame:    {stats['t_vo_ms_per_frame']:.4f} ms")
    print()
    print("=== Naive optical-flow accumulation ===")
    print(f"  final yaw (should return to 0): {naive_yaw[-1]:+.2f} deg")
    print(f"  final drift error:              {naive_err[-1]:+.2f} deg")
    print(f"  RMS error over trajectory:      {np.sqrt(np.mean(naive_err**2)):.2f} deg")
    print(f"  max |error|:                    {np.max(np.abs(naive_err)):.2f} deg")
    print()
    print("=== recoverPose (essential matrix VO) ===")
    print(f"  final yaw (should return to 0): {vo_yaw[-1]:+.2f} deg")
    print(f"  final drift error:              {vo_err[-1]:+.2f} deg")
    print(f"  RMS error over trajectory:      {np.sqrt(np.mean(vo_err**2)):.2f} deg")
    print(f"  max |error|:                    {np.max(np.abs(vo_err)):.2f} deg")

    np.savez(
        "results.npz",
        gt_yaw=gt_yaw, naive_yaw=naive_yaw, vo_yaw=vo_yaw,
        naive_err=naive_err, vo_err=vo_err,
    )
