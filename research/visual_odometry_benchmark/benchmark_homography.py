import numpy as np
import cv2

from sim import K, FX, camera_trajectory, make_scene, render_frame, N_FRAMES
from benchmark import FEATURE_PARAMS, LK_PARAMS, MIN_TRACKED, yaw_from_R


def run():
    scene = make_scene()
    poses = camera_trajectory(N_FRAMES)
    frames = [render_frame(scene, R, C) for (R, C, _) in poses]
    gt_yaw = np.array([p[2] for p in poses])

    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, mask=None, **FEATURE_PARAMS)

    homog_yaw = np.zeros(N_FRAMES)
    R_global = np.eye(3)

    for i in range(1, N_FRAMES):
        gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        if pts_prev is None or len(pts_prev) < MIN_TRACKED:
            pts_prev = cv2.goodFeaturesToTrack(gray_prev, mask=None, **FEATURE_PARAMS)
            if pts_prev is None:
                homog_yaw[i] = homog_yaw[i - 1]
                gray_prev = gray_curr
                continue

        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts_prev, None, **LK_PARAMS)
        status = status.reshape(-1).astype(bool)
        good_prev = pts_prev[status]
        good_curr = pts_curr[status]

        if len(good_prev) < 8:
            homog_yaw[i] = homog_yaw[i - 1]
            pts_prev = cv2.goodFeaturesToTrack(gray_curr, mask=None, **FEATURE_PARAMS)
            gray_prev = gray_curr
            continue

        H, mask_h = cv2.findHomography(good_prev.reshape(-1, 2), good_curr.reshape(-1, 2), cv2.RANSAC, 2.0)
        if H is not None:
            n_sols, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
            # pure-rotation-dominant motion -> pick the solution with smallest translation
            best = min(range(n_sols), key=lambda k: np.linalg.norm(Ts[k]))
            R_incr = Rs[best]
            R_global = R_incr @ R_global
        homog_yaw[i] = yaw_from_R(R_global)

        pts_prev = good_curr
        gray_prev = gray_curr

    return gt_yaw, homog_yaw


if __name__ == "__main__":
    gt_yaw, homog_yaw = run()
    err = homog_yaw - gt_yaw
    print("=== Homography-decomposition VO (rotation-only model) ===")
    print(f"  final yaw (should return to 0): {homog_yaw[-1]:+.2f} deg")
    print(f"  final drift error:              {err[-1]:+.2f} deg")
    print(f"  RMS error over trajectory:      {np.sqrt(np.mean(err**2)):.2f} deg")
    print(f"  max |error|:                    {np.max(np.abs(err)):.2f} deg")
