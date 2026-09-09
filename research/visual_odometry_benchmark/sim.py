import numpy as np
import cv2

W, H = 640, 480
FX = FY = 800.0
CX, CY = W / 2, H / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

FPS = 30
DURATION_S = 30
N_FRAMES = FPS * DURATION_S
YAW_AMPLITUDE_DEG = 25.0
RNG = np.random.default_rng(42)


def make_scene(n_points=250):
    x = RNG.uniform(-4, 4, n_points)
    y = RNG.uniform(-3, 3, n_points)
    z = RNG.uniform(3, 12, n_points)
    return np.stack([x, y, z], axis=1)


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def camera_trajectory(n_frames):
    """Smooth pan-and-return: yaw goes 0 -> +A -> 0, with small handheld
    jitter (pitch wobble + tiny translational sway) layered on top so the
    motion isn't a degenerate pure rotation about the camera center."""
    t = np.linspace(0, 1, n_frames)
    yaw = YAW_AMPLITUDE_DEG * np.sin(np.pi * t)

    jitter_pitch = 1.2 * np.sin(2 * np.pi * 3.7 * t) * np.sin(np.pi * t)
    jitter_roll = 0.6 * np.sin(2 * np.pi * 2.3 * t + 1.0) * np.sin(np.pi * t)

    sway_x = 0.15 * np.sin(2 * np.pi * 1.5 * t) * np.sin(np.pi * t)
    sway_y = 0.05 * np.sin(2 * np.pi * 4.1 * t + 0.5) * np.sin(np.pi * t)

    poses = []
    for i in range(n_frames):
        R = rot_y(yaw[i]) @ rot_x(jitter_pitch[i])
        roll = jitter_roll[i]
        a = np.radians(roll)
        Rroll = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        R = R @ Rroll
        C = np.array([sway_x[i], sway_y[i], 0.0])
        poses.append((R, C, yaw[i]))
    return poses


def render_frame(points_world, R, C, noise_px=0.4):
    Rt = R.T
    pts_cam = (Rt @ (points_world - C).T).T
    z = pts_cam[:, 2]
    valid = z > 0.5
    pts_cam = pts_cam[valid]
    z = z[valid]

    uv = (K @ (pts_cam / z[:, None]).T).T[:, :2]
    uv += RNG.normal(0, noise_px, uv.shape)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    in_bounds = (uv[:, 0] >= 4) & (uv[:, 0] < W - 4) & (uv[:, 1] >= 4) & (uv[:, 1] < H - 4)
    uv_vis = uv[in_bounds]
    for (u, v) in uv_vis:
        cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img
