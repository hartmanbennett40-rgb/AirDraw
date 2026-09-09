import cv2

from airdraw.capture import Capture
from airdraw.motion import MotionTracker


def draw_motion_overlay(frame, dx, dy, scale=15):
    # Displacement is usually only a few pixels/frame, so it's amplified
    # here to stay visible. This is a debug aid, not a to-scale measurement.
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    tip = (int(center[0] + dx * scale), int(center[1] + dy * scale))
    cv2.arrowedLine(frame, center, tip, (0, 0, 255), 2, tipLength=0.3)


def main():
    capture = Capture()
    tracker = MotionTracker()
    try:
        while True:
            frame = capture.get_frame()
            if frame is None:
                break
            dx, dy = tracker.update(frame)
            draw_motion_overlay(frame, dx, dy)
            cv2.imshow("AirDraw", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
