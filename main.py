import cv2

from airdraw.capture import Capture
from airdraw.hands import HandTracker
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
    hand_tracker = HandTracker()
    motion_tracker = MotionTracker()
    was_pinching = False
    try:
        while True:
            frame = capture.get_frame()
            if frame is None:
                break

            found, fingertip_px, is_pinching = hand_tracker.update(frame)

            if found and fingertip_px is not None:
                color = (0, 0, 255) if is_pinching else (0, 255, 0)
                cv2.circle(frame, fingertip_px, 10, color, -1)

            if is_pinching != was_pinching:
                print("PINCH start" if is_pinching else "PINCH end")
                was_pinching = is_pinching

            dx, dy = motion_tracker.update(frame)
            draw_motion_overlay(frame, dx, dy)

            cv2.imshow("AirDraw", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        hand_tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
