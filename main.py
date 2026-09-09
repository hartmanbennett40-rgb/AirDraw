import cv2

from airdraw.capture import Capture
from airdraw.hands import HandTracker


def main():
    capture = Capture()
    tracker = HandTracker()
    was_pinching = False
    try:
        while True:
            frame = capture.get_frame()
            if frame is None:
                break

            found, fingertip_px, is_pinching = tracker.update(frame)

            if found and fingertip_px is not None:
                color = (0, 0, 255) if is_pinching else (0, 255, 0)
                cv2.circle(frame, fingertip_px, 10, color, -1)

            if is_pinching != was_pinching:
                print("PINCH start" if is_pinching else "PINCH end")
                was_pinching = is_pinching

            cv2.imshow("AirDraw", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
