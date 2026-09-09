import cv2

from airdraw.capture import Capture


def main():
    capture = Capture()
    try:
        while True:
            frame = capture.get_frame()
            if frame is None:
                break
            cv2.imshow("AirDraw", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
