import cv2


class Capture:
    def __init__(self, device_index=0):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {device_index}")

    def get_frame(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        self._cap.release()
