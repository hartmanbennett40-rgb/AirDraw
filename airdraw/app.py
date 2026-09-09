import os
import time
from datetime import datetime

import cv2

from airdraw.canvas import Canvas
from airdraw.capture import Capture

COLORS = [
    (0, 0, 255),    # 1 red
    (0, 255, 0),    # 2 green
    (255, 0, 0),    # 3 blue
    (0, 255, 255),  # 4 yellow
    (255, 0, 255),  # 5 magenta
]

INSTRUCTIONS = [
    "1-5: color  |  c: clear  |  z: undo  |  s: save  |  q/ESC: quit",
]


class App:
    # Wires capture, hand tracking, motion, and canvas into the main loop.
    def __init__(self, device_index=0, save_dir="."):
        self.capture = Capture(device_index)
        self.canvas = None
        self.color_index = 0
        self.save_dir = save_dir
        self.drawing = False

    def _on_mouse(self, event, x, y, flags, param):
        # Stand-in input source: hand/motion tracking isn't implemented yet,
        # so drawing is driven by the mouse until that lands.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.canvas.start_stroke((x, y), COLORS[self.color_index])
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.canvas.extend_stroke((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.canvas.end_stroke()

    def _draw_ui(self, frame):
        h, w = frame.shape[:2]
        for i, line in enumerate(INSTRUCTIONS):
            cv2.putText(frame, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

        swatch_size = 28
        margin = 10
        start_x = w - (swatch_size + margin) * len(COLORS) - margin
        y = margin
        for idx, color in enumerate(COLORS):
            x = start_x + idx * (swatch_size + margin)
            cv2.rectangle(frame, (x, y), (x + swatch_size, y + swatch_size), color, -1)
            border = (255, 255, 255) if idx == self.color_index else (0, 0, 0)
            thickness = 3 if idx == self.color_index else 1
            cv2.rectangle(frame, (x, y), (x + swatch_size, y + swatch_size), border, thickness)
            cv2.putText(frame, str(idx + 1), (x + 8, y + swatch_size + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _save_canvas(self):
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, f"airdraw_{datetime.now():%Y%m%d_%H%M%S}.png")
        cv2.imwrite(filename, self.canvas.render_blank())
        return filename

    def run(self):
        window = "AirDraw"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._on_mouse)

        try:
            while True:
                frame = self.capture.get_frame()
                if frame is None:
                    break

                if self.canvas is None:
                    h, w = frame.shape[:2]
                    self.canvas = Canvas(w, h)

                self.canvas.render(frame)
                self._draw_ui(frame)
                cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('c'):
                    self.canvas.clear()
                elif key == ord('z'):
                    self.canvas.undo_last()
                elif key == ord('s'):
                    self._save_canvas()
                elif ord('1') <= key <= ord('5'):
                    self.color_index = key - ord('1')
        finally:
            self.capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    App().run()
