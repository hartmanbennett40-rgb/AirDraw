import cv2
import numpy as np


class Canvas:
    # Holds a persistent drawing surface as a list of strokes so undo/clear are cheap.
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.strokes = []  # list of (points, color, thickness)
        self._current = None

    def start_stroke(self, point, color, thickness=4):
        self._current = [list(point)], color, thickness
        self.strokes.append(self._current)

    def extend_stroke(self, point):
        if self._current is not None:
            self._current[0].append(list(point))

    def end_stroke(self):
        self._current = None

    def undo_last(self):
        if self.strokes:
            self.strokes.pop()
        self._current = None

    def clear(self):
        self.strokes = []
        self._current = None

    def render(self, frame):
        for points, color, thickness in self.strokes:
            for i in range(1, len(points)):
                cv2.line(frame, tuple(points[i - 1]), tuple(points[i]), color, thickness, cv2.LINE_AA)
        return frame

    def render_blank(self):
        img = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        return self.render(img)
