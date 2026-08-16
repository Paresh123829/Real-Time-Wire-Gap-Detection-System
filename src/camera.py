import cv2
import time

class CameraStream:
    """
    Handles live camera acquisition or video simulation.
    Tracks latency (ms) and processing FPS.
    """
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            return self.cap.isOpened()
        except Exception:
            return False

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if ret:
            now = time.time()
            dt = now - self.last_frame_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.last_frame_time = now
            return frame
        return None

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
