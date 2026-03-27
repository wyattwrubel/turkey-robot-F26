import os
import time
import cv2
from config import CONFIG


class USBCamera:
    def __init__(self):
        self.device_index = CONFIG.camera.device_index
        self.width = CONFIG.camera.width
        self.height = CONFIG.camera.height
        self.fps = CONFIG.camera.fps
        self.fourcc = CONFIG.camera.fourcc
        self.backend = CONFIG.camera.backend
        self.cap = None

    def _create_capture(self):
        if self.backend == "dshow":
            return cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        elif self.backend == "msmf":
            return cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
        elif self.backend == "v4l2":
            return cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
        else:
            return cv2.VideoCapture(self.device_index)

    def open(self) -> None:
        self.cap = self._create_capture()

        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera] Failed to open camera index {self.device_index}")

        # 方案 A：先设 FOURCC，再设分辨率/FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        time.sleep(0.3)

        actual = self.get_actual_settings()

        # 如果 FOURCC 还不是目标格式，再试一次不同顺序
        if actual["fourcc"] != self.fourcc:
            self.release()
            self.cap = self._create_capture()

            if not self.cap.isOpened():
                raise RuntimeError(f"[Camera] Failed to reopen camera index {self.device_index}")

            # 方案 B：先设分辨率/FPS，最后设 FOURCC
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))

            time.sleep(0.3)

    def read(self):
        if self.cap is None:
            raise RuntimeError("[Camera] Camera has not been opened yet")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_actual_settings(self) -> dict:
        if self.cap is None:
            raise RuntimeError("[Camera] Camera has not been opened yet")

        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int > 0:
            fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        else:
            fourcc_str = "UNKNOWN"

        return {
            "device_index": self.device_index,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc_str,
        }

    def print_actual_settings(self) -> None:
        settings = self.get_actual_settings()
        print("[Camera] Actual settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")