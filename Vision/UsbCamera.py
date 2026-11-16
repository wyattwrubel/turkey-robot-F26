import time
import cv2


# ── Camera settings: from Vision/config.py (one place, contract C9) ───────────
from Vision import config as _vcfg
DEVICE_PATH  = _vcfg.CAMERA_DEVICE
WIDTH        = _vcfg.CAMERA_WIDTH
HEIGHT       = _vcfg.CAMERA_HEIGHT
FPS          = _vcfg.CAMERA_FPS
FOURCC       = _vcfg.CAMERA_FOURCC   # MJPG recommended for USB2.0
BACKEND      = "v4l2"   # "v4l2" on Pi, "dshow" on Windows, "auto" for either
# ─────────────────────────────────────────────────────────────────────────────


class USBCamera:
    def __init__(self):
        self.device_path  = DEVICE_PATH
        self.width        = WIDTH
        self.height       = HEIGHT
        self.fps          = FPS
        self.fourcc       = FOURCC
        self.backend      = BACKEND
        self.cap          = None

    def _create_capture(self):
        if self.backend == "dshow":
            return cv2.VideoCapture(self.device_path, cv2.CAP_DSHOW)
        elif self.backend == "msmf":
            return cv2.VideoCapture(self.device_path, cv2.CAP_MSMF)
        elif self.backend == "v4l2":
            return cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        else:
            return cv2.VideoCapture(self.device_path)

    def open(self) -> None:
        self.cap = self._create_capture()
        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera] Failed to open camera at {self.device_path}")

        # Attempt A: set FOURCC first, then resolution/FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        time.sleep(0.3)

        # Attempt B: if FOURCC didn't take, try reversed order
        if self.get_actual_settings()["fourcc"] != self.fourcc:
            self.release()
            self.cap = self._create_capture()
            if not self.cap.isOpened():
                raise RuntimeError(f"[Camera] Failed to reopen camera at {self.device_path}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
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
        fourcc_str = (
            "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
            if fourcc_int > 0 else "UNKNOWN"
        )
        return {
            "device_path": self.device_path,
            "width":  int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":    self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": fourcc_str,
        }

    def print_actual_settings(self) -> None:
        print("[Camera] Actual settings:")
        for key, value in self.get_actual_settings().items():
            print(f"  {key}: {value}")