"""
Vision/Detector.py: run the detection model on the robot (no laptop).

    det = Detector()                 # backend from Vision/config.py
    detections, info = det(frame)    # frame: BGR uint8 numpy array, kept in memory only
    det.close()

Backends:
  "hailo": the Hailo-8L. Needs the system Python on the Pi (/usr/bin/python3),
           which has hailo_platform 4.23. Opens the device once per process.
  "pt":    Ultralytics on CPU. For laptop dev on recorded video, and for scoring
           W26's baseline (turkey_1280.pt at 640) offline.

Threading: the Hailo pipeline must be created and used from the same thread.
VisionObject creates the Detector inside its single detection worker thread
and only calls it from there.
"""

import contextlib
import time
from dataclasses import dataclass

import cv2
import numpy as np

from Vision import config
from Vision.yolo_decode import decode, decode_split, guess_layout, letterbox, unletterbox


@dataclass
class Detection:
    cls: int              # model class id
    name: str             # "turkey" | "person" | other
    conf: float
    box: tuple            # x1, y1, x2, y2 in image pixels


class _HailoBackend:
    def __init__(self, hef_path, conf):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams,
                                    InputVStreamParams, OutputVStreamParams, FormatType,
                                    InferVStreams)
        self.conf = conf
        self.hef = HEF(str(hef_path))
        self._stack = contextlib.ExitStack()
        self.device = self._stack.enter_context(VDevice())
        cfg = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.device.configure(self.hef, cfg)[0]
        inp = self.hef.get_input_vstream_infos()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = inp.shape[0], inp.shape[1]
        outs = self.hef.get_output_vstream_infos()
        self.output_names = [o.name for o in outs]
        # several outputs = split heads (box + class per stride); one output = raw/e2e/nms
        self.split = len(outs) > 1
        ip = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32)
        self._stack.enter_context(self.ng.activate(self.ng.create_params()))
        self.pipe = self._stack.enter_context(InferVStreams(self.ng, ip, op))
        self.layout = None

    def __call__(self, bgr):
        lb, r, pad = letterbox(bgr, (self.in_h, self.in_w))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        res = self.pipe.infer({self.input_name: rgb[np.newaxis]})
        if self.split:
            self.layout = "split"
            boxes, scores, classes = decode_split(res.values(), (self.in_w, self.in_h), self.conf)
        else:
            out = res[self.output_names[0]]
            if self.layout is None:
                self.layout = guess_layout(out)
            boxes, scores, classes = decode(out, self.layout, (self.in_w, self.in_h), self.conf)
        boxes = unletterbox(boxes, r, pad, (bgr.shape[1], bgr.shape[0]))
        return boxes, scores, classes

    def close(self):
        self._stack.close()


class _PtBackend:
    def __init__(self, pt_path, imgsz, conf):
        from ultralytics import YOLO
        self.model, self.imgsz, self.conf = YOLO(str(pt_path)), imgsz, conf

    def __call__(self, bgr):
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)

    def close(self):
        pass


class Detector:
    def __init__(self, backend=None, hef_path=None, pt_path=None, classes=None):
        self.backend_name = backend or config.BACKEND
        cs = config.CLASS_SETS[classes or config.MODEL_CLASSES]
        self.names, self.turkey_ids, self.person_ids = cs["names"], cs["turkey"], cs["person"]
        if self.backend_name == "hailo":
            self._b = _HailoBackend(hef_path or config.HEF_PATH, config.DECODE_CONF)
        elif self.backend_name == "pt":
            self._b = _PtBackend(pt_path or config.PT_PATH, config.PT_IMGSZ, config.DECODE_CONF)
        else:
            raise ValueError(f"Detector backend must be 'hailo' or 'pt', got {self.backend_name!r}")
        self._t_last = None
        self.fps = 0.0

    def __call__(self, bgr):
        """Returns (list[Detection], {"infer_ms", "fps", "backend"}). Keeps only turkey and
        person classes that pass their per-class threshold."""
        t0 = time.perf_counter()
        boxes, scores, classes = self._b(bgr)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        out = []
        for box, s, c in zip(boxes, scores, classes):
            c = int(c)
            if c in self.turkey_ids:
                name = "turkey"
            elif c in self.person_ids:
                name = "person"
            else:
                continue
            if s < config.CONF_THRESH[name]:
                continue
            out.append(Detection(c, name, float(s), tuple(float(v) for v in box)))

        now = time.perf_counter()
        if self._t_last is not None:
            inst = 1.0 / max(now - self._t_last, 1e-6)
            self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst
        self._t_last = now
        return out, {"infer_ms": infer_ms, "fps": self.fps, "backend": self.backend_name}

    def close(self):
        self._b.close()
