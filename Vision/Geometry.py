"""
Vision/Geometry.py: pixel box -> bearing and camera-frame position (contract C2).

Bearing sign: + = LEFT of the optical axis (counterclockwise), radians. This is
what the planner already receives: W26's laptop code negated its + = right
degrees (coordinate_output.py: theta = radians(-angle_deg)).

Camera frame handed to VisionObject._to_world (unchanged for DR2):
    x_cam = d * tan(bearing)    (+ = left)
    y_cam = d                   (forward)

Bearing uses the checkerboard calibration when Vision/calib/camera_WxH.json
exists (undistort the box's bottom-center, then atan), else a pinhole model
from config.HFOV_DEG. W26 used a linear map across an assumed 102 deg, which is
up to ~6 deg off at the edges against F5's 2.5 deg.

Range here is the camera-only fallback: d = fy * TURKEY_HEIGHT_M / box_height_px.
It is labeled range_source "camera" and gets replaced by the LiDAR range along
the bearing once the Hokuyo is on the robot.
"""

import json
import math

import numpy as np

from Vision import config


class CameraModel:
    def __init__(self, width=None, height=None, hfov_deg=None, calib_path=None):
        self.w = width or config.CAMERA_WIDTH
        self.h = height or config.CAMERA_HEIGHT
        path = calib_path or config.CALIB_PATH
        self.K = self.dist = None
        try:
            with open(path) as f:
                c = json.load(f)
            self.fx, self.fy, self.cx, self.cy = c["fx"], c["fy"], c["cx"], c["cy"]
            self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)
            self.dist = np.array(c.get("dist", [0, 0, 0, 0, 0]), float)
            self.source = f"calibration {path}"
        except (OSError, KeyError, ValueError):
            hfov = math.radians(hfov_deg or config.HFOV_DEG)
            self.fx = self.fy = (self.w / 2.0) / math.tan(hfov / 2.0)
            self.cx, self.cy = self.w / 2.0, self.h / 2.0
            self.source = f"pinhole from HFOV {math.degrees(hfov):.1f} deg (not calibrated)"

    def _undistort(self, u, v):
        if self.K is None:
            return u, v
        import cv2
        p = cv2.undistortPoints(np.array([[[u, v]]], float), self.K, self.dist, P=self.K)
        return float(p[0, 0, 0]), float(p[0, 0, 1])

    def bearing(self, u, v=None):
        """Bearing in radians of image column u (+ = left)."""
        v = self.cy if v is None else v
        u, _ = self._undistort(u, v)
        return math.atan2(self.cx - u, self.fx)

    def range_from_height(self, box_h_px, height_m=None):
        h = height_m or config.TURKEY_HEIGHT_M
        return self.fy * h / max(box_h_px, 1.0)

    def locate(self, box):
        """box x1, y1, x2, y2 -> (x_cam, y_cam, bearing_rad, range_m)."""
        x1, y1, x2, y2 = box
        b = self.bearing((x1 + x2) / 2.0, y2)
        d = self.range_from_height(y2 - y1)
        return d * math.tan(b), d, b, d
