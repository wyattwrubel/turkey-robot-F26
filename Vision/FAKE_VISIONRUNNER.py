#!/usr/bin/env python3
"""
test_vision.py  —  Live detection test for the full vision pipeline.
Press Q or ESC to quit.

Run from the Vision directory:
    python3 test_vision.py
"""

import sys
import os
import time
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera import USBCamera
from model import YOLODetector
from processing import TemporalManager, PositionEstimator
from output import CoordinateOutput

# ── Colours per class ────────────────────────────────────────────────
COLOURS = {
    "person":  (0, 255, 0),    # green
    "turkey":  (0, 165, 255),  # orange
    "vehicle": (255, 0, 0),    # blue
}
DEFAULT_COLOUR = (200, 200, 200)

def draw(frame, outputs):
    h, w = frame.shape[:2]
    for out in outputs:
        x1, y1, x2, y2 = out.bbox
        if x2 <= 0 or x1 >= w or y2 <= 0 or y1 >= h:
            continue
        colour = COLOURS.get(out.class_name, DEFAULT_COLOUR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = (f"{out.class_name} {out.confidence:.2f} | "
                 f"x={out.x_m:.1f}m y={out.y_m:.1f}m "
                 f"ang={out.angle_deg:.1f}°")
        cv2.putText(frame, label, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    return frame


def main():
    print("=== Vision pipeline test ===")
    print("Initialising camera ...")
    cam = USBCamera()
    cam.open()
    cam.print_actual_settings()

    print("Loading detector ...")
    detector = YOLODetector()

    # Match the settings used in VisionObject
    manager   = TemporalManager(init_window=10, init_min_hits=7, delete_missing_frames=30)
    estimator = PositionEstimator()
    outputter = CoordinateOutput()

    print("\nRunning — press Q or ESC in the window to quit.\n")

    frame_times = []

    while True:
        t0 = time.time()

        ret, frame = cam.read()
        if not ret or frame is None:
            print("[WARN] Camera read failed, retrying ...")
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # ── Full pipeline ─────────────────────────────────────────────
        detections    = detector.detect(frame)
        confirmed     = manager.update(detections)
        positions     = estimator.estimate_all(confirmed, w, h)
        coord_outputs = outputter.convert_all(positions)

        visible = [
            o for o in coord_outputs
            if not (o.bbox[2] <= 0 or o.bbox[0] >= w or
                    o.bbox[3] <= 0 or o.bbox[1] >= h)
        ]

        # ── Terminal output ───────────────────────────────────────────
        if visible:
            for o in visible:
                print(f"  [{o.class_name:7s}] conf={o.confidence:.2f}  "
                      f"bbox={o.bbox}  x={o.x_m:.2f}m  y={o.y_m:.2f}m  "
                      f"angle={o.angle_deg:.1f}°")
        else:
            # Raw detections even if not yet temporally confirmed
            if detections:
                names = [f"{d.class_name}({d.confidence:.2f})" for d in detections]
                print(f"  [unconfirmed] {', '.join(names)}")
            else:
                print("  no detections")

        # ── FPS ───────────────────────────────────────────────────────
        t1 = time.time()
        frame_times.append(t1 - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        # ── Draw & show ───────────────────────────────────────────────
        vis = draw(frame.copy(), visible)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(vis, f"confirmed={len(visible)}  raw={len(detections)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Vision Test", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()