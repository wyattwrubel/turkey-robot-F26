import time
import cv2

from camera import USBCamera
from model import YOLODetector
from processing import TemporalManager, PositionEstimator
from output import CoordinateOutput


# =========================
# Config switches
# =========================
DEBUG_DRAW = True
DEBUG_PRINT = True


def draw_outputs(frame, outputs):
    """Draw bbox + coordinate info on frame"""
    for out in outputs:
        x1, y1, x2, y2 = out.bbox

        label = (
            f"{out.class_name} | "
            f"x={out.x_m:.2f}, y={out.y_m:.2f}"
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame


def main():
    # =========================
    # Initialize modules
    # =========================
    cam = USBCamera()
    detector = YOLODetector()

    manager = TemporalManager(
        init_window=10,
        init_min_hits=7,
        delete_missing_frames=30,
    )

    estimator = PositionEstimator()
    outputter = CoordinateOutput()

    # =========================
    # Open camera
    # =========================
    cam.open()
    cam.print_actual_settings()

    print("[Main] Starting pipeline... Press 'q' or ESC to quit.")

    prev_time = time.time()

    try:
        while True:
            # =========================
            # Read frame
            # =========================
            ret, frame = cam.read()
            if not ret:
                print("[Main] Failed to read frame")
                break

            h, w = frame.shape[:2]

            # =========================
            # CV Pipeline
            # =========================
            detections = detector.detect(frame)

            confirmed_targets = manager.update(detections)

            position_results = estimator.estimate_all(
                confirmed_targets, w, h
            )

            coord_outputs = outputter.convert_all(position_results)

            # =========================
            # Debug print
            # =========================
            if DEBUG_PRINT:
                outputter.print_all(coord_outputs)

            # =========================
            # Visualization
            # =========================
            vis = frame

            if DEBUG_DRAW:
                vis = draw_outputs(vis, coord_outputs)

                # FPS display
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time

                cv2.putText(
                    vis,
                    f"FPS: {fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                cv2.imshow("CV Pipeline", vis)

            # =========================
            # Exit condition
            # =========================
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("[Main] Exiting...")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()