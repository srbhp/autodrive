import argparse
import logging

import cv2
from ultralytics import YOLO

# separate lane departure logic
from lane_departure import LaneDepartureMonitor


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Minimal lane departure demo")
    parser.add_argument(
        "--source", default="0", help="Video source (camera index or file). Default 0"
    )
    parser.add_argument(
        "--weights", default="./saved_model/yolo-seg-roadlanes.pt", help="YOLO Weights"
    )
    parser.add_argument(
        "--offset-threshold",
        default=0.4,
        type=float,
        help="Lateral offset in meters to trigger lane departure (default 0.4m)",
    )
    parser.add_argument(
        "--frames-to-trigger",
        default=3,
        type=int,
        help="Number of consecutive frames over threshold to trigger warning",
    )
    parser.add_argument(
        "--beep",
        action="store_true",
        help="Play a terminal beep when lane departure is triggered",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=True,
        help="Show GUI windows (default True)",
    )

    args = parser.parse_args()
    src = args.source
    if src.isdigit():
        src = int(src)
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logging.error("Cannot open source %s", args.source)
        return

    cap = cv2.VideoCapture(src)

    # lane departure monitor
    monitor = LaneDepartureMonitor(
        offset_threshold=args.offset_threshold,
        frames_to_trigger=args.frames_to_trigger,
        beep=args.beep,
        model_imgsz=640,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # If your model expects RGB, convert. If it accepts BGR, skip conversion.
        results = model.track(
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640)),
            persist=True,
            show=False,
            conf=0.3,  # Lower confidence threshold
            iou=0.6,  # Adjust IoU threshold for NMS
            max_det=3,  # Increase max detections
        )
        # Use the LaneDepartureMonitor to process and annotate
        annotated, offset, curvature, departure_active = monitor.process_frame(
            frame, results
        )

        if args.display:
            cv2.imshow("YOLO Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
