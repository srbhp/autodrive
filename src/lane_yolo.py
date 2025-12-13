import cv2
import argparse
from ultralytics import YOLO
import logging


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Minimal lane departure demo")
    parser.add_argument(
        "--source", default="0", help="Video source (camera index or file). Default 0"
    )
    parser.add_argument(
        "--weights", default="./saved_model/yolo-seg-roadlanes.pt", help="YOLO Weights"
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # If your model expects RGB, convert. If it accepts BGR, skip conversion.
        results = model.track(
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 640)),
            persist=True,
            conf=0.3,  # Lower confidence threshold
            iou=0.6,  # Adjust IoU threshold for NMS
            max_det=3,  # Increase max detections
        )

        plotted = None
        for result in results:
            plotted = result.plot(color_mode="instance", boxes=False)

        # If result.plot returned None, fallback to original frame
        if plotted is None:
            plotted_bgr = frame
        else:
            # If plotted is RGB, convert to BGR for OpenCV display
            if plotted.ndim == 3 and plotted.shape[2] == 3:
                plotted_bgr = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
            else:
                # grayscale or other format: convert to BGR
                plotted_bgr = cv2.cvtColor(plotted, cv2.COLOR_GRAY2BGR)

        cv2.imshow("YOLO Stream", plotted_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
