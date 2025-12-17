#!/usr/bin/env python3
"""
Minimal lane-departure demo using OpenCV (training-free) with an optional
pre-trained segmentation model (PyTorch/torchvision) to mask the road area.

Usage:
    python src/lane_demo.py --source 0           # webcam
    python src/lane_demo.py --source video.mp4   # video file

"""

from __future__ import annotations

import argparse
import logging
import time

import cv2

from ui import road_lanes


def main():
    parser = argparse.ArgumentParser(description="Minimal lane departure demo")
    parser.add_argument(
        "--source", default="0", help="Video source (camera index or file). Default 0"
    )
    parser.add_argument(
        "--save", default=None, help="Optional output path to save video"
    )
    parser.add_argument("--display", default=True, help="Show GUI windows")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logging.error("Cannot open source %s", args.source)
        return

    writer = None
    if args.save:
        # Prepare VideoWriter later when we know frame size
        pass

    prev_time = time.time()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result = road_lanes.lane_finding_pipeline(frame)

        if result is None:
            result = frame

        # Display frame rate
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv2.putText(
            result,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        if args.display:
            cv2.imshow("AutoDrive  ", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save:
            h, w = result.shape[:2]
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, 25.0, (w, h))
            writer.write(result)

    cap.release()
    if writer:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
