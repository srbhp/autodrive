#!/usr/bin/env python3
"""
Minimal lane-departure demo using OpenCV (training-free) with an optional
pre-trained segmentation model (PyTorch/torchvision) to mask the road area.

Usage:
    python src/lane_demo.py --source 0            # webcam
    python src/lane_demo.py --source video.mp4   # video file

"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from typing import Optional

import cv2
import numpy as np

from ui import road_lanes


def try_load_yolo_model(weights_path: str = "yolov8n.pt", imgsz: int = 640):
    """Try to load a YOLOv8 model (ultralytics). Returns a callable that
    takes a BGR frame (HxWx3) and returns a list of detections each as
    (x1, y1, x2, y2, conf, cls). If ultralytics isn't installed or loading
    fails, returns None.
    """
    try:
        import torch
        from ultralytics import YOLO

        model = YOLO(weights_path)
        # Move to CUDA if available and use half precision for faster inference
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model.to(device)
            # fuse conv/bn for speed if available
            try:
                model.model.fuse()
            except Exception:
                pass
        except Exception:
            pass
        try:
            if device == "cuda":
                # enable half precision if underlying model supports it
                model.model.half()
        except Exception:
            pass

        # tracking timings
        timings = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

        def predict(frame: np.ndarray):
            # ultralytics expects BGR images as numpy arrays; it returns
            # a Results object. We extract boxes as numpy array.
            import time as _time

            t0 = _time.perf_counter()
            # minimal preprocessing is done by the model itself; record preprocessing time as zero
            timings["preprocess"] = _time.perf_counter() - t0
            t1 = _time.perf_counter()
            try:
                res = model(frame, imgsz=imgsz)[0]
            except Exception:
                res = model(frame)[0]
            timings["inference"] = _time.perf_counter() - t1
            if hasattr(res, "boxes") and len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()  # Nx4
                conf = res.boxes.conf.cpu().numpy()  # N
                cls = res.boxes.cls.cpu().numpy().astype(int)  # N
                detections = [
                    (int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(c), int(cl))
                    for b, c, cl in zip(boxes, conf, cls)
                ]
            else:
                detections = []
            timings["postprocess"] = (
                _time.perf_counter() - (t1 + timings["inference"])
                if timings["inference"]
                else 0
            )
            # attach latest timings for diagnostics
            predict.last_timings = timings.copy()
            return detections

        logging.info("Loaded YOLO model from %s (device=%s)", weights_path, device)
        return predict
    except Exception as exc:  # pylint: disable=broad-except
        logging.debug("YOLO model unavailable: %s", exc)
        return None


def get_birdseye_transform(
    image: np.ndarray, dst_width: int = 640, dst_height: int = 480
):
    """Compute perspective transform matrix from image to bird's-eye view.
    Default trapezoid is based on image size. Returns (M, Minv, dst_size).
    """
    h, w = image.shape[:2]
    # source points (trapezoid) — tweak these for your camera and mounting
    src = np.float32(
        [
            [w * 0.1, h * 0.95],  # bottom-left
            [w * 0.45, h * 0.6],  # top-left
            [w * 0.55, h * 0.6],  # top-right
            [w * 0.95, h * 0.95],  # bottom-right
        ]
    )
    dst = np.float32(
        [
            [0, dst_height],
            [0, 0],
            [dst_width, 0],
            [dst_width, dst_height],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, (dst_width, dst_height)


def warp_points(points: np.ndarray, M: np.ndarray):
    """Warp Nx2 points using perspective matrix M and return warped Nx2 points.
    Points must be an array of shape (N, 2) of floats.
    """
    if len(points) == 0:
        return points
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    warped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    return warped


def fit_lane_polynomial(points):
    """Given points in bird's-eye coords (x,y), separate left/right and
    fit x(y) polynomials for each side. Returns (left_poly, right_poly,
    left_pts, right_pts) where poly are np.poly1d or None.
    """
    if points is None or len(points) == 0:
        return None, None, [], []
    # points are Nx2: x,y in BEV
    # Split into left/right by x position relative to image center
    x_mid = np.median(points[:, 0])
    left = points[points[:, 0] < x_mid]
    right = points[points[:, 0] >= x_mid]
    left_poly = None
    right_poly = None
    if len(left) >= 3 and np.std(left[:, 1]) > 1e-6:
        # fit x = Ay^2 + By + C
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=np.RankWarning)
                left_poly = np.poly1d(np.polyfit(left[:, 1], left[:, 0], 2))
        except Exception as ex:
            logging.debug("left quadratic fit failed (%s), falling back to linear", ex)
            # fallback to linear fit
            left_poly = np.poly1d(np.polyfit(left[:, 1], left[:, 0], 1))
    elif len(left) >= 2 and np.std(left[:, 1]) > 1e-6:
        left_poly = np.poly1d(np.polyfit(left[:, 1], left[:, 0], 1))
    if len(right) >= 3 and np.std(right[:, 1]) > 1e-6:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=np.RankWarning)
                right_poly = np.poly1d(np.polyfit(right[:, 1], right[:, 0], 2))
        except Exception as ex:
            logging.debug("right quadratic fit failed (%s), falling back to linear", ex)
            right_poly = np.poly1d(np.polyfit(right[:, 1], right[:, 0], 1))
    elif len(right) >= 2 and np.std(right[:, 1]) > 1e-6:
        right_poly = np.poly1d(np.polyfit(right[:, 1], right[:, 0], 1))
    return left_poly, right_poly, left, right


def median_point_filter(points: np.ndarray, num_bins: int = 8):
    """Reduce outliers by binning points along y and taking median x per bin.
    Returns filtered points as Nx2 array sorted by y.
    """
    if points is None or len(points) == 0:
        return points
    ys = points[:, 1]
    bins = np.linspace(ys.min(), ys.max(), num_bins + 1)
    filtered = []
    for i in range(num_bins):
        mask = (
            (ys >= bins[i]) & (ys < bins[i + 1])
            if i < num_bins - 1
            else (ys >= bins[i]) & (ys <= bins[i + 1])
        )
        if not np.any(mask):
            continue
        xs = points[mask][:, 0]
        ys_bin = points[mask][:, 1]
        # use median x and median y for the bin
        mx = np.median(xs)
        my = np.median(ys_bin)
        filtered.append((mx, my))
    if len(filtered) == 0:
        return np.empty((0, 2), dtype=float)
    filtered = np.array(filtered, dtype=float)
    # sort by y
    filtered = filtered[np.argsort(filtered[:, 1])]
    return filtered


class PolynomialSmoother:
    """Simple EMA smoother for polynomial coefficients (degree 2).

    It keeps a current estimate of coefficients [A,B,C] and updates via
    new_coeffs = alpha*new + (1-alpha)*old.
    If a measurement is missing, it will keep the previous value for up to
    `max_missing_frames` frames, then reset.
    """

    def __init__(self, alpha: float = 0.85, max_missing_frames: int = 8):
        self.alpha = float(alpha)
        self.max_missing_frames = int(max_missing_frames)
        self.coeffs = None
        self.missing = 0

    def _to_degree2(self, poly: np.poly1d):
        if poly is None:
            return None
        # Ensure polynomial coefficients are length 3 representing degree 2
        coeffs = np.atleast_1d(poly.c).astype(float)
        if coeffs.shape[0] == 3:
            return coeffs.copy()
        # if linear (degree 1), pad with zero A coeff
        if coeffs.shape[0] == 2:
            return np.array([0.0, coeffs[0], coeffs[1]], dtype=float)
        # if constant or other, pad/truncate to degree2
        if coeffs.shape[0] < 3:
            coeffs = np.concatenate((np.zeros(3 - len(coeffs)), coeffs))
            return coeffs
        return coeffs[-3:]

    def update(self, poly: Optional[np.poly1d]):
        new_coeffs = self._to_degree2(poly)
        if new_coeffs is None:
            # no new measurement
            self.missing += 1
            if self.missing > self.max_missing_frames:
                self.coeffs = None
                return None
            if self.coeffs is None:
                return None
            return np.poly1d(self.coeffs)
        self.missing = 0
        if self.coeffs is None:
            self.coeffs = new_coeffs
        else:
            self.coeffs = self.alpha * new_coeffs + (1.0 - self.alpha) * self.coeffs
        return np.poly1d(self.coeffs)


def compute_curvature_and_offset(left_poly, right_poly, bev_shape):
    """Compute curvature radius (px) and vehicle offset (meters approximated)
    from left and right polynomials (x(y)). Returns (curvature_m, offset_m, lane_width_pixels)
    """
    # If both polynomials exist, compute lane center at bottom y
    h, w = bev_shape[:2]
    y_eval = h  # bottom of BEV
    # estimate lane width in pixels
    if left_poly is None or right_poly is None:
        return None, None, None
    left_x = left_poly(y_eval)
    right_x = right_poly(y_eval)
    lane_width_pixels = abs(right_x - left_x)
    # vehicle center is w/2
    vehicle_center = w / 2.0
    lane_center_x = (left_x + right_x) / 2.0
    offset_pixels = vehicle_center - lane_center_x
    # assume lane width = 3.7 meters
    xm_per_pix = 3.7 / lane_width_pixels if lane_width_pixels != 0 else 1.0
    offset_m = offset_pixels * xm_per_pix

    # curvature: use average of left & right curvature at y_eval
    def radius_of_curvature(poly, y):
        # poly is function x(y) = Ay^2 + By + C => derivative dx/dy = 2Ay + B
        # but curvature for lane lines parametrized as x(y) -> use formula adapted
        A = poly.c[0]
        B = poly.c[1]
        denom = abs(2 * A)
        if denom == 0:
            return float("inf")
        R = (1 + (2 * A * y + B) ** 2) ** 1.5 / denom
        return R

    R_left = radius_of_curvature(left_poly, y_eval)
    R_right = radius_of_curvature(right_poly, y_eval)
    # convert to meters using xm_per_pix factor; curvature px->m: multiply by xm_per_pix
    curvature_m = ((R_left + R_right) / 2.0) * xm_per_pix
    return curvature_m, offset_m, lane_width_pixels


def main():
    parser = argparse.ArgumentParser(description="Minimal lane departure demo")
    parser.add_argument(
        "--source", default="0", help="Video source (camera index or file). Default 0"
    )
    parser.add_argument(
        "--use-yolo",
        action="store_true",
        help="Use YOLOv8n for lane/drivable-edge detection (if weights available)",
    )
    parser.add_argument(
        "--yolo-weights",
        default="yolov8n.pt",
        help="Path to YOLOv8 weights to load (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        default=640,
        type=int,
        help="YOLO inference resolution (square size, e.g., 640 or 384).",
    )
    parser.add_argument(
        "--save", default=None, help="Optional output path to save video"
    )
    parser.add_argument("--display", action="store_true", help="Show GUI windows")
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
        help="Play a terminal beep (\a) when lane departure is triggered",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Attempt to load models if requested
    predict_yolo = None
    if args.use_yolo:
        predict_yolo = try_load_yolo_model(
            weights_path=args.yolo_weights, imgsz=args.yolo_imgsz
        )
        if predict_yolo is None:
            logging.warning(
                "YOLO model unavailable — falling back to OpenCV-only pipeline."
            )

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
    bev_M = None
    bev_Minv = None
    bev_size = None
    # smoothing state for left and right lane polynomials
    left_smoother = PolynomialSmoother(alpha=0.75)
    right_smoother = PolynomialSmoother(alpha=0.75)
    # state for lane departure warnings
    departure_counter = 0
    departure_active = False
    lost_lane_counter = 0
    lost_lane_active = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Optional: apply segmentation to mask out non-road areas
        img_for_pipeline = frame
        result = frame.copy()
        # If YOLO is enabled and present, use it to estimate lane points
        curvature = None
        offset = None
        if predict_yolo is not None:
            try:
                detections = predict_yolo(frame)
                # attach timings for display if available
                if hasattr(predict_yolo, "last_timings"):
                    yolo_times = predict_yolo.last_timings
                else:
                    yolo_times = None
                # detections are (x1,y1,x2,y2,conf,cls)
                points = []
                for x1, y1, x2, y2, conf, cls in detections:
                    # bottom center of bounding box
                    cx = (x1 + x2) / 2.0
                    cy = y2
                    points.append((cx, cy))
                if len(points) > 0:
                    pts = np.array(points, dtype=np.float32)
                    # remove outlier points by binning (y) and taking medians
                    pts = median_point_filter(pts, num_bins=8)
                    # compute BEV transforms if needed
                    if bev_M is None:
                        bev_M, bev_Minv, bev_size = get_birdseye_transform(frame)
                    bev_pts = warp_points(pts, bev_M)
                    left_poly, right_poly, left_pts, right_pts = fit_lane_polynomial(
                        bev_pts
                    )
                    # smooth polynomial coefficients across frames to reduce jitter
                    left_poly = left_smoother.update(left_poly)
                    right_poly = right_smoother.update(right_poly)
                    if left_poly is not None and right_poly is not None:
                        curvature, offset, lane_width_px = compute_curvature_and_offset(
                            left_poly, right_poly, bev_size
                        )
                        # draw lane lines and center on result using inverse warp
                        h_bev = bev_size[1]
                        ys = np.linspace(0, h_bev, num=50)
                        left_xy = np.column_stack((left_poly(ys), ys))
                        right_xy = np.column_stack((right_poly(ys), ys))
                        # warp back to original image
                        left_img = warp_points(left_xy, bev_Minv)
                        right_img = warp_points(right_xy, bev_Minv)
                        # draw
                        for i in range(len(left_img) - 1):
                            x1, y1 = map(int, left_img[i])
                            x2, y2 = map(int, left_img[i + 1])
                            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        for i in range(len(right_img) - 1):
                            x1, y1 = map(int, right_img[i])
                            x2, y2 = map(int, right_img[i + 1])
                            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        # draw lane center
                        mid_img = (left_img + right_img) / 2.0
                        for i in range(len(mid_img) - 1):
                            x1, y1 = map(int, mid_img[i])
                            x2, y2 = map(int, mid_img[i + 1])
                            cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        # Draw vehicle center for reference
                        h, w = result.shape[:2]
                        cv2.circle(
                            result, (int(w / 2), int(h * 0.92)), 6, (255, 255, 0), -1
                        )
            except Exception as e:  # pylint: disable=broad-except
                logging.exception("YOLO detection failed: %s", e)

        # If YOLO was not used or did not find lanes, fallback to classic pipeline
        if predict_yolo is None or (
            predict_yolo is not None and (curvature is None or offset is None)
        ):
            result = road_lanes.lane_finding_pipeline(img_for_pipeline)

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
        if curvature is not None:
            cv2.putText(
                result,
                f"Curv(m): {curvature:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        if offset is not None:
            cv2.putText(
                result,
                f"Offset(m): {offset:.2f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
        # Lane departure warning logic
        if offset is not None:
            if abs(offset) > args.offset_threshold:
                departure_counter += 1
            else:
                # decay counter
                departure_counter = max(0, departure_counter - 1)

            if departure_counter >= args.frames_to_trigger:
                if not departure_active:
                    departure_active = True
                    logging.warning(
                        "Lane departure detected: offset=%.3fm (threshold %.2fm)",
                        offset,
                        args.offset_threshold,
                    )
                    if args.beep:
                        print("\a", end="")
                # overlay warning visuals
                direction = "LEFT" if offset > 0 else "RIGHT"
                cv2.putText(
                    result,
                    f"LANE DEPARTURE: {direction}",
                    (result.shape[1] // 4, result.shape[0] // 4),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )
                # highlight centerline
                for i in range(len(mid_img) - 1):
                    x1, y1 = map(int, mid_img[i])
                    x2, y2 = map(int, mid_img[i + 1])
                    cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 6)
            else:
                if departure_active and departure_counter == 0:
                    departure_active = False
                    logging.info("Lane departure cleared.")
        else:
            # no offset computed; decrement counters and try to mark lost lane
            departure_counter = max(0, departure_counter - 1)
        if "yolo_times" in locals() and yolo_times is not None:
            cv2.putText(
                result,
                f"yolo pre:{yolo_times['preprocess'] * 1000:.1f}ms inf:{yolo_times['inference'] * 1000:.1f}ms post:{yolo_times['postprocess'] * 1000:.1f}ms",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 0),
                2,
            )

        if args.display:
            cv2.imshow("LaneDemo", result)
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
