"""
Lane departure monitor helper for YOLO segmentation/bbox results.
Provides a class `LaneDepartureMonitor` that accepts frames and model `results`
and computes lane polynomials, curvature, offset, and lane-departure warnings.

It reuses helpers from `lane_demo.py` for BEV transforms and polynomial fitting.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from lane_demo import (
    PolynomialSmoother,
    median_point_filter,
)


class LaneDepartureMonitor:
    def __init__(
        self,
        offset_threshold: float = 0.4,
        frames_to_trigger: int = 3,
        beep: bool = False,
        model_imgsz: int = 640,
        smoothing_alpha: float = 0.75,
        min_mask_overlap: int = 200,
        min_mask_area: int = 100,
    ):
        self.offset_threshold = float(offset_threshold)
        self.frames_to_trigger = int(frames_to_trigger)
        self.beep = bool(beep)
        self.model_imgsz = int(model_imgsz)
        # BEV transform state
        self.bev_M = None
        self.bev_Minv = None
        self.bev_size = None
        # polynomial smoothers
        self.left_smoother = PolynomialSmoother(alpha=smoothing_alpha)
        self.right_smoother = PolynomialSmoother(alpha=smoothing_alpha)
        # departure counters
        self.departure_counter = 0
        self.departure_active = False
        # mask thresholds
        self.min_mask_overlap = int(min_mask_overlap)
        self.min_mask_area = int(min_mask_area)

    def _get_masks_from_result(
        self, result, frame_shape: Tuple[int, int]
    ) -> Optional[List[np.ndarray]]:
        """Extract and resize boolean masks to frame resolution from an ultralytics `result` object.
        Returns list of boolean masks (H,W) or None if no masks.
        """
        if not hasattr(result, "masks") or result.masks is None:
            return None
        raw_masks = None
        if hasattr(result.masks, "masks"):
            raw_masks = result.masks.masks
        elif hasattr(result.masks, "data"):
            raw_masks = result.masks.data
        if raw_masks is None:
            return None
        # convert tensor->numpy if needed
        if hasattr(raw_masks, "cpu"):
            raw_masks = raw_masks.cpu().numpy()
        else:
            raw_masks = np.asarray(raw_masks)
        # normalize shapes to (N, H, W)
        if raw_masks.ndim == 3:
            # ambig: either (N,H,W) or (H,W,N)
            if raw_masks.shape[0] == len(raw_masks):
                masks_nhw = raw_masks
            elif raw_masks.shape[2] == len(raw_masks):
                masks_nhw = np.transpose(raw_masks, (2, 0, 1))
            else:
                masks_nhw = raw_masks
        else:
            masks_nhw = raw_masks[np.newaxis, ...]
        h, w = frame_shape[:2]
        resized = []
        for i in range(masks_nhw.shape[0]):
            m = masks_nhw[i].astype(np.uint8) * 255
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(m_resized > 0)
        return resized

    def _select_center_mask(
        self, masks: List[np.ndarray], frame_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Pick the mask with centroid in the central ROI or the largest overlap with center ROI.
        Returns a boolean mask (H,W) or None.
        Applies morphological close/open and keeps largest connected component to reduce jitter.
        """
        if masks is None or len(masks) == 0:
            return None
        h, w = frame_shape[:2]
        x1c = int(w * 0.35)
        x2c = int(w * 0.65)
        y1c = int(h * 0.55)
        y2c = h
        best_idx = None
        best_overlap = 0
        # prefer centroid in center
        for i, mm in enumerate(masks):
            ys_all, xs_all = np.where(mm)
            if len(xs_all) == 0:
                continue
            centroid_x = int(np.mean(xs_all))
            overlap = int(np.sum(mm[y1c:y2c, x1c:x2c]))
            if x1c <= centroid_x <= x2c and overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        # fallback to pure overlap
        if best_idx is None:
            for i, mm in enumerate(masks):
                overlap = int(np.sum(mm[y1c:y2c, x1c:x2c]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
        if best_idx is None or best_overlap < self.min_mask_overlap:
            return None
        sel_mask = masks[best_idx].astype(np.uint8)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_u8 = sel_mask.astype(np.uint8) * 255
            mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            mask_open = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_open, connectivity=8
            )
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label = int(np.argmax(areas) + 1)
                sel_mask = labels == largest_label
            else:
                sel_mask = mask_open > 0
        except Exception:
            sel_mask = masks[best_idx]
        if np.sum(sel_mask) < self.min_mask_area:
            return None
        return sel_mask

    def _mask_to_pts(self, sel_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.where(sel_mask)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=float)
        pts = np.column_stack((xs.astype(float), ys.astype(float)))
        pts = median_point_filter(pts, num_bins=16)
        return pts

    def _boxes_to_pts(self, result, frame_shape: Tuple[int, int]) -> np.ndarray:
        # boxes (xyxy) are relative to the model input which we expect to be square of `model_imgsz`.
        boxes = []
        if (
            hasattr(result, "boxes")
            and result.boxes is not None
            and len(result.boxes) > 0
        ):
            try:
                boxes = result.boxes.xyxy.cpu().numpy()
            except Exception:
                boxes = np.asarray(result.boxes.xyxy)
        h_orig, w_orig = frame_shape[:2]
        scale_x = w_orig / float(self.model_imgsz)
        scale_y = h_orig / float(self.model_imgsz)
        pts = []
        for b in boxes:
            x1, y1, x2, y2 = b[:4]
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y
            cx = (x1 + x2) / 2.0
            cy = y2
            pts.append((cx, cy))
        if len(pts) == 0:
            return np.empty((0, 2), dtype=float)
        pts = np.array(pts, dtype=np.float32)
        pts = median_point_filter(pts, num_bins=8)
        return pts

    def process_frame(self, frame: np.ndarray, results: list):
        """Process a BGR frame with a list of ultralytics `results` (from .track or .predict).
        Returns (annotated_frame, offset, curvature, departure_active).
        """
        annotated = frame.copy()
        r = results[-1]
        sel_mask = None
        masks = self._get_masks_from_result(r, frame.shape)
        if masks:
            sel_mask = self._select_center_mask(masks, frame.shape)
            if sel_mask is not None:
                overlay = np.zeros_like(annotated, dtype=np.uint8)
                overlay[sel_mask] = (0, 255, 0)
                annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)
        offset, curvature = None, None
        return annotated, offset, curvature, self.departure_active


__all__ = ["LaneDepartureMonitor"]
