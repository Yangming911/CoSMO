"""
YOLOObstacleMap: builds a 2D obstacle map from YOLO bounding box detections
and depth back-projection.

For each detected bbox we:
  1. Take the median valid depth within the bounding box.
  2. Back-project the bbox centre to a 2D map cell using the same camera model
     as OneMap (x_world = depth, y_world = -(u - cx)*d/fx, then yaw rotation).
  3. Record (frame_idx, px, py) for that label in a rolling event buffer.

Seed accumulation is bounded by a sliding window of the last `window_size`
frames.  Old events are discarded as new frames arrive, so seeds decay once
an object leaves the robot's recent field of view.

At query time, each label's seeds within the window are dilated by its
semantic radius from _SEMANTIC_SAFETY_RADIUS_CELLS, and the result is
subtracted from the base navigable map.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from eval.semantic_collision import (
    _SEMANTIC_SAFETY_RADIUS_CELLS,
    normalize_semantic_label,
)

# import lazily to avoid circular imports at module level
_YOLOObstacleDebugCollector = None
def _get_debug_cls():
    global _YOLOObstacleDebugCollector
    if _YOLOObstacleDebugCollector is None:
        from mapping.semantic_debug import YOLOObstacleDebugCollector
        _YOLOObstacleDebugCollector = YOLOObstacleDebugCollector
    return _YOLOObstacleDebugCollector


class YOLOObstacleMap:
    """Sliding-window per-episode obstacle map built from YOLO detections.

    Args:
        n_cells:     side length of the square map (cells).
        cell_size:   metres per cell.
        window_size: number of recent frames whose seeds are retained.
                     Older events are discarded.  Default 50.
    """

    def __init__(self, n_cells: int, cell_size: float, window_size: int = 50) -> None:
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.map_center = n_cells // 2
        self.window_size = window_size

        # Rolling event buffer: deque of (frame_idx, label, px, py)
        self._events: Deque[Tuple[int, str, int, int]] = deque()
        self._frame_idx: int = 0

        # Events projected in the most recent update() call — used by OACP calibration
        self.latest_projected: List[Tuple[str, int, int]] = []  # [(label, px, py), ...]

        # Pre-compute dilation kernels per radius
        self._kernels: Dict[int, np.ndarray] = {}
        # Current query label — excluded from obstacle marking
        self._query_label: Optional[str] = None
        # Optional debug collector (set by evaluator)
        self.debug_collector = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._events.clear()
        self._frame_idx = 0

    def set_query_label(self, raw_label: str) -> None:
        """Exclude this label from obstacle seeds (it is the navigation target)."""
        self._query_label = normalize_semantic_label(raw_label)

    # ------------------------------------------------------------------
    def _get_kernel(self, radius: int) -> np.ndarray:
        if radius not in self._kernels:
            if radius <= 0:
                self._kernels[radius] = np.ones((1, 1), np.uint8)
            else:
                d = radius * 2 + 1
                self._kernels[radius] = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (d, d)
                )
        return self._kernels[radius]

    # ------------------------------------------------------------------
    def _prune_old_events(self) -> None:
        """Remove events older than window_size frames."""
        cutoff = self._frame_idx - self.window_size
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

    # ------------------------------------------------------------------
    def update(
        self,
        detections_by_class: Dict[str, List[List[float]]],
        depth: np.ndarray,
        tf: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_h: int,
        image_w: int,
    ) -> None:
        """Project YOLO bbox detections into the rolling event buffer.

        Args:
            detections_by_class: {class_name: [[x1,y1,x2,y2], ...]} pixel coords
            depth: (H, W) depth image in metres
            tf:    4×4 camera-to-episodic transform
            fx, fy, cx, cy: camera intrinsics
            image_h, image_w: image dimensions
        """
        self._frame_idx += 1
        self._prune_old_events()
        self.latest_projected = []  # reset per-frame list

        yaw = float(np.arctan2(tf[1, 0], tf[0, 0]))
        cam_x = float(tf[0, 3] / tf[3, 3])
        cam_y = float(tf[1, 3] / tf[3, 3])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        for class_name, boxes in detections_by_class.items():
            label = normalize_semantic_label(class_name)

            # Skip the current navigation target — must remain reachable
            if label == self._query_label:
                continue

            radius = _SEMANTIC_SAFETY_RADIUS_CELLS.get(label, -1)
            # Always project for debug collection; only add to obstacle events if in dict
            debug_only = radius < 0

            for box in boxes:
                x1, y1, x2, y2 = box
                x1_i = max(0, int(x1))
                y1_i = max(0, int(y1))
                x2_i = min(image_w - 1, int(x2) + 1)
                y2_i = min(image_h - 1, int(y2) + 1)
                if x1_i >= x2_i or y1_i >= y2_i:
                    continue

                # --- Method B: per-pixel back-projection ---
                ys_arr, xs_arr = np.mgrid[y1_i:y2_i, x1_i:x2_i]
                ds_arr = depth[y1_i:y2_i, x1_i:x2_i]

                valid_mask = (ds_arr > 0) & np.isfinite(ds_arr)
                if not valid_mask.any():
                    continue

                med_d = float(np.median(ds_arr[valid_mask]))
                # Keep foreground pixels (≤ median × 1.2) to exclude background walls
                fg_mask = valid_mask & (ds_arr <= med_d * 1.2)
                if not fg_mask.any():
                    fg_mask = valid_mask

                ds_v = ds_arr[fg_mask]
                us_v = xs_arr[fg_mask].astype(np.float32)

                x_world_v = ds_v
                y_world_v = -(us_v - cx) * ds_v / fx
                x_rot_v = x_world_v * cos_yaw - y_world_v * sin_yaw + cam_x
                y_rot_v = x_world_v * sin_yaw + y_world_v * cos_yaw + cam_y

                pxs_v = (x_rot_v / self.cell_size).astype(int) + self.map_center
                pys_v = (y_rot_v / self.cell_size).astype(int) + self.map_center

                in_bounds = (
                    (pxs_v >= 0) & (pxs_v < self.n_cells) &
                    (pys_v >= 0) & (pys_v < self.n_cells)
                )
                pxs_b = pxs_v[in_bounds]
                pys_b = pys_v[in_bounds]
                if len(pxs_b) == 0:
                    continue

                # Deduplicate projected cells
                unique_cells = np.unique(np.stack([pxs_b, pys_b], axis=1), axis=0)
                for cell_px, cell_py in unique_cells:
                    if not debug_only:
                        self._events.append((self._frame_idx, label, int(cell_px), int(cell_py)))
                    self.latest_projected.append((label, int(cell_px), int(cell_py)))
                if self.debug_collector is not None:
                    # Report representative single point (median-depth cell) for debug
                    med_u = float(x1 + x2) / 2.0
                    x_w = med_d
                    y_w = -(med_u - cx) * med_d / fx
                    xr = x_w * cos_yaw - y_w * sin_yaw + cam_x
                    yr = x_w * sin_yaw + y_w * cos_yaw + cam_y
                    px_dbg = int(xr / self.cell_size) + self.map_center
                    py_dbg = int(yr / self.cell_size) + self.map_center
                    if 0 <= px_dbg < self.n_cells and 0 <= py_dbg < self.n_cells:
                        self.debug_collector.record(label, px_dbg, py_dbg)

    # ------------------------------------------------------------------
    def get_obstacle_mask(self) -> np.ndarray:
        """Return dilated obstacle mask (True = blocked) from window seeds."""
        # Collect per-label seed sets from events within the window
        label_cells: Dict[str, set] = {}
        for _frame, label, px, py in self._events:
            if label not in label_cells:
                label_cells[label] = set()
            label_cells[label].add((px, py))

        combined = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        for label, cells in label_cells.items():
            radius = _SEMANTIC_SAFETY_RADIUS_CELLS.get(label, 0)
            seed = np.zeros((self.n_cells, self.n_cells), dtype=np.uint8)
            for px, py in cells:
                seed[px, py] = 1
            kernel = self._get_kernel(radius)
            dilated = cv2.dilate(seed, kernel, iterations=1)
            combined |= dilated.astype(bool)
        return combined

    # ------------------------------------------------------------------
    def apply_to_navigable_map(self, base_nav_map: np.ndarray) -> np.ndarray:
        """Return navigable map with YOLO-detected obstacles removed."""
        obstacle_mask = self.get_obstacle_mask()
        return base_nav_map & ~obstacle_mask
