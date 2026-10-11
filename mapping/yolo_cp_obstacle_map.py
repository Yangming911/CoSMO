"""
YOLO-CP Obstacle Map: Conformalized YOLO-based obstacle detection.

Design (mirrors Yannis et al., "Safe Planning in Unknown Environments using
Conformalized Semantic Maps", 2026):

  Predictor : YOLO detection confidence stored per grid cell.
               conf_map[px, py, class_idx] = max confidence seen at that cell
               for each obstacle class.

  Oracle    : GT semantic collision data (Habitat simulator).
               At each frame, GT obstacle cells visible in the camera frustum
               are used as calibration events:
                 - YOLO detected the cell  → score = 1 - conf  (low = good)
                 - YOLO missed the cell    → score = 1.0        (max penalty)

  CP update : τ = 1 - quantile(scores, target_coverage) when ≥ min_samples.
               Cells with conf ≥ τ are marked as obstacles and dilated.

Frustum optimisation: the camera FOV is fixed, so the set of grid-cell offsets
visible from the robot (in the robot's local frame) is precomputed once and
rotated per-frame with a 2-D rotation matrix.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from eval.semantic_collision import (
    _SEMANTIC_SAFETY_RADIUS_CELLS,
    normalize_semantic_label,
)


class YOLOCPObstacleMap:
    def __init__(
        self,
        n_cells: int,
        cell_size: float,
        hfov_deg: float = 90.0,
        max_depth: float = 5.0,
        threshold: float = 0.5,
        target_coverage: float = 0.9,
        window_size: int = 100,
        min_calibration_samples: int = 10,
    ) -> None:
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.map_center = n_cells // 2

        # Obstacle classes tracked (from shared dict)
        self._obstacle_classes: List[str] = list(_SEMANTIC_SAFETY_RADIUS_CELLS.keys())
        self._class_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self._obstacle_classes)
        }
        n_cls = len(self._obstacle_classes)

        # Per-cell max confidence: [n_cells, n_cells, n_obstacle_classes]
        self._conf_map = np.zeros((n_cells, n_cells, n_cls), dtype=np.float32)
        # Frame index at which each cell was last updated (0 = never seen)
        self._conf_frame = np.zeros((n_cells, n_cells, n_cls), dtype=np.int32)
        self._frame_idx: int = 0
        self._conf_window: int = window_size  # same window used for CP scores

        # CP state
        self.threshold = threshold
        self.target_coverage = target_coverage
        self._scores: deque = deque(maxlen=window_size)
        self._min_samples = min_calibration_samples

        # Dilation kernels keyed by radius
        self._kernels: Dict[int, np.ndarray] = {}

        # Precompute frustum offsets in robot-local frame (robot faces +x)
        self._frustum_offsets = self._precompute_frustum(hfov_deg, max_depth, cell_size)

    # ------------------------------------------------------------------
    @staticmethod
    def _precompute_frustum(hfov_deg: float, max_depth: float, cell_size: float) -> np.ndarray:
        """Return (N, 2) integer offsets [dx, dy] inside the camera frustum.

        Robot faces +x; frustum spans ±hfov/2 horizontally up to max_depth.
        """
        half_fov = np.radians(hfov_deg / 2.0)
        max_cells = int(max_depth / cell_size) + 1
        offsets = []
        for dx in range(1, max_cells + 1):
            for dy in range(-max_cells, max_cells + 1):
                angle = abs(np.arctan2(abs(dy), dx))
                dist = np.sqrt(dx * dx + dy * dy) * cell_size
                if angle <= half_fov and dist <= max_depth:
                    offsets.append([dx, dy])
        if not offsets:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(offsets, dtype=np.float32)  # [N, 2]

    def _get_kernel(self, radius: int) -> np.ndarray:
        if radius not in self._kernels:
            if radius <= 0:
                self._kernels[radius] = np.ones((1, 1), dtype=np.uint8)
            else:
                d = radius * 2 + 1
                self._kernels[radius] = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (d, d)
                )
        return self._kernels[radius]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._conf_map.fill(0.0)
        self._conf_frame.fill(0)
        self._frame_idx = 0
        self._scores.clear()

    # ------------------------------------------------------------------
    def update(
        self,
        detections_with_conf: Dict[str, List[Tuple[List[float], float]]],
        depth: np.ndarray,
        tf: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_h: int,
        image_w: int,
    ) -> None:
        """Project YOLO detections and store max confidence per cell.

        Method B: all depth pixels within the bounding box are back-projected
        to map cells (rather than a single bbox-centre point).  Only foreground
        pixels (depth ≤ median × 1.2) are used to avoid projecting background
        geometry behind the detected object.

        Args:
            detections_with_conf: {class_name: [(box, conf), ...]}
                                   box = [x1, y1, x2, y2] in pixel coords.
        """
        self._frame_idx += 1
        yaw = float(np.arctan2(tf[1, 0], tf[0, 0]))
        cam_x = float(tf[0, 3] / tf[3, 3])
        cam_y = float(tf[1, 3] / tf[3, 3])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        for class_name, det_list in detections_with_conf.items():
            label = normalize_semantic_label(class_name)
            if label not in self._class_to_idx:
                continue
            class_idx = self._class_to_idx[label]

            for box, conf in det_list:
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
                # Keep only foreground (≤ median × 1.2 to exclude background walls)
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

                # Deduplicate and update conf_map + frame stamp
                unique_cells = np.unique(np.stack([pxs_b, pys_b], axis=1), axis=0)
                pxs_u, pys_u = unique_cells[:, 0], unique_cells[:, 1]
                update_mask = conf > self._conf_map[pxs_u, pys_u, class_idx]
                self._conf_map[pxs_u[update_mask], pys_u[update_mask], class_idx] = conf
                # Stamp ALL projected cells (even if conf didn't improve) so the
                # window keeps them alive as long as they're still being seen.
                self._conf_frame[pxs_u, pys_u, class_idx] = self._frame_idx

    # ------------------------------------------------------------------
    def calibrate_with_gt(
        self,
        gt_label_map: np.ndarray,
        gt_labels: List[str],
        robot_px: int,
        robot_py: int,
        robot_yaw: float,
    ) -> None:
        """Online CP calibration using GT semantic map as oracle.

        For each GT obstacle cell visible in the camera frustum:
          - YOLO detected it (conf > 0) → score = 1 - conf
          - YOLO missed it              → score = 1.0
        τ is updated to 1 - quantile(scores, target_coverage).

        Args:
            gt_label_map: [n_cells, n_cells] uint16, 0=free, idx=obstacle class
                          (indices into gt_labels, 1-based).
            gt_labels:    List of obstacle label strings (1-based into label_map).
            robot_px/py:  Robot position in map grid coordinates.
            robot_yaw:    Robot heading in radians.
        """
        if self._frustum_offsets.shape[0] == 0:
            return

        # Rotate precomputed frustum offsets by current yaw
        cos_y, sin_y = np.cos(robot_yaw), np.sin(robot_yaw)
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)
        rotated = (R @ self._frustum_offsets.T).T  # [N, 2]
        cell_coords = np.round(rotated + np.array([robot_px, robot_py])).astype(int)

        # Filter in-bounds (vectorized)
        valid_mask = (
            (cell_coords[:, 0] >= 0) & (cell_coords[:, 0] < self.n_cells) &
            (cell_coords[:, 1] >= 0) & (cell_coords[:, 1] < self.n_cells)
        )
        cell_coords = cell_coords[valid_mask]
        if len(cell_coords) == 0:
            return

        # Vectorized GT lookup — skip free cells early
        gt_values = gt_label_map[cell_coords[:, 0], cell_coords[:, 1]]  # [N]
        obs_mask = gt_values > 0
        if not obs_mask.any():
            return

        obs_coords = cell_coords[obs_mask]       # [M, 2]
        obs_gt_idx = gt_values[obs_mask] - 1     # [M] 0-based into gt_labels

        # Loop only over obstacle cells (M << N in practice).
        # Only calibrate on cells where YOLO produced a detection (conf > 0).
        # Cells YOLO never saw cannot be covered by our predictor regardless of τ,
        # so including them as score=1.0 would collapse τ→0 (marking whole map blocked).
        for i in range(len(obs_coords)):
            gt_label = normalize_semantic_label(gt_labels[int(obs_gt_idx[i])])
            if gt_label not in self._class_to_idx:
                continue
            class_idx = self._class_to_idx[gt_label]
            cpx, cpy = obs_coords[i]
            conf = float(self._conf_map[cpx, cpy, class_idx])
            frame = int(self._conf_frame[cpx, cpy, class_idx])
            in_window = (self._frame_idx - frame) <= self._conf_window
            if conf > 0.0 and in_window:        # skip cells YOLO never detected or too old
                self._scores.append(1.0 - conf)

        if len(self._scores) >= self._min_samples:
            self.threshold = float(
                1.0 - np.quantile(list(self._scores), self.target_coverage)
            )

    # ------------------------------------------------------------------
    def get_obstacle_mask(self) -> np.ndarray:
        """Return dilated obstacle mask (True = blocked).

        Only cells updated within the last _conf_window frames are considered,
        so stale projections from previous rooms cannot permanently block paths.
        """
        cutoff = self._frame_idx - self._conf_window
        combined = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        for class_name, class_idx in self._class_to_idx.items():
            radius = _SEMANTIC_SAFETY_RADIUS_CELLS[class_name]
            in_window = self._conf_frame[:, :, class_idx] > cutoff
            seed = (self._conf_map[:, :, class_idx] >= self.threshold) & in_window
            if not seed.any():
                continue
            kernel = self._get_kernel(radius)
            dilated = cv2.dilate(seed.astype(np.uint8), kernel, iterations=1)
            combined |= dilated.astype(bool)
        return combined

    def apply_to_navigable_map(self, base: np.ndarray) -> np.ndarray:
        return base & ~self.get_obstacle_mask()
