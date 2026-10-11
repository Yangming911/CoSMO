from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from eval.semantic_collision import get_semantic_safety_radius_cells, normalize_semantic_label, _NON_OBSTACLE_LABELS, _SEMANTIC_SAFETY_RADIUS_CELLS


@dataclass
class SemanticLabelConfig:
    labels: List[str]
    radii: np.ndarray
    max_radius: int


def build_semantic_label_config(raw_labels: List[str]) -> SemanticLabelConfig:
    labels: List[str] = []
    label_to_idx: Dict[str, int] = {}
    radii: List[int] = []

    for raw_label in raw_labels:
        label = normalize_semantic_label(raw_label)
        if not label:
            continue
        if label in label_to_idx:
            continue
        label_to_idx[label] = len(labels)
        labels.append(label)
        if label in _SEMANTIC_SAFETY_RADIUS_CELLS:
            radii.append(int(_SEMANTIC_SAFETY_RADIUS_CELLS[label]))
        else:
            radii.append(-1)  # not in obstacle whitelist → treat as free space

    radii_arr = np.asarray(radii, dtype=np.int32) if radii else np.zeros((0,), dtype=np.int32)
    positive_radii = radii_arr[radii_arr >= 0]
    max_radius = int(positive_radii.max()) if positive_radii.size > 0 else 0
    return SemanticLabelConfig(labels=labels, radii=radii_arr, max_radius=max_radius)


class SemanticNavigableMapUpdater:
    def __init__(self, n_cells: int, label_config: SemanticLabelConfig) -> None:
        self.n_cells = n_cells
        self.labels = label_config.labels
        self.radii = label_config.radii
        self.max_radius = label_config.max_radius
        self._kernels = [
            self._make_disk_kernel(int(r)) if r >= 0 else None
            for r in self.radii.tolist()
        ]
        self.seed_label_map = np.zeros((n_cells, n_cells), dtype=np.uint16)
        self.semantic_blocked_map = np.zeros((n_cells, n_cells), dtype=bool)
        self.semantic_navigable_map = np.ones((n_cells, n_cells), dtype=bool)

    @staticmethod
    def _make_disk_kernel(radius_cells: int) -> np.ndarray:
        if radius_cells <= 0:
            return np.ones((1, 1), dtype=np.uint8)
        diameter = radius_cells * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

    @staticmethod
    def _expand_bounds(
        x_min: int,
        x_max_excl: int,
        y_min: int,
        y_max_excl: int,
        margin: int,
        n_cells: int,
    ) -> Tuple[int, int, int, int]:
        return (
            max(0, x_min - margin),
            min(n_cells, x_max_excl + margin),
            max(0, y_min - margin),
            min(n_cells, y_max_excl + margin),
        )

    def reset(self, base_navigable_map: np.ndarray) -> None:
        self.seed_label_map.fill(0)
        self.semantic_blocked_map.fill(False)
        self.semantic_navigable_map = np.asarray(base_navigable_map, dtype=bool).copy()

    def update(
        self,
        base_navigable_map: np.ndarray,
        updated_mask: np.ndarray,
        label_ids_for_updated_cells: np.ndarray,
    ) -> np.ndarray:
        base = np.asarray(base_navigable_map, dtype=bool)
        changed_mask = np.asarray(updated_mask, dtype=bool)
        if not changed_mask.any():
            self.semantic_navigable_map = base & (~self.semantic_blocked_map)
            return self.semantic_navigable_map

        labels_updated = np.asarray(label_ids_for_updated_cells, dtype=np.uint16)
        if labels_updated.shape != self.seed_label_map.shape:
            raise ValueError("label_ids_for_updated_cells shape must match map shape")

        old_labels = self.seed_label_map[changed_mask]
        new_labels = labels_updated[changed_mask]
        if np.array_equal(old_labels, new_labels):
            self.semantic_navigable_map = base & (~self.semantic_blocked_map)
            return self.semantic_navigable_map

        self.seed_label_map[changed_mask] = new_labels

        changed_x, changed_y = np.nonzero(changed_mask)
        x_min = int(changed_x.min())
        x_max_excl = int(changed_x.max()) + 1
        y_min = int(changed_y.min())
        y_max_excl = int(changed_y.max()) + 1

        tx0, tx1, ty0, ty1 = self._expand_bounds(
            x_min,
            x_max_excl,
            y_min,
            y_max_excl,
            self.max_radius,
            self.n_cells,
        )
        sx0, sx1, sy0, sy1 = self._expand_bounds(
            tx0,
            tx1,
            ty0,
            ty1,
            self.max_radius,
            self.n_cells,
        )

        target_slice = (slice(tx0, tx1), slice(ty0, ty1))
        source_seed = self.seed_label_map[sx0:sx1, sy0:sy1]
        source_h, source_w = source_seed.shape
        local_blocked = np.zeros((source_h, source_w), dtype=bool)

        for label_idx, kernel in enumerate(self._kernels, start=1):
            if kernel is None:
                continue
            class_seed = (source_seed == label_idx).astype(np.uint8)
            if class_seed.max() == 0:
                continue
            local_blocked |= cv2.dilate(class_seed, kernel, iterations=1).astype(bool)

        rel_tx0 = tx0 - sx0
        rel_tx1 = tx1 - sx0
        rel_ty0 = ty0 - sy0
        rel_ty1 = ty1 - sy0
        self.semantic_blocked_map[target_slice] = local_blocked[rel_tx0:rel_tx1, rel_ty0:rel_ty1]
        self.semantic_navigable_map = base & (~self.semantic_blocked_map)
        return self.semantic_navigable_map
