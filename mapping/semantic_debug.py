"""
SemanticPredGTCollector: accumulates (CLIP-predicted label, GT label) pairs
and max-similarity score distributions across the full evaluation run.

Outputs two CSVs:
  - semantic_pred_gt_pairs.csv   : (pred, gt) count table
  - semantic_sim_distribution.csv: per-GT-category histogram of max_sim scores
"""
import csv
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np


# Bin edges for max_sim histogram (cosine similarity range roughly [-0.3, 0.5])
_SIM_BINS = np.arange(-0.3, 0.55, 0.025)


class SemanticPredGTCollector:
    def __init__(self) -> None:
        self.counts: Counter = Counter()                         # (pred_label, gt_label) -> count
        # gt_category -> histogram counts over _SIM_BINS
        self.sim_hist: dict = defaultdict(lambda: np.zeros(len(_SIM_BINS) - 1, dtype=np.int64))
        self.gt_seed_map: Optional[np.ndarray] = None
        self.gt_labels: List[str] = []

    def set_gt_map(self, gt_seed_map: np.ndarray, gt_labels: List[str]) -> None:
        self.gt_seed_map = gt_seed_map
        self.gt_labels = gt_labels

    def record(
        self,
        valid_mask: np.ndarray,
        pred_ids: np.ndarray,
        pred_labels: List[str],
        max_sims: Optional[np.ndarray] = None,  # 1-D float array, same length as pred_ids
    ) -> None:
        if self.gt_seed_map is None or valid_mask.sum() == 0:
            return

        gt_at_cells = self.gt_seed_map[valid_mask]

        for i, (pred_idx, gt_idx) in enumerate(zip(pred_ids, gt_at_cells)):
            pred_label = (
                pred_labels[int(pred_idx) - 1]
                if 1 <= int(pred_idx) <= len(pred_labels)
                else "unknown"
            )
            gt_label = (
                self.gt_labels[int(gt_idx) - 1]
                if int(gt_idx) > 0 and int(gt_idx) <= len(self.gt_labels)
                else "free"
            )
            self.counts[(pred_label, gt_label)] += 1

            if max_sims is not None:
                sim_val = float(max_sims[i])
                bin_idx = np.searchsorted(_SIM_BINS, sim_val, side="right") - 1
                bin_idx = int(np.clip(bin_idx, 0, len(_SIM_BINS) - 2))
                self.sim_hist[gt_label][bin_idx] += 1

    def save_csv(self, path: str) -> None:
        total = sum(self.counts.values())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pred_label", "gt_label", "count", "pct"])
            for (pred, gt), count in sorted(self.counts.items(), key=lambda x: -x[1]):
                writer.writerow([pred, gt, count, f"{count / total * 100:.2f}"])
        print(f"[SemanticDebug] Saved {len(self.counts)} pred-GT pairs ({total} cells) → {path}")

    def save_sim_distribution_csv(self, path: str) -> None:
        bin_labels = [f"{_SIM_BINS[i]:.3f}~{_SIM_BINS[i+1]:.3f}" for i in range(len(_SIM_BINS) - 1)]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gt_label", "bin", "bin_center", "count"])
            for gt_label, hist in sorted(self.sim_hist.items()):
                total_cat = hist.sum()
                for i, count in enumerate(hist):
                    if count == 0:
                        continue
                    bin_center = round(float((_SIM_BINS[i] + _SIM_BINS[i + 1]) / 2), 4)
                    writer.writerow([gt_label, bin_labels[i], bin_center, int(count)])
        print(f"[SemanticDebug] Saved sim distribution ({len(self.sim_hist)} categories) → {path}")


class YOLOObstacleDebugCollector:
    """Records (YOLO-detected label, GT label at projected cell) pairs.

    Two GT maps are maintained:
      - gt_seed_map   : exact per-instance object centres (1 cell each)
      - gt_dilated_map: each seed dilated by _DILATE_R cells — a loose
                        neighbourhood check that tolerates bbox projection noise

    We report hits against BOTH maps so we can distinguish:
      - True localisation error (miss even the dilated zone)
      - Quantisation / bbox-centre noise (miss exact seed but hit dilated zone)
    """

    _DILATE_R = 5   # cells (~0.5 m) used for loose GT matching

    def __init__(self) -> None:
        self.counts_exact:   Counter = Counter()   # vs exact seed map
        self.counts_dilated: Counter = Counter()   # vs dilated zone
        self.gt_seed_map:    Optional[np.ndarray] = None
        self.gt_dilated_map: Optional[np.ndarray] = None
        self.gt_labels: List[str] = []

    def set_gt_map(self, gt_seed_map: np.ndarray, gt_labels: List[str]) -> None:
        import cv2 as _cv2
        self.gt_seed_map = gt_seed_map
        self.gt_labels   = gt_labels
        # Build per-label dilated map (same label index as seed map)
        r = self._DILATE_R
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
        dilated = np.zeros_like(gt_seed_map)
        for idx in range(1, len(gt_labels) + 1):
            mask = (gt_seed_map == idx).astype(np.uint8)
            if mask.any():
                d = _cv2.dilate(mask, kernel, iterations=1)
                dilated[d > 0] = idx   # later labels overwrite earlier; fine for counting
        self.gt_dilated_map = dilated

    def _lookup(self, label_map: np.ndarray, px: int, py: int) -> str:
        gt_idx = int(label_map[px, py])
        return (
            self.gt_labels[gt_idx - 1]
            if gt_idx > 0 and gt_idx <= len(self.gt_labels)
            else "free"
        )

    def record(self, yolo_label: str, px: int, py: int) -> None:
        """Record one projected YOLO detection against both GT maps."""
        if self.gt_seed_map is None:
            return
        if not (0 <= px < self.gt_seed_map.shape[0] and 0 <= py < self.gt_seed_map.shape[1]):
            return
        self.counts_exact[(yolo_label,   self._lookup(self.gt_seed_map,    px, py))] += 1
        self.counts_dilated[(yolo_label, self._lookup(self.gt_dilated_map, px, py))] += 1

    def save_csv(self, path: str) -> None:
        total = sum(self.counts_exact.values())
        if total == 0:
            print(f"[YOLODebug] No YOLO detections recorded, skipping {path}")
            return

        def _write_table(writer, counts, label):
            writer.writerow([f"--- {label} ---"])
            writer.writerow(["yolo_label", "gt_label", "count", "pct"])
            tot = sum(counts.values())
            for (yolo, gt), count in sorted(counts.items(), key=lambda x: -x[1]):
                writer.writerow([yolo, gt, count, f"{count / tot * 100:.2f}"])
            writer.writerow([])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            _write_table(writer, self.counts_exact,   "EXACT seed match")
            _write_table(writer, self.counts_dilated, f"DILATED match (±{self._DILATE_R} cells = ±{self._DILATE_R*0.1:.1f}m)")
        print(f"[YOLODebug] Saved {len(self.counts_exact)} pairs ({total} detections) → {path}")