from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class OracleDetection:
    label: str
    cell_x: int
    cell_y: int


class ObstacleOracle(ABC):
    """Interface for obstacle detection oracles used by OACP.

    An oracle observes the environment each step and returns per-cell
    obstacle detections.  It also provides the initial safety radius
    for newly discovered obstacle categories.

    Implementations:
      - GTLabelMapOracle  : simulation GT semantic map (current default)
      - (future) GroundingDINOOracle, VLMOracle, ...
    """

    @abstractmethod
    def detect(
        self,
        agent_cell: Tuple[int, int],
    ) -> List[OracleDetection]:
        """Return obstacle detections visible from the agent's current position.

        Args:
            agent_cell: (px, py) agent position in map cell coordinates.

        Returns:
            List of OracleDetection(label, cell_x, cell_y).
        """
        ...

    @abstractmethod
    def get_safety_radius(self, label: str) -> Optional[float]:
        """Return the raw safety radius (in cells) for a label, or None if unknown."""
        ...

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass


class GTLabelMapOracle(ObstacleOracle):
    """Oracle backed by a ground-truth semantic label map (simulation)."""

    def __init__(
        self,
        safety_dict: dict,
        observe_radius_m: float = 3.0,
        cell_size: float = 0.1,
    ):
        self._safety_dict = safety_dict
        self._observe_radius_m = observe_radius_m
        self._cell_size = cell_size
        self._gt_label_map: Optional[np.ndarray] = None
        self._gt_labels: List[str] = []

    def set_gt_label_map(self, label_map: np.ndarray, labels: List[str]) -> None:
        self._gt_label_map = label_map
        self._gt_labels = labels

    def detect(
        self,
        agent_cell: Tuple[int, int],
    ) -> List[OracleDetection]:
        if self._gt_label_map is None:
            return []

        px, py = agent_cell
        n_cells = self._gt_label_map.shape[0]
        delta = max(1, int(self._observe_radius_m / self._cell_size))
        x0 = max(0, px - delta)
        x1 = min(n_cells, px + delta + 1)
        y0 = max(0, py - delta)
        y1 = min(n_cells, py + delta + 1)

        region = self._gt_label_map[x0:x1, y0:y1]
        if not region.any():
            return []

        detections = []
        xs, ys = np.nonzero(region)
        for dx, dy in zip(xs, ys):
            lbl_idx = int(region[dx, dy])
            true_label = self._gt_labels[lbl_idx - 1]
            detections.append(OracleDetection(
                label=true_label,
                cell_x=x0 + dx,
                cell_y=y0 + dy,
            ))
        return detections

    def get_safety_radius(self, label: str) -> Optional[float]:
        raw_r = self._safety_dict.get(label)
        if raw_r is None:
            return None
        return raw_r

    def reset(self) -> None:
        pass


class GroundingDINOOracle(ObstacleOracle):
    """Oracle that builds a label map online using Grounding DINO detections
    projected to grid cells via depth back-projection.

    Same interface as GTLabelMapOracle so it can drop-in replace GT calibration.
    """

    def __init__(
        self,
        safety_dict: dict,
        n_cells: int,
        cell_size: float,
        obstacle_labels: List[str],
        confidence_threshold: float = 0.3,
        observe_radius_m: float = 3.0,
    ):
        self._safety_dict = safety_dict
        self._n_cells = n_cells
        self._cell_size = cell_size
        self._observe_radius_m = observe_radius_m
        self._confidence_threshold = confidence_threshold

        from eval.semantic_collision import normalize_semantic_label
        self._obstacle_labels = [normalize_semantic_label(l) for l in obstacle_labels]
        self._label_to_idx: Dict[str, int] = {
            l: i + 1 for i, l in enumerate(self._obstacle_labels)
        }
        self._gdino_text = " . ".join(self._obstacle_labels) + " ."

        self._label_map = np.zeros((n_cells, n_cells), dtype=np.uint16)
        self._conf_map = np.zeros((n_cells, n_cells), dtype=np.float32)

        self._last_detections: List[Tuple[int, int, int, int, float, str]] = []

        self._detector = None
        self._fx = self._fy = self._cx = self._cy = 0.0

    def _ensure_detector(self):
        if self._detector is not None:
            return
        from pathlib import Path
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        local_path = Path(__file__).resolve().parent.parent / "weights" / "grounding-dino-base"
        model_src = str(local_path) if local_path.exists() else "IDEA-Research/grounding-dino-base"
        self._processor = AutoProcessor.from_pretrained(model_src)
        self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_src).to("cuda")
        self._detector.eval()

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        self._fx, self._fy, self._cx, self._cy = fx, fy, cx, cy

    def update(
        self,
        image_hwc: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_world: np.ndarray,
    ) -> None:
        """Run Grounding DINO on current frame and project detections to grid."""
        self._ensure_detector()
        from PIL import Image as PILImage

        h, w = depth.shape
        pil_img = PILImage.fromarray(image_hwc)

        inputs = self._processor(
            images=pil_img, text=self._gdino_text, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = self._detector(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._confidence_threshold,
            text_threshold=self._confidence_threshold,
            target_sizes=[(h, w)],
        )[0]

        if len(results["boxes"]) == 0:
            self._last_detections = []
            return

        from eval.semantic_collision import normalize_semantic_label

        # Build per-pixel label image from detections (highest confidence wins)
        pixel_label = np.zeros((h, w), dtype=np.uint16)
        pixel_conf = np.zeros((h, w), dtype=np.float32)

        frame_dets: List[Tuple[int, int, int, int, float, str]] = []
        for box, score, raw_label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = (
                int(box[0].item()), int(box[1].item()),
                int(box[2].item()), int(box[3].item()),
            )
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            conf = float(score)
            label = normalize_semantic_label(raw_label)
            if label not in self._label_to_idx:
                continue
            lbl_idx = self._label_to_idx[label]
            frame_dets.append((x1, y1, x2, y2, conf, label))
            mask = pixel_conf[y1:y2, x1:x2] < conf
            pixel_label[y1:y2, x1:x2] = np.where(mask, lbl_idx, pixel_label[y1:y2, x1:x2])
            pixel_conf[y1:y2, x1:x2] = np.where(mask, conf, pixel_conf[y1:y2, x1:x2])
        self._last_detections = frame_dets

        labeled_mask = pixel_label > 0
        if not labeled_mask.any():
            return

        # Back-project labeled pixels to grid cells
        depth_t = torch.tensor(depth, dtype=torch.float32, device="cuda")
        u_coords = torch.arange(w, device="cuda", dtype=torch.float32)
        v_coords = torch.arange(h, device="cuda", dtype=torch.float32)
        uu, vv = torch.meshgrid(u_coords, v_coords, indexing="xy")
        uu = uu.flatten()
        vv = vv.flatten()
        zz = depth_t.flatten()

        x_cam = (uu - self._cx) * zz / self._fx
        y_cam = (vv - self._cy) * zz / self._fy
        # Camera frame: z=depth, x=right, y=down → world: (z, -x, -y)
        pcl = torch.stack([zz, -x_cam, -y_cam], dim=1)  # [H*W, 3]

        tf = torch.tensor(tf_camera_to_world, dtype=torch.float32, device="cuda")
        rot = tf[:3, :3]
        yaw = torch.atan2(rot[1, 0], rot[0, 0])
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        rot2d = torch.tensor([[cos_y, -sin_y], [sin_y, cos_y]], device="cuda")
        pcl[:, :2] = (rot2d @ pcl[:, :2].T).T
        cam_x = tf[0, 3] / tf[3, 3]
        cam_y = tf[1, 3] / tf[3, 3]
        pcl[:, 0] += cam_x
        pcl[:, 1] += cam_y

        gx = torch.floor(pcl[:, 0] / self._cell_size).to(torch.int32) + self._n_cells // 2
        gy = torch.floor(pcl[:, 1] / self._cell_size).to(torch.int32) + self._n_cells // 2

        labeled_flat = torch.tensor(pixel_label.flatten().astype(np.int32), device="cuda")
        conf_flat = torch.tensor(pixel_conf.flatten(), device="cuda")
        valid = (
            (labeled_flat > 0) &
            (zz > 0) & (zz != float("inf")) &
            (gx >= 0) & (gx < self._n_cells) &
            (gy >= 0) & (gy < self._n_cells)
        )
        if not valid.any():
            return

        gx_v = gx[valid].cpu().numpy().astype(np.intp)
        gy_v = gy[valid].cpu().numpy().astype(np.intp)
        lbl_v = labeled_flat[valid].cpu().numpy().astype(np.uint16)
        conf_v = conf_flat[valid].cpu().numpy()

        # Per-cell max-confidence aggregation: for duplicate grid cells, keep
        # the detection with highest confidence using lexsort + unique.
        cell_ids = gx_v * self._n_cells + gy_v
        order = np.lexsort((conf_v, cell_ids))
        cell_ids_sorted = cell_ids[order]
        unique_mask = np.empty(len(cell_ids_sorted), dtype=bool)
        unique_mask[-1] = True
        unique_mask[:-1] = cell_ids_sorted[:-1] != cell_ids_sorted[1:]
        # After lexsort by (conf asc, cell_id), the last occurrence per cell
        # has the highest confidence.
        idx = order[unique_mask]
        ux, uy = gx_v[idx], gy_v[idx]
        uc, ul = conf_v[idx], lbl_v[idx]
        better = uc > self._conf_map[ux, uy]
        self._label_map[ux[better], uy[better]] = ul[better]
        self._conf_map[ux[better], uy[better]] = uc[better]

    def detect(
        self,
        agent_cell: Tuple[int, int],
    ) -> List[OracleDetection]:
        px, py = agent_cell
        delta = max(1, int(self._observe_radius_m / self._cell_size))
        x0 = max(0, px - delta)
        x1 = min(self._n_cells, px + delta + 1)
        y0 = max(0, py - delta)
        y1 = min(self._n_cells, py + delta + 1)

        region = self._label_map[x0:x1, y0:y1]
        if not region.any():
            return []

        detections = []
        xs, ys = np.nonzero(region)
        for dx, dy in zip(xs, ys):
            lbl_idx = int(region[dx, dy])
            label = self._obstacle_labels[lbl_idx - 1]
            detections.append(OracleDetection(
                label=label,
                cell_x=x0 + dx,
                cell_y=y0 + dy,
            ))
        return detections

    def get_safety_radius(self, label: str) -> Optional[float]:
        return self._safety_dict.get(label)

    def reset(self) -> None:
        self._label_map[:] = 0
        self._conf_map[:] = 0.0
        self._last_detections = []
