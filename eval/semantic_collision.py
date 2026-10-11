from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


_SEMANTIC_LABEL_ALIASES = {
    "tv monitor": "tv",
    "tvmonitor": "tv",
    "television": "tv",
    "monitor": "tv",
    "plant": "potted plant",
    "potted plant": "potted plant",
    "flower pot": "potted plant",
    "flowerpot": "potted plant",
    "decorative plant": "potted plant",
    "diningtable": "dining table",
    "coffee table": "dining table",
    "sofa": "couch",
    "cushion": "pillow",
    # lamp aliases → "lamp"
    "floor lamp": "lamp",
    "table lamp": "lamp",
    "desk lamp": "lamp",
    "lamp table": "lamp",
    "lamps": "lamp",
    "lamp ceiling": "lamp",
    "lamp desk": "lamp",
    "台灯": "lamp",
    "落地灯": "lamp",
    "floor light": "lamp",
    "standing lamp": "lamp",
    # MP3D category aliases
    "tv_monitor": "tv",
    "chest_of_drawers": "chest of drawers",
    "dresser": "chest of drawers",
}

_SEMANTIC_SAFETY_RADIUS_CELLS = {
    "shower":           3,
    "cabinet":          4,
    "chest of drawers": 4,
    "table":            5,
    "tv":               2,
}

_DEFAULT_SAFETY_RADIUS_CELLS = 1

# COCO classes that never appear indoors — excluded from argmax background competitors
_OUTDOOR_COCO_LABELS = frozenset({
    # vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # street furniture
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    # large outdoor animals
    "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    # outdoor sports equipment
    "frisbee", "skis", "snowboard", "kite", "skateboard", "surfboard",
    "baseball bat", "baseball glove", "tennis racket", "sports ball",
})

_NON_OBSTACLE_LABELS = frozenset({
    "unknown", "ceiling", "floor", "wall", "void", "misc",
    "window", "curtain", "stairs", "staircase",
    "door", "dorr", "door frame", "pillar", "balustrade",
    "chandelier", "light", "ventilation", "fire alarm",
    "handle", "alarm control", "shower dial", "electrical controller",
    "curtain rod",
    "photo", "picture", "frame", "flag", "decorative plate",
    "kitchen countertop item",
})


@dataclass(frozen=True)
class SemanticCollisionData:
    collision_map: np.ndarray
    label_map: np.ndarray
    labels: List[str]


def normalize_semantic_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return _SEMANTIC_LABEL_ALIASES.get(normalized, normalized)


def metric_to_px(x: float, y: float, n_cells: int, cell_size: float) -> Tuple[int, int]:
    center = n_cells // 2
    epsilon = 1e-9
    return (
        int(x / cell_size + center + epsilon),
        int(y / cell_size + center + epsilon),
    )


def get_semantic_safety_radius_cells(
    label: str,
    query_label: str,
    max_query_radius_cells: Optional[int],
) -> int:
    normalized_label = normalize_semantic_label(label)
    normalized_query = normalize_semantic_label(query_label)
    if normalized_label == normalized_query:
        return -1  # navigation target → exclude from collision map entirely
    radius = _SEMANTIC_SAFETY_RADIUS_CELLS.get(normalized_label)
    if radius is None:
        return -1  # not in obstacle whitelist → treat as free space
    return max(int(radius), 0)


# Height thresholds relative to floor_y:
#   bottom must be within 0.3 m above floor (not elevated on shelves)
#   centre must be within 1.5 m above floor (not hanging/wall-mounted)
#   bottom must be within 0.5 m below floor (not on a lower storey — phantom fix)
_MAX_BOTTOM_ABOVE_FLOOR = 0.3
_MAX_CENTER_ABOVE_FLOOR = 1.5
_MAX_BOTTOM_BELOW_FLOOR = 0.5


def _iter_object_metric_points(
    objects: Sequence,
    is_gibson: bool,
    floor_y: Optional[float] = None,
) -> Iterable[Tuple[float, float]]:
    for obj in objects:
        if is_gibson:
            points = np.asarray(obj, dtype=np.float32)
            if points.ndim == 1 and points.shape[0] >= 2:
                yield float(points[0]), float(points[1])
                continue
            for pt in points:
                if len(pt) >= 2:
                    yield float(pt[0]), float(pt[1])
            continue

        center = np.asarray(obj.bbox.center, dtype=np.float32)
        if center.shape[0] < 3:
            continue

        # Height filter relative to this episode's floor Y.
        # Skips objects on shelves, hanging on walls, mounted too high, etc.
        if floor_y is not None:
            try:
                sizes = np.asarray(obj.bbox.sizes, dtype=np.float32)
                bottom_y = float(center[1]) - float(sizes[1]) / 2.0
                if (bottom_y - floor_y) > _MAX_BOTTOM_ABOVE_FLOOR:
                    continue
                if (floor_y - bottom_y) > _MAX_BOTTOM_BELOW_FLOOR:
                    continue
            except Exception:
                pass
            if (float(center[1]) - floor_y) > _MAX_CENTER_ABOVE_FLOOR:
                continue

        yield float(-center[2]), float(-center[0])


def _build_label_seed_map(
    object_locations: Dict[str, Sequence],
    n_cells: int,
    cell_size: float,
    is_gibson: bool,
    floor_y: Optional[float] = None,
    label_dict: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if label_dict is None:
        label_dict = _SEMANTIC_SAFETY_RADIUS_CELLS
    labels: List[str] = []
    label_to_idx: Dict[str, int] = {}
    seed_map = np.zeros((n_cells, n_cells), dtype=np.uint16)

    for raw_label, objects in object_locations.items():
        label = normalize_semantic_label(raw_label)
        if label not in label_dict:
            continue
        if label not in label_to_idx:
            label_to_idx[label] = len(labels) + 1
            labels.append(label)
        label_idx = label_to_idx[label]

        for x, y in _iter_object_metric_points(objects, is_gibson, floor_y=floor_y):
            px, py = metric_to_px(x, y, n_cells=n_cells, cell_size=cell_size)
            if 0 <= px < n_cells and 0 <= py < n_cells:
                seed_map[px, py] = label_idx

    return seed_map, labels


def _make_disk_kernel(radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    diameter = radius_cells * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def build_semantic_collision_data(
    object_locations: Dict[str, Sequence],
    n_cells: int,
    size: float,
    is_gibson: bool,
    query_label: str,
    max_query_radius_cells: int,
    floor_y: Optional[float] = None,
    oacp_radius_override: Optional[int] = None,
    label_dict: Optional[Dict[str, int]] = None,
) -> SemanticCollisionData:
    cell_size = size / float(n_cells)
    seed_map, labels = _build_label_seed_map(object_locations, n_cells, cell_size, is_gibson,
                                              floor_y=floor_y, label_dict=label_dict)

    collision_map = np.zeros((n_cells, n_cells), dtype=bool)
    label_map = np.zeros((n_cells, n_cells), dtype=np.uint16)

    for idx, label in enumerate(labels, start=1):
        class_seed = (seed_map == idx).astype(np.uint8)
        if not class_seed.any():
            continue

        if oacp_radius_override is not None:
            radius_cells = oacp_radius_override
        else:
            radius_cells = get_semantic_safety_radius_cells(label, query_label, max_query_radius_cells)
            if radius_cells < 0:
                continue  # excluded (e.g. query label or not in whitelist)
        kernel = _make_disk_kernel(radius_cells)
        dilated = cv2.dilate(class_seed, kernel, iterations=1).astype(bool)
        collision_map |= dilated
        label_map[dilated] = idx

    return SemanticCollisionData(
        collision_map=collision_map,
        label_map=label_map,
        labels=labels,
    )
