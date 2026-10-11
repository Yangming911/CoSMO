"""
This module contains the Navigator class, which is responsible for the main functionality. It updates Onemap and uses it
for navigation and exploration.
"""
import time
from collections import deque

from mapping import (OneMap, detect_frontiers, get_frontier_midpoint,
                     cluster_high_similarity_regions, find_local_maxima,
                     watershed_clustering, gradient_based_clustering, cluster_thermal_image,
                     Cluster, NavGoal, Frontier)

from planning import Planning
from vision_models.base_model import BaseModel
from vision_models.coco_classes import COCO_CLASSES
from eval.semantic_collision import _NON_OBSTACLE_LABELS
from vision_models.yolo_world_detector import YOLOWorldDetector
from onemap_utils import monochannel_to_inferno_rgb, log_map_rerun
from config import Conf, load_config
from config import SpotControllerConf
from mobile_sam import sam_model_registry, SamPredictor
from mapping.semantic_navigable_map import SemanticNavigableMapUpdater, build_semantic_label_config
from mapping.semantic_debug import SemanticPredGTCollector
from mapping.yolo_obstacle_map import YOLOObstacleMap
from mapping.clip_cp_obstacle_map import CLIPCPObstacleMap
from mapping.yolo_cp_obstacle_map import YOLOCPObstacleMap
from mapping.oracle import ObstacleOracle, GTLabelMapOracle, GroundingDINOOracle, OracleDetection

# numpy
import numpy as np

# typing
from typing import List, Optional, Set, Any, Union

# torch
import torch
import torch.nn.functional as F

# warnings
import warnings

# rerun
import rerun as rr

# cv2
import cv2


def closest_point_within_threshold(nav_goals: List[NavGoal], target_point: np.ndarray, threshold: float) -> int:
    """Find the point within the threshold distance that is closest to the target_point.

    Args:
        nav_goals (List[NavGoal]): An array of potential nav points, where each point is retrieved by nav_goal.get_descr_point
            (x, y).
        target_point (np.ndarray): The target 2D point (x, y).
        threshold (float): The maximum distance threshold.

    Returns:
        int: The index of the closest point within the threshold distance.
    """
    points_array = np.array([nav_goal.get_descr_point() for nav_goal in nav_goals])
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)

    return -1


class HistoricDetectData:
    def __init__(self, position: np.ndarray, action: str, other: Any = None):
        self.position = position
        self.other = other
        self.action = action

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.action}_{self.other}"
        return hash(string_repr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HistoricDetectData):
            return NotImplemented
        return (np.array_equal(self.position, other.position) and
                self.action == other.action and
                self.other == other.other)


class CyclicDetectChecker:
    history: Set[HistoricDetectData] = set()

    def check_cyclic(self, position: np.ndarray, action: str, other: Any = None) -> bool:
        state_action = HistoricDetectData(position, action, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: str, other: Any = None) -> None:
        state_action = HistoricDetectData(position, action, other)
        self.history.add(state_action)


class HistoricData:
    def __init__(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None):
        self.position = position
        self.frontier_pt = frontier_pt
        self.other = other

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.frontier_pt}_{self.other}"
        return hash(string_repr)


class CyclicChecker:
    history: Set[HistoricData] = set()

    def check_cyclic(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> bool:
        state_action = HistoricData(position, frontier_pt, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, frontier_pt: np.ndarray, other: Any = None) -> None:
        state_action = HistoricData(position, frontier_pt, other)
        self.history.add(state_action)


class Navigator:
    query_text: List[str]  # the query texts as a list. The first element will be used for planning and frontier score
    # computation
    query_text_features: torch.Tensor
    points_of_interest: List[Cluster]
    blacklisted_nav_goals: List[np.ndarray]
    nav_goals: List[NavGoal]
    last_nav_goal: Union[NavGoal, None]

    def __init__(self,
                 model: BaseModel,
                 detector: YOLOWorldDetector,
                 config: Conf
                 ) -> None:

        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.config = config

        # Models
        self.model = model
        self.detector = detector
        sam_model_t = "vit_t"
        sam_checkpoint = "weights/mobile_sam.pt"
        self.sam = sam_model_registry[sam_model_t](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda")
        self.sam.eval()
        self.sam_predictor = SamPredictor(self.sam)

        self.one_map = OneMap(self.model.feature_dim, config.mapping, map_device="cpu")

        # Dual-model: GCLIP for obstacle detection (separate from navigation model)
        self.use_gclip_for_obstacles = bool(getattr(config.mapping, "use_gclip_for_obstacles", False))
        self.gclip_model = None
        if self.use_gclip_for_obstacles:
            from vision_models.gclip_dense import GCLIPModel
            self.gclip_model = GCLIPModel(clip_input_size=640)
            self.gclip_model.eval()
            self.one_map.init_gclip_map(
                self.gclip_model.feature_dim,
                aggregation=str(getattr(config.mapping, "clip_gclip_aggregation", "min_depth")),
            )

        self.use_clip_semantic_nav_map = bool(getattr(config.mapping, "use_clip_semantic_nav_map", False))
        self.clip_semantic_sim_threshold = float(getattr(config.mapping, "clip_semantic_sim_threshold", 0.0))
        self.semantic_label_config = None
        self.semantic_text_features = None
        self.semantic_map_updater = None
        if self.use_clip_semantic_nav_map:
            self.semantic_label_config = build_semantic_label_config(list(COCO_CLASSES) + list(_NON_OBSTACLE_LABELS))
            self.semantic_text_features = self.model.get_text_features(
                [f"a {label}" for label in self.semantic_label_config.labels]
            ).to(self.one_map.map_device)
            self.semantic_map_updater = SemanticNavigableMapUpdater(self.one_map.n_cells, self.semantic_label_config)
            self.semantic_map_updater.reset(self.one_map.navigable_map)
        self.semantic_navigable_map = self.one_map.navigable_map.copy()
        self.debug_collector: SemanticPredGTCollector = None

        self.use_yolo_obstacle_map = bool(getattr(config.mapping, "use_yolo_obstacle_map", False))
        self.yolo_obstacle_map: YOLOObstacleMap = None
        if self.use_yolo_obstacle_map:
            window_size = int(getattr(config.mapping, "yolo_window_size", 50))
            self.yolo_obstacle_map = YOLOObstacleMap(
                self.one_map.n_cells, self.one_map.cell_size,
                window_size=window_size,
            )

        self.use_clip_argmax_obstacle_map = bool(getattr(config.mapping, "use_clip_argmax_obstacle_map", False))
        self.use_clip_cp_obstacle_map = bool(getattr(config.mapping, "use_clip_cp_obstacle_map", False))
        self._cp_goal_zone_disable = bool(getattr(config.mapping, "clip_cp_goal_zone_disable", True))
        self._cp_step_guard = bool(getattr(config.mapping, "clip_cp_step_guard", False))
        self._cp_step_guard_lookahead = int(getattr(config.mapping, "clip_cp_step_guard_lookahead", 3))
        self._cp_detection_gate = bool(getattr(config.mapping, "clip_cp_detection_gate", False))
        # ACI calibration delay (Proposition 2): buffer GT samples for k steps before calibrating.
        # delay=0 preserves the original direct-call behavior.
        self._calibration_delay = int(getattr(config.mapping, "clip_cp_calibration_delay", 0))
        self._delay_adaptive_gamma = bool(getattr(config.mapping, "clip_cp_delay_adaptive_gamma", False))
        self._delay_buffer: deque = deque()
        self._step_count = 0
        self._cells_per_step_ema = 0.0  # exponential moving average of cells buffered per step
        self._gclip_query_feat: Optional[torch.Tensor] = None  # GCLIP query embedding, set in set_query()
        self._open_vocab = False
        self._oracle_dict: dict = {}           # label -> effective radius (cells)
        self._oracle_text_feats: Optional[torch.Tensor] = None
        self._oracle_labels_list: List[str] = []
        self._discovered_labels: Set[str] = set()
        self._discovered_novel: Set[str] = set()
        self.clip_cp_obstacle_map: CLIPCPObstacleMap = None
        if self.use_clip_argmax_obstacle_map or self.use_clip_cp_obstacle_map:
            from eval.semantic_collision import (
                _SEMANTIC_SAFETY_RADIUS_CELLS, _NON_OBSTACLE_LABELS, _OUTDOOR_COCO_LABELS,
                normalize_semantic_label,
            )
            from vision_models.coco_classes import COCO_CLASSES
            self._safety_dict = _SEMANTIC_SAFETY_RADIUS_CELLS
            _radius_scale = float(getattr(config.mapping, "clip_cp_safety_radius_scale", 1.0))
            _radius_add = int(getattr(config.mapping, "clip_cp_safety_radius_add", 0))

            self._open_vocab = bool(getattr(config.mapping, "clip_cp_open_vocab", False))
            if self._open_vocab:
                _oracle_str = str(getattr(config.mapping, "clip_cp_oracle_labels", ""))
                if _oracle_str:
                    _expert_labels = [
                        normalize_semantic_label(s.strip())
                        for s in _oracle_str.split(",") if s.strip()
                    ]
                else:
                    _expert_labels = ["shower", "cabinet", "chest of drawers", "table", "tv"]
                for lbl in _expert_labels:
                    if lbl not in _SEMANTIC_SAFETY_RADIUS_CELLS:
                        print(f"[OPEN-VOCAB] WARNING: expert label '{lbl}' not in safety dict, skipping")
                        continue
                    raw_r = _SEMANTIC_SAFETY_RADIUS_CELLS[lbl]
                    eff_r = max(1, int(round(raw_r * _radius_scale)) + _radius_add)
                    self._oracle_dict[lbl] = eff_r

                # Open-vocab: start with empty obstacle dictionary
                _cp_labels = []
                _cp_radii = []
            elif bool(getattr(config.mapping, "clip_cp_use_mp3d_labels", False)):
                from eval.dataset_utils.mp3d_dataset import MP3D_GOAL_CATEGORIES
                _seen: Set[str] = set()
                _cp_labels = []
                for _raw in MP3D_GOAL_CATEGORIES:
                    _nl = normalize_semantic_label(_raw)
                    if _nl not in _seen and _nl not in _NON_OBSTACLE_LABELS:
                        _seen.add(_nl)
                        _cp_labels.append(_nl)
                _cp_radii = [
                    max(1, int(round(_SEMANTIC_SAFETY_RADIUS_CELLS[l] * _radius_scale)) + _radius_add)
                    if l in _SEMANTIC_SAFETY_RADIUS_CELLS else -1
                    for l in _cp_labels
                ]
                print(f"[MP3D-LABELS] Using {len(_cp_labels)} MP3D categories: {_cp_labels}")
                print(f"[MP3D-LABELS] Safety-dict labels: "
                      f"{[l for l in _cp_labels if l in _SEMANTIC_SAFETY_RADIUS_CELLS]}")
            else:
                _cp_labels = list(_SEMANTIC_SAFETY_RADIUS_CELLS.keys())
                _cp_radii = [max(1, int(round(_SEMANTIC_SAFETY_RADIUS_CELLS[l] * _radius_scale)) + _radius_add)
                             for l in _cp_labels]

            # Use GCLIP text features for obstacle CP if dual-model enabled
            _cp_feat_model = self.gclip_model if self.gclip_model is not None else self.model
            if _cp_labels:
                _cp_text_feats = _cp_feat_model.get_text_features(
                    [f"a {l}" for l in _cp_labels]
                ).to(self.one_map.map_device)
            else:
                _feat_dim = self.gclip_model.feature_dim if self.gclip_model is not None else self.model.feature_dim
                _cp_text_feats = torch.zeros(0, _feat_dim).to(self.one_map.map_device)

            # Precompute text features for oracle-known labels
            if self._open_vocab and self._oracle_dict:
                self._oracle_labels_list = list(self._oracle_dict.keys())
                self._oracle_text_feats = _cp_feat_model.get_text_features(
                    [f"a {l}" for l in self._oracle_labels_list]
                ).to(self.one_map.map_device)
                print(
                    f"\033[1;36m[OPEN-VOCAB] Initial obstacle dictionary: EMPTY\033[0m",
                    flush=True,
                )
                print(
                    f"\033[1;36m[OPEN-VOCAB] Expert knowledge (to discover): "
                    f"{self._oracle_labels_list}\033[0m", flush=True,
                )

            # Argmax mode: pass COCO + NON_OBSTACLE as background competitors
            # Margin-OACP mode: also needs BG features for margin computation
            _cp_use_margin = bool(getattr(config.mapping, "clip_cp_use_margin", False))
            _bg_text_feats = None
            _bg_labels = []
            if self.use_clip_argmax_obstacle_map or _cp_use_margin:
                _obs_set = set(_cp_labels)
                _bg_labels = [l for l in list(COCO_CLASSES) + list(_NON_OBSTACLE_LABELS)
                              if l not in _obs_set and l not in _OUTDOOR_COCO_LABELS]
                _bg_text_feats = _cp_feat_model.get_text_features(
                    [f"a {l}" for l in _bg_labels]
                ).to(self.one_map.map_device)
            self.clip_cp_obstacle_map = CLIPCPObstacleMap(
                n_cells=self.one_map.n_cells,
                cell_size=self.one_map.cell_size,
                threshold=float(getattr(config.mapping, "clip_cp_threshold", 0.05)),
                use_oacp=bool(getattr(config.mapping, "clip_cp_use_oacp", False)),
                target_coverage=float(getattr(config.mapping, "clip_cp_target_coverage", 0.9)),
                window_size=int(getattr(config.mapping, "clip_cp_window_size", 200)),
                gamma=float(getattr(config.mapping, "clip_cp_gamma", 0.05)),
                initial_alpha=float(getattr(config.mapping, "clip_cp_initial_alpha", 0.5)),
                use_margin_score=_cp_use_margin,
                max_seeds_per_label=int(getattr(config.mapping, "clip_cp_max_seeds_per_label", 0)),
                temporal_persistence=int(getattr(config.mapping, "clip_cp_temporal_persistence", 0)),
                argmax_confidence=float(getattr(config.mapping, "clip_cp_argmax_confidence", 0.0)),
                use_softmax_score=bool(getattr(config.mapping, "clip_cp_use_softmax_score", False)),
                softmax_recompute=bool(getattr(config.mapping, "clip_cp_softmax_recompute", False)),
            )
            self.clip_cp_obstacle_map.set_text_features(
                _cp_text_feats, _cp_labels, _cp_radii,
                bg_text_features=_bg_text_feats,
                bg_labels=_bg_labels if (self.use_clip_argmax_obstacle_map or _cp_use_margin) else None,
            )
            if self._open_vocab:
                self.clip_cp_obstacle_map.save_initial_label_state()

        self.use_yolo_cp_obstacle_map = bool(getattr(config.mapping, "use_yolo_cp_obstacle_map", False))
        self.yolo_cp_obstacle_map: YOLOCPObstacleMap = None
        if self.use_yolo_cp_obstacle_map:
            self.yolo_cp_obstacle_map = YOLOCPObstacleMap(
                n_cells=self.one_map.n_cells,
                cell_size=self.one_map.cell_size,
                hfov_deg=90.0,
                max_depth=float(getattr(config.mapping, "yolo_cp_frustum_max_depth", 5.0)),
                threshold=float(getattr(config.mapping, "yolo_cp_threshold", 0.5)),
                target_coverage=float(getattr(config.mapping, "yolo_cp_target_coverage", 0.9)),
                window_size=int(getattr(config.mapping, "yolo_cp_window_size", 200)),
            )

        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)
        self.previous_sims = None

        # Frontier and POIs
        self.nav_goals = []
        self.blacklisted_nav_goals = []
        self.artificial_obstacles = []

        self.last_nav_goal = None
        self.last_pose = None
        self.saw_left = False
        self.saw_right = False

        self.first_obs = True
        self.similar_points = None
        self.similar_scores = None
        self.object_detected = False
        self.chosen_detection = None
        self.is_goal_path = False
        self.navigation_scores = np.zeros_like(self.semantic_navigable_map, dtype=np.float32)
        self.path = None
        self.is_spot = type(config.controller) == SpotControllerConf
        self.initializing = True

        self._oracle: Optional[ObstacleOracle] = None
        self._use_gdino_oracle = bool(getattr(config.mapping, "use_gdino_oracle", False))
        if self._use_gdino_oracle and (self.use_clip_cp_obstacle_map or self.use_clip_argmax_obstacle_map):
            from eval.semantic_collision import _SEMANTIC_SAFETY_RADIUS_CELLS
            _gdino_labels = list(_SEMANTIC_SAFETY_RADIUS_CELLS.keys())
            self._oracle = GroundingDINOOracle(
                safety_dict=_SEMANTIC_SAFETY_RADIUS_CELLS,
                n_cells=self.one_map.n_cells,
                cell_size=self.one_map.cell_size,
                obstacle_labels=_gdino_labels,
                confidence_threshold=float(getattr(config.mapping, "gdino_confidence", 0.3)),
            )
            print(
                f"\033[1;35m[GDINO-ORACLE] Using Grounding DINO as calibration oracle "
                f"(conf={float(getattr(config.mapping, 'gdino_confidence', 0.3))}, "
                f"labels={_gdino_labels})\033[0m",
                flush=True,
            )
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0

        self.percentile_exploitation = config.planner.percentile_exploitation
        self.frontier_depth = int(config.planner.frontier_depth / self.one_map.cell_size)
        self.no_nav_radius = int(config.planner.no_nav_radius / self.one_map.cell_size)
        self.max_detect_distance = int(config.planner.max_detect_distance / self.one_map.cell_size)
        self.obstcl_kernel_size = int(config.planner.obstcl_kernel_size / self.one_map.cell_size)
        self.min_goal_dist = int(config.planner.min_goal_dist / self.one_map.cell_size)

        self.path_id = 0
        self.filter_detections_depth = config.planner.filter_detections_depth
        self.consensus_filtering = config.planner.consensus_filtering

        self.log = config.log_rerun
        self.allow_replan = config.planner.allow_replan
        self.use_frontiers = config.planner.use_frontiers
        self.allow_far_plan = config.planner.allow_far_plan

        # For the closed-vocabulary object detector, not needed for OneMap
        self.class_map = {}
        self.class_map["chair"] = "chair"
        self.class_map["tv_monitor"] = "tv"
        self.class_map["tv"] = "tv"
        self.class_map["plant"] = "potted plant"
        self.class_map["potted plant"] = "potted plant"
        self.class_map["sofa"] = "couch"
        self.class_map["couch"] = "couch"
        self.class_map["bed"] = "bed"
        self.class_map["toilet"] = "toilet"

    def reset(self):
        self.query_text = ["Other."]
        self.query_text_features = self.model.get_text_features(self.query_text).to(self.one_map.map_device)
        self.previous_sims = None
        self.similar_points = None
        self.similar_scores = None
        self.object_detected = False
        self.chosen_detection = None
        self.last_pose = None
        self.stuck_at_nav_goal_counter = 0
        self.stuck_at_cell_counter = 0
        self.is_goal_path = False
        self.path = None
        self.path_id = 0
        self.initializing = True
        self.one_map.reset()
        if self.use_clip_semantic_nav_map and self.semantic_map_updater is not None:
            self.semantic_map_updater.reset(self.one_map.navigable_map)
        self.semantic_navigable_map = self.one_map.navigable_map.copy()
        if self.use_yolo_obstacle_map and self.yolo_obstacle_map is not None:
            self.yolo_obstacle_map.reset()
        if (self.use_clip_argmax_obstacle_map or self.use_clip_cp_obstacle_map) and self.clip_cp_obstacle_map is not None:
            self.clip_cp_obstacle_map.reset()
            if self._open_vocab:
                self.clip_cp_obstacle_map.restore_initial_label_state()
                self._discovered_labels = set()
                self._discovered_novel = set()
        self._delay_buffer.clear()
        self._step_count = 0
        if isinstance(self._oracle, GroundingDINOOracle):
            self._oracle.reset()
        if self.use_yolo_cp_obstacle_map and self.yolo_cp_obstacle_map is not None:
            self.yolo_cp_obstacle_map.reset()
        self.navigation_scores = np.zeros_like(self.semantic_navigable_map, dtype=np.float32)
        self.first_obs = True
        self.cyclic_checker = CyclicChecker()
        self.cyclic_detect_checker = CyclicDetectChecker()
        self.points_of_interest = []
        self.nav_goals = []
        self.blacklisted_nav_goals = []
        self.artificial_obstacles = []

    def set_camera_matrix(self,
                          camera_matrix: np.ndarray
                          ) -> None:
        self.one_map.set_camera_matrix(camera_matrix)
        if isinstance(self._oracle, GroundingDINOOracle):
            self._oracle.set_camera_intrinsics(
                float(self.one_map.fx), float(self.one_map.fy),
                float(self.one_map.cx), float(self.one_map.cy),
            )

    def set_oracle(self, oracle: ObstacleOracle) -> None:
        self._oracle = oracle

    def set_gt_label_map(self, label_map: np.ndarray, labels: List[str]) -> None:
        """Backward-compat: create/update a GTLabelMapOracle internally."""
        if self._oracle is None:
            from eval.semantic_collision import _SEMANTIC_SAFETY_RADIUS_CELLS
            self._oracle = GTLabelMapOracle(
                safety_dict=_SEMANTIC_SAFETY_RADIUS_CELLS,
                cell_size=self.one_map.cell_size,
            )
        if isinstance(self._oracle, GTLabelMapOracle):
            self._oracle.set_gt_label_map(label_map, labels)

    def set_query(self,
                  txt: List[str]
                  ) -> None:
        """
        Sets the query text
        :param txt: List of strings
        :return:
        """
        txt = [self.class_map.get(t, t) for t in txt]
        if txt != self.query_text:
            print(f"Setting query to {txt}")
            self.query_text = txt
            self.query_text_features = self.model.get_text_features(["a " + self.query_text[0]]).to(
                self.one_map.map_device)
            self.previous_sims = None
            self.one_map.reset_checked_map()
            self.detector.set_classes(self.query_text)
            if self.use_yolo_obstacle_map and self.yolo_obstacle_map is not None:
                self.yolo_obstacle_map.set_query_label(self.query_text[0])
            self.object_detected = False
            if self.gclip_model is not None and self._cp_detection_gate:
                self._gclip_query_feat = self.gclip_model.get_text_features(
                    [f"a {self.query_text[0]}"]
                ).to(self.one_map.map_device)
            self.get_map(False)

    def get_active_navigable_map(self, robot_pos: np.ndarray = None) -> np.ndarray:
        if self.use_clip_semantic_nav_map and self.semantic_navigable_map is not None:
            base = self.semantic_navigable_map
        else:
            base = self.one_map.navigable_map
        if self.use_yolo_obstacle_map and self.yolo_obstacle_map is not None:
            base = self.yolo_obstacle_map.apply_to_navigable_map(base)
        # Goal-zone anti-STUCK: skip CP obstacle map when robot is within 1.5m of detected goal.
        # Prevents the obstacle ring around the target object from blocking the final approach.
        _goal_zone_cells = 15  # 1.5m at 0.1m/cell
        _near_goal = (
            self._cp_goal_zone_disable
            and robot_pos is not None
            and self.object_detected
            and self.chosen_detection is not None
            and np.linalg.norm(robot_pos - np.array(self.chosen_detection)) < _goal_zone_cells
        )
        if (self.use_clip_argmax_obstacle_map or self.use_clip_cp_obstacle_map) and self.clip_cp_obstacle_map is not None:
            if not _near_goal:
                base = self.clip_cp_obstacle_map.apply_to_navigable_map(base)
        if self.use_yolo_cp_obstacle_map and self.yolo_cp_obstacle_map is not None:
            base = self.yolo_cp_obstacle_map.apply_to_navigable_map(base)
        return base

    @torch.no_grad()
    def _update_semantic_navigable_map(self) -> None:
        if not self.use_clip_semantic_nav_map:
            self.semantic_navigable_map = self.one_map.navigable_map.copy()
            return

        if self.semantic_map_updater is None or self.semantic_text_features is None:
            self.semantic_navigable_map = self.one_map.navigable_map.copy()
            return

        updated_all = self.one_map.updated_mask
        if updated_all.max() == 0:
            self.semantic_navigable_map = self.semantic_map_updater.update(
                self.one_map.navigable_map,
                np.zeros_like(self.one_map.navigable_map, dtype=bool),
                self.semantic_map_updater.seed_label_map,
            )
            return

        changed_mask = updated_all.detach().cpu().numpy().astype(bool)
        label_ids_map = self.semantic_map_updater.seed_label_map.copy()
        label_ids_map[changed_mask] = 0

        valid_updated = updated_all & (self.one_map.confidence_map > 0)
        if valid_updated.max() > 0:
            feats = self.one_map.feature_map[valid_updated, :]
            text_features = self.semantic_text_features.to(feats.device)
            if text_features.dtype != feats.dtype:
                text_features = text_features.to(feats.dtype)
            sims = feats @ text_features.T
            max_sims, argmax_ids = torch.max(sims, dim=1)
            pred_ids = (argmax_ids + 1).detach().cpu().numpy().astype(np.uint16)
            if self.clip_semantic_sim_threshold > 0.0:
                below_thresh = (max_sims < self.clip_semantic_sim_threshold).detach().cpu().numpy()
                pred_ids[below_thresh] = 0  # 0 = no label → not blocked
            valid_mask_np = valid_updated.detach().cpu().numpy().astype(bool)
            label_ids_map[valid_mask_np] = pred_ids
            if self.debug_collector is not None:
                max_sims_np = max_sims.detach().cpu().numpy().astype(np.float32)
                self.debug_collector.record(valid_mask_np, pred_ids, self.semantic_label_config.labels, max_sims_np)

        self.semantic_navigable_map = self.semantic_map_updater.update(
            self.one_map.navigable_map,
            changed_mask,
            label_ids_map,
        )

    def get_path(self
                 ) -> Union[np.ndarray, str]:
        if not self.path:
            return None
        if not self.object_detected:
            if self.saw_left:
                return "L"
            if self.saw_right:
                return "R"
        return self.path[min(self.path_id, len(self.path)):]

    def compute_best_path(self,
                          start: np.ndarray) -> None:
        """
        Computes the best path from the start point to a point on a frontier, or a point of interest
        :param start: start point as [X, Y]
        :return:
        """
        self.path_id = 0
        if not self.object_detected:
            # We are exploring
            if len(self.nav_goals) == 0:
                if not self.initializing:
                    self.one_map.reset_checked_map()  # We need new points of interest
                    self.compute_frontiers_and_POIs(start[0], start[1])
            # If we still have no nav goals, we can't plan anything
            if len(self.nav_goals) == 0:
                return
            self.initializing = False
            self.is_goal_path = False
            self.path = None
            while self.path is None and len(self.nav_goals) > 0:
                best_idx = None
                if len(self.nav_goals) == 1:
                    top_two_vals = tuple((self.nav_goals[0].get_score(), self.nav_goals[0].get_score()))
                else:
                    top_two_vals = tuple((self.nav_goals[0].get_score(), self.nav_goals[1].get_score()))

                # We have a frontier and we need to consider following up on that
                curr_index = None
                if self.last_nav_goal is not None:
                    last_pt = self.last_nav_goal.get_descr_point()
                    for nav_id in range(len(self.nav_goals)):
                        if np.array_equal(last_pt, self.nav_goals[nav_id].get_descr_point()):
                            # frontier still exists!
                            curr_index = nav_id
                            break
                    if curr_index is None:
                        closest_index = closest_point_within_threshold(self.nav_goals, last_pt,
                                                                       0.5 / self.one_map.cell_size)
                        if closest_index != -1:
                            curr_index = closest_index
                            # there is a close point to the previous frontier that we could consider instead
                    if curr_index is not None:
                        curr_value = self.nav_goals[curr_index].get_score()
                        if curr_value + 0.01 > self.last_nav_goal.get_score():
                            best_idx = curr_index
                if best_idx is None:
                    # Select the current best nav_goal, and check for cyclic
                    for nav_id in range(len(self.nav_goals)):
                        nav_goal = self.nav_goals[nav_id]
                        cyclic = self.cyclic_checker.check_cyclic(start, nav_goal.get_descr_point(), top_two_vals)
                        if cyclic:
                            continue
                        best_idx = nav_id
                        break
                best_nav_goal = self.nav_goals[best_idx]
                self.cyclic_checker.add_state_action(start, best_nav_goal.get_descr_point(), top_two_vals)
                if isinstance(best_nav_goal, Frontier):
                    # NOTE Allow more aggressive planning through unknown regions
                    self.path = Planning.compute_to_goal(start, self.get_active_navigable_map(),
                                                         (self.one_map.confidence_map > 0).cpu().numpy(),
                                                         best_nav_goal.get_descr_point(),
                                                         self.obstcl_kernel_size, 2)
                elif isinstance(best_nav_goal, Cluster):
                    # NOTE Allow more aggressive planning through unknown regions
                    self.path = Planning.compute_to_goal(start, self.get_active_navigable_map(),
                                                         (self.one_map.confidence_map > 0).cpu().numpy(),
                                                         best_nav_goal.get_descr_point(),
                                                         self.obstcl_kernel_size, 4)
                if self.path is None:
                    # remove the nav goal from the list, we don't know how to reach it
                    self.nav_goals.pop(best_idx)
            if self.path is None:
                if not self.initializing:
                    if self.log:
                        rr.log("path_updates", rr.TextLog(f"Resetting checked map as no path found."))
                    self.one_map.reset_checked_map()
            if self.last_nav_goal is not None and not np.array_equal(self.last_nav_goal.get_descr_point(),
                                                                     best_nav_goal.get_descr_point()):
                self.stuck_at_nav_goal_counter = 0
            else:
                if self.last_pose is not None:
                    if self.path is not None:
                        if len(self.path) < 5 and self.last_pose[0] == start[0] and self.last_pose[1] == start[1]:
                            self.stuck_at_nav_goal_counter += 1
            if self.stuck_at_nav_goal_counter > 10:
                # We probably are trying to reach an unreachable goal, for instance a frontier to the void in habitat
                self.blacklisted_nav_goals.append(best_nav_goal.get_descr_point())
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"Frontier at position {best_nav_goal.get_descr_point()[0]}"
                                                      f",{best_nav_goal.get_descr_point()[1]} invalid."))
            self.last_nav_goal = best_nav_goal

            if self.path:
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"Computed path of length {len(self.path)}"))
                    rr.log("map/path", rr.LineStrips2D(self.path, colors=np.repeat(np.array([0, 0, 255])[np.newaxis, :],
                                                                                   len(self.path), axis=0)))
        else:
            # We go to an object
            if np.linalg.norm(start - self.chosen_detection) < self.max_detect_distance:
                self.path = [start] * 5
                # We are close to the object, we don't need to move
                return
            self.path = Planning.compute_to_goal(start, self.get_active_navigable_map(robot_pos=start),
                                                 (self.one_map.confidence_map > 0).cpu().numpy(),
                                                 self.chosen_detection,
                                                 self.obstcl_kernel_size, self.min_goal_dist)
            self.is_goal_path = True
            if self.path and len(self.path) > 0:
                if self.log:
                    rr.log("map/path", rr.LineStrips2D(self.path, colors=np.repeat(np.array([0, 255, 0])[np.newaxis, :],
                                                                                   len(self.path), axis=0)))
                    rr.log("path_updates",
                           rr.TextLog(f"Path to object {self.query_text[0]} of length {len(self.path)} computed."))
            else:
                self.object_detected = False
                if self.log:
                    rr.log("path_updates", rr.TextLog(f"No path to object {self.query_text[0]} found."))

    def compute_frontiers_and_POIs(self, px, py):
        """
        Computes the frontiers (at the border from fully explored to confidence > 0),
        and points of interest (high similarity regions within the fully explored, but not checked map)
        :return:
        """
        self.nav_goals = []
        if self.previous_sims is not None:
            # Compute the frontiers
            frontiers, unexplored_map, largest_contour = detect_frontiers(
                self.get_active_navigable_map().astype(np.uint8),
                self.one_map.fully_explored_map.astype(np.uint8),
                self.one_map.confidence_map > 0,
                int(1.0 * ((
                                   self.one_map.n_cells /
                                   self.one_map.size) ** 2)))

            # moreover we compute points of interest. These are high similarity regions within the fully explored,
            # but not checked map
            # For that we make use of the cluster_high_similarity_regions function, and project the points to the
            # navigable map
            if self.previous_sims is not None:
                adjusted_score = self.previous_sims[0].cpu().numpy() + 1.0  # only positive scores
                map_def = self.previous_sims[0].numpy()
                normalized_map = (map_def - map_def.min()) / (map_def.max() - map_def.min())
                clusters = cluster_high_similarity_regions(normalized_map,
                                                           (self.one_map.confidence_map > 0.0).cpu().numpy())
                for cluster in clusters:
                    cluster.compute_score(adjusted_score)
            else:
                clusters = []

            if clusters:
                centers = np.array([c.center for c in clusters])          # [K, 2]
                explored_ok = self.one_map.fully_explored_map[centers[:, 0], centers[:, 1]]
                not_checked  = ~self.one_map.checked_map[centers[:, 0], centers[:, 1]]

                if len(self.blacklisted_nav_goals) > 0:
                    pts = np.array([c.get_descr_point() for c in clusters])  # [K, 2]
                    bl  = np.asarray(self.blacklisted_nav_goals)              # [B, 2]
                    blacklisted = np.any(np.all(pts[:, None, :] == bl[None, :, :], axis=2), axis=1)
                else:
                    blacklisted = np.zeros(len(clusters), dtype=bool)

                for i, cluster in enumerate(clusters):
                    if blacklisted[i]:
                        continue
                    contour_ok = (largest_contour is None or
                                  cv2.pointPolygonTest(largest_contour,
                                                       cluster.center.astype(float),
                                                       measureDist=True) > -15.0)
                    if (contour_ok or explored_ok[i]) and not_checked[i]:
                        self.nav_goals.append(cluster)
            if self.log:
                cluster_max_similarity = np.zeros_like(self.previous_sims[0])

                # Fill each cluster with its maximum similarity value
                min_c = np.min([cluster.get_score() for cluster in clusters])
                max_c = np.max([cluster.get_score() for cluster in clusters])
                for cluster in clusters:
                    cluster_pts = cluster.points
                    score = (cluster.get_score() - min_c) / (max_c - min_c)
                    cluster_max_similarity[cluster_pts[:, 0], cluster_pts[:, 1]] = score
                log_map_rerun(cluster_max_similarity, path="map/similarity_th2")

            if self.log:
                log_map_rerun(unexplored_map, path="map/unexplored")

            frontiers = [f[:, :, ::-1] for f in frontiers]  # need to flip coords for some reason
            adjusted_score_frontier = adjusted_score.copy()

            # set the score of the fully explored map to 0 for the frontiers
            valid_frontiers_mask = np.zeros((len(frontiers),), dtype=bool)

            confidence_np = (self.one_map.confidence_map > 0).cpu().numpy()
            bl_arr = np.asarray(self.blacklisted_nav_goals) if len(self.blacklisted_nav_goals) > 0 else None
            for i_frontier, frontier in enumerate(frontiers):
                frontier_mp = np.round(get_frontier_midpoint(frontier).astype(np.uint32))
                if bl_arr is not None and np.any(np.all(frontier_mp == bl_arr, axis=1)):
                    continue
                score, n_els, best_reachable, reachable_area = Planning.compute_reachable_area_score(
                    frontier_mp,
                    confidence_np,
                    adjusted_score_frontier,
                    self.frontier_depth)
                valid_frontiers_mask[i_frontier] = True
                self.nav_goals.append(
                    Frontier(frontier_midpoint=frontier_mp, points=frontier, frontier_score=score))

            if self.log:
                if len(self.nav_goals) > 0:
                    pts = np.array([nav_goal.get_descr_point() for nav_goal in self.nav_goals])
                    scores = np.array([nav_goal.get_score() for nav_goal in self.nav_goals])
                    rr.log("map/frontiers",
                           rr.Points2D(pts, colors=np.flip(monochannel_to_inferno_rgb(scores), axis=-1),
                                       radii=[1] * pts.shape[0]))

            self.nav_goals = sorted(self.nav_goals, key=lambda x: x.get_score(), reverse=True)

    def add_data(self,
                 image: np.ndarray,
                 depth: np.ndarray,
                 odometry: np.ndarray,
                 ) -> bool:
        """
        Adds data to the navigator
        :param image: RGB image of dimension [C, H, W]
        :param depth: depth image of dimension [H, W]
        :param odometry: 4x4 transformation matrix from camera to world
        :return: boolean indicating if the episode is over
        """
        self._step_count += 1
        odometry = odometry.astype(np.float32)
        x = odometry[0, 3]
        y = odometry[1, 3]
        yaw = np.arctan2(odometry[1, 0], odometry[0, 0])

        px, py = self.one_map.metric_to_px(x, y)
        if self.last_pose:
            if px == self.last_pose[0] and py == self.last_pose[1] and abs(yaw - self.last_pose[2]) < 0.001:
                if self.path is not None:
                    self.stuck_at_cell_counter += 1
            else:
                self.stuck_at_cell_counter = 0
        if self.stuck_at_cell_counter > 5:
            # we are stuck we need to add an obstacle right in front of us!
            dx = np.cos(yaw)
            dy = np.sin(yaw)

            # Round to nearest integer to get facing direction
            facing_dx = round(dx)
            facing_dy = round(dy)

            # Calculate coordinates of facing cell
            facing_px = px + facing_dx
            facing_py = py + facing_dy
            self.artificial_obstacles.append((facing_px, facing_py))

        if not self.one_map.camera_initialized:
            warnings.warn("Camera matrix not set, please set camera matrix first")
            return

        # Check if RGB or BGR correct?
        img_hwc = image.transpose(1, 2, 0)
        detections = self.detector.detect(img_hwc)
        if self.one_map.camera_initialized and hasattr(self.detector, "get_obstacle_detections"):
            if self.use_yolo_obstacle_map and self.yolo_obstacle_map is not None:
                obstacle_dets = self.detector.get_obstacle_detections()
                self.yolo_obstacle_map.update(
                    obstacle_dets, depth, odometry,
                    self.one_map.fx, self.one_map.fy, self.one_map.cx, self.one_map.cy,
                    img_hwc.shape[0], img_hwc.shape[1],
                )
            if self.use_yolo_cp_obstacle_map and self.yolo_cp_obstacle_map is not None and \
                    hasattr(self.detector, "get_obstacle_detections_with_conf"):
                dets_with_conf = self.detector.get_obstacle_detections_with_conf()
                self.yolo_cp_obstacle_map.update(
                    dets_with_conf, depth, odometry,
                    self.one_map.fx, self.one_map.fy, self.one_map.cx, self.one_map.cy,
                    img_hwc.shape[0], img_hwc.shape[1],
                )
        a = time.time()
        image_features = self.model.get_image_features(image[np.newaxis, ...]).squeeze(0)
        b = time.time()
        self.one_map.update(image_features, depth, odometry, self.artificial_obstacles)
        self._update_semantic_navigable_map()

        # GCLIP dual-model: extract GCLIP features and update separate feature map
        if self.gclip_model is not None:
            gclip_features = self.gclip_model.get_image_features(image[np.newaxis, ...]).squeeze(0)
            self.one_map.update_gclip(gclip_features, depth, odometry)

        # CLIP CP obstacle map: use GCLIP features if available, else ConvNeXt
        if (self.use_clip_argmax_obstacle_map or self.use_clip_cp_obstacle_map) and self.clip_cp_obstacle_map is not None:
            if self.gclip_model is not None:
                _cp_mask = self.one_map.updated_mask_gclip
                _cp_feats = self.one_map.feature_map_gclip
            else:
                _cp_mask = self.one_map.updated_mask
                _cp_feats = self.one_map.feature_map
            self.clip_cp_obstacle_map.update(_cp_mask, _cp_feats)
            # Grounding DINO oracle: update label map from current frame
            if isinstance(self._oracle, GroundingDINOOracle):
                self._oracle.update(img_hwc, depth, odometry)
            # Oracle: discover labels (open-vocab) and optionally calibrate ACI
            if self._oracle is not None:
                oracle_detections = self._oracle.detect(agent_cell=(px, py))
                n_cells_this_step = 0
                for det in oracle_detections:
                    cx, cy = det.cell_x, det.cell_y
                    self._delay_buffer.append((
                        self._step_count, cx, cy, det.label,
                        _cp_feats[cx, cy, :].clone(),
                        self.clip_cp_obstacle_map.threshold,
                    ))
                    n_cells_this_step += 1
                # Track cells-per-step EMA for adaptive gamma
                _ema_alpha = 0.01
                self._cells_per_step_ema = (
                    (1 - _ema_alpha) * self._cells_per_step_ema
                    + _ema_alpha * n_cells_this_step
                )
                # Drain entries whose oracle-publication step has arrived.
                # Collect all drainable entries, grouped by their observation sim step.
                _drain_batches = {}  # obs_step -> list of (cx, cy, label, feat, obs_tau)
                while self._delay_buffer and (
                    self._step_count - self._delay_buffer[0][0]
                ) >= self._calibration_delay:
                    obs_step, cx_d, cy_d, lbl_d, feat_d, tau_d = self._delay_buffer.popleft()
                    # Open-vocab discovery (immediate per entry)
                    if self._open_vocab:
                        if (lbl_d in self._oracle_dict
                                and lbl_d not in self._discovered_labels):
                            o_idx = self._oracle_labels_list.index(lbl_d)
                            o_feat = self._oracle_text_feats[o_idx]
                            o_radius = self._oracle_dict[lbl_d]
                            self.clip_cp_obstacle_map.expand_label(lbl_d, o_radius, o_feat)
                            self._discovered_labels.add(lbl_d)
                            dist_m = o_radius * self.one_map.cell_size
                            n_known = len(self.clip_cp_obstacle_map._labels)
                            print(
                                f"\n\033[1;33m{'='*72}\n"
                                f"  [OPEN-VOCAB DISCOVERY] External experts first find '{lbl_d}'!\n"
                                f"  Expert suggests its semantic safety distance as {dist_m:.2f} meter.\n"
                                f"  Safety dictionary expanded: {n_known} categories now known.\n"
                                f"{'='*72}\033[0m\n",
                                flush=True,
                            )
                        elif (lbl_d not in self.clip_cp_obstacle_map._labels
                                and lbl_d not in self._discovered_novel):
                            self._discovered_novel.add(lbl_d)
                            print(
                                f"\n\033[1;32m{'='*72}\n"
                                f"  [OPEN-VOCAB DISCOVERY] External experts first find '{lbl_d}'!\n"
                                f"  Expert confirms no semantic safety concern.\n"
                                f"{'='*72}\033[0m\n",
                                flush=True,
                            )
                    if lbl_d in self._safety_dict:
                        _drain_batches.setdefault(obs_step, []).append(
                            (cx_d, cy_d, lbl_d, feat_d, tau_d)
                        )
                # Per-sim-step aggregated ACI update: one α update per observation step.
                for obs_step in sorted(_drain_batches):
                    batch = _drain_batches[obs_step]
                    if self._calibration_delay > 0:
                        self.clip_cp_obstacle_map.calibrate_aci_batch(
                            [(f, l, (cx, cy), t) for cx, cy, l, f, t in batch]
                        )
                    else:
                        # delay=0: per-cell updates with live τ (original behavior)
                        for cx_d, cy_d, lbl_d, feat_d, tau_d in batch:
                            self.clip_cp_obstacle_map.calibrate_aci(
                                feat_d, lbl_d, cell_xy=(cx_d, cy_d),
                            )
            # Legacy OACP calibration via YOLO detections
            if self.use_yolo_obstacle_map and self.yolo_obstacle_map is not None:
                for label, px, py in self.yolo_obstacle_map.latest_projected:
                    self.clip_cp_obstacle_map.calibrate(
                        _cp_feats[px, py, :], label
                    )
        c = time.time()
        self.get_map(False)
        d = time.time()
        detected_just_now = False
        start = np.array([px, py])
        old_path = self.path.copy() if self.path else self.path
        old_id = self.path_id
        if self.first_obs:
            self.one_map.confidence_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.one_map.checked_conf_map[px - 10:px + 10, py - 10:py + 10] += 10
            self.first_obs = False
        last_saw_left = self.saw_left
        last_saw_right = self.saw_right
        self.saw_left = False
        self.saw_right = False
        if len(detections["boxes"]) > 0:
            # wants rgb
            self.sam_predictor.set_image(image.transpose(1, 2, 0))
            for area, confidence in zip(detections["boxes"], detections['scores']):
                if self.log:
                    rr.log("camera/detection", rr.Boxes2D(array_format=rr.Box2DFormat.XYXY, array=area))
                    rr.log("object_detections", rr.TextLog(f"Object {self.query_text[0]} detected"))

                chosen_detection = (
                    int((area[3] + area[1]) / 2), int((area[2] + area[0]) / 2))
                masks, _, _ = self.sam_predictor.predict(point_coords=None,
                                                         point_labels=None,
                                                         box=np.array(area)[None, :],
                                                         multimask_output=False, )
                # Project the points where the mask is one
                mask_ids = np.argwhere(masks[0] & (depth != 0))
                depth_detection = depth[chosen_detection[0], chosen_detection[1]]

                depths = depth[mask_ids[:, 0], mask_ids[:, 1]]

                if not self.filter_detections_depth or depth_detection < 2.5:
                    y_world = -(mask_ids[:, 1] - self.one_map.cx) * depths / self.one_map.fx
                    x_world = depths
                    r = np.array([[np.cos(yaw), -np.sin(yaw)],
                                  [np.sin(yaw), np.cos(yaw)]])
                    x_rot, y_rot = np.dot(r, np.stack((x_world, y_world)))
                    x_rot += odometry[0, 3]
                    y_rot += odometry[1, 3]

                    x_id = ((x_rot / self.one_map.cell_size)).astype(np.int32) + \
                           self.one_map.map_center_cells[0].item()
                    y_id = ((y_rot / self.one_map.cell_size)).astype(np.int32) + \
                           self.one_map.map_center_cells[1].item()
                    # Clamp to valid range
                    x_id = np.clip(x_id, 0, self.one_map.n_cells - 1)
                    y_id = np.clip(y_id, 0, self.one_map.n_cells - 1)

                    object_valid = True
                    if self.previous_sims is not None:
                        adjusted_score = self.previous_sims[0].cpu().numpy() + 1.0  # only positive scores
                    else:
                        adjusted_score = np.ones((self.one_map.n_cells, self.one_map.n_cells))
                    if self.log:
                        rr.log("map/proj_detect",
                               rr.Points2D(np.stack((x_id, y_id)).T, colors=[[0, 0, 255]], radii=[1]))
                        # log the segmentation mask as rgba
                        rr.log("camera", rr.SegmentationImage(masks[0].astype(np.uint8))
                               )
                    if self.consensus_filtering:
                        top_10 = np.percentile(adjusted_score[self.one_map.confidence_map > 0],
                                               self.percentile_exploitation)
                        top_map = (adjusted_score > top_10).astype(np.uint8)

                        print(top_10)
                        top_map[self.one_map.confidence_map == 0] = 0
                        k = np.ones((7, 7), np.uint8)
                        top_map = cv2.dilate(top_map, k, iterations=1)
                        top_map_projections = top_map[x_id, y_id]
                        if not np.any(top_map_projections):
                            object_valid = False

                        if object_valid:
                            mask = top_map_projections
                            x_masked = x_id[mask == 1]
                            y_masked = y_id[mask == 1]
                            depths_masked = depths[mask == 1]
                            best = np.argmin(depths_masked)

                            if self.object_detected and object_valid:
                                # we already have a goal point and will only update if the current one is better
                                if adjusted_score[x_masked[best], y_masked[best]] < \
                                        adjusted_score[self.chosen_detection[0], self.chosen_detection[1]] * 1.1:
                                    object_valid = False
                            if object_valid:
                                self.chosen_detection = (x_masked[best], y_masked[best])
                    else:
                        best = np.argmin(depths)
                        # NOTE More aggressive reselection of the best point
                        self.chosen_detection = (x_id[best], y_id[best])
                    # W: detection gate — reject if any GCLIP obstacle label outscores query at detection cell
                    if object_valid and self._cp_detection_gate and self.clip_cp_obstacle_map is not None and self._gclip_query_feat is not None:
                        cx, cy = self.chosen_detection
                        feat = self.one_map.feature_map_gclip[cx, cy]
                        if feat.norm() > 1e-6:
                            feat_n = F.normalize(feat.unsqueeze(0), dim=1)
                            obs_feats = self.clip_cp_obstacle_map._text_features.to(feat_n.device)
                            if obs_feats.numel() > 0:
                                if obs_feats.dtype != feat_n.dtype:
                                    obs_feats = obs_feats.to(feat_n.dtype)
                                max_obs_sim = float((feat_n @ obs_feats.T).max())
                                q_feat = self._gclip_query_feat.to(feat_n.device)
                                if q_feat.dtype != feat_n.dtype:
                                    q_feat = q_feat.to(feat_n.dtype)
                                if max_obs_sim > float((feat_n @ q_feat.T).squeeze()):
                                    object_valid = False
                    if object_valid:
                        self.object_detected = True
                        self.compute_best_path(start)
                        if not self.path:
                            self.object_detected = False
                            self.path = old_path
                            self.path_id = old_id
                        else:
                            if self.log:
                                rr.log("path_updates",
                                       rr.TextLog(f"The object {self.query_text[0]} has been detected just now."))
                                rr.log("map/goal_pos",
                                       rr.Points2D([self.chosen_detection], colors=[[0, 255, 0]], radii=[1]))
        elif not self.object_detected:
            self.chosen_detection = None
            self.object_detected = False
        # GCLIP-based stop: when YOLO misses the target, use GCLIP similarity
        # in the agent's neighborhood to detect proximity to query object.
        if not hasattr(self, '_gclip_stop_debug_counter'):
            self._gclip_stop_debug_counter = 0
        if not self.object_detected and self._cp_detection_gate and \
                self.gclip_model is not None and self._gclip_query_feat is not None and \
                hasattr(self.one_map, 'feature_map_gclip'):
            _r = 10  # check 21x21 neighborhood (~2m radius)
            _ax, _ay = int(px), int(py)
            _x0 = max(0, _ax - _r)
            _x1 = min(self.one_map.n_cells, _ax + _r + 1)
            _y0 = max(0, _ay - _r)
            _y1 = min(self.one_map.n_cells, _ay + _r + 1)
            _local_feats = self.one_map.feature_map_gclip[_x0:_x1, _y0:_y1]
            _local_norms = _local_feats.norm(dim=-1)
            _observed = _local_norms > 1e-6
            if _observed.sum() > 20:
                _obs_feats = _local_feats[_observed]
                _obs_feats_n = F.normalize(_obs_feats.float(), dim=1)
                _q = self._gclip_query_feat.to(_obs_feats_n.device).float()
                _sims = (_obs_feats_n @ _q.T).squeeze(-1)
                _top_sim = float(_sims.max())
                _mean_sim = float(_sims.mean())
                # Compare against the GLOBAL mean GCLIP query similarity for reference
                _all_feats = self.one_map.feature_map_gclip
                _all_norms = _all_feats.norm(dim=-1)
                _all_observed = _all_norms > 1e-6
                if _all_observed.sum() > 100:
                    _all_obs = _all_feats[_all_observed]
                    _all_obs_n = F.normalize(_all_obs.float(), dim=1)
                    _all_sims = (_all_obs_n @ _q.T).squeeze(-1)
                    _global_mean = float(_all_sims.mean())
                    _global_top = float(_all_sims.max())
                    # Local neighborhood should be significantly above global mean
                    # AND the local top should be close to the global top
                    _local_above = _mean_sim - _global_mean
                    _top_ratio = _top_sim / max(_global_top, 1e-6)
                    if self._gclip_stop_debug_counter % 100 == 0:
                        print(f"  [GCLIP-stop] local_top={_top_sim:.4f}, local_mean={_mean_sim:.4f}, "
                              f"global_mean={_global_mean:.4f}, global_top={_global_top:.4f}, "
                              f"delta={_local_above:.4f}, top_ratio={_top_ratio:.3f}")
                    self._gclip_stop_debug_counter += 1
                    # Trigger: local mean is well above global, and local top is near global top
                    if _local_above > 0.01 and _top_ratio > 0.85 and _top_sim > 0.25:
                        # Find the cell with max GCLIP query similarity in neighborhood
                        _local_sims_map = torch.full((_x1-_x0, _y1-_y0), -1.0)
                        _local_sims_map[_observed] = _sims
                        _best_local = torch.argmax(_local_sims_map.view(-1))
                        _bx = _best_local // (_y1-_y0) + _x0
                        _by = _best_local % (_y1-_y0) + _y0
                        self.chosen_detection = (int(_bx), int(_by))
                        self.object_detected = True
        if not self.object_detected:
            if self.saw_left:
                self.cyclic_detect_checker.add_state_action(np.array([px, py]), "L")
            elif self.saw_right:
                self.cyclic_detect_checker.add_state_action(np.array([px, py]), "R")
        self.compute_frontiers_and_POIs(*self.one_map.metric_to_px(odometry[0, 3], odometry[1, 3]))
        e = time.time()
        if self.log:
            adjusted_score = self.previous_sims[0].cpu().numpy() + 1.0  # only positive scores

            top_10 = np.percentile(adjusted_score[self.one_map.confidence_map > 0],
                                   self.percentile_exploitation)
            top_map = (adjusted_score > top_10).astype(np.uint8)
            k = np.ones((3, 3), np.uint8)
            top_map = cv2.dilate(top_map, k, iterations=1)
            log_map_rerun(top_map, path="map/similarity_th")
        # Compute the new path
        if self.object_detected:
            if np.linalg.norm(start - self.chosen_detection) <= self.max_detect_distance:
                self.object_detected = False
                return True
            if self.consensus_filtering and self.object_detected and self.previous_sims is not None:
                adjusted_score = self.previous_sims[0].cpu().numpy() + 1.0  # only positive scores

                top_10 = np.percentile(adjusted_score[self.one_map.confidence_map > 0],
                                       self.percentile_exploitation)
                top_map = (adjusted_score > top_10).astype(np.uint8)
                k = np.ones((7, 7), np.uint8)
                top_map = cv2.dilate(top_map, k, iterations=1)
                if not top_map[self.chosen_detection[0], self.chosen_detection[1]]:
                    self.object_detected = False
                    rr.log("path_updates", rr.TextLog("Current path lost similarity."))
        if self.allow_replan:
            self.compute_best_path(start)
        # Path-2: step-level safety guard. Halt in place if upcoming few cells of the
        # current path intersect the latest CP obstacle mask AND the hit cell is
        # outside the goal zone. (Inside the goal zone the planner intentionally
        # ignores CP via zone_disable, so guard must defer there or we'd deadlock.)
        if (self._cp_step_guard
                and (self.use_clip_argmax_obstacle_map or self.use_clip_cp_obstacle_map)
                and self.clip_cp_obstacle_map is not None
                and self.path is not None
                and len(self.path) > self.path_id):
            _mask = self.clip_cp_obstacle_map._obstacle_mask
            _k = self._cp_step_guard_lookahead
            _goal_zone_cells = 15
            _chosen = self.chosen_detection if (self._cp_goal_zone_disable and self.object_detected) else None
            _hit = False
            for _w in self.path[self.path_id:self.path_id + _k]:
                _cx, _cy = int(_w[0]), int(_w[1])
                if not (0 <= _cx < _mask.shape[0] and 0 <= _cy < _mask.shape[1]):
                    continue
                if not _mask[_cx, _cy]:
                    continue
                if _chosen is not None:
                    _dx, _dy = _cx - int(_chosen[0]), _cy - int(_chosen[1])
                    if _dx * _dx + _dy * _dy < _goal_zone_cells * _goal_zone_cells:
                        continue  # inside goal zone: defer to zone_disable
                _hit = True
                break
            if _hit:
                # Halt in place this frame: dummy path of the current cell so the
                # controller produces zero-velocity instead of falling back to
                # actor.py's move_forward default when path is None.
                self.path = [start] * 5
                self.path_id = 0
        if self.object_detected and self.path is not None and len(self.path) < 3:
            self.object_detected = False
            return True
        self.last_pose = (px, py, yaw)

    def add_side_data(self,
                      image: np.ndarray,
                      depth: np.ndarray,
                      odometry: np.ndarray,
                      ) -> None:
        """
        Updates the GCLIP semantic obstacle map from a side camera.

        Only runs GCLIP inference and update_gclip() — does NOT call update()
        (project_dense) because that path assumes a forward-facing camera and
        produces wrong world projections for off-axis sensors.

        The existing min-depth aggregation in update_gclip() naturally implements
        "best camera" selection: for each grid cell the camera with the smallest
        depth (most direct / perpendicular view) wins, so left-side obstacles are
        filled from the left camera and right-side from the right camera.

        :param image: RGB image [C, H, W]
        :param depth: depth image [H, W]
        :param odometry: 4x4 camera-to-world transform with rotation R_z(yaw ± π/2)
        """
        if not self.one_map.camera_initialized:
            return
        if self.gclip_model is None:
            return
        odometry = odometry.astype(np.float32)
        gclip_features = self.gclip_model.get_image_features(image[np.newaxis, ...]).squeeze(0)
        self.one_map.update_gclip(gclip_features, depth, odometry)

    def get_map(self,
                return_map=True
                ) -> Optional[np.ndarray]:
        """
        Returns the similarity map given the query text
        :return: map as numpy array
        """
        if self.query_text_features is None:
            raise ValueError("No query text set")
        map_features = self.one_map.feature_map  # [X, Y, F]
        mask = self.one_map.updated_mask
        if mask.max() == 0:
            if return_map:
                return self.previous_sims.cpu().numpy()
            else:
                return
        if self.previous_sims is not None:
            map_features = map_features[mask, :].permute(1, 0).unsqueeze(0)
        else:
            map_features = map_features.permute(2, 0, 1).unsqueeze(0)

        similarity = self.model.compute_similarity(map_features, self.query_text_features)

        if self.previous_sims is None:
            self.previous_sims = similarity
        else:
            # then, similarity is only updated where the mask is true, otherwise it is the previous similarity
            self.previous_sims[:, mask] = similarity
        self.one_map.reset_updated_mask()
        if return_map:
            return self.previous_sims.cpu().numpy()
        else:
            return

    def get_confidence_map(self,
                           ) -> np.ndarray:
        """
          Returns the confidence map
          :return: map as numpy array
          """
        return self.one_map.confidence_map.cpu().numpy()
