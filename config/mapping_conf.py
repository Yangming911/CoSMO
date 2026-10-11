from typing import Optional
from spock import spock

@spock
class MappingConf:
    n_points: int
    size: int
    agent_radius: float
    blur_kernel_size: float
    obstacle_map_threshold: float
    fully_explored_threshold: float
    checked_map_threshold: float

    depth_factor: float
    gradient_factor: float

    optimal_object_distance: float
    optimal_object_factor: float

    obstacle_min: float
    obstacle_max: float

    filter_stairs: bool
    floor_level: float
    floor_threshold: float
    use_clip_semantic_nav_map: bool
    clip_semantic_sim_threshold: float
    use_yolo_obstacle_map: bool
    yolo_window_size: int
    clip_model_type: Optional[str] = "convnext"
    use_gclip_for_obstacles: Optional[bool] = False
    # CLIP obstacle map — two modes (mutually exclusive):
    #   use_clip_argmax_obstacle_map: pure argmax over COCO+NON_OBSTACLE vs obstacle labels, no CP params needed
    #   use_clip_cp_obstacle_map:     threshold / OACP mode, requires clip_cp_* params below
    use_clip_argmax_obstacle_map: Optional[bool] = False
    use_clip_cp_obstacle_map: Optional[bool] = False
    clip_cp_threshold: Optional[float] = 0.05
    clip_cp_use_oacp: Optional[bool] = False
    clip_cp_target_coverage: Optional[float] = 0.9
    clip_cp_window_size: Optional[int] = 200
    clip_cp_gamma: Optional[float] = 0.05
    clip_cp_initial_alpha: Optional[float] = 0.5
    clip_cp_use_margin: Optional[bool] = False
    clip_cp_max_seeds_per_label: Optional[int] = 100
    clip_cp_temporal_persistence: Optional[int] = 0
    clip_cp_detection_gate: Optional[bool] = False
    clip_cp_safety_radius_scale: Optional[float] = 1.0
    clip_cp_safety_radius_add: Optional[int] = 0
    clip_cp_step_guard: Optional[bool] = False
    clip_cp_step_guard_lookahead: Optional[int] = 3
    clip_cp_argmax_confidence: Optional[float] = 0.0
    clip_cp_open_vocab: Optional[bool] = False
    clip_cp_oracle_labels: Optional[str] = ""
    clip_cp_use_mp3d_labels: Optional[bool] = False
    clip_cp_calibration_delay: Optional[int] = 0
    clip_cp_delay_adaptive_gamma: Optional[bool] = False
    clip_cp_use_softmax_score: Optional[bool] = False
    clip_cp_softmax_recompute: Optional[bool] = False
    use_yolo_cp_obstacle_map: Optional[bool] = False
    yolo_cp_threshold: Optional[float] = 0.5
    yolo_cp_target_coverage: Optional[float] = 0.9
    yolo_cp_window_size: Optional[int] = 200
    yolo_cp_frustum_max_depth: Optional[float] = 5.0
    use_gdino_oracle: Optional[bool] = False
    gdino_confidence: Optional[float] = 0.3

