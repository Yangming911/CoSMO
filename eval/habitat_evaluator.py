# eval utils
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from eval.dataset_utils.gibson_dataset import load_gibson_episodes
from eval.dataset_utils.mp3d_dataset import load_mp3d_episodes
from eval.semantic_collision import build_semantic_collision_data, metric_to_px, _build_label_seed_map
from mapping.semantic_debug import SemanticPredGTCollector, YOLOObstacleDebugCollector
from mapping import rerun_logger
from config import EvalConf
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *

# os / filsystem
import bz2
import os
from os import listdir
import gzip
import json
import pathlib

# cv2
import cv2

# numpy
import numpy as np

# skimage
import skimage


# dataclasses
from dataclasses import dataclass

# quaternion
import quaternion

# typing
from typing import Tuple, List, Dict
import enum
import time

# habitat
import habitat_sim
from habitat_sim import ActionSpec, ActuationSpec
from habitat_sim.utils import common as utils

# tabulate
from tabulate import tabulate

# rerun
import rerun as rr

# pandas
import pandas as pd

# pickle
import pickle

# scipy
from scipy.spatial.transform import Rotation as R

class Result(enum.Enum):
    SUCCESS = 1
    FAILURE_MISDETECT = 2
    FAILURE_STUCK = 3
    FAILURE_OOT = 4
    FAILURE_NOT_REACHED = 5
    FAILURE_ALL_EXPLORED = 6
    SEMANTIC_COLLISION = 7

class HabitatEvaluator:
    def __init__(self,
                 config: EvalConf,
                 actor: Actor,
                 ) -> None:
        self.config = config
        self.multi_object = config.multi_object
        self.max_steps = config.max_steps
        self.max_dist = config.max_dist
        self.controller = config.controller
        self.mapping = config.mapping
        self.planner = config.planner
        self.log_rerun = config.log_rerun
        self.object_nav_path = config.object_nav_path
        self.scene_path = config.scene_path
        self.scene_data = {}
        self.episodes = []
        self.exclude_ids = []
        self.is_gibson = config.is_gibson
        # dataset_type: explicit or derived from is_gibson for backward compat
        _dt = getattr(config, "dataset_type", None)
        if _dt:
            self.dataset_type = _dt
        else:
            self.dataset_type = "gibson" if config.is_gibson else "hm3d"
        self.sim = None
        self.actor = actor
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True
        self.control_frequency = config.controller.control_freq
        self.max_vel = config.controller.max_vel
        self.max_ang_vel = config.controller.max_ang_vel
        self.time_step = 1.0 / self.control_frequency
        if self.is_gibson:
            dataset_info_file = str(pathlib.Path(self.object_nav_path).parent.absolute()) + \
                                "/val_info.pbz2".format(split="val")
            with bz2.BZ2File(dataset_info_file, 'rb') as f:
                self.dataset_info = pickle.load(f)
        else:
            self.dataset_info = None
        if self.dataset_type == "gibson":
            self.episodes, self.scene_data = GibsonDataset.load_gibson_episodes(self.episodes,
                                                                                self.scene_data,
                                                                                self.dataset_info,
                                                                                self.object_nav_path)
        elif self.dataset_type == "mp3d":
            self.episodes, self.scene_data = MP3DDataset.load_mp3d_episodes(self.episodes,
                                                                            self.scene_data,
                                                                            self.object_nav_path)
        else:
            if self.multi_object:
                self.episodes, self.scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(self.episodes,
                                                                                           self.scene_data,
                                                                                           self.object_nav_path)
            else:
                self.episodes, self.scene_data = HM3DDataset.load_hm3d_episodes(self.episodes,
                                                                                self.scene_data,
                                                                                self.object_nav_path)
        if self.actor is not None:
            self.logger = rerun_logger.RerunLogger(self.actor.mapper, False, "") if self.log_rerun else None
        self.results_path = config.results_path
        for _sub in ("trajectories", "similarities", "state"):
            os.makedirs(os.path.join(self.results_path, _sub), exist_ok=True)
        ep_start = getattr(config, "ep_start", 0)
        ep_end = getattr(config, "ep_end", 999999)
        self.episodes = self.episodes[ep_start:ep_end]
        _inc = getattr(config, "include_ids", None)
        _inc_file = getattr(config, "include_ids_file", None)
        if _inc_file is not None and os.path.exists(_inc_file):
            with open(_inc_file) as _f:
                _file_ids = [int(line.strip()) for line in _f if line.strip()]
            _inc = _file_ids if _inc is None else list(_inc) + _file_ids
        if _inc is not None and len(_inc) > 0:
            _inc_set = set(int(i) for i in _inc)
            self.exclude_ids = [i for i in range(len(self.episodes)) if i not in _inc_set]
        self.semantic_collision_cache = {}
        self.debug_collector = SemanticPredGTCollector()
        if self.actor is not None:
            self.actor.mapper.debug_collector = self.debug_collector
        self.yolo_debug_collector = YOLOObstacleDebugCollector()
        if self.actor is not None:
            yolo_map = getattr(self.actor.mapper, "yolo_obstacle_map", None)
            if yolo_map is not None:
                yolo_map.debug_collector = self.yolo_debug_collector

    def load_scene(self, scene_id: str):
        if self.sim is not None:
            self.sim.close()
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = self.scene_path + scene_id
        if self.dataset_type == "gibson":
            pass
        elif self.dataset_type == "mp3d":
            pass  # MP3D loads .glb directly; semantic .ply is auto-detected by habitat-sim
        else:
            backend_cfg.scene_dataset_config_file = self.scene_path + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        hfov = 90
        rgb = habitat_sim.CameraSensorSpec()
        rgb.uuid = "rgb"
        rgb.hfov = hfov
        rgb.position = np.array([0, 0.88, 0])
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        res = 640
        rgb.resolution = [res, res]

        depth = habitat_sim.CameraSensorSpec()
        depth.uuid = "depth"
        depth.hfov = hfov
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.position = np.array([0, 0.88, 0])
        depth.resolution = [res, res]

        agent_cfg = habitat_sim.agent.AgentConfiguration(action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
            turn_left=ActionSpec("turn_left", ActuationSpec(amount=5.0)),
            turn_right=ActionSpec("turn_right", ActuationSpec(amount=5.0)),
        ))
        agent_cfg.sensor_specifications = [rgb, depth]
        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(sim_cfg)
        if not self.scene_data[scene_id].objects_loaded:
            if self.dataset_type == "gibson":
                self.scene_data = GibsonDataset.load_gibson_objects(self.scene_data, self.dataset_info, scene_id)
            elif self.dataset_type == "mp3d":
                self.scene_data = MP3DDataset.load_mp3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)
            else:
                self.scene_data = HM3DDataset.load_hm3d_objects(self.scene_data, self.sim.semantic_scene.objects, scene_id)
        cell_size = self.mapping.size / self.mapping.n_points
        gt_seed_map, gt_labels = _build_label_seed_map(
            self.scene_data[scene_id].object_locations, self.mapping.n_points, cell_size, self.is_gibson
        )
        self.debug_collector.set_gt_map(gt_seed_map, gt_labels)
        self.yolo_debug_collector.set_gt_map(gt_seed_map, gt_labels)

    def get_semantic_collision_data(self, scene_id: str, query_label: str,
                                     floor_y: float = None):
        # floor_y is per-episode (robot may be on different floors in same scene)
        floor_y_key = round(floor_y, 1) if floor_y is not None else None
        cache_key = (scene_id, query_label, floor_y_key)
        if cache_key not in self.semantic_collision_cache:
            cell_size = self.mapping.size / self.mapping.n_points
            max_query_radius_cells = int(self.planner.max_detect_distance / cell_size)
            self.semantic_collision_cache[cache_key] = build_semantic_collision_data(
                self.scene_data[scene_id].object_locations,
                self.mapping.n_points,
                self.mapping.size,
                self.is_gibson,
                query_label,
                max_query_radius_cells,
                floor_y=floor_y,
            )
        return self.semantic_collision_cache[cache_key]

    def check_semantic_collision(self, scene_id: str, query_label: str):
        """Return (collided: bool, causing_label: str | None)."""
        floor_y = getattr(self, "_episode_floor_y", None)
        collision_data = self.get_semantic_collision_data(scene_id, query_label, floor_y=floor_y)
        position = self.sim.get_agent(0).get_state().position
        x = -position[2]
        y = -position[0]
        px, py = metric_to_px(x, y, self.mapping.n_points, self.mapping.size / self.mapping.n_points)
        if 0 <= px < self.mapping.n_points and 0 <= py < self.mapping.n_points:
            if collision_data.collision_map[px, py]:
                idx = int(collision_data.label_map[px, py])
                label = collision_data.labels[idx - 1] if 0 < idx <= len(collision_data.labels) else "unknown"
                return True, label
        return False, None



    def execute_action(self, action: Dict
                       ):
        if 'discrete' in action.keys():
            # We have a discrete actor
            self.sim.step(action['discrete'])

        elif 'continuous' in action.keys():
            # We have a continuous actor
            self.vel_control.angular_velocity = action['continuous']['angular']
            self.vel_control.linear_velocity = action['continuous']['linear']
            agent_state = self.sim.get_agent(0).state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )

            # manually integrate the rigid state
            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            # snap rigid state to navmesh and set state to object/sim
            # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )

            # set the computed state
            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(
                target_rigid_state.rotation
            )
            self.sim.get_agent(0).set_state(agent_state)
            self.sim.step_physics(self.time_step)


    def read_results(self, path, sort_by):
        state_dir = os.path.join(path, 'state')
        state_results = {}
        object_query = {}
        scene_name = {}
        spl = {}

        # Check if the state directory exists
        if not os.path.isdir(state_dir):
            print(f"Error: {state_dir} is not a valid directory")
            return state_results
        pose_dir = os.path.join(os.path.abspath(os.path.join(state_dir, os.pardir)), "trajectories")

        # Iterate through all files in the state directory
        for filename in os.listdir(state_dir):
            if filename.startswith('state_') and filename.endswith('.txt'):
                try:
                    # Extract the experiment number from the filename
                    experiment_num = int(filename[6:-4])  # removes 'state_' and '.txt'
                    # Read the content of the file
                    with open(os.path.join(state_dir, filename), 'r') as file:
                        content = file.read().strip()

                    # Convert the content to a number (assuming it's a float)
                    state_value = int(content)
                    # Store the result in the dictionary
                    state_results[experiment_num] = state_value
                    object_query[experiment_num] = self.episodes[experiment_num].obj_sequence[0]
                    scene_name[experiment_num] = self.episodes[experiment_num].scene_id
                    poses = np.genfromtxt(os.path.join(pose_dir, "poses_" + str(experiment_num) + ".csv"), delimiter=",")
                    deltas = poses[1:, :2] - poses[:-1, :2]
                    distance_traveled = np.linalg.norm(deltas, axis=1).sum()
                    if state_value == 1:
                        spl[experiment_num] = self.episodes[experiment_num].best_dist / max(self.episodes[experiment_num].best_dist, distance_traveled)
                    else:
                        spl[experiment_num] = 0
                    if self.episodes[experiment_num].episode_id != experiment_num:
                        print(f"Warning, exerpiment_num {experiment_num} does not correctly resolve to episode_id {self.episodes[experiment_num].episode_id}")
                except ValueError:
                    print(f"Warning: Skipping {filename} due to invalid format")
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        dict_res = {"state": state_results, "obj" : object_query, "scene" : scene_name, "spl" : spl}
        data = pd.DataFrame.from_dict(dict_res)

        states = data["state"].unique()

        def calculate_percentages(group):
            total = len(group)
            result = pd.Series({Result(state).name: (group['state'] == state).sum() / total for state in states})

            # Calculate average SPL and multiply by 100
            avg_spl = group['spl'].mean()
            result['Average SPL'] = avg_spl

            return result

        # Per-object results
        object_results = data.groupby('obj').apply(calculate_percentages).reset_index()
        object_results = object_results.rename(columns={'obj': 'Object'})

        # Per-scene results
        scene_results = data.groupby('scene').apply(calculate_percentages).reset_index()
        scene_results = scene_results.rename(columns={'scene': 'Scene'})

        # Overall results
        overall_percentages = calculate_percentages(data)
        overall_row = pd.DataFrame([{**{'Object': 'Overall'}, **overall_percentages.to_dict()}])
        object_results = pd.concat([overall_row, object_results], ignore_index=True)

        overall_row = pd.DataFrame([{**{'Scene': 'Overall'}, **overall_percentages.to_dict()}])
        scene_results = pd.concat([overall_row, scene_results], ignore_index=True)

        # Sorting
        object_results = object_results.sort_values(by=sort_by, ascending=False)
        scene_results = scene_results.sort_values(by=sort_by, ascending=False)

        # Function to format percentages
        def format_percentages(val):
            return f"{val:.2%}" if isinstance(val, float) else val

        # Apply formatting to all columns except the first one (Object/Scene)
        object_table = object_results.iloc[:, 0].to_frame().join(
            object_results.iloc[:, 1:].applymap(format_percentages))
        scene_table = scene_results.iloc[:, 0].to_frame().join(
            scene_results.iloc[:, 1:].applymap(format_percentages))

        print(f"Results by Object (sorted by {sort_by} rate, descending):")
        print(tabulate(object_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))

        print(f"\nResults by Scene (sorted by {sort_by} rate, descending):")
        print(tabulate(scene_table, headers='keys', tablefmt='pretty', floatfmt='.2%'))
        return data

    def evaluate(self):
        eval_start = time.time()
        success = 0
        n_eps = 0
        success_per_obj = {}
        obj_count = {}
        results = []
        spl_accum = 0.0
        collision_log = []  # list of (episode_id, query_label, causing_label)
        path_lengths = {}
        # restart at 930
        for n_ep, episode in enumerate(self.episodes):
            poses = []
            results.append(Result.FAILURE_OOT)
            steps = 0
            if n_ep in self.exclude_ids:
                continue
            n_eps += 1
            if self.sim is None or not self.sim.curr_scene_name in episode.scene_id:
                self.load_scene(episode.scene_id)
            self.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
            self.actor.reset()
            current_obj_id = 0
            current_obj = episode.obj_sequence[current_obj_id]
            if current_obj not in success_per_obj:
                success_per_obj[current_obj] = 0
                obj_count[current_obj] = 1
            else:
                obj_count[current_obj] += 1
            self.actor.set_query(current_obj)
            # floor_y: agent start_position[1] IS the floor Y in world coords
            # (floor_level=-0.88 is camera-frame offset, not world offset)
            self._episode_floor_y = float(episode.start_position[1])
            gt_cd = self.get_semantic_collision_data(episode.scene_id, current_obj,
                                                     floor_y=self._episode_floor_y)
            # For OACP calibration: build inclusive label map (includes target object).
            # Normal gt_cd excludes the query label; OACP needs ALL objects incl. target.
            if getattr(self.actor.mapper, "use_clip_cp_obstacle_map", False):
                _cell_size = self.mapping.size / self.mapping.n_points
                _max_r = int(self.planner.max_detect_distance / _cell_size)
                _gt_label_dict = None
                if bool(getattr(self.mapping, "clip_cp_use_mp3d_labels", False)):
                    from eval.dataset_utils.mp3d_dataset import MP3D_GOAL_CATEGORIES
                    from eval.semantic_collision import normalize_semantic_label, _SEMANTIC_SAFETY_RADIUS_CELLS
                    _gt_label_dict = {normalize_semantic_label(l): 0 for l in MP3D_GOAL_CATEGORIES}
                    _gt_label_dict.update(_SEMANTIC_SAFETY_RADIUS_CELLS)
                _gt_oacp = build_semantic_collision_data(
                    self.scene_data[episode.scene_id].object_locations,
                    self.mapping.n_points, self.mapping.size, self.is_gibson,
                    query_label="_oacp_none_",  # dummy: nothing excluded
                    max_query_radius_cells=_max_r,
                    floor_y=self._episode_floor_y,
                    oacp_radius_override=15,  # 1.5m radius to cover GCLIP-projected surface cells
                    label_dict=_gt_label_dict,
                )
                self.actor.mapper.set_gt_label_map(_gt_oacp.label_map, _gt_oacp.labels)
            else:
                self.actor.mapper.set_gt_label_map(gt_cd.label_map, gt_cd.labels)
            if self.log_rerun:
                pts = []
                for obj in self.scene_data[episode.scene_id].object_locations[current_obj]:
                    if not self.is_gibson:
                        pt = obj.bbox.center[[0, 2]]
                        pt = (-pt[1], -pt[0])
                        pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                    else:
                        for pt_ in obj:
                            pt = (pt_[0], pt_[1])
                            pts.append(self.actor.mapper.one_map.metric_to_px(*pt))
                pts = np.array(pts)
                rr.log("map/ground_truth", rr.Points2D(pts, colors=[[255, 255, 0]], radii=[1]))

            while steps < self.max_steps and current_obj_id < len(episode.obj_sequence):
                observations = self.sim.get_sensor_observations()
                observations['state'] = self.sim.get_agent(0).get_state()
                pose = np.zeros((4, ))
                pose[0] = -observations['state'].position[2]
                pose[1] = -observations['state'].position[0]
                pose[2] = observations['state'].position[1]
                # yaw
                orientation = observations['state'].rotation
                q0 = orientation.x
                q1 = orientation.y
                q2 = orientation.z
                q3 = orientation.w
                r = R.from_quat([q0, q1, q2, q3])
                # r to euler
                yaw, _, _1 = r.as_euler("yxz")
                pose[3] = yaw

                poses.append(pose)
                if self.log_rerun:
                    cam_x = -self.sim.get_agent(0).get_state().position[2]
                    cam_y = -self.sim.get_agent(0).get_state().position[0]
                    rr.log("camera/rgb", rr.Image(observations["rgb"]).compress(jpeg_quality=50))
                    self.logger.log_pos(cam_x, cam_y)
                action, called_found = self.actor.act(observations)

                yolo_cp_map = getattr(self.actor.mapper, "yolo_cp_obstacle_map", None)
                if yolo_cp_map is not None and getattr(self.actor.mapper, "use_yolo_cp_obstacle_map", False):
                    collision_data = self.get_semantic_collision_data(episode.scene_id, current_obj,
                                                                       floor_y=self._episode_floor_y)
                    cell_size = self.mapping.size / self.mapping.n_points
                    robot_x = -observations['state'].position[2]
                    robot_y = -observations['state'].position[0]
                    robot_px, robot_py = metric_to_px(robot_x, robot_y, self.mapping.n_points, cell_size)
                    yolo_cp_map.calibrate_with_gt(
                        collision_data.label_map,
                        collision_data.labels,
                        robot_px, robot_py,
                        yaw,
                    )

                self.execute_action(action)
                if self.log_rerun:
                    self.logger.log_map()

                collided, cause_label = self.check_semantic_collision(episode.scene_id, current_obj)
                if collided:
                    results[n_ep] = Result.SEMANTIC_COLLISION
                    collision_log.append((episode.episode_id, current_obj, cause_label))
                    # Diagnostic: check if OACP predicted this obstacle cell
                    _cp = getattr(self.actor.mapper, "clip_cp_obstacle_map", None)
                    _cp_diag = ""
                    if _cp is not None:
                        position = self.sim.get_agent(0).get_state().position
                        from eval.semantic_collision import metric_to_px
                        _ax, _ay = metric_to_px(-position[2], -position[0],
                                                self.mapping.n_points,
                                                self.mapping.size / self.mapping.n_points)
                        _was_blocked = bool(_cp.get_obstacle_mask()[_ax, _ay]) if 0 <= _ax < self.mapping.n_points and 0 <= _ay < self.mapping.n_points else False
                        _seed_val = int(_cp._seed_map[_ax, _ay]) if 0 <= _ax < self.mapping.n_points and 0 <= _ay < self.mapping.n_points else -1
                        _n_seeds = int((_cp._seed_map > 0).sum())
                        _n_blocked = int(_cp._obstacle_mask.sum())
                        _n_base = int(self.actor.mapper.one_map.navigable_map.sum())
                        _cp_diag = (f" [OACP: agent_cell_blocked={_was_blocked}, seed={_seed_val}, "
                                    f"total_seeds={_n_seeds}, blocked={_n_blocked}({100*_n_blocked/_n_base:.1f}%), tau={_cp.threshold:.4f}]")
                    print(f"Semantic collision detected! cause={cause_label} step={steps}{_cp_diag}")
                    break

                if called_found:
                    # We will now compute the closest distance to the bounding box of the object
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    if dist < self.max_dist:
                        results[n_ep] = Result.SUCCESS
                        success += 1
                        print("Object found!")
                        success_per_obj[current_obj] += 1
                    else:
                        pos = self.actor.mapper.chosen_detection
                        pos_metric = self.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
                        dist_detect = get_closest_dist([-pos_metric[1], -pos_metric[0]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                        if dist_detect < self.max_dist:
                            results[n_ep] = Result.FAILURE_NOT_REACHED
                        else:
                            results[n_ep] = Result.FAILURE_MISDETECT
                        print(f"Object not found! Dist {dist}, detect dist: {dist_detect}.")
                    current_obj_id += 1

                if steps % 100 == 0:
                    dist = get_closest_dist(self.sim.get_agent(0).get_state().position[[0, 2]],
                                            self.scene_data[episode.scene_id].object_locations[current_obj], self.is_gibson)
                    _cp = getattr(self.actor.mapper, "clip_cp_obstacle_map", None)
                    _cp_stats = ""
                    if _cp is not None:
                        _nav = self.actor.mapper.get_active_navigable_map() if hasattr(self.actor.mapper, "get_active_navigable_map") else None
                        _base = self.actor.mapper.one_map.navigable_map
                        _mask = _cp.get_obstacle_mask()
                        _n_seeds = int((_cp._seed_map > 0).sum())
                        _n_blocked = int((_mask & _base).sum())
                        _n_base = int(_base.sum())
                        _pct = 100.0 * _n_blocked / _n_base if _n_base > 0 else 0
                        _tau = f"{_cp.threshold:.4f}"
                        _cp_stats = f"  [CP seeds={_n_seeds} blocked={_n_blocked}({_pct:.1f}%) tau={_tau}]"
                    print(f"Step {steps}, current object: {current_obj}, episode_id: {episode.episode_id}, distance to closest object: {dist}{_cp_stats}")
                    # Early stuck detection: if agent hasn't moved >0.1m in last 100 steps, abort episode
                    if steps >= 100:
                        _recent = np.array(poses[-100:])
                        _disp = float(np.linalg.norm(_recent[-1, :2] - _recent[0, :2]))
                        if _disp < 0.1:
                            results[n_ep] = Result.FAILURE_STUCK
                            print(f"Early stuck: displacement={_disp:.4f}m over last 100 steps, aborting episode")
                            break
                steps += 1
            poses = np.array(poses)
            # If the last 10 poses didn't change much and we have OOT, assume stuck
            if results[n_ep] == Result.FAILURE_OOT and np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
                results[n_ep] = Result.FAILURE_STUCK

            # Path length (xy plane) and SPL
            path_len = float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))) if len(poses) > 1 else 0.0
            path_lengths[episode.episode_id] = path_len
            geo_dist = episode.best_dist if isinstance(episode.best_dist, float) else float(episode.best_dist)
            is_success = results[n_ep] == Result.SUCCESS
            spl_accum += is_success * geo_dist / max(path_len, geo_dist) if geo_dist > 0 else 0.0

            num_frontiers = len(self.actor.mapper.nav_goals)
            np.savetxt(f"{self.results_path}/trajectories/poses_{episode.episode_id}.csv", poses, delimiter=",")
            # save final sim to image file
            final_sim = (self.actor.mapper.get_map() + 1.0) / 2.0
            final_sim = final_sim[0]
            final_sim = final_sim.transpose((1, 0))
            final_sim = np.flip(final_sim, axis=0)
            final_sim = monochannel_to_inferno_rgb(final_sim)
            cv2.imwrite(f"{self.results_path}/similarities/final_sim_{episode.episode_id}.png", final_sim)
            if (results[n_ep] == Result.FAILURE_STUCK or results[n_ep] == Result.FAILURE_OOT) and num_frontiers == 0:
                results[n_ep] = Result.FAILURE_ALL_EXPLORED
            print(f"Overall success: {success / (n_eps)}, per object: ")
            for obj in success_per_obj.keys():
                print(f"{obj}: {success_per_obj[obj] / obj_count[obj]}")
            print(
                f"Result distribution: successes: {results.count(Result.SUCCESS)}, misdetects: {results.count(Result.FAILURE_MISDETECT)}, OOT: {results.count(Result.FAILURE_OOT)}, stuck: {results.count(Result.FAILURE_STUCK)}, not reached: {results.count(Result.FAILURE_NOT_REACHED)}, all explored: {results.count(Result.FAILURE_ALL_EXPLORED)}, semantic collisions: {results.count(Result.SEMANTIC_COLLISION)}")
            # Write result to file
            with open(f"{self.results_path}/state/state_{episode.episode_id}.txt", 'w') as f:
                f.write(str(results[n_ep].value))

            # GCLIP argmax: per-episode alignment stats (print immediately, don't accumulate)
            _clip_cp = getattr(self.actor.mapper, "clip_cp_obstacle_map", None)
            if _clip_cp is not None and getattr(self.actor.mapper, "use_clip_argmax_obstacle_map", False):
                try:
                    import torch, torch.nn.functional as _F
                    _feat_map = getattr(self.actor.mapper.one_map, "feature_map_gclip", None)
                    _text_feats = _clip_cp._text_features
                    _obs_labels = _clip_cp._labels
                    _seed_map = _clip_cp._seed_map
                    _gt_lm = gt_cd.label_map
                    _gt_labels = gt_cd.labels
                    if _feat_map is not None and _text_feats is not None:
                        _fn = _feat_map if not hasattr(_feat_map, 'cpu') else _feat_map.cpu().numpy()
                        _nc = self.actor.mapper.one_map.n_cells
                        _flat = _fn.reshape(-1, _fn.shape[-1])
                        _flat_t = torch.from_numpy(_flat).float()
                        _norms = _flat_t.norm(dim=1)
                        _obs_mask = (_seed_map.reshape(-1) > 0) & (_norms.numpy() > 1e-6)
                        _nav_np = self.actor.mapper.one_map.navigable_map.astype(bool).reshape(-1)
                        _conf_np = (_norms.numpy() > 1e-6)
                        _free_mask = _nav_np & _conf_np & (_seed_map.reshape(-1) == 0)
                        _n_obs = int(_obs_mask.sum())
                        _n_free = int(_free_mask.sum())
                        _gt_hit, _gt_wrong, _gt_miss = 0, 0, 0
                        for _mask, _cell_type in [(_obs_mask, "argmax_obstacle"), (_free_mask, "free")]:
                            _idxs = np.where(_mask)[0]
                            if len(_idxs) == 0:
                                continue
                            _rng = np.random.default_rng(42)
                            _sample = _rng.choice(_idxs, size=min(20, len(_idxs)), replace=False)
                            for _idx in _sample:
                                _px, _py = int(_idx // _nc), int(_idx % _nc)
                                _pred_j = int(_seed_map.reshape(-1)[_idx]) - 1 if _seed_map.reshape(-1)[_idx] > 0 else -1
                                _pred_lbl = _obs_labels[_pred_j] if _pred_j >= 0 else "bg"
                                _gt_idx = int(_gt_lm[_px, _py])
                                _gt_lbl = _gt_labels[_gt_idx - 1] if 0 < _gt_idx <= len(_gt_labels) else "bg"
                                if _gt_lbl != "bg":
                                    if _cell_type == "free":
                                        _gt_miss += 1
                                    elif _pred_lbl == _gt_lbl:
                                        _gt_hit += 1
                                    else:
                                        _gt_wrong += 1
                        _gt_total = _gt_hit + _gt_wrong + _gt_miss
                        print(
                            f"[ALIGN] ep={episode.episode_id} obs_cells={_n_obs} free_cells={_n_free} "
                            f"gt_obs={_gt_total} hit={_gt_hit} wrong={_gt_wrong} miss={_gt_miss}",
                            flush=True,
                        )
                except Exception as _e:
                    print(f"Warning: GCLIP alignment failed ep={episode.episode_id}: {_e}")

            # OACP calibration log: print per-episode summary and accumulate for CSV
            if (_clip_cp is not None
                    and getattr(self.actor.mapper, "use_clip_cp_obstacle_map", False)
                    and getattr(_clip_cp, "use_oacp", False)):
                try:
                    _calib_log = _clip_cp.get_calibration_log()
                    _n_calls = len(_calib_log)
                    _coverage = (sum(1 for e in _calib_log if e["err"] == 0) / _n_calls) if _n_calls > 0 else float("nan")
                    _final_tau = _calib_log[-1]["tau"] if _calib_log else float("nan")
                    print(f"OACP ep={episode.episode_id}: n_calib={_n_calls}, coverage={_coverage:.3f}, final_tau={_final_tau:.4f}", flush=True)
                    if not hasattr(self, "_oacp_calib_rows"):
                        self._oacp_calib_rows = []
                    for _e in _calib_log:
                        self._oacp_calib_rows.append({"episode_id": episode.episode_id, **_e})
                except Exception as _oe:
                    print(f"Warning: OACP log collection failed ep={episode.episode_id}: {_oe}")

        total_time = time.time() - eval_start
        from datetime import datetime
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_collector.save_csv(f"{self.results_path}/semantic_pred_gt_pairs_{_ts}.csv")
        self.debug_collector.save_sim_distribution_csv(f"{self.results_path}/semantic_sim_distribution_{_ts}.csv")
        self.yolo_debug_collector.save_csv(f"{self.results_path}/yolo_obstacle_pred_gt_{_ts}.csv")
        # Save per-collision cause log
        import csv
        collision_csv = f"{self.results_path}/collision_causes_{_ts}.csv"
        with open(collision_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode_id", "query_label", "cause_label"])
            w.writerows(collision_log)
        if hasattr(self, "_gclip_align_rows") and self._gclip_align_rows:
            align_csv = f"{self.results_path}/gclip_alignment_{_ts}.csv"
            with open(align_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["episode_id","cell_type","px","py","pred_label","gt_label","sim_to_pred","sim_to_gt","sim_gap"])
                w.writeheader()
                w.writerows(self._gclip_align_rows)
            print(f"Saved GCLIP alignment CSV: {align_csv}")
        if hasattr(self, "_oacp_calib_rows") and self._oacp_calib_rows:
            oacp_csv = f"{self.results_path}/oacp_calibration_{_ts}.csv"
            with open(oacp_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["episode_id","step","tau","tau_after","err","alpha","s"], extrasaction='ignore')
                w.writeheader()
                w.writerows(self._oacp_calib_rows)
            print(f"Saved OACP calibration CSV: {oacp_csv}")
        sr = success / n_eps if n_eps > 0 else 0.0
        spl = spl_accum / n_eps if n_eps > 0 else 0.0
        result_counts = {r: results.count(r) for r in Result}
        step_times = getattr(self.actor, "step_times_ms", [])
        return {
            "n_eps": n_eps,
            "total_time_s": total_time,
            "sr": sr,
            "spl": spl,
            "result_counts": result_counts,
            "avg_step_time_ms": float(np.mean(step_times)) if step_times else None,
            "path_lengths": path_lengths,
        }
