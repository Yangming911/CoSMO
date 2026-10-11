from eval.dataset_utils import Episode, SceneData, SemanticObject

from typing import Dict, List

import os
from os import listdir
import gzip
import json


# MP3D ObjectNav v1: 21 goal categories
MP3D_GOAL_CATEGORIES = [
    "chair", "table", "picture", "cabinet", "cushion",
    "sofa", "bed", "chest_of_drawers", "plant", "sink",
    "toilet", "stool", "towel", "tv_monitor", "shower",
    "bathtub", "counter", "fireplace", "gym_equipment",
    "seating", "clothes",
]


def load_mp3d_episodes(episodes: List[Episode], scene_data: Dict[str, SceneData], object_nav_path: str):
    """Load MP3D ObjectNav episodes.

    MP3D episode files use the same per-scene .json.gz format as HM3D.
    Key differences:
      - scene_id format: 'mp3d/{scene}/{scene}.glb' (no .basis)
      - goals_by_category keys: '{scene}.glb_{category}'
      - 21 goal categories (vs HM3D's 6)
    """
    i = len(episodes)
    files = listdir(object_nav_path)
    files = sorted(files, key=str.casefold)
    for file in files:
        if file.endswith('.json.gz'):
            with gzip.open(os.path.join(object_nav_path, file), 'r') as f:
                json_data = json.load(f)
                scene_id = json_data['episodes'][0]['scene_id']
                if scene_id not in scene_data:
                    scene_data_ = SceneData(scene_id, {}, {})
                    for obj_ in json_data['goals_by_category']:
                        obj = json_data['goals_by_category'][obj_]
                        obj_name = obj[0]['object_category']
                        scene_data_.object_locations[obj_name] = []
                        scene_data_.object_ids[obj_name] = []
                        for obj_loc in obj:
                            scene_data_.object_ids[obj_name].append(obj_loc['object_id'])
                    scene_data[scene_id] = scene_data_
                for ep in json_data['episodes']:
                    episode = Episode(ep['scene_id'],
                                      i,
                                      ep['start_position'],
                                      ep['start_rotation'],
                                      [ep['object_category']],
                                      ep['info']['geodesic_distance'])
                    episodes.append(episode)
                    i += 1
    return episodes, scene_data


def load_mp3d_objects(scene_data: Dict[str, SceneData], semantic_objects, scene_id: str):
    """Load semantic object bounding boxes from simulator, same logic as HM3D.

    MP3D semantic annotations use the same habitat-sim API:
    scene_obj.category.name(), scene_obj.aabb, scene_obj.semantic_id
    """
    for scene_obj in semantic_objects:
        obj_name = scene_obj.category.name()
        if not obj_name:
            continue
        added_to_any = False
        for cat in list(scene_data[scene_id].object_locations.keys()):
            if scene_obj.id in scene_data[scene_id].object_locations[cat]:
                added_to_any = True
                continue
            if scene_obj.semantic_id in scene_data[scene_id].object_ids.get(cat, []):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
                added_to_any = True
            elif obj_name in cat or cat in obj_name:
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
                added_to_any = True
            # MP3D category aliases
            elif cat == "plant" and ("flower" in obj_name or "vase" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
                added_to_any = True
            elif cat == "sofa" and ("couch" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
                added_to_any = True
            elif cat == "tv_monitor" and ("tv" in obj_name or "monitor" in obj_name or "television" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
                added_to_any = True
            elif cat == "chest_of_drawers" and ("dresser" in obj_name or "drawer" in obj_name):
                scene_data[scene_id].object_locations[cat].append(
                    SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
                added_to_any = True
        if not added_to_any:
            if obj_name not in scene_data[scene_id].object_locations:
                scene_data[scene_id].object_locations[obj_name] = []
            scene_data[scene_id].object_locations[obj_name].append(
                SemanticObject(scene_obj.id, obj_name, scene_obj.aabb, scene_obj.semantic_id))
    return scene_data


if __name__ == '__main__':
    eps, scene_data = load_mp3d_episodes([], {}, "datasets/objectnav_mp3d_v1/val/content")
    print(f"Found {len(eps)} episodes")
    scene_dist = {}
    for ep in eps:
        if ep.scene_id not in scene_dist:
            scene_dist[ep.scene_id] = 1
        else:
            scene_dist[ep.scene_id] += 1
    for sc in scene_dist:
        print(f"Scene {sc}, number of eps {scene_dist[sc]}")

    obj_counts = {}
    for ep in eps:
        for obj in ep.obj_sequence:
            if obj not in obj_counts:
                obj_counts[obj] = 1
            else:
                obj_counts[obj] += 1
    total = sum(obj_counts.values())
    for obj in sorted(obj_counts, key=obj_counts.get, reverse=True):
        print(f"Object {obj}, count {obj_counts[obj]}, percentage {obj_counts[obj] / total:.2%}")
