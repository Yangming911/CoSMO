from typing import List, Optional
from spock import spock, SpockBuilder

from config import HabitatControllerConf, MappingConf, PlanningConf


@spock
class EvalConf:
    multi_object: bool
    max_steps: int
    max_dist: float
    log_rerun: bool
    is_gibson: bool
    controller: HabitatControllerConf
    mapping: MappingConf
    planner: PlanningConf
    object_nav_path: str
    scene_path: str
    use_pointnav: bool
    square_im: bool
    results_path: str = "results/"
    ep_start: int = 0        # first episode index (inclusive)
    ep_end: int = 999999     # last episode index (exclusive), 999999 = all
    include_ids: Optional[List[int]] = None  # if set, only run these post-slice episode indices
    include_ids_file: Optional[str] = None  # path to txt file with one episode ID per line
    dataset_type: Optional[str] = None  # "hm3d", "mp3d", "gibson"; None = auto from is_gibson


def load_eval_config():
    return SpockBuilder(EvalConf, HabitatControllerConf, PlanningConf, MappingConf,
                        desc='Default MON config.').generate()
