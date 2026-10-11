import os
import numpy as np
from datetime import datetime
from pathlib import Path

from eval.habitat_evaluator import HabitatEvaluator, Result
from eval.semantic_collision import _SEMANTIC_SAFETY_RADIUS_CELLS
from config import load_eval_config
from eval.actor import MONActor

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("COSMO_LOG_DIR", PROJECT_ROOT / "logs"))


def resolve_eval_paths(eval_conf) -> None:
    """Resolve dataset paths relative to this checkout.

    Keeping this small helper here avoids pulling the optional visualization
    script into the minimal evaluation package.
    """
    for field in ("object_nav_path", "scene_path"):
        path = Path(getattr(eval_conf, field))
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        value = str(path)
        object.__setattr__(
            eval_conf,
            field,
            value if value.endswith("/") else value + "/",
        )


def write_eval_log(summary: dict, cfg):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"eval_{timestamp}.log"

    ec = cfg.EvalConf
    pc = cfg.PlanningConf
    mc = cfg.MappingConf

    detector = "YOLOWorld" if pc.using_ov else "YOLOv7"
    n_eps = summary["n_eps"]
    total_s = summary["total_time_s"]
    h, rem = divmod(int(total_s), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    result_counts = summary["result_counts"]
    total_steps = sum(summary["path_lengths"].values())  # approx from path lengths

    lines = [
        "=" * 60,
        "  OneMap Evaluation Summary",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "[ Run Config ]",
        f"  Detector          : {detector}",
        f"  CLIP model        : weights/clip.pth",
        f"  Dataset           : {ec.object_nav_path}",
        f"  Max steps         : {ec.max_steps}",
        f"  Max dist (success): {ec.max_dist} m",
        f"  Map size          : {mc.n_points}x{mc.n_points} cells, {mc.size} m",
        f"  Obstacle kernel   : {pc.obstcl_kernel_size} m",
        f"  CLIP semantic map : {getattr(mc, 'use_clip_semantic_nav_map', False)}  (sim_threshold={getattr(mc, 'clip_semantic_sim_threshold', 0.0)})",
        f"  YOLO obstacle map : {getattr(mc, 'use_yolo_obstacle_map', False)}"
        f"  (window={getattr(mc, 'yolo_window_size', 50)})",
        f"  Obstacle dict     : {list(_SEMANTIC_SAFETY_RADIUS_CELLS.keys())}",
        f"  YOLO confidence   : {pc.yolo_confidence}",
        f"  Consensus filter  : {pc.consensus_filtering}",
        f"  Using frontiers   : {pc.use_frontiers}",
        "",
        "[ Performance ]",
        f"  Episodes          : {n_eps}",
        f"  SR  (Success Rate): {summary['sr']:.1%}",
        f"  SPL               : {summary['spl']:.3f}",
        "",
        "[ Timing ]",
        f"  Total runtime     : {time_str}  ({total_s:.0f} s)",
        f"  Avg time/episode  : {total_s / n_eps:.1f} s" if n_eps else "",
    ]

    if summary["avg_step_time_ms"] is not None:
        lines.append(f"  Avg planning/step : {summary['avg_step_time_ms']:.1f} ms  (RGB→action, excl. sim)")

    path_lens = list(summary["path_lengths"].values())
    if path_lens:
        avg_path = np.mean(path_lens)
        lines.append(f"  Avg path length   : {avg_path:.1f} m")

    lines += [
        "",
        "[ Result Breakdown ]",
    ]
    for r in Result:
        count = result_counts.get(r, 0)
        pct = count / n_eps * 100 if n_eps else 0
        lines.append(f"  {r.name:<25} {count:>3} / {n_eps}  ({pct:5.1f}%)")

    lines += ["", "=" * 60]

    log_text = "\n".join(lines)
    print("\n" + log_text)
    log_path.write_text(log_text)
    print(f"\nLog saved to: {log_path}")


def main():
    eval_config = load_eval_config()
    resolve_eval_paths(eval_config.EvalConf)
    evaluator = HabitatEvaluator(eval_config.EvalConf, MONActor(eval_config.EvalConf))
    summary = evaluator.evaluate()
    write_eval_log(summary, eval_config)


if __name__ == "__main__":
    main()
