# CoSMO: Conformal Semantic Mapping for Open-World Safe Robot Navigation

This repository contains the core simulation code for **CoSMO**, a framework that turns dense vision-language features into calibrated, category-dependent safety constraints for robot navigation.

CoSMO addresses two difficulties that arise when semantic maps are used for safety-critical planning:

- the set of safety-relevant object categories is not known in advance and can
  grow during deployment; and
- visual observations are non-stationary, while supervision from a stronger
  expert may arrive only after a delay.

CoSMO defines nonconformity in the vision-language embedding space. This makes previously collected calibration scores reusable when new categories are added. An online adaptive conformal update then controls the size of the semantic coverage sets.

On the full Matterport3D (MP3D) validation benchmark of 2,195 ObjectNav episodes, the paper reports a reduction in semantic collision rate from 25.5% for the geometry-only OneMap baseline to 9.8% for open-vocabulary CoSMO, a 61.6% relative reduction.



## Paper conditions included here

| condition | Configuration | Semantic obstacle handling | Expert used for calibration |
| --- | --- | --- | --- |
| Baseline (OneMap) | `config/mon/eval_baseline.yaml` | YOLO detections and geometric avoidance; no semantic safety dilation | None |
| GCLIP Argmax | `config/mon/eval_argmax.yaml` | Per-cell GCLIP argmax with category-dependent dilation; no CP | None |
| CoSMO (open-vocabulary) | `config/mon/eval_oacp.yaml` | Online coverage sets and worst-case semantic dilation | MP3D ground-truth semantic labels |
| CoSMO (calibrated with GDINO) | `config/mon/eval_oacp_gdino.yaml` | Same CoSMO pipeline | Grounding DINO detections |





## Repository structure

```text
cosmo_code/
├── config/
│   └── mon/                    # four evaluation modes and shared configs
├── eval/
│   ├── habitat_evaluator.py    # MP3D episode loop, SR/SCR/SPL, result export
│   └── semantic_collision.py   # safety dictionary and collision metric
├── mapping/
│   ├── feature_map.py          # dense RGB-D feature projection and fusion
│   ├── clip_cp_obstacle_map.py # coverage sets, ACI, semantic dilation
│   ├── navigator.py            # mapping/calibration/planning integration
│   └── oracle.py               # GT and Grounding DINO expert interfaces
├── planning/                   # path planning and controllers
├── planning_cpp/               # compiled shortest-path extension
├── vision_models/
│   ├── clip_dense.py           # ConvNeXt CLIP navigation features
│   ├── gclip_dense.py          # GCLIP dense semantic features
│   └── grounding_dino_detector.py
├── eval_habitat.py             # evaluation entry point
└── requirements.txt
```

The navigation map uses ConvNeXt-Large CLIP features. Conditions B, D, and F
add a GCLIP ViT-B/16 feature map for semantic obstacle reasoning. The default
configuration uses YOLOv7 for ObjectNav target detection and MobileSAM for
instance masks. Condition F additionally runs Grounding DINO as the noisy
expert.

## Installation

The experiments were developed with Python 3.8, PyTorch 2.4.1, CUDA 12.1,
Habitat-Sim 0.2.2, and Habitat-Lab 0.2.4. A CUDA-capable GPU is required by the
current implementation.

```bash
conda create -n cosmo python=3.8
conda activate cosmo

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ./planning_cpp
```

Habitat-Sim packages are sensitive to the local CUDA and compiler versions. If
the pinned package cannot be installed directly on your system, install a
Habitat-Sim 0.2.2 build compatible with your CUDA runtime before installing the
remaining requirements.

Clone YOLOv7 into the repository root:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
```

Run all commands from the `cosmo_code/` directory because the target detector
and model paths are resolved relative to the working directory.

## External data and checkpoints

No data or weights are distributed in this repository. Before evaluation,
prepare the following layout:

```text
cosmo_code/
├── datasets/
│   ├── objectnav_mp3d_v1/
│   │   └── val/content/        # per-scene MP3D ObjectNav episode files
│   └── scene_datasets/         # MP3D scene assets used by Habitat-Sim
├── weights/
│   ├── clip.pth                # ConvNeXt-Large dense CLIP checkpoint
│   ├── yolov7-e6e.pt           # YOLOv7-E6E checkpoint
│   ├── mobile_sam.pt           # MobileSAM checkpoint
│   └── grounding-dino-base/    # optional local cache for Condition F
└── yolov7/                     # YOLOv7 source checkout
```



## Running the benchmark

Run one condition at a time:

```bash
# geometry-only OneMap baseline
python eval_habitat.py --config config/mon/eval_baseline.yaml

# GCLIP argmax, without conformal prediction
python eval_habitat.py --config config/mon/eval_argmax.yaml

# open-vocabulary CoSMO with a GT semantic expert
python eval_habitat.py --config config/mon/eval_oacp.yaml

# open-vocabulary CoSMO with Grounding DINO as expert
python eval_habitat.py --config config/mon/eval_oacp_gdino.yaml
```



## Outputs

The selected YAML controls the output directory through `EvalConf.results_path`.
During evaluation the code writes:

- `state/state_<episode_id>.txt`: terminal result code for each episode;
- `trajectories/poses_<episode_id>.csv`: executed trajectory;
- `similarities/final_sim_<episode_id>.png`: final target-similarity map;
- `collision_causes_<timestamp>.csv`: semantic collision labels; and
- `oacp_calibration_<timestamp>.csv`: threshold, error, risk state, and score
  history for CoSMO runs.

At completion, the console and `logs/eval_<timestamp>.log` report SR, SPL,
result counts, runtime, and average planning time. Set `COSMO_LOG_DIR` to place
the summary logs elsewhere.



The navigation backbone derives from
[SemanticOneMap](https://github.com/Yangming911/SemanticOneMap). CoSMO adds the
open-vocabulary safety dictionary, expert interfaces, adaptive calibration,
semantic safety maps, and collision evaluation used in the paper. See
[`LICENSE`](LICENSE) for licensing information and preserve upstream notices
when redistributing derived files.

## Citation

Please cite xxxx
