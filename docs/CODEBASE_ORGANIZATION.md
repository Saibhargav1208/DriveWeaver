# DriveWeaver Codebase Organization

This repository is currently organized around one working hypothesis:

> Does VLM-guided C-JEPA produce better future slots for downstream planning than C-JEPA alone?

The current nuScenes experiments are:

- `ablation1`: VideoSAUR slots -> C-JEPA -> Planner A
- `ablation2`: VideoSAUR slots + VLM guidance -> C-JEPA+VLM -> Planner B

The same hypothesis should be reusable across datasets and VLM backbones. Treat the codebase as five layers:

```text
dataset adapter
  -> slot extraction / slot cache
  -> VLM feature cache, optional
  -> world-model training / future-slot precompute
  -> planner training / inference / visualization
```

## Source Vs Artifacts

Keep source code in git:

```text
configs/
datasets/
evaluation/
experiments/
models/
scripts/
slurm/
training/
utils/
```

Keep runtime artifacts out of git:

```text
data/
checkpoints/
outputs/
logs/
*.pt
*.pth
*.pkl
*.npz
```

These are already ignored by `.gitignore`, but they are present locally because they are needed for experiments.

## Current Source Layout

```text
datasets/
  nuscenes_dataset.py       # raw image/video scene loader for slot extraction
  slot_dataset.py           # temporal windows for C-JEPA world-model training
  planner_dataset.py        # planner windows and cached future-slot datasets

models/
  cjepa_predictor.py        # C-JEPA predictor with optional VLM guidance injection
  thinkjepa.py              # wrapper around C-JEPA guidance/inference
  planner.py                # multimodal trajectory planner and scorer
  complete_pipeline.py      # world model + planner composition

training/
  train_cjepa.py            # world model without VLM
  train_cjepa_vlm.py        # world model with VLM guidance
  train_planner_a.py        # C-JEPA planner, cached or on-the-fly
  train_planner_b.py        # C-JEPA+VLM planner, cached or on-the-fly
  losses_planner.py         # proposal and proposal-score losses

scripts/
  cache_vlm_*.py            # VLM feature extraction
  precompute_*_futures.py   # frozen world-model future-slot caches
  inference_planner_*.py    # scorer-based practical inference
  inference_planner_*_oracle.py
                            # diagnostic oracle/minADE inference
  visualize_*               # image overlays

slurm/
  ablation*_*.sh            # cluster entry points

experiments/
  registry.py               # loads high-level experiment specs

configs/
  datasets/                 # dataset adapter specs
  vlms/                     # VLM feature-cache specs
  experiments/              # full hypothesis variants
```

## Dataset Expansion

New datasets should implement the same logical surfaces:

```text
raw frames or sensor observations
scene token / sample token identity
ego poses over time
slot cache in the standard pickle format
optional image paths for VLM feature extraction
optional camera projection metadata for visualization
```

The standard slot cache expected by current C-JEPA/planner code is:

```text
{
  "train": {
    scene_name: {
      "slots": [T, N, D],
      "ego_poses": [T, 4],
      "timestamps": [T],
      "sample_tokens": list[str],
    }
  },
  "val": { ... }
}
```

NAVSIM and NVIDIA Physical AI adapters should convert into this format first. That lets the current world model and planner code run unchanged.

## Model Expansion

VLM changes should live in `configs/vlms/` first:

```text
model_name
feature_dim
hook target
hooked layers
prompt
resolution
num keyframes
max generated tokens
cache dtype
```

The current C-JEPA guidance code expects dual-path cached arrays:

```text
vlm_old: [num_layers, tokens_old, feature_dim]
vlm_new: [num_layers, tokens_new, feature_dim]
```

If a new VLM has a different feature shape, adapt it in the cache extractor, not inside planner training.

## Cleanup Policy

Safe cleanup:

```bash
git config core.fileMode false
find . -type d -name __pycache__ -prune -exec rm -rf {} +
```

Risky cleanup that should be confirmed first:

```text
delete old checkpoint backups
delete old logs
delete old outputs/visualizations
move vendored videosaur to a submodule or external dependency
rename existing scripts
```

Do not delete checkpoints or outputs during a refactor unless the exact directories have been approved.
