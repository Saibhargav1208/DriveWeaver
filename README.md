# DriveWeaver

DriveWeaver is a latent-space autonomous driving research codebase built around object-centric video slots, C-JEPA world modeling, and trajectory planning. The current experiment setup compares two planning pipelines:

- Ablation 1: C-JEPA world model -> Planner A
- Ablation 2: C-JEPA + cached VLM guidance -> Planner B

The key idea is simple: VideoSAUR slots represent the scene over time, C-JEPA predicts future slot states, and the planner predicts multiple future ego trajectories from the predicted latent future. Ablation 2 adds Qwen3-VL decoder features as semantic guidance inside C-JEPA.

## Current Pipeline

```text
nuScenes frames
  -> VideoSAUR slot extraction
  -> slots: [T, 11, 128]
  -> C-JEPA world model
  -> predicted future slots: [6, 11, 128]
  -> multimodal planner
  -> 32 ego trajectory proposals: [32, 6, 2]
```

Ablation 2 adds one extra offline input:

```text
nuScenes frames + prompt
  -> Qwen3-VL-2B-Thinking decoder cache
  -> vlm_old + vlm_new hidden states
  -> injected into C-JEPA layers during world-model prediction
```

## Data And Artifacts

Expected paths inside the Docker container:

```text
/work/data/slots/nuscenes_slots_full.pkl
/work/data/vlm_cache/nuscenes/
/work/checkpoints/cjepa/best_model.pth
/work/checkpoints/cjepa_vlm/
/work/checkpoints/planner_a/
/work/checkpoints/planner_b/
```

Important files:

```text
configs/cjepa/default.yaml              # baseline C-JEPA world model
configs/cjepa_vlm/default.yaml          # C-JEPA + VLM world model
configs/planner_a/default.yaml          # Ablation 1 planner
configs/planner_b/default.yaml          # Ablation 2 planner
slurm/ablation1_cjepa_planner.sh        # full Ablation 1 run
slurm/ablation2_cjepa_vlm_planner.sh    # full Ablation 2 run
training/train_cjepa.py                 # baseline world model
training/train_cjepa_vlm.py             # VLM-guided world model
training/train_planner_a.py             # planner without VLM guidance
training/train_planner_b.py             # planner with VLM-guided world model
```

## Ablation 1: C-JEPA -> Planner A

Ablation 1 is the non-VLM baseline.

```text
slots
  -> train C-JEPA
  -> /work/checkpoints/cjepa/best_model.pth
  -> freeze C-JEPA
  -> train Planner A
  -> /work/checkpoints/planner_a/best_model.pth
```

The planner never receives VLM guidance in this path. Calls into the world model use `vlm_guidance=None`.

Submit the full Slurm job:

```bash
cd /data1/work/j0987341/aadya/research/DriveWeaver
sbatch slurm/ablation1_cjepa_planner.sh
```

Manual container commands:

```bash
docker exec -it aadya_driveweaver bash
cd /work

PYTHONPATH=/work python training/train_cjepa.py --config-name default
PYTHONPATH=/work python training/train_planner_a.py --config-name default
```

Outputs:

```text
/work/checkpoints/cjepa/best_model.pth
/work/checkpoints/planner_a/best_model.pth
logs/ablation1-<jobid>.out
logs/ablation1-<jobid>.err
```

## Ablation 2: C-JEPA + VLM -> Planner B

Ablation 2 tests whether cached Qwen3-VL hidden states improve the C-JEPA world model and downstream planning.

```text
slots + VLM cache
  -> initialize from /work/checkpoints/cjepa/best_model.pth when available
  -> train C-JEPA + VLM guidance
  -> /work/checkpoints/cjepa_vlm/best_model.pth
  -> freeze C-JEPA + VLM world model
  -> train Planner B with the same cached VLM guidance
  -> /work/checkpoints/planner_b/best_model.pth
```

Submit the full Slurm job:

```bash
cd /data1/work/j0987341/aadya/research/DriveWeaver
sbatch slurm/ablation2_cjepa_vlm_planner.sh
```

Manual container commands:

```bash
docker exec -it aadya_driveweaver bash
cd /work

PYTHONPATH=/work python training/train_cjepa_vlm.py --config-name default
PYTHONPATH=/work python training/train_planner_b.py --config-name default
```

Outputs:

```text
/work/checkpoints/cjepa_vlm/best_model.pth
/work/checkpoints/planner_b/best_model.pth
logs/ablation2-<jobid>.out
logs/ablation2-<jobid>.err
```

## VLM Cache Details

The active VLM cache is produced by:

```text
scripts/cache_vlm_nuscenes_corrected.py
```

The Slurm script currently verifies that 850 cache files exist and skips extraction if the cache is complete.

VLM settings:

```text
model: Qwen/Qwen3-VL-2B-Thinking
hooked module: model.model.language_model.layers
hooked decoder layers: [6, 12, 18, 24]
hidden dim: 2048
prompt: "Describe what will happen next."
num keyframes: 16
image resolution: 384
max generated tokens: 16
cache dtype: fp16
```

Each scene cache stores two streams from the same Qwen decoder layers:

```text
vlm_old: decoder activations from the input/prompt prefill pass
vlm_new: decoder activations while generating reasoning tokens
```

Typical cached shapes:

```text
vlm_old: [4, 1, 2352, 2048]
vlm_new: [4, 1, 15, 2048]
```

The extra singleton dimension is removed by the C-JEPA loader before injection.

## VLM Injection Into C-JEPA

C-JEPA has 6 transformer layers. The cache contains 4 selected Qwen decoder layers. The current mapping intentionally does not repeat the last VLM layer:

```text
C-JEPA layer 0 <- Qwen decoder layer 6
C-JEPA layer 1 <- Qwen decoder layer 12
C-JEPA layer 2 <- Qwen decoder layer 18
C-JEPA layer 3 <- Qwen decoder layer 24
C-JEPA layer 4 <- unguided
C-JEPA layer 5 <- unguided
```

For FiLM guidance, old/new streams are projected separately and fused as:

```text
[old_summary, new_summary, abs(old_summary - new_summary), old_summary * new_summary]
```

The per-layer guidance gates are initialized at zero:

```text
gate = tanh(guidance_layer_scale)
initial gate = 0
```

This means Ablation 2 starts as an unguided C-JEPA predictor and learns how much VLM guidance to use. Early gate values near zero are expected.

## Monitoring Jobs

Check the queue:

```bash
squeue -u $USER
squeue -j <jobid>
```

Tail logs:

```bash
tail -f logs/ablation1-<jobid>.out
tail -f logs/ablation1-<jobid>.err

tail -f logs/ablation2-<jobid>.out
tail -f logs/ablation2-<jobid>.err
```

Cancel a job:

```bash
scancel <jobid>
```

For Ablation 2, useful signals to watch:

```text
train loss should trend down
validation loss should not explode
guided-layer gates should slowly move away from exactly zero
NaN/Inf warnings should be rare or absent
```

## Inference And Visualization

Ablation 1 inference:

```bash
PYTHONPATH=/work python scripts/inference_planner_a.py \
  --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
  --planner_ckpt /work/checkpoints/planner_a/best_model.pth
```

Ablation 2 inference:

```bash
PYTHONPATH=/work python scripts/inference_planner_b.py \
  --world_model_ckpt /work/checkpoints/cjepa_vlm/best_model.pth \
  --planner_ckpt /work/checkpoints/planner_b/best_model.pth \
  --vlm_cache_dir /work/data/vlm_cache/nuscenes/
```

Ablation 1 visualization:

```bash
PYTHONPATH=/work python scripts/visualize_planner_a_on_images.py \
  --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
  --planner_ckpt /work/checkpoints/planner_a/best_model.pth
```

Ablation 2 visualization:

```bash
PYTHONPATH=/work python scripts/visualize_planner_b_on_images.py \
  --world_model_ckpt /work/checkpoints/cjepa_vlm/best_model.pth \
  --planner_ckpt /work/checkpoints/planner_b/best_model.pth \
  --vlm_cache_dir /work/data/vlm_cache/nuscenes/
```

## Quick Sanity Checks

Check Python syntax inside the container:

```bash
docker exec aadya_driveweaver bash -lc "cd /work && PYTHONPYCACHEPREFIX=/tmp/driveweaver_pycache PYTHONPATH=/work python -m py_compile models/cjepa_predictor.py datasets/planner_dataset.py training/train_cjepa_vlm.py training/train_planner_b.py scripts/inference_planner_b.py scripts/visualize_planner_b_on_images.py"
```

Check VLM cache count and shape:

```bash
docker exec aadya_driveweaver bash -lc "python - <<'PY'
import glob, numpy as np
paths = sorted(glob.glob('/work/data/vlm_cache/nuscenes/*.npz'))
print('cache_count', len(paths))
z = np.load(paths[0])
print(paths[0])
print('keys', sorted(z.files))
print('vlm_old', z['vlm_old'].shape, z['vlm_old'].dtype)
print('vlm_new', z['vlm_new'].shape, z['vlm_new'].dtype)
PY"
```

Check the VLM-to-C-JEPA layer mapping:

```bash
docker exec aadya_driveweaver bash -lc "cd /work && PYTHONPATH=/work python - <<'PY'
from models.cjepa_predictor import CJEPAPredictor
m = CJEPAPredictor(num_slots=11, slot_dim=128, history_frames=4, pred_frames=6, depth=6, heads=8, dim_head=64, mlp_dim=2048, guidance_mode='film', guidance_dim=2048, guidance_hidden=512)
print(m._map_vlm_to_cjepa_layers(['L6','L12','L18','L24'], 4, 6))
print(m.transformer.guidance_layer_scale.detach().flatten().tolist())
PY"
```

Expected:

```text
['L6', 'L12', 'L18', 'L24', None, None]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```
