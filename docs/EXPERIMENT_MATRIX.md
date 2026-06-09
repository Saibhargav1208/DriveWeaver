# Experiment Matrix

The hypothesis should be evaluated as a matrix:

```text
dataset x world-model variant x VLM feature source x planner/scorer training
```

## Baseline Variants

```text
ablation1:
  dataset slots -> C-JEPA -> future slots -> planner

ablation2:
  dataset slots + VLM features -> C-JEPA+VLM -> future slots -> planner
```

For every dataset/model pair, report both:

```text
oracle metrics:
  best-of-32 minADE/minFDE, diagnostic only

scorer metrics:
  proposal_scores.argmax ADE/FDE/MR, practical inference
```

The oracle metric answers: did the proposal set contain a good trajectory?

The scorer metric answers: can the trained model choose it without ground truth?

## Dataset Specs

Current and planned dataset specs live in:

```text
configs/datasets/nuscenes.yaml
configs/datasets/navsim.yaml
configs/datasets/nvidia_physical_ai.yaml
```

Each dataset should eventually provide:

```text
raw_root
slot_cache_path
vlm_cache_dir
future_slot_cache_dir
checkpoint_root
output_root
camera/view metadata for visualization, if available
```

## VLM Specs

VLM specs live in:

```text
configs/vlms/qwen3_vl_2b_thinking.yaml
configs/vlms/vlm_template.yaml
```

The feature-cache interface should stay stable even when the VLM changes:

```text
vlm_old: cached input/prefill features
vlm_new: cached generated/reasoning features
```

## Experiment Specs

Full variants live in `configs/experiments/`.

Current concrete specs:

```text
nuscenes_ablation1_cjepa.yaml
nuscenes_ablation2_cjepa_vlm_qwen3.yaml
```

Templates for planned datasets:

```text
navsim_ablation1_cjepa.yaml
navsim_ablation2_cjepa_vlm_template.yaml
nvidia_physical_ai_ablation1_cjepa.yaml
nvidia_physical_ai_ablation2_cjepa_vlm_template.yaml
```

Use the helper to inspect a spec:

```bash
PYTHONPATH=/work python -m experiments.registry   configs/experiments/nuscenes_ablation2_cjepa_vlm_qwen3.yaml
```
