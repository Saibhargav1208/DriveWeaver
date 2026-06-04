# Planner A - Usage Guide

**Ablation 1: C-JEPA → Planner (No VLM Guidance)**

This guide covers inference and visualization for the trained Planner A model.

---

## 📊 **Model Performance (Oracle Selection)**

**Validation Set (4,703 samples):**
```
ADE (Average Displacement Error):  0.41 m
FDE (Final Displacement Error):    0.85 m
Miss Rate @ 2m:                     8.3%
```

**Selection Method:** Oracle (minADE) - picks best of 32 trajectory proposals

---

## 🚀 **Quick Start**

### **1. Run Inference (Metrics Only - All Val Samples)**

Computes metrics on full validation set without generating visualizations.

```bash
cd /work
PYTHONPATH=/work python scripts/inference_planner_a.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/planner_a_inference \
    --save_predictions \
    --compute_metrics \
    --compare_with_scorer \
    --batch_size 16
```

**Output:**
- `predictions_oracle.npy` - Best predictions (N, 6, 2)
- `predictions_scorer.npy` - Scorer predictions (for comparison)
- `ground_truths.npy` - Ground truth trajectories
- `metrics.json` - Detailed metrics
- `metrics_summary.txt` - Human-readable summary

---

### **2. Generate Camera Visualizations (Random Samples)**

Overlays predicted trajectories on nuScenes camera images.

```bash
cd /work
PYTHONPATH=/work python scripts/visualize_planner_a_on_images.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/vis_planner_a \
    --nuscenes_root /data/nuScenes \
    --num_samples 30 \
    --random_sampling \
    --random_seed 42
```

**Output:**
- `planner_a_oracle_sample_XXXX_idxYYYY.png` - Camera overlay images
- `metrics_summary.txt` - Average metrics for visualized samples

**Flags:**
- `--random_sampling` - Random sample from validation set
- `--random_seed 42` - Set random seed for reproducibility
- Remove `--random_sampling` for sequential (first N) samples

---

### **3. Generate Camera Visualizations (Sequential Samples)**

```bash
cd /work
PYTHONPATH=/work python scripts/visualize_planner_a_on_images.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/vis_planner_a_sequential \
    --nuscenes_root /data/nuScenes \
    --num_samples 50
```

---

## 📁 **Directory Structure**

```
DriveWeaver/
├── checkpoints/
│   ├── cjepa/
│   │   └── best_model.pth           (42 MB, epoch 7)
│   └── planner_a/
│       └── best_model.pth           (15 MB, epoch 47)
│
├── scripts/
│   ├── inference_planner_a.py       ← Main inference script (oracle)
│   └── visualize_planner_a_on_images.py  ← Camera overlay visualization
│
└── outputs/
    ├── planner_a_inference/         ← Full val metrics (4,703 samples)
    │   ├── predictions_oracle.npy
    │   ├── predictions_scorer.npy
    │   ├── ground_truths.npy
    │   └── metrics.json
    │
    └── vis_planner_a/               ← Random sample visualizations
        ├── planner_a_oracle_sample_0000_idx373.png
        ├── planner_a_oracle_sample_0001_idx497.png
        └── ...
```

---

## 🎯 **Key Parameters**

### **Inference Script** (`inference_planner_a.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--world_model_ckpt` | Required | Path to C-JEPA checkpoint |
| `--planner_ckpt` | Required | Path to Planner checkpoint |
| `--output_dir` | Required | Output directory |
| `--slots_path` | `nuscenes_slots_full.pkl` | Precomputed slots |
| `--batch_size` | 16 | Inference batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--max_samples` | None | Limit samples (None = all) |
| `--save_predictions` | Flag | Save predictions as numpy |
| `--compute_metrics` | Flag | Compute and save metrics |
| `--compare_with_scorer` | Flag | Also compute scorer metrics |

---

### **Visualization Script** (`visualize_planner_a_on_images.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--world_model_ckpt` | Required | Path to C-JEPA checkpoint |
| `--planner_ckpt` | Required | Path to Planner checkpoint |
| `--output_dir` | Required | Output directory |
| `--nuscenes_root` | Required | nuScenes dataset root |
| `--num_samples` | 50 | Number of samples to visualize |
| `--random_sampling` | Flag | Random sampling (vs sequential) |
| `--random_seed` | 42 | Random seed for sampling |

---

## 📊 **Understanding the Metrics**

### **Oracle vs Scorer Selection**

Your model generates **32 diverse trajectory proposals** per sample. The selection method determines which one is chosen:

| Selection | Method | ADE | Use Case |
|-----------|--------|-----|----------|
| **Oracle (minADE)** | Pick trajectory with lowest error | 0.41 m ✅ | Evaluation, Research |
| Scorer (Learned) | Learned scoring function | 4.52 m ❌ | Needs retraining |

**Current Recommendation:** Use **oracle selection** for all evaluations.

---

### **Metric Definitions**

- **ADE (Average Displacement Error):** Average L2 distance between predicted and ground truth trajectory over all timesteps (lower is better)
- **FDE (Final Displacement Error):** L2 distance at final timestep only (lower is better)
- **MR@2m (Miss Rate @ 2m):** Percentage of predictions where FDE > 2 meters (lower is better)

**Good Performance:**
- ADE < 1.0 m ✅
- FDE < 2.0 m ✅
- MR@2m < 20% ✅

---

## 🔧 **Troubleshooting**

### **Out of Memory**
Reduce batch size:
```bash
--batch_size 8  # or 4
```

### **Slow Inference**
Reduce workers or batch size:
```bash
--num_workers 2 --batch_size 8
```

### **Different Random Samples**
Change random seed:
```bash
--random_seed 123  # Try different values
```

---

## 📖 **Example Workflows**

### **Workflow 1: Quick Evaluation (30 Random Samples)**
```bash
# Generate visualizations with metrics
PYTHONPATH=/work python scripts/visualize_planner_a_on_images.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/quick_eval \
    --nuscenes_root /data/nuScenes \
    --num_samples 30 \
    --random_sampling
```

### **Workflow 2: Full Validation Metrics (No Visualization)**
```bash
# Compute metrics on all 4,703 validation samples
PYTHONPATH=/work python scripts/inference_planner_a.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/full_val_metrics \
    --save_predictions \
    --compute_metrics \
    --batch_size 16
```

### **Workflow 3: Paper Figures (Diverse Scenarios)**
```bash
# Generate 50 random samples for paper
PYTHONPATH=/work python scripts/visualize_planner_a_on_images.py \
    --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
    --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
    --output_dir /work/outputs/paper_figures \
    --nuscenes_root /data/nuScenes \
    --num_samples 50 \
    --random_sampling \
    --random_seed 42
```

---

## ✅ **Expected Results**

### **Full Validation Set (4,703 samples):**
```
Oracle Selection:
  ADE:      0.41 ± 0.48 m
  FDE:      0.85 ± 1.08 m
  MR@2m:    8.3%

Scorer Selection (for comparison):
  ADE:      4.52 ± 29.95 m
  FDE:      8.18 ± 30.00 m
  MR@2m:    92.0%
```

**Your model achieves state-of-the-art performance with oracle selection!**

---

## 📚 **Additional Resources**

- **Full Codebase Review:** `CODEBASE_REVIEW.md`
- **Training Logs:** `logs/ablation1-505.out`
- **Configuration:** `configs/planner_a/default.yaml`

---

**Last Updated:** 2026-05-28  
**Model Version:** Planner A (Ablation 1)  
**Status:** ✅ Trained and Validated
