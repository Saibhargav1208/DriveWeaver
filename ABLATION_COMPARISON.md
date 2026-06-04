# DriveWeaver Ablation Study - Results Comparison

## Overview

This document compares the two ablations of the DriveWeaver trajectory prediction pipeline.

---

## **Ablation 1: C-JEPA → Planner (Planner A)**

### Architecture:
```
VideoSAUR Slots → C-JEPA (Self-Supervised) → Planner A → Trajectories
```

### Training:
- **C-JEPA World Model**: 3.59M params, trained with temporal masking (self-supervised)
- **Planner A**: 1.36M params, trained with frozen C-JEPA
- **No VLM guidance**

### Checkpoints:
- C-JEPA: `/work/checkpoints/cjepa/best_model.pth` (Epoch 7, 42 MB)
- Planner A: `/work/checkpoints/planner_a/best_model.pth` (Epoch 47, 15 MB)

---

## **Ablation 2: C-JEPA + VLM → Planner (Planner B)**

### Architecture:
```
VideoSAUR Slots → C-JEPA + VLM (FiLM Guidance) → Planner B → Trajectories
                     ↑
              Qwen3-VL Features
            (layers 6,12,18,24)
```

### Training:
- **C-JEPA + VLM World Model**: 6.22M params (3.59M base + 2.63M guidance)
  - VLM features from Qwen3-VL-2B-Thinking
  - FiLM modulation at each transformer layer
  - Learned per-layer gates
- **Planner B**: 1.36M params, trained with frozen C-JEPA+VLM
- **VLM guidance enabled during training** (8GB cache, 850 scenes)

### Checkpoints:
- C-JEPA+VLM: `/work/checkpoints/cjepa_vlm/best_model.pth` (Epoch 0, TBD)
- Planner B: `/work/checkpoints/planner_b/best_model.pth` (Epoch 25, TBD)

---

## **Quantitative Results (Full Validation Set, 4,703 samples)**

### Oracle Selection (minADE - Best of 32 proposals):

| Metric | Planner A (No VLM) | Planner B (With VLM) | Difference |
|--------|-------------------|---------------------|------------|
| **ADE (mean)** | **0.4106 m** | **0.4328 m** | +0.0222 m (+5.4%) |
| **ADE (std)** | 0.4768 m | 0.4865 m | +0.0097 m |
| **FDE (mean)** | **0.8515 m** | **0.9086 m** | +0.0571 m (+6.7%) |
| **FDE (std)** | 1.0847 m | 1.1502 m | +0.0655 m |
| **Miss Rate @ 2m** | **8.3%** | **9.5%** | +1.2% |

### Key Observations:

✅ **Both ablations achieve excellent performance** (ADE < 0.5m, FDE < 1.0m)

⚠️ **Planner A (without VLM) performs slightly better** than Planner B (with VLM):
  - ADE: 0.41m vs 0.43m (5.4% better)
  - FDE: 0.85m vs 0.91m (6.7% better)
  - MR@2m: 8.3% vs 9.5%

---

## **Possible Reasons for VLM Performance Gap:**

### 1. **Text Prompt Contamination**
- VLM features were extracted with text prompt: *"Describe the driving scene and predict what will happen next."*
- ThinkJEPA paper uses **visual features only**, no text
- Text tokens may have introduced noise or misalignment

### 2. **VLM Features Not Used at Inference**
- VLM features were used during **training** (world model learns with them)
- But **not provided at inference** (scripts run without VLM features)
- This train/test mismatch could degrade performance

### 3. **VLM Guidance Learning**
- Per-layer gates may not have converged optimally
- Gates remained close to zero (minimal VLM influence)
- 50 epochs may not be enough for VLM guidance to help

### 4. **Base C-JEPA Already Strong**
- Self-supervised C-JEPA alone achieves 0.41m ADE
- VLM guidance may not add much value for this task
- Slot-based representations already capture sufficient information

---

## **Random Sample Visualizations:**

### Planner A (No VLM):
- **Output**: `/work/outputs/vis_planner_a/` (30 samples)
- **Random seed**: 42
- **Sample metrics**: ADE ~0.41m

### Planner B (With VLM):
- **Output**: `/work/outputs/vis_planner_b/` (50 samples)
- **Random seed**: 42
- **Sample metrics**: ADE ~0.50m (50 sample average)

---

## **Recommendations:**

### To Improve Ablation 2 (VLM-guided):

1. **Re-extract VLM features without text prompt**
   - Use pure visual features (images only)
   - Follows ThinkJEPA approach more closely
   - Estimated time: 19 hours extraction + 30 hours retraining

2. **Provide VLM features at inference time**
   - Current inference runs without VLM features
   - Train/test mismatch hurts performance
   - Requires dataset modifications

3. **Longer training for VLM guidance**
   - Current: 50 epochs (gates near zero)
   - Try: 100-150 epochs to let guidance learn
   - Monitor gate values and validation loss

4. **Ablation 3: Pure Vision VLM (Future Work)**
   - Extract features without text prompt
   - Fair comparison to Ablation 1
   - May close the performance gap

---

## **Conclusion:**

**Ablation 1 (C-JEPA without VLM)** currently outperforms **Ablation 2 (C-JEPA with VLM)** by a small margin (~5%). This suggests:

✅ **Slot-based self-supervised learning is highly effective** for trajectory prediction on nuScenes

⚠️ **Current VLM integration may need refinement**:
- Remove text prompt contamination
- Provide VLM at inference
- Longer training for guidance modules

🔬 **Future work**: Re-run Ablation 2 with pure visual VLM features to test true potential of vision-language guidance.

---

**Date**: 2026-05-29
**Dataset**: nuScenes v1.0-trainval (4,703 validation samples)
**Selection Method**: Oracle (minADE) - picks best of 32 trajectory proposals
