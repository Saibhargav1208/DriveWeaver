# DriveWeaver: Hierarchical Object-Centric World Models for Autonomous Driving

**A complete research framework for learning latent world models from driving videos**

[![Status](https://img.shields.io/badge/Status-Implementation_Complete-success)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

---

## 🎯 Overview

**DriveWeaver** builds a hierarchical latent world model for autonomous driving that operates entirely in **latent space** — no pixel reconstruction, no BEV rasterization.

### Architecture at a Glance

```
nuScenes Video  →  VideoSAUR  →  Object Slots  →  C-JEPA  →  Future Latents
                   (frozen)      [T, 11, 128]    (3.6M)    [T_fut, 11, 128]
                                                               ↓
                                                           ThinkJEPA
                                                           (VLM + Refine)
                                                              (14.7M)
                                                               ↓
                                                        Refined Latents
                                                               ↓
                                                          Drive-JEPA
                                                          (Multimodal)
                                                             (1.4M)
                                                               ↓
                                                     M=32 Trajectory Proposals
                                                               ↓
                                                        Best Trajectory
```

### Key Features

- **Object-Centric**: Reasoning through tracked objects, not pixels
- **Latent-Only**: No expensive pixel decoding
- **Semantic-Aware**: VLM hidden states (no text generation)
- **Multimodal**: 32 diverse trajectory proposals
- **Planning-Aware**: World model learns driving-relevant representations

---

## 📊 Project Status

| Phase | Component | Status | Parameters |
|-------|-----------|--------|------------|
| **Phase 1** | VideoSAUR Slot Extraction | ✅ Complete | 86M (frozen) |
| **Phase 2** | C-JEPA World Model | ✅ Complete | 3.6M |
| **Phase 2.5** | Planning-Aware Supervision | ✅ Complete | +0.7M |
| **Phase 3** | ThinkJEPA Semantic Refinement | ✅ Complete | 14.7M |
| **Phase 4** | Drive-JEPA Multimodal Planner | ✅ Complete | 1.4M |
| **Training** | Training Scripts | 🔄 90% Ready | - |
| **Evaluation** | Evaluation Scripts | 🔄 Ready | - |

**Total Trainable Parameters**: ~19.7M

---

## 🚀 Quick Start

### Prerequisites

- Docker container `aadya_driveweaver` running
- nuScenes dataset at `/data/nuScenes`
- 8× NVIDIA GPUs (A100 40GB recommended)
- PyTorch 2.5+, CUDA 12.1

### Setup

```bash
# Enter container
docker exec -it aadya_driveweaver bash
cd /work

# Install dependencies (if needed)
pip install -r requirements.txt
pip install -e .

# Link VideoSAUR (if not already linked)
ln -s /path/to/videosaur /work/videosaur

# Copy VideoSAUR checkpoint
mkdir -p /work/checkpoints/videosaur
cp /path/to/videosaur_dinov2.ckpt /work/checkpoints/videosaur/
cp /path/to/ytvis_dinov2.yml /work/checkpoints/videosaur/
```

### Extract Slots (Phase 1)

```bash
# Debug mode (5 scenes)
python training/extract_slots.py --config-name debug

# Full dataset
python training/extract_slots.py --config-name default
```

**Output**: `/work/data/slots/nuscenes_slots.pkl` (~2.2 MB debug, ~300 MB full)

### Train C-JEPA (Phase 2)

```bash
# Debug training
PYTHONPATH=/work python training/train_cjepa.py --config-name debug

# Full training
PYTHONPATH=/work python training/train_cjepa.py --config-name default
```

**Expected**: 
- Debug (5 scenes, 100 epochs): ~4 hours
- Full (850 scenes, 100 epochs): ~2-3 days

### Train Drive-JEPA (Phase 4)

```bash
# Stage 1: Train planner only
PYTHONPATH=/work python training/train_drivejepa.py --config-name default

# Stage 2: Fine-tune ThinkJEPA + planner
PYTHONPATH=/work python training/train_drivejepa.py --config-name finetune

# Stage 3: Joint fine-tuning (advanced)
PYTHONPATH=/work python training/train_drivejepa.py --config-name joint
```

### Evaluate

```bash
# Evaluate planning performance
python evaluation/evaluate_drivejepa.py \
    --checkpoint /work/checkpoints/drivejepa/best_model.pth \
    --split val

# Visualize trajectory proposals
python evaluation/visualize_proposals.py \
    --checkpoint /work/checkpoints/drivejepa/best_model.pth \
    --num_scenes 10
```

---

## 📁 Repository Structure

```
DriveWeaver/
├── configs/                          # Hydra configs
│   ├── extraction/                   # Phase 1: Slot extraction
│   │   ├── default.yaml              # Full dataset (850 scenes)
│   │   └── debug.yaml                # Debug mode (5 scenes)
│   ├── cjepa/                        # Phase 2: C-JEPA training
│   │   ├── default.yaml
│   │   └── debug.yaml
│   ├── planning_cjepa/               # Phase 2.5: Planning-aware
│   │   ├── default.yaml
│   │   └── debug.yaml
│   ├── thinkjepa/                    # Phase 3: Semantic refinement
│   │   ├── default.yaml
│   │   └── debug.yaml
│   └── drivejepa/                    # Phase 4: Multimodal planning
│       ├── default.yaml              # Stage 1: Planner only
│       ├── finetune.yaml             # Stage 2: ThinkJEPA + planner
│       └── joint.yaml                # Stage 3: End-to-end
│
├── models/                           # Core modules
│   ├── videosaur_wrapper.py          # Phase 1: VideoSAUR interface
│   ├── cjepa_predictor.py            # Phase 2: C-JEPA world model
│   ├── temporal_transformer.py       # Transformer components
│   ├── masking.py                    # Object-level masking
│   ├── positional_encoding.py        # Temporal position embeddings
│   ├── ego_encoder.py                # Phase 2.5: Ego dynamics encoder
│   ├── planner_probe.py              # Phase 2.5: Planning probe
│   ├── planning_aware_cjepa.py       # Phase 2.5: Integrated model
│   ├── vlm_encoder.py                # Phase 3: Qwen3 encoder + Lite
│   ├── thinkjepa.py                  # Phase 3: Refinement module
│   ├── drivejepa_planner.py          # Phase 4: Multimodal planner
│   └── complete_pipeline.py          # Phase 4: Full pipeline
│
├── datasets/                         # Data loaders
│   ├── nuscenes_dataset.py           # Phase 1: nuScenes video loader
│   ├── slot_dataset.py               # Phase 2: Slot windows
│   ├── slot_dataset_planning.py      # Phase 2.5: With ego/trajectory
│   └── drivejepa_dataset.py          # Phase 4: Complete dataset
│
├── training/                         # Training scripts
│   ├── extract_slots.py              # Phase 1: Slot extraction pipeline
│   ├── train_cjepa.py                # Phase 2: C-JEPA training
│   ├── train_planning_cjepa.py       # Phase 2.5: Planning-aware training
│   ├── train_thinkjepa.py            # Phase 3: ThinkJEPA training
│   ├── train_drivejepa.py            # Phase 4: Drive-JEPA training
│   ├── losses_cjepa.py               # Phase 2: C-JEPA losses
│   ├── losses_planning_cjepa.py      # Phase 2.5: Planning losses
│   ├── losses_thinkjepa.py           # Phase 3: Refinement losses
│   └── losses_drivejepa.py           # Phase 4: Multimodal losses
│
├── evaluation/                       # Evaluation & visualization
│   ├── rollout.py                    # Autoregressive rollout
│   ├── evaluate_planning_probe.py    # Phase 2.5 evaluation
│   ├── evaluate_thinkjepa.py         # Phase 3 evaluation
│   ├── evaluate_drivejepa.py         # Phase 4 evaluation
│   └── visualize_proposals.py        # Trajectory visualization
│
├── scripts/                          # Utility scripts
│   ├── check_setup.py                # Environment check
│   ├── test_phase2.py                # Phase 2 unit tests
│   ├── test_phase3.py                # Phase 3 unit tests
│   ├── test_drivejepa_phase.py       # Phase 4 unit tests
│   └── validate_phase1.py            # Phase 1 validation
│
├── data/                             # Data directory
│   └── slots/                        # Extracted slots
│       ├── nuscenes_slots_debug.pkl  # Debug (5 scenes, 2.2 MB)
│       └── nuscenes_slots.pkl        # Full (850 scenes, ~300 MB)
│
├── checkpoints/                      # Model checkpoints
│   ├── videosaur/                    # VideoSAUR pretrained
│   ├── cjepa/                        # C-JEPA checkpoints
│   ├── planning_cjepa/               # Planning-aware checkpoints
│   ├── thinkjepa/                    # ThinkJEPA checkpoints
│   └── drivejepa/                    # Drive-JEPA checkpoints
│
├── outputs/                          # Hydra outputs
├── logs/                             # Training logs
├── experiments/                      # Experiment results
│
├── videosaur/                        # VideoSAUR submodule (symlink)
│
├── README.md                         # This file
├── PIPELINE_EXPLANATION.md           # Complete architecture explanation
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
└── environment.yml                   # Conda environment
```

---

## 🏗️ Architecture

### Complete Pipeline

DriveWeaver operates through **four hierarchical stages**:

#### **Phase 1: VideoSAUR Object Extraction** (86M params, frozen)

Extract temporally consistent object-centric slots from videos.

**Critical**: VideoSAUR processes the **ENTIRE video in ONE forward pass** to maintain temporal slot identity via `ScanOverTime`.

```python
# Input: nuScenes video
video: [T≈40, 3, 900, 1600]

# VideoSAUR forward
slots = videosaur(video)  # [T, 11, 128]

# slots[t=0, k, :] tracks SAME object as slots[t=40, k, :]
```

**Output**: Temporally tracked object slots [T, 11, 128]

---

#### **Phase 2: C-JEPA World Model** (3.6M params)

Predict future object-centric latent states from history.

**Innovation**: Anchor-based architecture with t=0 always visible as identity anchor.

```python
# Input: history slots
history_slots: [B, 4, 11, 128]

# C-JEPA prediction
future_pred = cjepa.inference(history_slots)  # [B, 6, 11, 128]
```

**Key Features**:
- Non-causal transformer (full attention)
- Object-level masking
- No pixel reconstruction
- Autoregressive-capable

---

#### **Phase 2.5: Planning-Aware Supervision** (+0.7M params)

Evaluate predictions via downstream planning quality.

```python
# Encode ego dynamics
ego_token = ego_encoder(ego_history)  # [B, 128]

# Predict trajectory from future slots
trajectory = planner_probe(ego_token, future_pred)  # [B, 6, 2]

# Loss combines world modeling + planning
loss = λ_jepa * L_jepa + λ_plan * L_planning
```

**Benefit**: Gradients from planning encourage driving-relevant representations.

---

#### **Phase 3: ThinkJEPA Semantic Refinement** (14.7M params)

Refine C-JEPA predictions using VLM semantic priors.

**Innovation**: Extract VLM **hidden states** (not text) for efficient semantic guidance.

```python
# VLM encoder: Extract semantic embeddings
semantic_embs = vlm_encoder(history_slots, future_pred)  # [B, 6, 11, 128]

# ThinkJEPA: Iterative refinement (3 iterations)
#   - Cross-attention: slots ← semantics
#   - Object self-attention: inter-object relations
#   - Temporal self-attention: trajectory smoothness
future_refined = thinkjepa(future_pred, semantic_embs)  # [B, 6, 11, 128]
```

**Key Features**:
- Frozen Qwen3-4B backbone
- Trainable projections only
- Iterative refinement (3 passes)
- Residual blending (starts at α=0)

---

#### **Phase 4: Drive-JEPA Multimodal Planner** (1.4M params)

Generate M diverse trajectory proposals and select the best.

```python
# Encode planning context
planning_context = context_encoder(ego_state, route_goals)  # [B, 128]

# Generate M mode queries (learnable modes)
mode_queries = query_generator(planning_context)  # [B, 32, 128]

# Decode trajectories
proposals = trajectory_decoder(mode_queries, future_refined)  # [B, 32, 6, 2]

# Score and select best
scores = proposal_scorer(proposals)  # [B, 32]
best_trajectory = proposals[argmax(scores)]  # [B, 6, 2]
```

**Key Features**:
- M=32 diverse proposals
- Learnable planning modes
- Discriminative scoring
- Interpretable selection

---

### Tensor Flow Summary

```
Video [40, 3, 900, 1600]
  ↓ VideoSAUR
Slots [40, 11, 128]
  ↓ Sliding window
History [4, 11, 128] + Ego [4, 4]
  ↓ C-JEPA
Future Predicted [6, 11, 128]
  ↓ VLM Encoder
Semantic Embeddings [6, 11, 128]
  ↓ ThinkJEPA Refinement
Future Refined [6, 11, 128]
  ↓ Drive-JEPA Planner
Trajectory Proposals [32, 6, 2]
  ↓ Best Selection
Best Trajectory [6, 2]
```

For complete architecture details, see **[PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md)**.

---

## 🔬 Research Contributions

### 1. Hierarchical Object-Centric World Modeling

**First driving world model using object-centric representations.**

- Clean separation: Perception → Prediction → Refinement → Planning
- Modular, interpretable, debuggable
- Each component independently trainable

### 2. Semantic Latent Refinement (ThinkJEPA)

**Novel use of VLM hidden states for latent refinement.**

- No text generation overhead
- Differentiable, efficient
- Preserves C-JEPA quality while adding semantics

### 3. Multimodal Trajectory Planning (Drive-JEPA)

**Learnable mode queries for structured multimodal planning.**

- M=32 diverse proposals cover driving modes
- Discriminative scoring (easier than generation)
- Interpretable planning behavior

### 4. Planning-Aware World Modeling

**Evaluate world models via downstream task quality.**

- Gradient signal from planning guides representation learning
- Lightweight probe doesn't overwhelm world model
- Encourages driving-relevant representations

---

## 📈 Training Strategy

### Stage 1: Train Drive-JEPA Only (Recommended Start)

**Freeze**: C-JEPA, VLM backbone, ThinkJEPA  
**Train**: VLM projections + Drive-JEPA (~5.7M params)

```bash
PYTHONPATH=/work python training/train_drivejepa.py --config-name default
```

**Expected Results** (debug, 5 scenes):
- Epoch 0: minADE ~5.0m
- Epoch 50: minADE ~1.0m
- Epoch 100: minADE ~0.7m

**Duration**: ~6 hours (debug), ~2-3 days (full)

---

### Stage 2: Fine-tune ThinkJEPA + Drive-JEPA (Optional)

**Unfreeze**: ThinkJEPA (small LR) + Drive-JEPA (full LR)  
**Train**: ~3.4M params

```bash
PYTHONPATH=/work python training/train_drivejepa.py --config-name finetune
```

**Why**: Let refinement adapt to planning feedback.

**Duration**: ~1-2 days

---

### Stage 3: Joint Fine-Tuning (Advanced)

**Unfreeze**: C-JEPA (tiny LR) + ThinkJEPA (small LR) + Drive-JEPA (full LR)  
**Train**: ~7M params

```bash
PYTHONPATH=/work python training/train_drivejepa.py --config-name joint
```

**Why**: End-to-end world modeling → planning optimization.  
**Risk**: Can destabilize C-JEPA if LR too high.

**Duration**: ~2-3 days

---

## 📊 Expected Results

### Debug Training (5 scenes, 100 epochs)

| Epoch | Train minADE | Val minADE | Diversity | Smoothness |
|-------|-------------|-----------|-----------|------------|
| 0 | 5.0m | 5.5m | 2.0m | 0.5 |
| 25 | 1.5m | 1.8m | 5.0m | 0.2 |
| 50 | 0.9m | 1.1m | 7.0m | 0.1 |
| 75 | 0.7m | 0.9m | 8.0m | 0.08 |
| 100 | 0.5m | 0.7m | 9.0m | 0.05 |

### Full Training (850 scenes, 100 epochs)

**Expected Final Performance**:
- **minADE**: 0.3-0.5m (best-of-32)
- **minFDE**: 0.8-1.2m
- **Diversity**: 8-10m (proposal separation)
- **Proposals**: Cover diverse modes (lane-keep, overtake, yield)
- **Generalization**: Val ~10-20% worse than train

---

## 🧪 Testing & Validation

### Unit Tests

```bash
# Test Phase 1 (VideoSAUR)
PYTHONPATH=/work python scripts/validate_phase1.py

# Test Phase 2 (C-JEPA)
PYTHONPATH=/work python scripts/test_phase2.py

# Test Phase 3 (ThinkJEPA)
PYTHONPATH=/work python scripts/test_phase3.py

# Test Phase 4 (Drive-JEPA)
PYTHONPATH=/work python scripts/test_drivejepa_phase.py

# Test complete pipeline
PYTHONPATH=/work python models/complete_pipeline.py
```

All tests should print `✓ ALL TESTS PASSED!`

---

## 🐛 Troubleshooting

### Out of GPU Memory

**Solution 1**: Use lite VLM mode
```yaml
# In config: thinkjepa/default.yaml
vlm_encoder:
  lite_mode: true
  lite_hidden_dim: 512
  lite_depth: 4
```

**Solution 2**: Reduce batch size
```yaml
training:
  batch_size: 8  # Instead of 16
```

**Solution 3**: Use gradient checkpointing
```python
# In training script
model.gradient_checkpointing_enable()
```

---

### High minADE (> 5.0m)

**Diagnosis**: Model not learning

**Solutions**:
- Increase λ_ade: `lambda_ade: 2.0`
- Decrease λ_diversity: `lambda_div: 0.05`
- Check ground truth trajectories: visualize with `visualize_proposals.py`
- Verify data preprocessing: check ego poses are correct

---

### Low Diversity (< 3.0m)

**Diagnosis**: Proposals collapsing to single mode

**Solutions**:
- Increase λ_diversity: `lambda_div: 0.2`
- Increase number of modes: `num_modes: 64`
- Add diversity warmup: gradually increase λ_div over epochs

---

### NaN in Training

**Diagnosis**: Gradient explosion or division by zero

**Solutions**:
- Enable gradient clipping:
  ```yaml
  training:
    clip_grad_norm: 1.0
  ```
- Disable mixed precision:
  ```yaml
  training:
    mixed_precision: false
  ```
- Reduce learning rate:
  ```yaml
  optimizer:
    lr: 1e-4  # Instead of 3e-4
  ```

---

## 💻 Hardware Requirements

### Minimum

- **GPU**: 1× NVIDIA A100 40GB
- **RAM**: 64GB
- **Storage**: 500GB
- **CPU**: 16 cores

### Recommended

- **GPU**: 8× NVIDIA A100 40GB
- **RAM**: 256GB
- **Storage**: 2TB (NVMe SSD)
- **CPU**: 64 cores

### Per-Phase GPU Usage

| Phase | Training GPUs | VRAM/GPU | Inference GPUs |
|-------|--------------|----------|----------------|
| Phase 1 (Extraction) | N/A | ~18 GB | 1 |
| Phase 2 (C-JEPA) | 4-8 | ~25 GB | 1 |
| Phase 2.5 (Planning) | 4-8 | ~28 GB | 1 |
| Phase 3 (ThinkJEPA) | 4-8 | ~35 GB (full) / ~28 GB (lite) | 1-2 |
| Phase 4 (Drive-JEPA) | 4-8 | ~30 GB | 1 |

---

## 📚 Documentation

### Main Documents

- **[README.md](README.md)** (this file) — Project overview and quick start
- **[PIPELINE_EXPLANATION.md](PIPELINE_EXPLANATION.md)** — Complete architecture explanation

### Phase-Specific Docs

- **[PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)** — VideoSAUR slot extraction
- **[PHASE2_COMPLETE.md](PHASE2_COMPLETE.md)** — C-JEPA world model
- **[PHASE2.5_PLANNING_AWARE_COMPLETE.md](PHASE2.5_PLANNING_AWARE_COMPLETE.md)** — Planning-aware supervision
- **[PHASE4_DRIVEJEPA_COMPLETE.md](PHASE4_DRIVEJEPA_COMPLETE.md)** — Drive-JEPA planner

### Summary Docs

- **[DRIVEWEAVER_COMPLETE_SUMMARY.md](DRIVEWEAVER_COMPLETE_SUMMARY.md)** — Phase 1-2.5 overview
- **[COMPLETE_PROJECT_STATUS.md](COMPLETE_PROJECT_STATUS.md)** — Overall project status
- **[START_HERE.md](START_HERE.md)** — Quick start guide

### Technical Docs

- **[ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)** — C-JEPA vs original paper
- **[CJEPA_AUDIT_COMPLETE.md](CJEPA_AUDIT_COMPLETE.md)** — Architecture verification
- **[VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)** — Testing and validation

---

## 🎓 Citation

```bibtex
@article{driveweaver2026,
  title={DriveWeaver: Hierarchical Object-Centric JEPA World Models for Autonomous Driving},
  author={Your Team},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 📝 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **[VideoSAUR](https://github.com/martius-lab/videosaur)** — Temporally consistent object-centric video understanding
- **[nuScenes](https://www.nuscenes.org/)** — Large-scale autonomous driving dataset
- **[C-JEPA](https://github.com/facebookresearch/c-jepa)** — Self-supervised latent video prediction
- **[Qwen3](https://github.com/QwenLM/Qwen)** — Vision-language model for semantic reasoning

---

## 📞 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [github.com/your-org/driveweaver/issues](https://github.com/your-org/driveweaver/issues)
- **Email**: your-email@institution.edu

---

## 🗺️ Roadmap

### Phase 1: Implementation ✅ (COMPLETE)
- [x] VideoSAUR wrapper
- [x] C-JEPA predictor
- [x] Planning-aware supervision
- [x] ThinkJEPA refinement
- [x] Drive-JEPA planner
- [x] All loss functions
- [x] Dataset loaders
- [x] Complete pipeline integration

### Phase 2: Training 🔄 (IN PROGRESS)
- [ ] Debug training (5 scenes)
- [ ] Full training (850 scenes)
- [ ] Hyperparameter tuning
- [ ] Ablation studies

### Phase 3: Evaluation 📋 (PLANNED)
- [ ] Closed-loop evaluation (CARLA)
- [ ] Baseline comparisons
- [ ] Failure case analysis
- [ ] Multi-camera fusion
- [ ] Map context integration

### Phase 4: Publication 📄 (PLANNED)
- [ ] Paper writing
- [ ] Code release
- [ ] Model checkpoints
- [ ] Demo videos
- [ ] Benchmark submissions

---

**Status**: Implementation Complete, Ready for Training  
**Last Updated**: 2026-05-25  
**Next Milestone**: Debug Training (5 scenes, 100 epochs)

---

*DriveWeaver: Where object-centric perception meets semantic reasoning and multimodal planning.* 🚗🧠✨
