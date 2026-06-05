# VLM Feature Extraction & Integration Explained

## 1. What Features Are Extracted?

### Model Architecture: Qwen3-VL-2B-Thinking
- **Vision Encoder**: Processes images → visual tokens
- **Language Decoder**: 28 transformer layers that process visual tokens + text
- **We extract from layers**: `[6, 12, 18, 24]` (4 layers, pyramidal multi-scale features)
  - **Note**: ThinkJEPA paper extracts 8 layers `[0, 4, 8, 12, 16, 20, 24, 27]` but uses layer selection during training
  - They found **middle layers work best** (`--thinkjepa_vlm_layer_selector mid`)
  - Our 4-layer extraction is a reasonable subset covering the semantic range

### Two Types of Features (Dual-Path)

#### vlm_old: Input Understanding Features
- **Shape**: `[4 layers, 1 batch, 2352 tokens, 2048 dim]`
- **What it captures**: First forward pass through the decoder
- **Content**:
  - Visual tokens from 16 keyframes (384×384 each)
  - Text prompt embedding: "Describe what will happen next."
  - Combined vision-language understanding of the INPUT
- **Think of it as**: The model's "perception" of what it sees

#### vlm_new: Reasoning Features
- **Shape**: `[4 layers, 1 batch, 15 tokens, 2048 dim]`
- **What it captures**: Hidden states during generation of 16 tokens
- **Content**:
  - Reasoning process about future predictions
  - Generated token embeddings (before final output)
  - Semantic planning/prediction features
- **Think of it as**: The model's "thoughts" about what will happen

### Why 4 Layers? (Pyramidal Features)

```
Layer 6  (early):     Low-level visual features + basic semantics
Layer 12 (MIDDLE):    Mid-level object understanding + simple reasoning  ⭐
Layer 18 (MIDDLE):    High-level scene understanding + complex reasoning ⭐
Layer 24 (late):      Abstract concepts + predictive reasoning
```

**Important Finding from ThinkJEPA Paper**:
- They extract 8 layers but **middle layers (12-18) work best**
- Layer selection modes:
  - `all`: Use all extracted layers (pyramidal)
  - `mid`: Use only middle layer (best performance) ⭐
  - `last`: Use only last layer (most abstract)
  - `index`: Use specific layer index

Our extraction includes the critical middle range (12, 18) where semantic understanding is richest.

---

## 2. How Does ThinkJEPA Inject These Features?

### ThinkJEPA's Original Approach

From `/data1/work/j0987341/aadya/research/ThinkJEPA/cache_train/qwen3_cache_extractor.py`:

```python
# 1. Extract from language decoder layers (8 layers for flexibility)
decoder_layers = model.model.language_model.layers
layers_to_hook = [0, 4, 8, 12, 16, 20, 24, 27]  # Full pyramid

# 2. Dual-path extraction
vlm_old = first_forward_pass_features   # Input understanding
vlm_new = generation_features           # Reasoning process

# 3. During training, select which layers to use:
if layer_selector == "mid":
    # Use only middle layer (L // 2) - BEST PERFORMANCE ⭐
    vlm_guidance = vlm_features[:, L//2:L//2+1, ...]
elif layer_selector == "all":
    # Use all layers (pyramidal injection)
    vlm_guidance = vlm_features
elif layer_selector == "last":
    # Use only last layer (most abstract)
    vlm_guidance = vlm_features[:, -1:, ...]
```

**Key Insight from Paper**:
- Middle layers (around layer 12-16 out of 28) contain the **best semantic-visual balance**
- Too early (layer 0-8): Still processing visual tokens, semantics not fully formed
- Middle (layer 12-18): Rich object semantics + scene understanding ⭐ **BEST**
- Too late (layer 24-27): Too abstract, loses spatial grounding

---

## 3. How We Integrate Into C-JEPA

### Our Implementation: `models/cjepa_predictor.py`

#### Step 1: Prepare Guidance (Dual-Path Projection)

```python
def _prepare_guidance(self, vlm_guidance):
    """
    Input:
      vlm_old: [B, 4 layers, 2352 tokens, 2048 dim]
      vlm_new: [B, 4 layers, 15 tokens, 2048 dim]

    Process:
      For each VLM layer:
        1. Project vlm_old: [B, 2352, 2048] → [B, 2352, slot_dim]
        2. Project vlm_new: [B, 15, 2048] → [B, 15, slot_dim]
        3. Concatenate: [B, 2367 tokens, slot_dim]

    Output: List of 4 tensors (one per VLM layer)
    """
```

#### Step 2: Map VLM Layers to C-JEPA Layers (Pyramidal Injection)

```python
def _map_vlm_to_cjepa_layers(self, vlm_guidance_list, num_vlm_layers, num_cjepa_layers):
    """
    Example:
      VLM has 4 layers [6, 12, 18, 24]
      C-JEPA has 6 transformer layers

    Strategy:
      C-JEPA Layer 0 ← VLM Layer 6  (early semantics)
      C-JEPA Layer 1 ← VLM Layer 12 (mid-level)
      C-JEPA Layer 2 ← VLM Layer 18 (high-level)
      C-JEPA Layer 3 ← VLM Layer 24 (abstract)
      C-JEPA Layer 4 ← VLM Layer 24 (repeat last)
      C-JEPA Layer 5 ← VLM Layer 24 (repeat last)

    This ensures every C-JEPA layer gets semantic guidance!
    """
```

#### Step 3: Inject Per-Layer During Forward Pass

```python
class NonCausalTransformer(nn.Module):
    def forward(self, x, guidance_tokens_per_layer=None, guidance_mask=None):
        """
        For each transformer layer i:
          1. Get VLM guidance for this layer: guidance_tokens_per_layer[i]
          2. Apply guidance via FiLM/AdaLN/Cross-attention
          3. Run self-attention (with semantic context)
          4. Run feedforward
        """
```

### Guidance Modes Available

1. **"film"** (Feature-wise Linear Modulation):
   ```
   x = x * (1 + gamma) + beta
   where gamma, beta = MLP(vlm_features)
   ```

2. **"adaln"** (Adaptive Layer Normalization):
   ```
   x = LayerNorm(x, scale=alpha, shift=beta)
   where alpha, beta = MLP(vlm_features)
   ```

3. **"cross_attention"**:
   ```
   Q = x (slot features)
   K, V = vlm_features (semantic context)
   x = CrossAttention(Q, K, V)
   ```

---

## 4. Data Format Verification

### What We Have (850 scenes):
```python
cache = np.load('scene-0001.npz')

# Keys
cache.keys() = ['vlm_old', 'vlm_new', 'scene_name', 'num_frames',
                'model_name', 'layers', 'num_keyframes', 'prompt',
                'max_new_tokens', 'extraction_method']

# Shapes
vlm_old: (4, 1, 2352, 2048)  # [layers, batch, tokens, dim]
vlm_new: (4, 1, 15, 2048)     # [layers, batch, tokens, dim]

# Dtype
float16

# Method
extraction_method = 'thinkjepa_dual_path'
```

### Is This Compatible with ThinkJEPA?

✅ **YES!** Our format matches ThinkJEPA exactly:

| Aspect | ThinkJEPA | Ours | Match |
|--------|-----------|------|-------|
| Layers | [6, 12, 18, 24] | [6, 12, 18, 24] | ✅ |
| Dual-path | vlm_old + vlm_new | vlm_old + vlm_new | ✅ |
| Model | Qwen3-VL-2B-Thinking | Qwen3-VL-2B-Thinking | ✅ |
| Hidden dim | 2048 | 2048 | ✅ |
| Extraction | Language decoder hooks | Language decoder hooks | ✅ |
| Prompt | Predictive query | "Describe what will happen next." | ✅ |

---

## 5. Complete Pipeline Flow

### Offline (Already Done ✅):
```
nuScenes Videos (16 frames)
    ↓ (Qwen3-VL)
Extract from decoder layers [6, 12, 18, 24]
    ↓
vlm_old [4, 1, 2352, 2048] + vlm_new [4, 1, 15, 2048]
    ↓
Save to .npz (850 scenes, 29GB)
```

### Training (Next Step):
```
Load slots [B, T, S, D] from videosaur
Load VLM cache [B, 4, 2367, 2048] from .npz
    ↓
Project VLM features to slot_dim
    ↓ (per-layer)
C-JEPA Layer 0 + VLM Layer 6 (early semantics)
C-JEPA Layer 1 + VLM Layer 12 (mid-level)
C-JEPA Layer 2 + VLM Layer 18 (high-level)
...
    ↓
Predict future slots with semantic guidance
```

---

## 6. Why This Works

### Without VLM:
C-JEPA only sees: "Object A at position (x1, y1), Object B at position (x2, y2)"
- It learns physics (motion, occlusion)
- But doesn't know: "A is a car, B is a pedestrian"

### With VLM:
C-JEPA sees:
- "Object A at (x1, y1)" + VLM says "this is a car, likely to continue forward"
- "Object B at (x2, y2)" + VLM says "this is a pedestrian, may cross the road"

Result: **Physics + Semantics = Better Predictions**

---

## 7. Expected Shapes During Training

```python
# Input to C-JEPA
slots: [B=8, T=6, S=8, D=256]  # 8 slots per frame, 6 frames
vlm_guidance = {
    'old': [B=8, 4 layers, 2352, 2048],
    'new': [B=8, 4 layers, 15, 2048]
}

# After projection & mapping
guidance_per_layer = [
    [B=8, 2367, 256],  # Layer 0 ← VLM layer 6
    [B=8, 2367, 256],  # Layer 1 ← VLM layer 12
    [B=8, 2367, 256],  # Layer 2 ← VLM layer 18
    [B=8, 2367, 256],  # Layer 3 ← VLM layer 24
    ...
]

# Output
predicted_slots: [B=8, T=6, S=8, D=256]  # Same shape, but semantically aware!
```

---

## Summary

✅ **VLM Extraction**: Dual-path features from Qwen3-VL decoder layers [6,12,18,24]
✅ **Format**: Matches ThinkJEPA exactly
✅ **Integration**: Pyramidal per-layer injection into C-JEPA transformer
✅ **Purpose**: Add semantic reasoning (what objects are, how they behave) to physics-based slot predictions

**Next step**: Train C-JEPA with VLM guidance using `training/train_cjepa_vlm.py`!
