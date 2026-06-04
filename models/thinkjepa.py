"""
ThinkJEPA: Cortex-Guided C-JEPA Predictor (Paper-faithful)

Follows the architecture from:
    "ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model"
    Zhang et al., 2026.  arXiv:2603.22281

The VLM (Qwen3-VL) processes actual video frames OFFLINE and caches hidden states.
During training, these cached features are injected INTO the C-JEPA predictor
at every transformer layer via FiLM conditioning (or cross-attention / AdaLN).

Architecture:
    nuScenes frames → Qwen3-VL (offline, cached .npz)
                            ↓
    history_slots → C-JEPA predictor ← VLM guidance injected per-layer
                            ↓
                    predicted future_slots

The C-JEPA predictor remains the core world model. VLM guidance provides
semantic/visual reasoning that helps the predictor make better future predictions.
When guidance is absent (vlm_guidance=None), output is identical to vanilla C-JEPA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig

from models.cjepa_predictor import CJEPAPredictor, create_cjepa_from_config


class ThinkJEPA(nn.Module):
    """
    ThinkJEPA: C-JEPA with per-layer VLM guidance injection.

    This is NOT a post-hoc refiner. VLM features are injected INTO the
    C-JEPA transformer at every layer, conditioning the prediction process.

    The guidance modules (projection, fusion MLPs, layer gates) are initialized
    to produce zero contribution, so at init the model behaves identically to
    the pre-trained C-JEPA. Training gradually enables the guidance signal.

    Training strategy:
        Phase A: Freeze C-JEPA base weights, train only guidance modules
        Phase B: Unfreeze C-JEPA with small LR for joint fine-tuning
    """

    def __init__(
        self,
        cjepa: CJEPAPredictor,
    ):
        super().__init__()
        self.cjepa = cjepa

    def freeze_cjepa_base(self):
        """Freeze all non-guidance C-JEPA parameters."""
        for name, p in self.cjepa.named_parameters():
            if 'guidance' not in name:
                p.requires_grad = False

    def unfreeze_cjepa_base(self):
        """Unfreeze C-JEPA base parameters for joint fine-tuning."""
        for p in self.cjepa.parameters():
            p.requires_grad = True

    def get_guidance_params(self):
        """Return only the guidance-related parameters."""
        return [
            p for name, p in self.cjepa.named_parameters()
            if 'guidance' in name
        ]

    def get_base_params(self):
        """Return non-guidance parameters."""
        return [
            p for name, p in self.cjepa.named_parameters()
            if 'guidance' not in name
        ]

    def count_params(self) -> Dict[str, int]:
        """Count parameters by group."""
        guidance = sum(
            p.numel() for name, p in self.cjepa.named_parameters()
            if 'guidance' in name
        )
        base = sum(
            p.numel() for name, p in self.cjepa.named_parameters()
            if 'guidance' not in name
        )
        trainable = sum(
            p.numel() for p in self.cjepa.parameters() if p.requires_grad
        )
        return {'guidance': guidance, 'base': base, 'trainable': trainable}

    def get_guidance_gate_values(self) -> list:
        """Return current per-layer guidance gate values for monitoring."""
        if hasattr(self.cjepa.transformer, 'guidance_layer_scale'):
            scales = self.cjepa.transformer.guidance_layer_scale
            return torch.tanh(scales).squeeze().tolist()
        return []

    def forward(
        self,
        history_slots: torch.Tensor,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (training mode with masking).

        Args:
            history_slots: [B, T_hist, N, D]
            vlm_guidance: dict with 'vlm_features' [B, S_vlm, vlm_dim]

        Returns:
            out: [B, T_total, N, D] predicted slots (history + future)
            masked_indices: indices of masked slots
        """
        return self.cjepa(history_slots, vlm_guidance=vlm_guidance)

    @torch.no_grad()
    def inference(
        self,
        history_slots: torch.Tensor,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Inference without masking.

        Args:
            history_slots: [B, T_hist, N, D]
            vlm_guidance: dict with 'vlm_features' [B, S_vlm, vlm_dim]

        Returns:
            future_slots: [B, T_fut, N, D]
        """
        return self.cjepa.inference(history_slots, vlm_guidance=vlm_guidance)

    @torch.no_grad()
    def inference_both(
        self,
        history_slots: torch.Tensor,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference with AND without guidance for comparison.

        Returns:
            (guided_prediction, unguided_prediction) each [B, T_fut, N, D]
        """
        guided = self.cjepa.inference(history_slots, vlm_guidance=vlm_guidance)
        unguided = self.cjepa.inference(history_slots, vlm_guidance=None)
        return guided, unguided


def create_thinkjepa_from_config(cfg: DictConfig) -> ThinkJEPA:
    """
    Create ThinkJEPA model from config.

    Reads guidance parameters from cfg.vlm_guidance if present.
    """
    vlm_cfg = cfg.get('vlm_guidance', {})
    guidance_mode = vlm_cfg.get('guidance_mode', None) if vlm_cfg.get('enable', False) else None
    guidance_dim = vlm_cfg.get('vlm_dim', None)
    guidance_hidden = vlm_cfg.get('guidance_hidden', 512)

    cjepa = create_cjepa_from_config(
        cfg,
        guidance_mode=guidance_mode,
        guidance_dim=guidance_dim,
        guidance_hidden=guidance_hidden,
    )

    return ThinkJEPA(cjepa=cjepa)


def load_cjepa_checkpoint_into_thinkjepa(
    model: ThinkJEPA,
    checkpoint_path: str,
    device: str = 'cpu',
) -> Tuple[list, list]:
    """
    Load a pre-trained C-JEPA (or planning_cjepa) checkpoint into ThinkJEPA.

    Handles:
    - Standalone cjepa checkpoints (bare keys)
    - planning_cjepa checkpoints (keys prefixed with 'cjepa.')

    Guidance modules will be missing from old checkpoints — that's expected
    (they start at zero contribution anyway).

    Returns:
        (missing_keys, unexpected_keys)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    # Strip 'cjepa.' prefix if from planning_cjepa checkpoint
    if any(k.startswith('cjepa.') for k in state.keys()):
        state = {
            k.removeprefix('cjepa.'): v
            for k, v in state.items()
            if k.startswith('cjepa.')
        }

    # Load into the cjepa sub-module (strict=False allows new guidance keys to be missing)
    missing, unexpected = model.cjepa.load_state_dict(state, strict=False)

    # Filter out expected missing keys (guidance modules)
    unexpected_real = [k for k in unexpected if 'guidance' not in k]
    missing_base = [k for k in missing if 'guidance' not in k]

    if missing_base:
        print(f"  WARNING: Missing base keys: {missing_base}")
    if unexpected_real:
        print(f"  WARNING: Unexpected keys: {unexpected_real}")

    guidance_missing = [k for k in missing if 'guidance' in k]
    if guidance_missing:
        print(f"  Guidance modules (new, zero-init): {len(guidance_missing)} keys")

    return missing, unexpected


if __name__ == "__main__":
    print("Testing ThinkJEPA (paper-faithful) Architecture...")

    B, T_hist, T_fut, N, D = 2, 4, 6, 11, 128
    VLM_DIM = 3584

    # Create model with FiLM guidance
    cjepa = CJEPAPredictor(
        num_slots=N,
        slot_dim=D,
        history_frames=T_hist,
        pred_frames=T_fut,
        num_masked_slots=2,
        depth=2,
        heads=4,
        dim_head=32,
        mlp_dim=512,
        dropout=0.0,
        guidance_mode="film",
        guidance_dim=VLM_DIM,
        guidance_hidden=256,
    )
    model = ThinkJEPA(cjepa=cjepa)

    # Test without guidance (should behave like vanilla C-JEPA)
    history = torch.randn(B, T_hist, N, D)
    future_no_guidance = model.inference(history, vlm_guidance=None)
    assert future_no_guidance.shape == (B, T_fut, N, D)
    print(f"  No guidance: {future_no_guidance.shape} ✓")

    # Test with guidance
    vlm_features = torch.randn(B, 480, VLM_DIM)  # 480 VLM tokens
    vlm_guidance = {'vlm_features': vlm_features}
    future_guided = model.inference(history, vlm_guidance=vlm_guidance)
    assert future_guided.shape == (B, T_fut, N, D)
    print(f"  With guidance: {future_guided.shape} ✓")

    # At init, guided ≈ unguided (gates start at 0)
    delta = (future_guided - future_no_guidance).abs().mean().item()
    print(f"  Initial delta (guided vs unguided): {delta:.6f} (expected ~0)")

    # Training forward pass
    model.train()
    model.freeze_cjepa_base()
    out, masked_idx = model(history, vlm_guidance=vlm_guidance)
    assert out.shape == (B, T_hist + T_fut, N, D)
    print(f"  Training forward: {out.shape} ✓")

    # Backward
    loss = out.mean()
    loss.backward()
    print(f"  Backward pass: ✓")

    # Parameter counts
    counts = model.count_params()
    print(f"  Params — base: {counts['base']:,}, guidance: {counts['guidance']:,}, trainable: {counts['trainable']:,}")

    # Gate values
    gates = model.get_guidance_gate_values()
    print(f"  Layer gates: {[f'{g:.4f}' for g in gates]}")

    print("\n✓ All ThinkJEPA tests passed!")
