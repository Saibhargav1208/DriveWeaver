"""
Complete DriveWeaver Pipeline: C-JEPA (± VLM guidance) → Drive-JEPA

End-to-end pipeline for trajectory prediction from slot-based world models.

Supports two ablation modes:
    A) C-JEPA only:     history_slots → C-JEPA → future_slots → DriveJEPA → trajectory
    B) ThinkJEPA:       history_slots → C-JEPA+VLM guidance → future_slots → DriveJEPA → trajectory

Usage:
    pipeline = create_complete_pipeline(cfg)
    result = pipeline(history_slots, ego_state, vlm_guidance=vlm_guidance)
    best_trajectory = result['best_trajectory']  # [B, T_fut, 2]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

from models.cjepa_predictor import CJEPAPredictor, create_cjepa_from_config
from models.thinkjepa import ThinkJEPA, create_thinkjepa_from_config, load_cjepa_checkpoint_into_thinkjepa
from models.planner import Planner, create_planner


class DriveWeaverPipeline(nn.Module):
    """
    Complete DriveWeaver pipeline for trajectory prediction.

    Phases:
        1. VideoSAUR (offline): frames → slots (already done)
        2. World model: slots → future slots (C-JEPA ± VLM guidance)
        3. Planner: future slots + ego → trajectory (DriveJEPA)

    Ablations:
        - vlm_guidance=None → pure C-JEPA world model (Ablation A)
        - vlm_guidance={...} → ThinkJEPA: C-JEPA with VLM injection (Ablation B)
    """

    def __init__(
        self,
        world_model: ThinkJEPA,
        planner: Planner,
    ):
        super().__init__()
        self.world_model = world_model
        self.planner = planner

    def forward(
        self,
        history_slots: torch.Tensor,
        ego_state: torch.Tensor,
        route_goals: Optional[torch.Tensor] = None,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            history_slots: [B, T_hist, N, D]
            ego_state: [B, T_hist, 4]
            route_goals: [B, K, 2] (optional waypoints)
            vlm_guidance: dict with 'vlm_features' [B, S_vlm, vlm_dim] (None = no guidance)
            return_intermediates: whether to include world model outputs

        Returns:
            dict with 'best_trajectory' [B, T_fut, 2] and optionally intermediates
        """
        # World model: predict future slots
        future_slots = self.world_model.inference(history_slots, vlm_guidance=vlm_guidance)

        # Planner: future slots → trajectory
        planning_result = self.planner(
            refined_slots=future_slots,
            ego_state=ego_state,
            route_goals=route_goals,
            return_all_proposals=return_intermediates,
        )

        result = {
            'best_trajectory': planning_result['best_trajectory'],
            'proposal_scores': planning_result['proposal_scores'],
        }

        if return_intermediates:
            result['future_slots'] = future_slots
            result['trajectory_proposals'] = planning_result['trajectory_proposals']
            result['planning_context'] = planning_result['planning_context']
            result['mode_queries'] = planning_result['mode_queries']

            # Also get unguided prediction for comparison
            if vlm_guidance is not None:
                with torch.no_grad():
                    future_slots_no_vlm = self.world_model.inference(
                        history_slots, vlm_guidance=None
                    )
                result['future_slots_no_vlm'] = future_slots_no_vlm

        return result

    def inference(
        self,
        history_slots: torch.Tensor,
        ego_state: torch.Tensor,
        route_goals: Optional[torch.Tensor] = None,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Returns only best trajectory: [B, T_fut, 2]."""
        result = self.forward(
            history_slots, ego_state, route_goals, vlm_guidance,
            return_intermediates=False,
        )
        return result['best_trajectory']

    # --- Freezing utilities ---

    def freeze_world_model(self):
        for p in self.world_model.parameters():
            p.requires_grad = False
        print("[Pipeline] World model frozen")

    def unfreeze_world_model(self):
        for p in self.world_model.parameters():
            p.requires_grad = True
        print("[Pipeline] World model unfrozen")

    def freeze_planner(self):
        for p in self.planner.parameters():
            p.requires_grad = False
        print("[Pipeline] Planner frozen")

    def count_trainable_params(self) -> Dict[str, int]:
        return {
            'world_model': sum(p.numel() for p in self.world_model.parameters() if p.requires_grad),
            'planner': sum(p.numel() for p in self.planner.parameters() if p.requires_grad),
        }

    def print_parameter_summary(self):
        counts = self.count_trainable_params()
        total_trainable = sum(counts.values())
        total_all = sum(p.numel() for p in self.parameters())

        print("\n=== DriveWeaver Pipeline Parameters ===")
        print(f"World Model:  {counts['world_model']:>10,} trainable")
        print(f"Planner:      {counts['planner']:>10,} trainable")
        print(f"---")
        print(f"Total trainable: {total_trainable:>10,}")
        print(f"Total all:       {total_all:>10,}")


def create_complete_pipeline(
    # Model architecture
    slot_dim: int = 128,
    num_slots: int = 11,
    history_len: int = 4,
    future_len: int = 6,
    num_modes: int = 32,

    # C-JEPA config
    cjepa_depth: int = 6,
    cjepa_heads: int = 8,
    cjepa_mlp_dim: int = 2048,

    # VLM guidance config
    guidance_mode: Optional[str] = None,
    guidance_dim: Optional[int] = None,
    guidance_hidden: int = 512,

    # Planner config
    planner_decoder_layers: int = 3,
    planner_heads: int = 8,

    # Checkpoints
    cjepa_checkpoint: Optional[str] = None,
    planner_checkpoint: Optional[str] = None,

    # Device
    device: str = 'cuda',
) -> DriveWeaverPipeline:
    """
    Factory for complete pipeline.

    Args:
        guidance_mode: None for pure C-JEPA (Ablation A), 'film'/'crossattn' for ThinkJEPA (Ablation B)
        guidance_dim: VLM hidden dim (e.g. 3584 for Qwen3-VL-2B)
        cjepa_checkpoint: Path to trained C-JEPA or planning_cjepa checkpoint
        drivejepa_checkpoint: Path to trained DriveJEPA checkpoint
    """
    print("\n=== Building DriveWeaver Pipeline ===")

    # 1. World model (ThinkJEPA wrapping C-JEPA)
    print(f"\n[1/2] World Model (guidance={guidance_mode or 'None'})")
    cjepa = CJEPAPredictor(
        num_slots=num_slots,
        slot_dim=slot_dim,
        history_frames=history_len,
        pred_frames=future_len,
        num_masked_slots=2,
        depth=cjepa_depth,
        heads=cjepa_heads,
        dim_head=slot_dim // cjepa_heads,
        mlp_dim=cjepa_mlp_dim,
        dropout=0.0,
        guidance_mode=guidance_mode,
        guidance_dim=guidance_dim,
        guidance_hidden=guidance_hidden,
    )
    world_model = ThinkJEPA(cjepa=cjepa)

    if cjepa_checkpoint and Path(cjepa_checkpoint).exists():
        print(f"  Loading: {cjepa_checkpoint}")
        load_cjepa_checkpoint_into_thinkjepa(world_model, cjepa_checkpoint)

    wm_params = sum(p.numel() for p in world_model.parameters())
    print(f"  Parameters: {wm_params:,}")

    # 2. Planner
    print(f"\n[2/2] Planner ({num_modes} modes)")
    planner = create_planner(
        slot_dim=slot_dim,
        num_slots=num_slots,
        num_modes=num_modes,
        future_len=future_len,
        history_len=history_len,
        num_decoder_layers=planner_decoder_layers,
        num_heads=planner_heads,
    )

    if planner_checkpoint and Path(planner_checkpoint).exists():
        print(f"  Loading planner: {planner_checkpoint}")
        ckpt = torch.load(planner_checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        planner.load_state_dict(state)

    planner_params = sum(p.numel() for p in planner.parameters())
    print(f"  Parameters: {planner_params:,}")

    # Assemble
    pipeline = DriveWeaverPipeline(world_model=world_model, planner=planner)
    pipeline.to(device)
    pipeline.print_parameter_summary()

    return pipeline


if __name__ == "__main__":
    print("Testing DriveWeaver Pipeline...")

    B, T_hist, T_fut, N, D = 2, 4, 6, 11, 128

    # --- Ablation A: C-JEPA only (no guidance) ---
    print("\n" + "="*50)
    print("ABLATION A: C-JEPA only")
    print("="*50)
    pipeline_a = create_complete_pipeline(
        guidance_mode=None, guidance_dim=None,
        cjepa_depth=2, cjepa_heads=4, cjepa_mlp_dim=512,
        num_modes=8, planner_decoder_layers=2, planner_heads=4,
        device='cpu',
    )

    history = torch.randn(B, T_hist, N, D)
    ego = torch.randn(B, T_hist, 4)

    result_a = pipeline_a(history, ego, vlm_guidance=None, return_intermediates=True)
    print(f"\n  Best trajectory: {result_a['best_trajectory'].shape}")
    assert result_a['best_trajectory'].shape == (B, T_fut, 2)

    # --- Ablation B: ThinkJEPA (with VLM guidance) ---
    print("\n" + "="*50)
    print("ABLATION B: ThinkJEPA (FiLM guidance)")
    print("="*50)
    pipeline_b = create_complete_pipeline(
        guidance_mode='film', guidance_dim=512, guidance_hidden=256,
        cjepa_depth=2, cjepa_heads=4, cjepa_mlp_dim=512,
        num_modes=8, planner_decoder_layers=2, planner_heads=4,
        device='cpu',
    )

    vlm_guidance = {'vlm_features': torch.randn(B, 64, 512)}
    result_b = pipeline_b(history, ego, vlm_guidance=vlm_guidance, return_intermediates=True)
    print(f"\n  Best trajectory: {result_b['best_trajectory'].shape}")
    print(f"  Future slots (guided): {result_b['future_slots'].shape}")
    print(f"  Future slots (no VLM): {result_b['future_slots_no_vlm'].shape}")
    assert result_b['best_trajectory'].shape == (B, T_fut, 2)

    # Backward pass
    loss = result_b['best_trajectory'].mean()
    loss.backward()
    print(f"\n  Backward pass: ✓")

    print("\n✓ All pipeline tests passed!")
