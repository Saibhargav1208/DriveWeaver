"""
Diagnose Proposal Scorer Issue

Compares:
1. Best trajectory by ADE (oracle)
2. Best trajectory by scorer (what we use)

This will reveal if the scorer is selecting the wrong proposals.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.complete_pipeline import DriveWeaverPipeline
from models.cjepa_predictor import CJEPAPredictor
from models.thinkjepa import ThinkJEPA
from models.planner import Planner
from datasets.planner_dataset import create_planner_dataloaders


def compute_ade_per_mode(proposals, gt):
    """
    proposals: [B, M, T, 2]
    gt: [B, T, 2]
    Returns: [B, M] ADE per mode
    """
    B, M, T, _ = proposals.shape
    gt_exp = gt.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 2]
    errors = torch.norm(proposals - gt_exp, p=2, dim=-1)  # [B, M, T]
    ade = errors.mean(dim=-1)  # [B, M]
    return ade


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading pipeline...")

    # Load C-JEPA
    cjepa = CJEPAPredictor(
        num_slots=11, slot_dim=128, history_frames=4, pred_frames=6,
        depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout=0.0,
    )
    ckpt = torch.load('/work/checkpoints/cjepa/best_model.pth', map_location='cpu')
    cjepa.load_state_dict(ckpt['model_state_dict'])
    world_model = ThinkJEPA(cjepa=cjepa)
    world_model.to(device).eval()

    # Load Planner
    planner = Planner(slot_dim=128, num_slots=11, num_modes=32, future_len=6, history_len=4)
    ckpt = torch.load('/work/checkpoints/planner_a/best_model.pth', map_location='cpu')
    planner.load_state_dict(ckpt['model_state_dict'], strict=False)
    planner.to(device).eval()

    pipeline = DriveWeaverPipeline(world_model=world_model, planner=planner)
    pipeline.to(device).eval()

    # Load data
    print("Loading validation data...")
    _, val_loader = create_planner_dataloaders(
        slots_path='/work/data/slots/nuscenes_slots_full.pkl',
        batch_size=16, num_workers=4, history_length=4, future_length=6,
    )

    # Analyze
    print("\nAnalyzing scorer performance...\n")

    scorer_ades = []
    oracle_ades = []
    scorer_ranks = []  # Rank of scorer's choice among all modes (1=best, 32=worst)

    for i, batch in enumerate(tqdm(val_loader, total=100, desc="Analyzing")):
        if i >= 100:
            break

        history = batch['history_slots'].to(device)
        ego = batch['ego_history'].to(device)
        gt_traj = batch['ego_future_trajectory'].to(device)

        # Forward with all proposals
        result = pipeline(history, ego, vlm_guidance=None, return_intermediates=True)
        proposals = result['trajectory_proposals']  # [B, M, T, 2]
        scores = result['proposal_scores']  # [B, M]

        # Compute ADE for all modes
        ade_per_mode = compute_ade_per_mode(proposals, gt_traj)  # [B, M]

        # Best by ADE (oracle)
        oracle_best_idx = ade_per_mode.argmin(dim=1)  # [B]
        oracle_ade = ade_per_mode[range(len(oracle_best_idx)), oracle_best_idx]

        # Best by scorer (what we actually use)
        scorer_best_idx = scores.argmax(dim=1)  # [B]
        scorer_ade = ade_per_mode[range(len(scorer_best_idx)), scorer_best_idx]

        # Rank of scorer's choice
        # Sort modes by ADE, find where scorer's choice ranks
        ade_ranks = torch.argsort(ade_per_mode, dim=1)  # [B, M] indices sorted by ADE
        scorer_ranks_batch = []
        for b in range(len(scorer_best_idx)):
            rank = (ade_ranks[b] == scorer_best_idx[b]).nonzero(as_tuple=True)[0].item() + 1
            scorer_ranks_batch.append(rank)

        scorer_ades.extend(scorer_ade.cpu().numpy())
        oracle_ades.extend(oracle_ade.cpu().numpy())
        scorer_ranks.extend(scorer_ranks_batch)

    scorer_ades = np.array(scorer_ades)
    oracle_ades = np.array(oracle_ades)
    scorer_ranks = np.array(scorer_ranks)

    print(f"\n{'='*70}")
    print("SCORER DIAGNOSIS")
    print(f"{'='*70}\n")

    print(f"Oracle (best by ADE):")
    print(f"  Mean ADE: {oracle_ades.mean():.3f} m")
    print(f"  Std ADE:  {oracle_ades.std():.3f} m\n")

    print(f"Scorer (what we use):")
    print(f"  Mean ADE: {scorer_ades.mean():.3f} m")
    print(f"  Std ADE:  {scorer_ades.std():.3f} m\n")

    print(f"Scorer Performance:")
    print(f"  ADE gap:  {scorer_ades.mean() - oracle_ades.mean():.3f} m")
    print(f"  Relative: {100 * (scorer_ades.mean() / oracle_ades.mean() - 1):.1f}% worse\n")

    print(f"Scorer's Rank Distribution:")
    print(f"  Mean rank: {scorer_ranks.mean():.1f} / 32")
    print(f"  Median rank: {np.median(scorer_ranks):.0f} / 32")
    print(f"  Rank 1 (optimal): {100 * (scorer_ranks == 1).mean():.1f}%")
    print(f"  Rank 1-5 (top 5):  {100 * (scorer_ranks <= 5).mean():.1f}%")
    print(f"  Rank 1-10 (top 10): {100 * (scorer_ranks <= 10).mean():.1f}%")
    print(f"  Rank 16-32 (bottom half): {100 * (scorer_ranks > 16).mean():.1f}%")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}\n")

    if scorer_ades.mean() > oracle_ades.mean() * 1.5:
        print("❌ SCORER IS BROKEN!")
        print(f"   Scorer's choices are {100 * (scorer_ades.mean() / oracle_ades.mean() - 1):.0f}% worse than oracle")
        print(f"   Average rank: {scorer_ranks.mean():.1f}/32 (should be close to 1)")
        print("\n   RECOMMENDATION: Use oracle selection (minADE) for inference")
    elif scorer_ades.mean() > oracle_ades.mean() * 1.2:
        print("⚠️  SCORER IS SUBOPTIMAL")
        print(f"   Scorer could be 20%+ better")
        print("\n   RECOMMENDATION: Add supervision to scorer training")
    else:
        print("✅ SCORER IS WORKING")
        print(f"   Scorer is selecting near-optimal proposals")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
