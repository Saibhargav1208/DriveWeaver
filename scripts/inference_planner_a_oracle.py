"""
Inference Script with Oracle Selection (minADE)

Uses the BEST trajectory by ADE instead of the scorer's selection.
This gives the TRUE performance of your trained model.

Usage:
    python scripts/inference_planner_oracle.py \
        --world_model_ckpt /work/checkpoints/cjepa/best_model.pth \
        --planner_ckpt /work/checkpoints/planner_a/best_model.pth \
        --output_dir /work/outputs/inference_oracle \
        --save_predictions \
        --compute_metrics
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.complete_pipeline import DriveWeaverPipeline
from models.cjepa_predictor import CJEPAPredictor
from models.thinkjepa import ThinkJEPA
from models.planner import Planner
from datasets.planner_dataset import create_planner_dataloaders


def compute_ade_per_mode(proposals: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute ADE for each mode.

    Args:
        proposals: [B, M, T, 2]
        gt: [B, T, 2]
    Returns:
        ade: [B, M]
    """
    B, M, T, _ = proposals.shape
    gt_exp = gt.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 2]
    errors = torch.norm(proposals - gt_exp, p=2, dim=-1)  # [B, M, T]
    ade = errors.mean(dim=-1)  # [B, M]
    return ade


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="DriveWeaver Inference with Oracle Selection")

    # Model checkpoints
    parser.add_argument('--world_model_ckpt', type=str, required=True)
    parser.add_argument('--planner_ckpt', type=str, required=True)

    # Data
    parser.add_argument('--slots_path', type=str,
                        default='/work/data/slots/nuscenes_slots_full.pkl')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])

    # Inference settings
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)

    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--compute_metrics', action='store_true')

    # Oracle vs Scorer comparison
    parser.add_argument('--compare_with_scorer', action='store_true',
                        help='Also compute metrics with scorer selection for comparison')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"DriveWeaver Inference Pipeline (ORACLE SELECTION)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"\nLoading checkpoints:")
    print(f"  World Model: {args.world_model_ckpt}")
    print(f"  Planner:     {args.planner_ckpt}")

    # Load C-JEPA
    print(f"\n[1/2] Loading C-JEPA World Model...")
    cjepa = CJEPAPredictor(
        num_slots=11, slot_dim=128, history_frames=4, pred_frames=6,
        depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout=0.0,
    )
    ckpt = torch.load(args.world_model_ckpt, map_location='cpu', weights_only=False)
    cjepa.load_state_dict(ckpt['model_state_dict'])
    world_model = ThinkJEPA(cjepa=cjepa)
    world_model.to(device).eval()
    print(f"  ✓ Loaded from epoch {ckpt.get('epoch', 'unknown')}")

    # Load Planner
    print(f"\n[2/2] Loading Planner...")
    planner = Planner(slot_dim=128, num_slots=11, num_modes=32, future_len=6, history_len=4)
    ckpt = torch.load(args.planner_ckpt, map_location='cpu', weights_only=False)
    planner.load_state_dict(ckpt['model_state_dict'], strict=False)
    planner.to(device).eval()
    print(f"  ✓ Loaded from epoch {ckpt.get('epoch', 'unknown')}")

    # Create pipeline
    pipeline = DriveWeaverPipeline(world_model=world_model, planner=planner)
    pipeline.to(device).eval()
    print("\n✓ Pipeline loaded successfully!")

    # Load data
    print(f"\nLoading {args.split} data from {args.slots_path}...")
    train_loader, val_loader = create_planner_dataloaders(
        slots_path=args.slots_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        history_length=4,
        future_length=6,
    )
    dataloader = val_loader if args.split == 'val' else train_loader

    # Run inference
    print(f"\n{'='*70}")
    print(f"Running Inference (ORACLE = minADE selection)")
    print(f"{'='*70}")

    oracle_predictions = []
    scorer_predictions = []
    ground_truths = []
    scene_tokens = []
    start_indices = []

    oracle_ranks = []  # Track which rank oracle selected
    scorer_ranks = []  # Track which rank scorer selected

    total_samples = args.max_samples if args.max_samples else len(dataloader)

    for i, batch in enumerate(tqdm(dataloader, total=total_samples, desc="Inference")):
        if args.max_samples and i >= args.max_samples:
            break

        history = batch['history_slots'].to(device)
        ego = batch['ego_history'].to(device)
        gt_traj = batch['ego_future_trajectory']

        # Forward with all proposals
        result = pipeline(history, ego, vlm_guidance=None, return_intermediates=True)
        proposals = result['trajectory_proposals']  # [B, M, T, 2]
        scores = result['proposal_scores']  # [B, M]

        # Compute ADE for all modes
        ade_per_mode = compute_ade_per_mode(proposals, gt_traj.to(device))  # [B, M]

        # ORACLE: Select best by ADE
        oracle_best_idx = ade_per_mode.argmin(dim=1)  # [B]
        B = len(oracle_best_idx)
        oracle_traj = proposals[range(B), oracle_best_idx].cpu().numpy()  # [B, T, 2]

        # SCORER: Select by scorer (for comparison)
        scorer_best_idx = scores.argmax(dim=1)  # [B]
        scorer_traj = proposals[range(B), scorer_best_idx].cpu().numpy()  # [B, T, 2]

        # Track ranks
        ade_ranks = torch.argsort(ade_per_mode, dim=1)  # [B, M] sorted by ADE
        for b in range(B):
            oracle_rank = (ade_ranks[b] == oracle_best_idx[b]).nonzero(as_tuple=True)[0].item() + 1
            scorer_rank = (ade_ranks[b] == scorer_best_idx[b]).nonzero(as_tuple=True)[0].item() + 1
            oracle_ranks.append(oracle_rank)
            scorer_ranks.append(scorer_rank)

        # Collect results
        oracle_predictions.append(oracle_traj)
        scorer_predictions.append(scorer_traj)
        ground_truths.append(gt_traj.cpu().numpy())
        scene_tokens.extend(batch.get('scene_tokens', ['unknown'] * B))
        start_indices.extend(batch.get('start_indices', [0] * B))

    # Concatenate
    oracle_predictions = np.concatenate(oracle_predictions, axis=0)  # [N, T, 2]
    scorer_predictions = np.concatenate(scorer_predictions, axis=0)  # [N, T, 2]
    ground_truths = np.concatenate(ground_truths, axis=0)  # [N, T, 2]

    print(f"\n✓ Inference complete!")
    print(f"  Total samples: {len(oracle_predictions)}")

    # Compute metrics
    if args.compute_metrics:
        print(f"\n{'='*70}")
        print("Computing Metrics")
        print(f"{'='*70}")

        # Oracle metrics
        oracle_errors = np.linalg.norm(oracle_predictions - ground_truths, axis=-1)  # [N, T]
        oracle_ade = oracle_errors.mean(axis=1).mean()
        oracle_ade_std = oracle_errors.mean(axis=1).std()
        oracle_fde = oracle_errors[:, -1].mean()
        oracle_fde_std = oracle_errors[:, -1].std()
        oracle_mr2m = (oracle_errors[:, -1] > 2.0).mean()

        print(f"\n✅ ORACLE Selection (minADE):")
        print(f"  ADE (mean): {oracle_ade:.4f} m  (± {oracle_ade_std:.4f})")
        print(f"  FDE (mean): {oracle_fde:.4f} m  (± {oracle_fde_std:.4f})")
        print(f"  Miss Rate @ 2m: {100*oracle_mr2m:.1f}%")
        print(f"  Average rank selected: {np.mean(oracle_ranks):.1f} / 32")

        # Scorer metrics (for comparison)
        if args.compare_with_scorer:
            scorer_errors = np.linalg.norm(scorer_predictions - ground_truths, axis=-1)
            scorer_ade = scorer_errors.mean(axis=1).mean()
            scorer_fde = scorer_errors[:, -1].mean()
            scorer_mr2m = (scorer_errors[:, -1] > 2.0).mean()

            print(f"\n❌ SCORER Selection (what was used before):")
            print(f"  ADE (mean): {scorer_ade:.4f} m")
            print(f"  FDE (mean): {scorer_fde:.4f} m")
            print(f"  Miss Rate @ 2m: {100*scorer_mr2m:.1f}%")
            print(f"  Average rank selected: {np.mean(scorer_ranks):.1f} / 32")

            print(f"\n📊 Improvement:")
            print(f"  ADE: {scorer_ade:.4f} → {oracle_ade:.4f} m  ({100*(1-oracle_ade/scorer_ade):.1f}% better)")
            print(f"  FDE: {scorer_fde:.4f} → {oracle_fde:.4f} m  ({100*(1-oracle_fde/scorer_fde):.1f}% better)")

        # Save metrics
        metrics = {
            'oracle': {
                'ADE_mean': float(oracle_ade),
                'ADE_std': float(oracle_ade_std),
                'FDE_mean': float(oracle_fde),
                'FDE_std': float(oracle_fde_std),
                'MR@2m': float(oracle_mr2m),
                'avg_rank': float(np.mean(oracle_ranks)),
            }
        }

        if args.compare_with_scorer:
            metrics['scorer'] = {
                'ADE_mean': float(scorer_ade),
                'FDE_mean': float(scorer_fde),
                'MR@2m': float(scorer_mr2m),
                'avg_rank': float(np.mean(scorer_ranks)),
            }

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Human-readable summary
        with open(output_dir / 'metrics_summary.txt', 'w') as f:
            f.write("DriveWeaver Inference Metrics (ORACLE SELECTION)\n")
            f.write("=" * 70 + "\n\n")
            f.write("✅ Oracle Selection (minADE):\n")
            f.write(f"  ADE (mean):     {oracle_ade:.4f} m  (± {oracle_ade_std:.4f})\n")
            f.write(f"  FDE (mean):     {oracle_fde:.4f} m  (± {oracle_fde_std:.4f})\n")
            f.write(f"  Miss Rate @ 2m: {100*oracle_mr2m:.1f}%\n\n")

            if args.compare_with_scorer:
                f.write("❌ Scorer Selection (original):\n")
                f.write(f"  ADE (mean):     {scorer_ade:.4f} m\n")
                f.write(f"  FDE (mean):     {scorer_fde:.4f} m\n")
                f.write(f"  Miss Rate @ 2m: {100*scorer_mr2m:.1f}%\n\n")
                f.write("📊 Improvement:\n")
                f.write(f"  ADE: {100*(1-oracle_ade/scorer_ade):.1f}% better\n")
                f.write(f"  FDE: {100*(1-oracle_fde/scorer_fde):.1f}% better\n")

        print(f"\n✓ Metrics saved to {output_dir}")

    # Save predictions
    if args.save_predictions:
        np.save(output_dir / 'predictions_oracle.npy', oracle_predictions)
        np.save(output_dir / 'ground_truths.npy', ground_truths)

        metadata = {
            'scene_tokens': scene_tokens,
            'start_indices': start_indices,
            'num_samples': len(oracle_predictions),
            'world_model_ckpt': args.world_model_ckpt,
            'planner_ckpt': args.planner_ckpt,
            'split': args.split,
            'variant': 'planner_a',
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Predictions saved to {output_dir}")

    print(f"\n{'='*70}")
    print(f"✓ Inference Complete!")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
