"""
Analyze planner predictions to understand why some are not visible in camera view.

Checks:
- Coordinate frame correctness
- Trajectory distribution (forward/backward, left/right)
- Camera visibility statistics
"""

import sys
import argparse
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_model_ckpt', type=str, required=True)
    parser.add_argument('--planner_ckpt', type=str, required=True)
    parser.add_argument('--slots_path', type=str, default='/work/data/slots/nuscenes_slots_full.pkl')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pipeline
    print("Loading pipeline...")
    cjepa = CJEPAPredictor(
        num_slots=11, slot_dim=128, history_frames=4, pred_frames=6,
        depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout=0.1,
    )
    world_ckpt = torch.load(args.world_model_ckpt, map_location='cpu')
    cjepa.load_state_dict(world_ckpt['model_state_dict'])
    world_model = ThinkJEPA(cjepa=cjepa)

    planner = Planner(slot_dim=128, num_slots=11, num_modes=32, future_len=6, history_len=4)
    planner_ckpt = torch.load(args.planner_ckpt, map_location='cpu')
    planner.load_state_dict(planner_ckpt['model_state_dict'], strict=False)

    pipeline = DriveWeaverPipeline(world_model=world_model, planner=planner)
    pipeline.to(device).eval()

    # Load data
    print("Loading validation data...")
    _, val_loader = create_planner_dataloaders(
        slots_path=args.slots_path, batch_size=1, num_workers=0,
        history_length=4, future_length=6,
    )

    # Analyze predictions
    print(f"\nAnalyzing {args.num_samples} samples...")

    pred_forward = []  # X > 0
    pred_backward = []  # X < 0
    pred_lateral = []  # |Y|
    pred_distance = []  # final distance

    gt_forward = []
    gt_lateral = []
    gt_distance = []

    errors = []

    for i, batch in enumerate(tqdm(val_loader, total=args.num_samples)):
        if i >= args.num_samples:
            break

        history_slots = batch['history_slots'].to(device)
        ego_history = batch['ego_history'].to(device)
        gt_traj = batch['ego_future_trajectory'].cpu().numpy()[0]  # [T, 2]

        # Forward pass
        out = pipeline(history_slots, ego_state=ego_history)
        pred_traj = out['best_trajectory'].cpu().numpy()[0]  # [T, 2]

        # Analyze predicted trajectory
        pred_x = pred_traj[:, 0]  # longitudinal (forward/back)
        pred_y = pred_traj[:, 1]  # lateral (left/right)
        pred_final_dist = np.linalg.norm(pred_traj[-1])

        pred_forward.append(pred_x.mean())
        pred_backward.append((pred_x < 0).sum())  # count backward points
        pred_lateral.append(np.abs(pred_y).mean())
        pred_distance.append(pred_final_dist)

        # Analyze ground truth
        gt_x = gt_traj[:, 0]
        gt_y = gt_traj[:, 1]
        gt_final_dist = np.linalg.norm(gt_traj[-1])

        gt_forward.append(gt_x.mean())
        gt_lateral.append(np.abs(gt_y).mean())
        gt_distance.append(gt_final_dist)

        # Error
        error = np.linalg.norm(pred_traj - gt_traj, axis=-1).mean()
        errors.append(error)

    # Convert to arrays
    pred_forward = np.array(pred_forward)
    pred_backward = np.array(pred_backward)
    pred_lateral = np.array(pred_lateral)
    pred_distance = np.array(pred_distance)

    gt_forward = np.array(gt_forward)
    gt_lateral = np.array(gt_lateral)
    gt_distance = np.array(gt_distance)

    errors = np.array(errors)

    # Print analysis
    print(f"\n{'='*70}")
    print("PREDICTION ANALYSIS")
    print(f"{'='*70}")

    print(f"\n--- Predicted Trajectories ---")
    print(f"  Avg forward distance (X):  {pred_forward.mean():.2f}m  (std: {pred_forward.std():.2f}m)")
    print(f"  Samples with backward pts:  {(pred_backward > 0).sum()}/{len(pred_backward)} ({100*(pred_backward > 0).mean():.1f}%)")
    print(f"  Avg lateral deviation (|Y|): {pred_lateral.mean():.2f}m  (std: {pred_lateral.std():.2f}m)")
    print(f"  Avg final distance:         {pred_distance.mean():.2f}m  (std: {pred_distance.std():.2f}m)")

    print(f"\n--- Ground Truth Trajectories ---")
    print(f"  Avg forward distance (X):   {gt_forward.mean():.2f}m  (std: {gt_forward.std():.2f}m)")
    print(f"  Avg lateral deviation (|Y|): {gt_lateral.mean():.2f}m  (std: {gt_lateral.std():.2f}m)")
    print(f"  Avg final distance:         {gt_distance.mean():.2f}m  (std: {gt_distance.std():.2f}m)")

    print(f"\n--- Errors ---")
    print(f"  Mean ADE:  {errors.mean():.2f}m")
    print(f"  Std ADE:   {errors.std():.2f}m")
    print(f"  Min ADE:   {errors.min():.2f}m")
    print(f"  Max ADE:   {errors.max():.2f}m")

    # Camera visibility analysis
    print(f"\n--- Camera Visibility Heuristic ---")
    # Camera typically sees forward (X > 1m) and within ~30° horizontal FOV (~tan(30°) ≈ 0.58)
    # For final waypoint visibility
    visible_forward = pred_distance * np.cos(np.arctan2(pred_traj[:, -1, 1], pred_traj[:, -1, 0]))
    visible_count = np.sum((pred_forward > 1.0) & (np.abs(pred_lateral) < pred_forward * 0.58))

    print(f"  Predicted final points likely visible: {visible_count}/{args.num_samples} ({100*visible_count/args.num_samples:.1f}%)")
    print(f"  (Heuristic: X > 1m AND |Y| < 0.58*X)")

    # Identify problematic predictions
    backward_samples = np.where(pred_forward < 0)[0]
    extreme_lateral = np.where(pred_lateral > 10)[0]
    very_close = np.where(pred_distance < 2)[0]

    if len(backward_samples) > 0:
        print(f"\n  ⚠️  {len(backward_samples)} samples predict BACKWARD trajectory (negative X)")
    if len(extreme_lateral) > 0:
        print(f"  ⚠️  {len(extreme_lateral)} samples have extreme lateral deviation (>10m)")
    if len(very_close) > 0:
        print(f"  ⚠️  {len(very_close)} samples predict very close trajectory (<2m)")

    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")

    if pred_forward.mean() < gt_forward.mean() * 0.5:
        print("  ❌ Predictions are much SHORTER than ground truth")
        print("     → Planner may be too conservative or not learning proper scale")

    if np.abs(pred_lateral.mean() - gt_lateral.mean()) > 2:
        print("  ❌ Lateral deviation mismatch")
        print("     → Coordinate frame issue or poor lateral prediction")

    if (pred_backward > 0).mean() > 0.1:
        print("  ❌ >10% samples predict backward motion")
        print("     → Coordinate transformation bug or planner instability")

    if errors.mean() > 3:
        print("  ❌ High average error (>3m)")
        print("     → Planner needs more training or better world model")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
