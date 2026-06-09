"""
Visualize Planner B Predictions with scorer selection on nuScenes Images
(Ablation 2: C-JEPA + VLM -> Planner)

Projects predicted and ground truth trajectories onto CAM_FRONT images
using camera intrinsics and extrinsics.

Uses proposal_scores.argmax, so no ground truth is used for selecting the displayed proposal.

Usage:
    python scripts/visualize_planner_b_on_images.py \
        --world_model_ckpt /work/checkpoints/cjepa_vlm/best_model.pth \
        --planner_ckpt /work/checkpoints/planner_b/best_model.pth \
        --output_dir /work/outputs/vis_planner_b \
        --nuscenes_root /data/nuScenes \
        --num_samples 30 \
        --random_sampling
"""

import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from models.complete_pipeline import DriveWeaverPipeline
from models.cjepa_predictor import CJEPAPredictor
from models.thinkjepa import ThinkJEPA
from models.planner import Planner
from datasets.planner_dataset import (
    CachedFutureSlotsDataset,
    DriveJEPADataset,
    cached_collate_fn,
    collate_fn as planner_collate_fn,
)
from torch.utils.data import DataLoader

# Import nuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute trajectory metrics."""
    distances = np.linalg.norm(pred - gt, axis=-1)
    ade = distances.mean()
    fde = distances[-1]
    mr_2m = 1.0 if fde > 2.0 else 0.0
    return {'ADE': ade, 'FDE': fde, 'MR@2m': mr_2m}


def compute_ade_per_mode(proposals: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute ADE for each mode."""
    B, M, T, _ = proposals.shape
    gt_exp = gt.unsqueeze(1).expand(-1, M, -1, -1)
    errors = torch.norm(proposals - gt_exp, p=2, dim=-1)
    return errors.mean(dim=-1)


def get_camera_intrinsic(nusc: NuScenes, sample_token: str, camera_channel: str = 'CAM_FRONT') -> np.ndarray:
    """Get camera intrinsic matrix."""
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][camera_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    return np.array(cam_calib['camera_intrinsic'])


def get_camera_transform(nusc: NuScenes, sample_token: str, camera_channel: str = 'CAM_FRONT') -> Tuple[np.ndarray, Quaternion]:
    """Get camera to ego transformation."""
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][camera_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

    # Camera to ego transformation
    translation = np.array(cam_calib['translation'])
    rotation = Quaternion(cam_calib['rotation'])

    return translation, rotation


def project_ego_to_image(
    points_ego: np.ndarray,
    cam_translation: np.ndarray,
    cam_rotation: Quaternion,
    cam_intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project points in ego frame to camera image coordinates.

    Args:
        points_ego: [N, 3] points in ego vehicle frame (x, y, z)
        cam_translation: [3] camera position in ego frame
        cam_rotation: camera orientation quaternion
        cam_intrinsic: [3, 3] camera intrinsic matrix

    Returns:
        points_2d: [N, 2] projected image coordinates
        depths: [N] depth values (for filtering points behind camera)
    """
    # Transform from ego to camera frame
    points_cam = points_ego - cam_translation
    points_cam = np.dot(points_cam, cam_rotation.rotation_matrix)

    # Filter points behind camera (z > 0 in camera frame)
    depths = points_cam[:, 2]

    # Project to image using intrinsics
    points_2d = view_points(points_cam.T, cam_intrinsic, normalize=True)[:2, :].T

    return points_2d, depths


def visualize_trajectory_on_image(
    image_path: str,
    pred_traj_ego: np.ndarray,
    gt_traj_ego: np.ndarray,
    cam_translation: np.ndarray,
    cam_rotation: Quaternion,
    cam_intrinsic: np.ndarray,
    metrics: Dict[str, float],
    save_path: Path,
    title: str = "",
    selected_rank: int = 1,
):
    """
    Overlay predicted and ground truth trajectories on camera image.
    """
    # Load image
    img = np.array(Image.open(image_path))
    h, w = img.shape[:2]

    # Add z=0 (ground plane) to trajectories
    pred_traj_3d = np.concatenate([pred_traj_ego, np.zeros((len(pred_traj_ego), 1))], axis=-1)
    gt_traj_3d = np.concatenate([gt_traj_ego, np.zeros((len(gt_traj_ego), 1))], axis=-1)

    # Project to image
    pred_2d, pred_depths = project_ego_to_image(pred_traj_3d, cam_translation, cam_rotation, cam_intrinsic)
    gt_2d, gt_depths = project_ego_to_image(gt_traj_3d, cam_translation, cam_rotation, cam_intrinsic)

    # Filter points that are in front of camera and within image bounds
    pred_valid = (pred_depths > 0) & \
                 (pred_2d[:, 0] >= 0) & (pred_2d[:, 0] < w) & \
                 (pred_2d[:, 1] >= 0) & (pred_2d[:, 1] < h)
    gt_valid = (gt_depths > 0) & \
               (gt_2d[:, 0] >= 0) & (gt_2d[:, 0] < w) & \
               (gt_2d[:, 1] >= 0) & (gt_2d[:, 1] < h)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.axis('off')

    # Plot trajectories if visible
    if gt_valid.any():
        gt_visible = gt_2d[gt_valid]
        ax.plot(gt_visible[:, 0], gt_visible[:, 1], 'o-',
                color='lime', linewidth=4, markersize=10,
                label='Ground Truth', zorder=100, markeredgecolor='black', markeredgewidth=2)

        # Start and end markers
        if len(gt_visible) > 0:
            ax.scatter(gt_visible[0, 0], gt_visible[0, 1],
                      s=300, c='green', marker='o', edgecolors='white',
                      linewidths=3, label='Start', zorder=101)
            ax.scatter(gt_visible[-1, 0], gt_visible[-1, 1],
                      s=400, c='red', marker='X', edgecolors='white',
                      linewidths=3, label='Goal', zorder=102)

    if pred_valid.any():
        pred_visible = pred_2d[pred_valid]
        ax.plot(pred_visible[:, 0], pred_visible[:, 1], '^-',
                color='cyan', linewidth=4, markersize=10,
                label='Scorer Selected', zorder=90, alpha=1.0,
                markeredgecolor='blue', markeredgewidth=2)

    # Add metrics text box with scorer rank for diagnostics.
    metrics_text = (
        f"SCORER (ADE rank {selected_rank}/32)\n"
        f"ADE: {metrics['ADE']:.2f}m | "
        f"FDE: {metrics['FDE']:.2f}m | "
        f"MR@2m: {metrics['MR@2m']:.0f}"
    )
    ax.text(0.5, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=16, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white', weight='bold')

    if title:
        ax.text(0.5, 0.02, title, transform=ax.transAxes,
                fontsize=14, verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                weight='bold')

    # Legend
    if gt_valid.any() or pred_valid.any():
        ax.legend(loc='upper left', fontsize=12, framealpha=0.8)

    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_model_ckpt', type=str, required=True)
    parser.add_argument('--planner_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--nuscenes_root', type=str, required=True)
    parser.add_argument('--slots_path', type=str,
                        default='/work/data/slots/nuscenes_slots_full.pkl')
    parser.add_argument('--vlm_cache_dir', type=str,
                        default='/work/data/vlm_cache/nuscenes/',
                        help='Path to VLM feature cache')
    parser.add_argument('--future_slots_cache_dir', type=str, default=None,
                        help='Optional precomputed future-slot cache dir with val.pt; skips world-model inference')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--random_sampling', action='store_true',
                        help='Randomly sample from validation set instead of first N')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading nuScenes from {args.nuscenes_root}...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.nuscenes_root, verbose=False)

    use_cached_future_slots = args.future_slots_cache_dir is not None

    print(f"\nLoading checkpoints (SCORER SELECTION):")
    if use_cached_future_slots:
        print(f"  Future-slot cache: {args.future_slots_cache_dir}")
    else:
        print(f"  World model: {args.world_model_ckpt}")
    print(f"  Planner:     {args.planner_ckpt}")

    pipeline = None
    if not use_cached_future_slots:
        # Build pipeline
        print(f"\nBuilding DriveWeaver pipeline...")

        # 1. Load C-JEPA + VLM
        print("  [1/2] Loading C-JEPA + VLM world model...")
        cjepa = CJEPAPredictor(
            num_slots=11, slot_dim=128, history_frames=4, pred_frames=6,
            depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout=0.0,
            guidance_mode='film',
            guidance_dim=2048,
            guidance_hidden=512,
        )
        world_model = ThinkJEPA(cjepa=cjepa)
        world_ckpt = torch.load(args.world_model_ckpt, map_location='cpu', weights_only=False)
        world_model.load_state_dict(world_ckpt['model_state_dict'])
        world_model.to(device).eval()
        print(f"    Loaded from epoch {world_ckpt.get('epoch', '?')}")

    # 2. Load Planner
    print("  [2/2] Loading Planner..." if not use_cached_future_slots else "  [1/1] Loading Planner...")
    planner = Planner(slot_dim=128, num_slots=11, num_modes=32, future_len=6, history_len=4)
    planner_ckpt = torch.load(args.planner_ckpt, map_location='cpu', weights_only=False)
    planner.load_state_dict(planner_ckpt['model_state_dict'], strict=False)
    planner.to(device).eval()
    print(f"    Loaded from epoch {planner_ckpt.get('epoch', '?')}")

    if not use_cached_future_slots:
        # 3. Create pipeline
        pipeline = DriveWeaverPipeline(world_model=world_model, planner=planner)
        pipeline.to(device).eval()
        print("  Pipeline ready!")
    else:
        print("  Planner ready with precomputed future slots!")

    # Load data. Cached future slots are the fast path for the newly trained Planner B.
    if use_cached_future_slots:
        cache_path = Path(args.future_slots_cache_dir) / 'val.pt'
        print(f"\nLoading cached validation future slots from {cache_path}...")
        val_dataset = CachedFutureSlotsDataset(str(cache_path))
    else:
        print("\nLoading validation data...")
        print(f"VLM cache: {args.vlm_cache_dir}")
        val_dataset = DriveJEPADataset(
            slots_path=args.slots_path,
            split='val',
            history_length=4,
            future_length=6,
            vlm_cache_dir=args.vlm_cache_dir,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=cached_collate_fn if use_cached_future_slots else planner_collate_fn,
    )

    # Load slots data to get sample tokens
    import pickle
    with open(args.slots_path, 'rb') as f:
        slots_data = pickle.load(f)

    # Determine which samples to visualize
    total_val_samples = len(val_dataset)

    if args.random_sampling:
        # Random sampling
        np.random.seed(args.random_seed)
        sample_indices = np.random.choice(total_val_samples, size=args.num_samples, replace=False)
        sample_indices = sorted(sample_indices.tolist())
        print(f"\nRandomly sampling {args.num_samples} samples (seed={args.random_seed})...")
        print(f"Sample indices: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
    else:
        # Sequential sampling (first N)
        sample_indices = list(range(min(args.num_samples, total_val_samples)))
        print(f"\nVisualizing first {args.num_samples} samples...")

    def build_single_vlm_guidance(batch_data: Dict) -> Dict[str, torch.Tensor] | None:
        if 'vlm_features' not in batch_data:
            return None
        result = {}
        for key, tensor in batch_data['vlm_features'].items():
            result[key] = tensor.unsqueeze(0).to(device) if tensor is not None else None
        return result

    # Run inference
    print(f"Visualizing on images with scorer selection...")
    all_metrics = []

    for vis_idx, sample_idx in enumerate(tqdm(sample_indices, desc="Processing")):
        # Get specific sample from dataset
        batch_data = val_dataset[sample_idx]

        # Add batch dimension
        ego_history = batch_data['ego_history'].unsqueeze(0).to(device)
        gt_traj = batch_data['ego_future_trajectory'].unsqueeze(0).to(device)
        scene_token = batch_data['scene_token']
        start_idx = batch_data['start_idx']

        if use_cached_future_slots:
            future_slots = batch_data['future_slots'].unsqueeze(0).to(device).float()
            out = planner(
                refined_slots=future_slots,
                ego_state=ego_history,
                return_all_proposals=True,
            )
        else:
            history_slots = batch_data['history_slots'].unsqueeze(0).to(device)
            vlm_guidance = build_single_vlm_guidance(batch_data)
            # Forward pass with ALL proposals and cached VLM guidance.
            out = pipeline(history_slots, ego_state=ego_history, vlm_guidance=vlm_guidance, return_intermediates=True)

        proposals = out['trajectory_proposals']  # [1, 32, 6, 2]
        scores = out['proposal_scores']  # [1, 32]

        # Practical selection: pick the proposal with the highest learned score.
        ade_per_mode = compute_ade_per_mode(proposals, gt_traj)  # used only for rank diagnostics
        selected_idx = scores[0].argmax().item()
        rank_tensor = (torch.argsort(ade_per_mode, dim=1)[0] == selected_idx).nonzero(as_tuple=False)
        selected_rank = int(rank_tensor[0].item() + 1) if rank_tensor.numel() else -1
        pred_traj = proposals[0, selected_idx].cpu().numpy()  # [6, 2]

        gt_traj_np = gt_traj[0].cpu().numpy()

        # Compute metrics
        metrics = compute_metrics(pred_traj, gt_traj_np)
        metrics['selected_rank'] = float(selected_rank)
        metrics['selected_score'] = float(scores[0, selected_idx].cpu().item())
        all_metrics.append(metrics)

        # Get sample token for the FIRST frame of the trajectory (history end)
        scene_data = slots_data['val'][scene_token]
        sample_tokens = scene_data['sample_tokens']
        history_end_idx = start_idx + 4 - 1  # Last history frame

        if history_end_idx >= len(sample_tokens):
            print(f"  Skipping {scene_token} @ {start_idx}: index out of range")
            continue

        sample_token = sample_tokens[history_end_idx]

        # Get image path
        sample = nusc.get('sample', sample_token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        image_path = str(Path(args.nuscenes_root) / cam_data['filename'])

        # Get camera parameters
        cam_intrinsic = get_camera_intrinsic(nusc, sample_token)
        cam_translation, cam_rotation = get_camera_transform(nusc, sample_token)

        # Visualize
        title = f"Planner B (Scorer + VLM) | Sample {vis_idx+1} | {scene_token} @ frame {start_idx}"
        save_path = output_dir / f"planner_b_scorer_sample_{vis_idx:04d}_idx{sample_idx}.png"

        visualize_trajectory_on_image(
            image_path=image_path,
            pred_traj_ego=pred_traj,
            gt_traj_ego=gt_traj_np,
            cam_translation=cam_translation,
            cam_rotation=cam_rotation,
            cam_intrinsic=cam_intrinsic,
            metrics=metrics,
            save_path=save_path,
            title=title,
            selected_rank=selected_rank,
        )

    # Aggregate metrics
    print(f"\n{'='*60}")
    print("Validation Metrics (SCORER)")
    print(f"{'='*60}")

    metrics_summary = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    for k, v in metrics_summary.items():
        print(f"  {k:15s}: {v:.4f}")

    # Save summary
    summary_path = output_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Planner B (Ablation 2) - Validation Metrics (SCORER SELECTION)\n{'='*60}\n")
        for k, v in metrics_summary.items():
            f.write(f"{k:15s}: {v:.4f}\n")
        f.write(f"\nCheckpoints:\n")
        f.write(f"  World Model (C-JEPA+VLM): {args.world_model_ckpt}\n")
        f.write(f"  Planner B: {args.planner_ckpt}\n")
        f.write(f"Samples: {len(all_metrics)}\n")
        f.write(f"\nSelection: proposal_scores.argmax - no GT used for selection\n")
        f.write(f"Future-slot cache: {args.future_slots_cache_dir}\n")
        f.write(f"VLM Guidance: {'Precomputed C-JEPA+VLM future slots' if use_cached_future_slots else 'Enabled (Qwen3-VL layers 6,12,18,24)'}\n")

    print(f"\nOK Visualizations saved to: {output_dir}")
    print(f"OK Metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
