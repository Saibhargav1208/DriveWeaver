"""
Dataset for Drive-JEPA Training (Phase 4)

Loads:
    - History slots [T_hist, N, D]
    - Future slots (for ThinkJEPA) [T_fut, N, D]
    - Ego state history [T_hist, 4]
    - Ego future trajectory (ground truth) [T_fut, 2]
    - Optional: Route waypoints [K, 2]

This dataset assumes Phase 3 (ThinkJEPA) is already trained, so we either:
    Option A: Load pre-refined slots from disk (fast, recommended)
    Option B: Run ThinkJEPA on-the-fly during training (slower, but always fresh)

For now, we'll use Option B (on-the-fly) to keep things simple.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class DriveJEPADataset(Dataset):
    """
    Dataset for Drive-JEPA multimodal trajectory planning.

    Loads VideoSAUR slots + ego poses, creates temporal windows.

    Each sample contains:
        - history_slots: [T_hist, N, D]
        - future_slots_gt: [T_fut, N, D]  (ground truth, for ThinkJEPA)
        - ego_history: [T_hist, 4]  (x, y, yaw, speed)
        - ego_future_trajectory: [T_fut, 2]  (x, y) - SUPERVISION SIGNAL
        - scene_token: str
        - start_idx: int

    The ego_future_trajectory is the ground truth for trajectory prediction.
    """

    def __init__(
        self,
        slots_path: str,
        split: str = 'train',
        history_length: int = 4,
        future_length: int = 6,
        stride: int = 1,
    ):
        """
        Args:
            slots_path: Path to pickled slots file (from Phase 1)
            split: 'train' or 'val'
            history_length: T_hist
            future_length: T_fut
            stride: Stride between windows (1 = dense sampling)
        """
        super().__init__()

        self.history_length = history_length
        self.future_length = future_length
        self.stride = stride
        self.window_size = history_length + future_length

        # Load slots
        print(f"Loading slots from {slots_path}...")
        with open(slots_path, 'rb') as f:
            data = pickle.load(f)

        if split not in data:
            raise ValueError(f"Split '{split}' not found in data. Available: {list(data.keys())}")

        self.split_data = data[split]
        self.scene_tokens = list(self.split_data.keys())

        # Build index: (scene_idx, start_idx) for each valid window
        self.index = []
        for scene_idx, scene_token in enumerate(self.scene_tokens):
            scene_data = self.split_data[scene_token]
            slots_shape = scene_data['slots'].shape  # [T, N, D]
            T = slots_shape[0]

            # Create windows with stride
            for start_idx in range(0, T - self.window_size + 1, self.stride):
                self.index.append((scene_idx, start_idx))

        print(f"  {split} split: {len(self.scene_tokens)} scenes, {len(self.index)} windows")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_idx, start_idx = self.index[idx]
        scene_token = self.scene_tokens[scene_idx]
        scene_data = self.split_data[scene_token]

        # Extract window
        end_idx = start_idx + self.window_size

        # Slots: [T, N, D]
        slots = scene_data['slots'][start_idx:end_idx]  # [T_hist+T_fut, N, D]
        history_slots = slots[:self.history_length]      # [T_hist, N, D]
        future_slots_gt = slots[self.history_length:]    # [T_fut, N, D]

        # Ego poses: [T, 4] (x, y, yaw, speed)
        ego_poses = scene_data['ego_poses'][start_idx:end_idx]  # [T_hist+T_fut, 4]
        ego_history = ego_poses[:self.history_length]            # [T_hist, 4]
        ego_future = ego_poses[self.history_length:]             # [T_fut, 4]

        # Transform to ego-relative frame at last history timestep
        # Reference: position and heading at t=T_hist-1
        ref_x, ref_y, ref_yaw = ego_history[-1, 0], ego_history[-1, 1], ego_history[-1, 2]
        cos_r, sin_r = np.cos(-ref_yaw), np.sin(-ref_yaw)

        # Future trajectory in ego-relative coords
        future_xy = ego_future[:, :2]  # [T_fut, 2]
        dx = future_xy[:, 0] - ref_x
        dy = future_xy[:, 1] - ref_y
        ego_future_trajectory = np.stack([
            dx * cos_r - dy * sin_r,
            dx * sin_r + dy * cos_r,
        ], axis=-1)  # [T_fut, 2] in ego frame

        # History also in ego-relative coords
        hist_xy = ego_history[:, :2]
        hdx = hist_xy[:, 0] - ref_x
        hdy = hist_xy[:, 1] - ref_y
        ego_history_local = ego_history.copy()
        ego_history_local[:, 0] = hdx * cos_r - hdy * sin_r
        ego_history_local[:, 1] = hdx * sin_r + hdy * cos_r
        ego_history_local[:, 2] = ego_history[:, 2] - ref_yaw  # relative yaw
        ego_history = ego_history_local

        # Convert to tensors
        sample = {
            'history_slots': torch.from_numpy(history_slots).float(),         # [T_hist, N, D]
            'future_slots_gt': torch.from_numpy(future_slots_gt).float(),     # [T_fut, N, D]
            'ego_history': torch.from_numpy(ego_history).float(),             # [T_hist, 4]
            'ego_future_trajectory': torch.from_numpy(ego_future_trajectory).float(),  # [T_fut, 2]
            'scene_token': scene_token,
            'start_idx': start_idx,
        }

        return sample


def collate_fn(batch):
    """
    Collate function for DriveJEPADataset.

    Stacks tensors, keeps scene_token and start_idx as lists.
    """
    history_slots = torch.stack([item['history_slots'] for item in batch])
    future_slots_gt = torch.stack([item['future_slots_gt'] for item in batch])
    ego_history = torch.stack([item['ego_history'] for item in batch])
    ego_future_trajectory = torch.stack([item['ego_future_trajectory'] for item in batch])

    scene_tokens = [item['scene_token'] for item in batch]
    start_indices = [item['start_idx'] for item in batch]

    return {
        'history_slots': history_slots,             # [B, T_hist, N, D]
        'future_slots_gt': future_slots_gt,         # [B, T_fut, N, D]
        'ego_history': ego_history,                 # [B, T_hist, 4]
        'ego_future_trajectory': ego_future_trajectory,  # [B, T_fut, 2]
        'scene_tokens': scene_tokens,
        'start_indices': start_indices,
    }


def create_planner_dataloaders(
    slots_path: str,
    batch_size: int = 16,
    history_length: int = 4,
    future_length: int = 6,
    stride: int = 1,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders for Drive-JEPA.

    Args:
        slots_path: Path to VideoSAUR slots pickle
        batch_size: Batch size
        history_length: T_hist
        future_length: T_fut
        stride: Window stride
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader
    """

    # Train dataset
    train_dataset = DriveJEPADataset(
        slots_path=slots_path,
        split='train',
        history_length=history_length,
        future_length=future_length,
        stride=stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Val dataset
    val_dataset = DriveJEPADataset(
        slots_path=slots_path,
        split='val',
        history_length=history_length,
        future_length=future_length,
        stride=stride,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing Drive-JEPA Dataset...")

    # Test with debug slots
    slots_path = "/work/data/slots/nuscenes_slots_debug.pkl"

    # Create dataset
    dataset = DriveJEPADataset(
        slots_path=slots_path,
        split='train',
        history_length=4,
        future_length=6,
        stride=1,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get sample
    sample = dataset[0]
    print("\nSample 0:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # Test dataloader
    print("\nTesting DataLoader...")
    train_loader, val_loader = create_drivejepa_dataloaders(
        slots_path=slots_path,
        batch_size=4,
        history_length=4,
        future_length=6,
        stride=1,
        num_workers=0,
    )

    # Get batch
    batch = next(iter(train_loader))
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {len(v)} items")

    # Verify shapes
    B = 4
    assert batch['history_slots'].shape == (B, 4, 11, 128)
    assert batch['future_slots_gt'].shape == (B, 6, 11, 128)
    assert batch['ego_history'].shape == (B, 4, 4)
    assert batch['ego_future_trajectory'].shape == (B, 6, 2)

    print("\n✓ All dataset tests passed!")
