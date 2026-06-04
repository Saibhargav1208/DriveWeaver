"""
Sliding Window Dataset for C-JEPA Training

Loads VideoSAUR slots from Phase 1 and creates temporal windows:
- History: past T_hist frames
- Future: next T_future frames

Handles scene boundaries and generates valid temporal windows.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TemporalWindow:
    """Single temporal window sample."""
    history_slots: np.ndarray      # [T_hist, N, D]
    future_slots: np.ndarray       # [T_future, N, D]
    history_timestamps: np.ndarray # [T_hist]
    future_timestamps: np.ndarray  # [T_future]
    ego_poses: np.ndarray          # [T_hist + T_future, 4]
    scene_name: str
    start_idx: int


class SlotDataset(Dataset):
    """
    Dataset for loading temporally windowed slot embeddings.

    Creates sliding windows from Phase 1 VideoSAUR slots:
        Scene: [T_total frames]
        Window: [start:start+T_hist] -> [start+T_hist:start+T_hist+T_future]

    Optionally loads cached VLM features (from offline Qwen3-VL extraction)
    for ThinkJEPA guidance injection.
    """

    def __init__(
        self,
        slots_path: str,
        split: str = 'train',
        history_length: int = 4,
        future_length: int = 6,
        stride: int = 1,
        min_sequence_length: Optional[int] = None,
        vlm_cache_dir: Optional[str] = None,
        vlm_random_dim: Optional[int] = None,
        vlm_random_tokens: int = 480,
    ):
        """
        Args:
            slots_path: Path to Phase 1 slot pickle file
            split: 'train' or 'val'
            history_length: Number of history frames (T_hist)
            future_length: Number of future frames (T_future)
            stride: Sliding window stride
            min_sequence_length: Minimum scene length to include
            vlm_cache_dir: Path to directory with per-scene .npz VLM cache files
            vlm_random_dim: If set, generate random VLM features with this dim (debug)
            vlm_random_tokens: Number of random tokens when using vlm_random_dim
        """
        self.slots_path = Path(slots_path)
        self.split = split
        self.history_length = history_length
        self.future_length = future_length
        self.stride = stride
        self.window_length = history_length + future_length
        self.vlm_random_dim = vlm_random_dim
        self.vlm_random_tokens = vlm_random_tokens

        if min_sequence_length is None:
            min_sequence_length = self.window_length
        self.min_sequence_length = min_sequence_length

        # Load slot data
        print(f"Loading slots from {self.slots_path}...")
        with open(self.slots_path, 'rb') as f:
            self.data = pickle.load(f)

        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in data. "
                             f"Available: {list(self.data.keys())}")

        self.split_data = self.data[split]
        print(f"Loaded {len(self.split_data)} scenes for split '{split}'")

        # Build VLM cache index
        self.vlm_cache = {}
        if vlm_cache_dir:
            self._load_vlm_cache_index(vlm_cache_dir)

        # Generate temporal windows
        self.windows = self._generate_windows()
        print(f"Generated {len(self.windows)} temporal windows "
              f"(hist={history_length}, fut={future_length}, stride={stride})")

    def _load_vlm_cache_index(self, vlm_cache_dir: str):
        """Build index mapping scene_name -> npz path."""
        cache_dir = Path(vlm_cache_dir)
        if not cache_dir.exists():
            print(f"  VLM cache dir not found: {cache_dir}, will use random features")
            return

        count = 0
        for npz_path in cache_dir.glob("*.npz"):
            scene_name = npz_path.stem  # e.g. "scene-0001"
            self.vlm_cache[scene_name] = npz_path
            count += 1

        print(f"  VLM cache: indexed {count} scenes from {cache_dir}")

    def _generate_windows(self) -> List[Tuple[str, int]]:
        """
        Generate all valid temporal windows across scenes.

        Returns:
            List of (scene_name, start_idx) tuples
        """
        windows = []

        for scene_name, scene_data in self.split_data.items():
            slots = scene_data['slots']  # [T, N, D]
            T = slots.shape[0]

            # Check minimum length
            if T < self.min_sequence_length:
                print(f"  Skipping {scene_name}: only {T} frames "
                      f"(min={self.min_sequence_length})")
                continue

            # Generate sliding windows
            for start_idx in range(0, T - self.window_length + 1, self.stride):
                windows.append((scene_name, start_idx))

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a temporal window sample.

        Returns:
            dict with keys:
                - history_slots: [T_hist, N, D]
                - future_slots: [T_future, N, D]
                - history_timestamps: [T_hist]
                - future_timestamps: [T_future]
                - ego_poses: [T_hist + T_future, 4]
                - vlm_features: [S_vlm, vlm_dim] (if VLM cache available)
                - scene_name: str
                - start_idx: int
        """
        scene_name, start_idx = self.windows[idx]
        scene_data = self.split_data[scene_name]

        # Extract temporal window
        end_idx = start_idx + self.window_length

        slots = scene_data['slots']  # [T, N, D]
        timestamps = scene_data['timestamps']  # [T]
        ego_poses = scene_data['ego_poses']  # [T, 4]

        # Split into history and future
        hist_end = start_idx + self.history_length

        history_slots = slots[start_idx:hist_end]      # [T_hist, N, D]
        future_slots = slots[hist_end:end_idx]         # [T_future, N, D]
        history_timestamps = timestamps[start_idx:hist_end]
        future_timestamps = timestamps[hist_end:end_idx]
        window_ego_poses = ego_poses[start_idx:end_idx]

        result = {
            'history_slots': torch.from_numpy(history_slots).float(),
            'future_slots': torch.from_numpy(future_slots).float(),
            'history_timestamps': torch.from_numpy(history_timestamps).float(),
            'future_timestamps': torch.from_numpy(future_timestamps).float(),
            'ego_poses': torch.from_numpy(window_ego_poses).float(),
            'scene_name': scene_name,
            'start_idx': start_idx,
        }

        # Load VLM features if available
        vlm_features = self._get_vlm_features(scene_name)
        if vlm_features is not None:
            result['vlm_features'] = vlm_features

        return result

    def _get_vlm_features(self, scene_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load cached dual-path VLM features for a scene.

        Returns:
            dict with 'old' and 'new' keys, or None
            - 'old': [num_layers, num_tokens_old, vlm_dim]
            - 'new': [num_layers, num_tokens_new, vlm_dim]
        """
        if scene_name in self.vlm_cache:
            npz = np.load(self.vlm_cache[scene_name])

            # Check if this is new dual-path cache
            if 'vlm_old' in npz and 'vlm_new' in npz:
                vlm_old = torch.from_numpy(npz['vlm_old'].astype(np.float32).copy())
                vlm_new = torch.from_numpy(npz['vlm_new'].astype(np.float32).copy())
                return {'old': vlm_old, 'new': vlm_new}

            # Legacy single-path cache (backward compatibility)
            elif 'vlm_features' in npz:
                features = torch.from_numpy(npz['vlm_features'].astype(np.float32).copy())
                # Treat as old-style: add dummy layer dimension
                return {'old': features.unsqueeze(0), 'new': None}

        # Random features for debugging
        if self.vlm_random_dim is not None:
            num_layers = 4
            vlm_old = torch.randn(num_layers, self.vlm_random_tokens, self.vlm_random_dim)
            vlm_new = torch.randn(num_layers, 16, self.vlm_random_dim)
            return {'old': vlm_old, 'new': vlm_new}

        return None

    def get_scene_names(self) -> List[str]:
        """Get all scene names in this split."""
        return list(self.split_data.keys())

    def get_slot_dim(self) -> int:
        """Get slot dimension from data."""
        first_scene = next(iter(self.split_data.values()))
        return first_scene['slots'].shape[-1]

    def get_num_slots(self) -> int:
        """Get number of slots from data."""
        first_scene = next(iter(self.split_data.values()))
        return first_scene['slots'].shape[1]


def collate_with_vlm(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length VLM features.

    Pads VLM features to max length in batch and creates attention mask.
    """
    # Stack fixed-size tensors normally
    result = {
        'history_slots': torch.stack([item['history_slots'] for item in batch]),
        'future_slots': torch.stack([item['future_slots'] for item in batch]),
        'history_timestamps': torch.stack([item['history_timestamps'] for item in batch]),
        'future_timestamps': torch.stack([item['future_timestamps'] for item in batch]),
        'ego_poses': torch.stack([item['ego_poses'] for item in batch]),
        'scene_name': [item['scene_name'] for item in batch],
        'start_idx': [item['start_idx'] for item in batch],
    }

    # Handle dual-path VLM features if present
    if 'vlm_features' in batch[0] and batch[0]['vlm_features'] is not None:
        vlm_features_list = [item['vlm_features'] for item in batch]

        # Extract old and new separately
        old_list = [f['old'] for f in vlm_features_list]  # Each: [num_layers, S_old, D]
        new_list = [f['new'] for f in vlm_features_list if f['new'] is not None]

        # Stack old features (no padding needed - same shape per scene)
        vlm_old = torch.stack(old_list)  # [B, num_layers, S_old, D]

        # Stack new features if available
        if new_list:
            vlm_new = torch.stack(new_list)  # [B, num_layers, S_new, D]
        else:
            # Fallback for legacy cache without vlm_new
            vlm_new = None

        result['vlm_features'] = {
            'old': vlm_old,  # [B, num_layers, S_old, D]
            'new': vlm_new,  # [B, num_layers, S_new, D] or None
        }

    return result


def create_dataloaders(
    slots_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    history_length: int = 4,
    future_length: int = 6,
    stride: int = 1,
    pin_memory: bool = True,
    vlm_cache_dir: Optional[str] = None,
    vlm_random_dim: Optional[int] = None,
    vlm_random_tokens: int = 480,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        slots_path: Path to Phase 1 slot pickle
        batch_size: Batch size
        num_workers: Number of dataloader workers
        history_length: History frames
        future_length: Future frames
        stride: Sliding window stride
        pin_memory: Pin memory for faster GPU transfer
        vlm_cache_dir: Path to VLM cache directory (for ThinkJEPA guidance)
        vlm_random_dim: If set, generate random VLM features (debug mode)
        vlm_random_tokens: Number of tokens for random VLM features
        persistent_workers: Keep worker processes alive between epochs
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SlotDataset(
        slots_path=slots_path,
        split='train',
        history_length=history_length,
        future_length=future_length,
        stride=stride,
        vlm_cache_dir=vlm_cache_dir,
        vlm_random_dim=vlm_random_dim,
        vlm_random_tokens=vlm_random_tokens,
    )

    val_dataset = SlotDataset(
        slots_path=slots_path,
        split='val',
        history_length=history_length,
        future_length=future_length,
        stride=stride,
        vlm_cache_dir=vlm_cache_dir,
        vlm_random_dim=vlm_random_dim,
        vlm_random_tokens=vlm_random_tokens,
    )

    # Use custom collate function if VLM features are enabled
    collate_fn = collate_with_vlm if vlm_cache_dir or vlm_random_dim else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For stable batch norm
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    print(f"\n=== Dataloader Summary ===")
    print(f"Train: {len(train_dataset)} windows, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} windows, {len(val_loader)} batches")
    print(f"Slot dim: {train_dataset.get_slot_dim()}, "
          f"Num slots: {train_dataset.get_num_slots()}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing SlotDataset...")

    slots_path = "/work/data/slots/nuscenes_slots_debug.pkl"

    # Create dataset
    dataset = SlotDataset(
        slots_path=slots_path,
        split='train',
        history_length=4,
        future_length=6,
        stride=1
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Slot dim: {dataset.get_slot_dim()}")
    print(f"Num slots: {dataset.get_num_slots()}")
    print(f"Scene names: {dataset.get_scene_names()}")

    # Test __getitem__
    sample = dataset[0]
    print(f"\n=== Sample 0 ===")
    print(f"History slots: {sample['history_slots'].shape}")
    print(f"Future slots: {sample['future_slots'].shape}")
    print(f"History timestamps: {sample['history_timestamps'].shape}")
    print(f"Ego poses: {sample['ego_poses'].shape}")
    print(f"Scene: {sample['scene_name']}")
    print(f"Start idx: {sample['start_idx']}")

    # Test dataloader
    print(f"\n=== Testing DataLoader ===")
    train_loader, val_loader = create_dataloaders(
        slots_path=slots_path,
        batch_size=4,
        num_workers=0,  # Single process for testing
        history_length=4,
        future_length=6
    )

    # Get first batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  history_slots: {batch['history_slots'].shape}")
    print(f"  future_slots: {batch['future_slots'].shape}")
    print(f"  history_timestamps: {batch['history_timestamps'].shape}")
    print(f"  ego_poses: {batch['ego_poses'].shape}")

    # Verify data integrity
    assert not torch.isnan(batch['history_slots']).any(), "NaN in history!"
    assert not torch.isnan(batch['future_slots']).any(), "NaN in future!"
    print(f"\n✓ No NaN values found")

    print("\n✓ All dataset tests passed!")
