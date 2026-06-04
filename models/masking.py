"""
C-JEPA Style Masking Strategies

Implements three masking modes:
1. Object-level masking: Mask entire object trajectories
2. Temporal masking: Mask specific timesteps
3. Future masking: Mask random future slots

Goal: Force the model to infer missing object dynamics from context.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Literal, Optional
from dataclasses import dataclass


@dataclass
class MaskConfig:
    """Configuration for masking strategy."""
    strategy: Literal['object', 'temporal', 'future', 'mixed'] = 'mixed'

    # Object masking
    object_prob: float = 0.5
    object_mask_ratio: float = 0.3
    min_masked_objects: int = 2
    max_masked_objects: int = 4

    # Temporal masking
    temporal_prob: float = 0.3
    temporal_mask_ratio: float = 0.25
    min_masked_frames: int = 1
    max_masked_frames: int = 2

    # Future masking
    future_prob: float = 0.2
    future_mask_ratio: float = 0.3


class ObjectMasker(nn.Module):
    """
    Object-level masking: Mask entire object trajectories in history.

    Example:
        If slot index 2 and 5 are masked, all timesteps for those slots
        are replaced with a learnable [MASK] token.
    """

    def __init__(self, slot_dim: int):
        super().__init__()
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(slot_dim) * 0.02)

    def forward(
        self,
        slots: torch.Tensor,
        mask_ratio: float,
        min_masked: int = 2,
        max_masked: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: [B, T, N, D] slot embeddings
            mask_ratio: Ratio of objects to mask
            min_masked: Minimum number of objects to mask
            max_masked: Maximum number of objects to mask

        Returns:
            masked_slots: [B, T, N, D] with masked objects
            mask: [B, N] binary mask (1 = masked)
        """
        B, T, N, D = slots.shape

        # Determine number of objects to mask per sample
        num_masked = int(N * mask_ratio)
        num_masked = max(min_masked, min(num_masked, max_masked))
        num_masked = min(num_masked, N - 1)  # Keep at least one visible

        # Create mask: [B, N]
        mask = torch.zeros(B, N, device=slots.device, dtype=torch.bool)

        for b in range(B):
            # Randomly select objects to mask
            masked_indices = torch.randperm(N, device=slots.device)[:num_masked]
            mask[b, masked_indices] = True

        # Apply masking: replace masked slots with mask token
        masked_slots = slots.clone()
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        masked_slots = torch.where(
            mask_expanded,
            self.mask_token.view(1, 1, 1, D),  # Broadcast mask token
            masked_slots
        )

        return masked_slots, mask


class TemporalMasker(nn.Module):
    """
    Temporal masking: Mask specific timesteps across all objects.

    Example:
        If timestep 2 is masked, all slots at t=2 are masked.
        Forces temporal interpolation.
    """

    def __init__(self, slot_dim: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(slot_dim) * 0.02)

    def forward(
        self,
        slots: torch.Tensor,
        mask_ratio: float,
        min_masked: int = 1,
        max_masked: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slots: [B, T, N, D] slot embeddings
            mask_ratio: Ratio of timesteps to mask
            min_masked: Minimum timesteps to mask
            max_masked: Maximum timesteps to mask

        Returns:
            masked_slots: [B, T, N, D] with masked timesteps
            mask: [B, T] binary mask (1 = masked)
        """
        B, T, N, D = slots.shape

        # Determine number of timesteps to mask
        num_masked = int(T * mask_ratio)
        num_masked = max(min_masked, min(num_masked, max_masked))
        num_masked = min(num_masked, T - 1)  # Keep at least one visible

        # Create mask: [B, T]
        mask = torch.zeros(B, T, device=slots.device, dtype=torch.bool)

        for b in range(B):
            # Randomly select timesteps to mask
            masked_indices = torch.randperm(T, device=slots.device)[:num_masked]
            mask[b, masked_indices] = True

        # Apply masking
        masked_slots = slots.clone()
        mask_expanded = mask.unsqueeze(2).unsqueeze(-1)  # [B, T, 1, 1]
        masked_slots = torch.where(
            mask_expanded,
            self.mask_token.view(1, 1, 1, D),
            masked_slots
        )

        return masked_slots, mask


class FutureMasker(nn.Module):
    """
    Future masking: Mask random slots in future predictions.

    Used during training to predict masked future slots from history.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        future_slots: torch.Tensor,
        mask_ratio: float
    ) -> torch.Tensor:
        """
        Args:
            future_slots: [B, T_future, N, D] target future slots
            mask_ratio: Ratio of future slots to mask

        Returns:
            mask: [B, T_future, N] binary mask for loss computation
        """
        B, T, N, _ = future_slots.shape

        # Random masking
        mask = torch.rand(B, T, N, device=future_slots.device) < mask_ratio

        return mask


class CJEPAMasker(nn.Module):
    """
    Combined C-JEPA masking strategy.

    Randomly samples from three masking modes:
    - Object-level masking
    - Temporal masking
    - Future masking
    """

    def __init__(self, slot_dim: int, config: MaskConfig):
        super().__init__()
        self.config = config

        self.object_masker = ObjectMasker(slot_dim)
        self.temporal_masker = TemporalMasker(slot_dim)
        self.future_masker = FutureMasker()

    def forward(
        self,
        history_slots: torch.Tensor,
        future_slots: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply masking strategy during training.

        Args:
            history_slots: [B, T_hist, N, D] history slot embeddings
            future_slots: [B, T_future, N, D] future slots (optional)
            training: Whether in training mode

        Returns:
            masked_history: [B, T_hist, N, D] masked history
            history_mask: [B, T_hist, N] or [B, N] binary mask (or None)
            future_mask: [B, T_future, N] binary mask for future (or None)
        """
        if not training:
            # No masking during inference
            return history_slots, None, None

        B, T, N, D = history_slots.shape

        # Select masking strategy
        if self.config.strategy == 'mixed':
            # Sample strategy based on probabilities
            strategies = ['object', 'temporal', 'future']
            probs = [
                self.config.object_prob,
                self.config.temporal_prob,
                self.config.future_prob
            ]
            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / probs.sum()
            strategy = np.random.choice(strategies, p=probs)
        else:
            strategy = self.config.strategy

        # Apply selected masking strategy
        history_mask = None
        future_mask = None

        if strategy == 'object':
            masked_history, object_mask = self.object_masker(
                history_slots,
                self.config.object_mask_ratio,
                self.config.min_masked_objects,
                self.config.max_masked_objects
            )
            # Expand object mask to all timesteps: [B, N] -> [B, T, N]
            history_mask = object_mask.unsqueeze(1).expand(-1, T, -1)

        elif strategy == 'temporal':
            masked_history, temporal_mask = self.temporal_masker(
                history_slots,
                self.config.temporal_mask_ratio,
                self.config.min_masked_frames,
                self.config.max_masked_frames
            )
            # Expand temporal mask to all slots: [B, T] -> [B, T, N]
            history_mask = temporal_mask.unsqueeze(2).expand(-1, -1, N)

        elif strategy == 'future' and future_slots is not None:
            # No history masking, only future masking
            masked_history = history_slots
            future_mask = self.future_masker(
                future_slots,
                self.config.future_mask_ratio
            )

        else:
            # No masking
            masked_history = history_slots

        return masked_history, history_mask, future_mask


if __name__ == "__main__":
    # Test masking strategies
    print("Testing Masking Strategies...")

    B, T_hist, T_future, N, D = 4, 4, 6, 11, 128
    history = torch.randn(B, T_hist, N, D)
    future = torch.randn(B, T_future, N, D)

    # Test object masking
    obj_masker = ObjectMasker(slot_dim=D)
    masked_hist, obj_mask = obj_masker(history, mask_ratio=0.3)
    print(f"Object Masking:")
    print(f"  Input: {history.shape}")
    print(f"  Masked: {masked_hist.shape}")
    print(f"  Mask: {obj_mask.shape}, masked objects: {obj_mask.sum(dim=1)}")

    # Test temporal masking
    temp_masker = TemporalMasker(slot_dim=D)
    masked_hist, temp_mask = temp_masker(history, mask_ratio=0.25)
    print(f"\nTemporal Masking:")
    print(f"  Masked: {masked_hist.shape}")
    print(f"  Mask: {temp_mask.shape}, masked frames: {temp_mask.sum(dim=1)}")

    # Test future masking
    fut_masker = FutureMasker()
    fut_mask = fut_masker(future, mask_ratio=0.3)
    print(f"\nFuture Masking:")
    print(f"  Mask: {fut_mask.shape}, masked ratio: {fut_mask.float().mean():.3f}")

    # Test combined masker
    config = MaskConfig(strategy='mixed')
    cjepa_masker = CJEPAMasker(slot_dim=D, config=config)

    print(f"\nC-JEPA Combined Masking:")
    for i in range(3):
        masked_hist, hist_mask, fut_mask = cjepa_masker(
            history, future, training=True
        )
        print(f"  Trial {i+1}: hist_mask={hist_mask is not None}, "
              f"fut_mask={fut_mask is not None}")

    # Test inference mode (no masking)
    masked_hist, hist_mask, fut_mask = cjepa_masker(
        history, future, training=False
    )
    print(f"\nInference mode: hist_mask={hist_mask}, fut_mask={fut_mask}")

    print("\n✓ All masking tests passed!")
