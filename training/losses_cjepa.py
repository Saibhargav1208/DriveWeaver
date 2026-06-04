"""
C-JEPA Loss Functions (Exact Match to Original)

Based on: /research/cjepa/src/train/train_causalwm_from_clevrer_slot.py

The original C-JEPA uses ONLY MSE loss:
1. loss_masked_history: MSE on masked slots in history
2. loss_future: MSE on future prediction
3. total_loss = loss_masked_history + loss_future

NO temporal smoothness, NO diversity regularization, NO other losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def compute_cjepa_loss(
    pred_full: torch.Tensor,
    history: torch.Tensor,
    target_future: torch.Tensor,
    masked_indices: torch.Tensor,
    history_size: int
) -> Dict[str, torch.Tensor]:
    """
    Compute C-JEPA loss exactly as in the original paper.

    Args:
        pred_full: [B, T_total, N, D] predicted full sequence (history + future)
        history: [B, T_hist, N, D] ground truth history
        target_future: [B, T_future, N, D] ground truth future
        masked_indices: [num_masked] indices of masked slots
        history_size: Number of history frames (T_hist)

    Returns:
        Dict with keys:
            - loss: total loss
            - loss_masked_history: loss on masked history slots
            - loss_future: loss on future prediction
    """
    # Split prediction into history and future
    pred_history = pred_full[:, :history_size, :, :]  # [B, T_hist, N, D]
    pred_future = pred_full[:, history_size:, :, :]    # [B, T_future, N, D]

    losses = {}

    # 1. Loss on masked slots in history
    if len(masked_indices) > 0:
        loss_masked_history = F.mse_loss(
            pred_history[:, :, masked_indices, :],
            history[:, :, masked_indices, :].detach()
        )
        losses['loss_masked_history'] = loss_masked_history
    else:
        loss_masked_history = torch.tensor(0.0, device=pred_full.device)
        losses['loss_masked_history'] = loss_masked_history

    # 2. Loss on future prediction
    loss_future = F.mse_loss(pred_future, target_future.detach())
    losses['loss_future'] = loss_future

    # 3. Total loss
    total_loss = loss_masked_history + loss_future
    losses['loss'] = total_loss

    return losses


def compute_cjepa_loss_inference(
    pred_future: torch.Tensor,
    target_future: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute C-JEPA loss during inference (no masking).

    Args:
        pred_future: [B, T_future, N, D] predicted future
        target_future: [B, T_future, N, D] ground truth future

    Returns:
        Dict with keys:
            - loss: total loss
            - loss_future: loss on future prediction
            - loss_masked_history: zero (for consistency)
    """
    loss_future = F.mse_loss(pred_future, target_future.detach())

    losses = {
        'loss_future': loss_future,
        'loss_masked_history': torch.tensor(0.0, device=pred_future.device),
        'loss': loss_future
    }

    return losses


class CJEPALoss(nn.Module):
    """
    C-JEPA Loss Module (wrapper for compatibility with existing code).

    Uses pure MSE loss as in the original paper.
    """

    def __init__(self):
        super().__init__()
        # No parameters needed - just MSE

    def forward(
        self,
        pred_full: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        target_future: Optional[torch.Tensor] = None,
        masked_indices: Optional[torch.Tensor] = None,
        history_size: Optional[int] = None,
        pred_future: Optional[torch.Tensor] = None,
        inference: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (supports both training and inference modes).

        Training mode (with masking):
            pred_full: [B, T_total, N, D]
            history: [B, T_hist, N, D]
            target_future: [B, T_future, N, D]
            masked_indices: [num_masked]
            history_size: int

        Inference mode (no masking):
            pred_future: [B, T_future, N, D]
            target_future: [B, T_future, N, D]
        """
        if inference or pred_future is not None:
            # Inference mode
            if pred_future is None:
                raise ValueError("pred_future required for inference mode")
            if target_future is None:
                raise ValueError("target_future required for inference mode")
            return compute_cjepa_loss_inference(pred_future, target_future)
        else:
            # Training mode
            if pred_full is None or history is None or target_future is None:
                raise ValueError("pred_full, history, target_future required for training")
            if masked_indices is None:
                masked_indices = torch.tensor([], dtype=torch.long, device=pred_full.device)
            if history_size is None:
                raise ValueError("history_size required for training")

            return compute_cjepa_loss(
                pred_full, history, target_future, masked_indices, history_size
            )


if __name__ == "__main__":
    # Test C-JEPA loss
    print("Testing C-JEPA Loss...")

    B, T_hist, T_fut, N, D = 4, 4, 6, 11, 128

    # Test training mode
    pred_full = torch.randn(B, T_hist + T_fut, N, D)
    history = torch.randn(B, T_hist, N, D)
    target_future = torch.randn(B, T_fut, N, D)
    masked_indices = torch.tensor([0, 5])  # 2 masked slots

    losses = compute_cjepa_loss(
        pred_full, history, target_future, masked_indices, T_hist
    )

    print(f"\nTraining mode:")
    print(f"  loss: {losses['loss'].item():.4f}")
    print(f"  loss_masked_history: {losses['loss_masked_history'].item():.4f}")
    print(f"  loss_future: {losses['loss_future'].item():.4f}")

    # Test inference mode
    pred_future = torch.randn(B, T_fut, N, D)
    losses_inf = compute_cjepa_loss_inference(pred_future, target_future)

    print(f"\nInference mode:")
    print(f"  loss: {losses_inf['loss'].item():.4f}")
    print(f"  loss_future: {losses_inf['loss_future'].item():.4f}")

    # Test module
    criterion = CJEPALoss()

    losses_module = criterion(
        pred_full=pred_full,
        history=history,
        target_future=target_future,
        masked_indices=masked_indices,
        history_size=T_hist
    )

    print(f"\nModule (training):")
    print(f"  loss: {losses_module['loss'].item():.4f}")

    losses_module_inf = criterion(
        pred_future=pred_future,
        target_future=target_future,
        inference=True
    )

    print(f"\nModule (inference):")
    print(f"  loss: {losses_module_inf['loss'].item():.4f}")

    # Test backward
    losses_module['loss'].backward()
    print(f"\n✓ Backward pass successful")

    print("\n✓ All C-JEPA loss tests passed!")
