"""
Autoregressive Rollout Evaluation for C-JEPA

Evaluates long-horizon prediction via autoregressive rollout:
1. Predict future from history
2. Use predicted frames as new history
3. Repeat for N steps

Metrics:
- Short-term accuracy (1-3 steps)
- Long-term rollout error (6+ steps)
- Temporal consistency
- Visualization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


def compute_rollout_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for rollout evaluation.

    Args:
        predictions: [B, T, N, D] predicted slots
        targets: [B, T, N, D] target slots

    Returns:
        dict of metrics
    """
    B, T, N, D = predictions.shape

    # MSE per timestep
    mse_per_step = []
    for t in range(T):
        mse = torch.mean((predictions[:, t] - targets[:, t]) ** 2).item()
        mse_per_step.append(mse)

    # Overall MSE
    overall_mse = torch.mean((predictions - targets) ** 2).item()

    # MAE
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # Temporal consistency: correlation between consecutive frames
    pred_diff = predictions[:, 1:] - predictions[:, :-1]
    target_diff = targets[:, 1:] - targets[:, :-1]
    consistency = torch.nn.functional.cosine_similarity(
        pred_diff.reshape(-1, D),
        target_diff.reshape(-1, D),
        dim=-1
    ).mean().item()

    # Slot diversity: std across slot dimension
    pred_diversity = torch.std(predictions, dim=2).mean().item()
    target_diversity = torch.std(targets, dim=2).mean().item()

    return {
        'overall_mse': overall_mse,
        'mae': mae,
        'temporal_consistency': consistency,
        'pred_diversity': pred_diversity,
        'target_diversity': target_diversity,
        'mse_per_step': mse_per_step,
        'short_term_mse': np.mean(mse_per_step[:3]),  # First 3 steps
        'long_term_mse': np.mean(mse_per_step[3:]) if T > 3 else 0.0
    }


def visualize_rollout_pca(
    history: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[Path] = None,
    slot_idx: int = 0
):
    """
    Visualize rollout in PCA space for a single slot.

    Args:
        history: [T_hist, N, D] history slots
        predictions: [T_future, N, D] predicted future
        targets: [T_future, N, D] target future
        save_path: Path to save figure
        slot_idx: Which slot to visualize
    """
    T_hist = history.shape[0]
    T_future = predictions.shape[0]

    # Extract single slot trajectory
    hist_slot = history[:, slot_idx, :].cpu().numpy()      # [T_hist, D]
    pred_slot = predictions[:, slot_idx, :].cpu().numpy()  # [T_future, D]
    target_slot = targets[:, slot_idx, :].cpu().numpy()    # [T_future, D]

    # Combine for PCA
    all_slots = np.concatenate([hist_slot, pred_slot, target_slot], axis=0)

    # PCA to 2D
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_slots)

    hist_pca = all_pca[:T_hist]
    pred_pca = all_pca[T_hist:T_hist + T_future]
    target_pca = all_pca[T_hist + T_future:]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # History (black)
    ax.plot(hist_pca[:, 0], hist_pca[:, 1], 'o-', color='black',
            label='History', linewidth=2, markersize=8)

    # Target (green)
    ax.plot(target_pca[:, 0], target_pca[:, 1], 'o-', color='green',
            label='Target Future', linewidth=2, markersize=8, alpha=0.7)

    # Prediction (red)
    ax.plot(pred_pca[:, 0], pred_pca[:, 1], 'o-', color='red',
            label='Predicted Future', linewidth=2, markersize=8, alpha=0.7)

    # Mark start and end
    ax.scatter(hist_pca[0, 0], hist_pca[0, 1], s=200, color='blue',
               marker='s', label='Start', zorder=5)
    ax.scatter(target_pca[-1, 0], target_pca[-1, 1], s=200, color='purple',
               marker='X', label='Target End', zorder=5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title(f'Rollout Trajectory in PCA Space (Slot {slot_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved rollout visualization: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_mse_per_step(
    mse_per_step: List[float],
    save_path: Optional[Path] = None
):
    """
    Plot MSE vs rollout step.

    Args:
        mse_per_step: List of MSE values per step
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = list(range(1, len(mse_per_step) + 1))
    ax.plot(steps, mse_per_step, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Rollout Step')
    ax.set_ylabel('MSE')
    ax.set_title('Prediction Error vs Rollout Step')
    ax.grid(True, alpha=0.3)

    # Mark short-term vs long-term
    if len(mse_per_step) > 3:
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.5,
                   label='Short/Long-term boundary')
        ax.legend()

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


@torch.no_grad()
def evaluate_rollout(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_rollout_steps: int = 12,
    device: torch.device = torch.device('cuda'),
    num_visualizations: int = 4,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Evaluate model with autoregressive rollout.

    Args:
        model: C-JEPA predictor model
        dataloader: Validation dataloader
        num_rollout_steps: Number of steps to roll out
        device: Device to run on
        num_visualizations: Number of samples to visualize
        output_dir: Directory to save visualizations

    Returns:
        dict of aggregated metrics
    """
    model.eval()

    all_metrics = {
        'overall_mse': [],
        'mae': [],
        'temporal_consistency': [],
        'short_term_mse': [],
        'long_term_mse': []
    }

    visualization_count = 0

    print(f"\nRunning autoregressive rollout ({num_rollout_steps} steps)...")

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        history_slots = batch['history_slots'].to(device)  # [B, T_hist, N, D]
        future_slots = batch['future_slots'].to(device)    # [B, T_fut, N, D]

        B, T_hist, N, D = history_slots.shape
        T_fut = future_slots.shape[1]

        # Determine how many full rollout steps we can do
        max_steps = min(num_rollout_steps, T_fut)

        # Autoregressive rollout
        rollout = model.autoregressive_rollout(
            history_slots,
            num_steps=max_steps
        )  # [B, max_steps, N, D]

        # Get targets (first max_steps of future)
        targets = future_slots[:, :max_steps, :, :]

        # Compute metrics
        metrics = compute_rollout_metrics(rollout, targets)

        for key in all_metrics.keys():
            if key in metrics:
                all_metrics[key].append(metrics[key])

        # Visualize first few samples
        if visualization_count < num_visualizations and output_dir:
            for b in range(min(B, num_visualizations - visualization_count)):
                # Visualize in PCA space
                vis_path = output_dir / f'rollout_batch{batch_idx}_sample{b}.png'
                visualize_rollout_pca(
                    history=history_slots[b],
                    predictions=rollout[b],
                    targets=targets[b],
                    save_path=vis_path,
                    slot_idx=0  # Visualize first slot
                )

                # Plot MSE per step
                mse_path = output_dir / f'mse_batch{batch_idx}_sample{b}.png'
                visualize_mse_per_step(
                    mse_per_step=metrics['mse_per_step'],
                    save_path=mse_path
                )

                visualization_count += 1

                if visualization_count >= num_visualizations:
                    break

    # Aggregate metrics
    aggregated = {
        key: np.mean(values) for key, values in all_metrics.items()
    }

    print("\n=== Rollout Evaluation Results ===")
    print(f"Overall MSE: {aggregated['overall_mse']:.4f}")
    print(f"MAE: {aggregated['mae']:.4f}")
    print(f"Temporal Consistency: {aggregated['temporal_consistency']:.4f}")
    print(f"Short-term MSE (steps 1-3): {aggregated['short_term_mse']:.4f}")
    print(f"Long-term MSE (steps 4+): {aggregated['long_term_mse']:.4f}")

    return aggregated


if __name__ == "__main__":
    # Test rollout evaluation
    print("Testing Rollout Evaluation...")

    B, T_hist, T_fut, N, D = 4, 4, 12, 11, 128

    # Dummy data
    predictions = torch.randn(B, T_fut, N, D)
    targets = torch.randn(B, T_fut, N, D)

    # Compute metrics
    metrics = compute_rollout_metrics(predictions, targets)
    print("\n=== Metrics ===")
    for key, value in metrics.items():
        if key != 'mse_per_step':
            print(f"  {key}: {value:.4f}")

    print(f"  MSE per step: {[f'{x:.3f}' for x in metrics['mse_per_step']]}")

    # Test visualization
    print("\nTesting visualization...")
    history = torch.randn(T_hist, N, D)
    pred_fut = torch.randn(T_fut, N, D)
    target_fut = torch.randn(T_fut, N, D)

    visualize_rollout_pca(
        history=history,
        predictions=pred_fut,
        targets=target_fut,
        save_path=None,  # Show instead of save
        slot_idx=0
    )

    print("\n✓ All rollout tests passed!")
