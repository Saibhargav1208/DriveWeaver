"""
Drive-JEPA Loss Functions (Phase 4)

Multimodal trajectory planning losses:
    1. minADE: Minimum average displacement error (best-of-M)
    2. minFDE: Minimum final displacement error
    3. Winner-Takes-All: Strongly supervise best proposal
    4. Diversity: Encourage proposal diversity
    5. Smoothness: Penalize high jerk/acceleration
    6. Feasibility: Kinematic constraints (optional)

Key principle: Balance between accuracy (minADE) and diversity (coverage of modes).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_ade(
    predictions: torch.Tensor,    # [B, M, T, 2]
    targets: torch.Tensor,        # [B, T, 2]
) -> torch.Tensor:
    """
    Compute Average Displacement Error per proposal.

    Args:
        predictions: [B, M, T, 2]  M trajectory proposals
        targets: [B, T, 2]  ground truth trajectory

    Returns:
        ade_per_proposal: [B, M]  ADE for each proposal
    """
    B, M, T, _ = predictions.shape

    # Expand targets to match proposals
    targets_exp = targets.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 2]

    # L2 distance per timestep
    errors = torch.norm(predictions - targets_exp, p=2, dim=-1)  # [B, M, T]

    # Average over time
    ade = errors.mean(dim=-1)  # [B, M]

    return ade


def compute_fde(
    predictions: torch.Tensor,    # [B, M, T, 2]
    targets: torch.Tensor,        # [B, T, 2]
) -> torch.Tensor:
    """
    Compute Final Displacement Error per proposal.

    Args:
        predictions: [B, M, T, 2]
        targets: [B, T, 2]

    Returns:
        fde_per_proposal: [B, M]
    """
    B, M = predictions.shape[:2]

    # Final timestep
    pred_final = predictions[:, :, -1, :]  # [B, M, 2]
    target_final = targets[:, -1, :].unsqueeze(1).expand(-1, M, -1)  # [B, M, 2]

    # L2 distance
    fde = torch.norm(pred_final - target_final, p=2, dim=-1)  # [B, M]

    return fde


def compute_min_ade(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minimum ADE across all proposals (best-of-M).

    Returns:
        min_ade: scalar loss
        best_indices: [B] indices of best proposal per batch
    """
    ade_per_proposal = compute_ade(predictions, targets)  # [B, M]
    min_ade, best_indices = ade_per_proposal.min(dim=1)  # [B]
    return min_ade.mean(), best_indices


def compute_min_fde(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minimum FDE across all proposals.

    Returns:
        min_fde: scalar loss
        best_indices: [B] indices of best proposal per batch
    """
    fde_per_proposal = compute_fde(predictions, targets)  # [B, M]
    min_fde, best_indices = fde_per_proposal.min(dim=1)  # [B]
    return min_fde.mean(), best_indices


def winner_takes_all_loss(
    predictions: torch.Tensor,    # [B, M, T, 2]
    targets: torch.Tensor,        # [B, T, 2]
    best_indices: torch.Tensor,   # [B] indices of best proposals
) -> torch.Tensor:
    """
    Winner-takes-all loss: Strongly supervise the best proposal.

    This encourages the best mode to become even better, rather than just
    improving the worst modes.

    Args:
        predictions: [B, M, T, 2]
        targets: [B, T, 2]
        best_indices: [B] which proposal is best per batch element

    Returns:
        wta_loss: scalar
    """
    B = predictions.shape[0]

    # Select best proposal per batch element
    best_proposals = predictions[range(B), best_indices]  # [B, T, 2]

    # MSE loss on best proposal
    loss = F.mse_loss(best_proposals, targets)

    return loss


def diversity_loss(
    predictions: torch.Tensor,    # [B, M, T, 2]
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Diversity loss: Encourage proposals to be different from each other.

    Uses negative pairwise distances to maximize diversity.

    Args:
        predictions: [B, M, T, 2]
        sigma: temperature for distance (higher = more diversity encouraged)

    Returns:
        div_loss: scalar (negative, to maximize diversity)
    """
    B, M, T, _ = predictions.shape

    # Flatten trajectory to vector per proposal
    pred_flat = predictions.reshape(B, M, -1)  # [B, M, T*2]

    # Compute pairwise L2 distances between proposals
    # pred_flat: [B, M, D]
    # dist[b, i, j] = ||pred[b,i] - pred[b,j]||

    diff = pred_flat.unsqueeze(2) - pred_flat.unsqueeze(1)  # [B, M, M, D]
    pairwise_dist = torch.norm(diff, p=2, dim=-1)  # [B, M, M]

    # Average distance (excluding diagonal)
    mask = 1.0 - torch.eye(M, device=predictions.device).unsqueeze(0)  # [1, M, M]
    masked_dist = pairwise_dist * mask
    avg_dist = masked_dist.sum() / (mask.sum() + 1e-8)

    # Negative loss (we want to maximize distance)
    # Log-space keeps gradient stable and prevents dominating the total loss
    loss = -torch.log(avg_dist / sigma + 1.0)

    return loss


def smoothness_loss(
    predictions: torch.Tensor,    # [B, M, T, 2]
    penalize_jerk: bool = True,
) -> torch.Tensor:
    """
    Smoothness loss: Penalize high acceleration or jerk.

    Jerk = third derivative of position (derivative of acceleration).
    High jerk = uncomfortable driving.

    Args:
        predictions: [B, M, T, 2]
        penalize_jerk: If True, penalize jerk; else penalize acceleration

    Returns:
        smooth_loss: scalar
    """
    if predictions.shape[2] < 3:
        return torch.tensor(0.0, device=predictions.device)

    # First derivative: velocity (approximate)
    vel = predictions[:, :, 1:] - predictions[:, :, :-1]  # [B, M, T-1, 2]

    if not penalize_jerk:
        # Penalize high velocity changes (acceleration)
        accel = vel[:, :, 1:] - vel[:, :, :-1]  # [B, M, T-2, 2]
        loss = (accel ** 2).mean()
        return loss

    # Second derivative: acceleration
    if predictions.shape[2] < 4:
        accel = vel[:, :, 1:] - vel[:, :, :-1]
        loss = (accel ** 2).mean()
        return loss

    accel = vel[:, :, 1:] - vel[:, :, :-1]  # [B, M, T-2, 2]

    # Third derivative: jerk
    jerk = accel[:, :, 1:] - accel[:, :, :-1]  # [B, M, T-3, 2]

    # L2 penalty
    loss = (jerk ** 2).mean()

    return loss


def feasibility_loss(
    predictions: torch.Tensor,    # [B, M, T, 2]
    max_speed: float = 15.0,      # m/s (~54 km/h for nuScenes)
    max_accel: float = 4.0,       # m/s^2
    dt: float = 0.5,              # time step (nuScenes is 2Hz)
) -> torch.Tensor:
    """
    Feasibility loss: Penalize kinematic constraint violations.

    Constraints:
        - Maximum speed
        - Maximum acceleration

    Args:
        predictions: [B, M, T, 2]
        max_speed: maximum allowed speed (m/s)
        max_accel: maximum allowed acceleration (m/s^2)
        dt: time step between frames

    Returns:
        feasibility_loss: scalar
    """
    if predictions.shape[2] < 2:
        return torch.tensor(0.0, device=predictions.device)

    # Velocity (displacement per timestep)
    displacement = predictions[:, :, 1:] - predictions[:, :, :-1]  # [B, M, T-1, 2]
    speed = torch.norm(displacement, p=2, dim=-1) / dt  # [B, M, T-1]

    # Speed violation
    speed_violation = F.relu(speed - max_speed)
    loss_speed = speed_violation.mean()

    if predictions.shape[2] < 3:
        return loss_speed

    # Acceleration
    velocity = displacement / dt  # [B, M, T-1, 2]
    accel_vec = velocity[:, :, 1:] - velocity[:, :, :-1]  # [B, M, T-2, 2]
    accel_mag = torch.norm(accel_vec, p=2, dim=-1) / dt  # [B, M, T-2]

    # Acceleration violation
    accel_violation = F.relu(accel_mag - max_accel)
    loss_accel = accel_violation.mean()

    return loss_speed + loss_accel


class PlannerLoss(nn.Module):
    """
    Combined loss for multimodal trajectory planning.

    The trajectory terms train the proposal set. The score terms train the
    proposal scorer so normal inference can select a proposal without GT.

    L_total = trajectory losses
            + lambda_score_ce * CE(proposal_scores, argmin_ADE)
            + lambda_score_kl * KL(softmax(scores), softmax(-ADE / tau))
            + lambda_all_ade * mean_ADE_over_all_modes
    """

    def __init__(
        self,
        lambda_ade: float = 1.0,
        lambda_fde: float = 0.5,
        lambda_wta: float = 0.5,
        lambda_diversity: float = 0.1,
        lambda_smoothness: float = 0.1,
        lambda_feasibility: float = 0.05,
        diversity_sigma: float = 1.0,
        penalize_jerk: bool = True,
        max_speed: float = 15.0,
        max_accel: float = 4.0,
        dt: float = 0.5,
        lambda_score_ce: float = 1.0,
        lambda_score_kl: float = 0.0,
        score_temperature: float = 0.5,
        lambda_all_ade: float = 0.05,
    ):
        super().__init__()

        self.lambda_ade = lambda_ade
        self.lambda_fde = lambda_fde
        self.lambda_wta = lambda_wta
        self.lambda_diversity = lambda_diversity
        self.lambda_smoothness = lambda_smoothness
        self.lambda_feasibility = lambda_feasibility
        self.lambda_score_ce = lambda_score_ce
        self.lambda_score_kl = lambda_score_kl
        self.lambda_all_ade = lambda_all_ade

        self.diversity_sigma = diversity_sigma
        self.penalize_jerk = penalize_jerk
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.dt = dt
        self.score_temperature = score_temperature

    def forward(
        self,
        predictions: torch.Tensor,    # [B, M, T, 2]
        targets: torch.Tensor,        # [B, T, 2]
        proposal_scores: Optional[torch.Tensor] = None,  # [B, M]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined proposal and proposal-selection loss."""

        ade_per_proposal = compute_ade(predictions, targets)  # [B, M]
        loss_ade, best_indices = ade_per_proposal.min(dim=1)
        loss_ade = loss_ade.mean()

        fde_per_proposal = compute_fde(predictions, targets)  # [B, M]
        loss_fde = fde_per_proposal.min(dim=1)[0].mean()

        loss_wta = winner_takes_all_loss(predictions, targets, best_indices)
        loss_div = diversity_loss(predictions, sigma=self.diversity_sigma)
        loss_smooth = smoothness_loss(predictions, penalize_jerk=self.penalize_jerk)
        loss_feas = feasibility_loss(
            predictions,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
            dt=self.dt,
        )
        loss_all_ade = ade_per_proposal.mean()

        zero = predictions.new_tensor(0.0)
        loss_score_ce = zero
        loss_score_kl = zero
        score_acc = zero
        score_avg_rank = zero

        if proposal_scores is not None:
            best_indices_detached = best_indices.detach()
            if self.lambda_score_ce > 0:
                loss_score_ce = F.cross_entropy(proposal_scores, best_indices_detached)

            if self.lambda_score_kl > 0:
                tau = max(float(self.score_temperature), 1e-6)
                target_probs = F.softmax(-ade_per_proposal.detach() / tau, dim=1)
                score_log_probs = F.log_softmax(proposal_scores, dim=1)
                loss_score_kl = F.kl_div(score_log_probs, target_probs, reduction='batchmean')

            selected_indices = proposal_scores.argmax(dim=1)
            score_acc = (selected_indices == best_indices_detached).float().mean()
            sorted_by_ade = torch.argsort(ade_per_proposal.detach(), dim=1)
            rank_matches = (sorted_by_ade == selected_indices.unsqueeze(1)).nonzero(as_tuple=False)
            if rank_matches.numel() > 0:
                score_avg_rank = (rank_matches[:, 1].float() + 1.0).mean()

        total_loss = (
            self.lambda_ade * loss_ade
            + self.lambda_fde * loss_fde
            + self.lambda_wta * loss_wta
            + self.lambda_diversity * loss_div
            + self.lambda_smoothness * loss_smooth
            + self.lambda_feasibility * loss_feas
            + self.lambda_score_ce * loss_score_ce
            + self.lambda_score_kl * loss_score_kl
            + self.lambda_all_ade * loss_all_ade
        )

        metrics = {
            'loss_total': total_loss.item(),
            'loss_minADE': loss_ade.item(),
            'loss_minFDE': loss_fde.item(),
            'loss_wta': loss_wta.item(),
            'loss_diversity': loss_div.item(),
            'loss_smoothness': loss_smooth.item(),
            'loss_feasibility': loss_feas.item(),
            'loss_score_ce': loss_score_ce.item(),
            'loss_score_kl': loss_score_kl.item(),
            'loss_allADE': loss_all_ade.item(),
            'score_acc': score_acc.item(),
            'score_avg_rank': score_avg_rank.item(),
        }

        return total_loss, metrics

def compute_trajectory_metrics(
    predictions: torch.Tensor,    # [B, M, T, 2]
    targets: torch.Tensor,        # [B, T, 2]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for trajectory predictions.

    Returns dict with:
        - minADE: best-of-M average displacement error
        - minFDE: best-of-M final displacement error
        - avgADE: average ADE across all proposals
        - avgFDE: average FDE across all proposals
        - diversity: average pairwise distance between proposals
    """

    # ADE per proposal
    ade_per_proposal = compute_ade(predictions, targets)  # [B, M]
    min_ade = ade_per_proposal.min(dim=1)[0].mean().item()
    avg_ade = ade_per_proposal.mean().item()

    # FDE per proposal
    fde_per_proposal = compute_fde(predictions, targets)  # [B, M]
    min_fde = fde_per_proposal.min(dim=1)[0].mean().item()
    avg_fde = fde_per_proposal.mean().item()

    # Diversity (average pairwise distance)
    B, M, T, _ = predictions.shape
    pred_flat = predictions.reshape(B, M, -1)
    diff = pred_flat.unsqueeze(2) - pred_flat.unsqueeze(1)
    pairwise_dist = torch.norm(diff, p=2, dim=-1)
    mask = 1.0 - torch.eye(M, device=predictions.device).unsqueeze(0)
    diversity = (pairwise_dist * mask).sum() / (mask.sum() + 1e-8)

    return {
        'minADE': min_ade,
        'minFDE': min_fde,
        'avgADE': avg_ade,
        'avgFDE': avg_fde,
        'diversity': diversity.item(),
    }


if __name__ == "__main__":
    print("Testing Drive-JEPA Loss Functions...")

    B, M, T = 4, 32, 6

    # Random predictions and targets
    predictions = torch.randn(B, M, T, 2, requires_grad=True)
    targets = torch.randn(B, T, 2)

    # Test individual losses
    print("\n[1/7] minADE")
    loss_ade, best_idx = compute_min_ade(predictions, targets)
    print(f"  minADE: {loss_ade.item():.4f}")
    print(f"  Best indices: {best_idx}")

    print("\n[2/7] minFDE")
    loss_fde, _ = compute_min_fde(predictions, targets)
    print(f"  minFDE: {loss_fde.item():.4f}")

    print("\n[3/7] Winner-takes-all")
    loss_wta = winner_takes_all_loss(predictions, targets, best_idx)
    print(f"  WTA: {loss_wta.item():.4f}")

    print("\n[4/7] Diversity")
    loss_div = diversity_loss(predictions)
    print(f"  Diversity: {loss_div.item():.4f} (negative = good)")

    print("\n[5/7] Smoothness")
    loss_smooth = smoothness_loss(predictions, penalize_jerk=True)
    print(f"  Smoothness: {loss_smooth.item():.4f}")

    print("\n[6/7] Feasibility")
    loss_feas = feasibility_loss(predictions)
    print(f"  Feasibility: {loss_feas.item():.4f}")

    print("\n[7/7] Combined Loss")
    loss_fn = PlannerLoss(
        lambda_ade=1.0,
        lambda_fde=0.5,
        lambda_wta=0.5,
        lambda_diversity=0.1,
        lambda_smoothness=0.1,
        lambda_feasibility=0.05,
    )

    total_loss, metrics = loss_fn(predictions, targets)
    print(f"  Total loss: {total_loss.item():.4f}")
    print("  Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Test metrics
    print("\n[Evaluation Metrics]")
    eval_metrics = compute_trajectory_metrics(predictions, targets)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Test backward
    print("\n[Backward Pass]")
    total_loss.backward()
    print("  ✓ Gradients computed")

    print("\n✓ All loss function tests passed!")
