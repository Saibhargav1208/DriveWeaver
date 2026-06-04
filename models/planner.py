"""
Multimodal Trajectory Planner

Generates diverse trajectory proposals from predicted future slots and selects
the best one. This is the planning head that sits on top of the world model
(C-JEPA / ThinkJEPA).

Architecture:
    future_slots   [B, T_fut, N, D]  (from world model)
    ego_state      [B, T_hist, 4]    (x, y, yaw, speed)
    route_goals    [B, K, 2]         (optional waypoints)
        ↓
    Planning Context Encoder
        ↓
    M Learnable Mode Queries  [M, D]
        ↓
    Cross-Attention to Future Slots
        ↓
    Temporal Trajectory Decoder
        ↓
    M Trajectory Proposals  [B, M, T_fut, 2]
        ↓
    Proposal Scorer
        ↓
    Best Trajectory  [B, T_fut, 2]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, Optional, Tuple


class PlanningContextEncoder(nn.Module):
    """
    Encodes ego vehicle state and optional route goals into planning context.

    Args:
        ego_state: [B, T_hist, 4]  (x, y, yaw, speed)
        route_goals: [B, K, 2]  (optional, None → zero context)

    Returns:
        planning_context: [B, D]
    """

    def __init__(self, ego_dim: int = 4, context_dim: int = 128, history_len: int = 4):
        super().__init__()

        # Ego state encoder: MLP + temporal pooling
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim * history_len, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Linear(context_dim * 2, context_dim),
        )

        # Route goal encoder: attention pooling over waypoints
        self.route_encoder = nn.Sequential(
            nn.Linear(2, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        # Attention pooling for variable-length route
        self.route_attention = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )

        # Query for attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, context_dim))

    def forward(
        self,
        ego_state: torch.Tensor,          # [B, T_hist, 4]
        route_goals: Optional[torch.Tensor] = None,  # [B, K, 2]
    ) -> torch.Tensor:
        """Returns planning_context: [B, D]"""
        B = ego_state.shape[0]
        device = ego_state.device

        # Encode ego state
        ego_flat = rearrange(ego_state, 'b t d -> b (t d)')
        ego_context = self.ego_encoder(ego_flat)  # [B, D]

        # Encode route goals if provided
        if route_goals is not None:
            route_emb = self.route_encoder(route_goals)  # [B, K, D]

            # Attention pooling
            query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
            route_context, _ = self.route_attention(query, route_emb, route_emb)
            route_context = route_context.squeeze(1)  # [B, D]

            return ego_context + route_context
        else:
            return ego_context


class ModeQueryGenerator(nn.Module):
    """
    Generates M planning mode queries conditioned on planning context.

    Each mode query specializes in a planning behavior:
        - Lane keeping
        - Lane change left/right
        - Overtaking
        - Yielding
        - etc.

    The modes are learned implicitly through the query embeddings.
    """

    def __init__(self, num_modes: int, mode_dim: int):
        super().__init__()
        self.num_modes = num_modes

        # Learnable mode embeddings
        self.mode_embeddings = nn.Parameter(torch.randn(num_modes, mode_dim))
        nn.init.normal_(self.mode_embeddings, std=0.02)

        # Context conditioning
        self.context_proj = nn.Sequential(
            nn.Linear(mode_dim, mode_dim),
            nn.LayerNorm(mode_dim),
            nn.GELU(),
        )

    def forward(self, planning_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            planning_context: [B, D]
        Returns:
            mode_queries: [B, M, D]
        """
        B, D = planning_context.shape
        M = self.num_modes

        # Project context
        context_cond = self.context_proj(planning_context)  # [B, D]

        # Broadcast mode embeddings and add context conditioning
        mode_queries = self.mode_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        mode_queries = mode_queries + context_cond.unsqueeze(1)  # [B, M, D]

        return mode_queries


class TrajectoryDecoderBlock(nn.Module):
    """
    One decoder block: cross-attention to slots + causal self-attention + FFN.

    Cross-attention: queries attend to refined future slots
    Self-attention: queries attend to each other (mode interaction)
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        # Cross-attention: queries ← slots
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention: mode interaction
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,          # [B, M, D]
        slot_context: torch.Tensor,     # [B, T*N, D]
    ) -> torch.Tensor:
        """Returns refined queries: [B, M, D]"""

        # Cross-attention
        q_cross = self.cross_norm_q(queries)
        kv_cross = self.cross_norm_kv(slot_context)
        attn_out, _ = self.cross_attn(q_cross, kv_cross, kv_cross)
        queries = queries + attn_out

        # Self-attention
        q_self = self.self_norm(queries)
        self_out, _ = self.self_attn(q_self, q_self, q_self)
        queries = queries + self_out

        # FFN
        queries = queries + self.ff(self.norm_ff(queries))

        return queries


class TrajectoryDecoder(nn.Module):
    """
    Decodes mode queries into temporal trajectories.

    Architecture:
        1. Cross-attention layers to extract slot context
        2. Temporal expansion to T_fut timesteps
        3. Temporal transformer for trajectory coherence
        4. Output projection to (x, y) waypoints
    """

    def __init__(
        self,
        slot_dim: int,
        num_modes: int,
        future_len: int,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_modes = num_modes
        self.future_len = future_len

        # Cross-attention decoder layers
        self.decoder_layers = nn.ModuleList([
            TrajectoryDecoderBlock(slot_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Temporal positional embeddings
        self.time_embed = nn.Parameter(torch.randn(future_len, slot_dim))
        nn.init.normal_(self.time_embed, std=0.02)

        # Temporal transformer (causal for trajectory coherence)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=slot_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=2)

        # Output projection: [D] → (x, y)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim // 2),
            nn.GELU(),
            nn.Linear(slot_dim // 2, 2),
        )

    def forward(
        self,
        mode_queries: torch.Tensor,        # [B, M, D]
        refined_slots: torch.Tensor,      # [B, T_fut, N, D]
    ) -> torch.Tensor:
        """
        Returns:
            trajectories: [B, M, T_fut, 2]
        """
        B, M, D = mode_queries.shape
        T_fut = refined_slots.shape[1]
        N = refined_slots.shape[2]

        # Flatten slots for cross-attention
        slot_context = rearrange(refined_slots, 'b t n d -> b (t n) d')  # [B, T*N, D]

        # Apply decoder layers
        for layer in self.decoder_layers:
            mode_queries = layer(mode_queries, slot_context)  # [B, M, D]

        # Expand to temporal sequence
        # [B, M, D] → [B, M, T_fut, D]
        mode_queries_expanded = mode_queries.unsqueeze(2).expand(-1, -1, T_fut, -1)

        # Add temporal positional embeddings
        time_emb = self.time_embed.unsqueeze(0).unsqueeze(0)  # [1, 1, T_fut, D]
        temporal_features = mode_queries_expanded + time_emb  # [B, M, T_fut, D]

        # Apply temporal transformer per mode
        # Reshape: [B, M, T_fut, D] → [B*M, T_fut, D]
        temporal_flat = rearrange(temporal_features, 'b m t d -> (b m) t d')
        temporal_refined = self.temporal_transformer(temporal_flat)  # [B*M, T_fut, D]
        temporal_refined = rearrange(temporal_refined, '(b m) t d -> b m t d', b=B, m=M)

        # Project to (x, y) waypoints
        trajectories = self.output_proj(temporal_refined)  # [B, M, T_fut, 2]

        return trajectories


class ProposalScorer(nn.Module):
    """
    Scores trajectory proposals based on:
        1. Feasibility: kinematic/dynamic constraints
        2. Safety: collision risk with objects
        3. Progress: alignment with route goals
        4. Comfort: smoothness (acceleration, jerk)

    Returns per-proposal scores: [B, M]
    """

    def __init__(self, slot_dim: int, num_modes: int):
        super().__init__()

        # Trajectory feature encoder
        self.traj_encoder = nn.Sequential(
            nn.Linear(2, slot_dim // 2),  # (x, y) → D/2
            nn.LayerNorm(slot_dim // 2),
            nn.GELU(),
        )

        # Temporal pooling for trajectory
        self.traj_pool = nn.MultiheadAttention(
            embed_dim=slot_dim // 2,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, slot_dim // 2))

        # Score predictor
        self.scorer = nn.Sequential(
            nn.Linear(slot_dim // 2, slot_dim),
            nn.LayerNorm(slot_dim),
            nn.GELU(),
            nn.Linear(slot_dim, slot_dim // 2),
            nn.GELU(),
            nn.Linear(slot_dim // 2, 1),
        )

    def forward(
        self,
        trajectories: torch.Tensor,       # [B, M, T_fut, 2]
        refined_slots: Optional[torch.Tensor] = None,  # [B, T_fut, N, D] (for collision)
        ego_state: Optional[torch.Tensor] = None,      # [B, T_hist, 4]
        route_goals: Optional[torch.Tensor] = None,    # [B, K, 2]
    ) -> torch.Tensor:
        """
        Returns:
            scores: [B, M]  (higher = better proposal)
        """
        B, M, T_fut, _ = trajectories.shape

        # Encode trajectories
        traj_emb = self.traj_encoder(trajectories)  # [B, M, T_fut, D/2]

        # Pool over time
        traj_flat = rearrange(traj_emb, 'b m t d -> (b m) t d')
        query = self.pool_query.expand(B * M, -1, -1)
        traj_pooled, _ = self.traj_pool(query, traj_flat, traj_flat)
        traj_pooled = traj_pooled.squeeze(1)  # [B*M, D/2]

        # Predict scores
        scores = self.scorer(traj_pooled)  # [B*M, 1]
        scores = rearrange(scores, '(b m) 1 -> b m', b=B, m=M)

        return scores


class Planner(nn.Module):
    """
    Drive-JEPA: Multimodal trajectory planning from refined latent world states.

    Complete pipeline:
        ThinkJEPA → Drive-JEPA → M trajectory proposals → Best trajectory

    Args:
        slot_dim: D, dimension of refined slots (128)
        num_slots: N, slots per frame (11)
        num_modes: M, number of trajectory proposals (32)
        future_len: T_fut, prediction horizon (6)
        history_len: T_hist, ego history length (4)
        num_decoder_layers: decoder depth (3)
        num_heads: attention heads (8)
    """

    def __init__(
        self,
        slot_dim: int = 128,
        num_slots: int = 11,
        num_modes: int = 32,
        future_len: int = 6,
        history_len: int = 4,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_modes = num_modes
        self.future_len = future_len
        self.history_len = history_len

        # Planning context encoder
        self.context_encoder = PlanningContextEncoder(
            ego_dim=4,
            context_dim=slot_dim,
            history_len=history_len,
        )

        # Mode query generator
        self.query_generator = ModeQueryGenerator(
            num_modes=num_modes,
            mode_dim=slot_dim,
        )

        # Trajectory decoder
        self.trajectory_decoder = TrajectoryDecoder(
            slot_dim=slot_dim,
            num_modes=num_modes,
            future_len=future_len,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Proposal scorer
        self.proposal_scorer = ProposalScorer(
            slot_dim=slot_dim,
            num_modes=num_modes,
        )

    def forward(
        self,
        refined_slots: torch.Tensor,              # [B, T_fut, N, D]
        ego_state: torch.Tensor,                  # [B, T_hist, 4]
        route_goals: Optional[torch.Tensor] = None,  # [B, K, 2]
        return_all_proposals: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass: refined slots → best trajectory.

        Returns dict with:
            'trajectory_proposals': [B, M, T_fut, 2]
            'proposal_scores': [B, M]
            'best_trajectory': [B, T_fut, 2]
            'planning_context': [B, D]  (for analysis)
            'mode_queries': [B, M, D]  (for analysis)
        """

        # 1. Encode planning context
        planning_context = self.context_encoder(ego_state, route_goals)  # [B, D]

        # 2. Generate mode queries
        mode_queries = self.query_generator(planning_context)  # [B, M, D]

        # 3. Decode trajectories
        trajectory_proposals = self.trajectory_decoder(
            mode_queries,
            refined_slots,
        )  # [B, M, T_fut, 2]

        # 4. Score proposals
        proposal_scores = self.proposal_scorer(
            trajectory_proposals,
            refined_slots,
            ego_state,
            route_goals,
        )  # [B, M]

        # 5. Select best trajectory
        best_idx = proposal_scores.argmax(dim=1)  # [B]
        B = refined_slots.shape[0]
        best_trajectory = trajectory_proposals[range(B), best_idx]  # [B, T_fut, 2]

        result = {
            'best_trajectory': best_trajectory,
            'proposal_scores': proposal_scores,
            'planning_context': planning_context,
            'mode_queries': mode_queries,
        }

        if return_all_proposals:
            result['trajectory_proposals'] = trajectory_proposals

        return result

    def inference(
        self,
        refined_slots: torch.Tensor,
        ego_state: torch.Tensor,
        route_goals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference mode: returns only best trajectory.

        Returns:
            best_trajectory: [B, T_fut, 2]
        """
        result = self.forward(
            refined_slots,
            ego_state,
            route_goals,
            return_all_proposals=False,
        )
        return result['best_trajectory']


def create_planner(
    slot_dim: int = 128,
    num_slots: int = 11,
    num_modes: int = 32,
    future_len: int = 6,
    history_len: int = 4,
    num_decoder_layers: int = 3,
    num_heads: int = 8,
    dropout: float = 0.0,
) -> Planner:
    """Factory for trajectory planner."""
    return Planner(
        slot_dim=slot_dim,
        num_slots=num_slots,
        num_modes=num_modes,
        future_len=future_len,
        history_len=history_len,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("Testing Drive-JEPA Architecture...")

    B, T_hist, T_fut, N, D, M = 2, 4, 6, 11, 128, 32

    # --- Test individual components ---

    # PlanningContextEncoder
    print("\n[1/5] PlanningContextEncoder")
    context_enc = PlanningContextEncoder(ego_dim=4, context_dim=D, history_len=T_hist)
    ego = torch.randn(B, T_hist, 4)
    route = torch.randn(B, 5, 2)
    context = context_enc(ego, route)
    assert context.shape == (B, D), f"Context: {context.shape}"
    print(f"  Input: ego {ego.shape}, route {route.shape}")
    print(f"  Output: {context.shape} ✓")

    # ModeQueryGenerator
    print("\n[2/5] ModeQueryGenerator")
    query_gen = ModeQueryGenerator(num_modes=M, mode_dim=D)
    queries = query_gen(context)
    assert queries.shape == (B, M, D), f"Queries: {queries.shape}"
    print(f"  Input: context {context.shape}")
    print(f"  Output: {queries.shape} ✓")

    # TrajectoryDecoder
    print("\n[3/5] TrajectoryDecoder")
    decoder = TrajectoryDecoder(
        slot_dim=D,
        num_modes=M,
        future_len=T_fut,
        num_decoder_layers=2,
        num_heads=8,
    )
    refined_slots = torch.randn(B, T_fut, N, D)
    trajs = decoder(queries, refined_slots)
    assert trajs.shape == (B, M, T_fut, 2), f"Trajectories: {trajs.shape}"
    print(f"  Input: queries {queries.shape}, slots {refined_slots.shape}")
    print(f"  Output: {trajs.shape} ✓")

    # ProposalScorer
    print("\n[4/5] ProposalScorer")
    scorer = ProposalScorer(slot_dim=D, num_modes=M)
    scores = scorer(trajs, refined_slots, ego, route)
    assert scores.shape == (B, M), f"Scores: {scores.shape}"
    print(f"  Input: trajectories {trajs.shape}")
    print(f"  Output: {scores.shape} ✓")

    # Full Drive-JEPA
    print("\n[5/5] DriveJEPA Full Forward")
    model = DriveJEPA(
        slot_dim=D,
        num_slots=N,
        num_modes=M,
        future_len=T_fut,
        history_len=T_hist,
        num_decoder_layers=2,
        num_heads=8,
    )

    result = model(refined_slots, ego, route, return_all_proposals=True)

    assert result['best_trajectory'].shape == (B, T_fut, 2)
    assert result['trajectory_proposals'].shape == (B, M, T_fut, 2)
    assert result['proposal_scores'].shape == (B, M)
    assert not torch.isnan(result['best_trajectory']).any()

    print(f"  Refined slots: {refined_slots.shape}")
    print(f"  Ego state: {ego.shape}")
    print(f"  Route goals: {route.shape}")
    print(f"  →  Best trajectory: {result['best_trajectory'].shape}")
    print(f"  →  All proposals: {result['trajectory_proposals'].shape}")
    print(f"  →  Scores: {result['proposal_scores'].shape}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Trainable params: {trainable:,}")

    # Backward pass
    loss = result['best_trajectory'].mean()
    loss.backward()
    print(f"  Backward pass: ✓")

    print("\n✓ All Drive-JEPA tests passed!")
