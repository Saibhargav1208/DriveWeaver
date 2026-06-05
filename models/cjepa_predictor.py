"""
C-JEPA Predictor

Based on the original C-JEPA paper architecture:
- Anchor-based approach (t=0 always visible)
- ID Projector: Projects t=0 into query instruction
- Non-causal full attention transformer
- Object-level masking

Reference: /research/cjepa/src/cjepa_predictor.py
Paper: https://arxiv.org/abs/2602.11389
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
from omegaconf import DictConfig


class NonCausalTransformer(nn.Module):
    """
    Standard Transformer Encoder with Non-Causal (Full) Attention.
    Every token attends to every other token.

    Optionally accepts per-layer VLM guidance (FiLM / cross-attention / AdaLN)
    injected before each self-attention block.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        guidance_mode: Optional[str] = None,
        guidance_dim: Optional[int] = None,
        guidance_hidden: int = 512,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depth = depth
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True
                ),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

        # --- VLM Guidance injection (optional) ---
        self.guidance_mode = guidance_mode
        if guidance_mode and guidance_dim:
            # Separate projectors for old (input) and new (reasoning)
            self.guidance_proj_old = nn.Linear(guidance_dim, dim, bias=False)
            self.guidance_proj_new = nn.Linear(guidance_dim, dim, bias=False)

            # Initialize both projectors to near-zero for smooth warmup and stability
            nn.init.normal_(self.guidance_proj_old.weight, std=0.01)
            nn.init.normal_(self.guidance_proj_new.weight, std=0.01)

            if guidance_mode in ("film", "adaln"):
                self.guidance_fusion_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(dim * 4),
                        nn.Linear(dim * 4, guidance_hidden, bias=True),
                        nn.GELU(),
                        nn.Linear(guidance_hidden, 2 * dim, bias=True),
                    )
                    for _ in range(depth)
                ])
                # Initialize final layer of fusion MLPs to near-zero for stability
                for mlp in self.guidance_fusion_mlps:
                    nn.init.normal_(mlp[-1].weight, std=0.01)
                    if mlp[-1].bias is not None:
                        nn.init.zeros_(mlp[-1].bias)
                if guidance_mode == "adaln":
                    self.guidance_prenorms = nn.ModuleList([
                        nn.LayerNorm(dim) for _ in range(depth)
                    ])
            elif guidance_mode == "crossattn":
                self.guidance_query_norms = nn.ModuleList([
                    nn.LayerNorm(dim) for _ in range(depth)
                ])
                self.guidance_cross_attn = nn.ModuleList([
                    nn.MultiheadAttention(
                        embed_dim=dim,
                        num_heads=heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(depth)
                ])

            # Match ThinkJEPA: start from the unguided predictor and let training
            # learn how much VLM guidance each layer should use.
            self.guidance_layer_scale = nn.Parameter(torch.zeros(depth, 1, 1))

    def _inject_guidance(
        self,
        layer_idx: int,
        x: torch.Tensor,
        guidance_tokens: Optional[torch.Tensor],
        guidance_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inject VLM guidance into the stream at a given layer.

        Args:
            layer_idx: Which transformer layer
            x: [B, SeqLen, D] input sequence
            guidance_tokens: [B, S_vlm, D] VLM features (may be padded)
            guidance_mask: [B, S_vlm] bool mask (True=valid, False=padding)
        """
        if self.guidance_mode is None or guidance_tokens is None:
            return x

        gate = torch.tanh(self.guidance_layer_scale[layer_idx])

        if self.guidance_mode == "crossattn":
            q = self.guidance_query_norms[layer_idx](x)
            if isinstance(guidance_tokens, tuple):
                memory_parts = [tokens for tokens in guidance_tokens if tokens is not None]
                guidance_tokens = torch.cat(memory_parts, dim=1) if memory_parts else None
            if guidance_tokens is None:
                return x
            # Use mask as key_padding_mask (True = ignore)
            key_padding_mask = ~guidance_mask if guidance_mask is not None else None
            attn_out, _ = self.guidance_cross_attn[layer_idx](
                q, guidance_tokens, guidance_tokens,
                key_padding_mask=key_padding_mask
            )
            return x + attn_out * gate
        else:
            # FiLM / AdaLN: ThinkJEPA-style dual-path summary.
            # Official fusion signature: [old, new, |old-new|, old*new].
            if isinstance(guidance_tokens, tuple):
                old_tokens, new_tokens = guidance_tokens
                old_summary = old_tokens.mean(dim=1) if old_tokens is not None else None
                new_summary = new_tokens.mean(dim=1) if new_tokens is not None else None
                if old_summary is None and new_summary is None:
                    return x
                if old_summary is None:
                    old_summary = torch.zeros_like(new_summary)
                if new_summary is None:
                    new_summary = torch.zeros_like(old_summary)
            else:
                if guidance_mask is not None:
                    # Masked average: sum valid tokens / count valid tokens
                    mask_expanded = guidance_mask.unsqueeze(-1).float()  # [B, S_vlm, 1]
                    masked_tokens = guidance_tokens * mask_expanded
                    old_summary = masked_tokens.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    old_summary = guidance_tokens.mean(dim=1)  # [B, dim]
                new_summary = torch.zeros_like(old_summary)

            sig = torch.cat([
                old_summary,
                new_summary,
                (old_summary - new_summary).abs(),
                old_summary * new_summary,
            ], dim=-1)  # [B, 4*dim]
            scale_shift = self.guidance_fusion_mlps[layer_idx](sig)  # [B, 2*dim]
            scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, dim]

            # Clamp scale/shift to prevent NaN explosion
            scale = torch.clamp(scale, min=-10.0, max=10.0)
            shift = torch.clamp(shift, min=-10.0, max=10.0)

            # Apply gate and reshape
            scale = (scale * gate).unsqueeze(1)  # [B, 1, dim]
            shift = (shift * gate).unsqueeze(1)

            if self.guidance_mode == "adaln":
                x = self.guidance_prenorms[layer_idx](x)

            # Apply FiLM with additional safety check
            output = x * (1.0 + scale) + shift

            # Safety: check for NaN/Inf and fall back to input if detected
            if torch.isnan(output).any() or torch.isinf(output).any():
                return x  # Fallback to unmodified input
            return output

    def forward(
        self,
        x: torch.Tensor,
        guidance_tokens_per_layer: Optional[List[torch.Tensor]] = None,
        guidance_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, SeqLen, D] input sequence
            guidance_tokens_per_layer: List of [B, S_vlm, D] per layer (one per transformer layer)
            guidance_mask: [B, S_vlm] bool mask (True=valid, False=padding)

        Returns:
            [B, SeqLen, D] encoded sequence
        """
        if guidance_tokens_per_layer is None:
            guidance_tokens_per_layer = [None] * self.depth

        for layer_idx, (attn, ff) in enumerate(self.layers):
            # Get guidance for THIS specific layer
            guidance_tokens = guidance_tokens_per_layer[layer_idx]

            # Inject VLM guidance before self-attention
            x = self._inject_guidance(layer_idx, x, guidance_tokens, guidance_mask)

            # Self-attention with no mask (full attention)
            attn_out, _ = attn(x, x, x)
            x = x + attn_out

            # Feed-forward with residual
            x = x + ff(x)

        return self.norm(x)


class CJEPAPredictor(nn.Module):
    """
    C-JEPA: Causal-JEPA World Model

    Architecture:
    1. Anchor-based: t=0 always visible as identity anchor
    2. ID Projector: Projects t=0 into query instruction
    3. Query composition: MaskToken + TimePE + AnchorQuery
    4. Object-level masking: Masks specific slots across time
    5. Non-causal: Full attention across all tokens

    Input:  [B, T_hist, N_slots, D_slot]
    Output: [B, T_total, N_slots, D_slot] (includes history + future)
    """

    def __init__(
        self,
        num_slots: int,
        slot_dim: int = 128,
        history_frames: int = 4,
        pred_frames: int = 6,
        num_masked_slots: int = 2,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        seed: int = 42,
        guidance_mode: Optional[str] = None,
        guidance_dim: Optional[int] = None,
        guidance_hidden: int = 512,
    ):
        """
        Args:
            num_slots: Total number of slots per frame (N)
            slot_dim: Slot embedding dimension (D)
            history_frames: Number of input history frames (T_hist)
            pred_frames: Number of future frames to predict (T_future)
            num_masked_slots: Number of slots to mask (object-level masking)
            depth: Number of transformer layers
            heads: Number of attention heads
            dim_head: Dimension per attention head
            mlp_dim: MLP hidden dimension
            dropout: Dropout probability
            seed: Random seed for masking
            guidance_mode: VLM guidance mode — "film", "crossattn", "adaln", or None
            guidance_dim: Raw VLM hidden dimension (e.g. 3584 for Qwen3-VL-2B)
            guidance_hidden: Hidden dim for guidance fusion MLP
        """
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_frames = history_frames
        self.pred_frames = pred_frames
        self.total_frames = history_frames + pred_frames
        self.num_masked_slots = num_masked_slots
        self.seed = seed

        # 1. Learnable Mask Token (Query Base)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 2. Time Positional Embedding
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, self.total_frames, 1, slot_dim)
        )

        # 3. ID Projector (The "Anchor" mechanism)
        self.id_projector = nn.Linear(slot_dim, slot_dim)

        # 4. Backbone (Non-Causal Transformer with optional VLM guidance)
        self.transformer = NonCausalTransformer(
            dim=slot_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            guidance_mode=guidance_mode,
            guidance_dim=guidance_dim,
            guidance_hidden=guidance_hidden,
        )

        # 5. Output Head
        self.to_out = nn.Linear(slot_dim, slot_dim)

    def get_mask_indices(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects N slots to be masked per sample.

        Returns:
            is_slot_masked: [N_slots] boolean mask (True = masked)
            masked_indices: [num_masked_slots] indices of masked slots
        """
        rng = np.random.RandomState(self.seed)

        # Select N indices out of num_slots
        masked_indices = rng.choice(
            self.num_slots,
            self.num_masked_slots,
            replace=False
        )

        # Create boolean mask (True = Masked/Target, False = Visible/Context)
        is_slot_masked = torch.zeros(
            self.num_slots,
            dtype=torch.bool,
            device=device
        )
        is_slot_masked[masked_indices] = True

        return is_slot_masked, torch.from_numpy(masked_indices).to(device)

    def prepare_input(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs the input sequence for the Transformer.

        Logic:
        - t=0: ALWAYS Visible (Identity Anchor) for ALL slots
        - Masked Slots (Target): Visible at t=0, Masked at t=1 ~ T_total
        - Unmasked Slots (Context): Visible at t=0 ~ T_hist, Masked at Future

        Args:
            x: [B, T_hist, S, D] - Ground Truth History

        Returns:
            full_input: [B, T_total, S, D]
            masked_indices: indices of slots that were masked
        """
        B, T_hist, S, D = x.shape
        T_total = self.total_frames
        device = x.device

        # 1. Get Mask Indices
        if self.num_masked_slots > 0:
            is_slot_masked, masked_indices = self.get_mask_indices(B, device)
        else:
            masked_indices = torch.tensor([], dtype=torch.long, device=device)
            is_slot_masked = torch.zeros(S, dtype=torch.bool, device=device)

        # 2. Prepare Base Components
        # Anchors: First frame of all slots [B, S, D]
        anchors = x[:, 0, :, :]

        # Project anchors to create Identity Queries [B, S, D]
        anchor_queries = self.id_projector(anchors)

        # 3. Construct the "Query Grid" (Default for everything)
        # Shape: [B, T_total, S, D]
        # Base = MaskToken + TimePE + AnchorQuery
        # This represents "Predict the state of [Anchor] at [Time]"

        # MaskToken: [1, 1, 1, D] -> [B, T_total, S, D]
        tokens_grid = self.mask_token.expand(B, T_total, S, D)

        # TimePE: [1, T_total, 1, D] -> [B, T_total, S, D]
        pos_grid = self.time_pos_embed.expand(B, T_total, S, D)

        # AnchorQueries: [B, 1, S, D] -> [B, T_total, S, D]
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)

        # Full Query Input
        query_input = tokens_grid + pos_grid + anchor_grid

        # 4. Construct the "Real Data Grid"
        # Start by cloning query input, then overwrite visible parts with real data
        final_input = query_input.clone()

        # --- Overwrite Logic ---

        # (A) ALWAYS overwrite t=0 with Real Data + TimePE(0) for ALL slots
        # This ensures the Anchor is physically present in the input
        final_input[:, 0, :, :] = x[:, 0, :, :] + self.time_pos_embed[:, 0, :, :]

        # (B) For UNMASKED (Context) slots, overwrite history (t=1 to T_hist-1)
        if self.num_masked_slots > 0:
            unmasked_indices = torch.where(~is_slot_masked)[0]
        else:
            unmasked_indices = torch.arange(S, device=device)

        if len(unmasked_indices) > 0 and T_hist > 1:
            # Extract real history for unmasked slots
            real_history = x[:, 1:, unmasked_indices, :]  # [B, T_hist-1, len(unmasked), D]

            # Add corresponding TimePE
            history_pos = self.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist - 1, S, D)
            history_pos_unmasked = history_pos[:, :, unmasked_indices, :]

            # Overwrite in final_input
            final_input[:, 1:T_hist, unmasked_indices, :] = (
                real_history + history_pos_unmasked
            )

        # Note:
        # - Masked slots at t >= 1 remain as "Query Input"
        # - Unmasked slots at t >= T_hist (Future) remain as "Query Input"

        return final_input, masked_indices

    def forward(
        self,
        x: torch.Tensor,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with training-time masking.

        Args:
            x: [B, T_hist, S, D] history slots
            vlm_guidance: optional dict with 'old' and 'new' keys
                'old': [B, num_vlm_layers, S_old, vlm_dim]
                'new': [B, num_vlm_layers, S_new, vlm_dim] or None

        Returns:
            out: [B, T_total, S, D] predicted slots (history + future)
            masked_indices: indices of masked slots
        """
        B, T_hist, S, D = x.shape

        # 1. Prepare Input (Mix of Real Data and Queries)
        x_input, masked_indices = self.prepare_input(x)  # [B, T_total, S, D]

        # 2. Flatten for Transformer: [B, T*S, D]
        x_flat = rearrange(x_input, 'b t s d -> b (t s) d')

        # 3. Prepare per-layer VLM guidance
        guidance_tokens_list, guidance_mask, num_vlm_layers = self._prepare_guidance(vlm_guidance)

        # 4. Map VLM layers to C-JEPA layers
        if guidance_tokens_list is not None and num_vlm_layers > 0:
            guidance_per_cjepa_layer = self._map_vlm_to_cjepa_layers(
                guidance_tokens_list, num_vlm_layers, self.transformer.depth
            )
        else:
            guidance_per_cjepa_layer = None

        # 5. Non-Causal Full Attention with per-layer guidance injection
        out_flat = self.transformer(
            x_flat,
            guidance_tokens_per_layer=guidance_per_cjepa_layer,
            guidance_mask=guidance_mask
        )

        # 5. Unflatten
        out = rearrange(out_flat, 'b (t s) d -> b t s d', t=self.total_frames, s=S)

        # 6. Output Projection
        out = self.to_out(out)

        return out, masked_indices

    def _prepare_guidance(
        self, vlm_guidance: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor], int]:
        """
        Prepare dual-path VLM guidance for per-layer injection.

        Args:
            vlm_guidance: dict with 'old' and 'new' keys
                'old': [B, num_layers, S_old, vlm_dim]
                'new': [B, num_layers, S_new, vlm_dim] or None

        Returns:
            guidance_tokens_list: List of [B, S_combined, slot_dim] per VLM layer
            guidance_mask: None (features are not padded)
            num_layers: Number of VLM layers
        """
        if vlm_guidance is None or self.transformer.guidance_mode is None:
            return None, None, 0

        vlm_old = vlm_guidance.get('old')  # [B, num_layers, S_old, vlm_dim] or [B, num_layers, 1, S_old, vlm_dim]
        vlm_new = vlm_guidance.get('new')  # [B, num_layers, S_new, vlm_dim] or None

        # Backward compatibility for older callers/tests that pass a single
        # tensor as {'vlm_features': [B, S, D] or [B, L, S, D]}.
        if vlm_old is None and 'vlm_features' in vlm_guidance:
            vlm_old = vlm_guidance['vlm_features']
            if vlm_old.dim() == 3:
                vlm_old = vlm_old.unsqueeze(1)
            vlm_new = None

        if vlm_old is None:
            return None, None, 0

        # Handle extra dimension from dataloader: [B, num_layers, 1, S_old, vlm_dim] → [B, num_layers, S_old, vlm_dim]
        if vlm_old.dim() == 5 and vlm_old.shape[2] == 1:
            vlm_old = vlm_old.squeeze(2)
        if vlm_new is not None and vlm_new.dim() == 5 and vlm_new.shape[2] == 1:
            vlm_new = vlm_new.squeeze(2)

        B, num_layers, S_old, vlm_dim = vlm_old.shape

        # Project per-layer VLM streams. Keep old/new separate for FiLM/AdaLN
        # so fusion can use the ThinkJEPA signature [old, new, |old-new|, old*new].
        guidance_tokens_list = []
        for layer_idx in range(num_layers):
            old_layer = vlm_old[:, layer_idx, :, :]  # [B, S_old, vlm_dim]
            proj_old = self.transformer.guidance_proj_old(old_layer)  # [B, S_old, slot_dim]

            proj_new = None
            if vlm_new is not None:
                new_layer = vlm_new[:, layer_idx, :, :]  # [B, S_new, vlm_dim]
                proj_new = self.transformer.guidance_proj_new(new_layer)  # [B, S_new, slot_dim]

            if self.transformer.guidance_mode == "crossattn":
                parts = [proj_old] + ([proj_new] if proj_new is not None else [])
                guidance_tokens_list.append(torch.cat(parts, dim=1))
            else:
                guidance_tokens_list.append((proj_old, proj_new))

        return guidance_tokens_list, None, num_layers

    def _map_vlm_to_cjepa_layers(
        self,
        vlm_guidance_list: List[torch.Tensor],
        num_vlm_layers: int,
        num_cjepa_layers: int
    ) -> List[Optional[Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]]]:
        """
        Map VLM layers to C-JEPA layers.

        Example: VLM has 4 layers [6, 12, 18, 24], C-JEPA has 6 layers.
        Strategy: map available layers by index and leave remaining C-JEPA
        layers unguided. This matches ThinkJEPA's no-repeat behavior.

        Args:
            vlm_guidance_list: List of [B, S, D] tensors (one per VLM layer)
            num_vlm_layers: Number of VLM layers (e.g., 4)
            num_cjepa_layers: Number of C-JEPA layers (e.g., 6)

        Returns:
            List with one entry per C-JEPA layer. Entries are guidance tokens
            for guided layers and None for unguided layers.
        """
        if num_vlm_layers >= num_cjepa_layers:
            # More VLM layers than C-JEPA: use first N.
            return vlm_guidance_list[:num_cjepa_layers]

        # ThinkJEPA-style behavior: if fewer VLM layers are available, do not
        # repeat the final VLM layer. Later C-JEPA layers run unguided.
        result = vlm_guidance_list.copy()
        result.extend([None] * (num_cjepa_layers - num_vlm_layers))
        return result

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        vlm_guidance: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Inference function without masking.

        Args:
            x: [B, T_hist, S, D] - Fully visible history
            vlm_guidance: optional dict with 'vlm_features' [B, S_vlm, vlm_dim]

        Returns:
            future_prediction: [B, T_pred, S, D]

        Note:
            History length (T_hist) must match training history_frames.
            This is because positional encoding is fixed at initialization.
        """
        B, T_hist, S, D = x.shape
        T_pred = self.pred_frames
        T_total = T_hist + T_pred
        device = x.device

        # Validate history length matches training
        if T_hist != self.history_frames:
            raise ValueError(
                f"Inference requires history_length={self.history_frames}, "
                f"but got T_hist={T_hist}. Positional encoding is fixed at training time."
            )

        # Use last T_total timesteps for positional encoding
        inf_time_pos_embed = self.time_pos_embed[:, -T_total:, :, :]

        # 1. Anchor Query (t=0)
        anchors = x[:, 0, :, :]
        anchor_queries = self.id_projector(anchors)  # [B, S, D]

        # 2. History Part (NO MASK)
        input_history = x + inf_time_pos_embed[:, :T_hist, :, :]

        # 3. Future Part (Query Tokens)
        tokens_grid = self.mask_token.expand(B, T_pred, S, D)
        pos_grid = inf_time_pos_embed[:, T_hist:T_total, :, :].expand(B, T_pred, S, D)
        anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_pred, S, D)
        input_future = tokens_grid + pos_grid + anchor_grid

        # 4. Concatenate history and future
        full_input = torch.cat([input_history, input_future], dim=1)

        # 5. Flatten and process with guidance
        x_flat = rearrange(full_input, 'b t s d -> b (t s) d')

        # Prepare per-layer VLM guidance
        guidance_tokens_list, guidance_mask, num_vlm_layers = self._prepare_guidance(vlm_guidance)

        # Map VLM layers to C-JEPA layers
        if guidance_tokens_list is not None and num_vlm_layers > 0:
            guidance_per_cjepa_layer = self._map_vlm_to_cjepa_layers(
                guidance_tokens_list, num_vlm_layers, self.transformer.depth
            )
        else:
            guidance_per_cjepa_layer = None

        out_flat = self.transformer(
            x_flat,
            guidance_tokens_per_layer=guidance_per_cjepa_layer,
            guidance_mask=guidance_mask
        )

        # 6. Unflatten and project
        out = rearrange(out_flat, 'b (t s) d -> b t s d', t=T_total, s=S)
        out = self.to_out(out)

        # Return only future predictions
        return out[:, T_hist:, :, :]

    def predict(self, history_slots: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference.

        Args:
            history_slots: [B, T_hist, N, D]

        Returns:
            future_slots: [B, T_future, N, D]
        """
        return self.inference(history_slots)

    def autoregressive_rollout(
        self,
        initial_history: torch.Tensor,
        num_steps: int
    ) -> torch.Tensor:
        """
        Autoregressive rollout: iteratively predict multiple steps.

        Args:
            initial_history: [B, T_hist, N, D] initial history
            num_steps: Number of future steps to predict

        Returns:
            rollout: [B, num_steps, N, D] rolled-out predictions
        """
        B, T_hist, N, D = initial_history.shape

        rollout = []
        history = initial_history.clone()

        for step in range(num_steps):
            # Predict next future_length frames
            predicted = self.inference(history)  # [B, T_pred, N, D]

            # Take first predicted frame
            next_frame = predicted[:, 0:1, :, :]  # [B, 1, N, D]
            rollout.append(next_frame)

            # Update history: slide window forward
            history = torch.cat([history[:, 1:, :, :], next_frame], dim=1)

        # Concatenate all predictions
        rollout = torch.cat(rollout, dim=1)  # [B, num_steps, N, D]

        return rollout


def create_cjepa_from_config(
    cfg: DictConfig,
    guidance_mode: Optional[str] = None,
    guidance_dim: Optional[int] = None,
    guidance_hidden: int = 512,
) -> CJEPAPredictor:
    """
    Create C-JEPA model from Hydra config.

    Args:
        cfg: Hydra configuration
        guidance_mode: VLM guidance injection mode (film/crossattn/adaln/None)
        guidance_dim: Raw VLM hidden dimension (e.g. 3584 for Qwen3-VL-2B)
        guidance_hidden: Hidden dim for guidance fusion MLP

    Returns:
        CJEPAPredictor instance
    """
    depth = cfg.model.get('depth', cfg.model.encoder.num_layers)
    dim_head = cfg.model.get('dim_head', cfg.model.slot_dim // cfg.model.num_heads)
    mlp_dim = cfg.model.get('mlp_dim', cfg.model.encoder.feedforward_dim)

    model = CJEPAPredictor(
        num_slots=cfg.model.num_slots,
        slot_dim=cfg.model.slot_dim,
        history_frames=cfg.data.temporal.history_length,
        pred_frames=cfg.data.temporal.future_length,
        num_masked_slots=cfg.masking.object_mask.min_masked_objects,
        depth=depth,
        heads=cfg.model.num_heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        dropout=cfg.model.dropout,
        seed=cfg.system.seed,
        guidance_mode=guidance_mode,
        guidance_dim=guidance_dim,
        guidance_hidden=guidance_hidden,
    )

    return model


if __name__ == "__main__":
    # Test C-JEPA architecture
    print("Testing C-JEPA Architecture...")

    B, T_hist, T_fut, N, D = 4, 4, 6, 11, 128

    # Create model
    model = CJEPAPredictor(
        num_slots=N,
        slot_dim=D,
        history_frames=T_hist,
        pred_frames=T_fut,
        num_masked_slots=2,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1
    )

    print(f"\nModel created:")
    print(f"  Input: [B, {T_hist}, {N}, {D}]")
    print(f"  Output: [B, {T_hist + T_fut}, {N}, {D}]")

    # Test forward pass (training mode)
    history = torch.randn(B, T_hist, N, D)

    model.train()
    output, masked_indices = model(history)

    print(f"\nTraining mode:")
    print(f"  Output: {output.shape}")
    print(f"  Masked slots: {masked_indices}")

    # Test inference
    model.eval()
    future_pred = model.inference(history)
    print(f"\nInference mode:")
    print(f"  Future prediction: {future_pred.shape}")

    # Test autoregressive rollout
    rollout = model.autoregressive_rollout(history, num_steps=12)
    print(f"\nAutoregressive rollout:")
    print(f"  Input: {history.shape}")
    print(f"  Rollout: {rollout.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")

    # Check for NaN
    assert not torch.isnan(output).any(), "NaN in predictions!"
    print(f"\n✓ No NaN values")

    print("\n✓ All tests passed!")
