import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from videosaur.modules import networks
from videosaur.utils import make_build_fn


@make_build_fn(__name__, "grouper")
def build(config, name: str):
    pass  # No special module building needed


def _sinkhorn_normalize(masks: torch.Tensor, n_iters: int = 3) -> torch.Tensor:
    """
    Doubly-stochastic normalization (Sinkhorn) on slot-patch assignment masks.

    masks: [B, K, N]
    Alternates column normalization (each patch sums to 1 over K) and row
    normalization (each slot sums to 1 over N), driving the matrix toward a
    near-permutation where each patch is claimed by one slot. Prevents the
    slot overlap that pure softmax-over-N cross-attention produces.
    """
    for _ in range(n_iters):
        masks = masks / (masks.sum(dim=1, keepdim=True) + 1e-8)  # each patch → 1 over K
        masks = masks / (masks.sum(dim=2, keepdim=True) + 1e-8)  # each slot → 1 over N
    return masks


def _spatial_smooth(masks: torch.Tensor, side: int, kernel: int = 5) -> torch.Tensor:
    """
    Smooth [B, K, N] slot masks spatially to fill stray mis-assigned patches.
    Reshapes to [B*K, 1, side, side], applies depthwise avg_pool2d, renormalises.
    Only called when N == side*side (square grid).
    """
    B, K, N = masks.shape
    m2d = masks.reshape(B * K, 1, side, side)
    m2d = F.avg_pool2d(m2d, kernel_size=kernel, stride=1, padding=kernel // 2)
    masks = m2d.reshape(B, K, N)
    return masks / (masks.sum(dim=-1, keepdim=True) + 1e-8)


def _build_2d_sincos_pos_embed(n_patches: int, dim: int) -> torch.Tensor:
    """
    Build a 2D sine-cosine positional embedding for a square patch grid.

    n_patches must be a perfect square (e.g. 196 = 14×14, 256 = 16×16).
    Returns [1, n_patches, dim] — added to patch features before cross-attention
    so that nearby patches get similar positional context, encouraging spatial
    contiguity in the resulting slot attention maps.
    """
    side = int(math.isqrt(n_patches))
    assert side * side == n_patches, "n_patches must be a perfect square"
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos embed"

    half = dim // 2  # each axis gets half the dims
    omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)  # [half/2]

    grid_h = torch.arange(side, dtype=torch.float32)
    grid_w = torch.arange(side, dtype=torch.float32)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")  # [side, side]

    emb_h = torch.outer(gh.flatten(), omega)  # [N, half/2]
    emb_w = torch.outer(gw.flatten(), omega)  # [N, half/2]

    emb = torch.cat([emb_h.sin(), emb_h.cos(), emb_w.sin(), emb_w.cos()], dim=-1)  # [N, dim]
    return emb.unsqueeze(0)  # [1, N, dim]


class GATST(nn.Module):
    """Drop-in replacement for SlotAttention using transformer cross-attention.

    Architecture:
      1. TransformerDecoder builds rich slot representations via stacked
         self-attention + cross-attention blocks (much richer than SA's GRU).
      2. A competitive readout head re-computes masks using SA-style softmax
         over the slot dimension (dim K), so patches compete for slots exactly
         as in SlotAttention — giving clean non-overlapping regions by construction.
      3. 2D sin-cos positional embeddings injected into patch memory so slots
         learn spatially contiguous regions.
      4. Optional spatial smoothing (avg_pool2d) fills stray mis-assigned patches.

    This combines the representation power of transformer decoders with the
    clean spatial partitioning of SlotAttention's competitive assignment.
    """

    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        n_heads: int = 4,
        n_iters: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        initial_residual_scale: float = 0.1,
        n_patches: int = 196,
        pos_scale: float = 3.0,
        smooth_masks: bool = True,
        smooth_kernel: int = 5,
    ):
        super().__init__()
        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.feature_proj = nn.Linear(inp_dim, slot_dim, bias=False)
        self.decoder = networks.TransformerDecoder(
            dim=slot_dim,
            n_blocks=n_iters,
            n_heads=n_heads,
            memory_dim=slot_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            initial_residual_scale=initial_residual_scale,
        )
        # Competitive readout: project slots → keys, patches → queries,
        # then softmax over slot dim so each patch is assigned to one slot.
        self.readout_k = nn.Linear(slot_dim, slot_dim, bias=False)  # slots  → keys
        self.readout_q = nn.Linear(slot_dim, slot_dim, bias=False)  # patches → queries
        self.readout_scale = slot_dim ** -0.5

        self.n_iters = n_iters
        self.pos_scale = pos_scale
        self.smooth_masks = smooth_masks
        self.smooth_kernel = smooth_kernel

        # 2D sin-cos positional embedding — fixed (not learned), added after feature_proj.
        # Stored as buffer so it moves with .to(device) automatically.
        pos_embed = _build_2d_sincos_pos_embed(n_patches, slot_dim)
        self.register_buffer("pos_embed", pos_embed)  # [1, N, slot_dim]

    def forward(
        self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None
    ):
        # slots:    [B, K, slot_dim]
        # features: [B, N_patches, inp_dim]
        features = self.norm_features(features)
        memory = self.feature_proj(features)  # [B, N, slot_dim]

        # Inject spatial position so cross-attention forms contiguous regions.
        n = memory.shape[1]
        memory = memory + self.pos_scale * self.pos_embed[:, :n, :]

        # Transformer decoder: builds rich slot representations
        slots, _ = self.decoder(self.norm_slots(slots), memory, return_weights=True)

        # Competitive readout head: SA-style softmax over K so patches don't overlap.
        # keys:   [B, K, slot_dim]  — one key per slot
        # queries:[B, N, slot_dim]  — one query per patch
        # dots:   [B, K, N]         — slot-patch affinity
        # softmax over dim 1 (K) → each patch sums to 1 over slots (competition)
        keys    = self.readout_k(slots)                                  # [B, K, D]
        queries = self.readout_q(memory)                                 # [B, N, D]
        dots    = torch.einsum("bkd,bnd->bkn", keys, queries) * self.readout_scale
        masks   = torch.softmax(dots, dim=1)                             # [B, K, N]

        # Optional spatial smoothing to fill stray mis-assigned patches
        if self.smooth_masks:
            side = int(masks.shape[2] ** 0.5)
            if side * side == masks.shape[2]:
                masks = _spatial_smooth(masks, side, kernel=self.smooth_kernel)

        return {"slots": slots, "masks": masks}


class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
        smooth_masks: bool = False,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = networks.MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5
        self.smooth_masks = smooth_masks

    def step(
        self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)

        masks = pre_norm_attn  # [B, K, N] — softmax over K
        if self.smooth_masks:
            side = int(masks.shape[2] ** 0.5)
            if side * side == masks.shape[2]:
                # Renorm over N first (SA softmax is over K, not N)
                masks = masks / (masks.sum(dim=-1, keepdim=True) + 1e-8)
                masks = _spatial_smooth(masks, side, kernel=5)

        return {"slots": slots, "masks": masks}


class SpatialSlotAttention(SlotAttention):
    """SlotAttention with a spatial-coherence prior on slot assignments.

    Identical to SlotAttention in every way except that after the final
    iteration the slot masks are smoothed with a depthwise average-pooling
    kernel over the 2D patch grid.  This enforces the inductive bias that
    spatially adjacent patches are more likely to belong to the same slot,
    producing cleaner contiguous regions without changing the core attention
    dynamics or any learnable parameters.

    Use this as a drop-in replacement for SlotAttention when the patch grid
    is square (e.g. 14×14 = 196 or 16×16 = 256 patches).  For non-square
    grids the smoothing step is silently skipped and behaviour is identical
    to plain SlotAttention.
    """

    def __init__(self, *args, smooth_kernel: int = 5, **kwargs):
        # Force smooth_masks=True; expose kernel size as a first-class param.
        kwargs["smooth_masks"] = True
        super().__init__(*args, **kwargs)
        # Override the fixed kernel=5 hard-coded in SlotAttention.forward
        # by storing it and patching the forward call via _smooth_kernel.
        self._smooth_kernel = smooth_kernel

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)

        if n_iters is None:
            n_iters = self.n_iters

        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(slots, keys, values)

        masks = pre_norm_attn  # [B, K, N]
        side = int(masks.shape[2] ** 0.5)
        if side * side == masks.shape[2]:
            masks = masks / (masks.sum(dim=-1, keepdim=True) + 1e-8)
            masks = _spatial_smooth(masks, side, kernel=self._smooth_kernel)

        return {"slots": slots, "masks": masks}
