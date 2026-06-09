import pytest
import torch

from videosaur.modules import groupers


@pytest.mark.parametrize("use_mlp", [False, True])
def test_slot_attention(use_mlp):
    inp_dim, slot_dim, n_patches, n_slots = 5, 8, 6, 3
    slot_attention = groupers.SlotAttention(inp_dim, slot_dim, use_mlp=use_mlp)

    features = torch.randn(1, n_patches, inp_dim)
    slots = torch.randn(1, n_slots, slot_dim)

    with torch.no_grad():
        outp = slot_attention(slots, features)

    assert outp["slots"].shape == (1, n_slots, slot_dim)
    assert outp["masks"].shape == (1, n_slots, n_patches)


@pytest.mark.parametrize("n_iters", [1, 3])
def test_gatst(n_iters):
    inp_dim, slot_dim, n_patches, n_slots, n_heads = 16, 32, 10, 4, 4
    gatst = groupers.GATST(inp_dim=inp_dim, slot_dim=slot_dim, n_heads=n_heads, n_iters=n_iters)

    features = torch.randn(2, n_patches, inp_dim)
    slots = torch.randn(2, n_slots, slot_dim)

    with torch.no_grad():
        outp = gatst(slots, features)

    assert outp["slots"].shape == (2, n_slots, slot_dim), f"Unexpected slots shape: {outp['slots'].shape}"
    assert outp["masks"].shape == (2, n_slots, n_patches), f"Unexpected masks shape: {outp['masks'].shape}"
    assert torch.isfinite(outp["slots"]).all(), "GATST slots contain NaN/Inf"
    assert torch.isfinite(outp["masks"]).all(), "GATST masks contain NaN/Inf"
    # Masks use competitive softmax over slots (dim K): each patch sums to 1 over K.
    # n_patches=10 is not a perfect square so spatial smoothing is skipped here.
    mask_sums = outp["masks"].sum(dim=1)  # [B, N]
    assert torch.allclose(mask_sums, torch.ones_like(mask_sums), atol=1e-5), "GATST masks do not sum to 1 over slots"


def test_spatial_slot_attention():
    inp_dim, slot_dim, n_slots = 5, 8, 3
    # Use a square patch grid so smoothing is triggered (4x4 = 16 patches)
    n_patches = 16
    ssa = groupers.SpatialSlotAttention(inp_dim, slot_dim)

    features = torch.randn(1, n_patches, inp_dim)
    slots = torch.randn(1, n_slots, slot_dim)

    with torch.no_grad():
        outp = ssa(slots, features)

    assert outp["slots"].shape == (1, n_slots, slot_dim)
    assert outp["masks"].shape == (1, n_slots, n_patches)
    assert torch.isfinite(outp["masks"]).all()


def test_gatst_gradient_flow():
    inp_dim, slot_dim, n_patches, n_slots, n_heads = 16, 32, 10, 4, 4
    gatst = groupers.GATST(inp_dim=inp_dim, slot_dim=slot_dim, n_heads=n_heads, n_iters=3)

    features = torch.randn(1, n_patches, inp_dim, requires_grad=True)
    slots = torch.randn(1, n_slots, slot_dim, requires_grad=True)

    outp = gatst(slots, features)
    (outp["slots"].sum() + outp["masks"].sum()).backward()

    assert slots.grad is not None and torch.isfinite(slots.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()
