"""
Test script to verify dual-path VLM integration works correctly.

Tests:
1. Dataset loading with dual-path features
2. Batch collation
3. Model forward pass with per-layer guidance
4. Shape checking at each stage
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


def test_dataset_loading():
    """Test that dataset can load dual-path features."""
    print("=" * 70)
    print("Test 1: Dataset Loading (Dual-Path)")
    print("=" * 70)

    from datasets.slot_dataset import SlotDataset

    # Test with random features (no actual cache needed)
    dataset = SlotDataset(
        slots_path='/work/data/slots/nuscenes_slots_full.pkl',
        split='train',
        history_length=4,
        future_length=6,
        vlm_random_dim=3584,  # Random VLM features for testing
        vlm_random_tokens=256,
    )

    print(f"✓ Dataset created: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    print(f"✓ Sample keys: {sample.keys()}")

    vlm = sample.get('vlm_features')
    if vlm is not None:
        print(f"✓ VLM features type: {type(vlm)}")
        if isinstance(vlm, dict):
            print(f"  VLM keys: {vlm.keys()}")
            print(f"  vlm_old shape: {vlm['old'].shape}")  # [num_layers, S_old, D]
            print(f"  vlm_new shape: {vlm['new'].shape if vlm['new'] is not None else 'None'}")

            # Verify shapes
            assert vlm['old'].dim() == 3, f"Expected 3D tensor, got {vlm['old'].dim()}D"
            assert vlm['old'].shape[0] == 4, f"Expected 4 layers, got {vlm['old'].shape[0]}"
            assert vlm['old'].shape[1] == 256, f"Expected 256 tokens, got {vlm['old'].shape[1]}"
            assert vlm['old'].shape[2] == 3584, f"Expected 3584 dim, got {vlm['old'].shape[2]}"
            print("✓ vlm_old shape correct: [4, 256, 3584]")

            if vlm['new'] is not None:
                assert vlm['new'].shape[0] == 4, f"Expected 4 layers"
                assert vlm['new'].shape[1] == 16, f"Expected 16 tokens"
                assert vlm['new'].shape[2] == 3584, f"Expected 3584 dim"
                print("✓ vlm_new shape correct: [4, 16, 3584]")
        else:
            print(f"✗ Expected dict, got {type(vlm)}")
            return False
    else:
        print("✗ No VLM features in sample")
        return False

    print("\n✓ Test 1 PASSED\n")
    return True


def test_batch_collation():
    """Test that batching works correctly with dual-path features."""
    print("=" * 70)
    print("Test 2: Batch Collation")
    print("=" * 70)

    from datasets.slot_dataset import SlotDataset, collate_with_vlm

    dataset = SlotDataset(
        slots_path='/work/data/slots/nuscenes_slots_full.pkl',
        split='train',
        history_length=4,
        future_length=6,
        vlm_random_dim=3584,
        vlm_random_tokens=256,
    )

    # Create a batch
    batch = [dataset[i] for i in range(4)]
    print(f"✓ Created batch of {len(batch)} samples")

    # Collate
    collated = collate_with_vlm(batch)
    print(f"✓ Collated batch keys: {collated.keys()}")

    vlm = collated.get('vlm_features')
    if vlm is not None and isinstance(vlm, dict):
        print(f"✓ VLM features type: dict")
        print(f"  VLM keys: {vlm.keys()}")
        print(f"  vlm_old shape: {vlm['old'].shape}")  # [B, num_layers, S_old, D]
        print(f"  vlm_new shape: {vlm['new'].shape if vlm['new'] is not None else 'None'}")

        # Verify batch dimension
        assert vlm['old'].shape[0] == 4, f"Expected batch=4, got {vlm['old'].shape[0]}"
        assert vlm['old'].shape[1] == 4, f"Expected 4 layers"
        assert vlm['old'].shape[2] == 256, f"Expected 256 tokens"
        assert vlm['old'].shape[3] == 3584, f"Expected 3584 dim"
        print("✓ vlm_old batched correctly: [4, 4, 256, 3584]")

        if vlm['new'] is not None:
            assert vlm['new'].shape == (4, 4, 16, 3584), f"Wrong shape: {vlm['new'].shape}"
            print("✓ vlm_new batched correctly: [4, 4, 16, 3584]")
    else:
        print("✗ VLM features not properly collated")
        return False

    print("\n✓ Test 2 PASSED\n")
    return True


def test_model_forward():
    """Test that C-JEPA can process dual-path features."""
    print("=" * 70)
    print("Test 3: Model Forward Pass")
    print("=" * 70)

    from models.cjepa_predictor import CJEPAPredictor

    # Create model with VLM guidance
    model = CJEPAPredictor(
        num_slots=11,
        slot_dim=128,
        history_frames=4,
        pred_frames=6,
        num_masked_slots=3,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.0,
        guidance_mode="film",
        guidance_dim=3584,
        guidance_hidden=512,
    )

    print(f"✓ Model created")
    print(f"  C-JEPA depth: {model.transformer.depth} layers")
    print(f"  Guidance mode: {model.transformer.guidance_mode}")

    # Create dummy inputs
    B, T_hist, N, D = 2, 4, 11, 128
    history_slots = torch.randn(B, T_hist, N, D)

    # Create dual-path VLM guidance
    vlm_guidance = {
        'old': torch.randn(B, 4, 256, 3584),  # 4 VLM layers
        'new': torch.randn(B, 4, 16, 3584)
    }

    print(f"✓ Created dummy inputs:")
    print(f"  history_slots: {history_slots.shape}")
    print(f"  vlm_old: {vlm_guidance['old'].shape}")
    print(f"  vlm_new: {vlm_guidance['new'].shape}")

    # Forward pass
    try:
        out, masked_indices = model(history_slots, vlm_guidance=vlm_guidance)
        print(f"✓ Forward pass succeeded")
        print(f"  Output shape: {out.shape}")  # [B, T_total, N, D]

        expected_shape = (B, T_hist + 6, N, D)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        print(f"✓ Output shape correct: {out.shape}")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test inference
    try:
        future = model.inference(history_slots, vlm_guidance=vlm_guidance)
        print(f"✓ Inference succeeded")
        print(f"  Future shape: {future.shape}")  # [B, T_pred, N, D]

        assert future.shape == (B, 6, N, D), f"Wrong inference shape: {future.shape}"
        print(f"✓ Inference shape correct: {future.shape}")

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✓ Test 3 PASSED\n")
    return True


def test_layer_mapping():
    """Test VLM layer to C-JEPA layer mapping."""
    print("=" * 70)
    print("Test 4: Layer Mapping (VLM → C-JEPA)")
    print("=" * 70)

    from models.cjepa_predictor import CJEPAPredictor

    model = CJEPAPredictor(
        num_slots=11,
        slot_dim=128,
        history_frames=4,
        pred_frames=6,
        num_masked_slots=3,
        depth=6,  # 6 C-JEPA layers
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.0,
        guidance_mode="film",
        guidance_dim=3584,
    )

    # Create VLM guidance list (4 layers)
    vlm_guidance_list = [torch.randn(2, 272, 512) for _ in range(4)]

    # Map to 6 C-JEPA layers
    mapped = model._map_vlm_to_cjepa_layers(vlm_guidance_list, 4, 6)

    print(f"✓ VLM layers: 4")
    print(f"✓ C-JEPA layers: 6")
    print(f"✓ Mapped list length: {len(mapped)}")

    assert len(mapped) == 6, f"Expected 6 layers, got {len(mapped)}"

    # Check that layers 4-5 repeat layer 3
    assert torch.equal(mapped[4], mapped[3]), "Layer 4 should be repeat of layer 3"
    assert torch.equal(mapped[5], mapped[3]), "Layer 5 should be repeat of layer 3"

    print("✓ Layer mapping correct:")
    print("  C-JEPA 0 ← VLM 0")
    print("  C-JEPA 1 ← VLM 1")
    print("  C-JEPA 2 ← VLM 2")
    print("  C-JEPA 3 ← VLM 3")
    print("  C-JEPA 4 ← VLM 3 (repeated)")
    print("  C-JEPA 5 ← VLM 3 (repeated)")

    print("\n✓ Test 4 PASSED\n")
    return True


def main():
    print("\n" + "=" * 70)
    print("TESTING DUAL-PATH VLM INTEGRATION")
    print("=" * 70 + "\n")

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Batch Collation", test_batch_collation),
        ("Model Forward", test_model_forward),
        ("Layer Mapping", test_layer_mapping),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8s} {name}")

    all_passed = all(success for _, success in results)
    print("=" * 70)
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nDual-path VLM integration is working correctly!")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
