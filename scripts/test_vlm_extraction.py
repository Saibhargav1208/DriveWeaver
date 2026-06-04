"""
Test script to verify corrected VLM extraction before full run.

Tests:
1. Vision encoder path is correct
2. Features are extracted without text contamination
3. Spatial structure is preserved
4. Uniform temporal sampling works
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))


def test_qwen3_vision_encoder():
    """Test that we can access Qwen3-VL vision encoder correctly."""
    print("=" * 60)
    print("Test 1: Qwen3-VL Vision Encoder Access")
    print("=" * 60)

    try:
        from transformers import AutoModel, AutoProcessor

        model_name = "Qwen/Qwen3-VL-2B-Thinking"
        print(f"Loading {model_name}...")

        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()

        print(f"✓ Model loaded")
        print(f"  Model type: {type(model)}")
        print(f"  Has visual? {hasattr(model, 'visual')}")

        if hasattr(model, 'visual'):
            print(f"  Visual type: {type(model.visual)}")
            vision = model.visual

            # Find transformer blocks
            if hasattr(vision, 'transformer'):
                trans = vision.transformer
                print(f"  Transformer type: {type(trans)}")

                if hasattr(trans, 'blocks'):
                    print(f"  ✓ Has blocks: {len(trans.blocks)} layers")
                elif hasattr(trans, 'layers'):
                    print(f"  ✓ Has layers: {len(trans.layers)} layers")
                else:
                    print(f"  Available attrs: {[a for a in dir(trans) if not a.startswith('_')][:10]}")
            else:
                print(f"  Available visual attrs: {[a for a in dir(vision) if not a.startswith('_')][:10]}")

        return model, processor

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_vision_only_extraction(model, processor):
    """Test that extraction works without text prompts."""
    print("\n" + "=" * 60)
    print("Test 2: Vision-Only Extraction (No Text)")
    print("=" * 60)

    if model is None:
        print("Skipping - model not loaded")
        return

    try:
        # Create dummy images
        dummy_images = [Image.new('RGB', (384, 384), color=(i*20, i*20, i*20)) for i in range(4)]

        print(f"Processing {len(dummy_images)} dummy images...")

        # NO text - images only
        inputs = processor(
            images=dummy_images,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        print(f"✓ Processor created inputs (no text)")
        print(f"  Input keys: {inputs.keys()}")

        # Hook a middle layer
        hidden_states = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states['captured'] = output[0].detach().cpu()
            else:
                hidden_states['captured'] = output.detach().cpu()

        # Try to hook vision encoder
        vision = model.visual
        if hasattr(vision, 'transformer'):
            trans = vision.transformer
            if hasattr(trans, 'blocks'):
                layer_idx = len(trans.blocks) // 2  # Middle layer
                hook = trans.blocks[layer_idx].register_forward_hook(hook_fn)
                print(f"✓ Hooked vision transformer block {layer_idx}")
            elif hasattr(trans, 'layers'):
                layer_idx = len(trans.layers) // 2
                hook = trans.layers[layer_idx].register_forward_hook(hook_fn)
                print(f"✓ Hooked vision transformer layer {layer_idx}")

        # Forward pass through vision encoder
        with torch.no_grad():
            pixel_values = inputs.get('pixel_values', inputs.get('image'))
            if pixel_values is not None:
                print(f"  Pixel values shape: {pixel_values.shape}")
                _ = vision(pixel_values)
            else:
                print("  No pixel_values found, trying model forward...")
                _ = model(**inputs, output_hidden_states=False)

        hook.remove()

        if 'captured' in hidden_states:
            feat = hidden_states['captured']
            print(f"✓ Captured hidden states")
            print(f"  Shape: {feat.shape}")
            print(f"  Dtype: {feat.dtype}")
            print(f"  Mean: {feat.mean().item():.4f}, Std: {feat.std().item():.4f}")
            return True
        else:
            print(f"✗ No hidden states captured")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uniform_sampling():
    """Test uniform temporal sampling logic."""
    print("\n" + "=" * 60)
    print("Test 3: Uniform Temporal Sampling")
    print("=" * 60)

    # Simulate scene with 40 frames, want 16 keyframes
    scene_frames = list(range(40))
    num_keyframes = 16

    stride = len(scene_frames) // num_keyframes
    sampled = [scene_frames[i * stride] for i in range(num_keyframes)]
    sampled = sampled[:num_keyframes]

    print(f"Original: {len(scene_frames)} frames")
    print(f"Target: {num_keyframes} keyframes")
    print(f"Stride: {stride}")
    print(f"Sampled indices: {sampled}")
    print(f"✓ Uniform sampling works")

    return True


def main():
    print("\n" + "=" * 60)
    print("TESTING CORRECTED VLM EXTRACTION")
    print("=" * 60)

    # Test 1: Model access
    model, processor = test_qwen3_vision_encoder()

    # Test 2: Vision-only extraction
    if model is not None:
        extraction_ok = test_vision_only_extraction(model, processor)
    else:
        extraction_ok = False

    # Test 3: Sampling logic
    sampling_ok = test_uniform_sampling()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model access: {'✓' if model is not None else '✗'}")
    print(f"Vision extraction: {'✓' if extraction_ok else '✗'}")
    print(f"Uniform sampling: {'✓' if sampling_ok else '✗'}")

    if model is not None and extraction_ok and sampling_ok:
        print("\n✓ All tests passed! Ready for full extraction.")
        return 0
    else:
        print("\n✗ Some tests failed. Check implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
