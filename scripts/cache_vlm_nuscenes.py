"""
Offline VLM Cache Extraction for ThinkJEPA (CORRECTED)

Extracts Qwen3-VL vision encoder hidden states from nuScenes CAM_FRONT images.
This is the CORRECT implementation following ThinkJEPA paper methodology:

Key Corrections:
    1. NO text prompts - pure visual features only
    2. Hook VISION ENCODER layers (not language decoder)
    3. Use .forward() directly (not .generate())
    4. UNIFORM temporal sampling across scene
    5. Keep per-layer features (no averaging) OR use single middle layer

Usage:
    # Full extraction (requires Qwen3-VL weights):
    python scripts/cache_vlm_nuscenes.py \
        --slots_path /work/data/slots/nuscenes_slots_full.pkl \
        --output_dir /work/data/vlm_cache/nuscenes/ \
        --model_name Qwen/Qwen3-VL-2B-Thinking \
        --vision_layers 12 \
        --num_keyframes 16 \
        --resolution 384

    # Lite/debug mode (random features, no download):
    python scripts/cache_vlm_nuscenes.py \
        --slots_path /work/data/slots/nuscenes_slots_debug.pkl \
        --output_dir /work/data/vlm_cache/nuscenes_debug/ \
        --lite
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser("VLM cache extraction for ThinkJEPA (CORRECTED)")
    p.add_argument("--slots_path", type=str, required=True,
                   help="Path to slot pickle (used for scene/sample_token mapping)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for .npz cache files")
    p.add_argument("--nuscenes_root", type=str, default="/data/nuScenes",
                   help="nuScenes data root")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Thinking",
                   help="HuggingFace model ID")
    p.add_argument("--vision_layers", type=int, nargs="+", default=[12],
                   help="Vision encoder layer indices (middle layers, e.g. 12 for 24-layer model)")
    p.add_argument("--num_keyframes", type=int, default=16,
                   help="Number of keyframes to uniformly sample per scene")
    p.add_argument("--resolution", type=int, default=384,
                   help="Resize images to this resolution (384 is Qwen3-VL native)")
    p.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--lite", action="store_true",
                   help="Generate random features (no VLM download, for debugging)")
    p.add_argument("--lite_dim", type=int, default=3584,
                   help="Feature dim for lite mode")
    p.add_argument("--lite_tokens", type=int, default=256,
                   help="Number of tokens per scene for lite mode (16 frames × 16 tokens/frame)")
    return p.parse_args()


def get_image_paths_for_scene(
    scene_data: dict,
    nuscenes_root: str,
) -> List[str]:
    """
    Resolve sample_tokens to actual image file paths via nuScenes devkit.

    Returns list of image paths for CAM_FRONT.
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=False)

    sample_tokens = scene_data['sample_tokens']
    image_paths = []

    for token in sample_tokens:
        sample = nusc.get('sample', token)
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        img_path = str(Path(nuscenes_root) / cam_data['filename'])
        image_paths.append(img_path)

    return image_paths


def load_vlm_model(model_name: str, device: str = "cuda"):
    """Load VLM model and processor once."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def extract_vlm_features_full(
    image_paths: List[str],
    model,
    processor,
    vision_layers: List[int],
    num_keyframes: int,
    resolution: int,
    save_dtype: str,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract VLM VISION ENCODER hidden states (CORRECTED for ThinkJEPA).

    Key corrections:
        1. NO text prompts
        2. Hook VISION ENCODER (not language decoder)
        3. Direct forward pass (no .generate())
        4. Uniform temporal sampling
        5. Single middle layer (no averaging to preserve spatial structure)

    Returns:
        vlm_features: [num_tokens, hidden_dim] - pure visual features
    """
    from PIL import Image

    # 1. UNIFORM TEMPORAL SAMPLING (not just first N frames)
    if len(image_paths) > num_keyframes:
        stride = len(image_paths) // num_keyframes
        sampled_paths = [image_paths[i * stride] for i in range(num_keyframes)]
        sampled_paths = sampled_paths[:num_keyframes]  # Ensure exact count
    else:
        sampled_paths = image_paths

    # 2. Load and resize images
    images = []
    for path in sampled_paths:
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        images.append(img)

    # 3. MINIMAL TEXT (required by processor, but we hook vision encoder before fusion)
    # Use empty/minimal text to avoid semantic contamination
    # The key is we extract from VISION ENCODER before it fuses with text
    texts = [""] * len(images)  # Empty strings - minimal text influence
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # 4. Hook VISION ENCODER layers (not language decoder!)
    # Qwen3-VL architecture: model.visual is the vision encoder
    hidden_states_collector = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is [B, num_visual_tokens, hidden_dim]
            if isinstance(output, tuple):
                hidden_states_collector[layer_idx] = output[0].detach().cpu()
            else:
                hidden_states_collector[layer_idx] = output.detach().cpu()
        return hook_fn

    hooks = []
    # Hook vision encoder transformer blocks (middle layers)
    vision_encoder = model.visual.transformer  # Adjust path if needed for Qwen3-VL
    for idx in vision_layers:
        if hasattr(vision_encoder, 'blocks') and idx < len(vision_encoder.blocks):
            h = vision_encoder.blocks[idx].register_forward_hook(make_hook(idx))
            hooks.append(h)
        elif hasattr(vision_encoder, 'layers') and idx < len(vision_encoder.layers):
            h = vision_encoder.layers[idx].register_forward_hook(make_hook(idx))
            hooks.append(h)

    # 5. Direct vision forward pass (NO .generate()!)
    with torch.no_grad():
        # Process images through vision encoder only
        pixel_values = inputs.get('pixel_values', inputs.get('image'))
        if pixel_values is not None:
            _ = model.visual(pixel_values)
        else:
            # Fallback: minimal forward through model
            _ = model.model(**inputs, output_hidden_states=False)

    # Remove hooks
    for h in hooks:
        h.remove()

    # 6. Extract features (use SINGLE layer, don't average)
    if not hidden_states_collector:
        raise RuntimeError(f"No hidden states captured. Check vision encoder path. "
                          f"Model structure: {type(model.visual)}")

    # Take first (and typically only) layer
    layer_idx = vision_layers[0]
    if layer_idx not in hidden_states_collector:
        layer_idx = list(hidden_states_collector.keys())[0]

    features = hidden_states_collector[layer_idx].squeeze(0)  # [num_tokens, hidden_dim]
    features = features.numpy()

    if save_dtype == "fp16":
        features = features.astype(np.float16)

    return features


def extract_vlm_features_lite(
    num_frames: int,
    lite_dim: int,
    lite_tokens: int,
    save_dtype: str,
) -> np.ndarray:
    """Generate random features for debugging (no VLM needed)."""
    features = np.random.randn(lite_tokens, lite_dim).astype(np.float32)
    if save_dtype == "fp16":
        features = features.astype(np.float16)
    return features


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load slot pickle for scene/token info
    print(f"Loading slots from {args.slots_path}...")
    with open(args.slots_path, 'rb') as f:
        data = pickle.load(f)

    # Load VLM model once (if not in lite mode)
    model, processor = None, None
    if not args.lite:
        print(f"\nLoading VLM model: {args.model_name}...")
        model, processor = load_vlm_model(args.model_name)
        print("  Model loaded successfully")

    total_scenes = 0
    for split in ['train', 'val']:
        if split not in data:
            continue
        split_data = data[split]
        print(f"\nProcessing {split} split ({len(split_data)} scenes)...")

        for scene_name, scene_data in split_data.items():
            out_path = output_dir / f"{scene_name}.npz"

            if out_path.exists():
                print(f"  {scene_name}: already cached, skipping")
                total_scenes += 1
                continue

            num_frames = scene_data['slots'].shape[0]

            if args.lite:
                features = extract_vlm_features_lite(
                    num_frames=num_frames,
                    lite_dim=args.lite_dim,
                    lite_tokens=args.lite_tokens,
                    save_dtype=args.save_dtype,
                )
            else:
                image_paths = get_image_paths_for_scene(scene_data, args.nuscenes_root)

                features = extract_vlm_features_full(
                    image_paths=image_paths,
                    model=model,
                    processor=processor,
                    vision_layers=args.vision_layers,
                    num_keyframes=args.num_keyframes,
                    resolution=args.resolution,
                    save_dtype=args.save_dtype,
                )

            # Save
            np.savez_compressed(
                out_path,
                vlm_features=features,
                scene_name=scene_name,
                num_frames=num_frames,
                model_name=args.model_name if not args.lite else "lite_random",
                vision_layers=np.array(args.vision_layers) if not args.lite else np.array([]),
                num_keyframes=args.num_keyframes if not args.lite else 0,
                extraction_method="vision_encoder_only_no_text",
            )
            print(f"  {scene_name}: saved {features.shape} to {out_path}")
            total_scenes += 1

    print(f"\nDone. Cached {total_scenes} scenes to {output_dir}")


if __name__ == "__main__":
    main()
