"""
Offline VLM Cache Extraction for DriveWeaver (Following ThinkJEPA Official Implementation)

This is the CORRECTED implementation based on the actual ThinkJEPA codebase at:
/data1/work/j0987341/aadya/research/ThinkJEPA/cache_train/qwen3_cache_extractor.py

Key insights from ThinkJEPA paper implementation:
1. Text prompts ARE used (for reasoning guidance)
2. Language decoder layers ARE hooked (for semantic features)
3. .generate() IS used (to capture reasoning process)
4. TWO feature types: vlm_old (input) + vlm_new (generated reasoning)
5. Per-layer pyramid features (no averaging)

Usage:
    python scripts/cache_vlm_nuscenes_corrected.py \
        --slots_path /work/data/slots/nuscenes_slots_full.pkl \
        --output_dir /work/data/vlm_cache/nuscenes/ \
        --model_name Qwen/Qwen3-VL-2B-Thinking \
        --layers 6 12 18 24 \
        --num_keyframes 16 \
        --resolution 384 \
        --max_new_tokens 16
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser("VLM cache extraction for DriveWeaver (ThinkJEPA-aligned)")
    p.add_argument("--slots_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--nuscenes_root", type=str, default="/data/nuScenes")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Thinking")
    p.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24],
                   help="Language decoder layer indices (pyramid guidance)")
    p.add_argument("--num_keyframes", type=int, default=16,
                   help="Number of keyframes to uniformly sample")
    p.add_argument("--resolution", type=int, default=384,
                   help="Resize to NxN (384 is Qwen3-VL native)")
    p.add_argument("--prompt", type=str, default="Describe what will happen next.",
                   help="Text prompt for VLM reasoning")
    p.add_argument("--max_new_tokens", type=int, default=16,
                   help="Generate N tokens (captures reasoning process)")
    p.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--lite", action="store_true",
                   help="Generate random features (debug mode)")
    p.add_argument("--lite_dim", type=int, default=3584)
    p.add_argument("--lite_tokens_old", type=int, default=256)
    p.add_argument("--lite_tokens_new", type=int, default=16)
    return p.parse_args()


def get_image_paths_for_scene(scene_data: dict, nuscenes_root: str) -> List[str]:
    """Resolve sample_tokens to actual image file paths."""
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


def uniform_sample_keyframes(image_paths: List[str], num_keyframes: int) -> List[str]:
    """Uniformly sample keyframes across the full sequence."""
    if len(image_paths) <= num_keyframes:
        return image_paths

    stride = len(image_paths) / num_keyframes
    sampled_indices = [int(i * stride) for i in range(num_keyframes)]
    return [image_paths[i] for i in sampled_indices]


def load_vlm_model(model_name: str, device: str = "cuda"):
    """Load VLM model and processor."""
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


def register_decoder_hooks(model, layers: List[int]) -> Dict[str, List[torch.Tensor]]:
    """
    Hook language decoder layers (ThinkJEPA approach).

    Returns a dict where saved[f"dec_{layer_idx}"] accumulates hidden states.
    """
    # Locate decoder layers
    decoder_layers = model.model.language_model.layers

    # Verify layer indices
    for idx in layers:
        if idx < 0 or idx >= len(decoder_layers):
            raise ValueError(f"Layer {idx} out of range [0, {len(decoder_layers)})")

    # Prepare storage
    saved: Dict[str, List[torch.Tensor]] = {f"dec_{i}": [] for i in layers}

    def make_hook(name):
        def hook_fn(module, inp, out):
            h = out
            if isinstance(out, (tuple, list)):
                h = out[0]
            if torch.is_tensor(h):
                saved[name].append(h.detach().cpu())
        return hook_fn

    # Register hooks
    for layer_idx in layers:
        decoder_layers[layer_idx].register_forward_hook(make_hook(f"dec_{layer_idx}"))

    return saved


def stack_vlm_old_new(
    saved: Dict[str, List[torch.Tensor]],
    layers: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stack hidden states into vlm_old (input processing) and vlm_new (generation).

    vlm_old: Features from first forward pass (images + prompt → input tokens)
    vlm_new: Features from generation (reasoning tokens)

    Returns:
        vlm_old: [num_layers, num_input_tokens, hidden_dim]
        vlm_new: [num_layers, num_generated_tokens, hidden_dim]
    """
    old_list = []
    new_list = []

    for i in layers:
        key = f"dec_{i}"
        if key not in saved or len(saved[key]) == 0:
            continue

        # First captured state = input processing (vlm_old)
        old_list.append(saved[key][0])

        # Subsequent states = generation (vlm_new)
        if len(saved[key]) > 1:
            try:
                # Concatenate all generated tokens
                new = torch.cat(saved[key][1:], dim=1)
            except Exception:
                # Fallback: use last token
                new = saved[key][-1]
            new_list.append(new)

    # Stack into [num_layers, seq_len, hidden_dim]
    vlm_old = torch.stack(old_list, dim=0) if old_list else torch.empty((0,), dtype=torch.float16)
    vlm_new = torch.stack(new_list, dim=0) if new_list else torch.empty((0,), dtype=torch.float16)

    return vlm_old, vlm_new


def extract_vlm_features_full(
    image_paths: List[str],
    model,
    processor,
    layers: List[int],
    num_keyframes: int,
    resolution: int,
    prompt: str,
    max_new_tokens: int,
    save_dtype: str,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract VLM features following ThinkJEPA methodology.

    Returns:
        (vlm_old, vlm_new) both as numpy arrays
    """
    from PIL import Image

    # 1. Uniform temporal sampling
    sampled_paths = uniform_sample_keyframes(image_paths, num_keyframes)

    # 2. Load and resize images
    images = []
    for path in sampled_paths:
        img = Image.open(path).convert("RGB").resize((resolution, resolution))
        images.append(img)

    # 3. Build messages with images + text prompt
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    # 4. Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    # 5. Register hooks
    saved = register_decoder_hooks(model, layers)

    # 6. Generate (captures both input processing and reasoning)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 7. Extract vlm_old and vlm_new
    vlm_old, vlm_new = stack_vlm_old_new(saved, layers)

    # 8. Convert to numpy
    def to_numpy(x):
        if not torch.is_tensor(x) or x.numel() == 0:
            return np.zeros((0,), dtype=np.float16 if save_dtype == "fp16" else np.float32)
        arr = x.detach().float().cpu().numpy()
        if save_dtype == "fp16":
            arr = arr.astype(np.float16)
        return arr

    return to_numpy(vlm_old), to_numpy(vlm_new)


def extract_vlm_features_lite(
    num_frames: int,
    lite_dim: int,
    lite_tokens_old: int,
    lite_tokens_new: int,
    num_layers: int,
    save_dtype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random features for debugging."""
    dtype = np.float16 if save_dtype == "fp16" else np.float32
    vlm_old = np.random.randn(num_layers, lite_tokens_old, lite_dim).astype(dtype)
    vlm_new = np.random.randn(num_layers, lite_tokens_new, lite_dim).astype(dtype)
    return vlm_old, vlm_new


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load slot pickle
    print(f"Loading slots from {args.slots_path}...")
    with open(args.slots_path, 'rb') as f:
        data = pickle.load(f)

    # Load VLM model
    model, processor = None, None
    if not args.lite:
        print(f"\nLoading VLM model: {args.model_name}...")
        model, processor = load_vlm_model(args.model_name)
        print(f"  Model loaded successfully")
        print(f"  Decoder layers: {len(model.model.language_model.layers)}")
        print(f"  Hooking layers: {args.layers}")

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
                vlm_old, vlm_new = extract_vlm_features_lite(
                    num_frames=num_frames,
                    lite_dim=args.lite_dim,
                    lite_tokens_old=args.lite_tokens_old,
                    lite_tokens_new=args.lite_tokens_new,
                    num_layers=len(args.layers),
                    save_dtype=args.save_dtype,
                )
            else:
                image_paths = get_image_paths_for_scene(scene_data, args.nuscenes_root)

                vlm_old, vlm_new = extract_vlm_features_full(
                    image_paths=image_paths,
                    model=model,
                    processor=processor,
                    layers=args.layers,
                    num_keyframes=args.num_keyframes,
                    resolution=args.resolution,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    save_dtype=args.save_dtype,
                )

            # Save both vlm_old and vlm_new
            np.savez_compressed(
                out_path,
                vlm_old=vlm_old,
                vlm_new=vlm_new,
                scene_name=scene_name,
                num_frames=num_frames,
                model_name=args.model_name if not args.lite else "lite_random",
                layers=np.array(args.layers),
                num_keyframes=args.num_keyframes if not args.lite else 0,
                prompt=args.prompt if not args.lite else "",
                max_new_tokens=args.max_new_tokens if not args.lite else 0,
                extraction_method="thinkjepa_dual_path",
            )
            print(f"  {scene_name}: saved vlm_old={vlm_old.shape}, vlm_new={vlm_new.shape}")
            total_scenes += 1

    print(f"\nDone. Cached {total_scenes} scenes to {output_dir}")


if __name__ == "__main__":
    main()
