#!/usr/bin/env python3
"""
Precompute frozen C-JEPA future slots for fast planner training.

For Planner B this runs C-JEPA+VLM using cached VLM features. With
--no_vlm_guidance it runs pure C-JEPA for Planner A. Either way, this script
does the frozen world-model work once, scene-by-scene, and stores compact
future-slot tensors:

  output_dir/train.pt
  output_dir/val.pt

Each cache contains:
  - future_slots: [num_windows, T_fut, num_slots, slot_dim]
  - ego_history: [num_windows, T_hist, 4]
  - ego_future_trajectory: [num_windows, T_fut, 2]
  - scene_tokens, start_indices, metadata
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.cjepa_predictor import CJEPAPredictor
from models.thinkjepa import ThinkJEPA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute C-JEPA+VLM future slots")
    parser.add_argument('--slots_path', type=str, default='/work/data/slots/nuscenes_slots_full.pkl')
    parser.add_argument('--vlm_cache_dir', type=str, default='/work/data/vlm_cache/nuscenes')
    parser.add_argument('--world_model_ckpt', type=str, default='/work/checkpoints/cjepa_vlm/best_model.pth')
    parser.add_argument('--no_vlm_guidance', action='store_true', help='Run pure C-JEPA with no VLM features')
    parser.add_argument('--output_dir', type=str, default='/work/data/future_slots/cjepa_vlm')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--history_length', type=int, default=4)
    parser.add_argument('--future_length', type=int, default=6)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--save_dtype', choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def load_world_model(ckpt_path: str, device: torch.device, use_vlm_guidance: bool = True) -> ThinkJEPA:
    mode = 'C-JEPA+VLM' if use_vlm_guidance else 'C-JEPA'
    print(f"Loading {mode} world model: {ckpt_path}")
    cjepa = CJEPAPredictor(
        num_slots=11,
        slot_dim=128,
        history_frames=4,
        pred_frames=6,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.0,
        guidance_mode='film' if use_vlm_guidance else None,
        guidance_dim=2048 if use_vlm_guidance else None,
        guidance_hidden=512,
    )
    world_model = ThinkJEPA(cjepa=cjepa)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    if any(key.startswith('cjepa.') for key in state.keys()):
        # ThinkJEPA checkpoints already contain the wrapper prefix.
        missing, unexpected = world_model.load_state_dict(state, strict=False)
    else:
        # Standalone C-JEPA checkpoints store bare predictor keys.
        missing, unexpected = world_model.cjepa.load_state_dict(state, strict=False)

    missing_real = [key for key in missing if 'guidance' not in key]
    unexpected_real = [key for key in unexpected if 'guidance' not in key]
    if missing_real:
        print(f"WARNING: missing non-guidance keys: {missing_real}")
    if unexpected_real:
        print(f"WARNING: unexpected non-guidance keys: {unexpected_real}")

    guidance_missing = [key for key in missing if 'guidance' in key]
    if guidance_missing:
        print(f"  Guidance modules missing from checkpoint: {len(guidance_missing)} keys")

    loaded_params = sum(t.numel() for t in state.values() if torch.is_tensor(t))
    print(f"  Loaded checkpoint epoch={ckpt.get('epoch', 'unknown')} best_val_loss={ckpt.get('best_val_loss', 'unknown')}")
    print(f"  Checkpoint tensor parameters: {loaded_params:,}")
    world_model.to(device).eval()
    return world_model

def load_scene_vlm(scene_token: str, cache_dir: Path, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    path = cache_dir / f"{scene_token}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing VLM cache for scene {scene_token}: {path}")

    npz = np.load(path)
    if 'vlm_old' in npz and 'vlm_new' in npz:
        old = torch.from_numpy(npz['vlm_old'].astype(np.float32, copy=False)).to(device)
        new_arr = npz['vlm_new']
        new = (
            torch.from_numpy(new_arr.astype(np.float32, copy=False)).to(device)
            if new_arr.size > 0 else None
        )
        return {'old': old, 'new': new}

    if 'vlm_features' in npz:
        old = torch.from_numpy(npz['vlm_features'].astype(np.float32, copy=False)).unsqueeze(0).to(device)
        return {'old': old, 'new': None}

    raise KeyError(f"VLM cache {path} does not contain expected arrays")


def iter_chunks(items: List[int], chunk_size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def make_ego_tensors(
    ego_poses: np.ndarray,
    starts: List[int],
    history_length: int,
    future_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ego_histories = []
    ego_futures = []

    for start_idx in starts:
        end_idx = start_idx + history_length + future_length
        window = ego_poses[start_idx:end_idx]
        ego_history = window[:history_length]
        ego_future = window[history_length:]

        ref_x, ref_y, ref_yaw = ego_history[-1, 0], ego_history[-1, 1], ego_history[-1, 2]
        cos_r, sin_r = np.cos(-ref_yaw), np.sin(-ref_yaw)

        future_xy = ego_future[:, :2]
        dx = future_xy[:, 0] - ref_x
        dy = future_xy[:, 1] - ref_y
        ego_future_trajectory = np.stack([
            dx * cos_r - dy * sin_r,
            dx * sin_r + dy * cos_r,
        ], axis=-1)

        hist_xy = ego_history[:, :2]
        hdx = hist_xy[:, 0] - ref_x
        hdy = hist_xy[:, 1] - ref_y
        ego_history_local = ego_history.copy()
        ego_history_local[:, 0] = hdx * cos_r - hdy * sin_r
        ego_history_local[:, 1] = hdx * sin_r + hdy * cos_r
        ego_history_local[:, 2] = ego_history[:, 2] - ref_yaw

        ego_histories.append(torch.from_numpy(ego_history_local).float())
        ego_futures.append(torch.from_numpy(ego_future_trajectory).float())

    return torch.stack(ego_histories), torch.stack(ego_futures)


def build_history_batch(slots: np.ndarray, starts: List[int], history_length: int) -> torch.Tensor:
    histories = [slots[start_idx:start_idx + history_length] for start_idx in starts]
    return torch.from_numpy(np.stack(histories)).float()


def expand_guidance(scene_guidance: Dict[str, Optional[torch.Tensor]], batch_size: int) -> Dict[str, Optional[torch.Tensor]]:
    old_tensor = scene_guidance['old']
    old = old_tensor.unsqueeze(0).expand(batch_size, *([-1] * old_tensor.dim()))
    new_tensor = scene_guidance.get('new')
    new = None
    if new_tensor is not None:
        new = new_tensor.unsqueeze(0).expand(batch_size, *([-1] * new_tensor.dim()))
    return {'old': old, 'new': new}


def precompute_split(
    split: str,
    split_data: Dict,
    world_model: ThinkJEPA,
    vlm_cache_dir: Path,
    output_path: Path,
    args: argparse.Namespace,
    device: torch.device,
):
    if output_path.exists() and not args.overwrite:
        print(f"Skipping {split}: cache already exists at {output_path}")
        return

    future_chunks = []
    ego_history_chunks = []
    ego_future_chunks = []
    scene_tokens_all = []
    start_indices_all = []

    scene_tokens = list(split_data.keys())
    total_windows = 0
    for scene_token in scene_tokens:
        T = split_data[scene_token]['slots'].shape[0]
        total_windows += max(0, (T - args.history_length - args.future_length) // args.stride + 1)

    pbar = tqdm(scene_tokens, desc=f"Precompute {split} scenes")
    for scene_token in pbar:
        scene_data = split_data[scene_token]
        slots = scene_data['slots']
        ego_poses = scene_data['ego_poses']
        T = slots.shape[0]
        starts = list(range(0, T - args.history_length - args.future_length + 1, args.stride))
        if not starts:
            continue

        scene_guidance = load_scene_vlm(scene_token, vlm_cache_dir, device) if args.use_vlm_guidance else None
        pbar.set_postfix(scene=scene_token, windows=len(starts))

        for chunk_starts in iter_chunks(starts, args.batch_size):
            history = build_history_batch(slots, chunk_starts, args.history_length).to(device, non_blocking=True)
            ego_history, ego_future = make_ego_tensors(
                ego_poses, chunk_starts, args.history_length, args.future_length
            )
            guidance = (
                expand_guidance(scene_guidance, len(chunk_starts))
                if scene_guidance is not None else None
            )

            with torch.no_grad(), autocast(enabled=(device.type == 'cuda' and not args.no_amp)):
                future_slots = world_model.inference(history, vlm_guidance=guidance)

            if args.save_dtype == 'fp16':
                future_slots = future_slots.cpu().half()
            else:
                future_slots = future_slots.cpu().float()

            future_chunks.append(future_slots)
            ego_history_chunks.append(ego_history.cpu())
            ego_future_chunks.append(ego_future.cpu())
            scene_tokens_all.extend([scene_token] * len(chunk_starts))
            start_indices_all.extend(chunk_starts)

        if scene_guidance is not None:
            del scene_guidance
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    cache = {
        'future_slots': torch.cat(future_chunks, dim=0),
        'ego_history': torch.cat(ego_history_chunks, dim=0),
        'ego_future_trajectory': torch.cat(ego_future_chunks, dim=0),
        'scene_tokens': scene_tokens_all,
        'start_indices': torch.tensor(start_indices_all, dtype=torch.long),
        'metadata': {
            'split': split,
            'slots_path': args.slots_path,
            'vlm_cache_dir': args.vlm_cache_dir if args.use_vlm_guidance else None,
            'use_vlm_guidance': args.use_vlm_guidance,
            'world_model_ckpt': args.world_model_ckpt,
            'history_length': args.history_length,
            'future_length': args.future_length,
            'stride': args.stride,
            'save_dtype': args.save_dtype,
            'num_windows': total_windows,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    print(f"Saving {split} cache: {output_path}")
    print(f"  future_slots: {tuple(cache['future_slots'].shape)} {cache['future_slots'].dtype}")
    torch.save(cache, tmp_path)
    tmp_path.replace(output_path)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Device: {device}")
    if args.device == 'cuda' and device.type != 'cuda':
        raise RuntimeError("CUDA requested but not available")

    args.use_vlm_guidance = not args.no_vlm_guidance

    with open(args.slots_path, 'rb') as f:
        data = pickle.load(f)

    world_model = load_world_model(args.world_model_ckpt, device, use_vlm_guidance=args.use_vlm_guidance)
    output_dir = Path(args.output_dir)
    vlm_cache_dir = Path(args.vlm_cache_dir) if args.use_vlm_guidance else None

    for split in args.splits:
        if split not in data:
            raise KeyError(f"Split {split} not found in {args.slots_path}")
        precompute_split(
            split=split,
            split_data=data[split],
            world_model=world_model,
            vlm_cache_dir=vlm_cache_dir,
            output_path=output_dir / f"{split}.pt",
            args=args,
            device=device,
        )

    mode = "C-JEPA+VLM" if args.use_vlm_guidance else "C-JEPA"
    print(f"Done precomputing {mode} future-slot caches.")


if __name__ == '__main__':
    main()
