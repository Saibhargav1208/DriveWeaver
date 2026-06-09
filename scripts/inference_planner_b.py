from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.complete_pipeline import DriveWeaverPipeline
from models.cjepa_predictor import CJEPAPredictor
from models.thinkjepa import ThinkJEPA
from models.planner import Planner
from datasets.planner_dataset import (
    CachedFutureSlotsDataset,
    DriveJEPADataset,
    cached_collate_fn,
    collate_fn as planner_collate_fn,
)


def compute_ade_per_mode(proposals: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gt_exp = gt.unsqueeze(1).expand(-1, proposals.shape[1], -1, -1)
    return torch.norm(proposals - gt_exp, p=2, dim=-1).mean(dim=-1)


def summarize_errors(predictions: np.ndarray, ground_truths: np.ndarray) -> Dict[str, float]:
    errors = np.linalg.norm(predictions - ground_truths, axis=-1)
    ade_per_sample = errors.mean(axis=1)
    fde_per_sample = errors[:, -1]
    return {
        'ADE_mean': float(ade_per_sample.mean()),
        'ADE_std': float(ade_per_sample.std()),
        'FDE_mean': float(fde_per_sample.mean()),
        'FDE_std': float(fde_per_sample.std()),
        'MR@2m': float((fde_per_sample > 2.0).mean()),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Planner B scorer-based inference')
    parser.add_argument('--world_model_ckpt', type=str, default='/work/checkpoints/cjepa_vlm/best_model.pth')
    parser.add_argument('--planner_ckpt', type=str, required=True)
    parser.add_argument('--vlm_cache_dir', type=str, default='/work/data/vlm_cache/nuscenes/')
    parser.add_argument('--future_slots_cache_dir', type=str, default=None)
    parser.add_argument('--slots_path', type=str, default='/work/data/slots/nuscenes_slots_full.pkl')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--compare_with_oracle', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cached = args.future_slots_cache_dir is not None

    print('\n' + '=' * 70)
    print('Planner B Inference (scorer selection)')
    print('=' * 70)
    print(f'Device: {device}')
    if use_cached:
        print(f'Future-slot cache: {args.future_slots_cache_dir}')
    else:
        print(f'World model: {args.world_model_ckpt}')
        print(f'VLM cache:   {args.vlm_cache_dir}')
    print(f'Planner:     {args.planner_ckpt}')

    planner = Planner(slot_dim=128, num_slots=11, num_modes=32, future_len=6, history_len=4)
    planner_ckpt = torch.load(args.planner_ckpt, map_location='cpu', weights_only=False)
    planner.load_state_dict(planner_ckpt['model_state_dict'], strict=False)
    planner.to(device).eval()
    print(f'Loaded planner epoch {planner_ckpt.get("epoch", "unknown")}')

    pipeline = None
    if not use_cached:
        cjepa = CJEPAPredictor(
            num_slots=11, slot_dim=128, history_frames=4, pred_frames=6,
            depth=6, heads=8, dim_head=64, mlp_dim=2048, dropout=0.0,
            guidance_mode='film', guidance_dim=2048, guidance_hidden=512,
        )
        world_model = ThinkJEPA(cjepa=cjepa)
        world_ckpt = torch.load(args.world_model_ckpt, map_location='cpu', weights_only=False)
        world_model.load_state_dict(world_ckpt['model_state_dict'])
        world_model.to(device).eval()
        pipeline = DriveWeaverPipeline(world_model=world_model, planner=planner).to(device).eval()
        print(f'Loaded C-JEPA+VLM epoch {world_ckpt.get("epoch", "unknown")}')

    if use_cached:
        dataset = CachedFutureSlotsDataset(str(Path(args.future_slots_cache_dir) / f'{args.split}.pt'))
        collate = cached_collate_fn
    else:
        dataset = DriveJEPADataset(
            slots_path=args.slots_path,
            split=args.split,
            history_length=4,
            future_length=6,
            vlm_cache_dir=args.vlm_cache_dir,
        )
        collate = planner_collate_fn

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    def build_vlm_guidance(batch: Dict) -> Optional[Dict[str, torch.Tensor]]:
        if 'vlm_features' not in batch:
            return None
        return {k: v.to(device) if v is not None else None for k, v in batch['vlm_features'].items()}

    selected_predictions = []
    oracle_predictions = []
    ground_truths = []
    scene_tokens = []
    start_indices = []
    selected_ranks = []
    selected_scores = []
    processed = 0

    for batch in tqdm(loader, desc='Inference'):
        if args.max_samples is not None and processed >= args.max_samples:
            break
        ego = batch['ego_history'].to(device)
        gt = batch['ego_future_trajectory'].to(device)

        if use_cached:
            future_slots = batch['future_slots'].to(device).float()
            result = planner(future_slots, ego, return_all_proposals=True)
        else:
            history = batch['history_slots'].to(device)
            result = pipeline(history, ego, vlm_guidance=build_vlm_guidance(batch), return_intermediates=True)

        proposals = result['trajectory_proposals']
        scores = result['proposal_scores']
        ade_per_mode = compute_ade_per_mode(proposals, gt)
        scorer_idx = scores.argmax(dim=1)
        oracle_idx = ade_per_mode.argmin(dim=1)
        b = torch.arange(proposals.shape[0], device=device)
        selected = proposals[b, scorer_idx]
        oracle = proposals[b, oracle_idx]

        ranks = (torch.argsort(ade_per_mode, dim=1) == scorer_idx[:, None]).nonzero()[:, 1] + 1
        selected_ranks.extend(ranks.cpu().tolist())
        selected_scores.extend(scores[b, scorer_idx].cpu().tolist())

        selected_predictions.append(selected.cpu().numpy())
        oracle_predictions.append(oracle.cpu().numpy())
        ground_truths.append(gt.cpu().numpy())
        scene_tokens.extend(batch.get('scene_tokens', ['unknown'] * proposals.shape[0]))
        start_indices.extend(batch.get('start_indices', [0] * proposals.shape[0]))
        processed += proposals.shape[0]

    predictions = np.concatenate(selected_predictions, axis=0)
    oracle_predictions = np.concatenate(oracle_predictions, axis=0)
    ground_truths = np.concatenate(ground_truths, axis=0)

    if args.max_samples is not None:
        predictions = predictions[:args.max_samples]
        oracle_predictions = oracle_predictions[:args.max_samples]
        ground_truths = ground_truths[:args.max_samples]
        scene_tokens = scene_tokens[:args.max_samples]
        start_indices = start_indices[:args.max_samples]
        selected_ranks = selected_ranks[:args.max_samples]
        selected_scores = selected_scores[:args.max_samples]

    print(f'Inference complete: {len(predictions)} samples')

    metrics = {}
    if args.compute_metrics:
        metrics['scorer'] = summarize_errors(predictions, ground_truths)
        metrics['scorer']['avg_rank'] = float(np.mean(selected_ranks))
        metrics['scorer']['score_mean'] = float(np.mean(selected_scores))
        if args.compare_with_oracle:
            metrics['oracle'] = summarize_errors(oracle_predictions, ground_truths)
            metrics['oracle']['avg_rank'] = 1.0
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(output_dir / 'metrics_summary.txt', 'w') as f:
            f.write('Planner B Inference (scorer selection)\n')
            f.write('=' * 70 + '\n\n')
            for section, vals in metrics.items():
                f.write(f'{section}\n')
                for key, value in vals.items():
                    f.write(f'  {key}: {value:.6f}\n')
                f.write('\n')
        print(json.dumps(metrics, indent=2))

    if args.save_predictions:
        np.save(output_dir / 'predictions.npy', predictions)
        np.save(output_dir / 'ground_truths.npy', ground_truths)
        metadata = {
            'scene_tokens': scene_tokens,
            'start_indices': start_indices,
            'num_samples': len(predictions),
            'world_model_ckpt': args.world_model_ckpt,
            'planner_ckpt': args.planner_ckpt,
            'future_slots_cache_dir': args.future_slots_cache_dir,
            'split': args.split,
            'variant': 'planner_b',
            'selection': 'proposal_scores_argmax',
        }
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    print(f'Output directory: {output_dir}')


if __name__ == '__main__':
    main()
