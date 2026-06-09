"""
Planner Training — Ablation 2: C-JEPA + VLM → Planner

Freezes the C-JEPA+VLM world model and trains only the trajectory planner head.
VLM guidance is injected into the world model during inference (features from cache
or random for debug).

Usage:
    PYTHONPATH=/work python training/train_planner_b.py --config-name debug_thinkjepa
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))

from datasets.planner_dataset import create_planner_dataloaders, create_cached_planner_dataloaders
from models.complete_pipeline import create_complete_pipeline
from models.planner import create_planner
from training.losses_planner import PlannerLoss


class PlannerTrainerB:
    """Ablation 2: C-JEPA + VLM (frozen) -> Planner (trained)."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if cfg.system.cuda and torch.cuda.is_available() else 'cpu'
        )
        self._set_seed(cfg.system.seed)

        print("\n=== Ablation 2: C-JEPA + VLM -> Planner ===")
        print(f"Device: {self.device}")

        self.use_cached_future_slots = bool(cfg.data.get('future_slots_cache_dir', None))
        self.pipeline = None

        vlm_cfg = cfg.vlm_guidance
        self.use_vlm_guidance = bool(vlm_cfg.get('enable', False))
        self.vlm_cache_dir = vlm_cfg.get('cache_dir', None) if self.use_vlm_guidance else None
        self.vlm_random_dim = (
            vlm_cfg.get('vlm_dim', None)
            if self.use_vlm_guidance and vlm_cfg.get('use_random', False) else None
        )
        self.vlm_random_tokens_old = vlm_cfg.get('random_tokens_old', vlm_cfg.get('random_tokens', 480))
        self.vlm_random_tokens_new = vlm_cfg.get('random_tokens_new', 16)

        if self.use_cached_future_slots:
            print(f"\nUsing precomputed future slots: {cfg.data.future_slots_cache_dir}")
            self.planner = create_planner(
                slot_dim=cfg.model.slot_dim,
                num_slots=cfg.model.num_slots,
                num_modes=cfg.model.num_modes,
                future_len=cfg.model.future_length,
                history_len=cfg.model.history_length,
                num_decoder_layers=cfg.model.planner_decoder_layers,
                num_heads=cfg.model.planner_heads,
            ).to(self.device)

            planner_ckpt = cfg.checkpoints.get('planner', None)
            if planner_ckpt and Path(planner_ckpt).exists():
                print(f"  Loading planner: {planner_ckpt}")
                ckpt = torch.load(planner_ckpt, map_location='cpu', weights_only=False)
                state = ckpt.get('model_state_dict', ckpt)
                self.planner.load_state_dict(state, strict=False)

            total = sum(p.numel() for p in self.planner.parameters())
            trainable = sum(p.numel() for p in self.planner.parameters() if p.requires_grad)
            print(f"\n=== Planner Parameters ===")
            print(f"Trainable: {trainable:,}")
            print(f"Total:     {total:,}")

            self.train_loader, self.val_loader = create_cached_planner_dataloaders(
                cache_dir=cfg.data.future_slots_cache_dir,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
            )
        else:
            # Build pipeline (with VLM guidance) for the uncached fallback path.
            self.pipeline = create_complete_pipeline(
                slot_dim=cfg.model.slot_dim,
                num_slots=cfg.model.num_slots,
                history_len=cfg.model.history_length,
                future_len=cfg.model.future_length,
                num_modes=cfg.model.num_modes,
                cjepa_depth=cfg.model.cjepa_depth,
                cjepa_heads=cfg.model.cjepa_heads,
                cjepa_mlp_dim=cfg.model.cjepa_mlp_dim,
                guidance_mode=vlm_cfg.guidance_mode,
                guidance_dim=vlm_cfg.vlm_dim,
                guidance_hidden=vlm_cfg.get('guidance_hidden', 512),
                planner_decoder_layers=cfg.model.planner_decoder_layers,
                planner_heads=cfg.model.planner_heads,
                cjepa_checkpoint=cfg.checkpoints.get('cjepa_vlm', None) or cfg.checkpoints.get('cjepa', None),
                device=str(self.device),
            )

            self.pipeline.freeze_world_model()
            self.pipeline.print_parameter_summary()
            self.planner = self.pipeline.planner

            self.train_loader, self.val_loader = create_planner_dataloaders(
                slots_path=cfg.data.slots_path,
                batch_size=cfg.data.batch_size,
                history_length=cfg.model.history_length,
                future_length=cfg.model.future_length,
                stride=cfg.data.stride,
                num_workers=cfg.data.num_workers,
                vlm_cache_dir=self.vlm_cache_dir,
                vlm_random_dim=self.vlm_random_dim,
                vlm_random_tokens_old=self.vlm_random_tokens_old,
                vlm_random_tokens_new=self.vlm_random_tokens_new,
            )

        self.criterion = PlannerLoss(
            lambda_ade=cfg.loss.lambda_ade,
            lambda_fde=cfg.loss.lambda_fde,
            lambda_wta=cfg.loss.lambda_wta,
            lambda_diversity=cfg.loss.lambda_diversity,
            lambda_smoothness=cfg.loss.lambda_smoothness,
            lambda_feasibility=cfg.loss.lambda_feasibility,
            diversity_sigma=cfg.loss.diversity_sigma,
            penalize_jerk=cfg.loss.penalize_jerk,
            max_speed=cfg.loss.max_speed,
            max_accel=cfg.loss.max_accel,
            dt=cfg.loss.dt,
            lambda_score_ce=cfg.loss.get('lambda_score_ce', 1.0),
            lambda_score_kl=cfg.loss.get('lambda_score_kl', 0.0),
            score_temperature=cfg.loss.get('score_temperature', 0.5),
            lambda_all_ade=cfg.loss.get('lambda_all_ade', 0.05),
        )

        planner_params = [p for p in self.planner.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            planner_params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
        )
        print(f"\n  Optimizer: {sum(p.numel() for p in planner_params):,} planner params @ LR={cfg.training.learning_rate}")

        self.use_amp = bool(cfg.training.mixed_precision and self.device.type == 'cuda')
        self.scaler = GradScaler() if self.use_amp else None
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _build_vlm_guidance(self, batch: Dict) -> Optional[Dict[str, torch.Tensor]]:
        if not self.use_vlm_guidance or 'vlm_features' not in batch:
            return None

        vlm_features = batch['vlm_features']
        if isinstance(vlm_features, dict):
            result = {}
            for key, tensor in vlm_features.items():
                result[key] = tensor.to(self.device, non_blocking=True) if tensor is not None else None
            return result

        return {'old': vlm_features.to(self.device, non_blocking=True).unsqueeze(1), 'new': None}

    def _forward_planner(self, batch: Dict):
        ego = batch['ego_history'].to(self.device, non_blocking=True)
        gt_traj = batch['ego_future_trajectory'].to(self.device, non_blocking=True)

        if self.use_cached_future_slots:
            future_slots = batch['future_slots'].to(self.device, non_blocking=True).float()
        else:
            history = batch['history_slots'].to(self.device, non_blocking=True)
            vlm_guidance = self._build_vlm_guidance(batch)
            with torch.no_grad():
                future_slots = self.pipeline.world_model.inference(history, vlm_guidance=vlm_guidance)

        result = self.planner(
            refined_slots=future_slots,
            ego_state=ego,
            return_all_proposals=True,
        )
        return result, gt_traj

    def train_epoch(self) -> Dict[str, float]:
        self.planner.train()
        if self.pipeline is not None:
            self.pipeline.world_model.eval()

        total_loss = 0.0
        metrics_sum = {}
        n = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [train]"):
            with autocast(enabled=self.use_amp):
                result, gt_traj = self._forward_planner(batch)
                proposals = result['trajectory_proposals']
                scores = result['proposal_scores']
                loss, metrics = self.criterion(proposals, gt_traj, scores)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.planner.parameters(), self.cfg.training.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.planner.parameters(), self.cfg.training.clip_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            n += 1
            self.global_step += 1

        avg = {k: v / n for k, v in metrics_sum.items()}
        avg['loss'] = total_loss / n
        return avg

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.planner.eval()
        if self.pipeline is not None:
            self.pipeline.world_model.eval()

        total_loss = 0.0
        metrics_sum = {}
        n = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [val] "):
            with autocast(enabled=self.use_amp):
                result, gt_traj = self._forward_planner(batch)
                proposals = result['trajectory_proposals']
                scores = result['proposal_scores']
                loss, metrics = self.criterion(proposals, gt_traj, scores)

            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            n += 1

        avg = {k: v / n for k, v in metrics_sum.items()}
        avg['loss'] = total_loss / n
        return avg

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.planner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'uses_cached_future_slots': self.use_cached_future_slots,
            'uses_score_loss': True,
        }
        path = self.checkpoint_dir / f'checkpoint_epoch{epoch:04d}.pth'
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pth')
            print(f"  Best model saved (val loss={self.best_val_loss:.4f})")

        self._cleanup_old_checkpoints(keep_last_n=2)

    def _cleanup_old_checkpoints(self, keep_last_n: int):
        """Remove old checkpoint files, keeping only the last N."""
        import glob

        checkpoint_pattern = str(self.checkpoint_dir / 'checkpoint_epoch*.pth')
        checkpoints = sorted(glob.glob(checkpoint_pattern))

        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                try:
                    Path(old_ckpt).unlink()
                    print(f"  Removed old checkpoint: {Path(old_ckpt).name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {old_ckpt}: {e}")

    def train(self):
        print(f"\n{'='*60}")
        print(f"Ablation 2: C-JEPA + VLM -> Planner")
        print(f"{'='*60}")

        for epoch in range(self.cfg.training.max_epochs):
            self.current_epoch = epoch
            train_m = self.train_epoch()
            val_m = self.validate()

            print(
                f"Epoch {epoch:03d} | "
                f"Train: loss={train_m['loss']:.2f} ADE={train_m.get('loss_minADE',0):.2f} "
                f"ScoreAcc={train_m.get('score_acc',0):.2f} Rank={train_m.get('score_avg_rank',0):.1f} | "
                f"Val: loss={val_m['loss']:.2f} ADE={val_m.get('loss_minADE',0):.2f} "
                f"ScoreAcc={val_m.get('score_acc',0):.2f} Rank={val_m.get('score_avg_rank',0):.1f}"
            )

            is_best = val_m['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m['loss']
            if epoch % self.cfg.training.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        print(f"\nDone. Best val loss: {self.best_val_loss:.4f}")


@hydra.main(version_base=None, config_path="../configs/planner_b", config_name="debug")
def main(cfg: DictConfig):
    trainer = PlannerTrainerB(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
