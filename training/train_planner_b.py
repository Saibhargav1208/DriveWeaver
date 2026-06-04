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

from datasets.planner_dataset import create_planner_dataloaders
from models.complete_pipeline import create_complete_pipeline, DriveWeaverPipeline
from training.losses_planner import PlannerLoss


class PlannerTrainerB:
    """Ablation 2: C-JEPA + VLM (frozen) → Planner (trained)."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if cfg.system.cuda and torch.cuda.is_available() else 'cpu'
        )
        self._set_seed(cfg.system.seed)

        print("\n=== Ablation 2: C-JEPA + VLM → Planner ===")

        vlm_cfg = cfg.vlm_guidance
        self.vlm_random_dim = vlm_cfg.get('vlm_dim', None) if vlm_cfg.get('use_random', False) else None
        self.vlm_random_tokens = vlm_cfg.get('random_tokens', 480)

        # Build pipeline (with VLM guidance)
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

        # Freeze world model
        self.pipeline.freeze_world_model()
        self.pipeline.print_parameter_summary()

        # Data
        self.train_loader, self.val_loader = create_planner_dataloaders(
            slots_path=cfg.data.slots_path,
            batch_size=cfg.data.batch_size,
            history_length=cfg.model.history_length,
            future_length=cfg.model.future_length,
            stride=cfg.data.stride,
            num_workers=cfg.data.num_workers,
        )

        # Loss
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
        )

        # Optimizer (planner only)
        planner_params = [p for p in self.pipeline.planner.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            planner_params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay,
        )
        print(f"\n  Optimizer: {sum(p.numel() for p in planner_params):,} planner params @ LR={cfg.training.learning_rate}")

        self.use_amp = cfg.training.mixed_precision
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

    def _get_vlm_guidance(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        if self.vlm_random_dim:
            features = torch.randn(batch_size, self.vlm_random_tokens, self.vlm_random_dim, device=self.device)
            return {'vlm_features': features}
        return None

    def train_epoch(self) -> Dict[str, float]:
        self.pipeline.planner.train()
        self.pipeline.world_model.eval()
        total_loss = 0.0
        metrics_sum = {}
        n = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [train]"):
            history = batch['history_slots'].to(self.device)
            ego = batch['ego_history'].to(self.device)
            gt_traj = batch['ego_future_trajectory'].to(self.device)
            B = history.shape[0]
            vlm_guidance = self._get_vlm_guidance(B)

            with autocast(enabled=self.use_amp):
                result = self.pipeline(history, ego, vlm_guidance=vlm_guidance, return_intermediates=True)
                proposals = result['trajectory_proposals']
                loss, metrics = self.criterion(proposals, gt_traj)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.pipeline.planner.parameters(), self.cfg.training.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.pipeline.planner.parameters(), self.cfg.training.clip_grad_norm)
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
        self.pipeline.eval()
        total_loss = 0.0
        metrics_sum = {}
        n = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [val] "):
            history = batch['history_slots'].to(self.device)
            ego = batch['ego_history'].to(self.device)
            gt_traj = batch['ego_future_trajectory'].to(self.device)
            B = history.shape[0]
            vlm_guidance = self._get_vlm_guidance(B)

            result = self.pipeline(history, ego, vlm_guidance=vlm_guidance, return_intermediates=True)
            proposals = result['trajectory_proposals']
            loss, metrics = self.criterion(proposals, gt_traj)

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
            'model_state_dict': self.pipeline.planner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        path = self.checkpoint_dir / f'checkpoint_epoch{epoch:04d}.pth'
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pth')
            print(f"  Best model saved (val loss={self.best_val_loss:.4f})")

        # Clean up old checkpoints (keep only last 2)
        self._cleanup_old_checkpoints(keep_last_n=2)

    def _cleanup_old_checkpoints(self, keep_last_n: int):
        """Remove old checkpoint files, keeping only the last N."""
        import glob

        # Get all checkpoint files (exclude best_model.pth)
        checkpoint_pattern = str(self.checkpoint_dir / 'checkpoint_epoch*.pth')
        checkpoints = sorted(glob.glob(checkpoint_pattern))

        # Remove older checkpoints
        if len(checkpoints) > keep_last_n:
            for old_ckpt in checkpoints[:-keep_last_n]:
                try:
                    Path(old_ckpt).unlink()
                    print(f"  Removed old checkpoint: {Path(old_ckpt).name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {old_ckpt}: {e}")

    def train(self):
        print(f"\n{'='*60}")
        print(f"Ablation 2: C-JEPA + VLM → Planner")
        print(f"{'='*60}")

        for epoch in range(self.cfg.training.max_epochs):
            self.current_epoch = epoch
            train_m = self.train_epoch()
            val_m = self.validate()

            print(
                f"Epoch {epoch:03d} | "
                f"Train: loss={train_m['loss']:.2f} ADE={train_m.get('loss_minADE',0):.2f} | "
                f"Val: loss={val_m['loss']:.2f} ADE={val_m.get('loss_minADE',0):.2f}"
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
