"""
C-JEPA + VLM Training (Ablation 2: World Model)

Trains the C-JEPA predictor with per-layer VLM guidance injection (FiLM).
This is the world model for Ablation 2. Initializes from a pre-trained C-JEPA
checkpoint and jointly trains both the predictor and guidance modules.

The VLM features come from cached Qwen3-VL extractions (offline) or random
tensors (for debug).

Usage:
    # Debug (random VLM features):
    PYTHONPATH=/work python training/train_cjepa_vlm.py --config-name debug

    # Full (requires VLM cache):
    PYTHONPATH=/work python training/train_cjepa_vlm.py --config-name default
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from datasets.slot_dataset import create_dataloaders
from models.thinkjepa import ThinkJEPA, create_thinkjepa_from_config, load_cjepa_checkpoint_into_thinkjepa


class CJEPAWithVLMTrainer:
    """Trains C-JEPA world model with VLM guidance injection."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if cfg.system.cuda and torch.cuda.is_available() else 'cpu'
        )
        self._set_seed(cfg.system.seed)

        # Build model
        print("\n=== Building C-JEPA + VLM World Model ===")
        self.model = create_thinkjepa_from_config(cfg)
        self.model.to(self.device)

        # Load C-JEPA checkpoint as initialization
        cjepa_ckpt = cfg.get('cjepa_checkpoint', None)
        if cjepa_ckpt and Path(cjepa_ckpt).exists():
            print(f"  Initializing from: {cjepa_ckpt}")
            load_cjepa_checkpoint_into_thinkjepa(self.model, cjepa_ckpt)
        else:
            print(f"  No checkpoint — training from scratch")

        # Report params
        counts = self.model.count_params()
        print(f"\n  Parameters:")
        print(f"    Base (C-JEPA):  {counts['base']:>10,}")
        print(f"    Guidance (VLM): {counts['guidance']:>10,}")
        print(f"    Total:          {counts['base'] + counts['guidance']:>10,}")

        # Data
        print("\n  Loading data...")
        vlm_cfg = cfg.get('vlm_guidance', {})
        vlm_cache_dir = vlm_cfg.get('cache_dir', None)
        vlm_random_dim = vlm_cfg.get('vlm_dim', None) if vlm_cfg.get('use_random', False) else None
        vlm_random_tokens = vlm_cfg.get('random_tokens', 480)

        self.train_loader, self.val_loader = create_dataloaders(
            slots_path=cfg.data.slots_path,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            history_length=cfg.data.temporal.history_length,
            future_length=cfg.data.temporal.future_length,
            stride=cfg.data.temporal.stride,
            pin_memory=cfg.data.pin_memory,
            vlm_cache_dir=vlm_cache_dir,
            vlm_random_dim=vlm_random_dim,
            vlm_random_tokens=vlm_random_tokens,
        )

        # Optimizer: separate LRs for base vs guidance
        base_lr = cfg.training.learning_rate
        guidance_lr = cfg.training.get('guidance_lr', base_lr * 10)
        base_params = [p for p in self.model.get_base_params() if p.requires_grad]
        guidance_params = [p for p in self.model.get_guidance_params() if p.requires_grad]

        param_groups = []
        if base_params:
            param_groups.append({'params': base_params, 'lr': base_lr})
        if guidance_params:
            param_groups.append({'params': guidance_params, 'lr': guidance_lr})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.training.weight_decay,
            betas=tuple(cfg.training.betas),
            eps=cfg.training.eps,
        )
        print(f"\n  Optimizer: AdamW")
        print(f"    Base:     {sum(p.numel() for p in base_params):,} params @ LR={base_lr}")
        print(f"    Guidance: {sum(p.numel() for p in guidance_params):,} params @ LR={guidance_lr}")

        # Scheduler
        sched_cfg = cfg.training.scheduler
        if sched_cfg.type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sched_cfg.total_steps, eta_min=sched_cfg.min_lr,
            )
        else:
            self.scheduler = None

        # AMP
        self.use_amp = cfg.training.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _build_vlm_guidance(self, batch: Dict) -> Optional[Dict[str, torch.Tensor]]:
        if 'vlm_features' not in batch:
            return None
        return {'vlm_features': batch['vlm_features'].to(self.device)}

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_improvement = 0.0
        n = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [train]")
        for batch in pbar:
            history = batch['history_slots'].to(self.device)
            future_gt = batch['future_slots'].to(self.device)
            vlm_guidance = self._build_vlm_guidance(batch)

            T_hist = history.shape[1]

            if self.use_amp:
                with autocast():
                    out, _ = self.model(history, vlm_guidance=vlm_guidance)
                    pred_future = out[:, T_hist:, :, :]
                    loss = F.mse_loss(pred_future, future_gt)
            else:
                out, _ = self.model(history, vlm_guidance=vlm_guidance)
                pred_future = out[:, T_hist:, :, :]
                loss = F.mse_loss(pred_future, future_gt)

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.cfg.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.training.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            # Improvement over unguided
            with torch.no_grad():
                unguided = self.model.inference(history, vlm_guidance=None)
                mse_unguided = F.mse_loss(unguided, future_gt).item()
                mse_guided = loss.item()
                improvement = 1.0 - mse_guided / max(mse_unguided, 1e-12)

            epoch_loss += loss.item()
            epoch_improvement += improvement
            n += 1

            gates = self.model.get_guidance_gate_values()
            gate_str = f"{gates[0]:.3f}" if gates else "N/A"
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'gate[0]': gate_str})
            self.global_step += 1

        return {'loss': epoch_loss / n, 'improvement': epoch_improvement / n}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        val_improvement = 0.0
        n = 0

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [val] "):
            history = batch['history_slots'].to(self.device)
            future_gt = batch['future_slots'].to(self.device)
            vlm_guidance = self._build_vlm_guidance(batch)

            guided = self.model.inference(history, vlm_guidance=vlm_guidance)
            unguided = self.model.inference(history, vlm_guidance=None)

            mse_guided = F.mse_loss(guided, future_gt).item()
            mse_unguided = F.mse_loss(unguided, future_gt).item()
            improvement = 1.0 - mse_guided / max(mse_unguided, 1e-12)

            val_loss += mse_guided
            val_improvement += improvement
            n += 1

        return {'loss': val_loss / n, 'improvement': val_improvement / n}

    def save_checkpoint(self, filename: str, is_best: bool = False):
        ckpt = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }
        path = self.checkpoint_dir / filename
        torch.save(ckpt, path)
        print(f"  Saved: {path}")
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pth')
            print(f"  Best model updated")

        # Clean up old checkpoints (keep only last N)
        keep_last_n = self.cfg.training.get('keep_last_n', 3)
        self._cleanup_old_checkpoints(keep_last_n)

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
                    print(f"  Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"  Warning: Could not remove {old_ckpt}: {e}")

    def train(self):
        print(f"\n{'='*60}")
        print(f"C-JEPA + VLM Training")
        print(f"  Guidance mode: {self.cfg.vlm_guidance.guidance_mode}")
        print(f"{'='*60}")

        for epoch in range(self.current_epoch, self.cfg.training.max_epochs):
            self.current_epoch = epoch

            train_m = self.train_epoch()
            val_m = self.validate()

            gates = self.model.get_guidance_gate_values()
            gate_str = ", ".join(f"{g:.3f}" for g in gates[:4]) if gates else "N/A"
            print(
                f"\nEpoch {epoch:03d} | "
                f"Train: loss={train_m['loss']:.4f} impr={train_m['improvement']:.4f} | "
                f"Val: loss={val_m['loss']:.4f} impr={val_m['improvement']:.4f} | "
                f"Gates: [{gate_str}]"
            )

            is_best = val_m['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m['loss']

            if epoch % self.cfg.training.save_frequency == 0 or is_best:
                self.save_checkpoint(f'checkpoint_epoch{epoch:04d}.pth', is_best=is_best)

        print(f"\n{'='*60}")
        print(f"Training Complete | Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")


@hydra.main(version_base=None, config_path="../configs/cjepa_vlm", config_name="default")
def main(cfg: DictConfig):
    print("C-JEPA + VLM Configuration:")
    print(OmegaConf.to_yaml(cfg))
    trainer = CJEPAWithVLMTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
