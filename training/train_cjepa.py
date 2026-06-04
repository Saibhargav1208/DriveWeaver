"""
C-JEPA Training Pipeline

Main training script for Phase 2: Temporal World Model
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.slot_dataset import create_dataloaders
from models.cjepa_predictor import create_cjepa_from_config
from training.losses_cjepa import compute_cjepa_loss, compute_cjepa_loss_inference


class CJEPATrainer:
    """
    Trainer for C-JEPA temporal world model.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if cfg.system.cuda and torch.cuda.is_available() else 'cpu')

        # Set random seeds
        self._set_seed(cfg.system.seed)

        # Create model
        print("Creating C-JEPA model...")
        self.model = create_cjepa_from_config(cfg)
        self.model.to(self.device)
        print(f"Model on device: {self.device}")

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Create dataloaders
        print("\nCreating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            slots_path=cfg.data.slots_path,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            history_length=cfg.data.temporal.history_length,
            future_length=cfg.data.temporal.future_length,
            stride=cfg.data.temporal.stride,
            pin_memory=cfg.data.pin_memory
        )

        # C-JEPA uses simple MSE loss (no separate criterion needed)

        # Create optimizer
        self.optimizer = self._create_optimizer(cfg)

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler(cfg)

        # Mixed precision training
        self.use_amp = cfg.training.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if configured
        self.use_wandb = False
        if 'wandb' in cfg and cfg.get('wandb', {}).get('enabled', False):
            self._init_wandb(cfg)

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if self.cfg.system.cudnn_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def _create_optimizer(self, cfg: DictConfig):
        """Create optimizer."""
        if cfg.training.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
                betas=cfg.training.betas,
                eps=cfg.training.eps
            )
        elif cfg.training.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
                betas=cfg.training.betas,
                eps=cfg.training.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

        return optimizer

    def _create_scheduler(self, cfg: DictConfig):
        """Create learning rate scheduler."""
        if cfg.training.scheduler.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.training.scheduler.total_steps,
                eta_min=cfg.training.scheduler.min_lr
            )
        elif cfg.training.scheduler.type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.training.scheduler.get('step_size', 10),
                gamma=cfg.training.scheduler.get('gamma', 0.1)
            )
        elif cfg.training.scheduler.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=cfg.training.scheduler.get('factor', 0.5),
                patience=cfg.training.scheduler.get('patience', 5)
            )
        else:
            scheduler = None

        return scheduler

    def _init_wandb(self, cfg: DictConfig):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=cfg.experiment.project,
            name=cfg.experiment.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.experiment.tags,
            notes=cfg.experiment.notes
        )
        wandb.watch(self.model, log='all', log_freq=100)
        self.use_wandb = True

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'loss': 0.0,
            'loss_masked_history': 0.0,
            'loss_future': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            history_slots = batch['history_slots'].to(self.device)  # [B, T_hist, N, D]
            future_slots = batch['future_slots'].to(self.device)    # [B, T_fut, N, D]

            # Forward pass with mixed precision
            # C-JEPA returns: (full_sequence, masked_indices)
            # full_sequence: [B, T_hist+T_future, N, D]
            T_hist = history_slots.shape[1]

            if self.use_amp:
                with autocast():
                    full_output, masked_indices = self.model(history_slots)

                    # Compute C-JEPA loss (MSE only)
                    losses = compute_cjepa_loss(
                        pred_full=full_output,
                        history=history_slots,
                        target_future=future_slots,
                        masked_indices=masked_indices,
                        history_size=T_hist
                    )
                    loss = losses['loss']
            else:
                full_output, masked_indices = self.model(history_slots)

                losses = compute_cjepa_loss(
                    pred_full=full_output,
                    history=history_slots,
                    target_future=future_slots,
                    masked_indices=masked_indices,
                    history_size=T_hist
                )
                loss = losses['loss']

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                if self.cfg.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg.training.gradient_clip
                    )
                self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None and self.cfg.training.scheduler.type != 'plateau':
                self.scheduler.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key].item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to wandb
            if self.use_wandb and self.global_step % self.cfg.training.log_frequency == 0:
                log_dict = {f'train/{k}': v.item() for k, v in losses.items()}
                log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(log_dict, step=self.global_step)

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        val_losses = {
            'loss': 0.0,
            'loss_masked_history': 0.0,
            'loss_future': 0.0
        }

        for batch in tqdm(self.val_loader, desc="Validation"):
            history_slots = batch['history_slots'].to(self.device)
            future_slots = batch['future_slots'].to(self.device)

            # Forward pass using inference mode (no masking)
            predicted = self.model.inference(history_slots)

            # Compute losses (inference mode)
            losses = compute_cjepa_loss_inference(predicted, future_slots)

            # Accumulate
            for key in val_losses.keys():
                val_losses[key] += losses[key].item()

        # Average
        num_batches = len(self.val_loader)
        for key in val_losses.keys():
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': OmegaConf.to_container(self.cfg, resolve=True)
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

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
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Warning: Could not remove {old_ckpt}: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        """Main training loop."""
        print(f"\n{'='*50}")
        print(f"Starting C-JEPA Training")
        print(f"{'='*50}\n")

        for epoch in range(self.current_epoch, self.cfg.training.max_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_losses = self.train_epoch()
            print(f"\nEpoch {epoch} - Train Loss: {train_losses['loss']:.4f}")

            # Validate
            if epoch % self.cfg.training.get('val_every', 1) == 0:
                val_losses = self.validate()
                print(f"Epoch {epoch} - Val Loss: {val_losses['loss']:.4f}")

                # Log to wandb
                if self.use_wandb:
                    log_dict = {f'val/{k}': v for k, v in val_losses.items()}
                    wandb.log(log_dict, step=self.global_step)

                # Save best model
                is_best = val_losses['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['loss']
                    print(f"New best validation loss: {self.best_val_loss:.4f}")

                # Save checkpoint
                if epoch % self.cfg.training.save_frequency == 0 or is_best:
                    self.save_checkpoint(
                        filename=f'checkpoint_epoch{epoch}.pth',
                        is_best=is_best
                    )

                # Update scheduler (for ReduceLROnPlateau)
                if self.scheduler and self.cfg.training.scheduler.type == 'plateau':
                    self.scheduler.step(val_losses['loss'])

        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"{'='*50}\n")

        if self.use_wandb:
            wandb.finish()


@hydra.main(version_base=None, config_path="../configs/cjepa", config_name="default")
def main(cfg: DictConfig):
    """Main entry point."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Create trainer
    trainer = CJEPATrainer(cfg)

    # Resume from checkpoint if specified
    if cfg.get('resume_from', None):
        trainer.load_checkpoint(cfg.resume_from)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
