"""
Training script for V1 model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List
import sys
import os
import math

# Add parent directory (V1_implementation) to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.v1_model import V1MultimodalTransformer
from training.config import get_default_config
from training.metrics import compute_comprehensive_metrics

class WarmupCosineAnnealingLR:
    """
    Learning rate scheduler that combines warmup with cosine annealing.
    
    During warmup, linearly increases learning rate from 0 to target_lr.
    After warmup, applies cosine annealing from target_lr down to min_lr.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs: Optional[int] = None,
        warmup_ratio: float = 0.075,  # 7.5% of total epochs (middle of 5-10%)
        warmup_steps: Optional[int] = None,
        max_epochs=200,
        target_lr: Optional[float] = None,
        min_lr: Optional[float] = None,
        min_lr_ratio: float = 0.1  # min_lr = target_lr * min_lr_ratio
    ):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.target_lr = target_lr or optimizer.param_groups[0]['lr']
        
        # Calculate warmup epochs dynamically if not provided
        if warmup_epochs is None:
            self.warmup_epochs = max(1, int(warmup_ratio * max_epochs))  # At least 1 epoch
        else:
            self.warmup_epochs = warmup_epochs
        
        self.warmup_steps = warmup_steps
        
        # Calculate min_lr from ratio if not provided
        if min_lr is None:
            self.min_lr = self.target_lr * min_lr_ratio
        else:
            self.min_lr = min_lr
        
        self.current_epoch = 0
        self.current_step = 0
        self.in_warmup = True
        
    def step(self, metrics=None, epoch=None):
        """Update learning rate based on epoch (metrics ignored for cosine annealing)"""
        if epoch is not None:
            self.current_epoch = epoch
        
        # Warmup phase
        if self.in_warmup:
            if self.warmup_steps is not None:
                # Step-based warmup
                self.current_step += 1
                if self.current_step < self.warmup_steps:
                    lr = self.target_lr * (self.current_step / self.warmup_steps)
                    self._set_lr(lr)
                    return
                else:
                    self.in_warmup = False
                    self._set_lr(self.target_lr)
            else:
                # Epoch-based warmup
                if self.current_epoch < self.warmup_epochs:
                    lr = self.target_lr * ((self.current_epoch + 1) / self.warmup_epochs)
                    self._set_lr(lr)
                    return
                else:
                    self.in_warmup = False
                    self._set_lr(self.target_lr)
        
        # Cosine annealing phase
        if self.current_epoch >= self.warmup_epochs:
            # Calculate progress through cosine annealing (0 to 1)
            # Decay spans from warmup_end to max_epochs
            decay_epochs = self.max_epochs - self.warmup_epochs
            if decay_epochs > 0:
                progress = (self.current_epoch - self.warmup_epochs) / decay_epochs
                progress = min(progress, 1.0)  # Clamp to 1.0
                
                # Cosine annealing: lr = min_lr + (target_lr - min_lr) * (1 + cos(π * progress)) / 2
                import math
                lr = self.min_lr + (self.target_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                self._set_lr(lr)
            else:
                # Edge case: warmup_epochs >= max_epochs
                self._set_lr(self.target_lr)
    
    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get the last learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]

class WarmupReduceLROnPlateau:
    """
    Learning rate scheduler that combines warmup with ReduceLROnPlateau.
    
    During warmup, linearly increases learning rate from 0 to target_lr.
    After warmup, uses ReduceLROnPlateau behavior.
    """
    def __init__(
        self,
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        warmup_epochs=5,
        warmup_steps: Optional[int] = None,
        target_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr or optimizer.param_groups[0]['lr']
        self.base_lr = self.target_lr
        
        # ReduceLROnPlateau state
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.cooldown_epochs = 0
        
        # Warmup state
        self.current_epoch = 0
        self.current_step = 0
        self.in_warmup = True
        
    def step(self, metrics, epoch=None):
        """Update learning rate based on metrics (validation loss)"""
        if epoch is not None:
            self.current_epoch = epoch
        
        # Warmup phase
        if self.in_warmup:
            if self.warmup_steps is not None:
                # Step-based warmup
                self.current_step += 1
                if self.current_step < self.warmup_steps:
                    lr = self.target_lr * (self.current_step / self.warmup_steps)
                    self._set_lr(lr)
                    return
                else:
                    self.in_warmup = False
                    self._set_lr(self.target_lr)
            else:
                # Epoch-based warmup
                if self.current_epoch < self.warmup_epochs:
                    lr = self.target_lr * ((self.current_epoch + 1) / self.warmup_epochs)
                    self._set_lr(lr)
                    return
                else:
                    self.in_warmup = False
                    self._set_lr(self.target_lr)
        
        # ReduceLROnPlateau phase
        current = float(metrics)
        if epoch is None:
            epoch = self.current_epoch
        
        if self.best is None:
            self.best = current
        else:
            if self._is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0
    
    def _is_better(self, current, best):
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _reduce_lr(self, epoch):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            print(f"  LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
    
    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        """Get the last learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]


class V1Trainer:
    """Trainer for V1 Multimodal Transformer"""
    
    def __init__(
        self,
        model: V1MultimodalTransformer,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        training_config = config.training
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # self.scheduler = WarmupReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     warmup_epochs=training_config.warmup_epochs,
        #     warmup_steps=training_config.warmup_steps,
        #     target_lr=training_config.learning_rate
        # )

        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=training_config.warmup_epochs,
            warmup_ratio=training_config.warmup_ratio,
            warmup_steps=training_config.warmup_steps,
            max_epochs=training_config.max_epochs,
            target_lr=training_config.learning_rate,
            min_lr_ratio=training_config.min_lr_ratio
        )
        
        self.lambda_slope = training_config.lambda_slope
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_loss_next_visit': [],
            'train_loss_slope': [],
            'val_loss': [],
            'val_loss_next_visit': [],
            'val_loss_slope': [],
            'learning_rate': [],
            'val_metrics': []
        }
        self.best_val_metrics = None
        self.current_val_metrics = None
        
        self.save_dir = Path(config.data.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'loss': 0.0,
            'loss_next_visit': 0.0,
            'loss_slope': 0.0
        }
        
        n_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = self._move_batch_to_device(batch)
                predictions = self._forward_batch(batch)
                targets = self._extract_targets(batch)
                
                losses = self.model.compute_loss(
                    predictions, targets, batch['attention_mask'], self.lambda_slope
                )
                
                self._backward_step(losses)
                
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                
                pbar.set_postfix({
                    'loss': losses['loss'].item(),
                    'next': losses['loss_next_visit'].item(),
                    'slope': losses['loss_slope'].item()
                })
        
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    def _forward_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Run forward pass on a batch."""
        return self.model(
            batch['static_values'], batch['static_mask'],
            batch['motor_values'], batch['motor_mask'],
            batch['nonmotor_values'], batch['nonmotor_mask'],
            batch['med_values'], batch['med_mask'],
            batch['age_at_visit_values'], batch['age_at_visit_mask'],
            batch['time_months'], batch['attention_mask']
        )
    
    def _extract_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Extract target tensors from batch."""
        return {
            'next_visit': batch['next_visit_targets'],
            'next_visit_mask': batch.get('next_visit_label_mask'),
            'slope': batch['slope_targets']
        }
    
    def _backward_step(self, losses: Dict[str, torch.Tensor]):
        """Perform backward pass and optimizer step."""
        self.optimizer.zero_grad()
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        epoch_losses = {
            'loss': 0.0,
            'loss_next_visit': 0.0,
            'loss_slope': 0.0
        }
        
        n_batches = len(self.val_loader)
        
        all_predictions = {'next_visit': [], 'slope': []}
        all_targets = {'next_visit': [], 'next_visit_mask': [], 'slope': []}
        all_attention_masks = []
        all_time_months = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = self._move_batch_to_device(batch)
            predictions = self._forward_batch(batch)
            targets = self._extract_targets(batch)
            
            losses = self.model.compute_loss(
                predictions, targets, batch['attention_mask'], self.lambda_slope
            )
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            self._accumulate_validation_batch(
                predictions, targets, batch, all_predictions, all_targets,
                all_attention_masks, all_time_months
            )
        
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        predictions_cat, targets_cat, attention_mask_cat, time_months_cat = \
            self._concatenate_validation_results(
                all_predictions, all_targets, all_attention_masks, all_time_months
            )
        
        val_metrics = self._compute_validation_metrics(
            predictions_cat, targets_cat, attention_mask_cat, time_months_cat
        )
        self.current_val_metrics = val_metrics
        
        return epoch_losses
    
    def _accumulate_validation_batch(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict,
        all_predictions: Dict[str, List],
        all_targets: Dict[str, List],
        all_attention_masks: List,
        all_time_months: List
    ):
        """Accumulate batch results for validation metrics."""
        all_predictions['next_visit'].append(predictions['next_visit'].detach().cpu())
        all_predictions['slope'].append(predictions['slope'].detach().cpu())
        all_targets['next_visit'].append(targets['next_visit'].detach().cpu())
        if targets['next_visit_mask'] is not None:
            all_targets['next_visit_mask'].append(targets['next_visit_mask'].detach().cpu())
        all_targets['slope'].append(targets['slope'].detach().cpu())
        all_attention_masks.append(batch['attention_mask'].detach().cpu())
        all_time_months.append(batch['time_months'].detach().cpu())
        
    def _concatenate_validation_results(
        self,
        all_predictions: Dict[str, List],
        all_targets: Dict[str, List],
        all_attention_masks: List,
        all_time_months: List
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Concatenate accumulated validation batch results."""
        predictions_cat = {
            'next_visit': torch.cat(all_predictions['next_visit'], dim=0),
            'slope': torch.cat(all_predictions['slope'], dim=0),
        }
        targets_cat = {
            'next_visit': torch.cat(all_targets['next_visit'], dim=0),
            'slope': torch.cat(all_targets['slope'], dim=0),
            'next_visit_mask': (
                torch.cat(all_targets['next_visit_mask'], dim=0)
                if all_targets['next_visit_mask'] else None
            ),
        }
        attention_mask_cat = torch.cat(all_attention_masks, dim=0)
        time_months_cat = torch.cat(all_time_months, dim=0)
        return predictions_cat, targets_cat, attention_mask_cat, time_months_cat
    
    def _compute_validation_metrics(
        self,
        predictions_cat: Dict[str, torch.Tensor],
        targets_cat: Dict[str, torch.Tensor],
        attention_mask_cat: torch.Tensor,
        time_months_cat: torch.Tensor
    ) -> Dict:
        """Compute comprehensive validation metrics."""
        target_names = getattr(
            self.config.features, 'all_updrs_totals', ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        )
        return compute_comprehensive_metrics(
            predictions_cat, targets_cat, attention_mask_cat,
            target_names=target_names, time_months=time_months_cat
        )
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        scheduler_state = self._get_scheduler_state()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'best_epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state': scheduler_state,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
            'val_metrics': self.current_val_metrics,
            'best_val_metrics': self.best_val_metrics,
        }
        
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pt')
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def _get_scheduler_state(self) -> Dict:
        """Extract scheduler state for checkpoint saving."""
        try:
            scheduler_state = (
                self.scheduler.state_dict()
                if hasattr(self.scheduler, 'state_dict')
                else {
                    'current_epoch': self.scheduler.current_epoch,
                    'current_step': self.scheduler.current_step,
                    'in_warmup': self.scheduler.in_warmup,
                    'best': self.scheduler.best,
                    'num_bad_epochs': self.scheduler.num_bad_epochs
                }
            )
        except:
            scheduler_state = {
                'current_epoch': getattr(self.scheduler, 'current_epoch', 0),
                'current_step': getattr(self.scheduler, 'current_step', 0),
                'in_warmup': getattr(self.scheduler, 'in_warmup', False),
                'best': getattr(self.scheduler, 'best', None),
                'num_bad_epochs': getattr(self.scheduler, 'num_bad_epochs', 0)
            }
        return scheduler_state
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only = False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._load_scheduler_state(checkpoint)
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])
        self.training_history = checkpoint['training_history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def _load_scheduler_state(self, checkpoint: Dict):
        """Load scheduler state from checkpoint."""
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                pass
        elif 'scheduler_state' in checkpoint:
            scheduler_state = checkpoint['scheduler_state']
            if hasattr(self.scheduler, 'load_state_dict'):
                try:
                    self.scheduler.load_state_dict(scheduler_state)
                except:
                    self._restore_scheduler_state_manually(scheduler_state)
    
    def _restore_scheduler_state_manually(self, scheduler_state: Dict):
        """Manually restore scheduler state attributes."""
        self.scheduler.current_epoch = scheduler_state.get('current_epoch', 0)
        self.scheduler.current_step = scheduler_state.get('current_step', 0)
        self.scheduler.in_warmup = scheduler_state.get('in_warmup', False)
        self.scheduler.best = scheduler_state.get('best', None)
        self.scheduler.num_bad_epochs = scheduler_state.get('num_bad_epochs', 0)
    
    def train(self, max_epochs: int, early_stopping_patience: int = 15):
        """Main training loop."""
        print("=" * 80)
        print("Starting training")
        print("=" * 80)
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            train_losses = self.train_epoch()
            val_losses = self.validate()
            self.scheduler.step(val_losses['loss'], epoch=epoch)
            
            self._update_training_history(train_losses, val_losses)
            is_best = self._check_and_update_best(val_losses, epoch)
            self._print_epoch_summary(epoch, train_losses, val_losses, is_best, early_stopping_patience)
            
            self.save_checkpoint(is_best)
            
            if self.patience_counter >= early_stopping_patience:
                self._print_early_stopping_message(epoch, early_stopping_patience)
                break
        
        if self.current_epoch == max_epochs - 1:
            self._print_max_epochs_reached(max_epochs, early_stopping_patience)
        
        self._print_training_complete()
        self._save_training_history()
    
    def _update_training_history(self, train_losses: Dict[str, float], val_losses: Dict[str, float]):
        """Update training history with epoch losses."""
        self.training_history['train_loss'].append(train_losses['loss'])
        self.training_history['train_loss_next_visit'].append(train_losses['loss_next_visit'])
        self.training_history['train_loss_slope'].append(train_losses['loss_slope'])
        self.training_history['val_loss'].append(val_losses['loss'])
        self.training_history['val_loss_next_visit'].append(val_losses['loss_next_visit'])
        self.training_history['val_loss_slope'].append(val_losses['loss_slope'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        if self.current_val_metrics is not None:
            metrics_serializable = self._serialize_metrics(self.current_val_metrics)
            self.training_history['val_metrics'].append(metrics_serializable)

    def _serialize_metrics(self, metrics: Dict) -> Dict:
        """Convert metrics dictionary to JSON-serializable format."""
        def convert_value(v):
            if isinstance(v, (torch.Tensor, np.ndarray)):
                return float(v.item()) if v.numel() == 1 else v.tolist()
            elif isinstance(v, (int, float, str, bool, type(None))):
                return v
            elif isinstance(v, dict):
                return {k: convert_value(v2) for k, v2 in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            else:
                return str(v)

        return convert_value(metrics)

    def _check_and_update_best(self, val_losses: Dict[str, float], epoch: int) -> bool:
        """Check if current epoch is best and update state."""
        is_best = val_losses['loss'] < self.best_val_loss
        if is_best:
            self.best_val_loss = val_losses['loss']
            self.best_epoch = epoch
            self.best_val_metrics = self.current_val_metrics
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return is_best
    
    def _print_epoch_summary(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        is_best: bool,
        early_stopping_patience: int
    ):
        """Print epoch training summary."""
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_losses['loss']:.4f} "
              f"(next: {train_losses['loss_next_visit']:.4f}, "
              f"slope: {train_losses['loss_slope']:.4f})")
        print(f"  Val Loss:   {val_losses['loss']:.4f} "
              f"(next: {val_losses['loss_next_visit']:.4f}, "
              f"slope: {val_losses['loss_slope']:.4f})")
        warmup_status = " [WARMUP]" if getattr(self.scheduler, 'in_warmup', False) else ""
        print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}{warmup_status}")
        if is_best:
            print(f"  ✓ New best! (patience: {self.patience_counter}/{early_stopping_patience})")
        else:
            print(f"  No improvement (patience: {self.patience_counter}/{early_stopping_patience})")
            
    def _print_early_stopping_message(self, epoch: int, early_stopping_patience: int):
        """Print early stopping message."""
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        print(f"  No improvement for {early_stopping_patience} epochs")
        print(f"  Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        
    def _print_max_epochs_reached(self, max_epochs: int, early_stopping_patience: int):
        """Print message when max epochs reached."""
        print(f"\nReached maximum epochs ({max_epochs})")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
        if self.patience_counter > 0:
            print(f"  No improvement for {self.patience_counter} epochs (patience: {early_stopping_patience})")
        
    def _print_training_complete(self):
        """Print training completion message."""
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        print(f"Total epochs trained: {len(self.training_history['train_loss'])}")
        print("=" * 80)
        
    def _save_training_history(self):
        """Save training history to JSON file."""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
