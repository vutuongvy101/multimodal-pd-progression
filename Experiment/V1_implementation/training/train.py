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
from typing import Dict, Tuple
import sys
import os

# Add parent directory (V1_implementation) to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.v1_model import V1MultimodalTransformer
from training.config import get_default_config


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
        
        training_config = config['training']
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss weights
        self.lambda_slope = training_config.lambda_slope
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_loss_next_visit': [],
            'train_loss_slope': [],
            'val_loss': [],
            'val_loss_next_visit': [],
            'val_loss_slope': [],
            'learning_rate': []
        }
        
        # Create save directory
        self.save_dir = Path(config['data'].model_save_dir)
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
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(
                    batch['static_values'],
                    batch['static_mask'],
                    batch['motor_values'],
                    batch['motor_mask'],
                    batch['nonmotor_values'],
                    batch['nonmotor_mask'],
                    batch['med_values'],
                    batch['med_mask'],
                    batch['time_months'],
                    batch['attention_mask']
                )
                
                # Compute loss
                targets = {
                    'next_visit': batch['next_visit_targets'],
                    'slope': batch['slope_targets']
                }
                
                losses = self.model.compute_loss(
                    predictions,
                    targets,
                    batch['attention_mask'],
                    self.lambda_slope
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update metrics
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': losses['loss'].item(),
                    'next': losses['loss_next_visit'].item(),
                    'slope': losses['loss_slope'].item()
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
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
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                batch['static_values'],
                batch['static_mask'],
                batch['motor_values'],
                batch['motor_mask'],
                batch['nonmotor_values'],
                batch['nonmotor_mask'],
                batch['med_values'],
                batch['med_mask'],
                batch['time_months'],
                batch['attention_mask']
            )
            
            # Compute loss
            targets = {
                'next_visit': batch['next_visit_targets'],
                'slope': batch['slope_targets']
            }
            
            losses = self.model.compute_loss(
                predictions,
                targets,
                batch['attention_mask'],
                self.lambda_slope
            )
            
            # Update metrics
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_checkpoint.pt')
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, max_epochs: int, early_stopping_patience: int = 15):
        """Main training loop"""
        print("=" * 80)
        print("Starting training")
        print("=" * 80)
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_losses['loss'])
            
            # Update history
            self.training_history['train_loss'].append(train_losses['loss'])
            self.training_history['train_loss_next_visit'].append(train_losses['loss_next_visit'])
            self.training_history['train_loss_slope'].append(train_losses['loss_slope'])
            self.training_history['val_loss'].append(val_losses['loss'])
            self.training_history['val_loss_next_visit'].append(val_losses['loss_next_visit'])
            self.training_history['val_loss_slope'].append(val_losses['loss_slope'])
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_losses['loss']:.4f} "
                  f"(next: {train_losses['loss_next_visit']:.4f}, "
                  f"slope: {train_losses['loss_slope']:.4f})")
            print(f"  Val Loss:   {val_losses['loss']:.4f} "
                  f"(next: {val_losses['loss_next_visit']:.4f}, "
                  f"slope: {val_losses['loss_slope']:.4f})")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check for improvement
            is_best = val_losses['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 80)
        
        # Save training history
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)


if __name__ == "__main__":
    print("V1 Model Training Script")
    print("=" * 80)
    
    # Load config
    config = get_default_config()
    
    # Create model
    print("\nCreating model...")
    model = V1MultimodalTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    print("\nLoading data...")
    print("WARNING: Data loading not yet implemented")
    print("You need to:")
    print("  1. Prepare your data using data_preparation.py")
    print("  2. Create train/val/test dataloaders")
    print("  3. Pass them to the Trainer")
    
    # Example usage (once data is ready):
    # from data.dataset import create_dataloaders
    # train_loader, val_loader, test_loader = create_dataloaders(prepared_data, config)
    # 
    # trainer = V1Trainer(model, config, train_loader, val_loader, device='cuda')
    # trainer.train(max_epochs=100, early_stopping_patience=15)
