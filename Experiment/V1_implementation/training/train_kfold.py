"""
Example script for k-fold cross-validation training
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.v1_model import V1MultimodalTransformer
from training.config import get_default_config
from training.train import V1Trainer
from data.data_integrator import DataIntegrator
from data.dataset import create_kfold_dataloaders


def train_kfold(config=None, n_splits=5, n_epochs=50, device='cuda'):
    """
    Train model using k-fold cross-validation
    
    Args:
        config: Configuration object (if None, uses default)
        n_splits: Number of folds (default: 5)
        n_epochs: Number of epochs per fold (default: 50)
        device: Device to use ('cuda' or 'cpu')
    """
    if config is None:
        config = get_default_config()
    
    print("=" * 80)
    print("K-FOLD CROSS-VALIDATION TRAINING")
    print("=" * 80)
    print(f"Folds: {n_splits}")
    print(f"Epochs per fold: {n_epochs}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Prepare data
    print("\n--- Preparing Data ---")
    integrator = DataIntegrator(config, normalize_features=True)
    prepared_data = integrator.prepare_final_dataset()
    
    # Create k-fold dataloaders
    print("\n--- Creating K-Fold Dataloaders ---")
    fold_dataloaders, test_loader = create_kfold_dataloaders(
        prepared_data=prepared_data,
        config=config,
        n_splits=n_splits,
        test_ratio=0.2,
        random_seed=42
    )
    
    # Train on each fold
    fold_results = []
    
    for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print("=" * 80)
        
        # Create model for this fold
        model = V1MultimodalTransformer(config)
        
        # Create trainer
        trainer = V1Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Train
        trainer.train(max_epochs=n_epochs, early_stopping_patience=15)
        
        # Evaluate on validation set
        val_metrics = trainer.validate()
        
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_loss': trainer.best_val_loss,
            'val_loss': val_metrics['loss'],
            'val_loss_next_visit': val_metrics['loss_next_visit'],
            'val_loss_slope': val_metrics['loss_slope']
        })
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")
        print(f"  Final Val Loss: {val_metrics['loss']:.4f}")
        print(f"    - Next Visit: {val_metrics['loss_next_visit']:.4f}")
        print(f"    - Slope: {val_metrics['loss_slope']:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    avg_val_loss = sum(r['val_loss'] for r in fold_results) / len(fold_results)
    avg_val_next_visit = sum(r['val_loss_next_visit'] for r in fold_results) / len(fold_results)
    avg_val_slope = sum(r['val_loss_slope'] for r in fold_results) / len(fold_results)
    
    std_val_loss = torch.std(torch.tensor([r['val_loss'] for r in fold_results])).item()
    
    print(f"\nAverage Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"  - Next Visit: {avg_val_next_visit:.4f}")
    print(f"  - Slope: {avg_val_slope:.4f}")
    
    print("\nPer-fold results:")
    for result in fold_results:
        print(f"  Fold {result['fold']}: Val Loss = {result['val_loss']:.4f}")
    
    # Final evaluation on test set (optional)
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    print("Note: For proper test evaluation, you should:")
    print("  1. Retrain on ALL training data (all folds combined)")
    print("  2. Fit scalers on all training data")
    print("  3. Evaluate on test set")
    print("=" * 80)
    
    return fold_results


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    
    # Train with 5-fold CV
    results = train_kfold(
        config=config,
        n_splits=5,
        n_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
