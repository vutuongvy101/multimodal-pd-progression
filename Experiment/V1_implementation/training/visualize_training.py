"""
Visualization utilities for training metrics.
Plots training loss, validation loss, and accuracy (R²) across epochs.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history_file: str,
    output_dir: Optional[str] = None,
    show_plot: bool = True,
    figsize: tuple = (14, 10)
):
    """
    Plot training history from training_history.json file.
    
    Args:
        history_file: Path to training_history.json file
        output_dir: Directory to save plots (default: same as history_file)
        show_plot: Whether to display the plot
        figsize: Figure size for matplotlib
    """
    history_path = Path(history_file)
    
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = history_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = np.arange(len(history['train_loss']))
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs, history['val_loss'], label='Val Loss', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Overall Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Next-Visit Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_loss_next_visit'], label='Train Next-Visit', marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs, history['val_loss_next_visit'], label='Val Next-Visit', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Next-Visit Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Slope Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_loss_slope'], label='Train Slope', marker='o', markersize=3, alpha=0.7)
    ax.plot(epochs, history['val_loss_slope'], label='Val Slope', marker='s', markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Slope Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy (R²)
    ax = axes[1, 1]
    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
        ax.plot(epochs, history['val_accuracy'], label='Val R² (Accuracy)', 
                marker='o', markersize=5, alpha=0.7, linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('R² Score')
        ax.set_title('Validation Accuracy (R²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
    else:
        ax.text(0.5, 0.5, 'Accuracy metrics\nnot available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(output_dir) / 'training_metrics.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_file}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def print_training_summary(history_file: str):
    """
    Print summary statistics of training.
    
    Args:
        history_file: Path to training_history.json file
    """
    history_path = Path(history_file)
    
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    n_epochs = len(history['train_loss'])
    print(f"\nTotal Epochs: {n_epochs}")
    
    # Loss statistics
    train_loss_final = history['train_loss'][-1]
    val_loss_final = history['val_loss'][-1]
    val_loss_best = min(history['val_loss'])
    val_loss_best_epoch = history['val_loss'].index(val_loss_best)
    
    print(f"\nLoss Statistics:")
    print(f"  Final Train Loss: {train_loss_final:.4f}")
    print(f"  Final Val Loss: {val_loss_final:.4f}")
    print(f"  Best Val Loss: {val_loss_best:.4f} (Epoch {val_loss_best_epoch})")
    print(f"  Loss Improvement: {((val_loss_final - val_loss_best) / val_loss_best * 100):.2f}%")
    
    # Accuracy statistics
    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
        val_acc_final = history['val_accuracy'][-1]
        val_acc_best = max(history['val_accuracy'])
        val_acc_best_epoch = history['val_accuracy'].index(val_acc_best)
        
        print(f"\nAccuracy (R²) Statistics:")
        print(f"  Final Val R²: {val_acc_final:.4f}")
        print(f"  Best Val R²: {val_acc_best:.4f} (Epoch {val_acc_best_epoch})")
        print(f"  Accuracy Improvement: {((val_acc_best - val_acc_final) / max(abs(val_acc_final), 0.01) * 100):.2f}%")
    
    # Learning rate
    lr_initial = history['learning_rate'][0]
    lr_final = history['learning_rate'][-1]
    print(f"\nLearning Rate:")
    print(f"  Initial: {lr_initial:.2e}")
    print(f"  Final: {lr_final:.2e}")
    
    print("=" * 80 + "\n")


def plot_kfold_results(kfold_results_file: str, output_dir: Optional[str] = None):
    """
    Plot k-fold cross-validation results.
    
    Args:
        kfold_results_file: Path to kfold_results.json file
        output_dir: Directory to save plots (default: same as results_file)
    """
    results_path = Path(kfold_results_file)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Set output directory
    if output_dir is None:
        output_dir = results_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract fold results
    fold_results = results.get('fold_results', [])
    summary = results.get('summary', {})
    
    if not fold_results:
        print("No fold results found")
        return
    
    n_folds = len(fold_results)
    fold_indices = np.arange(1, n_folds + 1)
    
    # Extract metrics per fold
    val_losses = [r['val_loss'] for r in fold_results]
    val_accuracies = [r.get('val_accuracy', 0.0) for r in fold_results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'K-Fold Cross-Validation Results (n_splits={n_folds})', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Validation Loss per Fold
    ax = axes[0]
    ax.bar(fold_indices, val_losses, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(summary.get('mean_val_loss', 0), color='red', linestyle='--', 
               label=f"Mean: {summary.get('mean_val_loss', 0):.4f}")
    ax.set_xlabel('Fold')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss by Fold')
    ax.set_xticks(fold_indices)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Validation Accuracy (R²) per Fold
    ax = axes[1]
    if any(v > 0 for v in val_accuracies):
        ax.bar(fold_indices, val_accuracies, alpha=0.7, color='green', edgecolor='black')
        ax.axhline(summary.get('mean_val_r2', 0), color='red', linestyle='--',
                   label=f"Mean: {summary.get('mean_val_r2', 0):.4f}")
        ax.set_xlabel('Fold')
        ax.set_ylabel('R² Score')
        ax.set_title('Validation Accuracy (R²) by Fold')
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Accuracy metrics\nnot available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path(output_dir) / 'kfold_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved k-fold plot to {plot_file}")
    
    plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--history', type=str, help='Path to training_history.json')
    parser.add_argument('--kfold', type=str, help='Path to kfold_results.json')
    parser.add_argument('--output-dir', type=str, help='Output directory for plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    if args.history:
        print_training_summary(args.history)
        plot_training_history(args.history, args.output_dir, show_plot=not args.no_show)
    
    if args.kfold:
        plot_kfold_results(args.kfold, args.output_dir)
