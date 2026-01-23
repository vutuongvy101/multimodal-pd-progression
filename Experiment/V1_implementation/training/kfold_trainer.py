"""
K-Fold Cross-Validation Trainer for V1 model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from models.v1_model import V1MultimodalTransformer
from training.train import V1Trainer
from training.metrics import compute_comprehensive_metrics


class KFoldTrainer:
    """
    K-Fold Cross-Validation trainer that orchestrates training across multiple folds.
    
    Each fold gets its own model and trainer instance. Results are aggregated
    across folds to provide robust performance estimates.
    """
    
    def __init__(
        self,
        config,
        prepared_data: Dict,
        n_splits: int = 5,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        device: str = 'cuda',
        num_workers: int = 0
    ):
        """
        Args:
            config: Configuration object
            prepared_data: Output from DataIntegrator.prepare_final_dataset() (DataFrames)
            n_splits: Number of CV folds
            test_ratio: Proportion of data to hold out as test set
            random_seed: Random seed for reproducibility
            device: Device to use for training
            num_workers: Number of DataLoader workers
        """
        self.config = config
        self.prepared_data = prepared_data
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.device = device
        self.num_workers = num_workers
        
        self.fold_results: List[Dict] = []
        self.test_loader: Optional[DataLoader] = None
        
        self.save_dir = Path(config.data.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_fold_dataloaders(self):
        """Create k-fold dataloaders using dataset utility"""
        from data.dataset import create_kfold_dataloaders
        
        fold_dataloaders, test_loader = create_kfold_dataloaders(
            prepared_data=self.prepared_data,
            config=self.config,
            n_splits=self.n_splits,
            num_workers=self.num_workers,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed
        )
        
        return fold_dataloaders, test_loader
    
    def train_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_fold_checkpoint: bool = True
    ) -> Dict[str, float]:
        """
        Train a single fold.
        
        Args:
            fold_idx: Index of the fold (0-indexed)
            train_loader: Training dataloader for this fold
            val_loader: Validation dataloader for this fold
            save_fold_checkpoint: Whether to save checkpoint for this fold
            
        Returns:
            Dictionary with fold results (val_loss, best_val_loss, etc.)
        """
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_idx + 1}/{self.n_splits}")
        print(f"{'=' * 80}")
        
        model = V1MultimodalTransformer(self.config)
        trainer = self._create_fold_trainer(fold_idx, model, train_loader, val_loader)
        
        max_epochs = self.config.training.max_epochs
        patience = self.config.training.early_stopping_patience
        trainer.train(max_epochs=max_epochs, early_stopping_patience=patience)
        
        val_metrics = trainer.validate()
        
        if save_fold_checkpoint:
            fold_save_dir = self.save_dir / f"fold_{fold_idx + 1}"
            fold_save_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(is_best=True)
        
        fold_result = self._extract_fold_results(fold_idx, trainer, val_metrics)
        self._print_fold_results(fold_idx, trainer, val_metrics)
        return fold_result
    
    def train(self, save_fold_checkpoints: bool = True) -> Dict:
        """
        Run k-fold cross-validation training.
        
        Args:
            save_fold_checkpoints: Whether to save checkpoints for each fold
            
        Returns:
            Dictionary with aggregated results and per-fold metrics
        """
        self._print_training_header()
        
        print("\n--- Creating K-Fold Dataloaders ---")
        fold_dataloaders, test_loader = self._create_fold_dataloaders()
        self.test_loader = test_loader
        print(f"  ✓ Created {self.n_splits} folds")
        print(f"  ✓ Test set: {len(test_loader.dataset)} patients (held out)")
        
        self.fold_results = []
        for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
            fold_result = self.train_fold(
                fold_idx=fold_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                save_fold_checkpoint=save_fold_checkpoints
            )
            self.fold_results.append(fold_result)
        
        summary = self._aggregate_results()
        self._print_summary(summary)
        self._save_results(summary)
        
        return {
            'fold_results': self.fold_results,
            'summary': summary,
            'test_loader': test_loader
        }
    
    def _print_training_header(self):
        """Print training configuration header."""
        print("=" * 80)
        print("K-FOLD CROSS-VALIDATION TRAINING")
        print("=" * 80)
        print(f"Folds: {self.n_splits}")
        print(f"Test ratio: {self.test_ratio}")
        print(f"Epochs per fold: {self.config.training.max_epochs}")
        print(f"Early stopping patience: {self.config.training.early_stopping_patience}")
        print(f"Device: {self.device}")
        print("=" * 80)
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across folds."""
        val_losses = [r['val_loss'] for r in self.fold_results]
        val_next_visit = [r['val_loss_next_visit'] for r in self.fold_results]
        val_slope = [r['val_loss_slope'] for r in self.fold_results]
        best_val_losses = [r['best_val_loss'] for r in self.fold_results]
        
        summary = {
            'mean_val_loss': float(torch.tensor(val_losses).mean().item()),
            'std_val_loss': float(torch.tensor(val_losses).std().item()),
            'mean_val_next_visit': float(torch.tensor(val_next_visit).mean().item()),
            'std_val_next_visit': float(torch.tensor(val_next_visit).std().item()),
            'mean_val_slope': float(torch.tensor(val_slope).mean().item()),
            'std_val_slope': float(torch.tensor(val_slope).std().item()),
            'mean_best_val_loss': float(torch.tensor(best_val_losses).mean().item()),
            'std_best_val_loss': float(torch.tensor(best_val_losses).std().item()),
        }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print aggregated results summary."""
        print("\n" + "=" * 80)
        print("K-FOLD CROSS-VALIDATION RESULTS")
        print("=" * 80)
        
        print(f"\nAverage Validation Loss: {summary['mean_val_loss']:.4f} ± {summary['std_val_loss']:.4f}")
        print(f"  - Next Visit: {summary['mean_val_next_visit']:.4f} ± {summary['std_val_next_visit']:.4f}")
        print(f"  - Slope: {summary['mean_val_slope']:.4f} ± {summary['std_val_slope']:.4f}")
        
        print(f"\nAverage Best Validation Loss: {summary['mean_best_val_loss']:.4f} ± {summary['std_best_val_loss']:.4f}")
        
        print("\nPer-fold results:")
        for result in self.fold_results:
            print(f"  Fold {result['fold']}: Val Loss = {result['val_loss']:.4f} "
                  f"(Best: {result['best_val_loss']:.4f} at epoch {result['best_epoch']})")
        
        print("\n" + "=" * 80)
        print("TEST SET EVALUATION")
        print("=" * 80)
        print("Note: Test set is held out. For final evaluation:")
        print("  1. Use train_full mode to train on ALL CV training data")
        print("  2. Evaluate on the held-out test set")
        print("=" * 80)
    
    def _save_results(self, summary: Dict):
        """Save CV results to JSON."""
        results = {
            'n_splits': self.n_splits,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'fold_results': self.fold_results,
            'summary': summary
        }
        
        results_path = self.save_dir / 'kfold_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Saved CV results to {results_path}")
    
    def evaluate_test(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate on test set using one of the trained fold models.
        
        Note: For proper test evaluation, you should use train_full mode
        to train on ALL CV training data, then evaluate on test.
        
        Args:
            model_path: Path to a trained model checkpoint. If None, uses
                       the best fold's checkpoint (fold with lowest val loss).
        
        Returns:
            Test set metrics
        """
        if self.test_loader is None:
            raise ValueError("Must call train() first to create test loader")
        
        model_path = self._resolve_model_path(model_path)
        model = self._load_model_for_evaluation(model_path)
        
        test_losses, all_predictions, all_targets, all_attention_masks, all_time_months = \
            self._evaluate_test_set(model)
        
        predictions_cat, targets_cat, attention_mask_cat, time_months_cat = \
            self._concatenate_test_results(all_predictions, all_targets, all_attention_masks, all_time_months)
        
        metrics = self._compute_test_metrics(predictions_cat, targets_cat, attention_mask_cat, time_months_cat)
        
        self._print_test_results(test_losses, metrics)
        self._save_test_metrics(test_losses, metrics, model_path)
        
        return {'losses': test_losses, 'metrics': metrics}
    
    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        """Resolve model path, defaulting to best fold if not provided."""
        if model_path is None:
            best_fold_idx = min(
                range(len(self.fold_results)),
                key=lambda i: self.fold_results[i]['best_val_loss']
            )
            fold_num = best_fold_idx + 1
            model_path = self.save_dir / f"fold_{fold_num}" / "best_checkpoint.pt"
            print(f"\nUsing best fold model (fold {fold_num}) for test evaluation")
        return Path(model_path)
    
    def _load_model_for_evaluation(self, model_path: Path) -> V1MultimodalTransformer:
        """Load model from checkpoint for evaluation."""
        model = V1MultimodalTransformer(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _evaluate_test_set(
        self,
        model: V1MultimodalTransformer
    ) -> Tuple[Dict[str, float], Dict[str, List], Dict[str, List], List, List]:
        """Evaluate model on test set and accumulate predictions/targets."""
        test_losses = {'loss': 0.0, 'loss_next_visit': 0.0, 'loss_slope': 0.0}
        n_batches = len(self.test_loader)
        
        all_predictions = {'next_visit': [], 'slope': []}
        all_targets = {'next_visit': [], 'next_visit_mask': [], 'slope': []}
        all_attention_masks = []
        all_time_months = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Test Evaluation"):
                batch = self._move_batch_to_device(batch)
                predictions = self._forward_batch(model, batch)
                targets = self._extract_targets(batch)
                
                losses = model.compute_loss(
                    predictions, targets, batch['attention_mask'], self.config.training.lambda_slope
                )
                
                for key in test_losses:
                    test_losses[key] += losses[key].item()
                
                self._accumulate_batch_results(
                    predictions, targets, batch, all_predictions, all_targets,
                    all_attention_masks, all_time_months
                )
        
        for key in test_losses:
            test_losses[key] /= max(n_batches, 1)
        
        return test_losses, all_predictions, all_targets, all_attention_masks, all_time_months
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    def _forward_batch(self, model: V1MultimodalTransformer, batch: Dict) -> Dict[str, torch.Tensor]:
        """Run forward pass on a batch."""
        return model(
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
    
    def _accumulate_batch_results(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict,
        all_predictions: Dict[str, List],
        all_targets: Dict[str, List],
        all_attention_masks: List,
        all_time_months: List
    ):
        """Accumulate batch results for metric computation."""
        all_predictions['next_visit'].append(predictions['next_visit'].detach().cpu())
        all_predictions['slope'].append(predictions['slope'].detach().cpu())
        all_targets['next_visit'].append(targets['next_visit'].detach().cpu())
        if targets['next_visit_mask'] is not None:
            all_targets['next_visit_mask'].append(targets['next_visit_mask'].detach().cpu())
        all_targets['slope'].append(targets['slope'].detach().cpu())
        all_attention_masks.append(batch['attention_mask'].detach().cpu())
        all_time_months.append(batch['time_months'].detach().cpu())
    
    def _concatenate_test_results(
        self,
        all_predictions: Dict[str, List],
        all_targets: Dict[str, List],
        all_attention_masks: List,
        all_time_months: List
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Concatenate accumulated batch results."""
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
    
    def _compute_test_metrics(
        self,
        predictions_cat: Dict[str, torch.Tensor],
        targets_cat: Dict[str, torch.Tensor],
        attention_mask_cat: torch.Tensor,
        time_months_cat: torch.Tensor
    ) -> Dict:
        """Compute comprehensive metrics on test set."""
        target_names = getattr(
            self.config.features, 'all_updrs_totals', ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        )
        return compute_comprehensive_metrics(
            predictions_cat, targets_cat, attention_mask_cat,
            target_names=target_names, time_months=time_months_cat
        )
    
    def _print_test_results(self, test_losses: Dict[str, float], metrics: Dict):
        """Print test set evaluation results."""
        print(f"\nTest Set Results (losses):")
        print(f"  Loss: {test_losses['loss']:.4f}")
        print(f"    - Next Visit: {test_losses['loss_next_visit']:.4f}")
        print(f"    - Slope: {test_losses['loss_slope']:.4f}")
        
        print("\nPer-UPDRS metrics (test set):")
        for name, m in metrics['next_visit'].items():
            print(
                f"  {name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
                f"R2={m['r2']:.4f}, Pearson={m['correlation']:.4f}, "
                f"Spearman={m.get('spearman', float('nan')):.4f}, n={m['n_samples']}"
            )
            if m.get('delta_t_buckets'):
                print(f"    Per-Δt buckets:")
                for bucket_name, bucket_metrics in m['delta_t_buckets'].items():
                    if bucket_metrics['n_samples'] > 0:
                        print(
                            f"      {bucket_name}: MAE={bucket_metrics['mae']:.4f}, "
                            f"RMSE={bucket_metrics['rmse']:.4f}, n={bucket_metrics['n_samples']}"
                        )
        
        if "next_visit_overall" in metrics:
            overall = metrics["next_visit_overall"]
            print(f"\nNext-visit overall: Macro MAE={overall.get('macro_avg_mae', float('nan')):.4f}, "
                  f"Weighted MAE={overall.get('weighted_avg_mae', float('nan')):.4f}, "
                  f"Total samples={overall.get('total_samples', 0)}")
        
        print("\nPer-target slope metrics:")
        for name, m in metrics.get("slope", {}).items():
            if m.get('n_samples', 0) > 0:
                print(
                    f"  {name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
                    f"Spearman={m.get('spearman', float('nan')):.4f}, n={m['n_samples']}"
                )
        
        if "slope_overall" in metrics:
            slope_overall = metrics["slope_overall"]
            print(
                f"\nSlope overall: MAE={slope_overall['mae']:.4f}, "
                f"RMSE={slope_overall['rmse']:.4f}, "
                f"Spearman={slope_overall.get('spearman', float('nan')):.4f}, "
                f"n={slope_overall['n_samples']}"
            )
    
    def _save_test_metrics(self, test_losses: Dict[str, float], metrics: Dict, model_path: Path):
        """Save test metrics to JSON file."""
        results_path = self.save_dir / 'test_metrics_best_fold.json'
        to_save = {
            'losses': test_losses,
            'metrics': metrics,
            'model_path': str(model_path),
        }
        with open(results_path, 'w') as f:
            json.dump(to_save, f, indent=2)
        print(f"\n✓ Saved test metrics to {results_path}")

        return {'losses': test_losses, 'metrics': metrics}
