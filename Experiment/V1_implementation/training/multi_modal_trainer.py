"""
Multi-Modal Ablation Trainer
Trains multiple models with different modality combinations for comparison.
"""

import copy
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from training.kfold_trainer import KFoldTrainer
from data.data_integrator import DataIntegrator


class MultiModalTrainer:
    """
    Orchestrates training of multiple models with different modality combinations.
    
    Each modality configuration gets its own directory with checkpoints and logs.
    Already-trained configurations are skipped unless --force-retrain is used.
    """
    
    # Predefined modality sets for common ablation studies
    MODALITY_SETS = {
        'all': ['static', 'motor', 'non_motor', 'medication', 'age_at_visit'],
        'static+motor': ['static', 'motor'],
        'static+nonmotor': ['static', 'non_motor'],
        'static+medication': ['static', 'medication'],
        'motor+nonmotor': ['motor', 'non_motor'],
        'motor_only': ['motor'],
        'static_only': ['static'],
        'no_static': ['motor', 'non_motor', 'medication', 'age_at_visit'],
        'no_motor': ['static', 'non_motor', 'medication', 'age_at_visit'],
        'no_nonmotor': ['static', 'motor', 'medication', 'age_at_visit'],
    }
    
    def __init__(
        self,
        config,
        prepared_data: Dict,
        base_save_dir: Optional[str] = None,
        n_splits: int = 5,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        device: str = 'cuda',
        num_workers: int = 0
    ):
        """
        Args:
            config: Base configuration object (will be cloned per modality set)
            prepared_data: Output from DataIntegrator.prepare_final_dataset()
            base_save_dir: Base directory for saving models (default: config.data.model_save_dir)
            n_splits: Number of CV folds per model
            test_ratio: Test set ratio
            random_seed: Random seed
            device: Device to use
            num_workers: DataLoader workers
        """
        self.base_config = config
        self.prepared_data = prepared_data
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.device = device
        self.num_workers = num_workers
        
        # Setup save directory
        if base_save_dir is None:
            base_save_dir = config.data.model_save_dir
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.training_results: Dict[str, Dict] = {}
        # Store kfold_trainer instances for test evaluation
        self.kfold_trainers: Dict[str, KFoldTrainer] = {}
    
    @staticmethod
    def get_modality_key(modalities: List[str]) -> str:
        """
        Generate a consistent key for a modality combination.
        
        Args:
            modalities: List of modality names
            
        Returns:
            String key like "static+motor+nonmotor"
        """
        return '+'.join(sorted(modalities))
    
    @staticmethod
    def parse_modality_spec(spec: str) -> List[str]:
        """
        Parse modality specification string.
        
        Supports:
        - Predefined sets: 'all', 'static+motor', etc.
        - Custom combinations: 'static,motor,nonmotor' (comma-separated)
        
        Args:
            spec: Modality specification string
            
        Returns:
            List of modality names
        """
        # Check if it's a predefined set
        if spec in MultiModalTrainer.MODALITY_SETS:
            return MultiModalTrainer.MODALITY_SETS[spec]
        
        # Parse comma-separated list
        modalities = [m.strip() for m in spec.split(',')]
        return modalities
    
    def get_save_dir(self, modalities: List[str]) -> Path:
        """Get save directory for a modality combination"""
        key = self.get_modality_key(modalities)
        return self.base_save_dir / f"modalities_{key}"
    
    def is_trained(self, modalities: List[str]) -> bool:
        """
        Check if a modality configuration has already been trained.
        
        Checks for kfold_results.json with status='completed' and fold results.
        
        Args:
            modalities: List of modality names
            
        Returns:
            True if training appears complete
        """
        save_dir = self.get_save_dir(modalities)
        results_file = save_dir / "kfold_results.json"
        
        if not results_file.exists():
            return False
        
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                if results_data.get('status') != 'completed':
                    return False
                
                if not results_data.get('fold_results'):
                    return False
                
                return True
        except (json.JSONDecodeError, KeyError):
            return False
    
    def create_config_for_modalities(self, modalities: List[str]):
        """
        Create a config copy with specified modalities enabled.
        
        Args:
            modalities: List of modality names to enable
            
        Returns:
            New config object with enabled_modalities set
        """
        # Deep copy config to avoid modifying the original
        new_config = copy.deepcopy(self.base_config)
        
        # Set enabled modalities (create a new list to avoid reference issues)
        new_config.model.enabled_modalities = list(modalities)
        
        # Update save directory to modality-specific one
        modality_key = self.get_modality_key(modalities)
        new_config.data.model_save_dir = str(self.get_save_dir(modalities))
        
        return new_config
    
    def train_modality_config(
        self,
        modalities: List[str],
        force_retrain: bool = False
    ) -> Dict:
        """
        Train a single modality configuration using k-fold CV.
        
        Args:
            modalities: List of modality names to enable
            force_retrain: If True, retrain even if already complete
            
        Returns:
            Dictionary with training results
        """
        modality_key = self.get_modality_key(modalities)
        
        print("\n" + "=" * 80)
        print(f"TRAINING MODALITY CONFIGURATION: {modality_key}")
        print(f"Modalities: {modalities}")
        print("=" * 80)
        
        # Check if already trained
        if not force_retrain and self.is_trained(modalities):
            print(f"\n✓ Model already trained. Skipping...")
            print(f"  To retrain, use force_retrain=True")
            
            # Load existing results
            results_file = self.get_save_dir(modalities) / "kfold_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    return {
                        'modalities': modalities,
                        'modality_key': modality_key,
                        'status': 'skipped',
                        'results': results
                    }
            
            return {
                'modalities': modalities,
                'modality_key': modality_key,
                'status': 'skipped',
                'results': None
            }
        
        # Create config for this modality combination
        config = self.create_config_for_modalities(modalities)
        
        # Train using k-fold CV
        kfold_trainer = KFoldTrainer(
            config=config,
            prepared_data=self.prepared_data,
            n_splits=self.n_splits,
            test_ratio=self.test_ratio,
            random_seed=self.random_seed,
            device=self.device,
            num_workers=self.num_workers,
            modalities=modalities,
            modality_key=modality_key
        )
        
        try:
            results = kfold_trainer.train(save_fold_checkpoints=True)
            
            # Store kfold_trainer for potential test evaluation
            self.kfold_trainers[modality_key] = kfold_trainer
            
            # kfold_results.json is already saved by KFoldTrainer with modality metadata
            
            return {
                'modalities': modalities,
                'modality_key': modality_key,
                'status': 'completed',
                'results': results
            }
        except Exception as e:
            print(f"\n✗ Training failed for {modality_key}: {e}")
            
            save_dir = self.get_save_dir(modalities)
            
            # Mark as failed in kfold_results.json
            results_file = save_dir / "kfold_results.json"
            failure_data = {
                'modalities': modalities,
                'modality_key': modality_key,
                'status': 'failed',
                'error': str(e),
                'n_splits': self.n_splits,
                'test_ratio': self.test_ratio
            }
            with open(results_file, 'w') as f:
                json.dump(failure_data, f, indent=2)
            
            return {
                'modalities': modalities,
                'modality_key': modality_key,
                'status': 'failed',
                'error': str(e),
                'results': None
            }
    
    def train_multiple(
        self,
        modality_specs: List[str],
        force_retrain: bool = False
    ) -> Dict[str, Dict]:
        """
        Train multiple modality configurations.
        
        Args:
            modality_specs: List of modality specifications (see parse_modality_spec)
            force_retrain: If True, retrain all even if already complete
            
        Returns:
            Dictionary mapping modality_key -> training results
        """
        print("=" * 80)
        print("MULTI-MODAL ABLATION TRAINING")
        print("=" * 80)
        print(f"Total configurations to train: {len(modality_specs)}")
        print(f"Force retrain: {force_retrain}")
        print("=" * 80)
        
        # Parse modality specs
        modality_configs = []
        for spec in modality_specs:
            modalities = self.parse_modality_spec(spec)
            modality_configs.append(modalities)
            print(f"  - {spec} -> {modalities}")
        
        # Train each configuration
        all_results = {}
        
        for i, modalities in enumerate(modality_configs, 1):
            print(f"\n[{i}/{len(modality_configs)}] ", end="")
            result = self.train_modality_config(modalities, force_retrain=force_retrain)
            all_results[result['modality_key']] = result
            # Summary is already saved in each modality's folder by train_modality_config
        
        # Print summary
        summary = {
            'total_configs': len(modality_configs),
            'completed': sum(1 for r in all_results.values() if r['status'] == 'completed'),
            'skipped': sum(1 for r in all_results.values() if r['status'] == 'skipped'),
            'failed': sum(1 for r in all_results.values() if r['status'] == 'failed'),
            'results': all_results
        }
        self._print_summary(summary)
        
        self.training_results = all_results
        return all_results
    
    def _print_summary(self, summary: Dict):
        """Print training summary across all modality configurations"""
        print("\n" + "=" * 80)
        print("MULTI-MODAL TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total configurations: {summary['total_configs']}")
        print(f"  Completed: {summary['completed']}")
        print(f"  Skipped (already trained): {summary['skipped']}")
        print(f"  Failed: {summary['failed']}")
        
        if summary['completed'] > 0 or summary['skipped'] > 0:
            print("\nResults by configuration:")
            for key, result in summary['results'].items():
                status_icon = {
                    'completed': '✓',
                    'skipped': '○',
                    'failed': '✗'
                }.get(result['status'], '?')
                
                print(f"  {status_icon} {key}: {result['status']}")
                
                if result['status'] == 'completed' and result.get('results'):
                    summary_data = result['results'].get('summary', {})
                    mean_val = summary_data.get('mean_val_loss', 0)
                    std_val = summary_data.get('std_val_loss', 0)
                    print(f"      Val Loss: {mean_val:.4f} ± {std_val:.4f}")
        
        print("\n" + "=" * 80)
        print("Results files saved in each modality's folder:")
        for key, result in summary['results'].items():
            save_dir = self.get_save_dir(result.get('modalities', []))
            print(f"  {key}: {save_dir / 'kfold_results.json'}")
        print("=" * 80)
    
    def aggregate_summaries(self) -> Dict:
        """
        Aggregate training summaries from all modality folders.
        Useful for reading results after async training.
        
        Returns:
            Dictionary with aggregated summary data
        """
        aggregated = {
            'total_configs': 0,
            'completed': 0,
            'skipped': 0,
            'failed': 0,
            'results': {}
        }
        
        # Find all modality directories
        for modality_dir in self.base_save_dir.glob("modalities_*"):
            if not modality_dir.is_dir():
                continue
            
            results_file = modality_dir / "kfold_results.json"
            if not results_file.exists():
                continue
            
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    
                modality_key = results_data.get('modality_key', modality_dir.name)
                status = results_data.get('status', 'unknown')
                
                aggregated['results'][modality_key] = results_data
                
                if status == 'completed':
                    aggregated['completed'] += 1
                elif status == 'skipped':
                    aggregated['skipped'] += 1
                elif status == 'failed':
                    aggregated['failed'] += 1
                
                aggregated['total_configs'] += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read results from {results_file}: {e}")
                continue
        
        return aggregated
    
    def compare_results(self) -> Dict:
        """
        Compare results across all trained modality configurations.
        
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'configurations': [],
            'best_config': None,
            'best_val_loss': float('inf')
        }
        
        for key, result in self.training_results.items():
            if result['status'] != 'completed' or not result.get('results'):
                continue
            
            summary = result['results'].get('summary', {})
            mean_val_loss = summary.get('mean_val_loss', float('inf'))
            
            comparison['configurations'].append({
                'modality_key': key,
                'modalities': result['modalities'],
                'mean_val_loss': mean_val_loss,
                'std_val_loss': summary.get('std_val_loss', 0),
            })
            
            if mean_val_loss < comparison['best_val_loss']:
                comparison['best_val_loss'] = mean_val_loss
                comparison['best_config'] = key
        
        # Sort by validation loss
        comparison['configurations'].sort(key=lambda x: x['mean_val_loss'])
        
        return comparison
    
    def evaluate_test_all(self) -> Dict[str, Dict]:
        """
        Evaluate test set for all trained modality configurations.
        
        Returns:
            Dictionary mapping modality_key -> test evaluation results
        """
        test_results = {}
        
        print("\n" + "=" * 80)
        print("EVALUATING TEST SET FOR ALL MODALITY CONFIGURATIONS")
        print("=" * 80)
        
        for modality_key, kfold_trainer in self.kfold_trainers.items():
            if kfold_trainer.test_loader is None:
                print(f"\n⚠ Skipping {modality_key}: test_loader not available")
                continue
            
            print(f"\nEvaluating test set for: {modality_key}")
            try:
                test_result = kfold_trainer.evaluate_test()
                test_results[modality_key] = test_result
            except Exception as e:
                print(f"✗ Test evaluation failed for {modality_key}: {e}")
                test_results[modality_key] = {'error': str(e)}
        
        print("\n" + "=" * 80)
        print("TEST EVALUATION COMPLETE")
        print("=" * 80)
        
        return test_results