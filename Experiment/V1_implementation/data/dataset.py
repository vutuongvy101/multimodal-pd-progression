"""
PyTorch Dataset for V1 model
Handles variable-length sequences with padding
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PPMILongitudinalDataset(Dataset):
    """
    Dataset for longitudinal PPMI data
    Each sample is one patient with their full visit sequence
    """
    
    def __init__(
        self,
        patient_ids: List[int],
        static_data: Dict[int, np.ndarray],  # PATNO -> static features
        longitudinal_data: Dict[int, List[Dict]],  # PATNO -> list of visits
        slopes: Dict[int, float],  # PATNO -> empirical slope
        max_seq_len: int = 20
    ):
        """
        Args:
            patient_ids: List of PATNOs in this split
            static_data: Static features for each patient
            longitudinal_data: List of visits for each patient, each visit is a dict with:
                - 'motor_values': np.ndarray
                - 'motor_mask': np.ndarray
                - 'nonmotor_values': np.ndarray
                - 'nonmotor_mask': np.ndarray
                - 'med_values': np.ndarray
                - 'med_mask': np.ndarray
                - 'time_months': float
            slopes: Empirical slope for each patient (-999 if not available)
            max_seq_len: Maximum sequence length (will truncate if longer)
        """
        self.patient_ids = patient_ids
        self.static_data = static_data
        self.longitudinal_data = longitudinal_data
        self.slopes = slopes
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.patient_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one patient's data
        
        Returns dict with:
            - static_values, static_mask
            - motor_values, motor_mask
            - nonmotor_values, nonmotor_mask
            - med_values, med_mask
            - time_months
            - next_visit_targets [seq_len, 4] - UPDRS totals (NP1TOT, NP2TOT, NP3TOT, NP4TOT)
            - next_visit_label_mask [seq_len, 4] - Label availability mask (1=present, 0=missing)
            - slope_target
            - seq_len (actual length before padding)
        
        Two types of masks:
            A) Input missingness masks (motor_mask, etc.): "Is feature X observed at visit t?"
            B) Label availability masks (next_visit_label_mask): "Is target Y available at visit t?"
        """
        patno = self.patient_ids[idx]
        
        # Static features
        static = self.static_data[patno]
        static_values = torch.FloatTensor(static['values'])
        static_mask = torch.FloatTensor(static['mask'])
        
        # Longitudinal features
        visits = self.longitudinal_data[patno]
        
        # Truncate if too long
        if len(visits) > self.max_seq_len:
            visits = visits[:self.max_seq_len]
        
        seq_len = len(visits)
        
        # Extract sequences
        motor_values = [visit['motor_values'] for visit in visits]
        motor_mask = [visit['motor_mask'] for visit in visits]
        nonmotor_values = [visit['nonmotor_values'] for visit in visits]
        nonmotor_mask = [visit['nonmotor_mask'] for visit in visits]
        med_values = [visit['med_values'] for visit in visits]
        med_mask = [visit['med_mask'] for visit in visits]
        time_months = [visit['time_months'] for visit in visits]
        
        # Extract UPDRS totals (NP1TOT, NP2TOT, NP3TOT, NP4TOT)
        if 'updrs_totals' in visits[0]:
            # Use the new format with all 4 totals
            updrs_totals = np.array([visit['updrs_totals'] for visit in visits])  # [seq_len, 4]
        else:
            # Fallback to old format (only NP3TOT) - pad with NaN for backward compatibility
            np3tot = [visit['np3tot'] for visit in visits]
            updrs_totals = np.full((len(visits), 4), np.nan)
            updrs_totals[:, 2] = np3tot  # NP3TOT is at index 2
        
        # Create label availability mask (Mask B)
        # 1 = label present and valid, 0 = label missing (NaN)
        # This mask is used in loss computation to exclude missing targets
        label_mask = (~np.isnan(updrs_totals)).astype(np.float32)  # [seq_len, 4]
        
        # Fill NaN values with 0 (mask tells the model they're missing)
        updrs_totals_filled = np.nan_to_num(updrs_totals, nan=0.0).astype(np.float32)
        
        # Convert to tensors
        motor_values = torch.FloatTensor(np.array(motor_values))
        motor_mask = torch.FloatTensor(np.array(motor_mask))
        nonmotor_values = torch.FloatTensor(np.array(nonmotor_values))
        nonmotor_mask = torch.FloatTensor(np.array(nonmotor_mask))
        med_values = torch.FloatTensor(np.array(med_values))
        med_mask = torch.FloatTensor(np.array(med_mask))
        time_months = torch.FloatTensor(time_months)
        next_visit_targets = torch.FloatTensor(updrs_totals_filled)  # [seq_len, 4]
        next_visit_label_mask = torch.FloatTensor(label_mask)  # [seq_len, 4]
        
        # Slope
        slope_target = torch.FloatTensor([self.slopes.get(patno, -999)])
        
        return {
            'static_values': static_values,
            'static_mask': static_mask,
            'motor_values': motor_values,
            'motor_mask': motor_mask,
            'nonmotor_values': nonmotor_values,
            'nonmotor_mask': nonmotor_mask,
            'med_values': med_values,
            'med_mask': med_mask,
            'time_months': time_months,
            'next_visit_targets': next_visit_targets,
            'next_visit_label_mask': next_visit_label_mask,  # NEW: Label availability mask
            'slope_target': slope_target,
            'seq_len': seq_len
        }


# Backwards-compatible alias used in some tests
PPMIDataset = PPMILongitudinalDataset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle variable-length sequences.
    Pads sequences to max length in batch.
    
    Creates two types of masks:
        - attention_mask: Which visit positions are valid (not padding)
        - next_visit_label_mask: Which target labels are available (not NaN)
    """
    # Static features (no padding needed)
    static_values = torch.stack([item['static_values'] for item in batch])
    static_mask = torch.stack([item['static_mask'] for item in batch])
    
    # Get max sequence length in this batch
    max_len = max(item['seq_len'] for item in batch)
    batch_size = len(batch)
    
    # Get feature dimensions
    n_motor = batch[0]['motor_values'].shape[-1]
    n_nonmotor = batch[0]['nonmotor_values'].shape[-1]
    n_med = batch[0]['med_values'].shape[-1]
    
    # Initialize padded tensors
    motor_values_padded = torch.zeros(batch_size, max_len, n_motor)
    motor_mask_padded = torch.ones(batch_size, max_len, n_motor)  # 1 = missing
    nonmotor_values_padded = torch.zeros(batch_size, max_len, n_nonmotor)
    nonmotor_mask_padded = torch.ones(batch_size, max_len, n_nonmotor)
    med_values_padded = torch.zeros(batch_size, max_len, n_med)
    med_mask_padded = torch.ones(batch_size, max_len, n_med)
    time_months_padded = torch.zeros(batch_size, max_len)
    
    # next_visit_targets is now [seq_len, n_targets] where n_targets=4
    n_targets = batch[0]['next_visit_targets'].shape[-1] if len(batch[0]['next_visit_targets'].shape) > 1 else 1
    next_visit_targets_padded = torch.zeros(batch_size, max_len, n_targets)
    next_visit_label_mask_padded = torch.zeros(batch_size, max_len, n_targets)  # NEW: 0 = label missing
    attention_mask = torch.zeros(batch_size, max_len)
    
    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        motor_values_padded[i, :seq_len] = item['motor_values']
        motor_mask_padded[i, :seq_len] = item['motor_mask']
        nonmotor_values_padded[i, :seq_len] = item['nonmotor_values']
        nonmotor_mask_padded[i, :seq_len] = item['nonmotor_mask']
        med_values_padded[i, :seq_len] = item['med_values']
        med_mask_padded[i, :seq_len] = item['med_mask']
        time_months_padded[i, :seq_len] = item['time_months']
        
        # Handle both old format [seq_len] and new format [seq_len, n_targets]
        if len(item['next_visit_targets'].shape) == 1:
            # Old format - expand to [seq_len, 1]
            next_visit_targets_padded[i, :seq_len, 0] = item['next_visit_targets']
            # For old format, assume all labels are available if value is not NaN
            next_visit_label_mask_padded[i, :seq_len, 0] = (~torch.isnan(item['next_visit_targets'])).float()
        else:
            # New format [seq_len, n_targets]
            next_visit_targets_padded[i, :seq_len, :] = item['next_visit_targets']
            # Use the label mask from the item
            if 'next_visit_label_mask' in item:
                next_visit_label_mask_padded[i, :seq_len, :] = item['next_visit_label_mask']
            else:
                # Fallback: derive from non-NaN values
                next_visit_label_mask_padded[i, :seq_len, :] = (~torch.isnan(item['next_visit_targets'])).float()
        
        attention_mask[i, :seq_len] = 1  # 1 = valid visit, 0 = padding
    
    # Slopes
    slope_targets = torch.stack([item['slope_target'] for item in batch]).squeeze(-1)
    
    return {
        'static_values': static_values,
        'static_mask': static_mask,
        'motor_values': motor_values_padded,
        'motor_mask': motor_mask_padded,
        'nonmotor_values': nonmotor_values_padded,
        'nonmotor_mask': nonmotor_mask_padded,
        'med_values': med_values_padded,
        'med_mask': med_mask_padded,
        'time_months': time_months_padded,
        'attention_mask': attention_mask,
        'next_visit_targets': next_visit_targets_padded,
        'next_visit_label_mask': next_visit_label_mask_padded,  # NEW: Label availability mask
        'slope_targets': slope_targets
    }


def create_dataloaders(
    prepared_data: Dict,
    config,
    num_workers: int = 0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from prepared data.
    Automatically converts DataFrames to feature vectors with masking if needed.
    
    Args:
        prepared_data: Output from data_integrator.prepare_final_dataset() (DataFrames)
                       OR feature_vectors from create_feature_vectors() (masked format)
        config: Configuration dict
        num_workers: Number of workers for DataLoader
        train_ratio: Proportion of patients for training (default: 0.7)
        val_ratio: Proportion of patients for validation (default: 0.15)
        test_ratio: Proportion of patients for testing (default: 0.15)
        random_seed: Random seed for patient splitting
        
    Returns:
        train_loader, val_loader, test_loader
    """
    import sys
    import os
    
    # Add parent directory to path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from data.data_integrator import DataIntegrator
    
    print("Creating dataloaders...")
    
    # Check if we have DataFrames (need to convert) or already have feature vectors
    if 'static' in prepared_data and isinstance(prepared_data['static'], pd.DataFrame):
        # Need to convert DataFrames to feature vectors with masking
        print("  Converting DataFrames to feature vectors (creating missing value masks)...")
        integrator = DataIntegrator(config)
        feature_vectors = integrator.create_feature_vectors(prepared_data)
    elif 'static_data' in prepared_data:
        # Already in feature vector format (has 'static_data' key)
        feature_vectors = prepared_data
    else:
        raise ValueError(
            "prepared_data must be either:\n"
            "  - Output from prepare_final_dataset() (with 'static', 'longitudinal', 'slopes' DataFrames)\n"
            "  - Output from create_feature_vectors() (with 'static_data', 'longitudinal_data', 'slopes' dicts)"
        )
    
    # Extract feature vectors
    static_data = feature_vectors['static_data']
    longitudinal_data = feature_vectors['longitudinal_data']
    slopes = feature_vectors['slopes']
    
    # Get all patient IDs
    patient_ids = list(static_data.keys())
    n_patients = len(patient_ids)
    
    print(f"  Total patients: {n_patients}")
    print(f"  Splitting: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Split patients into train/val/test
    np.random.seed(random_seed)
    shuffled_ids = patient_ids.copy()
    np.random.shuffle(shuffled_ids)
    
    # First split: train vs (val+test)
    train_size = int(n_patients * train_ratio)
    train_ids = shuffled_ids[:train_size]
    
    # Second split: val vs test
    remaining_ids = shuffled_ids[train_size:]
    val_size = int(len(remaining_ids) * (val_ratio / (val_ratio + test_ratio)))
    val_ids = remaining_ids[:val_size]
    test_ids = remaining_ids[val_size:]
    
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    # Create datasets
    max_seq_len = config.model.max_seq_len if hasattr(config, 'model') else getattr(config, 'max_seq_len', 20)
    
    train_dataset = PPMILongitudinalDataset(
        patient_ids=train_ids,
        static_data=static_data,
        longitudinal_data=longitudinal_data,
        slopes=slopes,
        max_seq_len=max_seq_len
    )
    
    val_dataset = PPMILongitudinalDataset(
        patient_ids=val_ids,
        static_data=static_data,
        longitudinal_data=longitudinal_data,
        slopes=slopes,
        max_seq_len=max_seq_len
    )
    
    test_dataset = PPMILongitudinalDataset(
        patient_ids=test_ids,
        static_data=static_data,
        longitudinal_data=longitudinal_data,
        slopes=slopes,
        max_seq_len=max_seq_len
    )
    
    # Get batch size from config
    batch_size = config.training.batch_size if hasattr(config, 'training') else getattr(config, 'batch_size', 32)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    print("  ✓ Dataloaders created successfully")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset and collate function
    print("Testing Dataset and collate function...")
    
    # Create dummy data
    n_static = 25
    n_motor = 35
    n_nonmotor = 20
    n_med = 5
    
    patient_ids = [1001, 1002, 1003]
    
    # Static data
    static_data = {}
    for patno in patient_ids:
        static_data[patno] = {
            'values': np.random.randn(n_static),
            'mask': np.random.binomial(1, 0.1, n_static)
        }
    
    # Longitudinal data (variable length sequences)
    longitudinal_data = {}
    for i, patno in enumerate(patient_ids):
        n_visits = np.random.randint(5, 12)  # Variable length
        visits = []
        for v in range(n_visits):
            visits.append({
                'motor_values': np.random.randn(n_motor),
                'motor_mask': np.random.binomial(1, 0.15, n_motor),
                'nonmotor_values': np.random.randn(n_nonmotor),
                'nonmotor_mask': np.random.binomial(1, 0.2, n_nonmotor),
                'med_values': np.random.randn(n_med),
                'med_mask': np.random.binomial(1, 0.05, n_med),
                'time_months': v * 6,
                'np3tot': np.random.randn() * 10 + 30,  # For backward compatibility
                'updrs_totals': np.array([  # All 4 UPDRS totals
                    np.random.randn() * 5 + 15,   # NP1TOT
                    np.random.randn() * 5 + 15,   # NP2TOT
                    np.random.randn() * 10 + 30,  # NP3TOT
                    np.random.randn() * 3 + 5     # NP4TOT
                ])
            })
        longitudinal_data[patno] = visits
    
    # Slopes
    slopes = {patno: np.random.randn() * 0.5 for patno in patient_ids}
    
    # Create dataset
    dataset = PPMILongitudinalDataset(
        patient_ids=patient_ids,
        static_data=static_data,
        longitudinal_data=longitudinal_data,
        slopes=slopes,
        max_seq_len=15
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single item
    item = dataset[0]
    print(f"\nSingle item shapes:")
    print(f"  Static: {item['static_values'].shape}")
    print(f"  Motor sequence: {item['motor_values'].shape}")
    print(f"  Time: {item['time_months'].shape}")
    print(f"  Slope: {item['slope_target'].shape}")
    
    # Test DataLoader with collate
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"  Static values: {batch['static_values'].shape}")
    print(f"  Motor values: {batch['motor_values'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Next-visit targets: {batch['next_visit_targets'].shape}")
    print(f"  Slope targets: {batch['slope_targets'].shape}")
    
    print(f"\nAttention mask (shows padding):")
    print(batch['attention_mask'])
    
    print("\n✓ Dataset and collate function working correctly!")
