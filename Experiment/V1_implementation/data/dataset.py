"""
PyTorch Dataset for V1 model
Handles variable-length sequences with padding
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
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
            - next_visit_targets (NP3TOT values)
            - slope_target
            - seq_len (actual length before padding)
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
        np3tot = [visit['np3tot'] for visit in visits]
        
        # Convert to tensors
        motor_values = torch.FloatTensor(np.array(motor_values))
        motor_mask = torch.FloatTensor(np.array(motor_mask))
        nonmotor_values = torch.FloatTensor(np.array(nonmotor_values))
        nonmotor_mask = torch.FloatTensor(np.array(nonmotor_mask))
        med_values = torch.FloatTensor(np.array(med_values))
        med_mask = torch.FloatTensor(np.array(med_mask))
        time_months = torch.FloatTensor(time_months)
        next_visit_targets = torch.FloatTensor(np3tot)
        
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
            'slope_target': slope_target,
            'seq_len': seq_len
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function to handle variable-length sequences
    Pads sequences to max length in batch
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
    next_visit_targets_padded = torch.zeros(batch_size, max_len)
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
        next_visit_targets_padded[i, :seq_len] = item['next_visit_targets']
        attention_mask[i, :seq_len] = 1  # 1 = valid, 0 = padding
    
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
        'slope_targets': slope_targets
    }


def create_dataloaders(
    prepared_data: Dict,
    config,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from prepared data
    
    Args:
        prepared_data: Output from data_preparation.prepare_dataset()
        config: Configuration dict
        num_workers: Number of workers for DataLoader
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # This is a placeholder - implement based on your data structure
    # Key steps:
    # 1. Split patient_ids into train/val/test
    # 2. Create Dataset objects for each split
    # 3. Create DataLoaders with collate_fn
    
    print("Creating dataloaders...")
    
    # Example structure:
    # train_dataset = PPMILongitudinalDataset(
    #     patient_ids=train_ids,
    #     static_data=static_data,
    #     longitudinal_data=longitudinal_data,
    #     slopes=slopes,
    #     max_seq_len=config['model'].max_seq_len
    # )
    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config['training'].batch_size,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=num_workers
    # )
    
    print("WARNING: create_dataloaders not yet fully implemented")
    return None, None, None


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
                'np3tot': np.random.randn() * 10 + 30
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
