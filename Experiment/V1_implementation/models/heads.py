"""
Prediction heads for V1 model
"""

import torch
import torch.nn as nn
from typing import List


class NextVisitPredictionHead(nn.Module):
    """
    Predicts next visit UPDRS totals from current visit hidden state
    Supports predicting multiple totals: NP1TOT, NP2TOT, NP3TOT, NP4TOT
    """
    
    def __init__(self, d_model: int, n_targets: int = 4, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        """
        Args:
            d_model: Input dimension (from transformer)
            n_targets: Number of UPDRS totals to predict (default: 4)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_targets = n_targets
        
        layers = []
        prev_dim = d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer: n_targets values (one per UPDRS total)
        layers.append(nn.Linear(prev_dim, n_targets))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] - from transformer
            
        Returns:
            predictions: [batch, seq_len, n_targets] - predicted UPDRS totals for next visit
        """
        predictions = self.mlp(hidden_states)  # [batch, seq_len, n_targets]
        return predictions


class ProgressionSlopeHead(nn.Module):
    """
    Predicts patient-level progression slope from pooled representation
    """
    
    def __init__(self, d_model: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, pooling: str = 'mean'):
        """
        Args:
            d_model: Input dimension (from transformer)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'last', or 'max')
        """
        super().__init__()
        
        self.pooling = pooling
        
        layers = []
        prev_dim = d_model
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer: single value (slope)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] - from transformer
            attention_mask: [batch, seq_len] - 1 for valid positions, 0 for padding
            
        Returns:
            slope: [batch] - predicted progression slope
        """
        # Pool sequence representation
        if self.pooling == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                count = attention_mask.sum(dim=1, keepdim=True)
                pooled = sum_hidden / count.clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)
                
        elif self.pooling == 'last':
            if attention_mask is not None:
                # Get last valid position for each sequence
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths.long()]
            else:
                pooled = hidden_states[:, -1, :]
                
        elif self.pooling == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states_masked = hidden_states.clone()
                hidden_states_masked[mask_expanded == 0] = -1e9
                pooled = hidden_states_masked.max(dim=1)[0]
            else:
                pooled = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Predict slope
        slope = self.mlp(pooled)
        return slope.squeeze(-1)  # [batch]


class MultiTaskHead(nn.Module):
    """
    Combines both prediction heads with separate losses
    """
    
    def __init__(
        self,
        d_model: int,
        n_targets: int = 4,
        next_visit_hidden: List[int] = [128, 64],
        slope_hidden: List[int] = [128, 64],
        dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        """
        Args:
            d_model: Input dimension
            n_targets: Number of UPDRS totals to predict (default: 4)
            next_visit_hidden: Hidden dims for next-visit head
            slope_hidden: Hidden dims for slope head
            dropout: Dropout rate
            pooling: Pooling method for slope head
        """
        super().__init__()
        
        self.n_targets = n_targets
        self.next_visit_head = NextVisitPredictionHead(d_model, n_targets, next_visit_hidden, dropout)
        self.slope_head = ProgressionSlopeHead(d_model, slope_hidden, dropout, pooling)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]
            
        Returns:
            dict with 'next_visit' and 'slope' predictions
        """
        next_visit_preds = self.next_visit_head(hidden_states)  # [batch, seq_len, n_targets]
        slope_preds = self.slope_head(hidden_states, attention_mask)  # [batch]
        
        return {
            'next_visit': next_visit_preds,  # [batch, seq_len, n_targets]
            'slope': slope_preds  # [batch]
        }




if __name__ == "__main__":
    # Test prediction heads
    print("Testing prediction heads...")
    
    batch_size = 4
    seq_len = 10
    d_model = 256
    
    # Create test hidden states
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len)
    # Mask some positions to simulate variable-length sequences
    attention_mask[0, 8:] = 0
    attention_mask[1, 6:] = 0
    attention_mask[2, 9:] = 0
    
    print(f"\nInput shape: {hidden_states.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Valid positions per sequence: {attention_mask.sum(dim=1).tolist()}")
    
    # Test next-visit head
    n_targets = 4  # NP1TOT, NP2TOT, NP3TOT, NP4TOT
    next_visit_head = NextVisitPredictionHead(d_model, n_targets, [128, 64])
    next_visit_preds = next_visit_head(hidden_states)
    print(f"\nNext-visit predictions shape: {next_visit_preds.shape}")
    print(f"Sample predictions (first visit, all 4 totals): {next_visit_preds[0, 0, :].tolist()}")
    
    # Test slope head with different pooling methods
    for pooling in ['mean', 'last', 'max']:
        slope_head = ProgressionSlopeHead(d_model, [128, 64], pooling=pooling)
        slope_preds = slope_head(hidden_states, attention_mask)
        print(f"\nSlope predictions ({pooling} pooling) shape: {slope_preds.shape}")
        print(f"Sample slopes: {slope_preds.tolist()}")
    
    # Test multi-task head
    print("\n" + "="*80)
    print("Testing multi-task head...")
    multi_head = MultiTaskHead(d_model, n_targets, [128, 64], [128, 64], pooling='mean')
    outputs = multi_head(hidden_states, attention_mask)
    
    print(f"\nMulti-task outputs:")
    print(f"  Next-visit shape: {outputs['next_visit'].shape}")
    print(f"  Slope shape: {outputs['slope'].shape}")
    
    print("\n✓ All prediction heads working correctly!")
