"""
Prediction heads for V1 model
"""

import torch
import torch.nn as nn
from typing import List, Optional


class NextVisitPredictionHead(nn.Module):
    """
    Predicts next visit UPDRS totals from current visit hidden state.
    
    Supports predicting multiple totals: NP1TOT, NP2TOT, NP3TOT, NP4TOT.
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
        
        layers.append(nn.Linear(prev_dim, n_targets))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] - from transformer
            
        Returns:
            predictions: [batch, seq_len, n_targets] - predicted UPDRS totals for next visit
        """
        return self.mlp(hidden_states)


class ProgressionSlopeHead(nn.Module):
    """
    Predicts patient-level progression slopes from pooled representation.
    
    Supports predicting multiple slopes: one per UPDRS total
    (NP1RTOT, NP2PTOT, NP3TOT, NP4TOT).
    """
    
    def __init__(
        self,
        d_model: int,
        n_targets: int = 4,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        """
        Args:
            d_model: Input dimension (from transformer)
            n_targets: Number of slopes to predict (default: 4, one per UPDRS total)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            pooling: Pooling method ('mean', 'last', or 'max')
        """
        super().__init__()

        self.n_targets = int(n_targets)
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
        
        layers.append(nn.Linear(prev_dim, self.n_targets))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model] - from transformer
            attention_mask: [batch, seq_len] - 1 for valid positions, 0 for padding
            
        Returns:
            slopes: [batch, n_targets] - predicted progression slopes (one per UPDRS total)
        """
        pooled = self._pool_sequence(hidden_states, attention_mask)
        return self.mlp(pooled)
    
    def _pool_sequence(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool sequence representation using specified pooling method."""
        if self.pooling == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                count = attention_mask.sum(dim=1, keepdim=True)
                return sum_hidden / count.clamp(min=1)
            return hidden_states.mean(dim=1)
        
        elif self.pooling == 'last':
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, seq_lengths.long()]
            return hidden_states[:, -1, :]
        
        elif self.pooling == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
                hidden_states_masked = hidden_states.clone()
                hidden_states_masked[mask_expanded == 0] = -1e9
                return hidden_states_masked.max(dim=1)[0]
            return hidden_states.max(dim=1)[0]
        
        raise ValueError(f"Unknown pooling method: {self.pooling}")


class MultiTaskHead(nn.Module):
    """
    Combines both prediction heads (next-visit and slope) with separate losses.
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
        self.slope_head = ProgressionSlopeHead(d_model, n_targets, slope_hidden, dropout, pooling)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, seq_len]
            
        Returns:
            dict with 'next_visit' and 'slope' predictions
        """
        return {
            'next_visit': self.next_visit_head(hidden_states),
            'slope': self.slope_head(hidden_states, attention_mask)
        }



