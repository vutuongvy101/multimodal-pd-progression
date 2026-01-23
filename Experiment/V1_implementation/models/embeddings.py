"""
Modality embedding modules with missingness handling
"""

import torch
import torch.nn as nn
import math
from typing import List


class ModalityEmbedding(nn.Module):
    """
    Embeds a modality (motor, non-motor, medication, or static features)
    Handles missingness by concatenating values with missing masks
    """
    
    def __init__(self, n_features: int, d_model: int, hidden_dims: List[int] = None, dropout: float = 0.1):
        """
        Args:
            n_features: Number of input features
            d_model: Output embedding dimension
            hidden_dims: List of hidden layer dimensions for MLP
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        input_dim = n_features * 2  # Values + masks
        
        if hidden_dims is None:
            hidden_dims = [128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Embed modality features with missingness handling.
        
        Args:
            values: Feature values tensor of shape [batch, seq_len, n_features] 
                    or [batch, n_features] for static features
            mask: Missingness mask tensor of same shape as values, where 1=missing, 0=present
            
        Returns:
            Embeddings tensor of shape [batch, seq_len, d_model] or [batch, d_model]
        """
        combined = torch.cat([values, mask], dim=-1)
        return self.mlp(combined)


class StaticFeatureEmbedding(ModalityEmbedding):
    """Embedding for static (patient-level) features"""
    pass


class VisitFeatureEmbedding(ModalityEmbedding):
    """Embedding for time-varying (visit-level) features"""
    pass


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal positional encoding based on continuous time"""
    
    def __init__(self, d_model: int, max_time: float = 120.0):
        """
        Args:
            d_model: Embedding dimension
            max_time: Maximum time value (e.g., 120 months = 10 years)
        """
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)  # Registered buffers automatically move to GPU
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional encoding based on continuous time.
        
        Args:
            time: Time tensor of shape [batch, seq_len] in months since baseline
            
        Returns:
            Time encoding tensor of shape [batch, seq_len, d_model]
        """
        time_normalized = (time / self.max_time).unsqueeze(-1)
        batch_size, seq_len = time.shape
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=time.device)
        
        pe[:, :, 0::2] = torch.sin(time_normalized * self.div_term)
        pe[:, :, 1::2] = torch.cos(time_normalized * self.div_term)
        
        return pe


class VisitTokenBuilder(nn.Module):
    """
    Builds visit tokens by combining embeddings from enabled modalities.
    
    Supports dynamic modality selection - only enabled modalities are concatenated
    and projected to form visit tokens.
    """
    
    def __init__(self, d_model: int, enabled_modalities: List[str], dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension
            enabled_modalities: List of enabled modality names 
                               (e.g., ['static', 'motor', 'nonmotor', 'medication'])
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.enabled_modalities = enabled_modalities
        
        n_modalities = len(enabled_modalities)
        if n_modalities == 0:
            raise ValueError("At least one modality must be enabled")
        
        self.projection = nn.Sequential(
            nn.Linear(d_model * n_modalities, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        embeddings: dict,
        seq_len: int
    ) -> torch.Tensor:
        """
        Build visit tokens by combining embeddings from enabled modalities.
        
        Args:
            embeddings: Dict mapping modality name to embedding tensor.
                        Static modalities: [batch, d_model]
                        Time-varying modalities: [batch, seq_len, d_model]
            seq_len: Sequence length (used to expand static embeddings)
            
        Returns:
            Visit tokens tensor of shape [batch, seq_len, d_model]
        """
        first_emb = next(iter(embeddings.values()))
        batch_size = first_emb.shape[0]
        
        components = []
        for mod in self.enabled_modalities:
            if mod not in embeddings:
                raise KeyError(f"Modality '{mod}' enabled but not found in embeddings dict")
            
            emb = embeddings[mod]
            if emb.dim() == 2:
                emb = emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
            
            components.append(emb)
        
        combined = torch.cat(components, dim=-1)
        return self.projection(combined)