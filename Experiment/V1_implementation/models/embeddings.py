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
        
        # Input dimension is 2x features (values + masks)
        input_dim = n_features * 2
        
        # Build MLP
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
        
        # Final projection to d_model
        layers.append(nn.Linear(prev_dim, d_model))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: [batch, seq_len, n_features] or [batch, n_features] for static
            mask: [batch, seq_len, n_features] or [batch, n_features], 1 if missing
            
        Returns:
            embeddings: [batch, seq_len, d_model] or [batch, d_model]
        """
        # Concatenate values and mask
        combined = torch.cat([values, mask], dim=-1)  # [..., n_features*2]
        
        # Pass through MLP
        embeddings = self.mlp(combined)  # [..., d_model]
        
        return embeddings


class StaticFeatureEmbedding(ModalityEmbedding):
    """Embedding for static (patient-level) features"""
    
    def __init__(self, n_features: int, d_model: int, hidden_dims: List[int] = [128], dropout: float = 0.1):
        super().__init__(n_features, d_model, hidden_dims, dropout)
        
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: [batch, n_features]
            mask: [batch, n_features]
            
        Returns:
            embeddings: [batch, d_model]
        """
        return super().forward(values, mask)


class VisitFeatureEmbedding(ModalityEmbedding):
    """Embedding for time-varying (visit-level) features"""
    
    def __init__(self, n_features: int, d_model: int, hidden_dims: List[int] = [128], dropout: float = 0.1):
        super().__init__(n_features, d_model, hidden_dims, dropout)
        
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: [batch, seq_len, n_features]
            mask: [batch, seq_len, n_features]
            
        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        return super().forward(values, mask)


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
        
        # Pre-compute division terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [batch, seq_len] - time in months since baseline
            
        Returns:
            time_encoding: [batch, seq_len, d_model]
        """
        # Normalize time to [0, 1] range
        time_normalized = time / self.max_time
        time_normalized = time_normalized.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Create encoding
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
        
        # Number of modalities determines projection input size
        n_modalities = len(enabled_modalities)
        
        if n_modalities == 0:
            raise ValueError("At least one modality must be enabled")
        
        # Projection layer to combine modalities
        # Input: concatenated embeddings from enabled modalities
        # Output: d_model
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
        Args:
            embeddings: Dict mapping modality name to embedding tensor
                       - Static: [batch, d_model]
                       - Time-varying: [batch, seq_len, d_model]
            seq_len: Sequence length (for expanding static embeddings)
            
        Returns:
            visit_tokens: [batch, seq_len, d_model]
        """
        # Get batch size from first embedding
        first_emb = next(iter(embeddings.values()))
        batch_size = first_emb.shape[0]
        
        components = []
        for mod in self.enabled_modalities:
            if mod not in embeddings:
                raise KeyError(f"Modality '{mod}' enabled but not found in embeddings dict")
            
            emb = embeddings[mod]
            
            # Expand static (2D) embeddings to sequence length
            if emb.dim() == 2:  # [B, d] -> [B, T, d]
                emb = emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
            
            components.append(emb)
        
        # Concatenate all enabled modalities
        combined = torch.cat(components, dim=-1)  # [B, T, d * n_modalities]
        
        # Project to d_model
        visit_tokens = self.projection(combined)  # [B, T, d]
        
        return visit_tokens