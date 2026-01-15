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
    Builds visit tokens by combining embeddings from all modalities
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Projection layer to combine modalities
        # Input: concatenated embeddings from static + motor + nonmotor + med
        # Output: d_model
        self.projection = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        static_emb: torch.Tensor,
        motor_emb: torch.Tensor,
        nonmotor_emb: torch.Tensor,
        med_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            static_emb: [batch, d_model] - static features (same for all visits)
            motor_emb: [batch, seq_len, d_model]
            nonmotor_emb: [batch, seq_len, d_model]
            med_emb: [batch, seq_len, d_model]
            
        Returns:
            visit_tokens: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = motor_emb.shape
        
        # Expand static embedding to match sequence length
        static_expanded = static_emb.unsqueeze(1).expand(batch_size, seq_len, self.d_model)
        
        # Concatenate all modalities
        combined = torch.cat([static_expanded, motor_emb, nonmotor_emb, med_emb], dim=-1)
        # Shape: [batch, seq_len, d_model * 4]
        
        # Project to d_model
        visit_tokens = self.projection(combined)
        # Shape: [batch, seq_len, d_model]
        
        return visit_tokens


if __name__ == "__main__":
    # Test embeddings
    print("Testing embedding modules...")
    
    batch_size = 4
    seq_len = 10
    n_motor_features = 35
    n_nonmotor_features = 20
    n_med_features = 5
    n_static_features = 25
    d_model = 256
    
    # Create test data
    motor_values = torch.randn(batch_size, seq_len, n_motor_features)
    motor_mask = torch.bernoulli(torch.ones_like(motor_values) * 0.1)  # 10% missing
    
    nonmotor_values = torch.randn(batch_size, seq_len, n_nonmotor_features)
    nonmotor_mask = torch.bernoulli(torch.ones_like(nonmotor_values) * 0.2)  # 20% missing
    
    med_values = torch.randn(batch_size, seq_len, n_med_features)
    med_mask = torch.bernoulli(torch.ones_like(med_values) * 0.05)  # 5% missing
    
    static_values = torch.randn(batch_size, n_static_features)
    static_mask = torch.bernoulli(torch.ones_like(static_values) * 0.15)  # 15% missing
    
    time_months = torch.linspace(0, 48, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Create embedding modules
    static_embedding = StaticFeatureEmbedding(n_static_features, d_model, [128])
    motor_embedding = VisitFeatureEmbedding(n_motor_features, d_model, [128])
    nonmotor_embedding = VisitFeatureEmbedding(n_nonmotor_features, d_model, [128])
    med_embedding = VisitFeatureEmbedding(n_med_features, d_model, [64])
    time_encoding = SinusoidalTimeEncoding(d_model, max_time=120.0)
    visit_builder = VisitTokenBuilder(d_model)
    
    # Forward pass
    print(f"\nInput shapes:")
    print(f"  Motor: {motor_values.shape}, mask: {motor_mask.shape}")
    print(f"  Non-motor: {nonmotor_values.shape}, mask: {nonmotor_mask.shape}")
    print(f"  Medication: {med_values.shape}, mask: {med_mask.shape}")
    print(f"  Static: {static_values.shape}, mask: {static_mask.shape}")
    print(f"  Time: {time_months.shape}")
    
    static_emb = static_embedding(static_values, static_mask)
    motor_emb = motor_embedding(motor_values, motor_mask)
    nonmotor_emb = nonmotor_embedding(nonmotor_values, nonmotor_mask)
    med_emb = med_embedding(med_values, med_mask)
    time_emb = time_encoding(time_months)
    
    print(f"\nEmbedding shapes:")
    print(f"  Static: {static_emb.shape}")
    print(f"  Motor: {motor_emb.shape}")
    print(f"  Non-motor: {nonmotor_emb.shape}")
    print(f"  Medication: {med_emb.shape}")
    print(f"  Time encoding: {time_emb.shape}")
    
    visit_tokens = visit_builder(static_emb, motor_emb, nonmotor_emb, med_emb)
    visit_tokens = visit_tokens + time_emb
    
    print(f"\nVisit tokens shape: {visit_tokens.shape}")
    print("\n✓ All embedding modules working correctly!")
