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


# Legacy compatibility: old interface for backward compatibility
class VisitTokenBuilderLegacy(nn.Module):
    """
    Legacy VisitTokenBuilder with fixed 4-modality interface.
    Use VisitTokenBuilder with enabled_modalities for new code.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.builder = VisitTokenBuilder(
            d_model, 
            ['static', 'motor', 'nonmotor', 'medication'], 
            dropout
        )
        
    def forward(
        self,
        static_emb: torch.Tensor,
        motor_emb: torch.Tensor,
        nonmotor_emb: torch.Tensor,
        med_emb: torch.Tensor
    ) -> torch.Tensor:
        seq_len = motor_emb.shape[1]
        embeddings = {
            'static': static_emb,
            'motor': motor_emb,
            'nonmotor': nonmotor_emb,
            'medication': med_emb
        }
        return self.builder(embeddings, seq_len)


if __name__ == "__main__":
    # Test embeddings
    print("Testing embedding modules...")
    print("=" * 60)
    
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
    
    # Forward pass through individual embeddings
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
    
    # Test new VisitTokenBuilder with all modalities
    print("\n--- Testing VisitTokenBuilder (all modalities) ---")
    enabled_all = ['static', 'motor', 'nonmotor', 'medication']
    visit_builder_all = VisitTokenBuilder(d_model, enabled_all)
    
    embeddings_dict = {
        'static': static_emb,
        'motor': motor_emb,
        'nonmotor': nonmotor_emb,
        'medication': med_emb
    }
    
    visit_tokens = visit_builder_all(embeddings_dict, seq_len)
    visit_tokens = visit_tokens + time_emb
    print(f"  Visit tokens shape (4 modalities): {visit_tokens.shape}")
    
    # Test with subset of modalities (ablation)
    print("\n--- Testing VisitTokenBuilder (motor + static only) ---")
    enabled_subset = ['static', 'motor']
    visit_builder_subset = VisitTokenBuilder(d_model, enabled_subset)
    
    embeddings_subset = {
        'static': static_emb,
        'motor': motor_emb
    }
    
    visit_tokens_subset = visit_builder_subset(embeddings_subset, seq_len)
    visit_tokens_subset = visit_tokens_subset + time_emb
    print(f"  Visit tokens shape (2 modalities): {visit_tokens_subset.shape}")
    
    # Test legacy interface
    print("\n--- Testing Legacy Interface ---")
    legacy_builder = VisitTokenBuilderLegacy(d_model)
    visit_tokens_legacy = legacy_builder(static_emb, motor_emb, nonmotor_emb, med_emb)
    print(f"  Legacy visit tokens shape: {visit_tokens_legacy.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All embedding modules working correctly!")
