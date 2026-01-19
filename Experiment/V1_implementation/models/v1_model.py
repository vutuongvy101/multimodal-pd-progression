"""
V1: Multimodal Longitudinal Transformer
Complete model combining embeddings, transformer, and prediction heads
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

# Handle both relative imports (when used as module) and absolute imports (when run as script)
try:
    from .embeddings import (
        StaticFeatureEmbedding,
        VisitFeatureEmbedding,
        SinusoidalTimeEncoding,
        VisitTokenBuilder
    )
    from .heads import MultiTaskHead
except ImportError:
    # When run as script, use absolute imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.embeddings import (
        StaticFeatureEmbedding,
        VisitFeatureEmbedding,
        SinusoidalTimeEncoding,
        VisitTokenBuilder
    )
    from models.heads import MultiTaskHead


class V1MultimodalTransformer(nn.Module):
    """
    V1 Model: Simplified multimodal longitudinal transformer
    
    Architecture:
        1. Embed each modality (static, motor, non-motor, medication)
        2. Build visit tokens by combining modality embeddings
        3. Add time encoding
        4. Pass through Transformer encoder
        5. Make predictions via two heads:
           - Next-visit UPDRS totals (NP1TOT, NP2TOT, NP3TOT, NP4TOT)
           - Patient-level progression slope
    """
    
    def __init__(self, config):
        """
        Args:
            config: Config object with model and features attributes
        """
        super().__init__()
        
        self.config = config
        model_config = config.model
        feature_config = config.features
        
        # Get feature dimensions
        n_static = len(feature_config.static_features)
        n_motor = len(feature_config.motor_features)
        n_nonmotor = len(feature_config.nonmotor_features)
        n_med = len(feature_config.medication_features)
        
        d_model = model_config.d_model
        
        # Get MLP dimensions (auto-calculated if not explicitly set)
        # Uses self.config (single source of truth) - no need to pass feature_config
        static_mlp_dims = model_config.get_mlp_dims(modality='static')
        motor_mlp_dims = model_config.get_mlp_dims(modality='part3')  # motor uses part3
        nonmotor_mlp_dims = model_config.get_mlp_dims(modality='part1')  # nonmotor uses part1
        med_mlp_dims = model_config.get_mlp_dims(modality='med')
        
        # 1. Modality embeddings
        self.static_embedding = StaticFeatureEmbedding(
            n_static,
            d_model,
            static_mlp_dims,
            model_config.dropout
        )
        
        self.motor_embedding = VisitFeatureEmbedding(
            n_motor,
            d_model,
            motor_mlp_dims,
            model_config.dropout
        )
        
        self.nonmotor_embedding = VisitFeatureEmbedding(
            n_nonmotor,
            d_model,
            nonmotor_mlp_dims,
            model_config.dropout
        )
        
        self.med_embedding = VisitFeatureEmbedding(
            n_med,
            d_model,
            med_mlp_dims,
            model_config.dropout
        )
        
        # 2. Visit token builder
        self.visit_builder = VisitTokenBuilder(d_model, model_config.dropout)
        
        # 3. Time encoding
        self.time_encoding = SinusoidalTimeEncoding(d_model, model_config.max_time_months)
        
        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=model_config.n_heads,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            activation=model_config.activation,
            batch_first=True,
            norm_first=True  # Pre-LN architecture (more stable)
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 5. Prediction heads
        n_targets = len(model_config.predict_totals)
        self.prediction_heads = MultiTaskHead(
            d_model,
            n_targets=n_targets,
            next_visit_hidden=model_config.next_visit_hidden_dims,
            slope_hidden=model_config.slope_hidden_dims,
            dropout=model_config.dropout,
            pooling='mean'
        )
        
    def forward(
        self,
        static_values: torch.Tensor,
        static_mask: torch.Tensor,
        motor_values: torch.Tensor,
        motor_mask: torch.Tensor,
        nonmotor_values: torch.Tensor,
        nonmotor_mask: torch.Tensor,
        med_values: torch.Tensor,
        med_mask: torch.Tensor,
        time_months: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            static_values: [batch, n_static_features]
            static_mask: [batch, n_static_features] - 1 if missing
            motor_values: [batch, seq_len, n_motor_features]
            motor_mask: [batch, seq_len, n_motor_features] - 1 if missing
            nonmotor_values: [batch, seq_len, n_nonmotor_features]
            nonmotor_mask: [batch, seq_len, n_nonmotor_features] - 1 if missing
            med_values: [batch, seq_len, n_med_features]
            med_mask: [batch, seq_len, n_med_features] - 1 if missing
            time_months: [batch, seq_len] - months since baseline
            attention_mask: [batch, seq_len] - 1 for valid, 0 for padding
            
        Returns:
            dict with:
                - 'next_visit': [batch, seq_len, n_targets] - predicted UPDRS totals for next visit
                - 'slope': [batch] - predicted progression slope
                - 'hidden_states': [batch, seq_len, d_model] - transformer outputs
        """
        # 1. Embed each modality
        static_emb = self.static_embedding(static_values, static_mask)
        motor_emb = self.motor_embedding(motor_values, motor_mask)
        nonmotor_emb = self.nonmotor_embedding(nonmotor_values, nonmotor_mask)
        med_emb = self.med_embedding(med_values, med_mask)
        
        # 2. Build visit tokens
        visit_tokens = self.visit_builder(static_emb, motor_emb, nonmotor_emb, med_emb)
        
        # 3. Add time encoding
        time_emb = self.time_encoding(time_months)
        visit_tokens = visit_tokens + time_emb
        
        # 4. Create attention mask for transformer
        # Transformer expects: True for positions to MASK (opposite of our convention)
        if attention_mask is not None:
            # Convert: 1 (valid) -> False, 0 (padding) -> True
            transformer_mask = (attention_mask == 0)
        else:
            transformer_mask = None
        
        # 5. Pass through transformer
        hidden_states = self.transformer(
            visit_tokens,
            src_key_padding_mask=transformer_mask
        )
        
        # 6. Make predictions
        predictions = self.prediction_heads(hidden_states, attention_mask)
        
        # Add hidden states to output
        predictions['hidden_states'] = hidden_states
        
        return predictions
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        lambda_slope: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: dict from forward pass
            targets: dict with 'next_visit' [batch, seq_len, n_targets] and 'slope' [batch]
            attention_mask: [batch, seq_len] - 1 for valid positions
            lambda_slope: weight for slope loss
            
        Returns:
            dict with 'loss', 'loss_next_visit', 'loss_slope'
        """
        # Next-visit prediction loss (multiple UPDRS totals)
        next_visit_preds = predictions['next_visit']  # [batch, seq_len, n_targets]
        next_visit_targets = targets['next_visit']  # [batch, seq_len, n_targets]
        
        if attention_mask is not None:
            # Only compute loss for valid positions
            # Shift mask by 1 (predicting t+1 from t)
            valid_mask = attention_mask[:, :-1].unsqueeze(-1)  # [batch, seq_len-1, 1] for broadcasting
            next_visit_preds = next_visit_preds[:, :-1, :]  # [batch, seq_len-1, n_targets]
            next_visit_targets = next_visit_targets[:, 1:, :]  # [batch, seq_len-1, n_targets] - Targets are shifted
            
            # Masked MSE across all targets
            mse = (next_visit_preds - next_visit_targets) ** 2  # [batch, seq_len-1, n_targets]
            loss_next_visit = (mse * valid_mask).sum() / (valid_mask.sum() * next_visit_preds.shape[-1]).clamp(min=1)
        else:
            # Simple MSE without mask (averaged across all targets)
            next_visit_preds = next_visit_preds[:, :-1, :]  # [batch, seq_len-1, n_targets]
            next_visit_targets = next_visit_targets[:, 1:, :]  # [batch, seq_len-1, n_targets]
            loss_next_visit = nn.functional.mse_loss(next_visit_preds, next_visit_targets)
        
        # Slope prediction loss
        slope_preds = predictions['slope']  # [batch]
        slope_targets = targets['slope']  # [batch]
        
        # Filter out patients without slope targets (NaN or -999)
        valid_slopes = ~torch.isnan(slope_targets) & (slope_targets != -999)
        if valid_slopes.any():
            loss_slope = nn.functional.mse_loss(
                slope_preds[valid_slopes],
                slope_targets[valid_slopes]
            )
        else:
            loss_slope = torch.tensor(0.0, device=slope_preds.device)
        
        # Combined loss
        total_loss = loss_next_visit + lambda_slope * loss_slope
        
        return {
            'loss': total_loss,
            'loss_next_visit': loss_next_visit,
            'loss_slope': loss_slope
        }


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from training.config import get_default_config
    
    print("Testing V1 Model...")
    print("=" * 80)
    
    # Get config
    config = get_default_config()
    
    # Create model
    model = V1MultimodalTransformer(config)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy data
    batch_size = 4
    seq_len = 10
    
    n_static = len(config.features.static_features)
    n_motor = len(config.features.motor_features)
    n_nonmotor = len(config.features.nonmotor_features)
    n_med = len(config.features.medication_features)
    
    print(f"\nCreating dummy data:")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"  Static features: {n_static}")
    print(f"  Motor features: {n_motor}")
    print(f"  Non-motor features: {n_nonmotor}")
    print(f"  Medication features: {n_med}")
    
    static_values = torch.randn(batch_size, n_static)
    static_mask = torch.bernoulli(torch.ones_like(static_values) * 0.1)
    
    motor_values = torch.randn(batch_size, seq_len, n_motor)
    motor_mask = torch.bernoulli(torch.ones_like(motor_values) * 0.15)
    
    nonmotor_values = torch.randn(batch_size, seq_len, n_nonmotor)
    nonmotor_mask = torch.bernoulli(torch.ones_like(nonmotor_values) * 0.2)
    
    med_values = torch.randn(batch_size, seq_len, n_med)
    med_mask = torch.bernoulli(torch.ones_like(med_values) * 0.05)
    
    time_months = torch.linspace(0, 48, seq_len).unsqueeze(0).expand(batch_size, -1)
    
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 8:] = 0
    attention_mask[1, 6:] = 0
    
    # Forward pass
    print("\nRunning forward pass...")
    predictions = model(
        static_values, static_mask,
        motor_values, motor_mask,
        nonmotor_values, nonmotor_mask,
        med_values, med_mask,
        time_months,
        attention_mask
    )
    
    print(f"\nPredictions:")
    print(f"  Next-visit shape: {predictions['next_visit'].shape}")
    print(f"  Slope shape: {predictions['slope'].shape}")
    print(f"  Hidden states shape: {predictions['hidden_states'].shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    n_targets = len(config.model.predict_totals)
    targets = {
        'next_visit': torch.randn(batch_size, seq_len, n_targets),  # Simulate all UPDRS totals
        'slope': torch.randn(batch_size) * 0.5  # Simulate slopes
    }
    
    losses = model.compute_loss(predictions, targets, attention_mask, lambda_slope=0.2)
    
    print(f"\nLosses:")
    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  Next-visit loss: {losses['loss_next_visit'].item():.4f}")
    print(f"  Slope loss: {losses['loss_slope'].item():.4f}")
    
    print("\n" + "=" * 80)
    print("✓ V1 Model working correctly!")
