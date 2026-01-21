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
    V1 Model: Multimodal longitudinal transformer with configurable modalities.
    
    Architecture:
        1. Embed each ENABLED modality (static, motor, non-motor, medication)
        2. Build visit tokens by combining modality embeddings
        3. Add time encoding
        4. Pass through Transformer encoder
        5. Make predictions via two heads:
           - Next-visit UPDRS totals (NP1TOT, NP2TOT, NP3TOT, NP4TOT)
           - Patient-level progression slope
    
    Two types of masks are supported:
        A) Input missingness masks: "Is feature X observed at visit t?" (per modality)
        B) Label availability masks: "Is target Y available at visit t+1?" (for loss)
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
        
        d_model = model_config.d_model
        
        # Get enabled modalities (supports ablation studies)
        self.enabled_modalities = model_config.enabled_modalities
        
        # Create embeddings only for enabled modalities
        # This allows for ablation studies (e.g., train with only static + motor)
        
        if 'static' in self.enabled_modalities:
            n_static = len(feature_config.static_features)
            static_mlp_dims = model_config.get_mlp_dims(modality='static')
            self.static_embedding = StaticFeatureEmbedding(
                n_static, d_model, static_mlp_dims, model_config.dropout
            )
        
        if 'motor' in self.enabled_modalities:
            n_motor = len(feature_config.motor_features)
            motor_mlp_dims = model_config.get_mlp_dims(modality='motor')
            self.motor_embedding = VisitFeatureEmbedding(
                n_motor, d_model, motor_mlp_dims, model_config.dropout
            )
        
        if 'non_motor' in self.enabled_modalities:
            n_nonmotor = len(feature_config.non_motor_features)
            nonmotor_mlp_dims = model_config.get_mlp_dims(modality='non_motor')
            self.nonmotor_embedding = VisitFeatureEmbedding(
                n_nonmotor, d_model, nonmotor_mlp_dims, model_config.dropout
            )
        
        if 'medication' in self.enabled_modalities:
            n_med = len(feature_config.medication_features)
            med_mlp_dims = model_config.get_mlp_dims(modality='med')
            self.med_embedding = VisitFeatureEmbedding(
                n_med, d_model, med_mlp_dims, model_config.dropout
            )

        if 'age_at_visit' in self.enabled_modalities:
            n_age = len(feature_config.age_at_visit_features)
            age_mlp_dims = model_config.get_mlp_dims(modality='age_at_visit')
            self.age_at_visit_embedding = VisitFeatureEmbedding(
                n_age, d_model, age_mlp_dims, model_config.dropout
            )
        
        # 2. Visit token builder (with dynamic modality support)
        self.visit_builder = VisitTokenBuilder(
            d_model, self.enabled_modalities, model_config.dropout
        )
        
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
        age_at_visit_values: torch.Tensor,
        age_at_visit_mask: torch.Tensor,
        time_months: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            static_values: [batch, n_static_features]
            static_mask: [batch, n_static_features] - 1 if missing (Mask A: input missingness)
            motor_values: [batch, seq_len, n_motor_features]
            motor_mask: [batch, seq_len, n_motor_features] - 1 if missing
            nonmotor_values: [batch, seq_len, n_nonmotor_features]
            nonmotor_mask: [batch, seq_len, n_nonmotor_features] - 1 if missing
            med_values: [batch, seq_len, n_med_features]
            med_mask: [batch, seq_len, n_med_features] - 1 if missing
            age_at_visit_values: [batch, seq_len, n_age_features]
            age_at_visit_mask: [batch, seq_len, n_age_features] - 1 if missing
            time_months: [batch, seq_len] - months since baseline
            attention_mask: [batch, seq_len] - 1 for valid visit, 0 for padding
            
        Returns:
            dict with:
                - 'next_visit': [batch, seq_len, n_targets] - predicted UPDRS totals for next visit
                - 'slope': [batch] - predicted progression slope
                - 'hidden_states': [batch, seq_len, d_model] - transformer outputs
        
        Note: Only enabled modalities are embedded. Disabled modalities' inputs are ignored.
        """
        # Get sequence length from time tensor
        seq_len = time_months.shape[1]
        
        # 1. Embed only ENABLED modalities
        embeddings = {}
        
        if 'static' in self.enabled_modalities:
            embeddings['static'] = self.static_embedding(static_values, static_mask)
        
        if 'motor' in self.enabled_modalities:
            embeddings['motor'] = self.motor_embedding(motor_values, motor_mask)
        
        if 'non_motor' in self.enabled_modalities:
            embeddings['non_motor'] = self.nonmotor_embedding(nonmotor_values, nonmotor_mask)
        
        if 'medication' in self.enabled_modalities:
            embeddings['medication'] = self.med_embedding(med_values, med_mask)

        if 'age_at_visit' in self.enabled_modalities:
            embeddings['age_at_visit'] = self.age_at_visit_embedding(age_at_visit_values, age_at_visit_mask)
        
        # 2. Build visit tokens from enabled modalities
        visit_tokens = self.visit_builder(embeddings, seq_len)
        
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
        Compute multi-task loss with proper label availability masking.
        
        Uses two types of masks:
            A) attention_mask: "Does this visit position exist?" (not padding)
            B) label_mask: "Is the target label available for this position?"
        
        The combined mask = attention_mask AND label_mask ensures we only compute
        loss where BOTH the visit exists AND the target label is available.
        
        Args:
            predictions: dict from forward pass
            targets: dict with:
                - 'next_visit': [batch, seq_len, n_targets] - target UPDRS totals
                - 'next_visit_mask': [batch, seq_len, n_targets] - label availability (1=present, 0=missing)
                - 'slope': [batch] - target slopes
            attention_mask: [batch, seq_len] - 1 for valid positions, 0 for padding
            lambda_slope: weight for slope loss
            
        Returns:
            dict with 'loss', 'loss_next_visit', 'loss_slope'
        """
        # Next-visit prediction loss (multiple UPDRS totals)
        next_visit_preds = predictions['next_visit']  # [batch, seq_len, n_targets]
        next_visit_targets = targets['next_visit']  # [batch, seq_len, n_targets]
        
        # Get label availability mask (Mask B)
        # This tells us which targets are available at each time step
        label_mask = targets.get('next_visit_mask', None)  # [batch, seq_len, n_targets]
        
        # Shift for next-visit prediction: predict t+1 from representation at t
        next_visit_preds = next_visit_preds[:, :-1, :]      # [batch, seq_len-1, n_targets]
        next_visit_targets = next_visit_targets[:, 1:, :]   # [batch, seq_len-1, n_targets]
        
        # Build combined mask
        if attention_mask is not None:
            # Mask A: which visits exist (shifted for next-visit prediction)
            time_mask = attention_mask[:, :-1].unsqueeze(-1)  # [batch, seq_len-1, 1]
        else:
            time_mask = torch.ones_like(next_visit_preds[..., :1])  # [batch, seq_len-1, 1]
        
        if label_mask is not None:
            # Mask B: which labels are available (shifted to match targets at t+1)
            label_mask = label_mask[:, 1:, :]  # [batch, seq_len-1, n_targets]
        else:
            # Fallback: assume all labels are available
            label_mask = torch.ones_like(next_visit_targets)
        
        # Combined mask: visit exists AND label available for that target
        # This is the key insight: we don't filter timelines, we mask the loss
        combined_mask = time_mask * label_mask  # [batch, seq_len-1, n_targets]
        
        # Compute masked MSE
        mse = (next_visit_preds - next_visit_targets) ** 2  # [batch, seq_len-1, n_targets]
        
        # Average only over positions where combined_mask is 1
        n_valid = combined_mask.sum().clamp(min=1.0)
        loss_next_visit = (mse * combined_mask).sum() / n_valid
        
        # Slope prediction loss
        slope_preds = predictions['slope']  # [batch]
        slope_targets = targets['slope']  # [batch]
        
        # Filter out patients without slope targets (NaN)
        valid_slopes = ~torch.isnan(slope_targets)
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