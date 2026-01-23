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
           - Patient-level progression slopes (one per UPDRS total: NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope)
    
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
        self.enabled_modalities = model_config.enabled_modalities
        
        self._create_modality_embeddings(model_config, feature_config, d_model)
        
        self.visit_builder = VisitTokenBuilder(
            d_model, self.enabled_modalities, model_config.dropout
        )
        
        self.time_encoding = SinusoidalTimeEncoding(d_model, model_config.max_time_months)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=model_config.n_heads,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            activation=model_config.activation,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        n_targets = len(model_config.predict_totals)
        self.prediction_heads = MultiTaskHead(
            d_model,
            n_targets=n_targets,
            next_visit_hidden=model_config.next_visit_hidden_dims,
            slope_hidden=model_config.slope_hidden_dims,
            dropout=model_config.dropout,
            pooling='mean'
        )
    
    def _create_modality_embeddings(self, model_config, feature_config, d_model):
        """Create embedding modules for enabled modalities."""
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
                - 'slope': [batch, n_targets] - predicted progression slopes (one per UPDRS total)
                - 'hidden_states': [batch, seq_len, d_model] - transformer outputs
        
        Note: Only enabled modalities are embedded. Disabled modalities' inputs are ignored.
        """
        seq_len = time_months.shape[1]
        
        embeddings = self._embed_modalities(
            static_values, static_mask,
            motor_values, motor_mask,
            nonmotor_values, nonmotor_mask,
            med_values, med_mask,
            age_at_visit_values, age_at_visit_mask
        )
        
        visit_tokens = self._build_visit_tokens(embeddings, seq_len, time_months)
        hidden_states = self._apply_transformer(visit_tokens, attention_mask)
        predictions = self._make_predictions(hidden_states, attention_mask)
        
        return predictions
    
    def _embed_modalities(
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
        age_at_visit_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Embed enabled modalities."""
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
        
        return embeddings
    
    def _build_visit_tokens(
        self,
        embeddings: Dict[str, torch.Tensor],
        seq_len: int,
        time_months: torch.Tensor
    ) -> torch.Tensor:
        """Build visit tokens from modality embeddings and add time encoding."""
        visit_tokens = self.visit_builder(embeddings, seq_len)
        time_emb = self.time_encoding(time_months)
        return visit_tokens + time_emb
    
    def _apply_transformer(
        self,
        visit_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Apply transformer encoder to visit tokens."""
        transformer_mask = None
        if attention_mask is not None:
            transformer_mask = (attention_mask == 0)
        
        return self.transformer(visit_tokens, src_key_padding_mask=transformer_mask)
    
    def _make_predictions(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Make predictions from transformer hidden states."""
        predictions = self.prediction_heads(hidden_states, attention_mask)
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
                - 'slope': [batch, n_targets] - target slopes (one per UPDRS total)
            attention_mask: [batch, seq_len] - 1 for valid positions, 0 for padding
            lambda_slope: weight for slope loss
            
        Returns:
            dict with 'loss', 'loss_next_visit', 'loss_slope'
        """
        loss_next_visit = self._compute_next_visit_loss(
            predictions, targets, attention_mask
        )
        loss_slope = self._compute_slope_loss(predictions, targets)
        total_loss = loss_next_visit + lambda_slope * loss_slope
        
        return {
            'loss': total_loss,
            'loss_next_visit': loss_next_visit,
            'loss_slope': loss_slope
        }
    
    def _compute_next_visit_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute masked MSE loss for next-visit predictions."""
        next_visit_preds = predictions['next_visit']
        next_visit_targets = targets['next_visit']
        label_mask = targets.get('next_visit_mask', None)
        
        next_visit_preds = next_visit_preds[:, :-1, :]
        next_visit_targets = next_visit_targets[:, 1:, :]
        
        if attention_mask is not None:
            time_mask = attention_mask[:, :-1].unsqueeze(-1)
        else:
            time_mask = torch.ones_like(next_visit_preds[..., :1])
        
        if label_mask is not None:
            label_mask = label_mask[:, 1:, :]
        else:
            label_mask = torch.ones_like(next_visit_targets)
        
        combined_mask = time_mask * label_mask
        mse = (next_visit_preds - next_visit_targets) ** 2
        n_valid = combined_mask.sum().clamp(min=1.0)
        
        return (mse * combined_mask).sum() / n_valid
    
    def _compute_slope_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute masked MSE loss for slope predictions."""
        slope_preds = predictions['slope']
        slope_targets = targets['slope']
        valid_slope_mask = ~torch.isnan(slope_targets)
        
        if valid_slope_mask.any():
            # Mask out NaN targets before computing MSE to prevent NaN propagation
            # Replace NaN targets with 0 (they'll be masked out anyway)
            slope_targets_masked = torch.where(
                valid_slope_mask,
                slope_targets,
                torch.zeros_like(slope_targets)
            )
            mse_slope = (slope_preds - slope_targets_masked) ** 2
            n_valid_slopes = valid_slope_mask.sum().clamp(min=1.0)
            return (mse_slope * valid_slope_mask).sum() / n_valid_slopes
        else:
            return torch.tensor(0.0, device=slope_preds.device)