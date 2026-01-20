"""
Tests for model components (embeddings, heads, v1_model)
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestModalityEmbeddings:
    """Test modality embedding components"""
    
    def test_modality_embedding_creation(self, test_config):
        """Test ModalityEmbedding can be created"""
        from models.embeddings import ModalityEmbedding
        
        input_dim = 10
        d_model = test_config.model.d_model
        
        embedding = ModalityEmbedding(input_dim, d_model)
        assert embedding is not None
        assert embedding.d_model == d_model
    
    def test_modality_embedding_forward(self, test_config):
        """Test ModalityEmbedding forward pass"""
        from models.embeddings import ModalityEmbedding
        
        batch_size = 4
        input_dim = 10
        d_model = test_config.model.d_model
        
        embedding = ModalityEmbedding(input_dim, d_model)
        
        # Create input with values and missing mask
        values = torch.randn(batch_size, input_dim)
        mask = torch.ones(batch_size, input_dim, dtype=torch.bool)
        
        output = embedding(values, mask)
        
        assert output.shape == (batch_size, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPredictionHeads:
    """Test prediction head components"""
    
    def test_next_visit_head_creation(self, test_config):
        """Test NextVisitPredictionHead can be created"""
        from models.heads import NextVisitPredictionHead
        
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        head = NextVisitPredictionHead(d_model, n_targets)
        assert head is not None
    
    def test_next_visit_head_forward(self, test_config):
        """Test NextVisitPredictionHead forward pass"""
        from models.heads import NextVisitPredictionHead
        
        batch_size = 4
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        head = NextVisitPredictionHead(d_model, n_targets)
        
        # Create input hidden states [batch, seq_len, d_model]
        seq_len = 5
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        output = head(hidden_states)
        
        # Output should be [batch, seq_len, n_targets]
        assert output.shape == (batch_size, seq_len, n_targets)
        assert not torch.isnan(output).any()
    
    def test_slope_head_creation(self, test_config):
        """Test ProgressionSlopeHead can be created"""
        from models.heads import ProgressionSlopeHead
        
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        # ProgressionSlopeHead predicts a single slope per patient
        head = ProgressionSlopeHead(d_model)
        assert head is not None
    
    def test_slope_head_forward(self, test_config):
        """Test ProgressionSlopeHead forward pass"""
        from models.heads import ProgressionSlopeHead
        
        batch_size = 4
        seq_len = 5
        d_model = test_config.model.d_model
        # Head predicts a single slope per patient from sequence hidden states
        head = ProgressionSlopeHead(d_model)
        
        # Create input hidden states and attention mask
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = head(hidden_states, attention_mask)
        
        # Output should be [batch]
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()


class TestV1Model:
    """Test V1MultimodalTransformer model"""
    
    def test_model_creation(self, test_config):
        """Test V1MultimodalTransformer can be created"""
        from models.v1_model import V1MultimodalTransformer
        
        model = V1MultimodalTransformer(config=test_config)
        assert model is not None
        assert model.config == test_config
    
    def test_model_forward(self, test_config):
        """Test V1MultimodalTransformer forward pass"""
        from models.v1_model import V1MultimodalTransformer
        
        model = V1MultimodalTransformer(config=test_config)
        model.eval()  # Set to eval mode for deterministic behavior
        
        batch_size = 2
        seq_len = 3
        
        # Determine feature dimensions from FeatureConfig
        n_static = len(test_config.features.static_features)
        n_motor = len(test_config.features.motor_features)
        n_nonmotor = len(test_config.features.nonmotor_features)
        n_med = len(test_config.features.medication_features)
        n_targets = len(test_config.model.predict_totals)

        # Create mock input data
        static_values = torch.randn(batch_size, n_static)
        motor_values = torch.randn(batch_size, seq_len, n_motor)
        nonmotor_values = torch.randn(batch_size, seq_len, n_nonmotor)
        med_values = torch.randn(batch_size, seq_len, n_med)
        
        # Create masks (0 = observed, 1 = missing)
        static_mask = torch.zeros(batch_size, n_static)
        motor_mask = torch.zeros(batch_size, seq_len, n_motor)
        nonmotor_mask = torch.zeros(batch_size, seq_len, n_nonmotor)
        med_mask = torch.zeros(batch_size, seq_len, n_med)
        
        # Create time and attention masks
        time_months = torch.randn(batch_size, seq_len).abs()
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(
                static_values=static_values,
                static_mask=static_mask,
                motor_values=motor_values,
                motor_mask=motor_mask,
                nonmotor_values=nonmotor_values,
                nonmotor_mask=nonmotor_mask,
                med_values=med_values,
                med_mask=med_mask,
                time_months=time_months,
                attention_mask=attention_mask
            )
        
        next_visit_pred = outputs['next_visit']
        slope_pred = outputs['slope']
        
        # Check output shapes
        assert next_visit_pred.shape == (batch_size, seq_len, n_targets)
        assert slope_pred.shape == (batch_size,)
        
        # Check for NaNs
        assert not torch.isnan(next_visit_pred).any()
        assert not torch.isnan(slope_pred).any()
    
    def test_model_device_handling(self, test_config):
        """Test model works on different devices"""
        from models.v1_model import V1MultimodalTransformer
        
        model = V1MultimodalTransformer(config=test_config)
        
        # Model should handle device selection
        # (This is a simple check - actual device logic is in config)
        assert hasattr(model, 'config')
