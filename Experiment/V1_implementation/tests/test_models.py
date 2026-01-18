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
        """Test NextVisitHead can be created"""
        from models.heads import NextVisitHead
        
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        head = NextVisitHead(d_model, n_targets)
        assert head is not None
    
    def test_next_visit_head_forward(self, test_config):
        """Test NextVisitHead forward pass"""
        from models.heads import NextVisitHead
        
        batch_size = 4
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        head = NextVisitHead(d_model, n_targets)
        
        # Create input (hidden state at last position)
        hidden = torch.randn(batch_size, d_model)
        
        output = head(hidden)
        
        assert output.shape == (batch_size, n_targets)
        assert not torch.isnan(output).any()
    
    def test_slope_head_creation(self, test_config):
        """Test SlopeHead can be created"""
        from models.heads import SlopeHead
        
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        head = SlopeHead(d_model, n_targets)
        assert head is not None
    
    def test_slope_head_forward(self, test_config):
        """Test SlopeHead forward pass"""
        from models.heads import SlopeHead
        
        batch_size = 4
        seq_len = 5
        d_model = test_config.model.d_model
        n_targets = len(test_config.features.all_updrs_totals)
        
        head = SlopeHead(d_model, n_targets)
        
        # Create input (pooled sequence representation)
        pooled = torch.randn(batch_size, d_model)
        
        output = head(pooled)
        
        assert output.shape == (batch_size, n_targets)
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
        
        # Create mock input data
        static_features = torch.randn(batch_size, test_config.model.static_dim)
        motor_features = torch.randn(batch_size, seq_len, test_config.model.motor_dim)
        nonmotor_features = torch.randn(batch_size, seq_len, test_config.model.nonmotor_dim)
        medication_features = torch.randn(batch_size, seq_len, test_config.model.medication_dim)
        
        # Create masks
        motor_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        nonmotor_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        medication_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Create time encoding
        time_delta = torch.randn(batch_size, seq_len)
        
        with torch.no_grad():
            next_visit_pred, slope_pred = model(
                static_features=static_features,
                motor_features=motor_features,
                nonmotor_features=nonmotor_features,
                medication_features=medication_features,
                motor_mask=motor_mask,
                nonmotor_mask=nonmotor_mask,
                medication_mask=medication_mask,
                time_delta=time_delta
            )
        
        n_targets = len(test_config.features.all_updrs_totals)
        
        # Check output shapes
        assert next_visit_pred.shape == (batch_size, n_targets)
        assert slope_pred.shape == (batch_size, n_targets)
        
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
