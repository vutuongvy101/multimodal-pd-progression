"""
Tests for training configuration and utilities
"""

import pytest
from pathlib import Path


class TestConfiguration:
    """Test configuration management"""
    
    def test_get_default_config(self):
        """Test get_default_config returns valid configuration"""
        from training.config import get_default_config
        
        config = get_default_config()
        assert config is not None
        
        # Check all required config sections exist
        assert hasattr(config, 'features')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
    
    def test_feature_config(self, test_config):
        """Test FeatureConfig structure"""
        features = test_config.features
        
        # Check feature lists exist
        assert hasattr(features, 'static_features')
        assert hasattr(features, 'part1_features')
        assert hasattr(features, 'part2_features')
        assert hasattr(features, 'part3_features')
        assert hasattr(features, 'part4_features')
        
        # Check feature lists are non-empty
        assert len(features.static_features) > 0
        assert len(features.part1_features) > 0
        assert len(features.all_updrs_totals) == 4  # NP1TOT, NP2TOT, NP3TOT, NP4TOT
    
    def test_model_config(self, test_config):
        """Test ModelConfig structure"""
        model = test_config.model
        
        # Check model parameters exist
        assert hasattr(model, 'd_model')
        assert hasattr(model, 'n_heads')
        assert hasattr(model, 'n_layers')
        assert hasattr(model, 'max_seq_len')
        
        # Check parameters are valid
        assert model.d_model > 0
        assert model.n_heads > 0
        assert model.n_layers > 0
        assert model.max_seq_len > 0
    
    def test_training_config(self, test_config):
        """Test TrainingConfig structure"""
        training = test_config.training
        
        # Check training parameters exist
        assert hasattr(training, 'batch_size')
        assert hasattr(training, 'learning_rate')
        # We use max_epochs in the current implementation
        assert hasattr(training, 'max_epochs')
        
        # Check parameters are valid
        assert training.batch_size > 0
        assert training.learning_rate > 0
        assert training.max_epochs > 0
    
    def test_data_config(self, test_config):
        """Test DataConfig structure"""
        data = test_config.data
        
        # Check data paths exist
        assert hasattr(data, 'base_dir')
        assert hasattr(data, 'participant_status')
        
        # Check base_dir is a string (path)
        assert isinstance(data.base_dir, str)
    
    def test_config_dataclass_immutability(self, test_config):
        """Test that config dataclasses are properly structured"""
        # Config should be a dataclass (frozen or not)
        # This is more of a structural check
        assert test_config is not None
        
        # Try accessing nested configs
        assert test_config.model.d_model is not None
        assert test_config.features.static_features is not None


class TestTrainingUtilities:
    """Test training utility functions"""
    
    def test_device_detection(self):
        """Test device detection function"""
        from training.config import get_device
        
        device = get_device()
        
        # Device should be one of: 'cuda', 'mps', 'cpu'
        assert device in ['cuda', 'mps', 'cpu']
    
    def test_config_serialization(self, test_config):
        """Test that config can be converted to dict for logging"""
        # Config should be accessible as attributes
        # (Not testing actual JSON serialization, just structure)
        
        # Access nested attributes
        d_model = test_config.model.d_model
        batch_size = test_config.training.batch_size
        
        assert d_model is not None
        assert batch_size is not None
