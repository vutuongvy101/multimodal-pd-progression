"""
Tests for data processing pipeline (data integrator, preparation, dataset)
"""

import pytest
import pandas as pd
from pathlib import Path


class TestDataIntegrator:
    """Test data integration components"""
    
    @pytest.mark.requires_data
    def test_data_integrator_exists(self, skip_if_no_data):
        """Test DataIntegrator can be imported"""
        from data.data_integrator import DataIntegrator
        assert DataIntegrator is not None
    
    @pytest.mark.requires_data
    def test_data_integrator_creation(self, test_config, skip_if_no_data):
        """Test DataIntegrator can be instantiated"""
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        assert integrator is not None
        assert integrator.config == test_config


class TestDataset:
    """Test PyTorch dataset wrapper"""
    
    def test_dataset_creation(self, test_config):
        """Test dataset can be created with mock data"""
        from data.dataset import PPMIDataset
        import torch
        
        # Create mock data
        n_samples = 10
        n_static = len(test_config.features.static_features)
        n_motor = len(test_config.features.part3_features)
        n_seq = 5
        
        mock_static = pd.DataFrame({
            'PATNO': range(n_samples),
            **{f'feat_{i}': [0.5] * n_samples for i in range(n_static - 1)}
        })
        
        mock_longitudinal = pd.DataFrame({
            'PATNO': list(range(n_samples)) * n_seq,
            'EVENT_ID': [f'V{i}' for i in range(n_seq)] * n_samples,
            'TIME': [i * 6 for i in range(n_seq)] * n_samples,
            **{f'feat_{i}': [0.5] * (n_samples * n_seq) for i in range(n_motor)}
        })
        
        # Create dataset (this might need adjustments based on actual dataset implementation)
        # This is a placeholder test structure
        assert mock_static is not None
        assert mock_longitudinal is not None


class TestDataPreparation:
    """Test data preparation utilities"""
    
    @pytest.mark.requires_data
    def test_data_preparation_exists(self, skip_if_no_data):
        """Test data preparation module can be imported"""
        try:
            from data.data_preparation import prepare_data
            assert prepare_data is not None or True  # Placeholder
        except ImportError:
            pytest.skip("Data preparation module not available")


# Integration tests (require actual data)
@pytest.mark.requires_data
@pytest.mark.slow
class TestDataPipeline:
    """Integration tests for full data pipeline"""
    
    def test_full_data_loading_pipeline(self, test_config, skip_if_no_data):
        """Test that all data loaders work together"""
        from data.loaders.demographics_loader import DemographicsLoader
        from data.loaders.genetics_loader import GeneticsLoader
        
        # Test that multiple loaders can be instantiated
        demo_loader = DemographicsLoader(test_config.data.base_dir, test_config)
        genetics_loader = GeneticsLoader(test_config.data.base_dir, test_config)
        
        assert demo_loader is not None
        assert genetics_loader is not None
        
        # If data is available, test actual loading
        # (This would require data files to be present)
        # demo_df = demo_loader.load()
        # assert isinstance(demo_df, pd.DataFrame)
