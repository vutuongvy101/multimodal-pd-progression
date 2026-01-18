"""
Tests for base loader classes
"""

import pytest
import pandas as pd


class TestBaseLoaders:
    """Test base loader classes"""
    
    def test_base_data_loader_abstract(self):
        """Test that BaseDataLoader is abstract and cannot be instantiated"""
        from data.base_loader import BaseDataLoader
        
        with pytest.raises(TypeError):
            # Should fail because BaseDataLoader is abstract
            BaseDataLoader("/fake/path")
    
    def test_static_data_loader_interface(self):
        """Test StaticDataLoader interface"""
        from data.base_loader import StaticDataLoader
        
        # StaticDataLoader should have load() method
        assert hasattr(StaticDataLoader, 'load')
        # Should be callable (not abstract for StaticDataLoader)
        assert callable(StaticDataLoader.load)
    
    def test_longitudinal_data_loader_interface(self):
        """Test LongitudinalDataLoader interface"""
        from data.base_loader import LongitudinalDataLoader
        
        # LongitudinalDataLoader should have load() method
        assert hasattr(LongitudinalDataLoader, 'load')
        assert callable(LongitudinalDataLoader.load)


class TestLoaderInterface:
    """Test common loader interface methods"""
    
    def test_loader_validation(self, test_config):
        """Test that loaders have validate method"""
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        assert hasattr(loader, 'validate')
        assert callable(loader.validate)
    
    def test_loader_summary(self, test_config):
        """Test that loaders have get_summary method"""
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        assert hasattr(loader, 'get_summary')
        assert callable(loader.get_summary)
