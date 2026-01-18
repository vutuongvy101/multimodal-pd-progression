"""
Tests for GeneticsLoader
"""

import pytest
import pandas as pd


class TestGeneticsLoader:
    """Test GeneticsLoader"""
    
    @pytest.mark.requires_data
    def test_genetics_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test GeneticsLoader can load data with full validation and summary"""
        from data.loaders.genetics_loader import GeneticsLoader
        
        loader = GeneticsLoader(test_config.data.base_dir, test_config)
        genetics_df = loader.load()
        
        # Basic assertions
        assert isinstance(genetics_df, pd.DataFrame)
        assert 'PATNO' in genetics_df.columns
        assert len(genetics_df) > 0
        
        # Validate
        loader.validate(genetics_df)
        
        # Get summary
        summary = loader.get_summary(genetics_df)
        unique_patients = genetics_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in genetics_df.columns
        
        # Check that genetics columns exist (excluding PATNO)
        genetics_cols = [c for c in genetics_df.columns if c != 'PATNO']
        assert len(genetics_cols) > 0, "No genetics columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestGeneticsLoaderIntegration:
    """Integration tests for GeneticsLoader matching __main__ block functionality"""
    
    def test_genetics_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test GeneticsLoader with full validation and summary"""
        from data.loaders.genetics_loader import GeneticsLoader
        
        loader = GeneticsLoader(test_config.data.base_dir, test_config)
        genetics_df = loader.load()
        
        # Validate
        loader.validate(genetics_df)
        
        # Get summary
        summary = loader.get_summary(genetics_df)
        unique_patients = genetics_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        assert 'PATNO' in genetics_df.columns
        
        # Check missingness (genetics data often has missing values)
        # Similar to __main__ block: show missingness by column
        missing_pct = (genetics_df.isnull().sum() / len(genetics_df) * 100).sort_values(ascending=False)
        
        # Verify missingness data exists
        assert len(missing_pct) > 0
        
        # Check that some columns have data (not all missing)
        non_missing_cols = missing_pct[missing_pct < 100]
        assert len(non_missing_cols) > 0, "All genetics columns are missing"
        
        # Verify PATNO has no missing values (should be 0%)
        if 'PATNO' in missing_pct.index:
            assert missing_pct['PATNO'] == 0, "PATNO should have no missing values"
