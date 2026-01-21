"""Tests for SmellLoader"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestSmellLoader:
    """Test SmellLoader"""
    
    @pytest.mark.requires_data
    def test_smell_loader_load(self, test_config_v2, skip_if_no_data, capsys):
        """Test SmellLoader can load data with full validation and summary"""
        from data.loaders_v2 import SmellLoader
        
        loader = SmellLoader(test_config_v2.data.base_dir, test_config_v2)
        smell_df = loader.load()
        
        # Basic assertions
        assert isinstance(smell_df, pd.DataFrame)
        assert 'PATNO' in smell_df.columns
        assert len(smell_df) > 0
        
        # Validate
        loader.validate(smell_df)

        # Get summary
        summary = get_df_summary(smell_df)
        unique_patients = smell_df['PATNO'].nunique()

        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in smell_df.columns
        assert 'EVENT_ID' in smell_df.columns or 'VISIT_ID' in smell_df.columns

        # Check that smell assessment columns exist (excluding metadata columns)
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        smell_cols = [col for col in smell_df.columns if col not in metadata_cols]
        assert len(smell_cols) > 0, "No smell assessment columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestSmellLoaderIntegration:
    """Integration tests for SmellLoader"""
    
    def test_smell_loader_integration(self, test_config_v2, skip_if_no_data, capsys):
        """Test SmellLoader with full validation and summary"""
        from data.loaders_v2 import SmellLoader
        
        loader = SmellLoader(test_config_v2.data.base_dir, test_config_v2)
        smell_df = loader.load()
        
        # Validate
        loader.validate(smell_df)
        
        # Get summary
        summary = get_df_summary(smell_df)
        unique_patients = smell_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in smell_df.columns

        # Check smell assessment distributions (mean ± std) if available
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        smell_cols = [col for col in smell_df.columns if col not in metadata_cols]

        if len(smell_cols) > 0:
            # Verify that at least some smell assessments have valid values
            valid_cols = []
            for col in smell_cols:
                n_valid = smell_df[col].notna().sum()
                if n_valid > 0:
                    mean_val = smell_df[col].mean()
                    std_val = smell_df[col].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) or pd.notna(std_val)
                    valid_cols.append(col)

            # Assert that at least some smell columns have valid data
            assert len(valid_cols) > 0, "No smell assessment columns with valid data"