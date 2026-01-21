"""Tests for CognitiveLoader"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestCognitiveLoader:
    """Test CognitiveLoader"""
    
    @pytest.mark.requires_data
    def test_cognitive_loader_load(self, test_config_v2, skip_if_no_data, capsys):
        """Test CognitiveLoader can load data with full validation and summary"""
        from data.loaders_v2 import CognitiveLoader
        
        loader = CognitiveLoader(test_config_v2.data.base_dir, test_config_v2)
        cognitive_df = loader.load()
        
        # Basic assertions
        assert isinstance(cognitive_df, pd.DataFrame)
        assert 'PATNO' in cognitive_df.columns
        assert len(cognitive_df) > 0
        
        # Validate
        loader.validate(cognitive_df)

        # Get summary
        summary = get_df_summary(cognitive_df)
        unique_patients = cognitive_df['PATNO'].nunique()

        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in cognitive_df.columns
        assert 'EVENT_ID' in cognitive_df.columns or 'VISIT_ID' in cognitive_df.columns

        # Check that cognitive assessment columns exist (excluding metadata columns)
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        cognitive_cols = [col for col in cognitive_df.columns if col not in metadata_cols]
        assert len(cognitive_cols) > 0, "No cognitive assessment columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestCognitiveLoaderIntegration:
    """Integration tests for CognitiveLoader"""
    
    def test_cognitive_loader_integration(self, test_config_v2, skip_if_no_data, capsys):
        """Test CognitiveLoader v2 with full validation and summary"""
        from data.loaders_v2 import CognitiveLoader
        
        loader = CognitiveLoader(test_config_v2.data.base_dir, test_config_v2)
        cognitive_df = loader.load()
        
        # Validate
        loader.validate(cognitive_df)
        
        # Get summary
        summary = get_df_summary(cognitive_df)
        unique_patients = cognitive_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in cognitive_df.columns

        # Check cognitive assessment distributions (mean ± std) if available
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        cognitive_cols = [col for col in cognitive_df.columns if col not in metadata_cols]

        if len(cognitive_cols) > 0:
            # Verify that at least some cognitive assessments have valid values
            valid_cols = []
            for col in cognitive_cols:
                n_valid = cognitive_df[col].notna().sum()
                if n_valid > 0:
                    mean_val = cognitive_df[col].mean()
                    std_val = cognitive_df[col].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) or pd.notna(std_val)
                    valid_cols.append(col)

            # Assert that at least some cognitive columns have valid data
            assert len(valid_cols) > 0, "No cognitive assessment columns with valid data"