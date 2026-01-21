"""Tests for BehavioralLoader"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestBehavioralLoader:
    """Test BehavioralLoader"""
    
    @pytest.mark.requires_data
    def test_behavioral_loader_load(self, test_config_v2, skip_if_no_data, capsys):
        """Test BehavioralLoader can load data with full validation and summary"""
        from data.loaders_v2 import BehavioralLoader
        
        loader = BehavioralLoader(test_config_v2.data.base_dir, test_config_v2)
        behavioral_df = loader.load()
        
        # Basic assertions
        assert isinstance(behavioral_df, pd.DataFrame)
        assert 'PATNO' in behavioral_df.columns
        assert len(behavioral_df) > 0
        
        # Validate
        loader.validate(behavioral_df)

        # Get summary
        summary = get_df_summary(behavioral_df)
        unique_patients = behavioral_df['PATNO'].nunique()

        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in behavioral_df.columns
        assert 'EVENT_ID' in behavioral_df.columns or 'VISIT_ID' in behavioral_df.columns

        # Check that behavioral assessment columns exist (excluding metadata columns)
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        behavioral_cols = [col for col in behavioral_df.columns if col not in metadata_cols]
        assert len(behavioral_cols) > 0, "No behavioral assessment columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestBehavioralLoaderIntegration:
    """Integration tests for BehavioralLoader"""
    
    def test_behavioral_loader_integration(self, test_config_v2, skip_if_no_data, capsys):
        """Test BehavioralLoader v2 with full validation and summary"""
        from data.loaders_v2 import BehavioralLoader
        
        loader = BehavioralLoader(test_config_v2.data.base_dir, test_config_v2)
        behavioral_df = loader.load()
        
        # Validate
        loader.validate(behavioral_df)
        
        # Get summary
        summary = get_df_summary(behavioral_df)
        unique_patients = behavioral_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in behavioral_df.columns

        # Check behavioral assessment distributions (mean ± std) if available
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        behavioral_cols = [col for col in behavioral_df.columns if col not in metadata_cols]

        if len(behavioral_cols) > 0:
            # Verify that at least some behavioral assessments have valid values
            valid_cols = []
            for col in behavioral_cols:
                n_valid = behavioral_df[col].notna().sum()
                if n_valid > 0:
                    mean_val = behavioral_df[col].mean()
                    std_val = behavioral_df[col].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) or pd.notna(std_val)
                    valid_cols.append(col)

            # Assert that at least some behavioral columns have valid data
            assert len(valid_cols) > 0, "No behavioral assessment columns with valid data"