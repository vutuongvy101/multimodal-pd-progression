"""Tests for SleepLoader"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestSleepLoader:
    """Test SleepLoader"""
    
    @pytest.mark.requires_data
    def test_sleep_loader_load(self, test_config_v2, skip_if_no_data, capsys):
        """Test SleepLoader can load data with full validation and summary"""
        from data.loaders_v2 import SleepLoader
        
        loader = SleepLoader(test_config_v2.data.base_dir, test_config_v2)
        sleep_df = loader.load()
        
        # Basic assertions
        assert isinstance(sleep_df, pd.DataFrame)
        assert 'PATNO' in sleep_df.columns
        assert len(sleep_df) > 0
        
        # Validate
        loader.validate(sleep_df)

        # Get summary
        summary = get_df_summary(sleep_df)
        unique_patients = sleep_df['PATNO'].nunique()

        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in sleep_df.columns
        assert 'EVENT_ID' in sleep_df.columns or 'VISIT_ID' in sleep_df.columns

        # Check that sleep assessment columns exist (excluding metadata columns)
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        sleep_cols = [col for col in sleep_df.columns if col not in metadata_cols]
        assert len(sleep_cols) > 0, "No sleep assessment columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestSleepLoaderIntegration:
    """Integration tests for SleepLoader"""
    
    def test_sleep_loader_integration(self, test_config_v2, skip_if_no_data, capsys):
        """Test SleepLoader with full validation and summary"""
        from data.loaders_v2 import SleepLoader
        
        loader = SleepLoader(test_config_v2.data.base_dir, test_config_v2)
        sleep_df = loader.load()
        
        # Validate
        loader.validate(sleep_df)
        
        # Get summary
        summary = get_df_summary(sleep_df)
        unique_patients = sleep_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in sleep_df.columns

        # Check sleep assessment distributions (mean ± std) if available
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        sleep_cols = [col for col in sleep_df.columns if col not in metadata_cols]

        if len(sleep_cols) > 0:
            # Verify that at least some sleep assessments have valid values
            valid_cols = []
            for col in sleep_cols:
                n_valid = sleep_df[col].notna().sum()
                if n_valid > 0:
                    mean_val = sleep_df[col].mean()
                    std_val = sleep_df[col].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) or pd.notna(std_val)
                    valid_cols.append(col)

            # Assert that at least some sleep columns have valid data
            assert len(valid_cols) > 0, "No sleep assessment columns with valid data"