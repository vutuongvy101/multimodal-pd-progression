"""
Tests for NonMotorAssessmentsLoader
"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestNonMotorAssessmentsLoader:
    """Test NonMotorAssessmentsLoader"""
    
    @pytest.mark.requires_data
    def test_non_motor_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test NonMotorAssessmentsLoader can load data with full validation and summary"""
        from data.loaders.non_motor_loader import NonMotorAssessmentsLoader
        
        loader = NonMotorAssessmentsLoader(test_config.data.base_dir, test_config)
        non_motor_df = loader.load()
        
        # Basic assertions
        assert isinstance(non_motor_df, pd.DataFrame)
        assert 'PATNO' in non_motor_df.columns
        assert len(non_motor_df) > 0
        
        # Validate
        loader.validate(non_motor_df)
        
        # Get summary
        summary = get_df_summary(non_motor_df)
        unique_patients = non_motor_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in non_motor_df.columns
        assert 'EVENT_ID' in non_motor_df.columns or 'VISIT_ID' in non_motor_df.columns
        
        # Check that non-motor assessment columns exist (excluding metadata columns)
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        non_motor_cols = [col for col in non_motor_df.columns if col not in metadata_cols]
        assert len(non_motor_cols) > 0, "No non-motor assessment columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestNonMotorAssessmentsLoaderIntegration:
    """Integration tests for NonMotorAssessmentsLoader matching __main__ block functionality"""
    
    def test_non_motor_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test NonMotorAssessmentsLoader with full validation and summary"""
        from data.loaders.non_motor_loader import NonMotorAssessmentsLoader
        
        loader = NonMotorAssessmentsLoader(test_config.data.base_dir, test_config)
        non_motor_df = loader.load()
        
        # Validate
        loader.validate(non_motor_df)
        
        # Get summary
        summary = get_df_summary(non_motor_df)
        unique_patients = non_motor_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in non_motor_df.columns
        
        # Check non-motor assessment distributions (mean ± std) if available
        # This matches the __main__ block logic for showing distributions
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        non_motor_cols = [col for col in non_motor_df.columns if col not in metadata_cols]
        
        if len(non_motor_cols) > 0:
            # Verify that at least some non-motor assessments have valid values
            # Similar to __main__ block: print mean ± std for each column
            valid_cols = []
            for col in non_motor_cols:
                n_valid = non_motor_df[col].notna().sum()
                if n_valid > 0:
                    mean_val = non_motor_df[col].mean()
                    std_val = non_motor_df[col].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) or pd.notna(std_val)
                    valid_cols.append(col)
            
            # Assert that at least some non-motor columns have valid data
            assert len(valid_cols) > 0, "No non-motor assessment columns with valid data"
