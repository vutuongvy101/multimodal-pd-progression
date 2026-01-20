"""
Tests for ClinicalAssessmentsLoader
"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestClinicalAssessmentsLoader:
    """Test ClinicalAssessmentsLoader"""
    
    @pytest.mark.requires_data
    def test_clinical_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test ClinicalAssessmentsLoader can load data with full validation and summary"""
        from data.loaders.clinical_loader import ClinicalAssessmentsLoader
        
        loader = ClinicalAssessmentsLoader(test_config.data.base_dir, test_config)
        clinical_df = loader.load()
        
        # Basic assertions
        assert isinstance(clinical_df, pd.DataFrame)
        assert 'PATNO' in clinical_df.columns
        assert len(clinical_df) > 0
        
        # Validate
        loader.validate(clinical_df)
        
        # Get summary
        summary = get_df_summary(clinical_df)
        unique_patients = clinical_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in clinical_df.columns
        assert 'EVENT_ID' in clinical_df.columns or 'VISIT_ID' in clinical_df.columns
        
        # Check that clinical assessment columns exist (excluding metadata columns)
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        clinical_cols = [col for col in clinical_df.columns if col not in metadata_cols]
        assert len(clinical_cols) > 0, "No clinical assessment columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestClinicalAssessmentsLoaderIntegration:
    """Integration tests for ClinicalAssessmentsLoader matching __main__ block functionality"""
    
    def test_clinical_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test ClinicalAssessmentsLoader with full validation and summary"""
        from data.loaders.clinical_loader import ClinicalAssessmentsLoader
        
        loader = ClinicalAssessmentsLoader(test_config.data.base_dir, test_config)
        clinical_df = loader.load()
        
        # Validate
        loader.validate(clinical_df)
        
        # Get summary
        summary = get_df_summary(clinical_df)
        unique_patients = clinical_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in clinical_df.columns
        
        # Check clinical assessment distributions (mean ± std) if available
        # This matches the __main__ block logic for showing distributions
        metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
        clinical_cols = [col for col in clinical_df.columns if col not in metadata_cols]
        
        if len(clinical_cols) > 0:
            # Verify that at least some clinical assessments have valid values
            # Similar to __main__ block: print mean ± std for each column
            valid_cols = []
            for col in clinical_cols:
                n_valid = clinical_df[col].notna().sum()
                if n_valid > 0:
                    mean_val = clinical_df[col].mean()
                    std_val = clinical_df[col].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) or pd.notna(std_val)
                    valid_cols.append(col)
            
            # Assert that at least some clinical columns have valid data
            assert len(valid_cols) > 0, "No clinical assessment columns with valid data"