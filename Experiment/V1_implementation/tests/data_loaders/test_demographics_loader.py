"""
Tests for DemographicsLoader
"""

import pytest
import pandas as pd


class TestDemographicsLoader:
    """Test DemographicsLoader"""
    
    @pytest.mark.requires_data
    def test_demographics_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test DemographicsLoader can load data with full validation and summary"""
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        # Basic assertions
        assert isinstance(demo_df, pd.DataFrame)
        assert 'PATNO' in demo_df.columns
        assert len(demo_df) > 0
        
        # Validate
        loader.validate(demo_df)
        
        # Get summary
        summary = loader.get_summary(demo_df)
        unique_patients = demo_df['PATNO'].nunique()
        
        # Summary assertions
        assert unique_patients > 0
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        
        # Check distributions
        if 'ENROLL_AGE' in demo_df.columns:
            assert demo_df['ENROLL_AGE'].min() >= 0
            assert demo_df['ENROLL_AGE'].max() <= 150
            # Check age statistics are reasonable
            assert demo_df['ENROLL_AGE'].mean() > 0
        
        # Check cohort distribution if available
        if 'COHORT_DEFINITION' in demo_df.columns:
            # Count cohorts by unique PATNO (one patient counted once)
            unique_cohort = demo_df[['PATNO', 'COHORT_DEFINITION']].drop_duplicates(subset='PATNO')
            cohort_counts = unique_cohort['COHORT_DEFINITION'].value_counts()
            assert len(cohort_counts) > 0


@pytest.mark.requires_data
@pytest.mark.slow
class TestDemographicsLoaderIntegration:
    """Integration tests for DemographicsLoader matching __main__ block functionality"""
    
    def test_demographics_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test DemographicsLoader with full validation and summary"""
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        # Validate
        loader.validate(demo_df)
        
        # Get summary
        summary = loader.get_summary(demo_df)
        unique_patients = demo_df['PATNO'].nunique()
        
        assert unique_patients > 0
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        
        # Check for expected columns
        assert 'PATNO' in demo_df.columns
        
        # Check distributions if available
        if 'ENROLL_AGE' in demo_df.columns:
            assert demo_df['ENROLL_AGE'].min() >= 0
            assert demo_df['ENROLL_AGE'].max() <= 150
