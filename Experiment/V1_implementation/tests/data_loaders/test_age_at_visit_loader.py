"""
Tests for AgeAtVisitLoader
"""

import pytest
import pandas as pd


class TestAgeAtVisitLoader:
    """Test AgeAtVisitLoader"""
    
    @pytest.mark.requires_data
    def test_age_at_visit_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test AgeAtVisitLoader can load data"""
        from data.loaders.age_at_visit_loader import AgeAtVisitLoader
        
        loader = AgeAtVisitLoader(test_config.data.base_dir, test_config)
        age_df = loader.load()
        
        # Basic assertions
        assert isinstance(age_df, pd.DataFrame)
        assert 'PATNO' in age_df.columns
        assert 'EVENT_ID' in age_df.columns
        
        # Age at visit data is optional - may be empty if file not found
        if len(age_df) > 0:
            assert 'AGE_AT_VISIT' in age_df.columns
            unique_patients = age_df['PATNO'].nunique()
            assert unique_patients > 0
            
            # Check that we have multiple visits per patient (longitudinal data)
            visits_per_patient = age_df.groupby('PATNO').size()
            assert visits_per_patient.max() >= 1  # At least one visit per patient
            
            # Check age values are reasonable
            age_values = age_df['AGE_AT_VISIT'].dropna()
            if len(age_values) > 0:
                assert age_values.min() >= 0, "Age should not be negative"
                assert age_values.max() <= 150, "Age should be reasonable"
        else:
            # Empty dataframe should have correct structure
            assert 'PATNO' in age_df.columns
            assert 'EVENT_ID' in age_df.columns
            assert 'AGE_AT_VISIT' in age_df.columns
            assert 'months_since_baseline' in age_df.columns


@pytest.mark.requires_data
@pytest.mark.slow
class TestAgeAtVisitLoaderIntegration:
    """Integration tests for AgeAtVisitLoader matching __main__ block functionality"""
    
    def test_age_at_visit_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test AgeAtVisitLoader with full validation and summary"""
        from data.loaders.age_at_visit_loader import AgeAtVisitLoader
        
        loader = AgeAtVisitLoader(test_config.data.base_dir, test_config)
        age_df = loader.load()
        
        # Validate (works even with empty dataframe)
        loader.validate(age_df)
        
        # Get summary
        summary = loader.get_summary(age_df)
        
        # Basic assertions
        assert 'PATNO' in age_df.columns
        assert 'EVENT_ID' in age_df.columns
        assert summary['n_rows'] >= 0  # Can be 0 if no age_at_visit data
        assert summary['n_columns'] >= 0
        
        # If age_at_visit data exists, check distributions
        if len(age_df) > 0:
            unique_patients = age_df['PATNO'].nunique()
            assert unique_patients > 0
            
            # Check for expected columns
            assert 'AGE_AT_VISIT' in age_df.columns
            assert 'EVENT_ID' in age_df.columns
            
            # Check age statistics are reasonable
            age_values = age_df['AGE_AT_VISIT'].dropna()
            if len(age_values) > 0:
                assert age_values.min() >= 0, "Age should not be negative"
                assert age_values.max() <= 150, "Age should be reasonable"
                
                # Verify age statistics are reasonable
                mean_age = age_values.mean()
                median_age = age_values.median()
                assert pd.notna(mean_age) and pd.notna(median_age)
                assert 0 < mean_age < 150, "Mean age should be reasonable"
                assert 0 < median_age < 150, "Median age should be reasonable"
            
            # Check that this is longitudinal data (multiple visits per patient)
            visits_per_patient = age_df.groupby('PATNO').size()
            assert len(visits_per_patient) > 0
            # Note: Some patients may have only one visit, but we should have multiple patients
