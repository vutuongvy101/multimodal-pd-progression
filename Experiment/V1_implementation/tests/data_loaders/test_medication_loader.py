"""
Tests for MedicationLoader
"""

import pytest
import pandas as pd


class TestMedicationLoader:
    """Test MedicationLoader"""
    
    @pytest.mark.requires_data
    def test_medication_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test MedicationLoader can load data (handles empty data gracefully)"""
        from data.loaders.medication_loader import MedicationLoader
        
        loader = MedicationLoader(test_config.data.base_dir, test_config)
        med_df = loader.load()
        
        # Basic assertions
        assert isinstance(med_df, pd.DataFrame)
        assert 'PATNO' in med_df.columns
        
        # Medication data is optional - may be empty
        if len(med_df) > 0:
            assert 'EVENT_ID' in med_df.columns or 'VISIT_ID' in med_df.columns
            unique_patients = med_df['PATNO'].nunique()
            assert unique_patients > 0
        else:
            # Empty dataframe should have correct structure
            assert 'PATNO' in med_df.columns
            assert 'EVENT_ID' in med_df.columns or 'VISIT_ID' in med_df.columns


@pytest.mark.requires_data
@pytest.mark.slow
class TestMedicationLoaderIntegration:
    """Integration tests for MedicationLoader matching __main__ block functionality"""
    
    def test_medication_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test MedicationLoader with full validation and summary"""
        from data.loaders.medication_loader import MedicationLoader
        
        loader = MedicationLoader(test_config.data.base_dir, test_config)
        med_df = loader.load()
        
        # Validate (works even with empty dataframe)
        loader.validate(med_df)
        
        # Get summary
        summary = loader.get_summary(med_df)
        
        # Basic assertions
        assert 'PATNO' in med_df.columns
        assert summary['n_rows'] >= 0  # Can be 0 if no medication data
        assert summary['n_columns'] >= 0
        
        # If medication data exists, check distributions
        if len(med_df) > 0:
            unique_patients = med_df['PATNO'].nunique()
            assert unique_patients > 0
            
            # Check for expected columns
            assert 'EVENT_ID' in med_df.columns or 'VISIT_ID' in med_df.columns
            
            # Check medication features (excluding metadata columns)
            metadata_cols = ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']
            med_cols = [c for c in med_df.columns if c not in metadata_cols]
            
            # If LEDD column exists, check distribution (similar to __main__ block)
            if 'LEDD' in med_df.columns:
                ledd_valid = med_df['LEDD'].dropna()
                if len(ledd_valid) > 0:
                    # Verify LEDD statistics are reasonable
                    assert ledd_valid.min() >= 0, "LEDD should not be negative"
                    # Note: High LEDD (>3000) is warned but not an error
                    mean_ledd = ledd_valid.mean()
                    median_ledd = ledd_valid.median()
                    assert pd.notna(mean_ledd) and pd.notna(median_ledd)
