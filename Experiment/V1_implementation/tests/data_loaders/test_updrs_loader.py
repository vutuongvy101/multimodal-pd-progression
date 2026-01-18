"""
Tests for UPDRSLoader
"""

import pytest
import pandas as pd


class TestUPDRSLoader:
    """Test UPDRSLoader"""
    
    @pytest.mark.requires_data
    def test_updrs_loader_load(self, test_config, skip_if_no_data, capsys):
        """Test UPDRSLoader can load data with full validation and summary"""
        from data.loaders.updrs_loader import UPDRSLoader
        
        loader = UPDRSLoader(test_config.data.base_dir, test_config)
        updrs_df = loader.load()
        
        # Basic assertions
        assert isinstance(updrs_df, pd.DataFrame)
        assert 'PATNO' in updrs_df.columns
        assert 'EVENT_ID' in updrs_df.columns or 'VISIT_ID' in updrs_df.columns
        assert len(updrs_df) > 0
        
        # Validate
        loader.validate(updrs_df)
        
        # Get summary
        summary = loader.get_summary(updrs_df)
        unique_patients = updrs_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in updrs_df.columns
        assert 'EVENT_ID' in updrs_df.columns or 'VISIT_ID' in updrs_df.columns
        
        # Check that UPDRS columns exist (NP1*, NP2*, NP3*, NP4*)
        updrs_cols = [c for c in updrs_df.columns if any(c.startswith(prefix) for prefix in ['NP1', 'NP2', 'NP3', 'NP4'])]
        assert len(updrs_cols) > 0, "No UPDRS columns found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestUPDRSLoaderIntegration:
    """Integration tests for UPDRSLoader matching __main__ block functionality"""
    
    def test_updrs_loader_integration(self, test_config, skip_if_no_data, capsys):
        """Test UPDRSLoader with full validation and summary"""
        from data.loaders.updrs_loader import UPDRSLoader
        
        loader = UPDRSLoader(test_config.data.base_dir, test_config)
        updrs_df = loader.load()
        
        # Validate
        loader.validate(updrs_df)
        
        # Get summary
        summary = loader.get_summary(updrs_df)
        unique_patients = updrs_df['PATNO'].nunique()
        
        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0
        
        # Check for expected columns
        assert 'PATNO' in updrs_df.columns
        assert 'EVENT_ID' in updrs_df.columns or 'VISIT_ID' in updrs_df.columns
        
        # Check UPDRS Totals distribution (similar to __main__ block)
        # NP1TOT, NP2TOT, NP3TOT, NP4TOT
        totals = ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']
        totals_found = []
        for total in totals:
            if total in updrs_df.columns:
                n_valid = updrs_df[total].notna().sum()
                if n_valid > 0:
                    mean_val = updrs_df[total].mean()
                    std_val = updrs_df[total].std()
                    # Verify statistics are reasonable (not all NaN)
                    assert pd.notna(mean_val) and pd.notna(std_val)
                    # Verify totals are non-negative
                    assert updrs_df[total].dropna().min() >= 0, f"{total} should be non-negative"
                    totals_found.append(total)
        
        # Assert that at least some UPDRS totals exist
        assert len(totals_found) > 0, "No UPDRS totals found with valid data"
        
        # Check time distribution if available (similar to __main__ block)
        if 'months_since_baseline' in updrs_df.columns:
            time_valid = updrs_df['months_since_baseline'].dropna()
            if len(time_valid) > 0:
                min_time = time_valid.min()
                max_time = time_valid.max()
                mean_time = time_valid.mean()
                # Verify time statistics are reasonable
                assert pd.notna(min_time) and pd.notna(max_time) and pd.notna(mean_time)
                # Verify time is non-negative (months since baseline)
                assert min_time >= 0, "months_since_baseline should be non-negative"
                # Verify max time is reasonable (less than 20 years = 240 months)
                assert max_time <= 240, "months_since_baseline seems unreasonably large"
