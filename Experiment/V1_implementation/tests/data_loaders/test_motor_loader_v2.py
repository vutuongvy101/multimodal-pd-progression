"""Tests for MotorLoader (v2)"""

import pytest
import pandas as pd

from tests.utils.summary import get_df_summary


class TestMotorLoader:
    """Tests for the v2 MotorLoader"""

    @pytest.mark.requires_data
    def test_motor_loader_load(self, test_config_v2, skip_if_no_data, capsys):
        """MotorLoader v2: load, validate and basic summary checks"""
        # Import the v2 loader
        from data.loaders_v2 import MotorLoader

        loader = MotorLoader(test_config_v2.data.base_dir, test_config_v2)
        motor_df = loader.load()

        # Basic assertions
        assert isinstance(motor_df, pd.DataFrame)
        assert 'PATNO' in motor_df.columns
        assert 'EVENT_ID' in motor_df.columns or 'VISIT_ID' in motor_df.columns
        assert len(motor_df) > 0

        # Validate using loader's validate
        loader.validate(motor_df)

        # Get summary
        summary = get_df_summary(motor_df)
        unique_patients = motor_df['PATNO'].nunique()

        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0

        # Check that at least one motor-related column exists (from config)
        totals = test_config_v2.features.overall_motor_severity_score
        totals_found = [t for t in totals if t in motor_df.columns]
        assert len(totals_found) > 0, "No overall motor severity totals found"


@pytest.mark.requires_data
@pytest.mark.slow
class TestMotorLoaderIntegration:
    """Integration tests for MotorLoader v2"""

    def test_motor_loader_integration(self, test_config_v2, skip_if_no_data, capsys):
        from data.loaders_v2 import MotorLoader

        loader = MotorLoader(test_config_v2.data.base_dir, test_config_v2)
        motor_df = loader.load()

        # Validate
        loader.validate(motor_df)

        # Get summary
        summary = get_df_summary(motor_df)
        unique_patients = motor_df['PATNO'].nunique()

        # Summary assertions
        assert summary['n_rows'] > 0
        assert summary['n_columns'] > 0
        assert unique_patients > 0

        # Check for expected columns
        assert 'PATNO' in motor_df.columns
        assert 'EVENT_ID' in motor_df.columns or 'VISIT_ID' in motor_df.columns

        # Check UPDRS Totals distribution using v2 config grouping
        totals = test_config_v2.features.overall_motor_severity_score
        totals_found = []
        for total in totals:
            if total in motor_df.columns:
                n_valid = motor_df[total].notna().sum()
                if n_valid > 0:
                    mean_val = motor_df[total].mean()
                    std_val = motor_df[total].std()
                    assert pd.notna(mean_val) and pd.notna(std_val)
                    assert motor_df[total].dropna().min() >= 0, f"{total} should be non-negative"
                    totals_found.append(total)

        # At least one total should have valid data
        assert len(totals_found) > 0, "No UPDRS totals found with valid data"

        # Check time distribution if available
        if 'months_since_baseline' in motor_df.columns:
            time_valid = motor_df['months_since_baseline'].dropna()
            if len(time_valid) > 0:
                min_time = time_valid.min()
                max_time = time_valid.max()
                mean_time = time_valid.mean()
                assert pd.notna(min_time) and pd.notna(max_time) and pd.notna(mean_time)
                # Allow small negative tolerance if baseline calcs vary, but warn on large negatives
                assert min_time >= -1, "months_since_baseline seems unreasonably negative"
                assert max_time <= 240, "months_since_baseline seems unreasonably large"
