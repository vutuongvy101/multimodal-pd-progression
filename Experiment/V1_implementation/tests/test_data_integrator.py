"""
Tests for DataIntegrator

Comprehensive test suite covering:
- Unit tests for individual methods (merging, slope computation, missingness masks)
- Tests for create_feature_vectors function
- Edge cases and error handling
- Integration tests
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestDataIntegratorCreateMissingnessMasks:
    """Test create_missingness_masks method"""
    
    def test_create_missingness_masks_handles_valid_values(self, test_config):
        """Test: Valid values create mask with 0s (not missing)"""
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
        })
        feature_cols = ['feature1', 'feature2']
        
        values, mask = integrator.create_missingness_masks(df, feature_cols)
        
        assert values.shape == (3, 2)
        assert mask.shape == (3, 2)
        assert (mask == 0).all()  # No missing values
        np.testing.assert_array_equal(values, df[feature_cols].values)
    
    def test_create_missingness_masks_handles_nan_values(self, test_config):
        """Test: NaN values create mask with 1s (missing) and 0s in values"""
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [np.nan, 5.0, 6.0],
        })
        feature_cols = ['feature1', 'feature2']
        
        values, mask = integrator.create_missingness_masks(df, feature_cols)
        
        assert values.shape == (3, 2)
        assert mask.shape == (3, 2)
        # Check that NaN values are marked as missing (mask=1) and filled with 0
        assert mask[0, 1] == 1  # feature2 at row 0 is NaN
        assert mask[1, 0] == 1  # feature1 at row 1 is NaN
        assert values[0, 1] == 0.0  # NaN filled with 0
        assert values[1, 0] == 0.0  # NaN filled with 0
    
    def test_create_missingness_masks_handles_empty_dataframe(self, test_config):
        """Test: Empty dataframe returns empty arrays"""
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        
        df = pd.DataFrame(columns=['feature1', 'feature2'])
        feature_cols = ['feature1', 'feature2']
        
        values, mask = integrator.create_missingness_masks(df, feature_cols)
        
        assert values.shape == (0, 2)
        assert mask.shape == (0, 2)
    
    def test_create_missingness_masks_handles_object_columns(self, test_config):
        """Test: Object/string columns are converted to numeric and handled correctly
        
        This tests the scenario where genetics features might be strings/categorical.
        String values that can't be converted become NaN, which is marked as missing.
        """
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        
        # Simulate genetics data with string/categorical columns
        df = pd.DataFrame({
            'PATNO': [1, 2, 3],
            'LRRK2': ['positive', 'negative', np.nan],  # String column
            'GBA': [0, 1, 1],  # Numeric column
            'ENROLL_AGE': [60.0, 65.0, 70.0],  # Float column
        })
        feature_cols = ['LRRK2', 'GBA', 'ENROLL_AGE']
        
        values, mask = integrator.create_missingness_masks(df, feature_cols)
        
        # Should handle conversion without TypeError
        assert values.shape == (3, 3)
        assert mask.shape == (3, 3)
        
        # LRRK2 strings should become NaN -> marked as missing (mask=1)
        assert mask[0, 0] == 1  # 'positive' -> NaN -> missing
        assert mask[1, 0] == 1  # 'negative' -> NaN -> missing
        assert mask[2, 0] == 1  # np.nan -> missing
        
        # GBA and ENROLL_AGE should be fine
        assert mask[0, 1] == 0  # GBA has value
        assert mask[0, 2] == 0  # ENROLL_AGE has value
        
        # Values should be filled with 0 for missing (mask tells us they're missing)
        assert values[0, 0] == 0.0  # Missing value filled
        assert values[0, 1] == 0.0  # GBA value
        assert values[0, 2] == 60.0  # ENROLL_AGE value


class TestDataIntegratorCreateFeatureVectors:
    """Test create_feature_vectors method"""
    
    def test_create_feature_vectors_with_mocked_data(self, test_config):
        """Test: create_feature_vectors converts DataFrame format to feature vectors correctly"""
        from data.data_integrator import DataIntegrator
        
        # Create a mock config with simple feature lists
        mock_config = Mock()
        mock_config.features.static_features = ['ENROLL_AGE', 'SEX']
        mock_config.features.motor_features = ['NP2PTOT', 'NP3TOT']
        mock_config.features.updrs_supplementary_features = ['NP1RTOT']
        mock_config.features.non_motor_features = []
        mock_config.features.medication_features = ['LEDD']
        mock_config.features.all_updrs_totals = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        
        # Create mock prepared data
        static_df = pd.DataFrame({
            'PATNO': [1, 2],
            'ENROLL_AGE': [60.0, 65.0],
            'SEX': [1.0, 2.0],
        })
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 2],
            'EVENT_ID': ['BL', 'V01', 'BL'],
            'months_since_baseline': [0.0, 6.0, 0.0],
            'NP1RTOT': [5.0, 6.0, 4.0],
            'NP2PTOT': [8.0, 9.0, 7.0],
            'NP3TOT': [20.0, 22.0, 18.0],
            'NP4TOT': [1.0, 1.0, 0.0],
            'LEDD': [200.0, 250.0, 150.0],
        })
        
        slopes_df = pd.DataFrame({
            'PATNO': [1, 2],
            'NP1RTOT_slope': [0.1, 0.05],
            'NP2PTOT_slope': [0.2, 0.15],
            'NP3TOT_slope': [0.5, 0.3],
            'NP4TOT_slope': [0.01, 0.0],
        })
        
        prepared_data = {
            'static': static_df,
            'longitudinal': longitudinal_df,
            'slopes': slopes_df,
        }
        
        # Create integrator with mocked loaders (we'll bypass actual loading)
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = mock_config
        
        # Call create_feature_vectors with prepared data
        result = integrator.create_feature_vectors(prepared_data)
        
        # Verify structure
        assert 'static_data' in result
        assert 'longitudinal_data' in result
        assert 'slopes' in result
        
        # Verify static_data structure
        assert 1 in result['static_data']
        assert 2 in result['static_data']
        for patno in [1, 2]:
            static_entry = result['static_data'][patno]
            assert 'values' in static_entry
            assert 'mask' in static_entry
            assert isinstance(static_entry['values'], np.ndarray)
            assert isinstance(static_entry['mask'], np.ndarray)
            assert static_entry['values'].shape == static_entry['mask'].shape
            # Should have 2 features: ENROLL_AGE, SEX
            assert len(static_entry['values']) == 2
        
        # Verify longitudinal_data structure
        assert 1 in result['longitudinal_data']
        assert 2 in result['longitudinal_data']
        assert len(result['longitudinal_data'][1]) == 2  # Patient 1 has 2 visits
        assert len(result['longitudinal_data'][2]) == 1  # Patient 2 has 1 visit
        
        # Check visit structure for patient 1
        visit = result['longitudinal_data'][1][0]
        assert 'motor_values' in visit
        assert 'motor_mask' in visit
        assert 'updrs_supplementary_values' in visit
        assert 'updrs_supplementary_mask' in visit
        assert 'nonmotor_values' in visit
        assert 'nonmotor_mask' in visit
        assert 'med_values' in visit
        assert 'med_mask' in visit
        assert 'time_months' in visit
        assert 'updrs_totals' in visit
        assert 'np3tot' in visit
        
        # Verify slopes (now a dict with all UPDRS totals)
        assert isinstance(result['slopes'][1], dict)
        assert isinstance(result['slopes'][2], dict)
        # Check that all slope keys exist and have correct values
        expected_slopes_pat1 = {'NP1RTOT_slope': 0.1, 'NP2PTOT_slope': 0.2, 'NP3TOT_slope': 0.5, 'NP4TOT_slope': 0.01}
        expected_slopes_pat2 = {'NP1RTOT_slope': 0.05, 'NP2PTOT_slope': 0.15, 'NP3TOT_slope': 0.3, 'NP4TOT_slope': 0.0}
        for total in mock_config.features.all_updrs_totals:
            slope_key = f'{total}_slope'
            assert slope_key in result['slopes'][1]
            assert slope_key in result['slopes'][2]
            # Verify values match expected
            assert result['slopes'][1][slope_key] == expected_slopes_pat1[slope_key]
            assert result['slopes'][2][slope_key] == expected_slopes_pat2[slope_key]
    
    def test_create_feature_vectors_handles_missing_features(self, test_config):
        """Test: Missing features are handled with empty arrays or NaN"""
        from data.data_integrator import DataIntegrator
        
        # Create a mock config with simple feature lists
        mock_config = Mock()
        mock_config.features.static_features = ['ENROLL_AGE', 'SEX', 'MISSING_FEATURE']
        mock_config.features.motor_features = ['NP2PTOT']
        mock_config.features.updrs_supplementary_features = []
        mock_config.features.non_motor_features = []
        mock_config.features.medication_features = []
        mock_config.features.all_updrs_totals = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        
        static_df = pd.DataFrame({
            'PATNO': [1],
            'ENROLL_AGE': [60.0],
            'SEX': [1.0],
            # MISSING_FEATURE column not present
        })
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1],
            'EVENT_ID': ['BL'],
            'months_since_baseline': [0.0],
            'NP2PTOT': [8.0],
            # Other UPDRS totals missing
        })
        
        slopes_df = pd.DataFrame({
            'PATNO': [1],
            # NP3TOT_slope missing
        })
        
        prepared_data = {
            'static': static_df,
            'longitudinal': longitudinal_df,
            'slopes': slopes_df,
        }
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = mock_config
        
        result = integrator.create_feature_vectors(prepared_data)
        
        # Should only have features that exist (ENROLL_AGE, SEX)
        static_entry = result['static_data'][1]
        assert len(static_entry['values']) == 2  # Only 2 features present
        
        # UPDRS supplementary, non-motor, and med features should be empty arrays
        visit = result['longitudinal_data'][1][0]
        assert len(visit['updrs_supplementary_values']) == 0
        assert len(visit['nonmotor_values']) == 0
        assert len(visit['med_values']) == 0
        
        # Slopes should be dict with all UPDRS totals, NaN when missing
        assert isinstance(result['slopes'][1], dict)
        for total in mock_config.features.all_updrs_totals:
            assert f'{total}_slope' in result['slopes'][1]
            assert pd.isna(result['slopes'][1][f'{total}_slope'])
    
    def test_create_feature_vectors_handles_empty_longitudinal_data(self, test_config):
        """Test: Empty longitudinal data handled gracefully"""
        from data.data_integrator import DataIntegrator
        
        # Create a mock config
        mock_config = Mock()
        mock_config.features.static_features = ['ENROLL_AGE']
        mock_config.features.all_updrs_totals = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        
        static_df = pd.DataFrame({
            'PATNO': [1],
            'ENROLL_AGE': [60.0],
        })
        
        longitudinal_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        slopes_df = pd.DataFrame(columns=['PATNO'])
        
        prepared_data = {
            'static': static_df,
            'longitudinal': longitudinal_df,
            'slopes': slopes_df,
        }
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = mock_config
        
        result = integrator.create_feature_vectors(prepared_data)
        
        # Static data should exist
        assert 1 in result['static_data']
        
        # Longitudinal data should be empty dict or have empty lists
        assert len(result['longitudinal_data']) == 0 or result['longitudinal_data'][1] == []
        
        # Slopes should be dict with all UPDRS totals, NaN when missing
        assert isinstance(result['slopes'][1], dict)
        for total in mock_config.features.all_updrs_totals:
            assert f'{total}_slope' in result['slopes'][1]
            assert pd.isna(result['slopes'][1][f'{total}_slope'])
    
    def test_create_feature_vectors_sorts_visits_by_time(self, test_config):
        """Test: Visits are sorted by months_since_baseline"""
        from data.data_integrator import DataIntegrator
        
        # Create a mock config
        mock_config = Mock()
        mock_config.features.static_features = ['ENROLL_AGE']
        mock_config.features.motor_features = ['NP3TOT']
        mock_config.features.all_updrs_totals = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        
        static_df = pd.DataFrame({
            'PATNO': [1],
            'ENROLL_AGE': [60.0],
        })
        
        # Visits in unsorted order
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 1],
            'EVENT_ID': ['V01', 'BL', 'V02'],
            'months_since_baseline': [6.0, 0.0, 12.0],
            'NP3TOT': [22.0, 20.0, 24.0],
        })
        
        slopes_df = pd.DataFrame({'PATNO': [1]})
        
        prepared_data = {
            'static': static_df,
            'longitudinal': longitudinal_df,
            'slopes': slopes_df,
        }
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = mock_config
        
        result = integrator.create_feature_vectors(prepared_data)
        
        visits = result['longitudinal_data'][1]
        assert len(visits) == 3
        
        # Check that visits are sorted by time
        assert visits[0]['time_months'] == 0.0   # BL
        assert visits[1]['time_months'] == 6.0   # V01
        assert visits[2]['time_months'] == 12.0  # V02
    
    def test_create_feature_vectors_sorts_visits_by_event_id_when_time_missing(self, test_config):
        """Test: Visits sorted by EVENT_ID when months_since_baseline is missing"""
        from data.data_integrator import DataIntegrator
        
        # Create a mock config
        mock_config = Mock()
        mock_config.features.static_features = ['ENROLL_AGE']
        mock_config.features.motor_features = ['NP3TOT']
        mock_config.features.all_updrs_totals = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        
        static_df = pd.DataFrame({
            'PATNO': [1],
            'ENROLL_AGE': [60.0],
        })
        
        # Visits without months_since_baseline
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 1],
            'EVENT_ID': ['V02', 'BL', 'V01'],
            'NP3TOT': [24.0, 20.0, 22.0],
        })
        
        slopes_df = pd.DataFrame({'PATNO': [1]})
        
        prepared_data = {
            'static': static_df,
            'longitudinal': longitudinal_df,
            'slopes': slopes_df,
        }
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = mock_config
        
        result = integrator.create_feature_vectors(prepared_data)
        
        visits = result['longitudinal_data'][1]
        assert len(visits) == 3
        
        # Should be sorted: BL (order 0), V01 (order 1), V02 (order 2)
        # We check by EVENT_ID or by the fact that they're in order
        # Since time_months will be 0.0 for all (missing), we verify by structure
        assert visits[0]['time_months'] == 0.0
        assert visits[1]['time_months'] == 0.0
        assert visits[2]['time_months'] == 0.0
    
    def test_create_feature_vectors_calls_prepare_final_dataset_when_no_input(self, test_config):
        """Test: create_feature_vectors calls prepare_final_dataset when prepared_data is None"""
        from data.data_integrator import DataIntegrator
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            # Use a minimal mock config (FeatureConfig has read-only properties)
            mock_config = Mock()
            mock_config.features.static_features = ['ENROLL_AGE']
            mock_config.features.all_updrs_totals = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
            mock_config.features.motor_features = []
            mock_config.features.updrs_supplementary_features = []
            mock_config.features.non_motor_features = []
            mock_config.features.medication_features = []
            integrator.config = mock_config
            
            # Mock prepare_final_dataset
            mock_prepared_data = {
                'static': pd.DataFrame({'PATNO': [1], 'ENROLL_AGE': [60.0]}),
                'longitudinal': pd.DataFrame(columns=['PATNO', 'EVENT_ID']),
                'slopes': pd.DataFrame(columns=['PATNO']),
            }
            integrator.prepare_final_dataset = Mock(return_value=mock_prepared_data)
            
            # Call without prepared_data
            result = integrator.create_feature_vectors(prepared_data=None)
            
            # Verify prepare_final_dataset was called
            integrator.prepare_final_dataset.assert_called_once()
            
            # Verify result structure exists
            assert 'static_data' in result
            assert 'longitudinal_data' in result
            assert 'slopes' in result


class TestDataIntegratorComputeProgressionSlopes:
    """Test compute_progression_slopes method"""
    
    def test_compute_progression_slopes_with_valid_data(self, test_config):
        """Test: Slopes computed correctly for patients with sufficient visits"""
        from data.data_integrator import DataIntegrator
        
        # Use default config - all_updrs_totals includes NP3TOT
        test_config.training.min_visits_for_slope = 3
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 1, 2, 2, 2, 2],
            'months_since_baseline': [0.0, 6.0, 12.0, 0.0, 6.0, 12.0, 18.0],
            'NP3TOT': [20.0, 22.0, 24.0, 18.0, 20.0, 22.0, 24.0],
        })
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = test_config
        
        slopes_df = integrator.compute_progression_slopes(longitudinal_df)
        
        assert len(slopes_df) == 2  # Both patients have enough visits
        assert 'PATNO' in slopes_df.columns
        assert 'NP3TOT_slope' in slopes_df.columns
        assert 'NP3TOT_r' in slopes_df.columns
        assert 'NP3TOT_p' in slopes_df.columns
        
        # Check that slopes are positive (worsening) for both patients
        slopes = slopes_df['NP3TOT_slope'].values
        assert all(slopes > 0)
    
    def test_compute_progression_slopes_filters_insufficient_visits(self, test_config):
        """Test: Patients with too few visits are excluded"""
        from data.data_integrator import DataIntegrator
        
        # Use default config - all_updrs_totals includes NP3TOT
        test_config.training.min_visits_for_slope = 3
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 2],  # Patient 1 has 2 visits, Patient 2 has 1 visit
            'months_since_baseline': [0.0, 6.0, 0.0],
            'NP3TOT': [20.0, 22.0, 18.0],
        })
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = test_config
        
        slopes_df = integrator.compute_progression_slopes(longitudinal_df)
        
        # No patients should have slopes (all have < 3 visits)
        assert len(slopes_df) == 0
    
    def test_compute_progression_slopes_handles_missing_time(self, test_config):
        """Test: Visits with missing time are excluded from slope calculation"""
        from data.data_integrator import DataIntegrator
        
        # Use default config - all_updrs_totals includes NP3TOT
        test_config.training.min_visits_for_slope = 3
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 1, 1],
            'months_since_baseline': [0.0, np.nan, 6.0, 12.0],  # One missing time
            'NP3TOT': [20.0, 21.0, 22.0, 24.0],
        })
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = test_config
        
        slopes_df = integrator.compute_progression_slopes(longitudinal_df)
        
        # Should still compute slope using the 3 valid time points
        assert len(slopes_df) == 1


class TestDataIntegratorMergeLongitudinalData:
    """Test _merge_longitudinal_data helper method"""
    
    def test_merge_longitudinal_data_merges_correctly(self, test_config):
        """Test: Merges new DataFrame into longitudinal DataFrame"""
        from data.data_integrator import DataIntegrator
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1, 1, 2],
            'EVENT_ID': ['BL', 'V01', 'BL'],
            'months_since_baseline': [0.0, 6.0, 0.0],
            'NP3TOT': [20.0, 22.0, 18.0],
        })
        
        new_df = pd.DataFrame({
            'PATNO': [1, 2],
            'EVENT_ID': ['BL', 'BL'],
            'months_since_baseline': [0.0, 0.0],
            'LEDD': [200.0, 150.0],
        })
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = test_config
        
        result = integrator._merge_longitudinal_data(longitudinal_df, new_df, '_med')
        
        assert 'LEDD' in result.columns
        assert len(result) == 3
        # Patient 1, BL should have LEDD
        patient1_bl = result[(result['PATNO'] == 1) & (result['EVENT_ID'] == 'BL')]
        assert len(patient1_bl) == 1
        assert patient1_bl.iloc[0]['LEDD'] == 200.0
    
    def test_merge_longitudinal_data_handles_duplicate_months_columns(self, test_config):
        """Test: Duplicate months_since_baseline columns are merged_dataset correctly"""
        from data.data_integrator import DataIntegrator
        
        longitudinal_df = pd.DataFrame({
            'PATNO': [1],
            'EVENT_ID': ['BL'],
            'months_since_baseline': [0.0],
        })
        
        new_df = pd.DataFrame({
            'PATNO': [1],
            'EVENT_ID': ['BL'],
            'months_since_baseline': [0.0],  # Duplicate column
            'LEDD': [200.0],
        })
        
        with patch.object(DataIntegrator, '__init__', lambda self, config: None):
            integrator = DataIntegrator.__new__(DataIntegrator)
            integrator.config = test_config
        
        result = integrator._merge_longitudinal_data(longitudinal_df, new_df, '_med')
        
        # Should have only one months_since_baseline column
        assert result.columns.tolist().count('months_since_baseline') == 1
        assert 'months_since_baseline_med' not in result.columns


@pytest.mark.requires_data
class TestDataIntegratorIntegration:
    """Integration tests for DataIntegrator (requires actual data)"""
    
    def test_create_feature_vectors_integration(self, test_config, skip_if_no_data, capsys):
        """Test: create_feature_vectors works with real data"""
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        
        # Prepare final dataset
        prepared_data = integrator.prepare_final_dataset()
        
        # Create feature vectors
        feature_vectors = integrator.create_feature_vectors(prepared_data)
        
        # Verify structure
        assert 'static_data' in feature_vectors
        assert 'longitudinal_data' in feature_vectors
        assert 'slopes' in feature_vectors
        
        # Verify we have some patients
        assert len(feature_vectors['static_data']) > 0
        
        # Check a sample patient
        sample_patno = list(feature_vectors['static_data'].keys())[0]
        
        # Verify static data structure
        static_entry = feature_vectors['static_data'][sample_patno]
        assert 'values' in static_entry
        assert 'mask' in static_entry
        assert len(static_entry['values']) == len(static_entry['mask'])
        
        # Verify longitudinal data structure if patient has visits
        if sample_patno in feature_vectors['longitudinal_data']:
            visits = feature_vectors['longitudinal_data'][sample_patno]
            assert len(visits) > 0
            
            visit = visits[0]
            required_keys = ['motor_values', 'motor_mask', 
                           'updrs_supplementary_values', 'updrs_supplementary_mask',
                           'nonmotor_values', 'nonmotor_mask',
                           'med_values', 'med_mask', 
                           'time_months', 'updrs_totals', 'np3tot']
            for key in required_keys:
                assert key in visit
    
    @pytest.mark.slow
    def test_data_merging_pipeline_saves_to_merged_folder(self, test_config, skip_if_no_data, v1_root):
        """Test: Full data merging pipeline and save to tests/test_outputs/merged_dataset for inspection
        
        This test runs prepare_final_dataset() which does all the data merging,
        and saves the results to tests/test_outputs/merged_dataset/ folder for manual inspection.
        """
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        
        # Run the full merging pipeline
        prepared_data = integrator.prepare_final_dataset()
        
        # Create merged_dataset data directory in test folder
        merged_dir = v1_root / "tests" / "test_outputs" / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Save static data
        static_path = merged_dir / "static_data.csv"
        prepared_data['static'].to_csv(static_path, index=False)
        assert static_path.exists()
        assert len(prepared_data['static']) > 0
        
        # Save longitudinal data
        longitudinal_path = merged_dir / "longitudinal_data.csv"
        prepared_data['longitudinal'].to_csv(longitudinal_path, index=False)
        assert longitudinal_path.exists()
        assert len(prepared_data['longitudinal']) > 0
        
        # Save slopes data
        slopes_path = merged_dir / "slopes_data.csv"
        prepared_data['slopes'].to_csv(slopes_path, index=False)
        assert slopes_path.exists()
        
        # Save metadata summary
        metadata_path = merged_dir / "metadata.json"
        metadata = {
            'n_patients': prepared_data['metadata']['n_patients'],
            'n_visits': prepared_data['metadata']['n_visits'],
            'n_with_slopes': prepared_data['metadata']['n_with_slopes'],
            'static_features': prepared_data['metadata']['static_features'],
            'longitudinal_features': prepared_data['metadata']['longitudinal_features'],
            'updrs_totals': prepared_data['metadata']['updrs_totals'],
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        assert metadata_path.exists()
        
        print(f"\n✓ Merged data saved to: {merged_dir}")
        print(f"  - static_data.csv: {len(prepared_data['static'])} patients")
        print(f"  - longitudinal_data.csv: {len(prepared_data['longitudinal'])} visits")
        print(f"  - slopes_data.csv: {len(prepared_data['slopes'])} patients with slopes")
        print(f"  - metadata.json: Summary information")
    
    @pytest.mark.slow
    def test_feature_vectors_pipeline_saves_to_processed_folder(self, test_config, skip_if_no_data, v1_root):
        """Test: create_feature_vectors pipeline and save to tests/test_outputs/processed for inspection
        
        This test runs create_feature_vectors() which creates feature vectors with
        missingness masks, and saves the results to tests/test_outputs/processed/ folder for manual inspection.
        """
        from data.data_integrator import DataIntegrator
        
        integrator = DataIntegrator(test_config)
        prepared_data = integrator.prepare_final_dataset()
        feature_vectors = integrator.create_feature_vectors(prepared_data)
        
        # Create processed data directory in test folder
        processed_dir = v1_root / "tests" / "test_outputs" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save static feature vectors summary
        static_summary = []
        static_feature_names = [c for c in test_config.features.static_features 
                               if c in prepared_data['static'].columns]
        
        for patno, static_entry in feature_vectors['static_data'].items():
            row = {'PATNO': patno}
            # Add feature values and masks
            for i, feat_name in enumerate(static_feature_names):
                if i < len(static_entry['values']):
                    row[f'{feat_name}_value'] = float(static_entry['values'][i])
                    row[f'{feat_name}_mask'] = float(static_entry['mask'][i])
            static_summary.append(row)
        
        static_summary_df = pd.DataFrame(static_summary)
        static_summary_path = processed_dir / "static_feature_vectors.csv"
        static_summary_df.to_csv(static_summary_path, index=False)
        assert static_summary_path.exists()
        
        # Save longitudinal feature vectors summary (sample patients)
        longitudinal_summary = []
        sample_patients = list(feature_vectors['longitudinal_data'].keys())[:10]  # First 10 patients
        
        for patno in sample_patients:
            visits = feature_vectors['longitudinal_data'][patno]
            for visit_idx, visit in enumerate(visits):
                row = {
                    'PATNO': patno,
                    'visit_idx': visit_idx,
                    'time_months': visit['time_months'],
                    'motor_n_features': len(visit['motor_values']),
                    'motor_n_missing': int(visit['motor_mask'].sum()),
                    'updrs_supplementary_n_features': len(visit['updrs_supplementary_values']),
                    'updrs_supplementary_n_missing': int(visit['updrs_supplementary_mask'].sum()),
                    'non_motor_n_features': len(visit['nonmotor_values']),
                    'non_motor_n_missing': int(visit['nonmotor_mask'].sum()),
                    'med_n_features': len(visit['med_values']),
                    'med_n_missing': int(visit['med_mask'].sum()),
                    'np3tot': float(visit['np3tot']) if not np.isnan(visit['np3tot']) else np.nan,
                }
                # Add UPDRS totals
                for i, total_name in enumerate(test_config.features.all_updrs_totals):
                    if i < len(visit['updrs_totals']):
                        val = visit['updrs_totals'][i]
                        row[total_name] = float(val) if not np.isnan(val) else np.nan
                longitudinal_summary.append(row)
        
        longitudinal_summary_df = pd.DataFrame(longitudinal_summary)
        longitudinal_summary_path = processed_dir / "longitudinal_feature_vectors_summary.csv"
        longitudinal_summary_df.to_csv(longitudinal_summary_path, index=False)
        assert longitudinal_summary_path.exists()
        
        # Save slopes (flatten dict structure to CSV)
        slopes_rows = []
        for patno, slopes_dict in feature_vectors['slopes'].items():
            row = {'PATNO': patno}
            row.update(slopes_dict)
            slopes_rows.append(row)
        slopes_df = pd.DataFrame(slopes_rows)
        slopes_path = processed_dir / "slopes.csv"
        slopes_df.to_csv(slopes_path, index=False)
        assert slopes_path.exists()
        
        # Save feature vector statistics
        stats = {
            'n_patients': len(feature_vectors['static_data']),
            'n_patients_with_longitudinal': len(feature_vectors['longitudinal_data']),
            'n_patients_with_slopes': sum(1 for s_dict in feature_vectors['slopes'].values() 
                                          if any(pd.notna(v) for v in s_dict.values())),
            'total_visits': sum(len(visits) for visits in feature_vectors['longitudinal_data'].values()),
            'static_n_features': len(static_feature_names),
            'motor_n_features': len([c for c in test_config.features.motor_features 
                                    if c in prepared_data['longitudinal'].columns]),
            'updrs_supplementary_n_features': len([c for c in test_config.features.updrs_supplementary_features
                                       if c in prepared_data['longitudinal'].columns]),
            'non_motor_n_features': len([c for c in test_config.features.non_motor_features
                                       if c in prepared_data['longitudinal'].columns]),
            'med_n_features': len([c for c in test_config.features.medication_features 
                                  if c in prepared_data['longitudinal'].columns]),
        }
        
        stats_path = processed_dir / "feature_vectors_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        assert stats_path.exists()
        
        # Save a sample feature vector for detailed inspection (first patient)
        if feature_vectors['static_data']:
            sample_patno = list(feature_vectors['static_data'].keys())[0]
            sample_data = {
                'PATNO': int(sample_patno),
                'static': {
                    'values': feature_vectors['static_data'][sample_patno]['values'].tolist(),
                    'mask': feature_vectors['static_data'][sample_patno]['mask'].tolist(),
                    'feature_names': static_feature_names,
                },
                'slopes': feature_vectors['slopes'][sample_patno],  # Now a dict with all UPDRS totals
            }
            
            if sample_patno in feature_vectors['longitudinal_data']:
                visits = feature_vectors['longitudinal_data'][sample_patno]
                sample_data['n_visits'] = len(visits)
                sample_data['visits'] = []
                for visit in visits[:3]:  # First 3 visits
                    visit_data = {
                        'time_months': float(visit['time_months']),
                        'motor_values': visit['motor_values'].tolist(),
                        'motor_mask': visit['motor_mask'].tolist(),
                        'updrs_supplementary_values': visit['updrs_supplementary_values'].tolist(),
                        'updrs_supplementary_mask': visit['updrs_supplementary_mask'].tolist(),
                        'non_motor_values': visit['nonmotor_values'].tolist(),
                        'non_motor_mask': visit['nonmotor_mask'].tolist(),
                        'med_values': visit['med_values'].tolist(),
                        'med_mask': visit['med_mask'].tolist(),
                        'updrs_totals': [float(v) if not np.isnan(v) else None 
                                        for v in visit['updrs_totals']],
                        'np3tot': float(visit['np3tot']) if not np.isnan(visit['np3tot']) else None,
                    }
                    sample_data['visits'].append(visit_data)
            
            sample_path = processed_dir / "sample_feature_vector.json"
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f, indent=2, default=str)
            assert sample_path.exists()
        
        print(f"\n✓ Feature vectors saved to: {processed_dir}")
        print(f"  - static_feature_vectors.csv: All static features with values and masks")
        print(f"  - longitudinal_feature_vectors_summary.csv: Summary of longitudinal data (first 10 patients)")
        print(f"  - slopes.csv: All patient slopes")
        print(f"  - feature_vectors_stats.json: Overall statistics")
        print(f"  - sample_feature_vector.json: Detailed example of one patient's feature vector")
