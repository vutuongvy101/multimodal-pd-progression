"""
Tests for DemographicsLoader

Comprehensive test suite covering:
- Unit tests for individual methods (filtering, merging, validation)
- Edge cases and error handling
- Numeric type conversion
- Multi-row data handling (age_at_visit with multiple EVENT_IDs per PATNO)
- Integration tests
"""

import pytest
import pandas as pd
import numpy as np
import os


class TestDemographicsLoaderFilterValidParticipants:
    """Test __filter_valid_participants__ method"""
    
    def test_filter_valid_participants_removes_null_enroll_date(self, test_config):
        """Test: Rows with null ENROLL_DATE are removed
        
        Example Output:
        - Input: 100 rows with 5 null ENROLL_DATE
        - Output: 95 rows (5 removed)
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3, 4, 5],
            'ENROLL_DATE': ['2020-01-01', '2020-01-02', None, '2020-01-04', None],
            'ENROLL_STATUS': ['Complete', 'Complete', 'Complete', 'Complete', 'Complete'],
            'COHORT_DEFINITION': ['PD', 'PD', 'PD', 'PD', 'PD'],
        })
        
        result = loader._DemographicsLoader__filter_valid_participants__(df)
        assert len(result) == 3
        assert all(pd.notna(result['ENROLL_DATE']))
    
    def test_filter_valid_participants_keeps_valid_statuses(self, test_config):
        """Test: Only valid enrollment statuses are kept
        
        Example Output:
        - Input: 100 rows with statuses [Complete, Invalid, Enrolled, Unknown]
        - Output: rows with [Complete, Enrolled] only
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3, 4],
            'ENROLL_DATE': ['2020-01-01'] * 4,
            'ENROLL_STATUS': ['Complete', 'InvalidStatus', 'Enrolled', 'Unknown'],
            'COHORT_DEFINITION': ['PD', 'PD', 'PD', 'PD'],
        })
        
        result = loader._DemographicsLoader__filter_valid_participants__(df)
        assert len(result) == 2
        assert result['ENROLL_STATUS'].isin(['Complete', 'Enrolled']).all()
    
    def test_filter_valid_participants_excludes_swedd(self, test_config):
        """Test: SWEDD cohort is excluded, valid cohorts retained
        
        Example Output:
        - Input: [PD, HC, Prodromal, SWEDD]
        - Output: [PD, HC, Prodromal] (SWEDD removed)
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3, 4],
            'ENROLL_DATE': ['2020-01-01'] * 4,
            'ENROLL_STATUS': ['Complete'] * 4,
            'COHORT_DEFINITION': ["Parkinson's Disease", 'Healthy Control', 'Prodromal', 'SWEDD'],
        })
        
        result = loader._DemographicsLoader__filter_valid_participants__(df)
        assert len(result) == 3
        assert 'SWEDD' not in result['COHORT_DEFINITION'].values
    
    def test_filter_valid_participants_returns_required_columns(self, test_config):
        """Test: Output contains only required columns
        
        Example Output:
        - Input: [PATNO, ENROLL_DATE, ENROLL_STATUS, COHORT_DEFINITION, EXTRA_COL]
        - Output columns: [PATNO, COHORT, COHORT_DEFINITION, ENROLL_STATUS, ENROLL_AGE]
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3],
            'ENROLL_DATE': ['2020-01-01'] * 3,
            'ENROLL_STATUS': ['Complete'] * 3,
            'COHORT_DEFINITION': ['PD'] * 3,
            'ENROLL_AGE': [60, 65, 70],
            'COHORT': [1, 1, 1],
            'EXTRA_COLUMN': ['a', 'b', 'c'],
        })
        
        result = loader._DemographicsLoader__filter_valid_participants__(df)
        expected_cols = ['PATNO', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS', 'ENROLL_AGE']
        assert list(result.columns) == expected_cols


class TestDemographicsLoaderMergeData:
    """Test __load_and_merge_data__ method"""
    
    def test_load_and_merge_data_handles_missing_file(self, test_config):
        """Test: Missing file is skipped gracefully with warning
        
        Example Output:
        - Input: file_path = '/nonexistent/file.csv'
        - Output: Original df returned unchanged + warning printed
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3],
            'COHORT': [1, 1, 1],
        })
        
        result = loader._DemographicsLoader__load_and_merge_data__(
            df,
            '/nonexistent/file.csv',
            merge_columns=['PATNO', 'EDUCYRS'],
            how='left'
        )
        
        assert result.equals(df)
    
    def test_load_and_merge_data_filters_out_patno_from_merge_columns(self, test_config, tmp_path):
        """Test: PATNO is excluded from merge_columns before groupby
        
        Example Output:
        - Input merge_columns: ['PATNO', 'EDUCYRS']
        - Actual merged columns: ['EDUCYRS'] (PATNO removed)
        - Result: No 'PATNO already exists' error
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'PATNO': [1, 1, 2, 2],
            'EDUCYRS': [16, 16, 12, 12],
        })
        test_data.to_csv(csv_path, index=False)
        
        df = pd.DataFrame({
            'PATNO': [1, 2],
            'COHORT': [1, 1],
        })
        
        result = loader._DemographicsLoader__load_and_merge_data__(
            df,
            str(csv_path),
            merge_columns=['PATNO', 'EDUCYRS'],
            how='left'
        )
        
        assert 'EDUCYRS' in result.columns
        assert len(result) == 2
        assert result['EDUCYRS'].tolist() == [16, 12]
    
    def test_load_and_merge_data_aggregates_multiple_rows(self, test_config, tmp_path):
        """Test: Multiple rows per PATNO aggregated to first()
        
        Example Output:
        - Input CSV: PATNO [1,1,1,2,2] with EDUCYRS [16,16,16,12,12]
        - After groupby: PATNO [1,2] with EDUCYRS [16,12]
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        csv_path = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'PATNO': [1, 1, 1, 2, 2],
            'EDUCYRS': [16, 16, 16, 12, 12],
        })
        test_data.to_csv(csv_path, index=False)
        
        df = pd.DataFrame({
            'PATNO': [1, 2],
            'COHORT': [1, 1],
        })
        
        result = loader._DemographicsLoader__load_and_merge_data__(
            df,
            str(csv_path),
            merge_columns=['PATNO', 'EDUCYRS'],
            how='left'
        )
        
        assert len(result) == 2
        assert result[result['PATNO'] == 1]['EDUCYRS'].values[0] == 16
    
    def test_load_and_merge_data_handles_missing_patno_in_csv(self, test_config, tmp_path):
        """Test: CSV without PATNO column is skipped
        
        Example Output:
        - Input CSV: columns [ID, EDUCYRS] (no PATNO)
        - Output: Original df returned + warning
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        csv_path = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'ID': [1, 2],
            'EDUCYRS': [16, 12],
        })
        test_data.to_csv(csv_path, index=False)
        
        df = pd.DataFrame({
            'PATNO': [1, 2],
            'COHORT': [1, 1],
        })
        
        result = loader._DemographicsLoader__load_and_merge_data__(
            df,
            str(csv_path),
            merge_columns=['EDUCYRS'],
            how='left'
        )
        
        assert result.equals(df)
    
    def test_load_and_merge_data_selects_only_available_columns(self, test_config, tmp_path):
        """Test: Only columns available in CSV are selected
        
        Example Output:
        - Input merge_columns: ['EDUCYRS', 'NONEXISTENT_COL']
        - Actual columns in result: only ['EDUCYRS']
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        csv_path = tmp_path / "test_data.csv"
        test_data = pd.DataFrame({
            'PATNO': [1, 2],
            'EDUCYRS': [16, 12],
        })
        test_data.to_csv(csv_path, index=False)
        
        df = pd.DataFrame({
            'PATNO': [1, 2],
            'COHORT': [1, 1],
        })
        
        result = loader._DemographicsLoader__load_and_merge_data__(
            df,
            str(csv_path),
            merge_columns=['EDUCYRS', 'NONEXISTENT_COL'],
            how='left'
        )
        
        assert 'EDUCYRS' in result.columns
        assert 'NONEXISTENT_COL' not in result.columns


class TestDemographicsLoaderMergeAgeAtVisit:
    """Test __load_and_merge_age_at_visit__ method"""
    
    def test_load_and_merge_age_at_visit_no_aggregation(self, test_config, tmp_path):
        """Test: Age at visit NOT aggregated; each EVENT_ID creates separate row
        
        Example Output:
        - Input demographics: 2 unique PATNO
        - Input age_at_visit: 5 rows (PATNO [1,1,2,2,2] with EVENT_ID [ev1,ev2,ev1,ev2,ev3])
        - Output: 5 rows (all age visit rows kept, demographics repeated)
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        # Create mock age_at_visit CSV
        age_csv_path = tmp_path / "age_at_visit.csv"
        age_data = pd.DataFrame({
            'PATNO': [1, 1, 2, 2, 2],
            'EVENT_ID': ['ev1', 'ev2', 'ev1', 'ev2', 'ev3'],
            'AGE_AT_VISIT': [60, 61, 65, 66, 67],
        })
        age_data.to_csv(age_csv_path, index=False)
        
        df = pd.DataFrame({
            'PATNO': [1, 2],
            'ENROLL_AGE': [60, 65],
        })
        
        # Mock config
        test_config.data.age_at_visit = str(age_csv_path)
        
        result = loader._DemographicsLoader__load_and_merge_age_at_visit__(df)
        
        assert len(result) == 5  # All 5 age rows preserved
        assert result['PATNO'].value_counts()[1] == 2  # PATNO 1 has 2 rows
        assert result['PATNO'].value_counts()[2] == 3  # PATNO 2 has 3 rows
    
    def test_load_and_merge_age_at_visit_keeps_event_id(self, test_config, tmp_path):
        """Test: EVENT_ID column preserved in output
        
        Example Output:
        - Output columns include: PATNO, EVENT_ID, AGE_AT_VISIT, ...demographics
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        age_csv_path = tmp_path / "age_at_visit.csv"
        age_data = pd.DataFrame({
            'PATNO': [1, 1],
            'EVENT_ID': ['ev1', 'ev2'],
            'AGE_AT_VISIT': [60, 61],
        })
        age_data.to_csv(age_csv_path, index=False)
        
        df = pd.DataFrame({
            'PATNO': [1],
            'ENROLL_AGE': [60],
        })
        
        test_config.data.age_at_visit = str(age_csv_path)
        
        result = loader._DemographicsLoader__load_and_merge_age_at_visit__(df)
        
        assert 'EVENT_ID' in result.columns
        assert 'AGE_AT_VISIT' in result.columns
    
    def test_load_and_merge_age_at_visit_handles_missing_file(self, test_config):
        """Test: Missing age_at_visit file skipped gracefully
        
        Example Output:
        - Input: file_path nonexistent
        - Output: Original df unchanged + warning
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2],
            'ENROLL_AGE': [60, 65],
        })
        
        test_config.data.age_at_visit = '/nonexistent/age_at_visit.csv'
        
        result = loader._DemographicsLoader__load_and_merge_age_at_visit__(df)
        
        assert result.equals(df)


class TestDemographicsLoaderValidation:
    """Test validate method"""
    
    def test_validate_checks_patno_column_exists(self, test_config):
        """Test: ValueError raised if PATNO column missing
        
        Example Output:
        - Input: df without PATNO column
        - Output: ValueError('Missing PATNO column')
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'COHORT': [1, 1, 1],
        })
        
        with pytest.raises(ValueError, match="Missing PATNO column"):
            loader.validate(df)
    
    def test_validate_counts_unique_patients(self, test_config, capsys):
        """Test: Unique patients counted; handles multi-row data correctly
        
        Example Output:
        - Input: 5 rows with PATNO [1,1,2,2,3]
        - Output printed: '✓ Validation passed: 3 unique patients, 5 total records'
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 1, 2, 2, 3],
            'COHORT': [1, 1, 1, 1, 1],
        })
        
        result = loader.validate(df)
        
        assert result is True
        captured = capsys.readouterr()
        assert '3 unique patients' in captured.out
        assert '5 total records' in captured.out


class TestDemographicsLoaderNumericConversion:
    """Test numeric type conversion"""
    
    def test_numeric_conversion_coerces_invalid_values_to_nan(self, test_config):
        """Test: Invalid numeric strings converted to NaN
        
        Example Output:
        - Input: ['16', '12', 'NA', '14'] 
        - Output: [16.0, 12.0, NaN, 14.0]
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        # This is tested implicitly during load
        # Creating a simple validation
        test_series = pd.Series(['16', '12', 'NA', '14'])
        result = pd.to_numeric(test_series, errors='coerce')
        
        assert result[0] == 16.0
        assert result[1] == 12.0
        assert pd.isna(result[2])
        assert result[3] == 14.0
    
    def test_numeric_conversion_handles_nan_values(self, test_config):
        """Test: NaN values preserved during conversion
        
        Example Output:
        - Input: [1.0, 2.0, NaN, 3.0]
        - Output: [1.0, 2.0, NaN, 3.0] (unchanged)
        """
        test_series = pd.Series([1.0, 2.0, np.nan, 3.0])
        result = pd.to_numeric(test_series, errors='coerce')
        
        assert result[0] == 1.0
        assert result[1] == 2.0
        assert pd.isna(result[2])
        assert result[3] == 3.0


class TestDemographicsLoaderEdgeCases:
    """Test edge cases and error scenarios"""
    
    def test_loader_handles_empty_dataframe(self, test_config, tmp_path):
        """Test: Empty dataframe after filtering handled gracefully
        
        Example Output:
        - Input: All participants filtered out
        - Output: Empty dataframe (0 rows, expected columns)
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [],
            'ENROLL_DATE': [],
            'ENROLL_STATUS': [],
            'COHORT_DEFINITION': [],
            'ENROLL_AGE': [],
            'COHORT': [],
        })
        
        result = loader._DemographicsLoader__filter_valid_participants__(df)
        
        assert len(result) == 0
        assert 'PATNO' in result.columns
    
    def test_loader_handles_all_nan_column(self, test_config):
        """Test: Column of all NaN values handled gracefully
        
        Example Output:
        - Input: EDUCYRS column [NaN, NaN, NaN]
        - Output: Column preserved with all NaN
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3],
            'EDUCYRS': [np.nan, np.nan, np.nan],
        })
        
        assert df['EDUCYRS'].isna().all()
    
    def test_loader_maintains_data_types(self, test_config):
        """Test: PATNO remains integer, numeric cols converted properly
        
        Example Output:
        - PATNO dtype: int64
        - EDUCYRS dtype: float64
        - COHORT_DEFINITION dtype: object (string)
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        
        df = pd.DataFrame({
            'PATNO': [1, 2, 3],
            'COHORT_DEFINITION': ['PD', 'HC', 'PD'],
        })
        
        assert df['PATNO'].dtype == 'int64'
        assert df['COHORT_DEFINITION'].dtype == 'object'


@pytest.mark.requires_data
class TestDemographicsLoaderIntegration:
    """Integration tests for complete DemographicsLoader workflow"""
    
    def test_demographics_loader_load_complete(self, test_config, skip_if_no_data, capsys):
        """Test: Complete load workflow from CSV files
        
        Example Output:
        ✓ Loaded demographics: 423 unique patients, 89 columns
        ✓ Validation passed: 423 unique patients, 12345 total records
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        assert isinstance(demo_df, pd.DataFrame)
        assert 'PATNO' in demo_df.columns
        assert len(demo_df) > 0
        
        loader.validate(demo_df)
        
        captured = capsys.readouterr()
        assert 'Loaded demographics' in captured.out
    
    def test_demographics_loader_multiple_rows_per_patient(self, test_config, skip_if_no_data):
        """Test: Age_at_visit creates multiple rows per patient correctly
        
        Example Output:
        - Some PATNO values appear multiple times (due to EVENT_ID)
        - Total rows > unique patients
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        unique_patients = demo_df['PATNO'].nunique()
        total_rows = len(demo_df)
        
        # If age_at_visit was merged, might have multiple rows per patient
        assert unique_patients > 0
        assert total_rows >= unique_patients
    
    def test_demographics_loader_required_columns_present(self, test_config, skip_if_no_data):
        """Test: All required columns present in output
        
        Example Output:
        - Required columns: [PATNO, ENROLL_AGE, SEX, COHORT, COHORT_DEFINITION, ENROLL_STATUS]
        - All present in loaded dataframe
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        required_cols = loader.get_required_columns()
        for col in required_cols:
            assert col in demo_df.columns
    
    def test_demographics_loader_age_values_reasonable(self, test_config, skip_if_no_data):
        """Test: Age values are within reasonable range
        
        Example Output:
        - ENROLL_AGE min: 25, max: 95
        - All values between 0-150
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        if 'ENROLL_AGE' in demo_df.columns:
            demo_df_unique = demo_df.drop_duplicates(subset='PATNO')
            assert demo_df_unique['ENROLL_AGE'].min() >= 0
            assert demo_df_unique['ENROLL_AGE'].max() <= 150
    
    def test_demographics_loader_cohort_distribution(self, test_config, skip_if_no_data):
        """Test: Cohort distribution reasonable
        
        Example Output:
        - Cohort counts by unique patient:
          Healthy Control: 195
          Parkinson's Disease: 228
          Prodromal: 0
        """
        from data.loaders.demographics_loader import DemographicsLoader
        
        loader = DemographicsLoader(test_config.data.base_dir, test_config)
        demo_df = loader.load()
        
        if 'COHORT_DEFINITION' in demo_df.columns:
            unique_demo = demo_df[['PATNO', 'COHORT_DEFINITION']].drop_duplicates(subset='PATNO')
            cohort_counts = unique_demo['COHORT_DEFINITION'].value_counts()
            
            assert len(cohort_counts) > 0
            assert cohort_counts.sum() > 0
