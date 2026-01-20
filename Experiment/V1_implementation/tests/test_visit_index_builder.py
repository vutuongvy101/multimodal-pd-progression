"""
Tests for VisitIndexBuilder

VisitIndexBuilder creates the master timeline for each patient that is independent
of any specific modality. All modalities are then aligned TO this master timeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestVisitIndexBuilder:
    """Test VisitIndexBuilder with mock data"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object"""
        class MockConfig:
            pass
        return MockConfig()
    
    @pytest.fixture
    def visit_index_builder(self, mock_config):
        """Create a VisitIndexBuilder instance"""
        from data.visit_index_builder import VisitIndexBuilder
        return VisitIndexBuilder(mock_config)
    
    @pytest.fixture
    def mock_updrs_data(self):
        """Mock UPDRS data with visits from multiple patients"""
        return pd.DataFrame({
            'PATNO': [1001, 1001, 1001, 1002, 1002, 1003],
            'EVENT_ID': ['BL', 'V01', 'V02', 'BL', 'V01', 'BL'],
            'INFODT': [
                '2020-01-15', '2020-04-20', '2020-07-10',
                '2020-02-01', '2020-05-15', '2020-03-01'
            ],
            'NP3TOT': [25.0, 27.0, 30.0, 18.0, 20.0, 22.0],
        })
    
    @pytest.fixture
    def mock_medication_data(self):
        """Mock medication data (different visits than UPDRS)"""
        return pd.DataFrame({
            'PATNO': [1001, 1001, 1002, 1004],  # 1004 not in UPDRS
            'EVENT_ID': ['BL', 'V01', 'BL', 'BL'],
            'INFODT': [
                '2020-01-15', '2020-04-20',
                '2020-02-01', '2020-04-01'
            ],
            'LEDD': [0.0, 100.0, 50.0, 75.0],
        })
    
    @pytest.fixture
    def mock_non_motor_data(self):
        """Mock non-motor data"""
        return pd.DataFrame({
            'PATNO': [1001, 1002],
            'EVENT_ID': ['V02', 'V01'],
            'INFODT': ['2020-07-10', '2020-05-15'],
            'MCATOT': [28.0, 26.0],
        })
    
    def test_build_from_sources_basic(self, visit_index_builder, mock_updrs_data):
        """Test building visit_index from a single source"""
        visit_index = visit_index_builder.build_from_sources([mock_updrs_data])
        
        # Check structure
        assert isinstance(visit_index, pd.DataFrame)
        assert len(visit_index) == 6  # All 6 visits from mock data
        assert visit_index['PATNO'].nunique() == 3  # 3 patients
        
        # Check required columns
        required_cols = ['PATNO', 'EVENT_ID', 'visit_date', 'months_since_baseline', 
                        'visit_order', 'delta_months']
        for col in required_cols:
            assert col in visit_index.columns, f"Missing column: {col}"
    
    def test_build_from_sources_multiple_sources(self, visit_index_builder, 
                                                  mock_updrs_data, mock_medication_data):
        """Test building visit_index from multiple sources (UNION)"""
        visit_index = visit_index_builder.build_from_sources([
            mock_updrs_data, mock_medication_data
        ])
        
        # Should include visits from both sources
        # UPDRS: P1001[BL,V01,V02], P1002[BL,V01], P1003[BL] = 6 visits
        # Meds: P1001[BL,V01], P1002[BL], P1004[BL] = 4 visits
        # UNION: P1001[BL,V01,V02], P1002[BL,V01], P1003[BL], P1004[BL] = 7 visits
        assert len(visit_index) == 7
        assert visit_index['PATNO'].nunique() == 4  # Should include P1004
        
        # Check that P1004 (only in meds) is included
        assert 1004 in visit_index['PATNO'].values
    
    def test_build_from_sources_duplicate_visits(self, visit_index_builder):
        """Test handling duplicate (PATNO, EVENT_ID) across sources"""
        # Same visit appears in multiple sources
        source1 = pd.DataFrame({
            'PATNO': [1001, 1001],
            'EVENT_ID': ['BL', 'V01'],
            'INFODT': ['2020-01-15', '2020-04-20'],
        })
        source2 = pd.DataFrame({
            'PATNO': [1001, 1001],
            'EVENT_ID': ['BL', 'V01'],  # Same visits
            'INFODT': ['2020-01-15', '2020-04-20'],  # Same dates
        })
        
        visit_index = visit_index_builder.build_from_sources([source1, source2])
        
        # Should deduplicate to 2 visits
        assert len(visit_index) == 2
        assert (visit_index['EVENT_ID'].values == ['BL', 'V01']).all()
    
    def test_build_from_sources_filter_valid_patnos(self, visit_index_builder,
                                                     mock_updrs_data):
        """Test filtering to valid patient IDs"""
        valid_patnos = [1001, 1002]  # Exclude 1003
        
        visit_index = visit_index_builder.build_from_sources(
            [mock_updrs_data], valid_patnos=valid_patnos
        )
        
        assert visit_index['PATNO'].nunique() == 2
        assert 1003 not in visit_index['PATNO'].values
        assert 1001 in visit_index['PATNO'].values
        assert 1002 in visit_index['PATNO'].values
    
    def test_build_from_sources_empty_source(self, visit_index_builder, mock_updrs_data):
        """Test handling empty DataFrames in source list"""
        empty_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        visit_index = visit_index_builder.build_from_sources([mock_updrs_data, empty_df])
        
        # Should work fine, empty DataFrame ignored
        assert len(visit_index) == 6
    
    def test_build_from_sources_missing_infodt(self, visit_index_builder):
        """Test building visit_index when INFODT is missing"""
        data_no_dates = pd.DataFrame({
            'PATNO': [1001, 1001],
            'EVENT_ID': ['BL', 'V01'],
            # No INFODT column
        })
        
        visit_index = visit_index_builder.build_from_sources([data_no_dates])
        
        # Should still work, but visit_date will be NaT
        assert len(visit_index) == 2
        assert 'visit_date' in visit_index.columns
        assert visit_index['visit_date'].isna().all()
    
    def test_compute_time_features(self, visit_index_builder):
        """Test time feature computation"""
        # Create visit_index with dates
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001, 1001, 1002, 1002],
            'EVENT_ID': ['BL', 'V01', 'V02', 'BL', 'V01'],
            'visit_date': pd.to_datetime([
                '2020-01-15', '2020-04-20', '2020-07-10',
                '2020-02-01', '2020-05-15'
            ])
        })
        
        visit_index = visit_index_builder._compute_time_features(visit_index)
        
        # Check months_since_baseline
        p1001_visits = visit_index[visit_index['PATNO'] == 1001].sort_values('visit_order')
        assert p1001_visits.iloc[0]['months_since_baseline'] == 0.0  # Baseline
        assert p1001_visits.iloc[1]['months_since_baseline'] > 0  # V01 > baseline
        
        # Check visit_order
        assert (visit_index['visit_order'].values == [0, 1, 2, 0, 1]).all()
        
        # Check delta_months
        assert p1001_visits.iloc[0]['delta_months'] == 0.0  # First visit
        assert p1001_visits.iloc[1]['delta_months'] > 0  # Gap from baseline
    
    def test_filter_by_min_visits(self, visit_index_builder, mock_updrs_data):
        """Test filtering patients by minimum number of visits"""
        visit_index = visit_index_builder.build_from_sources([mock_updrs_data])
        
        # P1001: 3 visits, P1002: 2 visits, P1003: 1 visit
        filtered = visit_index_builder.filter_by_min_visits(visit_index, min_visits=2)
        
        assert filtered['PATNO'].nunique() == 2  # Only P1001 and P1002
        assert 1003 not in filtered['PATNO'].values  # P1003 filtered out
        assert len(filtered) == 5  # 3 + 2 visits
    
    def test_filter_by_min_visits_all_kept(self, visit_index_builder, mock_updrs_data):
        """Test filtering when all patients meet min_visits threshold"""
        visit_index = visit_index_builder.build_from_sources([mock_updrs_data])
        
        filtered = visit_index_builder.filter_by_min_visits(visit_index, min_visits=1)
        
        # All patients should be kept
        assert filtered['PATNO'].nunique() == visit_index['PATNO'].nunique()
        assert len(filtered) == len(visit_index)
    
    def test_align_modality_complete_data(self, visit_index_builder):
        """Test aligning modality when all visits have data"""
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001, 1001],
            'EVENT_ID': ['BL', 'V01', 'V02'],
        })
        
        modality_df = pd.DataFrame({
            'PATNO': [1001, 1001, 1001],
            'EVENT_ID': ['BL', 'V01', 'V02'],
            'FEATURE1': [10.0, 12.0, 15.0],
            'FEATURE2': [5.0, 6.0, 7.0],
        })
        
        result = visit_index_builder.align_modality(
            visit_index, modality_df, ['FEATURE1', 'FEATURE2'], 'test_modality'
        )
        
        assert result['values'].shape == (3, 2)  # 3 visits, 2 features
        assert result['mask'].shape == (3, 2)
        
        # All data present, mask should be all zeros (not missing)
        assert (result['mask'] == 0).all()
        
        # Values should match
        assert result['values'][0, 0] == 10.0
        assert result['values'][1, 0] == 12.0
    
    def test_align_modality_missing_data(self, visit_index_builder):
        """Test aligning modality when some visits have missing data"""
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001, 1001],
            'EVENT_ID': ['BL', 'V01', 'V02'],
        })
        
        # V02 is missing in modality_df
        modality_df = pd.DataFrame({
            'PATNO': [1001, 1001],
            'EVENT_ID': ['BL', 'V01'],
            'FEATURE1': [10.0, 12.0],
        })
        
        result = visit_index_builder.align_modality(
            visit_index, modality_df, ['FEATURE1'], 'test_modality'
        )
        
        # V02 should have mask=1 (missing)
        assert result['mask'][2, 0] == 1.0  # V02 is missing
        assert result['mask'][0, 0] == 0.0  # BL has data
        assert result['mask'][1, 0] == 0.0  # V01 has data
    
    def test_align_modality_empty_modality(self, visit_index_builder):
        """Test aligning modality when modality DataFrame is empty"""
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001],
            'EVENT_ID': ['BL', 'V01'],
        })
        
        empty_df = pd.DataFrame()
        
        result = visit_index_builder.align_modality(
            visit_index, empty_df, ['FEATURE1'], 'empty_modality'
        )
        
        # All should be masked (missing)
        assert (result['mask'] == 1).all()
        assert (result['values'] == 0).all()
    
    def test_align_modality_partial_features(self, visit_index_builder):
        """Test aligning when only some requested features exist"""
        visit_index = pd.DataFrame({
            'PATNO': [1001],
            'EVENT_ID': ['BL'],
        })
        
        modality_df = pd.DataFrame({
            'PATNO': [1001],
            'EVENT_ID': ['BL'],
            'FEATURE1': [10.0],  # Only FEATURE1 exists
            # FEATURE2 missing
        })
        
        result = visit_index_builder.align_modality(
            visit_index, modality_df, ['FEATURE1', 'FEATURE2'], 'partial'
        )
        
        # FEATURE1 should be present (mask=0)
        # FEATURE2 should be missing (mask=1)
        assert result['mask'][0, 0] == 0.0  # FEATURE1 present
        assert result['mask'][0, 1] == 1.0  # FEATURE2 missing
    
    def test_event_order_sorting(self, visit_index_builder):
        """Test that visits are sorted correctly by EVENT_ID order when dates are missing"""
        # Visits in wrong order
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001, 1001, 1001],
            'EVENT_ID': ['V02', 'BL', 'V01', 'V03'],
            # No visit_date
        })
        visit_index['visit_date'] = pd.NaT
        
        visit_index = visit_index_builder._compute_time_features(visit_index)
        
        # Should be sorted: BL (0), V01 (1), V02 (2), V03 (3)
        sorted_visits = visit_index.sort_values('visit_order')
        assert (sorted_visits['EVENT_ID'].values == ['BL', 'V01', 'V02', 'V03']).all()
        assert (sorted_visits['visit_order'].values == [0, 1, 2, 3]).all()
    
    def test_build_from_sources_remote_visits(self, visit_index_builder):
        """Test handling of remote visits (R01, R02, etc.)"""
        data_with_remote = pd.DataFrame({
            'PATNO': [1001, 1001, 1001],
            'EVENT_ID': ['BL', 'V01', 'R01'],  # Remote visit
            'INFODT': ['2020-01-15', '2020-04-20', '2021-01-15'],
        })
        
        visit_index = visit_index_builder.build_from_sources([data_with_remote])
        
        # Remote visit should be included
        assert 'R01' in visit_index['EVENT_ID'].values
        assert len(visit_index) == 3
    
    def test_months_since_baseline_calculation(self, visit_index_builder):
        """Test accurate months_since_baseline calculation"""
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001],
            'EVENT_ID': ['BL', 'V01'],
            'visit_date': pd.to_datetime(['2020-01-15', '2020-04-20']),
        })
        
        visit_index = visit_index_builder._compute_time_features(visit_index)
        
        # ~3 months difference
        p1001 = visit_index[visit_index['PATNO'] == 1001].sort_values('visit_order')
        baseline_time = p1001.iloc[0]['months_since_baseline']
        v01_time = p1001.iloc[1]['months_since_baseline']
        
        assert baseline_time == 0.0
        assert 2.5 < v01_time < 4.0  # Approximately 3 months
    
    def test_delta_months_calculation(self, visit_index_builder):
        """Test delta_months calculation (time gap between consecutive visits)"""
        visit_index = pd.DataFrame({
            'PATNO': [1001, 1001, 1001],
            'EVENT_ID': ['BL', 'V01', 'V02'],
            'visit_date': pd.to_datetime([
                '2020-01-15', '2020-04-20', '2020-07-10'
            ]),
        })
        
        visit_index = visit_index_builder._compute_time_features(visit_index)
        
        p1001 = visit_index[visit_index['PATNO'] == 1001].sort_values('visit_order')
        
        assert p1001.iloc[0]['delta_months'] == 0.0  # First visit
        assert p1001.iloc[1]['delta_months'] > 0  # Gap from BL to V01
        assert p1001.iloc[2]['delta_months'] > 0  # Gap from V01 to V02


class TestVisitIndexBuilderIntegration:
    """Integration tests with test_config (requires data files)"""
    
    @pytest.mark.requires_data
    def test_build_from_real_sources(self, test_config, skip_if_no_data):
        """Test building visit_index from real data sources"""
        from data.visit_index_builder import VisitIndexBuilder
        from data.loaders.updrs_loader import UPDRSLoader
        from data.loaders.medication_loader import MedicationLoader
        
        builder = VisitIndexBuilder(test_config)
        
        # Load real data
        updrs_loader = UPDRSLoader(test_config.data.base_dir, test_config)
        updrs_df = updrs_loader.load()
        
        try:
            med_loader = MedicationLoader(test_config.data.base_dir, test_config)
            med_df = med_loader.load()
        except Exception:
            med_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        # Build visit_index
        if len(med_df) > 0:
            visit_index = builder.build_from_sources([updrs_df, med_df])
        else:
            visit_index = builder.build_from_sources([updrs_df])
        
        # Basic checks
        assert isinstance(visit_index, pd.DataFrame)
        assert len(visit_index) > 0
        assert visit_index['PATNO'].nunique() > 0
        
        # Check columns
        required_cols = ['PATNO', 'EVENT_ID', 'visit_order', 'months_since_baseline', 
                        'delta_months']
        for col in required_cols:
            assert col in visit_index.columns
        
        # Check that visit_order is sequential within each patient
        for patno in visit_index['PATNO'].unique()[:5]:  # Check first 5 patients
            patient_visits = visit_index[visit_index['PATNO'] == patno].sort_values('visit_order')
            visit_orders = patient_visits['visit_order'].values
            assert (visit_orders == np.arange(len(visit_orders))).all(), \
                f"Visit order not sequential for patient {patno}"
    
    @pytest.mark.requires_data
    def test_filter_by_min_visits_real_data(self, test_config, skip_if_no_data):
        """Test filtering with real data"""
        from data.visit_index_builder import VisitIndexBuilder
        from data.loaders.updrs_loader import UPDRSLoader
        
        builder = VisitIndexBuilder(test_config)
        updrs_loader = UPDRSLoader(test_config.data.base_dir, test_config)
        updrs_df = updrs_loader.load()
        
        visit_index = builder.build_from_sources([updrs_df])
        
        # Test with min_visits=3
        n_before = visit_index['PATNO'].nunique()
        filtered = builder.filter_by_min_visits(visit_index, min_visits=3)
        n_after = filtered['PATNO'].nunique()
        
        # Should have filtered out some patients
        assert n_after <= n_before
        assert len(filtered) > 0
        
        # Verify all remaining patients have >= 3 visits
        visit_counts = filtered.groupby('PATNO').size()
        assert (visit_counts >= 3).all()
    
    @pytest.mark.requires_data
    def test_align_modality_real_data(self, test_config, skip_if_no_data):
        """Test aligning real modality data to visit_index"""
        from data.visit_index_builder import VisitIndexBuilder
        from data.loaders.updrs_loader import UPDRSLoader
        
        builder = VisitIndexBuilder(test_config)
        updrs_loader = UPDRSLoader(test_config.data.base_dir, test_config)
        updrs_df = updrs_loader.load()
        
        visit_index = builder.build_from_sources([updrs_df])
        visit_index = builder.filter_by_min_visits(visit_index, min_visits=2)
        
        # Align motor features
        motor_features = test_config.features.motor_features[:5]  # First 5 features
        
        result = builder.align_modality(
            visit_index, updrs_df, motor_features, 'motor'
        )
        
        # Check output structure
        assert 'values' in result
        assert 'mask' in result
        assert result['values'].shape[0] == len(visit_index)
        assert result['values'].shape[1] == len(motor_features)
        
        # Check that some data is present (mask should have some zeros)
        mask_sum = result['mask'].sum()
        total_mask_elements = result['mask'].size
        data_coverage = 1.0 - (mask_sum / total_mask_elements)
        
        assert data_coverage > 0, "No data coverage - all masked"
        assert data_coverage <= 1.0, "Data coverage should be <= 1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
