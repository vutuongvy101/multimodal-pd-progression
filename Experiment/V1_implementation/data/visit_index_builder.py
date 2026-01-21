"""
Visit Index Builder - Creates the master timeline for each patient

The visit_index is NOT a modality - it's the indexing structure that defines 
which visits exist for each patient and when they occurred. All modalities 
are then aligned TO this master timeline.

Key insight: You do NOT filter timelines based on missing modalities.
You keep all visits and mask out data/loss contributions where data is missing.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings


class VisitIndexBuilder:
    """
    Builds the master visit timeline from all available data sources.
    
    The visit_index defines the "anchors" that your dataset will iterate over.
    For each visit anchor, you then look up whether motor/meds/clinical data 
    exist and create masks if they do not.
    
    visit_index[PATNO] = [ (EVENT_ID_0, t0), (EVENT_ID_1, t1), ..., (EVENT_ID_T, tT) ]
    
    Columns in visit_index:
        - PATNO: Patient ID
        - EVENT_ID: Visit code (BL, V01, V02, R01, U01, etc.)
        - visit_date: Actual date (from INFODT)
        - months_since_baseline: Continuous time since baseline
        - visit_order: Integer (0, 1, 2, ...) within patient
        - delta_months: Time gap from previous visit (0 for first visit)
    """
    
    # Standard PPMI event ordering
    # BL/SC are baseline (order 0), V01-V25 are regular visits
    # R01-R09 are remote visits, U01-U09 are unscheduled visits
    EVENT_ORDER = {
        'SC': 0,   # Screening (same as baseline for timeline purposes)
        'BL': 0,   # Baseline
        **{f'V{i:02d}': i for i in range(1, 26)},       # V01-V25
        **{f'R{i:02d}': 100 + i for i in range(1, 10)}, # Remote visits
        **{f'U{i:02d}': 200 + i for i in range(1, 10)}, # Unscheduled visits
    }
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with data and feature settings
        """
        self.config = config
    
    def build_from_sources(
        self, 
        source_dfs: List[pd.DataFrame],
        valid_patnos: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Build visit_index from multiple data sources.
        
        The visit_index is the UNION of all visit anchors that exist across 
        all sources, then filtered and sorted.
        
        Args:
            source_dfs: List of DataFrames, each with at least [PATNO, EVENT_ID].
                       INFODT column is used for dates if available.
            valid_patnos: Optional list of valid patient IDs to filter
            
        Returns:
            visit_index DataFrame with columns:
                - PATNO, EVENT_ID, visit_date
                - months_since_baseline, visit_order, delta_months
        """
        # Collect all (PATNO, EVENT_ID, date) tuples from all sources
        all_visits = []
        
        for df in source_dfs:
            if df is None or df.empty:
                continue
            if not {'PATNO', 'EVENT_ID'}.issubset(df.columns):
                print(f"  Warning: DataFrame missing PATNO or EVENT_ID columns, skipping")
                continue
                
            subset = df[['PATNO', 'EVENT_ID']].copy()
            
            # Include date if available
            if 'INFODT' in df.columns:
                # Parse dates, handling various formats
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    subset['visit_date'] = pd.to_datetime(df['INFODT'], errors='coerce')
            elif 'visit_date' in df.columns:
                subset['visit_date'] = df['visit_date']
            
            all_visits.append(subset)
        
        if not all_visits:
            raise ValueError("No valid visit data found in source DataFrames")
        
        # Combine all sources
        combined = pd.concat(all_visits, ignore_index=True)
        
        # For duplicate (PATNO, EVENT_ID), aggregate dates
        # Use first non-null date if multiple sources have dates
        if 'visit_date' in combined.columns:
            visit_index = combined.groupby(['PATNO', 'EVENT_ID'], as_index=False).agg({
                'visit_date': lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NaT
            })
        else:
            visit_index = combined.drop_duplicates(subset=['PATNO', 'EVENT_ID']).copy()
            visit_index['visit_date'] = pd.NaT
        
        # Filter to valid patients if specified
        if valid_patnos is not None:
            n_before = len(visit_index)
            visit_index = visit_index[visit_index['PATNO'].isin(valid_patnos)].copy()
            n_after = len(visit_index)
            if n_before != n_after:
                print(f"  Filtered visit_index: {n_before} → {n_after} visits (valid patients only)")
        
        # Compute derived time features
        visit_index = self._compute_time_features(visit_index)
        
        # Sort by patient and visit order
        visit_index = visit_index.sort_values(['PATNO', 'visit_order']).reset_index(drop=True)
        
        print(f"  ✓ Built visit_index: {len(visit_index)} visits for {visit_index['PATNO'].nunique()} patients")
        print(f"    Visits per patient: {len(visit_index) / visit_index['PATNO'].nunique():.1f} average")
        
        return visit_index
    
    def _compute_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-related features for visit_index.
        
        Adds:
            - months_since_baseline: Continuous time since baseline visit
            - visit_order: Integer (0, 1, 2, ...) within patient
            - delta_months: Time gap from previous visit
        """
        df = df.copy()
        
        # Add event order for sorting (used when dates are missing)
        df['_event_order'] = df['EVENT_ID'].map(lambda x: self.EVENT_ORDER.get(x, 999))
        
        # Sort within patient by date (if available) or event order
        if 'visit_date' in df.columns and df['visit_date'].notna().any():
            # Primary sort by date, secondary by event order (for same-day visits)
            df = df.sort_values(['PATNO', 'visit_date', '_event_order'])
        else:
            df = df.sort_values(['PATNO', '_event_order'])
        
        # Compute visit_order (0-indexed within patient)
        df['visit_order'] = df.groupby('PATNO').cumcount()
        
        # Get baseline date per patient (BL or SC, whichever comes first)
        baseline_mask = df['EVENT_ID'].isin(['BL', 'SC'])
        baseline_visits = df[baseline_mask].copy()
        baseline_dates = baseline_visits.groupby('PATNO')['visit_date'].first()
        
        # months_since_baseline
        def compute_months_since_baseline(row):
            if pd.isna(row.get('visit_date')):
                return np.nan
            patno = row['PATNO']
            if patno not in baseline_dates.index:
                return np.nan
            baseline_date = baseline_dates[patno]
            if pd.isna(baseline_date):
                return np.nan
            return (row['visit_date'] - baseline_date).days / 30.44
        
        df['months_since_baseline'] = df.apply(compute_months_since_baseline, axis=1)
        
        # delta_months (time since previous visit)
        df['delta_months'] = df.groupby('PATNO')['months_since_baseline'].diff().fillna(0)
        
        # Clean up temporary column
        df = df.drop(columns=['_event_order'])
        
        return df
    
    def align_modality(
        self,
        visit_index: pd.DataFrame,
        modality_df: pd.DataFrame,
        feature_cols: List[str],
        modality_name: str = "modality"
    ) -> Dict[str, np.ndarray]:
        """
        Align a modality DataFrame to the visit_index.
        
        For each visit in visit_index, look up whether data exists in modality_df.
        If data is missing for a visit, create a mask indicating missingness.
        
        Args:
            visit_index: Master timeline DataFrame
            modality_df: Modality data with [PATNO, EVENT_ID, ...features...]
            feature_cols: List of feature column names to extract
            modality_name: Name for logging purposes
            
        Returns:
            Dict with:
                - 'values': np.ndarray [n_visits, n_features] - feature values (0 where missing)
                - 'mask': np.ndarray [n_visits, n_features] - 1 if missing, 0 if present
        """
        if modality_df is None or modality_df.empty:
            # All missing - return zeros with all-ones mask
            n_visits = len(visit_index)
            n_features = len(feature_cols)
            print(f"    {modality_name}: no data available, all masked")
            return {
                'values': np.zeros((n_visits, n_features), dtype=np.float32),
                'mask': np.ones((n_visits, n_features), dtype=np.float32)
            }
        
        # Select relevant columns from modality
        available_cols = [c for c in feature_cols if c in modality_df.columns]
        
        if not available_cols:
            # No features available - return empty with all-ones mask
            n_visits = len(visit_index)
            n_features = len(feature_cols)
            print(f"    {modality_name}: 0/{len(feature_cols)} features available, all masked")
            return {
                'values': np.zeros((n_visits, n_features), dtype=np.float32),
                'mask': np.ones((n_visits, n_features), dtype=np.float32)
            }
        
        modality_subset = modality_df[['PATNO', 'EVENT_ID'] + available_cols].copy()
        
        # Deduplicate by (PATNO, EVENT_ID) - take mean for numeric features
        # This prevents the merge from creating extra rows when modality_df has duplicates
        if len(modality_subset) != len(modality_subset.drop_duplicates(subset=['PATNO', 'EVENT_ID'])):
            # Convert feature columns to numeric before aggregation
            for col in available_cols:
                modality_subset[col] = pd.to_numeric(modality_subset[col], errors='coerce')
            
            # Group by (PATNO, EVENT_ID) and take mean of feature columns
            modality_subset = modality_subset.groupby(['PATNO', 'EVENT_ID'], as_index=False).agg({
                col: 'mean' for col in available_cols
            })
        
        # Left join: keep all visits from index
        merged = visit_index[['PATNO', 'EVENT_ID']].merge(
            modality_subset,
            on=['PATNO', 'EVENT_ID'],
            how='left'
        )
        
        # Build values and mask arrays
        # Use len(visit_index) to ensure we match the expected number of visits
        # (left join should preserve all rows from visit_index after deduplication)
        n_visits = len(visit_index)
        n_features = len(feature_cols)
        
        # Ensure merged has the same number of rows (should be true after deduplication)
        if len(merged) != n_visits:
            raise ValueError(
                f"Merge result has {len(merged)} rows but visit_index has {n_visits} rows. "
                f"This suggests duplicate (PATNO, EVENT_ID) in modality_df that weren't deduplicated."
            )
        values = np.zeros((n_visits, n_features), dtype=np.float32)
        mask = np.ones((n_visits, n_features), dtype=np.float32)  # Default: all missing
        
        for i, col in enumerate(feature_cols):
            if col in available_cols:
                col_values = merged[col].values
                # Convert to numeric, handling non-numeric values
                col_values = pd.to_numeric(col_values, errors='coerce')
                
                # Set values and mask
                valid_mask = ~np.isnan(col_values)
                values[valid_mask, i] = col_values[valid_mask]
                mask[valid_mask, i] = 0.0  # Not missing
        
        # Compute coverage statistics
        n_available = len(available_cols)
        coverage = 1.0 - mask.mean()
        print(f"    {modality_name}: {n_available}/{len(feature_cols)} features, {coverage:.1%} data coverage")
        
        return {
            'values': values,
            'mask': mask
        }
    
    def filter_by_min_visits(
        self, 
        visit_index: pd.DataFrame, 
        min_visits: int = 2
    ) -> pd.DataFrame:
        """
        Filter visit_index to only include patients with at least min_visits.
        
        Args:
            visit_index: Visit index DataFrame
            min_visits: Minimum number of visits required
            
        Returns:
            Filtered visit_index
        """
        visit_counts = visit_index.groupby('PATNO').size()
        valid_patnos = visit_counts[visit_counts >= min_visits].index
        
        n_before = visit_index['PATNO'].nunique()
        filtered = visit_index[visit_index['PATNO'].isin(valid_patnos)].copy()
        n_after = filtered['PATNO'].nunique()
        
        print(f"  Filtered to patients with ≥{min_visits} visits: {n_before} → {n_after} patients")
        
        return filtered


if __name__ == "__main__":
    # Test the VisitIndexBuilder
    print("Testing VisitIndexBuilder...")
    print("=" * 60)
    
    # Create mock data
    updrs_df = pd.DataFrame({
        'PATNO': [1001, 1001, 1001, 1002, 1002],
        'EVENT_ID': ['BL', 'V01', 'V02', 'BL', 'V01'],
        'INFODT': ['2020-01-15', '2020-04-20', '2020-07-10', '2020-02-01', '2020-05-15'],
        'NP3TOT': [25.0, 27.0, 30.0, 18.0, 20.0],
    })
    
    medication_df = pd.DataFrame({
        'PATNO': [1001, 1001, 1002],  # Missing V02 for 1001, V01 for 1002
        'EVENT_ID': ['BL', 'V01', 'BL'],
        'INFODT': ['2020-01-15', '2020-04-20', '2020-02-01'],
        'LEDD': [0.0, 100.0, 50.0],
    })
    
    # Create mock config
    class MockConfig:
        class features:
            motor_features = ['NP3TOT']
            medication_features = ['LEDD']
    
    config = MockConfig()
    builder = VisitIndexBuilder(config)
    
    # Build visit index
    print("\n1. Building visit_index from sources...")
    visit_index = builder.build_from_sources([updrs_df, medication_df])
    
    print(f"\nVisit Index:\n{visit_index}")
    
    # Align modalities
    print("\n2. Aligning motor modality...")
    motor_aligned = builder.align_modality(
        visit_index, updrs_df, ['NP3TOT'], 'motor'
    )
    print(f"   Values shape: {motor_aligned['values'].shape}")
    print(f"   Mask shape: {motor_aligned['mask'].shape}")
    
    print("\n3. Aligning medication modality...")
    med_aligned = builder.align_modality(
        visit_index, medication_df, ['LEDD'], 'medication'
    )
    print(f"   Values shape: {med_aligned['values'].shape}")
    print(f"   Mask (shows missing): {med_aligned['mask'].flatten()}")
    
    print("\n" + "=" * 60)
    print("✓ VisitIndexBuilder working correctly!")
