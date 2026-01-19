"""
Data Integrator - Combines all loaders into final dataset
Run this AFTER all individual loaders are working
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try relative imports first, then absolute
try:
    from data.loaders.genetics_loader import GeneticsLoader
    from data.loaders.demographics_loader import DemographicsLoader
    from data.loaders.updrs_loader import UPDRSLoader
    from data.loaders.clinical_loader import ClinicalAssessmentsLoader
    from data.loaders.medication_loader import MedicationLoader
    from data.loaders.age_at_visit_loader import AgeAtVisitLoader
except ImportError:
    # Fall back to direct imports if running from data/ directory
    from loaders.genetics_loader import GeneticsLoader
    from loaders.demographics_loader import DemographicsLoader
    from loaders.updrs_loader import UPDRSLoader
    from loaders.clinical_loader import ClinicalAssessmentsLoader
    from loaders.medication_loader import MedicationLoader
    from loaders.age_at_visit_loader import AgeAtVisitLoader


class DataIntegrator:
    """
    Integrates data from all loaders into final dataset
    Handles merging, missing data, slope computation
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dict
        """
        self.config = config
        self.base_dir = config.data.base_dir
        
        # Load participant_status once and share across all loaders
        print("Loading participant status (shared across all loaders)...")
        self.valid_participants = self._load_valid_participants(config)
        print(f"  ✓ Loaded {len(self.valid_participants)} valid participants")
        
        # Initialize all loaders with shared valid_participants
        print("\nInitializing loaders...")
        self.genetics_loader = GeneticsLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.demographics_loader = DemographicsLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.updrs_loader = UPDRSLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.clinical_loader = ClinicalAssessmentsLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.medication_loader = MedicationLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.age_at_visit_loader = AgeAtVisitLoader(self.base_dir, config, valid_participants=self.valid_participants)
    
    def _load_valid_participants(self, config) -> pd.DataFrame:
        """
        Load and filter participant_status once.
        This DataFrame is shared across all loaders to avoid multiple file loads.
        
        Returns:
            DataFrame with filtered valid participants
        """
        import os
        
        def resolve_path(file_path):
            """Resolve file path, handling relative paths that start with ../"""
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        # Load participant status
        status_path = resolve_path(config.data.participant_status)
        if not os.path.exists(status_path):
            raise FileNotFoundError(f"Participant status file not found: {status_path}\n"
                                  f"  Checked: {os.path.abspath(status_path)}")
        df = pd.read_csv(status_path)
        
        # Filter valid participants (same logic as BaseDataLoader._filter_valid_participants)
        # Remove participants with null ENROLL_DATE
        df = df.dropna(subset=['ENROLL_DATE'])

        # Keep only participants with valid enrollment status
        valid_statuses = [
            'Complete', 
            'Enrolled', 
            'Withdraw Deceased', 
            'Withdrew'
        ]
        df = df[df['ENROLL_STATUS'].isin(valid_statuses)]

        # Exclude SWEDD cohort
        valid_cohorts = [
            'Healthy Control', 
            "Parkinson's Disease", 
            'Prodromal'
        ]
        df = df[df['COHORT_DEFINITION'].isin(valid_cohorts)]
    
        # Reset index for a clean dataframe
        df.reset_index(drop=True, inplace=True)
        
        return df
    
    def _merge_longitudinal_data(self, longitudinal_df: pd.DataFrame, new_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Helper method to merge additional longitudinal data into the main longitudinal DataFrame.
        Handles duplicate months_since_baseline columns by filling missing values and dropping duplicates.
        
        Args:
            longitudinal_df: Main longitudinal DataFrame to merge into
            new_df: New DataFrame to merge (e.g., clinical_df, medication_df)
            suffix: Suffix to use for duplicate columns (e.g., '_clinical', '_med', '_age')
            
        Returns:
            Updated longitudinal_df with merged data
        """
        if len(new_df) > 0:
            longitudinal_df = longitudinal_df.merge(
                new_df,
                on=['PATNO', 'EVENT_ID'],
                how='outer',
                suffixes=('', suffix)
            )
            # Keep the first months_since_baseline if there are duplicates
            months_col = f'months_since_baseline{suffix}'
            if months_col in longitudinal_df.columns:
                # Ensure numeric types to avoid FutureWarning about downcasting
                longitudinal_df['months_since_baseline'] = pd.to_numeric(
                    longitudinal_df['months_since_baseline'], errors='coerce'
                )
                longitudinal_df[months_col] = pd.to_numeric(
                    longitudinal_df[months_col], errors='coerce'
                )
                longitudinal_df['months_since_baseline'] = longitudinal_df['months_since_baseline'].fillna(
                    longitudinal_df[months_col]
                )
                longitudinal_df = longitudinal_df.drop(months_col, axis=1)
        
        return longitudinal_df
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from all loaders
        
        Returns:
            Dictionary with 'static' and 'longitudinal' DataFrames
        """
        print("\n" + "=" * 80)
        print("LOADING ALL DATA")
        print("=" * 80)
        
        # Load static data (patient-level)
        print("\n--- Loading Static Data ---")
        genetics_df = self.genetics_loader.load()
        demographics_df = self.demographics_loader.load()
        
        # Merge static data
        static_df = demographics_df.merge(genetics_df, on='PATNO', how='outer')
        print(f"\n✓ Static data merged: {len(static_df)} patients, {len(static_df.columns)-1} features")
        
        # Load longitudinal data (visit-level)
        print("\n--- Loading Longitudinal Data ---")
        updrs_df = self.updrs_loader.load()
        
        try:
            clinical_df = self.clinical_loader.load()
        except:
            print("  ⚠️  Clinical assessments not available, continuing without them")
            clinical_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        try:
            medication_df = self.medication_loader.load()
        except:
            print("  ⚠️  Medication data not available, continuing without it")
            medication_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        try:
            age_at_visit_df = self.age_at_visit_loader.load()
        except:
            print("  ⚠️  Age at visit data not available, continuing without it")
            age_at_visit_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        # Merge longitudinal data
        longitudinal_df = updrs_df.copy()
        longitudinal_df = self._merge_longitudinal_data(longitudinal_df, clinical_df, '_clinical')
        longitudinal_df = self._merge_longitudinal_data(longitudinal_df, medication_df, '_med')
        longitudinal_df = self._merge_longitudinal_data(longitudinal_df, age_at_visit_df, '_age')
        
        print(f"\n✓ Longitudinal data merged: {len(longitudinal_df)} visits, {len(longitudinal_df.columns)-3} features")
        print(f"  Visits per patient: {len(longitudinal_df) / longitudinal_df['PATNO'].nunique():.1f} average")
        
        return {
            'static': static_df,
            'longitudinal': longitudinal_df
        }
    
    def compute_progression_slopes(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute empirical progression slopes for each patient
        
        Args:
            longitudinal_df: Longitudinal data with PATNO, months_since_baseline, and UPDRS totals
            
        Returns:
            DataFrame with PATNO and slope for each UPDRS total
        """
        print("\n--- Computing Progression Slopes ---")
        
        min_visits = self.config.training.min_visits_for_slope
        slopes_data = []
        
        for patno, group in longitudinal_df.groupby('PATNO'):
            # Filter to visits with valid time
            valid_group = group.dropna(subset=['months_since_baseline'])
            
            if len(valid_group) < min_visits:
                continue
            
            times = valid_group['months_since_baseline'].values
            slope_entry = {'PATNO': patno, 'n_visits': len(valid_group)}
            
            # Compute slope for each UPDRS total
            for total in self.config.features.all_updrs_totals:
                if total in valid_group.columns:
                    scores = valid_group[total].dropna()
                    times_for_total = valid_group.loc[scores.index, 'months_since_baseline'].values
                    
                    if len(scores) >= min_visits:
                        # Check that we have at least 2 unique time points for regression
                        # Skip if all times are identical (variance is zero)
                        times_std = pd.Series(times_for_total).std()
                        if pd.isna(times_std) or times_std == 0:
                            # Skip if all visits are at the same time point (cannot compute slope)
                            continue
                        
                        result = linregress(times_for_total, scores.values)
                        slope_entry[f'{total}_slope'] = result.slope
                        slope_entry[f'{total}_r'] = result.rvalue
                        slope_entry[f'{total}_p'] = result.pvalue
            
            if len(slope_entry) > 2:  # Has at least one slope
                slopes_data.append(slope_entry)
        
        slopes_df = pd.DataFrame(slopes_data)
        
        print(f"✓ Computed slopes for {len(slopes_df)} patients")
        for total in self.config.features.all_updrs_totals:
            slope_col = f'{total}_slope'
            if slope_col in slopes_df.columns:
                n_slopes = slopes_df[slope_col].notna().sum()
                mean_slope = slopes_df[slope_col].mean()
                std_slope = slopes_df[slope_col].std()
                print(f"  {total}: {mean_slope:.3f} ± {std_slope:.3f} points/month (n={n_slopes})")
        
        return slopes_df
    
    def create_missingness_masks(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create value and mask arrays for a set of features.
        Handles missing values by creating explicit masks.
        
        Args:
            df: DataFrame containing features
            feature_cols: List of column names to extract
            
        Returns:
            Tuple of (values, mask) arrays where mask is 1 for missing, 0 for present
        """
        values = df[feature_cols].values
        mask = np.isnan(values).astype(np.float32)
        
        # Fill NaN with 0 (the mask tells the model it's missing)
        values = np.nan_to_num(values, nan=0.0)
        
        return values, mask
    
    def create_feature_vectors(self, prepared_data: Dict = None) -> Dict:
        """
        Create feature vectors with missingness masks for model input.
        Converts DataFrame format to the structure expected by PPMILongitudinalDataset.
        
        Args:
            prepared_data: Optional dict with 'static', 'longitudinal', 'slopes' DataFrames.
                          If None, calls prepare_final_dataset() internally.
            
        Returns:
            Dict with:
                - static_data: Dict[PATNO, {'values': np.array, 'mask': np.array}]
                - longitudinal_data: Dict[PATNO, List[Dict]] - list of visits per patient
                - slopes: Dict[PATNO, float] - progression slope per patient (uses NP3TOT_slope)
        """
        # If no prepared_data provided, use prepare_final_dataset output
        if prepared_data is None:
            prepared_data = self.prepare_final_dataset()
        
        static_df = prepared_data['static']
        longitudinal_df = prepared_data['longitudinal']
        slopes_df = prepared_data['slopes']
        
        # Static Features 
        static_cols = [c for c in self.config.features.static_features if c in static_df.columns]
        
        # Create per-patient static data
        static_data = {}
        for patno in static_df['PATNO'].unique():
            patient_row = static_df[static_df['PATNO'] == patno].iloc[0]
            
            # Get feature columns (exclude PATNO)
            feature_cols = [c for c in static_cols if c != 'PATNO']
            values, mask = self.create_missingness_masks(
                pd.DataFrame([patient_row]), 
                feature_cols
            )
            
            static_data[patno] = {
                'values': values[0],  # Remove batch dimension
                'mask': mask[0]
            }
        
        # Longitudinal Features
        longitudinal_data = {}
        
        if not longitudinal_df.empty:
            # Get feature column lists from config
            motor_cols = [c for c in self.config.features.motor_features if c in longitudinal_df.columns]
            nonmotor_cols = [c for c in self.config.features.nonmotor_features if c in longitudinal_df.columns]
            med_cols = [c for c in self.config.features.medication_features if c in longitudinal_df.columns]
            
            # Group by patient
            for patno, group in longitudinal_df.groupby('PATNO'):
                # Sort by time (months_since_baseline or EVENT_ID order)
                if 'months_since_baseline' in group.columns:
                    group = group.sort_values('months_since_baseline')
                elif 'EVENT_ID' in group.columns:
                    # Sort by visit order
                    # SC and BL are both baseline (order 0), then V01-V25 are visits 1-25
                    event_order = {'SC': 0, 'BL': 0}
                    # Dynamically generate V01 through V25
                    for i in range(1, 26):
                        event_order[f'V{i:02d}'] = i
                    group['_visit_order'] = group['EVENT_ID'].map(lambda x: event_order.get(x, 999))
                    group = group.sort_values('_visit_order')
                
                visits = []
                for _, visit_row in group.iterrows():
                    visit_dict = {}
                    
                    # Motor features
                    if motor_cols:
                        motor_values, motor_mask = self.create_missingness_masks(
                            pd.DataFrame([visit_row]),
                            motor_cols
                        )
                        visit_dict['motor_values'] = motor_values[0]
                        visit_dict['motor_mask'] = motor_mask[0]
                    else:
                        visit_dict['motor_values'] = np.array([])
                        visit_dict['motor_mask'] = np.array([])
                    
                    # Non-motor features
                    if nonmotor_cols:
                        nonmotor_values, nonmotor_mask = self.create_missingness_masks(
                            pd.DataFrame([visit_row]),
                            nonmotor_cols
                        )
                        visit_dict['nonmotor_values'] = nonmotor_values[0]
                        visit_dict['nonmotor_mask'] = nonmotor_mask[0]
                    else:
                        visit_dict['nonmotor_values'] = np.array([])
                        visit_dict['nonmotor_mask'] = np.array([])
                    
                    # Medication features
                    if med_cols:
                        med_values, med_mask = self.create_missingness_masks(
                            pd.DataFrame([visit_row]),
                            med_cols
                        )
                        visit_dict['med_values'] = med_values[0]
                        visit_dict['med_mask'] = med_mask[0]
                    else:
                        visit_dict['med_values'] = np.array([])
                        visit_dict['med_mask'] = np.array([])
                    
                    # Time information
                    if 'months_since_baseline' in visit_row:
                        visit_dict['time_months'] = float(visit_row['months_since_baseline']) if pd.notna(visit_row['months_since_baseline']) else 0.0
                    else:
                        visit_dict['time_months'] = 0.0
                    
                    # UPDRS totals (targets for next-visit prediction)
                    updrs_totals = self.config.features.all_updrs_totals
                    updrs_values = []
                    for total in updrs_totals:
                        if total in visit_row:
                            val = float(visit_row[total]) if pd.notna(visit_row[total]) else np.nan
                        else:
                            val = np.nan
                        updrs_values.append(val)
                    # Array of UPDRS totals: [NP1RTOT, NP2PTOT, NP3TOT, NP4TOT] from all_updrs_totals
                    visit_dict['updrs_totals'] = np.array(updrs_values)  # [4]
                    
                    # Keep np3tot for backward compatibility (single value)
                    visit_dict['np3tot'] = updrs_values[2] if len(updrs_values) > 2 else np.nan
                    
                    visits.append(visit_dict)
                
                longitudinal_data[patno] = visits
        else:
            # Empty longitudinal data - create empty structure
            print("WARNING: No longitudinal data available")
        
        # Slopes - use NP3TOT_slope if available, otherwise -999
        slopes = {}
        if not slopes_df.empty:
            for _, row in slopes_df.iterrows():
                patno = row['PATNO']
                # Use NP3TOT_slope (the main target for the model)
                if 'NP3TOT_slope' in row and pd.notna(row['NP3TOT_slope']):
                    slope_val = row['NP3TOT_slope']
                else:
                    slope_val = -999
                slopes[patno] = float(slope_val)
        else:
            # If no slopes computed, mark all as unavailable (-999)
            all_patnos = list(static_data.keys())
            slopes = {patno: -999.0 for patno in all_patnos}
        
        return {
            'static_data': static_data,
            'longitudinal_data': longitudinal_data,
            'slopes': slopes
        }
    
    def prepare_final_dataset(self) -> Dict:
        """
        Main integration pipeline
        
        Returns:
            Dictionary with 'static', 'longitudinal', 'slopes', and metadata
        """
        print("\n" + "=" * 80)
        print("DATA INTEGRATION PIPELINE")
        print("=" * 80)
        
        # # Load all data
        data = self.load_all_data()
        
        # Filter to only patients with both static and longitudinal data
        patients_with_both = set(data['static']['PATNO']) & set(data['longitudinal']['PATNO'])
        print(f"\n--- Filtering to Complete Cases ---")
        print(f"  Patients with static data: {len(data['static'])}")
        print(f"  Patients with longitudinal data: {data['longitudinal']['PATNO'].nunique()}")
        print(f"  Patients with both: {len(patients_with_both)}")
        
        data['static'] = data['static'][data['static']['PATNO'].isin(patients_with_both)]
        data['longitudinal'] = data['longitudinal'][data['longitudinal']['PATNO'].isin(patients_with_both)]

        # Compute slopes
        slopes_df = self.compute_progression_slopes(data['longitudinal'])
        data['slopes'] = slopes_df
        
        # Create metadata
        data['metadata'] = {
            'n_patients': len(data['static']),
            'n_visits': len(data['longitudinal']),
            'n_with_slopes': len(slopes_df),
            'static_features': [c for c in data['static'].columns if c != 'PATNO'],
            'longitudinal_features': [c for c in data['longitudinal'].columns 
                                     if c not in ['PATNO', 'EVENT_ID', 'INFODT', 'visit_date']],
            'updrs_totals': [c for c in data['longitudinal'].columns if c in self.config.features.all_updrs_totals]
        }
        
        print("\n" + "=" * 80)
        print("DATA INTEGRATION COMPLETE")
        print("=" * 80)
        print(f"\nFinal dataset:")
        print(f"  Patients: {data['metadata']['n_patients']}")
        print(f"  Total visits: {data['metadata']['n_visits']}")
        print(f"  Visits per patient: {data['metadata']['n_visits'] / data['metadata']['n_patients']:.1f}")
        print(f"  Patients with slopes: {data['metadata']['n_with_slopes']}")
        print(f"  Static features: {len(data['metadata']['static_features'])}")
        print(f"  Longitudinal features: {len(data['metadata']['longitudinal_features'])}")
        print(f"  UPDRS totals available: {', '.join(data['metadata']['updrs_totals'])}")
        
        return data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Ensure parent directory is in path (already done above, but keep for clarity)
    v1_dir = os.path.dirname(parent_dir)  # Go up one more level from data/ to V1_implementation/
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
    from training.config import get_default_config
    
    print("=" * 80)
    print("DATA INTEGRATOR - Final Integration")
    print("=" * 80)
    
    config = get_default_config()
    
    # Create integrator
    integrator = DataIntegrator(config)
    
    try:
        # Run pipeline prepare data
        final_data = integrator.prepare_final_dataset()
        
        # Save to CSV for inspection
        print("\n--- Saving Data ---")
        output_dir = config.data.processed_data_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        final_data['static'].to_csv(f"{output_dir}/static_data.csv", index=False)
        final_data['longitudinal'].to_csv(f"{output_dir}/longitudinal_data.csv", index=False)
        final_data['slopes'].to_csv(f"{output_dir}/slopes_data.csv", index=False)
        
        print(f"  ✓ Saved to {output_dir}/")
        print(f"    - static_data.csv")
        print(f"    - longitudinal_data.csv")
        print(f"    - slopes_data.csv")
        
        print("\n✓ DATA INTEGRATION SUCCESSFUL!")
        print("\nNext steps:")
        print("  1. Inspect the saved CSV files")
        print("  2. Check for missing data patterns")
        print("  3. Proceed to dataset.py to create PyTorch DataLoaders")
        
    except Exception as e:
        print(f"\n❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Make sure all individual loaders work (test each separately)")
        print("  2. Check that file paths in config.py are correct")
        print("  3. Verify data format matches expected structure")
