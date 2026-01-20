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
    from data.visit_index_builder import VisitIndexBuilder
    from data.base_loader import filter_valid_participants
except ImportError:
    # Fall back to direct imports if running from data/ directory
    from loaders.genetics_loader import GeneticsLoader
    from loaders.demographics_loader import DemographicsLoader
    from loaders.updrs_loader import UPDRSLoader
    from loaders.clinical_loader import ClinicalAssessmentsLoader
    from loaders.medication_loader import MedicationLoader
    from loaders.age_at_visit_loader import AgeAtVisitLoader
    from visit_index_builder import VisitIndexBuilder
    from base_loader import filter_valid_participants


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
        
        Uses shared utility function filter_valid_participants() to ensure
        consistent filtering logic with BaseDataLoader.
        
        Returns:
            DataFrame with filtered valid participants (all columns)
        """
        import os
        
        def resolve_path(file_path):
            """Resolve file path, handling relative paths that start with ../"""
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        # Resolve participant status path
        status_path = resolve_path(config.data.participant_status)
        
        full_df = pd.read_csv(status_path)
        # Use shared utility function from base_loader
        df = filter_valid_participants(full_df)
        
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
        Converts non-numeric columns to numeric where possible.
        
        Args:
            df: DataFrame containing features
            feature_cols: List of column names to extract
            
        Returns:
            Tuple of (values, mask) arrays where mask is 1 for missing, 0 for present
        """
        # Extract only available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        if not available_cols:
            # Return empty arrays if no columns available
            return np.array([]).reshape(len(df), 0).astype(np.float32), \
                   np.array([]).reshape(len(df), 0).astype(np.float32)
        
        # Extract subset of DataFrame with only the features we need
        feature_df = df[available_cols].copy()
        
        # Convert all columns to numeric, coercing errors to NaN
        # This handles string columns, categoricals, etc.
        for col in available_cols:
            if feature_df[col].dtype == 'object':
                # Try to convert to numeric
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            elif feature_df[col].dtype.name == 'category':
                # Convert categorical to numeric codes
                feature_df[col] = pd.to_numeric(feature_df[col].cat.codes, errors='coerce')
        
        # Now extract as numeric array
        values = feature_df.values.astype(np.float32)
        
        # Create mask for missing values (handles NaN, None, etc.)
        # Use pandas isna() which works for all dtypes, then convert to numpy
        mask = feature_df.isna().values.astype(np.float32)
        
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
                - slopes: Dict[PATNO, Dict[str, float]] - progression slopes per patient
                         Keys: 'NP1RTOT_slope', 'NP2PTOT_slope', 'NP3TOT_slope', 'NP4TOT_slope'
                         Values: slope value or -999 if not available
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
            # Use getattr with defaults so tests can provide partial mock configs
            motor_features = getattr(self.config.features, 'motor_features', [])
            updrs_supplementary_features = getattr(self.config.features, 'updrs_supplementary_features', [])
            clinical_features = getattr(self.config.features, 'clinical_features', [])
            medication_features = getattr(self.config.features, 'medication_features', [])

            # If a Mock() provides attributes but they aren't iterable lists, treat as empty.
            if not isinstance(motor_features, (list, tuple, set)):
                motor_features = []
            if not isinstance(updrs_supplementary_features, (list, tuple, set)):
                updrs_supplementary_features = []
            if not isinstance(clinical_features, (list, tuple, set)):
                clinical_features = []
            if not isinstance(medication_features, (list, tuple, set)):
                medication_features = []

            motor_cols = [c for c in motor_features if c in longitudinal_df.columns]
            updrs_supplementary_cols = [c for c in updrs_supplementary_features if c in longitudinal_df.columns]
            clinical_cols = [c for c in clinical_features if c in longitudinal_df.columns]
            med_cols = [c for c in medication_features if c in longitudinal_df.columns]
            
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
                    
                    # UPDRS supplementary features
                    if updrs_supplementary_cols:
                        updrs_supplementary_values, updrs_supplementary_mask = self.create_missingness_masks(
                            pd.DataFrame([visit_row]),
                            updrs_supplementary_cols
                        )
                        visit_dict['updrs_supplementary_values'] = updrs_supplementary_values[0]
                        visit_dict['updrs_supplementary_mask'] = updrs_supplementary_mask[0]
                    else:
                        visit_dict['updrs_supplementary_values'] = np.array([])
                        visit_dict['updrs_supplementary_mask'] = np.array([])
                    
                    # Clinical features
                    if clinical_cols:
                        clinical_values, clinical_mask = self.create_missingness_masks(
                            pd.DataFrame([visit_row]),
                            clinical_cols
                        )
                        visit_dict['clinical_values'] = clinical_values[0]
                        visit_dict['clinical_mask'] = clinical_mask[0]
                    else:
                        visit_dict['clinical_values'] = np.array([])
                        visit_dict['clinical_mask'] = np.array([])
                    
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
        
        # Slopes - extract all UPDRS total slopes for each patient
        slopes = {}
        all_patnos = list(static_data.keys())
        
        # Initialize all patients with -999 for all slopes
        for patno in all_patnos:
            slopes[patno] = {
                f'{total}_slope': -999.0
                for total in self.config.features.all_updrs_totals
            }
        
        # Fill in computed slopes from slopes_df
        if not slopes_df.empty:
            for _, row in slopes_df.iterrows():
                patno = row['PATNO']
                if patno not in slopes:
                    continue
                
                # Extract slope for each UPDRS total
                for total in self.config.features.all_updrs_totals:
                    slope_col = f'{total}_slope'
                    if slope_col in row and pd.notna(row[slope_col]):
                        slopes[patno][slope_col] = float(row[slope_col])
                    # else keep -999 (already initialized)
        
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
    
    def create_feature_vectors_with_visit_index(self, prepared_data: Dict = None, min_visits: int = 2) -> Dict:
        """
        Create feature vectors using the VisitIndexBuilder for proper timeline management.
        
        This method uses a master visit_index to:
        1. Define the timeline independently of any modality
        2. Align each modality to the timeline
        3. Create proper input missingness masks (Mask A)
        4. Create proper label availability masks (Mask B)
        
        Args:
            prepared_data: Optional dict with raw DataFrames. If None, loads data.
            min_visits: Minimum number of visits required per patient
            
        Returns:
            Dict with:
                - 'visit_index': Master timeline DataFrame
                - 'static_data': Dict[PATNO, {'values': np.array, 'mask': np.array}]
                - 'longitudinal_data': Dict[PATNO, List[Dict]] - aligned to visit_index
                - 'slopes': Dict[PATNO, Dict[str, float]]
        """
        print("\n" + "=" * 80)
        print("CREATING FEATURE VECTORS WITH VISIT INDEX")
        print("=" * 80)
        
        # Load raw data if not provided
        if prepared_data is None:
            # Load raw DataFrames from loaders
            print("\n--- Loading Raw Data ---")
            genetics_df = self.genetics_loader.load()
            demographics_df = self.demographics_loader.load()
            updrs_df = self.updrs_loader.load()
            
            try:
                clinical_df = self.clinical_loader.load()
            except Exception as e:
                print(f"  ⚠️ Clinical data not available: {e}")
                clinical_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
            
            try:
                medication_df = self.medication_loader.load()
            except Exception as e:
                print(f"  ⚠️ Medication data not available: {e}")
                medication_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
            
            try:
                age_at_visit_df = self.age_at_visit_loader.load()
            except Exception as e:
                print(f"  ⚠️ Age at visit data not available: {e}")
                age_at_visit_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
            
            # Static data
            static_df = demographics_df.merge(genetics_df, on='PATNO', how='outer')
        else:
            # Use prepared data
            static_df = prepared_data.get('static', pd.DataFrame())
            updrs_df = prepared_data.get('longitudinal', pd.DataFrame())
            clinical_df = prepared_data.get('clinical', pd.DataFrame(columns=['PATNO', 'EVENT_ID']))
            medication_df = prepared_data.get('medication', pd.DataFrame(columns=['PATNO', 'EVENT_ID']))
            age_at_visit_df = prepared_data.get('age_at_visit', pd.DataFrame(columns=['PATNO', 'EVENT_ID']))
        
        # Build visit_index from all longitudinal sources
        print("\n--- Building Visit Index (Master Timeline) ---")
        visit_index_builder = VisitIndexBuilder(self.config)
        
        visit_index = visit_index_builder.build_from_sources(
            source_dfs=[updrs_df, clinical_df, medication_df, age_at_visit_df],
            valid_patnos=self.valid_participants['PATNO'].tolist()
        )
        
        # Filter to patients with minimum visits
        visit_index = visit_index_builder.filter_by_min_visits(visit_index, min_visits)
        
        # Get valid patient IDs from visit_index
        valid_patnos = visit_index['PATNO'].unique()
        
        # Filter static data to match
        static_df = static_df[static_df['PATNO'].isin(valid_patnos)]
        
        # Create static feature vectors
        print("\n--- Creating Static Feature Vectors ---")
        static_cols = [c for c in self.config.features.static_features if c in static_df.columns]
        static_data = {}
        
        for patno in valid_patnos:
            patient_row = static_df[static_df['PATNO'] == patno]
            if len(patient_row) == 0:
                # Patient has no static data - create all-missing
                static_data[patno] = {
                    'values': np.zeros(len(static_cols), dtype=np.float32),
                    'mask': np.ones(len(static_cols), dtype=np.float32)
                }
            else:
                feature_cols = [c for c in static_cols if c != 'PATNO']
                values, mask = self.create_missingness_masks(patient_row, feature_cols)
                static_data[patno] = {
                    'values': values[0],
                    'mask': mask[0]
                }
        
        print(f"  ✓ Static features: {len(static_cols)} features for {len(static_data)} patients")
        
        # Align modalities to visit_index
        print("\n--- Aligning Modalities to Visit Index ---")
        
        motor_cols = [c for c in self.config.features.motor_features if c in updrs_df.columns]
        updrs_supplementary_cols = [c for c in self.config.features.updrs_supplementary_features 
                                   if c in updrs_df.columns]
        clinical_cols = [c for c in self.config.features.clinical_features 
                        if c in clinical_df.columns]
        med_cols = [c for c in self.config.features.medication_features if c in medication_df.columns]
        updrs_total_cols = self.config.features.all_updrs_totals
        
        # Create per-patient longitudinal data
        longitudinal_data = {}
        
        for patno in valid_patnos:
            patient_visits = visit_index[visit_index['PATNO'] == patno].sort_values('visit_order')
            visits = []
            
            for _, visit_row in patient_visits.iterrows():
                event_id = visit_row['EVENT_ID']
                visit_dict = {}
                
                # Motor features from UPDRS
                motor_visit = updrs_df[(updrs_df['PATNO'] == patno) & (updrs_df['EVENT_ID'] == event_id)]
                if len(motor_visit) > 0 and motor_cols:
                    motor_values, motor_mask = self.create_missingness_masks(motor_visit, motor_cols)
                    visit_dict['motor_values'] = motor_values[0]
                    visit_dict['motor_mask'] = motor_mask[0]
                else:
                    visit_dict['motor_values'] = np.zeros(len(motor_cols) if motor_cols else 0, dtype=np.float32)
                    visit_dict['motor_mask'] = np.ones(len(motor_cols) if motor_cols else 0, dtype=np.float32)
                
                # UPDRS supplementary features from UPDRS
                updrs_supplementary_visit = updrs_df[(updrs_df['PATNO'] == patno) & (updrs_df['EVENT_ID'] == event_id)]
                if len(updrs_supplementary_visit) > 0 and updrs_supplementary_cols:
                    updrs_supplementary_values, updrs_supplementary_mask = self.create_missingness_masks(updrs_supplementary_visit, updrs_supplementary_cols)
                    visit_dict['updrs_supplementary_values'] = updrs_supplementary_values[0]
                    visit_dict['updrs_supplementary_mask'] = updrs_supplementary_mask[0]
                else:
                    visit_dict['updrs_supplementary_values'] = np.zeros(len(updrs_supplementary_cols) if updrs_supplementary_cols else 0, dtype=np.float32)
                    visit_dict['updrs_supplementary_mask'] = np.ones(len(updrs_supplementary_cols) if updrs_supplementary_cols else 0, dtype=np.float32)
                
                # Clinical features from clinical assessments
                clinical_visit = clinical_df[(clinical_df['PATNO'] == patno) & (clinical_df['EVENT_ID'] == event_id)] if len(clinical_df) > 0 else pd.DataFrame()
                if len(clinical_visit) > 0 and clinical_cols:
                    clinical_values, clinical_mask = self.create_missingness_masks(clinical_visit, clinical_cols)
                    visit_dict['clinical_values'] = clinical_values[0]
                    visit_dict['clinical_mask'] = clinical_mask[0]
                else:
                    visit_dict['clinical_values'] = np.zeros(len(clinical_cols) if clinical_cols else 0, dtype=np.float32)
                    visit_dict['clinical_mask'] = np.ones(len(clinical_cols) if clinical_cols else 0, dtype=np.float32)
                
                # Medication features
                med_visit = medication_df[(medication_df['PATNO'] == patno) & (medication_df['EVENT_ID'] == event_id)] if len(medication_df) > 0 else pd.DataFrame()
                if len(med_visit) > 0 and med_cols:
                    med_values, med_mask = self.create_missingness_masks(med_visit, med_cols)
                    visit_dict['med_values'] = med_values[0]
                    visit_dict['med_mask'] = med_mask[0]
                else:
                    visit_dict['med_values'] = np.zeros(len(med_cols) if med_cols else 0, dtype=np.float32)
                    visit_dict['med_mask'] = np.ones(len(med_cols) if med_cols else 0, dtype=np.float32)
                
                # Time features from visit_index
                visit_dict['time_months'] = float(visit_row['months_since_baseline']) if pd.notna(visit_row['months_since_baseline']) else 0.0
                visit_dict['delta_months'] = float(visit_row['delta_months']) if pd.notna(visit_row['delta_months']) else 0.0
                visit_dict['visit_order'] = int(visit_row['visit_order'])
                
                # UPDRS totals (targets)
                updrs_visit = updrs_df[(updrs_df['PATNO'] == patno) & (updrs_df['EVENT_ID'] == event_id)]
                updrs_values = []
                for total in updrs_total_cols:
                    if len(updrs_visit) > 0 and total in updrs_visit.columns:
                        val = updrs_visit[total].iloc[0]
                        val = float(val) if pd.notna(val) else np.nan
                    else:
                        val = np.nan
                    updrs_values.append(val)
                visit_dict['updrs_totals'] = np.array(updrs_values, dtype=np.float32)
                visit_dict['np3tot'] = updrs_values[2] if len(updrs_values) > 2 else np.nan  # Backward compatibility
                
                visits.append(visit_dict)
            
            longitudinal_data[patno] = visits
        
        print(f"  ✓ Longitudinal data: {sum(len(v) for v in longitudinal_data.values())} visits")
        print(f"    Motor features: {len(motor_cols)}")
        print(f"    UPDRS supplementary features: {len(updrs_supplementary_cols)}")
        print(f"    Clinical features: {len(clinical_cols)}")
        print(f"    Medication features: {len(med_cols)}")
        
        # Compute slopes
        print("\n--- Computing Progression Slopes ---")
        if not updrs_df.empty:
            slopes_df = self.compute_progression_slopes(updrs_df)
        else:
            slopes_df = pd.DataFrame()
        
        # Create slopes dict
        slopes = {}
        for patno in valid_patnos:
            slopes[patno] = {
                f'{total}_slope': -999.0 
                for total in self.config.features.all_updrs_totals
            }
        
        if not slopes_df.empty:
            for _, row in slopes_df.iterrows():
                patno = row['PATNO']
                if patno not in slopes:
                    continue
                for total in self.config.features.all_updrs_totals:
                    slope_col = f'{total}_slope'
                    if slope_col in row and pd.notna(row[slope_col]):
                        slopes[patno][slope_col] = float(row[slope_col])
        
        print("\n" + "=" * 80)
        print("FEATURE VECTORS CREATED SUCCESSFULLY")
        print("=" * 80)
        print(f"  Patients: {len(valid_patnos)}")
        print(f"  Total visits: {len(visit_index)}")
        print(f"  Visits per patient: {len(visit_index) / len(valid_patnos):.1f} average")
        
        return {
            'visit_index': visit_index,
            'static_data': static_data,
            'longitudinal_data': longitudinal_data,
            'slopes': slopes
        }