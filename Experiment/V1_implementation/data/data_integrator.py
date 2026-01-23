"""
Data Integrator - Combines all loaders into final dataset
Run this AFTER all individual loaders are working
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
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
    from data.loaders.non_motor_loader import NonMotorAssessmentsLoader
    from data.loaders.medication_loader import MedicationLoader
    from data.loaders.age_at_visit_loader import AgeAtVisitLoader
    from data.visit_index_builder import VisitIndexBuilder
    from data.base_loader import filter_valid_participants
    from data.utils import FeatureScaler, resolve_data_path
    from data.feature_engineer import FeatureEngineer
    from data.label_engineer import LabelEngineer
except ImportError:
    # Fall back to direct imports if running from data/ directory
    from loaders.genetics_loader import GeneticsLoader
    from loaders.demographics_loader import DemographicsLoader
    from loaders.updrs_loader import UPDRSLoader
    from loaders.non_motor_loader import NonMotorAssessmentsLoader
    from loaders.medication_loader import MedicationLoader
    from loaders.age_at_visit_loader import AgeAtVisitLoader
    from visit_index_builder import VisitIndexBuilder
    from base_loader import filter_valid_participants
    from utils import FeatureScaler, resolve_data_path
    from feature_engineer import FeatureEngineer
    from label_engineer import LabelEngineer


# Constants
_VISIT_EVENT_ORDER = {
    'SC': 0,
    'BL': 0,
    **{f'V{i:02d}': i for i in range(1, 26)}
}
_DEFAULT_VISIT_ORDER = 999  # For unknown event IDs


class DataIntegrator:
    """
    Integrates data from all loaders into final dataset
    Handles merging, missing data, slope computation
    """
    
    def __init__(self, config, normalize_features: bool = True):
        """
        Args:
            config: Configuration dict
            normalize_features: Whether to normalize features (default: True)
        
        Note: Patient filtering (min_visits) should be done via VisitIndexBuilder,
              not in individual loaders. Loaders return all valid participants.
        """
        self.config = config
        self.base_dir = config.data.base_dir
        self.normalize_features = normalize_features

        self.feature_engineer = FeatureEngineer(config, normalize_features=normalize_features)
        self.label_engineer = LabelEngineer(config)

        self.static_scaler = self.feature_engineer.static_scaler
        self.motor_scaler = self.feature_engineer.motor_scaler
        self.updrs_supplementary_scaler = self.feature_engineer.updrs_supplementary_scaler
        self.non_motor_scaler = self.feature_engineer.non_motor_scaler
        self.medication_scaler = self.feature_engineer.medication_scaler
        self.age_at_visit_scaler = self.feature_engineer.age_at_visit_scaler
        
        print("Loading participant status (shared across all loaders)...")
        self.valid_participants = self._load_valid_participants(config)
        print(f"  ✓ Loaded {len(self.valid_participants)} valid participants")
        
        print("\nInitializing loaders...")
        self.genetics_loader = GeneticsLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.demographics_loader = DemographicsLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.updrs_loader = UPDRSLoader(self.base_dir, config, valid_participants=self.valid_participants)
        self.non_motor_loader = NonMotorAssessmentsLoader(self.base_dir, config, valid_participants=self.valid_participants)
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
        status_path = resolve_data_path(self.base_dir, config.data.participant_status)
        full_df = pd.read_csv(status_path)
        return filter_valid_participants(full_df)
    
    def _load_with_fallback(self, loader, loader_name: str) -> pd.DataFrame:
        """
        Load data with graceful fallback to empty DataFrame.
        
        Args:
            loader: Loader instance with .load() method
            loader_name: Human-readable name for error messages
            
        Returns:
            DataFrame with loaded data or empty DataFrame with PATNO/EVENT_ID columns
        """
        try:
            return loader.load()
        except Exception as e:
            print(f"  ⚠️  {loader_name} not available: {e}")
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
    
    def _extract_modality_features(
        self,
        df: pd.DataFrame,
        patno: int,
        event_id: str,
        feature_cols: List[str],
        scaler: FeatureScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for a modality at a visit.
        
        Args:
            df: Source DataFrame
            patno: Patient number
            event_id: Event ID
            feature_cols: List of feature column names
            scaler: FeatureScaler instance (can be None)
            
        Returns:
            Tuple of (values, mask) arrays
        """
        visit_data = df[(df['PATNO'] == patno) & (df['EVENT_ID'] == event_id)]
        
        if len(visit_data) > 0 and feature_cols:
            values, mask = self.create_missingness_masks(visit_data, feature_cols, scaler=scaler)
            return values[0], mask[0]
        else:
            n_features = len(feature_cols) if feature_cols else 0
            return (
                np.zeros(n_features, dtype=np.float32),
                np.ones(n_features, dtype=np.float32)
            )
    
    def _fit_scaler_for_features(
        self,
        feature_attr_name: str,
        df: pd.DataFrame,
        scaler: FeatureScaler,
        scaler_name: str
    ) -> None:
        """
        Fit a scaler for a given feature set.
        
        Args:
            feature_attr_name: Attribute name in config.features (e.g., 'motor_features')
            df: DataFrame containing the features
            scaler: FeatureScaler instance (can be None)
            scaler_name: Human-readable name for logging
        """
        if scaler is None:
            return
            
        features = getattr(self.config.features, feature_attr_name, [])
        if not isinstance(features, (list, tuple, set)) or not features:
            return
        
        feature_cols = [c for c in features if c in df.columns]
        if feature_cols:
            values, mask = self.create_missingness_masks(df, feature_cols, scaler=None)
            scaler.fit_transform(values, mask, feature_cols)
            print(f"  ✓ Fitted {scaler_name} scaler: {len(feature_cols)} features")
    
    def _initialize_slopes_dict(self, patnos: List[int]) -> Dict[int, Dict[str, float]]:
        self._ensure_scalers_exist()
        return self.label_engineer.initialize_slope_dict(patnos)
    
    def _fill_slopes_from_df(self, slopes: Dict[int, Dict[str, float]], slopes_df: pd.DataFrame) -> None:
        self._ensure_scalers_exist()
        self.label_engineer.fill_slope_dict_from_df(slopes, slopes_df)
    
    def _load_and_merge_static_data(self) -> pd.DataFrame:
        """Load and merge static (patient-level) data from genetics and demographics."""
        print("\n--- Loading Static Data ---")
        genetics_df = self.genetics_loader.load()
        demographics_df = self.demographics_loader.load()
        static_df = demographics_df.merge(genetics_df, on='PATNO', how='outer')
        print(f"\n✓ Static data merged: {len(static_df)} patients, {len(static_df.columns)-1} features")
        return static_df
    
    def _load_and_merge_longitudinal_data(self) -> pd.DataFrame:
        """Load and merge all longitudinal (visit-level) data."""
        print("\n--- Loading Longitudinal Data ---")
        updrs_df = self.updrs_loader.load()
        
        non_motor_df = self._load_with_fallback(self.non_motor_loader, "Non-motor assessments")
        medication_df = self._load_with_fallback(self.medication_loader, "Medication data")
        age_at_visit_df = self._load_with_fallback(self.age_at_visit_loader, "Age at visit data")
        
        longitudinal_df = updrs_df.copy()
        longitudinal_df = self._merge_longitudinal_data(longitudinal_df, non_motor_df, '_non_motor')
        longitudinal_df = self._merge_longitudinal_data(longitudinal_df, medication_df, '_med')
        longitudinal_df = self._merge_longitudinal_data(longitudinal_df, age_at_visit_df, '_age')
        
        return longitudinal_df
    
    def _ensure_scalers_exist(self):
        if not hasattr(self, "feature_engineer") or self.feature_engineer is None:
            normalize_features = bool(getattr(self, "normalize_features", True))
            self.feature_engineer = FeatureEngineer(self.config, normalize_features=normalize_features)

        if not hasattr(self, "label_engineer") or self.label_engineer is None:
            self.label_engineer = LabelEngineer(self.config)

        self.static_scaler = getattr(self.feature_engineer, "static_scaler", None)
        self.motor_scaler = getattr(self.feature_engineer, "motor_scaler", None)
        self.updrs_supplementary_scaler = getattr(self.feature_engineer, "updrs_supplementary_scaler", None)
        self.non_motor_scaler = getattr(self.feature_engineer, "non_motor_scaler", None)
        self.medication_scaler = getattr(self.feature_engineer, "medication_scaler", None)
        self.age_at_visit_scaler = getattr(self.feature_engineer, "age_at_visit_scaler", None)
    
    def _process_static_features_vectorized(self, static_df: pd.DataFrame) -> Dict[int, Dict]:
        self._ensure_scalers_exist()
        return self.feature_engineer._process_static_features_vectorized(static_df)
    
    def _get_feature_columns_from_config(
        self, 
        df: pd.DataFrame, 
        feature_attr_names: List[str]
    ) -> Dict[str, List[str]]:
        self._ensure_scalers_exist()
        return self.feature_engineer._get_feature_columns_from_config(df, feature_attr_names)
    
    def _extract_all_visit_features(
        self,
        group: pd.DataFrame,
        feature_cols: List[str],
        scaler: FeatureScaler
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_scalers_exist()
        return self.feature_engineer._extract_all_visit_features(group, feature_cols, scaler)
    
    def _sort_visits_by_time(self, group: pd.DataFrame) -> pd.DataFrame:
        self._ensure_scalers_exist()
        return self.feature_engineer._sort_visits_by_time(group)
    
    def _process_longitudinal_features_vectorized(self, longitudinal_df: pd.DataFrame) -> Dict[int, List[Dict]]:
        self._ensure_scalers_exist()
        return self.feature_engineer._process_longitudinal_features_vectorized(longitudinal_df)
    
    def _filter_to_complete_cases(self, data: Dict) -> Dict:
        """Filter to only patients with both static and longitudinal data."""
        patients_with_both = set(data['static']['PATNO']) & set(data['longitudinal']['PATNO'])
        print(f"\n--- Filtering to Complete Cases ---")
        print(f"  Patients with static data: {len(data['static'])}")
        print(f"  Patients with longitudinal data: {data['longitudinal']['PATNO'].nunique()}")
        print(f"  Patients with both: {len(patients_with_both)}")
        
        data['static'] = data['static'][data['static']['PATNO'].isin(patients_with_both)]
        data['longitudinal'] = data['longitudinal'][data['longitudinal']['PATNO'].isin(patients_with_both)]
        return data
    
    def _create_metadata(self, data: Dict, slopes_df: pd.DataFrame) -> Dict:
        """Create metadata dictionary for the integrated dataset."""
        return {
            'n_patients': len(data['static']),
            'n_visits': len(data['longitudinal']),
            'n_with_slopes': len(slopes_df),
            'static_features': [c for c in data['static'].columns if c != 'PATNO'],
            'longitudinal_features': [c for c in data['longitudinal'].columns 
                                     if c not in ['PATNO', 'EVENT_ID', 'INFODT', 'visit_date']],
            'updrs_totals': [c for c in data['longitudinal'].columns 
                            if c in self.config.features.all_updrs_totals]
        }
    
    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw DataFrames from all loaders."""
        print("\n--- Loading Raw Data ---")
        genetics_df = self.genetics_loader.load()
        demographics_df = self.demographics_loader.load()
        updrs_df = self.updrs_loader.load()
        
        non_motor_df = self._load_with_fallback(self.non_motor_loader, "Non-motor data")
        medication_df = self._load_with_fallback(self.medication_loader, "Medication data")
        age_at_visit_df = self._load_with_fallback(self.age_at_visit_loader, "Age at visit data")
        
        static_df = demographics_df.merge(genetics_df, on='PATNO', how='outer')
        return static_df, updrs_df, non_motor_df, medication_df, age_at_visit_df
    
    def _build_visit_index(
        self, 
        source_dfs: List[pd.DataFrame], 
        min_visits: int
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build visit index from all longitudinal sources and filter by minimum visits."""
        print("\n--- Building Visit Index (Master Timeline) ---")
        visit_index_builder = VisitIndexBuilder(self.config)
        
        visit_index = visit_index_builder.build_from_sources(
            source_dfs=source_dfs,
            valid_patnos=self.valid_participants['PATNO'].tolist()
        )
        visit_index = visit_index_builder.filter_by_min_visits(visit_index, min_visits)
        valid_patnos = visit_index['PATNO'].unique()
        
        return visit_index, valid_patnos
    
    def _create_static_feature_vectors(
        self, 
        static_df: pd.DataFrame, 
        valid_patnos: np.ndarray
    ) -> Dict[int, Dict]:
        """Create static feature vectors for valid patients."""
        print("\n--- Creating Static Feature Vectors ---")
        static_cols = [c for c in self.config.features.static_features if c in static_df.columns]
        static_data = {}
        
        for patno in valid_patnos:
            patient_row = static_df[static_df['PATNO'] == patno]
            if len(patient_row) == 0:
                static_data[patno] = {
                    'values': np.zeros(len(static_cols), dtype=np.float32),
                    'mask': np.ones(len(static_cols), dtype=np.float32)
                }
            else:
                feature_cols = [c for c in static_cols if c != 'PATNO']
                values, mask = self.create_missingness_masks(patient_row, feature_cols, scaler=self.static_scaler)
                static_data[patno] = {
                    'values': values[0],
                    'mask': mask[0]
                }
        
        print(f"  ✓ Static features: {len(static_cols)} features for {len(static_data)} patients")
        return static_data
    
    def _align_modalities_to_visit_index(
        self,
        visit_index: pd.DataFrame,
        valid_patnos: np.ndarray,
        updrs_df: pd.DataFrame,
        non_motor_df: pd.DataFrame,
        medication_df: pd.DataFrame,
        age_at_visit_df: pd.DataFrame,
        motor_cols: List[str],
        updrs_supplementary_cols: List[str],
        non_motor_cols: List[str],
        med_cols: List[str],
        age_at_visit_cols: List[str]
    ) -> Dict[int, List[Dict]]:
        """Align all modalities to the visit index timeline."""
        longitudinal_data = {}
        
        for patno in valid_patnos:
            patient_visits = visit_index[visit_index['PATNO'] == patno].sort_values('visit_order')
            visits = []
            
            for _, visit_row in patient_visits.iterrows():
                event_id = visit_row['EVENT_ID']
                visit_dict = {}
                
                motor_values, motor_mask = self._extract_modality_features(
                    updrs_df, patno, event_id, motor_cols, self.motor_scaler
                )
                visit_dict['motor_values'] = motor_values
                visit_dict['motor_mask'] = motor_mask
                
                updrs_supplementary_values, updrs_supplementary_mask = self._extract_modality_features(
                    updrs_df, patno, event_id, updrs_supplementary_cols, self.updrs_supplementary_scaler
                )
                visit_dict['updrs_supplementary_values'] = updrs_supplementary_values
                visit_dict['updrs_supplementary_mask'] = updrs_supplementary_mask
                
                non_motor_values, non_motor_mask = self._extract_modality_features(
                    non_motor_df, patno, event_id, non_motor_cols, self.non_motor_scaler
                )
                visit_dict['nonmotor_values'] = non_motor_values
                visit_dict['nonmotor_mask'] = non_motor_mask
                
                med_values, med_mask = self._extract_modality_features(
                    medication_df, patno, event_id, med_cols, self.medication_scaler
                )
                visit_dict['med_values'] = med_values
                visit_dict['med_mask'] = med_mask
                
                age_at_visit_values, age_at_visit_mask = self._extract_modality_features(
                    age_at_visit_df, patno, event_id, age_at_visit_cols, self.age_at_visit_scaler
                )
                visit_dict['age_at_visit_values'] = age_at_visit_values
                visit_dict['age_at_visit_mask'] = age_at_visit_mask
                
                metadata = self._extract_visit_metadata(visit_row, updrs_df, patno, event_id)
                visit_dict.update(metadata)
                visits.append(visit_dict)
            
            longitudinal_data[patno] = visits
        
        print(f"  ✓ Longitudinal data: {sum(len(v) for v in longitudinal_data.values())} visits")
        print(f"    Motor features: {len(motor_cols)}")
        print(f"    UPDRS supplementary features: {len(updrs_supplementary_cols)}")
        print(f"    Non-motor features: {len(non_motor_cols)}")
        print(f"    Medication features: {len(med_cols)}")
        
        return longitudinal_data
    
    def _extract_visit_metadata(
        self,
        visit_row: pd.Series,
        updrs_df: pd.DataFrame,
        patno: int,
        event_id: str
    ) -> Dict[str, Union[float, int, np.ndarray]]:
        """
        Extract time and UPDRS totals for a visit.
        
        Args:
            visit_row: Row from visit_index
            updrs_df: DataFrame with UPDRS totals
            patno: Patient number
            event_id: Event ID
            
        Returns:
            Dictionary with time_months, delta_months, visit_order, updrs_totals, np3tot
        """
        metadata = {
            'time_months': float(visit_row.get('months_since_baseline', 0.0) or 0.0),
            'delta_months': float(visit_row.get('delta_months', 0.0) or 0.0),
            'visit_order': int(visit_row.get('visit_order', 0))
        }
        
        updrs_visit = updrs_df[(updrs_df['PATNO'] == patno) & (updrs_df['EVENT_ID'] == event_id)]
        updrs_values = []
        for total in self.config.features.all_updrs_totals:
            if len(updrs_visit) > 0 and total in updrs_visit.columns:
                val = updrs_visit[total].iloc[0]
                val = float(val) if pd.notna(val) else np.nan
            else:
                val = np.nan
            updrs_values.append(val)
        
        metadata['updrs_totals'] = np.array(updrs_values, dtype=np.float32)
        metadata['np3tot'] = updrs_values[2] if len(updrs_values) > 2 else np.nan
        
        return metadata
    
    def _merge_longitudinal_data(self, longitudinal_df: pd.DataFrame, new_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """
        Helper method to merge additional longitudinal data into the main longitudinal DataFrame.
        Handles duplicate months_since_baseline columns by filling missing values and dropping duplicates.
        
        Args:
            longitudinal_df: Main longitudinal DataFrame to merge into
            new_df: New DataFrame to merge (e.g., non_motor_df, medication_df)
            suffix: Suffix to use for duplicate columns (e.g., '_non_motor', '_med', '_age')
            
        Returns:
            Updated longitudinal_df with merged_dataset data
        """
        if len(new_df) > 0:
            longitudinal_df = longitudinal_df.merge(
                new_df,
                on=['PATNO', 'EVENT_ID'],
                how='outer',
                suffixes=('', suffix)
            )
            months_col = f'months_since_baseline{suffix}'
            if months_col in longitudinal_df.columns:
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
        
        static_df = self._load_and_merge_static_data()
        longitudinal_df = self._load_and_merge_longitudinal_data()
        
        print(f"\n✓ Longitudinal data merged: {len(longitudinal_df)} visits, {len(longitudinal_df.columns)-3} features")
        print(f"  Visits per patient: {len(longitudinal_df) / longitudinal_df['PATNO'].nunique():.1f} average")
        
        return {
            'static': static_df,
            'longitudinal': longitudinal_df
        }
    
    def compute_progression_slopes(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_scalers_exist()
        return self.label_engineer.compute_progression_slopes(longitudinal_df)
    
    def create_missingness_masks(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str],
        scaler: FeatureScaler = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_scalers_exist()
        return self.feature_engineer.create_missingness_masks(df, feature_cols, scaler=scaler)
    
    def fit_scalers(self, prepared_data: Dict, train_patnos: List[int] = None):
        """
        Fit feature scalers on training data.
        Should be called before create_feature_vectors() if normalization is enabled.
        
        Args:
            prepared_data: Dict with 'static' and 'longitudinal' DataFrames
            train_patnos: Optional list of training patient IDs to fit on.
                         If None, fits on all patients in prepared_data.
        """
        self._ensure_scalers_exist()
        self.feature_engineer.fit_scalers(prepared_data, train_patnos=train_patnos)
        self.static_scaler = self.feature_engineer.static_scaler
        self.motor_scaler = self.feature_engineer.motor_scaler
        self.updrs_supplementary_scaler = self.feature_engineer.updrs_supplementary_scaler
        self.non_motor_scaler = self.feature_engineer.non_motor_scaler
        self.medication_scaler = self.feature_engineer.medication_scaler
        self.age_at_visit_scaler = self.feature_engineer.age_at_visit_scaler
    
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
                -             slopes: Dict[PATNO, Dict[str, float]] - progression slopes per patient
                         Keys: 'NP1RTOT_slope', 'NP2PTOT_slope', 'NP3TOT_slope', 'NP4TOT_slope'
                         Values: slope value or NaN if not available
        """
        # If no prepared_data provided, use prepare_final_dataset output
        if prepared_data is None:
            prepared_data = self.prepare_final_dataset()
        
        static_df = prepared_data['static']
        longitudinal_df = prepared_data['longitudinal']
        slopes_df = prepared_data['slopes']
        
        self._ensure_scalers_exist()
        features = self.feature_engineer.create_feature_vectors(prepared_data)
        static_data = features["static_data"]
        longitudinal_data = features["longitudinal_data"]

        all_patnos = list(static_data.keys())
        slopes = self.label_engineer.initialize_slope_dict(all_patnos)
        self.label_engineer.fill_slope_dict_from_df(slopes, slopes_df)
        
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
        
        data = self.load_all_data()
        data = self._filter_to_complete_cases(data)
        slopes_df = self.compute_progression_slopes(data['longitudinal'])
        data['slopes'] = slopes_df
        data['metadata'] = self._create_metadata(data, slopes_df)
        
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
        
        IMPORTANT: Patient filtering happens at the visit_index level (single source of truth).
        The loaders are initialized with apply_min_visits_filter=False to avoid redundant filtering.
        
        Args:
            prepared_data: Optional dict with raw DataFrames. If None, loads data.
            min_visits: Minimum number of visits required per patient.
                       Filtering is applied ONCE at the visit_index level.
            
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
        
        if prepared_data is None:
            static_df, updrs_df, non_motor_df, medication_df, age_at_visit_df = self._load_raw_data()
        else:
            static_df = prepared_data.get('static', pd.DataFrame())
            updrs_df = prepared_data.get('longitudinal', pd.DataFrame())
            non_motor_df = prepared_data.get('non_motor', pd.DataFrame(columns=['PATNO', 'EVENT_ID']))
            medication_df = prepared_data.get('medication', pd.DataFrame(columns=['PATNO', 'EVENT_ID']))
            age_at_visit_df = prepared_data.get('age_at_visit', pd.DataFrame(columns=['PATNO', 'EVENT_ID']))
        
        visit_index, valid_patnos = self._build_visit_index(
            [updrs_df, non_motor_df, medication_df, age_at_visit_df], min_visits
        )
        static_df = static_df[static_df['PATNO'].isin(valid_patnos)]
        static_data = self._create_static_feature_vectors(static_df, valid_patnos)
        
        print("\n--- Aligning Modalities to Visit Index ---")
        
        motor_cols = [c for c in self.config.features.motor_features if c in updrs_df.columns]
        updrs_supplementary_cols = [c for c in self.config.features.updrs_supplementary_features 
                                   if c in updrs_df.columns]
        non_motor_cols = [c for c in self.config.features.non_motor_features 
                        if c in non_motor_df.columns]
        med_cols = [c for c in self.config.features.medication_features if c in medication_df.columns]
        updrs_total_cols = self.config.features.all_updrs_totals
        age_at_visit_cols = self.config.features.age_at_visit_features
        
        longitudinal_data = self._align_modalities_to_visit_index(
            visit_index, valid_patnos, updrs_df, non_motor_df, medication_df, 
            age_at_visit_df, motor_cols, updrs_supplementary_cols, 
            non_motor_cols, med_cols, age_at_visit_cols
        )
        
        slopes_df = (
            self.compute_progression_slopes(updrs_df) 
            if not updrs_df.empty 
            else pd.DataFrame()
        )
        slopes = self._initialize_slopes_dict(valid_patnos.tolist())
        self._fill_slopes_from_df(slopes, slopes_df)
        
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