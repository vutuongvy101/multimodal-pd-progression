"""
Data preparation for V1 model
Extracts features, handles missingness, computes slopes
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PPMIDataPreparator:
    """Prepare PPMI data for V1 model"""
    
    def __init__(self, config):
        self.config = config
        self.feature_config = config.features
        self.training_config = config.training
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required CSV files"""
        data_config = self.config.data
        base_dir = data_config.base_dir
        
        data = {}
        
        print("Loading data files...")
        
        # Participant status (demographics, enrollment info)
        data['participant_status'] = pd.read_csv(f"{base_dir}/{data_config.participant_status}")
        
        # Genetics
        data['genetic_consensus'] = pd.read_csv(f"{base_dir}/{data_config.genetic_consensus}")
        data['prs_scores'] = pd.read_csv(f"{base_dir}/{data_config.prs_scores}")
        data['prs_pcs'] = pd.read_csv(f"{base_dir}/{data_config.prs_pcs}")
        
        # Clinical assessments (add more as needed)
        # data['updrs_part1'] = pd.read_csv(f"{base_dir}/{data_config.updrs_part1}")
        # data['updrs_part2'] = pd.read_csv(f"{base_dir}/{data_config.updrs_part2}")
        # data['updrs_part3'] = pd.read_csv(f"{base_dir}/{data_config.updrs_part3}")
        # data['updrs_part4'] = pd.read_csv(f"{base_dir}/{data_config.updrs_part4}")
        # data['moca'] = pd.read_csv(f"{base_dir}/{data_config.moca}")
        
        print(f"Loaded {len(data)} data files")
        return data
    
    def merge_static_features(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all static (patient-level) features"""
        
        print("\nMerging static features...")
        
        # Start with participant status
        static_df = raw_data['participant_status'][['PATNO', 'ENROLL_AGE', 'SEX']].copy()
        
        # Add genetics
        if 'genetic_consensus' in raw_data:
            genetic_cols = ['PATNO', 'LRRK2', 'GBA', 'SNCA', 'PRKN', 'PATHVAR_COUNT', 'APOE']
            genetic_cols = [c for c in genetic_cols if c in raw_data['genetic_consensus'].columns]
            static_df = static_df.merge(
                raw_data['genetic_consensus'][genetic_cols],
                on='PATNO',
                how='left'
            )
        
        # Add PRS scores
        if 'prs_scores' in raw_data:
            prs_cols = ['PATNO', 'GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS']
            prs_cols = [c for c in prs_cols if c in raw_data['prs_scores'].columns]
            static_df = static_df.merge(
                raw_data['prs_scores'][prs_cols],
                on='PATNO',
                how='left'
            )
        
        # Add principal components (population structure)
        if 'prs_pcs' in raw_data:
            pc_cols = ['PATNO'] + [f'Genetic_PRS_PC{i}' for i in range(1, 11)]
            pc_cols = [c for c in pc_cols if c in raw_data['prs_pcs'].columns]
            # Use baseline visit only (EVENT_ID == 'SC')
            pcs_df = raw_data['prs_pcs'][raw_data['prs_pcs']['EVENT_ID'] == 'SC']
            static_df = static_df.merge(
                pcs_df[pc_cols],
                on='PATNO',
                how='left'
            )
        
        print(f"Static features: {len(static_df)} patients, {len(static_df.columns)-1} features")
        return static_df
    
    def extract_longitudinal_features(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract time-varying features from all visits"""
        
        print("\nExtracting longitudinal features...")
        
        # This is a placeholder - you'll need to implement based on your actual data structure
        # Key requirements:
        # 1. Each row = one visit for one patient
        # 2. Must have: PATNO, EVENT_ID, visit_date or months_since_baseline
        # 3. Include all motor, non-motor, medication features
        
        # Example structure:
        longitudinal_data = []
        
        # For UPDRS Part III (motor)
        # if 'updrs_part3' in raw_data:
        #     motor_df = raw_data['updrs_part3'].copy()
        #     motor_df = self._compute_time_since_baseline(motor_df)
        #     longitudinal_data.append(motor_df)
        
        # For now, return a placeholder
        print("WARNING: longitudinal feature extraction not yet implemented")
        return pd.DataFrame()
    
    def _compute_time_since_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute months since baseline for each visit"""
        
        # Convert dates to datetime
        if 'INFODT' in df.columns:
            df['visit_date'] = pd.to_datetime(df['INFODT'], errors='coerce')
            
            # Get baseline date for each patient
            baseline = df[df['EVENT_ID'] == 'BL'].groupby('PATNO')['visit_date'].first()
            baseline = baseline.to_dict()
            
            # Compute months since baseline
            df['months_since_baseline'] = df.apply(
                lambda row: (row['visit_date'] - baseline.get(row['PATNO'])).days / 30.44
                if pd.notna(row['visit_date']) and row['PATNO'] in baseline
                else np.nan,
                axis=1
            )
        else:
            # Use EVENT_ID mapping if no dates available
            event_mapping = {
                'SC': 0, 'BL': 0, 'V01': 0, 'V02': 3, 'V03': 6, 'V04': 12,
                'V05': 18, 'V06': 24, 'V07': 30, 'V08': 36, 'V09': 42, 'V10': 48,
                'V11': 54, 'V12': 60, 'V13': 72, 'V14': 84, 'V15': 96
            }
            df['months_since_baseline'] = df['EVENT_ID'].map(event_mapping)
        
        return df
    
    def compute_progression_slopes(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        """Compute empirical progression slopes for each patient"""
        
        print("\nComputing progression slopes...")
        
        min_visits = self.training_config.min_visits_for_slope
        
        # Group by patient
        slopes = []
        
        for patno, group in longitudinal_df.groupby('PATNO'):
            # Need at least min_visits visits with valid NP3TOT
            valid_visits = group.dropna(subset=['months_since_baseline', 'NP3TOT'])
            
            if len(valid_visits) >= min_visits:
                times = valid_visits['months_since_baseline'].values
                scores = valid_visits['NP3TOT'].values
                
                # Linear regression
                result = linregress(times, scores)
                
                slopes.append({
                    'PATNO': patno,
                    'slope': result.slope,
                    'intercept': result.intercept,
                    'r_value': result.rvalue,
                    'p_value': result.pvalue,
                    'std_err': result.stderr,
                    'n_visits': len(valid_visits)
                })
        
        slopes_df = pd.DataFrame(slopes)
        print(f"Computed slopes for {len(slopes_df)} patients")
        
        return slopes_df
    
    def create_missingness_masks(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create value and mask arrays for a set of features"""
        
        values = df[feature_cols].values
        mask = np.isnan(values).astype(np.float32)
        
        # Fill NaN with 0 (the mask tells the model it's missing)
        values = np.nan_to_num(values, nan=0.0)
        
        return values, mask
    
    def prepare_dataset(self) -> Dict:
        """Main pipeline: prepare complete dataset for training"""
        
        print("=" * 80)
        print("Starting data preparation pipeline")
        print("=" * 80)
        
        # 1. Load raw data
        raw_data = self.load_raw_data()
        
        # 2. Merge static features
        static_df = self.merge_static_features(raw_data)
        
        # 3. Extract longitudinal features
        longitudinal_df = self.extract_longitudinal_features(raw_data)
        
        # 4. Compute progression slopes
        if not longitudinal_df.empty:
            slopes_df = self.compute_progression_slopes(longitudinal_df)
        else:
            slopes_df = pd.DataFrame()
        
        # 5. Return prepared data
        prepared_data = {
            'static': static_df,
            'longitudinal': longitudinal_df,
            'slopes': slopes_df,
            'raw': raw_data
        }
        
        print("\n" + "=" * 80)
        print("Data preparation complete")
        print("=" * 80)
        
        return prepared_data
    
    def create_feature_vectors(self, prepared_data: Dict) -> Dict:
        """Create feature vectors with missingness masks for model input"""
        
        static_df = prepared_data['static']
        longitudinal_df = prepared_data['longitudinal']
        
        # Static features
        static_cols = [c for c in self.feature_config.static_features if c in static_df.columns]
        static_values, static_mask = self.create_missingness_masks(static_df, static_cols)
        
        # Longitudinal features (grouped by modality)
        # This will be a list of sequences, one per patient
        # Each sequence is a list of visits
        
        # TODO: Implement based on actual longitudinal data structure
        
        return {
            'static_values': static_values,
            'static_mask': static_mask,
            'static_cols': static_cols,
            # Add longitudinal features here
        }


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from training.config import get_default_config
    
    # Test data preparation
    config = get_default_config()
    preparator = PPMIDataPreparator(config)
    
    try:
        prepared_data = preparator.prepare_dataset()
        
        print("\nPrepared data summary:")
        print(f"Static features shape: {prepared_data['static'].shape}")
        if not prepared_data['longitudinal'].empty:
            print(f"Longitudinal features shape: {prepared_data['longitudinal'].shape}")
        if not prepared_data['slopes'].empty:
            print(f"Slopes computed for: {len(prepared_data['slopes'])} patients")
            
    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
