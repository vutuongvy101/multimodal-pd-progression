"""
Task 3: UPDRS Data Loader
Load all four UPDRS parts (I, II, III, IV)
"""

import pandas as pd
import numpy as np
import logging
import traceback
from typing import List, Dict

from ..base_loader import LongitudinalDataLoader
from training.config import DataConfig

_LOG = logging.getLogger(__name__)


class UPDRSLoader(LongitudinalDataLoader):
    """
    Loads all UPDRS parts and supplementary motor assessments:
    
    Core UPDRS:
    - Part I: Non-motor experiences (NP1*) - loaded from both part1_ques and part1
    - Part II: Motor ADL (NP2*)
    - Part III: Motor examination (NP3*)
    - Part IV: Motor complications (NP4*)
    
    Supplementary Motor Assessments:
    - Schwab & England ADL scale
    - Neuro QoL Lower Extremity Function
    - Neuro QoL Upper Extremity Function
    - Participant Motor Function Questionnaire
    
    Output structure:
    - PATNO, EVENT_ID, INFODT, months_since_baseline
    - All core UPDRS features + supplementary features
    """
    
    def __init__(self, base_dir: str, config: DataConfig, valid_participants=None):
        """
        Args:
            base_dir: Base directory for data files
            config: Configuration dict with file paths
            valid_participants: Optional pre-filtered participant DataFrame (shared across loaders)
        """
        super().__init__(base_dir, config, valid_participants=valid_participants)
        # config is now stored in base class as self.config - no need to store again
    
    def __load_data_file__(self, file_path_config: str, feature_list: List[str], 
                          file_name: str, required: bool = False) -> pd.DataFrame:
        """
        Generic method to load a longitudinal data file
        
        Args:
            file_path_config: Config attribute name for the file path (e.g., 'updrs_part1')
            feature_list: List of feature columns to extract (from config)
            file_name: Name of the file for logging (e.g., 'Part I (UPDRS)')
        
        Returns:
            DataFrame with PATNO, EVENT_ID, INFODT, and selected features (or empty df if failed)
        """
        import os
        
        def resolve_path(file_path):
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        try:
            file_path = getattr(self.config.data, file_path_config, None)

            if file_path is None:
                msg = f"{file_name} config not found: {file_path_config}"
                _LOG.error(msg)
                if required:
                    raise FileNotFoundError(msg)
                return pd.DataFrame()

            resolved_path = resolve_path(file_path)

            if not os.path.exists(resolved_path):
                msg = f"{file_name} file not found: {resolved_path}"
                _LOG.error(msg)
                if required:
                    raise FileNotFoundError(msg)
                return pd.DataFrame()

            _LOG.info(f"Loading {file_name} from: {resolved_path}")
            data_df = pd.read_csv(resolved_path, low_memory=False)

            # Select required columns + requested features
            required_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            # Only select features that actually exist in the file
            available_features = [f for f in feature_list if f in data_df.columns]
            cols_to_keep = required_cols + available_features

            # Ensure key columns exist
            for rc in ['PATNO', 'EVENT_ID']:
                if rc not in data_df.columns:
                    msg = f"Required column '{rc}' missing in {file_name}: {resolved_path}"
                    _LOG.error(msg)
                    if required:
                        raise ValueError(msg)
                    return pd.DataFrame()

            data_df = data_df[cols_to_keep].copy()
            _LOG.info(f"{file_name}: {len(data_df)} visits, {len(available_features)} features")
            return data_df

        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            _LOG.exception(f"Could not load {file_name}: {e}")
            if required:
                raise
            return pd.DataFrame()
        except Exception as e:
            _LOG.exception(f"Unexpected error loading {file_name}")
            if required:
                raise
            return pd.DataFrame()
        
    def _load_raw(self) -> pd.DataFrame:
        """
        Load all UPDRS parts and supplementary motor assessments, then merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and all UPDRS + supplementary features
        """
        print("Loading UPDRS and supplementary motor assessment data...")
        
        # Core UPDRS Parts
        part1_ques_df = self.__load_data_file__(
            'updrs_part1_ques',
            self.config.features.part1_questionnaire_features,
            'Part I Questionnaire (NP1 questions)',
            required=True
        )
        
        part1_df = self.__load_data_file__(
            'updrs_part1',
            self.config.features.part1_updrs_features,
            'Part I (NP1 UPDRS)',
            required=True
        )
        
        part2_df = self.__load_data_file__(
            'updrs_part2',
            self.config.features.part2_features,
            'Part II (Motor ADL - NP2*)',
            required=True
        )
        
        part3_df = self.__load_data_file__(
            'updrs_part3',
            self.config.features.part3_features,
            'Part III (Motor exam - NP3*)',
            required=True
        )
        
        part4_df = self.__load_data_file__(
            'updrs_part4',
            self.config.features.part4_features,
            'Part IV (Motor complications - NP4*)',
            required=True
        )
        
        # Supplementary Motor Assessments
        schwab_df = self.__load_data_file__(
            'schwab_england',
            self.config.features.schwab_england_features,
            'Schwab & England ADL',
            required=False
        )
        
        neuro_qol_lower_df = self.__load_data_file__(
            'neuro_qol_lower',
            self.config.features.neuro_qol_lower_features,
            'Neuro QoL Lower Extremity',
            required=False
        )
        
        neuro_qol_upper_df = self.__load_data_file__(
            'neuro_qol_upper',
            self.config.features.neuro_qol_upper_features,
            'Neuro QoL Upper Extremity',
            required=False
        )
        
        participant_motor_df = self.__load_data_file__(
            'participant_motor',
            self.config.features.participant_motor_features,
            'Participant Motor Function Questionnaire',
            required=False
        )
        
        # Merge all files together on [PATNO, EVENT_ID]
        updrs_df = None
        all_dfs = [
            part1_ques_df, part1_df, part2_df, part3_df, part4_df,
            schwab_df, neuro_qol_lower_df, neuro_qol_upper_df, participant_motor_df
        ]
        
        for df in all_dfs:
            if len(df) > 0:
                if updrs_df is None:
                    updrs_df = df
                else:
                    updrs_df = updrs_df.merge(
                        df,
                        on=['PATNO', 'EVENT_ID'],
                        how='outer',
                        suffixes=('', '_dup')
                    )
                    # Remove duplicate INFODT columns
                    updrs_df = updrs_df.loc[:, ~updrs_df.columns.str.endswith('_dup')]
        
        if updrs_df is None or len(updrs_df) == 0:
            raise ValueError("Could not load any UPDRS or motor assessment data")

        print(f"✓ Merged UPDRS + supplementary data: {len(updrs_df)} visits, {len(updrs_df.columns)-3} features")
        
        return updrs_df
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO', 'EVENT_ID', 'months_since_baseline',
            'NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT'  # At minimum, the totals
        ]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate UPDRS data"""
        # Check required columns
        if 'PATNO' not in df.columns or 'EVENT_ID' not in df.columns:
            raise ValueError("Missing PATNO or EVENT_ID columns")
        
        # Check for at least one UPDRS total
        totals = [c for c in df.columns if c in self.config.features.all_updrs_totals]
        if len(totals) == 0:
            raise ValueError("No UPDRS total scores found")
        
        # Check time computation
        if 'months_since_baseline' not in df.columns:
            print("⚠️  Warning: months_since_baseline not computed")
        else:
            # Check for reasonable values
            time_range = df['months_since_baseline'].dropna()
            if len(time_range) > 0:
                if time_range.min() < -1 or time_range.max() > 200:
                    print(f"⚠️  Warning: Time range looks unusual: {time_range.min():.1f} - {time_range.max():.1f} months")
        
        print(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        print(f"  Available UPDRS totals: {', '.join(totals)}")
        
        return True

