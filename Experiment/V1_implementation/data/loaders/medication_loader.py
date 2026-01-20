"""
Task 5: Medication Data Loader
Load medication information (LEDD, medication history, ON/OFF status)
"""

import pandas as pd
import numpy as np
from typing import List

from ..base_loader import LongitudinalDataLoader
from training.config import DataConfig


class MedicationLoader(LongitudinalDataLoader):
    """
    Loads medication data:
    - LEDD (levodopa equivalent daily dose)
    - Medication history
    - ON/OFF status (often in UPDRS Part III, but can be separate)
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
        
    def __load_data_file__(self, file_path_config: str, feature_list: List[str], file_name: str) -> pd.DataFrame:
        """
        Generic loader for longitudinal files used by MedicationLoader.
        """
        import logging
        _log = logging.getLogger(__name__)
        try:
            file_path = getattr(self.config.data, file_path_config, None)
            if file_path is None:
                msg = f"{file_name} config not found: {file_path_config}"
                _log.error(msg)
                return pd.DataFrame()

            resolved = self.resolve_path(file_path)
            if not resolved or not os.path.exists(resolved):
                msg = f"{file_name} file not found: {resolved}"
                _log.error(msg)
                return pd.DataFrame()

            _log.info(f"Loading {file_name} from: {resolved}")
            df = pd.read_csv(resolved, low_memory=False)
            required_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            available = [c for c in feature_list if c in df.columns]

            for rc in ['PATNO', 'EVENT_ID']:
                if rc not in df.columns:
                    msg = f"Required column '{rc}' missing in {file_name}: {resolved}"
                    _log.error(msg)
                    return pd.DataFrame()

            cols = required_cols + available
            return df[cols].copy()
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            _log.exception(f"Could not parse {file_name}: {e}")
            return pd.DataFrame()
        except Exception:
            _log.exception(f"Unexpected error loading {file_name}")
            return pd.DataFrame()
    def load(self) -> pd.DataFrame:
        """
        Load medication data
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and medication features
        """
        med_dfs = []
        import os

        # Try to load a medication file defined in config first
        med_from_config = self.__load_data_file__('medication', self.config.features.medication_features, 'Medication (config)')
        if len(med_from_config) > 0:
            med_dfs.append(med_from_config)

        # If not found via config, try common filenames (legacy)
        if len(med_dfs) == 0:
            possible_files = [
                'Use_of_PD_Medication.csv',
                'PD_Medications.csv',
                'Concomitant_Medications.csv'
            ]
            for fname in possible_files:
                try:
                    path = self.resolve_path(fname)
                    if os.path.exists(path):
                        print(f"  Loading medication from: {path}")
                        df = pd.read_csv(path, low_memory=False)
                        # pick columns
                        req = ['PATNO', 'EVENT_ID', 'INFODT']
                        led_col = next((c for c in self.config.features.ledd_features if c in df.columns), None)
                        pdcol = next((c for c in self.config.features.pdmedyn_features if c in df.columns), None)
                        cols = req + [c for c in [led_col, pdcol] if c]
                        med_dfs.append(df[cols].copy())
                        print(f"    ✓ Loaded {len(df)} rows from {fname}")
                        break
                except Exception:
                    continue

        if len(med_dfs) == 0:
            print("  ⚠️  No medication files found. Creating empty medication dataframe.")
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline'])

        med_df = med_dfs[0]
        for df in med_dfs[1:]:
            med_df = med_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
            med_df = med_df.loc[:, ~med_df.columns.str.endswith('_dup')]

        # Compute time since baseline
        if 'INFODT' in med_df.columns or 'EVENT_ID' in med_df.columns:
            med_df = self.compute_time_since_baseline(med_df)

        print(f"✓ Medication data: {len(med_df)} visits, {len(med_df.columns)-3} features")
        return med_df
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO', 'EVENT_ID', 'months_since_baseline'
            # Note: LEDD and other med features are optional
        ]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate medication data"""
        # Check required columns
        if 'PATNO' not in df.columns or 'EVENT_ID' not in df.columns:
            raise ValueError("Missing PATNO or EVENT_ID columns")
        
        # Check LEDD if present
        if 'LEDD' in df.columns:
            ledd_range = df['LEDD'].dropna()
            if len(ledd_range) > 0:
                if ledd_range.min() < 0:
                    print(f"⚠️  Warning: Negative LEDD values found")
                if ledd_range.max() > 3000:
                    print(f"⚠️  Warning: Very high LEDD values (>{ledd_range.max():.0f})")
                print(f"  LEDD range: {ledd_range.min():.1f} - {ledd_range.max():.1f}")
        
        print(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        
        return True

