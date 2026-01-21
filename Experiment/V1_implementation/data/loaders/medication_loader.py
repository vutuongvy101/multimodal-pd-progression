"""
Task 5: Medication Data Loader
Load medication information (LEDD, medication history, ON/OFF status)
"""

import os
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
        
    def __load_data_file__(self, file_path_config: str, feature_list: List[str], file_name: str, require_infodt: bool = False) -> pd.DataFrame:
        """
        Generic loader for longitudinal files used by MedicationLoader.
        
        Args:
            file_path_config: Config attribute name (e.g., 'ledd', 'vital_signs')
            feature_list: List of features to extract from file
            file_name: Display name for logging
            require_infodt: Whether INFODT is required (default False, since vital_signs doesn't have it)
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
            
            # Check required columns
            for rc in ['PATNO', 'EVENT_ID']:
                if rc not in df.columns:
                    msg = f"Required column '{rc}' missing in {file_name}: {resolved}"
                    _log.error(msg)
                    return pd.DataFrame()

            # Extract available features
            available = [c for c in feature_list if c in df.columns]
            
            # Build column list: always include PATNO, EVENT_ID
            cols = ['PATNO', 'EVENT_ID']
            
            # Add INFODT if available (not required)
            if 'INFODT' in df.columns:
                cols.append('INFODT')
            
            # Add available features
            cols.extend(available)
            
            return df[cols].copy()
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            _log.exception(f"Could not parse {file_name}: {e}")
            return pd.DataFrame()
        except Exception:
            _log.exception(f"Unexpected error loading {file_name}")
            return pd.DataFrame()
    def _load_raw(self) -> pd.DataFrame:
        """
        Load medication data from three separate sources:
        - LEDD (Levodopa Equivalent Daily Dose)
        - Vital Signs
        - PD Diagnosis History
        
        These are merged on PATNO and EVENT_ID.
        INFODT is only available in some files and is optional.
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and medication features
        """
        med_dfs = []
        
        # Load LEDD data
        ledd_df = self.__load_data_file__(
            'ledd', 
            self.config.features.ledd_features, 
            'LEDD',
            require_infodt=False
        )
        if len(ledd_df) > 0:
            med_dfs.append(ledd_df)
            print(f"  ✓ Loaded LEDD: {len(ledd_df)} rows")
        
        # Load Vital Signs data (no INFODT in this file)
        vital_df = self.__load_data_file__(
            'vital_signs',
            self.config.features.vital_signs_features,
            'Vital Signs',
            require_infodt=False
        )
        if len(vital_df) > 0:
            med_dfs.append(vital_df)
            print(f"  ✓ Loaded Vital Signs: {len(vital_df)} rows")
        
        # Load PD Diagnosis data
        pd_diag_df = self.__load_data_file__(
            'pd_diagnosis',
            self.config.features.pd_diagnosis_features,
            'PD Diagnosis',
            require_infodt=False
        )
        if len(pd_diag_df) > 0:
            med_dfs.append(pd_diag_df)
            print(f"  ✓ Loaded PD Diagnosis: {len(pd_diag_df)} rows")
        
        if len(med_dfs) == 0:
            print("  ⚠️  No medication files found. Creating empty medication dataframe.")
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline'])
        
        # Merge all medication dataframes on PATNO and EVENT_ID
        med_df = med_dfs[0]
        for df in med_dfs[1:]:
            med_df = med_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            med_df = med_df.loc[:, ~med_df.columns.str.endswith('_dup')]
        
        print(f"✓ Medication data: {len(med_df)} visits, {len(med_df.columns)-2} features (after merge)")
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

