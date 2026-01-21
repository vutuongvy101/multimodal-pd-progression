"""
Task 4: Behavioral Loader
Load behavioral assessments
- SCOPA-AUT (SCale for Outcomes in Parkinson's Disease for Autonomic Symptoms - Autonomic)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict

from ..base_loader import LongitudinalDataLoader
from training.config_v2 import DataConfig_v2

_log = logging.getLogger(__name__)


class BehavioralLoader(LongitudinalDataLoader):
    """
    Loads behavioral assessments:
    - SCOPA-AUT (SCale for Outcomes in Parkinson's Disease for Autonomic Symptoms - Autonomic)
    """
    
    def __init__(self, base_dir: str, config: DataConfig_v2, valid_participants=None):
        """
        Args:
            base_dir: Base directory for data files
            config: Configuration dict with file paths
            valid_participants: Optional pre-filtered participant DataFrame (shared across loaders)
        """
        super().__init__(base_dir, config, valid_participants=valid_participants)
        # config is now stored in base class as self.config - no need to store again
        
    def __load_data_file__(self, file_path_config: str, feature_list: List[str], file_name: str, required: bool = False) -> pd.DataFrame:
        """
        Generic loader for behavioral assessment CSVs. Returns empty DataFrame on failure.
        """
        import logging
        _log = logging.getLogger(__name__)
        try:
            file_path = getattr(self.config.data, file_path_config, None)
            if file_path is None:
                msg = f"{file_name} config not found: {file_path_config}"
                _log.error(msg)
                if required:
                    raise FileNotFoundError(msg)
                return pd.DataFrame()

            resolved = self.resolve_path(file_path)
            if not resolved or not os.path.exists(resolved):
                msg = f"{file_name} file not found: {resolved}"
                _log.error(msg)
                if required:
                    raise FileNotFoundError(msg)
                return pd.DataFrame()

            _log.info(f"Loading {file_name} from: {resolved}")
            df = pd.read_csv(resolved, low_memory=False)
            required_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            available = [c for c in feature_list if c in df.columns]

            for rc in ['PATNO', 'EVENT_ID']:
                if rc not in df.columns:
                    msg = f"Required column '{rc}' missing in {file_name}: {resolved}"
                    _log.error(msg)
                    if required:
                        raise ValueError(msg)
                    return pd.DataFrame()

            cols = required_cols + available
            return df[cols].copy()
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            _log.exception(f"Could not load {file_name}: {e}")
            if required:
                raise
            return pd.DataFrame()
        except Exception:
            _log.exception(f"Unexpected error loading {file_name}")
            if required:
                raise
            return pd.DataFrame()
        
    def _load_raw(self) -> pd.DataFrame:
        """
        Load behavioral assessments and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and behavioral scores
        """
        behavioral_dfs = []

        scopa_df = self.__load_data_file__('scopa_aut', self.config.features.scopa_aut_features, 'SCOPA-AUT (SCales for Outcomes in PArkinson’s disease - Autonomic)', required=False)
        if len(scopa_df) > 0:
            behavioral_dfs.append(('SCOPA-AUT', scopa_df))

        if len(behavioral_dfs) == 0:
            raise ValueError("Could not load any behavioral assessments")

        behavioral_df = None
        for name, df in behavioral_dfs:
            if behavioral_df is None:
                behavioral_df = df
            else:
                behavioral_df = behavioral_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
                behavioral_df = behavioral_df.loc[:, ~behavioral_df.columns.str.endswith('_dup')]

        print(f"✓ Merged behavioral data: {len(behavioral_df)} visits, {len(behavioral_df.columns)-3} features")
        return behavioral_df

    def get_required_columns(self) -> List[str]:
        """Return list of required columns for behavioral assessments."""
        return ['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.behavioral_features

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate behavioral DataFrame has required columns and at least one behavioral feature."""
        if df is None or len(df) == 0:
            raise ValueError("Behavioral DataFrame is empty or None")
        behavioral_cols = [c for c in df.columns if c in self.config.features.behavioral_features]
        if len(behavioral_cols) == 0:
            raise ValueError("No behavioral assessment columns found")

        required_cols = self.get_required_columns()
        missing_cols = set(required_cols) - set(df.columns.to_list())
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        extra_cols = set(df.columns) - set(required_cols)
        if extra_cols:
            _log.warning(f"Extra columns found (not required): {sorted(extra_cols)}")

        _log.info(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        _log.info(f"  Available assessments: {', '.join(behavioral_cols)}")
        return True