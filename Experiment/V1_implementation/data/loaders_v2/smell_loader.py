"""
Task 4: Smell Loader
Load smell assessments 
- UPSIT (University of Pennsylvania Smell Identification Test)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict

from ..base_loader import LongitudinalDataLoader
from training.config_v2 import DataConfig_v2

_log = logging.getLogger(__name__)


class SmellLoader(LongitudinalDataLoader):
    """
    Loads smell assessments:
    - UPSIT (University of Pennsylvania Smell Identification Test)
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
        Generic loader for smell assessment CSVs. Returns empty DataFrame on failure.
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
        Load smell assessments and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and smell scores
        """
        smell_dfs = []

        upsit_df = self.__load_data_file__('upsit', self.config.features.upsit_features, 'UPSIT (University of Pennsylvania Smell Identification Test)', required=False)
        if len(upsit_df) > 0:
            smell_dfs.append(('UPSIT', upsit_df))

        if len(smell_dfs) == 0:
            raise ValueError("Could not load any smell assessments")

        smell_df = None
        for name, df in smell_dfs:
            if smell_df is None:
                smell_df = df
            else:
                smell_df = smell_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
                smell_df = smell_df.loc[:, ~smell_df.columns.str.endswith('_dup')]

        print(f"✓ Merged smell data: {len(smell_df)} visits, {len(smell_df.columns)-3} features")
        return smell_df

    def get_required_columns(self) -> List[str]:
        """Return list of required columns for smell assessments."""
        return ['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.smell_features

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate smell DataFrame has required columns and at least one smell feature."""
        if df is None or len(df) == 0:
            raise ValueError("Smell DataFrame is empty or None")

        smell_cols = [c for c in df.columns if c in self.config.features.smell_features]
        if len(smell_cols) == 0:
            raise ValueError("No smell assessment columns found")
        
        required_cols = self.get_required_columns()
        missing_cols = set(required_cols) - set(df.columns.to_list())
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        extra_cols = set(df.columns) - set(required_cols)
        if extra_cols:
            _log.warning(f"Extra columns found (not required): {sorted(extra_cols)}")

        _log.info(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        _log.info(f"  Available assessments: {', '.join(smell_cols)}")
        return True