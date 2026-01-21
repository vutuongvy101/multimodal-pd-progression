"""
Task 4: Cognitive Loader
Load Cognitive Assessments 
- MoCA (Montreal Cognitive Assessment)
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict

from ..base_loader import LongitudinalDataLoader
from training.config_v2 import DataConfig_v2

_log = logging.getLogger(__name__)


class CognitiveLoader(LongitudinalDataLoader):
    """
    Loads cognitive assessment:
    - MoCA (Montreal Cognitive Assessment)
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
        Generic loader for cognitive assessment CSVs. Returns empty DataFrame on failure.
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
        Load cognitive assessments (if multiple files) and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and cognitive score
        """
        cognitive_dfs = []

        moca_df = self.__load_data_file__('moca', self.config.features.moca_features, 'MoCA (Montreal Cognitive Assessment)', required=False)
        if len(moca_df) > 0:
            cognitive_dfs.append(('MoCA', moca_df))

        if len(cognitive_dfs) == 0:
            raise ValueError("Could not load any cognitive assessments")

        cognitive_df = None
        for name, df in cognitive_dfs:
            if cognitive_df is None:
                cognitive_df = df
            else:
                cognitive_df = cognitive_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
                cognitive_df = cognitive_df.loc[:, ~cognitive_df.columns.str.endswith('_dup')]

        print(f"✓ Merged cognitive data: {len(cognitive_df)} visits, {len(cognitive_df.columns)-3} features")
        return cognitive_df

    def get_required_columns(self) -> List[str]:
        """Return list of required columns for cognitive assessments."""
        return ['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.cognitive_features

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate cognitive DataFrame has required columns and at least one cognitive feature."""
        if df is None or len(df) == 0:
            raise ValueError("Cognitive DataFrame is empty or None")

        cognitive_cols = [c for c in df.columns if c in self.config.features.cognitive_features]
        if len(cognitive_cols) == 0:
            raise ValueError("No cognitive assessment columns found")

        required_cols = self.get_required_columns()
        missing_cols = set(required_cols) - set(df.columns.to_list())
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        extra_cols = set(df.columns) - set(required_cols)
        if extra_cols:
            _log.warning(f"Extra columns found (not required): {sorted(extra_cols)}")

        _log.info(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        _log.info(f"  Available assessments: {', '.join(cognitive_cols)}")
        return True