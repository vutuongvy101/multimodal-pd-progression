"""
Task 4: Non-Motor Assessments Loader
Load additional non-motor assessments (MoCA, sleep, autonomic, QoL)
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
    Loads non-UPDRS non-motor assessments:
    - MoCA (cognitive)
    - ESS (sleep)
    - SCOPA-AUT (autonomic)
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
        Generic loader for non-motor assessment CSVs. Returns empty DataFrame on failure.
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
        Load non-motor assessments and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and non-motor scores
        """
        non_motor_dfs = []

        moca_df = self.__load_data_file__('moca', self.config.features.moca_features, 'MoCA (cognitive)', required=False)
        if len(moca_df) > 0:
            non_motor_dfs.append(('MoCA', moca_df))

        ess_df = self.__load_data_file__('ess', self.config.features.ess_features, 'ESS (Epworth Sleepiness Scale)', required=False)
        if len(ess_df) > 0:
            non_motor_dfs.append(('ESS', ess_df))

        scopa_df = self.__load_data_file__('scopa_aut', self.config.features.scopa_aut_features, 'SCOPA-AUT (autonomic)', required=False)
        if len(scopa_df) > 0:
            non_motor_dfs.append(('SCOPA-AUT', scopa_df))

        se_df = self.__load_data_file__('schwab_england', self.config.features.schwab_england_features, 'Schwab & England (ADL)', required=False)
        if len(se_df) > 0:
            non_motor_dfs.append(('Schwab & England', se_df))

        if len(non_motor_dfs) == 0:
            raise ValueError("Could not load any non-motor assessments")

        non_motor_df = None
        for name, df in non_motor_dfs:
            if non_motor_df is None:
                non_motor_df = df
            else:
                non_motor_df = non_motor_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
                non_motor_df = non_motor_df.loc[:, ~non_motor_df.columns.str.endswith('_dup')]

        print(f"✓ Merged non-motor data: {len(non_motor_df)} visits, {len(non_motor_df.columns)-3} features")
        return non_motor_df

    def get_required_columns(self) -> List[str]:
        """Return list of required columns for non-motor assessments."""
        return ['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.non_motor_features

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate non-motor DataFrame has required columns and at least one non-motor feature."""
        if df is None or len(df) == 0:
            raise ValueError("Non-motor DataFrame is empty or None")

        non_motor_cols = [c for c in df.columns if c in self.config.features.non_motor_features]
        if len(non_motor_cols) == 0:
            raise ValueError("No non-motor assessment columns found")

        required_cols = self.get_required_columns()
        missing_cols = set(required_cols) - set(df.columns.to_list())
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        extra_cols = set(df.columns) - set(required_cols)
        if extra_cols:
            _log.warning(f"Extra columns found (not required): {sorted(extra_cols)}")

        _log.info(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        _log.info(f"  Available assessments: {', '.join(non_motor_cols)}")
        return True
