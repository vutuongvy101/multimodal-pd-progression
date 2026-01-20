"""
Age at Visit Data Loader
Load age at visit data (longitudinal - varies by visit)
"""

import pandas as pd
import numpy as np
import logging
from typing import List

from ..base_loader import LongitudinalDataLoader
from training.config import DataConfig

_LOG = logging.getLogger(__name__)


class AgeAtVisitLoader(LongitudinalDataLoader):
    """
    Loads age at visit data:
    - AGE_AT_VISIT (varies by visit, not static)
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
        
    def __load_data_file__(self, file_path_config: str, feature_list: List[str], file_name: str, required: bool = False) -> pd.DataFrame:
        """Generic loader for age/visit CSV files.

        Returns an empty DataFrame with standard columns on non-fatal errors.
        If `required=True`, errors are raised to fail-fast.
        """
        try:
            file_path = getattr(self.config.data, file_path_config, None)
            if file_path is None:
                msg = f"{file_name} config not found: {file_path_config}"
                _LOG.error(msg)
                if required:
                    raise FileNotFoundError(msg)
                return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])

            resolved = self.resolve_path(file_path)
            import os
            if not os.path.exists(resolved):
                msg = f"{file_name} file not found: {resolved}"
                _LOG.error(msg)
                if required:
                    raise FileNotFoundError(msg)
                return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])

            _LOG.info(f"Loading {file_name} from: {resolved}")
            df = pd.read_csv(resolved, low_memory=False)

            # Basic validation
            if 'PATNO' not in df.columns:
                msg = f"Required column 'PATNO' missing in {file_name}: {resolved}"
                _LOG.error(msg)
                if required:
                    raise ValueError(msg)
                return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])

            required_cols = ['PATNO', 'EVENT_ID']
            available = [c for c in feature_list if c in df.columns]
            cols = [c for c in required_cols if c in df.columns] + available

            out_df = df[cols].copy()

            # If INFODT present we could compute months_since_baseline here; otherwise initialize as None
            if 'INFODT' in out_df.columns:
                try:
                    out_df = self.compute_time_since_baseline(out_df)
                except Exception:
                    _LOG.exception("Error computing months_since_baseline for age_at_visit; leaving as None")
                    out_df['months_since_baseline'] = None
            else:
                out_df['months_since_baseline'] = None

            _LOG.info(f"{file_name}: {len(out_df)} records")
            return out_df

        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            _LOG.exception(f"Could not load {file_name}: {e}")
            if required:
                raise
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
        except Exception:
            _LOG.exception(f"Unexpected error loading {file_name}")
            if required:
                raise
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
        
    def _load_raw(self) -> pd.DataFrame:
        """
        Load age at visit data
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and AGE_AT_VISIT
        """
        # Use generic loader (age_at_visit is optional)
        age_df = self.__load_data_file__('age_at_visit', self.config.features.age_at_visit_features, 'Age at Visit', required=False)
        if age_df is None or len(age_df) == 0:
            _LOG.warning("No age_at_visit data loaded; returning empty DataFrame")
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
        _LOG.info(f"✓ Loaded age_at_visit: {len(age_df)} records")
        return age_df

    def load(self) -> pd.DataFrame:
        return self._load_raw()
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'
        ]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate age at visit data"""
        # Check required columns
        if 'PATNO' not in df.columns or 'EVENT_ID' not in df.columns:
            raise ValueError("Missing PATNO or EVENT_ID columns")
        
        if 'AGE_AT_VISIT' not in df.columns:
            raise ValueError("Missing AGE_AT_VISIT column")
        
        # Check age values are reasonable
        if len(df) > 0:
            age_values = df['AGE_AT_VISIT'].dropna()
            if len(age_values) > 0:
                if age_values.min() < 0 or age_values.max() > 150:
                    print(f"⚠️  Warning: Unusual age values found: {age_values.min():.1f} - {age_values.max():.1f}")
                print(f"  AGE_AT_VISIT range: {age_values.min():.1f} - {age_values.max():.1f}")
        
        print(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        
        return True
