"""
Age at Visit Data Loader
Load age at visit data (longitudinal - varies by visit)
"""

import pandas as pd
import numpy as np
from typing import List

from ..base_loader import LongitudinalDataLoader
from training.config import DataConfig


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
        
    def load(self) -> pd.DataFrame:
        """
        Load age at visit data
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and AGE_AT_VISIT
        """
        try:
            file_path = self.resolve_path(self.config.data.age_at_visit)
            print(f"Loading age_at_visit from: {file_path}")
            
            import os
            if not os.path.exists(file_path):
                print(f"  ⚠️  Warning: age_at_visit file not found: {file_path}")
                return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
            
            age_df = pd.read_csv(file_path)
            
            if 'PATNO' not in age_df.columns:
                print(f"  ⚠️  Warning: PATNO not found in age_at_visit, skipping")
                return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
            
            # Select only PATNO, EVENT_ID, and AGE_AT_VISIT columns
            age_cols_to_keep = ['PATNO', 'EVENT_ID', 'AGE_AT_VISIT']
            age_cols_available = [col for col in age_cols_to_keep if col in age_df.columns]
            
            if 'AGE_AT_VISIT' not in age_cols_available:
                print(f"  ⚠️  Warning: AGE_AT_VISIT column not found, skipping")
                return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
            
            age_df = age_df[age_cols_available].copy()
            
            # Note: age_at_visit file doesn't contain INFODT, so we cannot compute 
            # months_since_baseline here. It will be filled during merging with other 
            # longitudinal data (e.g., UPDRS) that contains INFODT.
            # Initialize months_since_baseline as None - it will be filled during merge
            age_df['months_since_baseline'] = None
            
            print(f"  ✓ Loaded age_at_visit: {len(age_df)} visit records")
            
            return age_df
            
        except Exception as e:
            print(f"  ⚠️  Could not load age_at_visit: {e}")
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline', 'AGE_AT_VISIT'])
    
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
