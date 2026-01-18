"""
Task 4: Clinical Assessments Loader
Load additional clinical assessments (MoCA, sleep, autonomic, QoL)
"""

import pandas as pd
import numpy as np
from typing import List, Dict

from ..base_loader import LongitudinalDataLoader
from training.config import DataConfig


class ClinicalAssessmentsLoader(LongitudinalDataLoader):
    """
    Loads non-UPDRS clinical assessments:
    - MoCA (cognitive)
    - ESS (sleep)
    - SCOPA-AUT (autonomic)
    - Schwab & England (ADL)
    - Other scales as available
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
        Load clinical assessments and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and clinical scores
        """
        clinical_dfs = []
        
        # Load MoCA (cognitive)
        try:
            moca_path = self.resolve_path(self.config.data.moca)
            print(f"Loading MoCA from: {moca_path}")
            moca_df = pd.read_csv(moca_path)
            moca_df = moca_df[['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.moca_features]
            clinical_dfs.append(('MoCA', moca_df))
            print(f"  ✓ MoCA: {len(moca_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load MoCA: {e}")
        
        # Load ESS (Epworth Sleepiness Scale)
        try:
            ess_path = self.resolve_path(self.config.data.ess)
            print(f"Loading ESS from: {ess_path}")
            ess_df = pd.read_csv(ess_path)
            ess_df = ess_df[['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.ess_features]
            clinical_dfs.append(('ESS', ess_df))
            print(f"  ✓ ESS: {len(ess_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load ESS: {e}")
        
        # Load SCOPA-AUT (autonomic)
        try:
            scopa_path = self.resolve_path(self.config.data.scopa_aut)
            print(f"Loading SCOPA-AUT from: {scopa_path}")
            scopa_df = pd.read_csv(scopa_path)
            scopa_df = scopa_df[['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.scopa_aut_features]
            clinical_dfs.append(('SCOPA-AUT', scopa_df))
            print(f"  ✓ SCOPA-AUT: {len(scopa_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load SCOPA-AUT: {e}")
        
        # Load Schwab & England (ADL)
        try:
            se_path = self.resolve_path(self.config.data.schwab_england)
            print(f"Loading Schwab & England from: {se_path}")
            se_df = pd.read_csv(se_path)
            se_df = se_df[['PATNO', 'EVENT_ID', 'INFODT'] + self.config.features.schwab_features]
            clinical_dfs.append(('Schwab & England', se_df))
            print(f"  ✓ Schwab & England: {len(se_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load Schwab & England: {e}")
        
        # Merge all clinical data
        if len(clinical_dfs) == 0:
            raise ValueError("Could not load any clinical assessments")
        
        clinical_df = None
        for name, df in clinical_dfs:
            if clinical_df is None:
                clinical_df = df
            else:
                clinical_df = clinical_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
                # Remove duplicate INFODT columns
                clinical_df = clinical_df.loc[:, ~clinical_df.columns.str.endswith('_dup')]
        
        # Compute time since baseline
        clinical_df = self.compute_time_since_baseline(clinical_df)
        
        print(f"✓ Merged clinical data: {len(clinical_df)} visits, {len(clinical_df.columns)-3} features")
        
        return clinical_df
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO', 'EVENT_ID', 'months_since_baseline'
            # Note: specific assessments are optional
        ]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate clinical data"""
        # Check required columns
        if 'PATNO' not in df.columns or 'EVENT_ID' not in df.columns:
            raise ValueError("Missing PATNO or EVENT_ID columns")
        
        # Check for at least one clinical assessment
        clinical_cols = [c for c in df.columns 
                        if c not in ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']]
        if len(clinical_cols) == 0:
            raise ValueError("No clinical assessment columns found")

        required_cols = self.get_required_columns()
        # Find missing columns
        missing_cols = set(required_cols) - set(df.columns.to_list())
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        extra_cols = set(df.columns) - set(required_cols)
        if extra_cols:
            print(f"  ⚠️  Extra columns found (not required): {sorted(extra_cols)}")
        
        print(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        print(f"  Available assessments: {', '.join(clinical_cols)}")
        
        return True

