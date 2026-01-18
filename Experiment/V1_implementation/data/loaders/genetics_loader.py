"""
Task 1: Genetics Data Loader
Load and merge genetics files (consensus, PRS, PCs)
"""

import pandas as pd
import numpy as np
from typing import List
import os

from ..base_loader import StaticDataLoader
from training.config import Config


class GeneticsLoader(StaticDataLoader):
    """
    Loads genetics data from three sources:
    1. Genetic consensus (LRRK2, GBA, SNCA, PRKN, APOE, etc.)
    2. PRS scores (GP2_PGS, META5_PGS, etc.)
    3. Principal components (PC1-PC10 for population structure)
    """
    
    def __init__(self, base_dir: str, config: Config, valid_participants=None):
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
        Load all genetics files and merge them
        
        Returns:
            DataFrame with PATNO + all genetics features
        """
        # Load genetic consensus
        consensus_path = self.resolve_path(self.config.data.genetic_consensus)
        print(f"Loading genetic consensus from: {consensus_path}")
        if not os.path.exists(consensus_path):
            raise FileNotFoundError(f"Genetic consensus file not found: {consensus_path}\n"
                                  f"  Checked: {os.path.abspath(consensus_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        consensus_df = pd.read_csv(consensus_path)
        consensus_df = consensus_df[['PATNO'] + self.config.features.monogenic_variants_features]
        
        # Load PRS scores
        prs_path = self.resolve_path(self.config.data.prs_scores)
        print(f"Loading PRS scores from: {prs_path}")
        if not os.path.exists(prs_path):
            raise FileNotFoundError(f"PRS scores file not found: {prs_path}\n"
                                  f"  Checked: {os.path.abspath(prs_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        prs_df = pd.read_csv(prs_path)
        prs_df = prs_df[['PATNO'] + self.config.features.polygenic_features]
        
        # Load principal components
        pcs_path = self.resolve_path(self.config.data.prs_pcs)
        print(f"Loading PCs from: {pcs_path}")
        if not os.path.exists(pcs_path):
            raise FileNotFoundError(f"Principal components file not found: {pcs_path}\n"
                                  f"  Checked: {os.path.abspath(pcs_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        pcs_df = pd.read_csv(pcs_path)
        pcs_df = pcs_df[['PATNO'] + self.config.features.genetic_principal_components_features]
        
        # Merge all genetics data
        genetics_df = consensus_df.merge(prs_df, on='PATNO', how='left')
        genetics_df = genetics_df.merge(pcs_df, on='PATNO', how='left')
        
        print(f"✓ Loaded genetics data: {len(genetics_df)} patients, {len(genetics_df.columns)-1} features")
        
        return genetics_df

    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        genetics_features = self.config.features.genetics_features  # Property, not a method
        return ['PATNO'] + genetics_features
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate genetics data"""
        # Check PATNO exists
        if 'PATNO' not in df.columns:
            raise ValueError("Missing PATNO column")
        
        # Check for duplicates
        if df['PATNO'].duplicated().any():
            raise ValueError("Duplicate PATNOs found")
        
        # Check at least some genetics data exists
        genetics_cols = [c for c in df.columns if c != 'PATNO']
        if len(genetics_cols) == 0:
            raise ValueError("No genetics columns found")

        required_cols = self.get_required_columns()
        # Find missing columns
        missing_cols = set(required_cols) - set(df.columns.to_list())
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

        extra_cols = set(df.columns) - set(required_cols)
        if extra_cols:
            print(f"  ⚠️  Extra columns found (not required): {sorted(extra_cols)}")

        print(f"✓ Validation passed: {len(df)} unique patients")
        return True

