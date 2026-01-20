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
    
    def __load_and_merge_data__(self, df: pd.DataFrame, file_path: str,
                            merge_columns: List[str], 
                            how: str = 'left') -> pd.DataFrame:
        """
        Generic method to load and merge a CSV file with the main dataframe
        
        Args:
            df: Main dataframe to merge into
            file_path: Path to the CSV file to load
            merge_columns: Columns to select from the CSV before merging (includes 'PATNO')
            how: Type of merge ('left', 'inner', 'outer')
            
        Returns:
            Merged dataframe
        """
        import os
        
        def resolve_path(file_path):
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        file_path = resolve_path(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}\n"
                                  f"  Checked: {os.path.abspath(file_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        
        print(f"  Loading: {file_path}")
        additional_df = pd.read_csv(file_path)
        
        # Ensure PATNO exists
        if 'PATNO' not in additional_df.columns:
            raise ValueError(f"PATNO not found in {file_path}")
        
        # Select only merge_columns that exist in the file (filter out PATNO from feature list)
        feature_cols = [col for col in merge_columns if col in additional_df.columns and col != 'PATNO']
        additional_df = additional_df[['PATNO'] + feature_cols].copy()
        
        # Handle multiple rows per patient - take first occurrence
        additional_df = additional_df.groupby('PATNO')[feature_cols].first().reset_index()
        
        # Merge
        df = df.merge(
            additional_df,
            on='PATNO',
            how=how,
            suffixes=('', f'_{file_path}')
        )
        
        print(f"  ✓ Merged {file_path}: {additional_df.shape[0]} records")
        return df
        
    def load(self) -> pd.DataFrame:
        """
        Load all genetics files and merge them
        
        Returns:
            DataFrame with PATNO + all genetics features
        """
        # Load and filter participant status first
        df = self._load_and_filter_participants(
            include_columns=['COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS', 'ENROLL_AGE']
        )

        # Merge genetic consensus
        consensus_cols = ['PATNO'] + self.config.features.monogenic_variants_features
        df = self.__load_and_merge_data__(
            df,
            self.config.data.genetic_consensus,
            merge_columns=consensus_cols,
            how='left'
        )
        
        # Merge PRS scores
        prs_cols = ['PATNO'] + self.config.features.polygenic_features
        df = self.__load_and_merge_data__(
            df,
            self.config.data.prs_scores,
            merge_columns=prs_cols,
            how='left'
        )
        
        # Merge principal components
        pcs_cols = ['PATNO'] + self.config.features.genetic_principal_components_features
        df = self.__load_and_merge_data__(
            df,
            self.config.data.prs_pcs,
            merge_columns=pcs_cols,
            how='left'
        )
        print(df.columns)
        
        print(f"✓ Loaded genetics data: {len(df)} patients, {len(df.columns)-1} features")
        
        return df

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

