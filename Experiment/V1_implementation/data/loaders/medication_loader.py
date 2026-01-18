"""
Task 5: Medication Data Loader
Load medication information (LEDD, medication history, ON/OFF status)
"""

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
        
    def load(self) -> pd.DataFrame:
        """
        Load medication data
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and medication features
        """
        med_dfs = []
        
        # Load LEDD/medication use file
        try:
            # Try common PPMI medication file names
            possible_files = [
                # 'Use_of_PD_Medication.csv',
                # 'PD_Medications.csv',
                # 'Concomitant_Medications.csv'
            ]
            
            med_df = None
            for filename in possible_files:
                try:
                    med_path = self.resolve_path(filename)
                    print(f"Trying to load medications from: {med_path}")
                    med_df = pd.read_csv(med_path)
                    print(f"  ✓ Loaded from {filename}")
                    break
                except:
                    continue
            
            if med_df is not None:
                # Select columns following clinical_loader pattern
                med_cols = ['PATNO', 'EVENT_ID', 'INFODT']
                # Find LEDD column from config
                found_ledd_col = next((col for col in self.config.features.ledd_features if col in med_df.columns), None)
                if found_ledd_col:
                    med_cols.append(found_ledd_col)
                # Find PDMEDYN column from config
                found_pdmedyn_col = next((col for col in self.config.features.pdmedyn_features if col in med_df.columns), None)
                if found_pdmedyn_col:
                    med_cols.append(found_pdmedyn_col)
                
                med_df = med_df[med_cols]
                
                # Rename to standard names if needed
                if found_ledd_col and found_ledd_col != 'LEDD':
                    med_df = med_df.rename(columns={found_ledd_col: 'LEDD'})
                if found_pdmedyn_col and found_pdmedyn_col != 'PDMEDYN':
                    med_df = med_df.rename(columns={found_pdmedyn_col: 'PDMEDYN'})
                
                med_dfs.append(med_df)
                print(f"  ✓ Medication data: {len(med_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load medication data: {e}")
        
        # If no medication data found, create empty dataframe
        if len(med_dfs) == 0:
            print("  ⚠️  No medication files found. Creating empty medication dataframe.")
            # Return minimal dataframe - will be handled during merging
            return pd.DataFrame(columns=['PATNO', 'EVENT_ID', 'months_since_baseline'])
        
        # Merge medication data
        med_df = med_dfs[0]
        for df in med_dfs[1:]:
            med_df = med_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
            med_df = med_df.loc[:, ~med_df.columns.str.endswith('_dup')]
        
        # Compute time since baseline
        if 'INFODT' in med_df.columns or 'EVENT_ID' in med_df.columns:
            med_df = self.compute_time_since_baseline(med_df)
        
        print(f"✓ Medication data: {len(med_df)} visits, {len(med_df.columns)-3} features")
        
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

