"""
Task 5: Medication Data Loader
Load medication information (LEDD, medication history, ON/OFF status)
"""

import pandas as pd
import numpy as np
from typing import List
import sys
import os

# Add parent directory to path for base_loader import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from base_loader import LongitudinalDataLoader


class MedicationLoader(LongitudinalDataLoader):
    """
    Loads medication data:
    - LEDD (levodopa equivalent daily dose)
    - Medication history
    - ON/OFF status (often in UPDRS Part III, but can be separate)
    """
    
    def __init__(self, base_dir: str, config: dict):
        """
        Args:
            base_dir: Base directory for data files
            config: Configuration dict with file paths
        """
        super().__init__(base_dir)
        self.config = config
        
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
                'Use_of_PD_Medication.csv',
                'PD_Medications.csv',
                'Concomitant_Medications.csv'
            ]
            
            med_df = None
            for filename in possible_files:
                try:
                    med_path = f"{self.base_dir}/{filename}"
                    print(f"Trying to load medications from: {med_path}")
                    med_df = pd.read_csv(med_path)
                    print(f"  ✓ Loaded from {filename}")
                    break
                except:
                    continue
            
            if med_df is not None:
                # Select relevant columns
                med_cols = ['PATNO', 'EVENT_ID']
                if 'INFODT' in med_df.columns:
                    med_cols.append('INFODT')
                
                # Look for LEDD or related columns
                for col in ['LEDD', 'LED', 'LEDD_TOTAL', 'LEDDTOT']:
                    if col in med_df.columns:
                        med_cols.append(col)
                
                # Look for medication flags
                for col in ['PDMEDYN', 'ON_OFF', 'PD_MED_USE']:
                    if col in med_df.columns:
                        med_cols.append(col)
                
                med_df = med_df[[c for c in med_cols if c in med_df.columns]]
                
                # Standardize column names
                if 'LED' in med_df.columns and 'LEDD' not in med_df.columns:
                    med_df = med_df.rename(columns={'LED': 'LEDD'})
                if 'LEDD_TOTAL' in med_df.columns:
                    med_df = med_df.rename(columns={'LEDD_TOTAL': 'LEDD'})
                if 'LEDDTOT' in med_df.columns:
                    med_df = med_df.rename(columns={'LEDDTOT': 'LEDD'})
                
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


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing MedicationLoader")
    print("=" * 80)
    
    # Add parent directories to path for config import
    v1_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
    from training.config import get_default_config
    
    config = get_default_config()
    
    # Create loader
    loader = MedicationLoader("../../ppmi_pd", config)
    
    try:
        # Load data
        med_df = loader.load()
        
        # Validate
        loader.validate(med_df)
        
        # Get summary
        summary = loader.get_summary(med_df)
        print(f"\n✓ Medication loader working!")
        print(f"  Total visits: {summary['n_rows']}")
        if len(med_df) > 0:
            print(f"  Unique patients: {med_df['PATNO'].nunique()}")
            print(f"  Features: {summary['n_columns']-3}")
            print(f"  Columns: {', '.join([c for c in med_df.columns if c not in ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline']])}")
            
            # Show LEDD distribution
            if 'LEDD' in med_df.columns:
                print(f"\nLEDD distribution:")
                print(f"  Mean: {med_df['LEDD'].mean():.1f}")
                print(f"  Median: {med_df['LEDD'].median():.1f}")
                print(f"  Range: {med_df['LEDD'].min():.1f} - {med_df['LEDD'].max():.1f}")
                print(f"  N with data: {med_df['LEDD'].notna().sum()}")
        else:
            print("  No medication data available")
            
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("This is OK - medication data is optional")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
