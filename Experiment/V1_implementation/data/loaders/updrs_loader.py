"""
Task 3: UPDRS Data Loader
Load all four UPDRS parts (I, II, III, IV)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import sys
import os

# Add parent directory to path for base_loader import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from base_loader import LongitudinalDataLoader


class UPDRSLoader(LongitudinalDataLoader):
    """
    Loads all four UPDRS parts and merges them
    - Part I: Non-motor experiences (NP1*)
    - Part II: Motor ADL (NP2*)
    - Part III: Motor examination (NP3*)
    - Part IV: Motor complications (NP4*)
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
        Load all UPDRS parts and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and all UPDRS items
        """
        # Initialize with None
        part1_df = None
        part2_df = None
        part3_df = None
        part4_df = None
        
        # Load Part I (Non-motor)
        try:
            part1_path = f"{self.base_dir}/{self.config['data'].updrs_part1}"
            print(f"Loading UPDRS Part I from: {part1_path}")
            part1_df = pd.read_csv(part1_path)
            # Select NP1* columns
            part1_cols = ['PATNO', 'EVENT_ID', 'INFODT'] + [c for c in part1_df.columns if c.startswith('NP1')]
            part1_df = part1_df[part1_cols]
            print(f"  ✓ Part I: {len(part1_df)} visits, {len([c for c in part1_df.columns if c.startswith('NP1')])} features")
        except Exception as e:
            print(f"  ⚠️  Could not load Part I: {e}")
        
        # Load Part II (Motor ADL)
        try:
            part2_path = f"{self.base_dir}/{self.config['data'].updrs_part2}"
            print(f"Loading UPDRS Part II from: {part2_path}")
            part2_df = pd.read_csv(part2_path)
            part2_cols = ['PATNO', 'EVENT_ID', 'INFODT'] + [c for c in part2_df.columns if c.startswith('NP2')]
            part2_df = part2_df[part2_cols]
            print(f"  ✓ Part II: {len(part2_df)} visits, {len([c for c in part2_df.columns if c.startswith('NP2')])} features")
        except Exception as e:
            print(f"  ⚠️  Could not load Part II: {e}")
        
        # Load Part III (Motor Examination)
        try:
            part3_path = f"{self.base_dir}/{self.config['data'].updrs_part3}"
            print(f"Loading UPDRS Part III from: {part3_path}")
            part3_df = pd.read_csv(part3_path)
            # Include PDMEDYN (ON/OFF status) if available
            part3_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            if 'PDMEDYN' in part3_df.columns:
                part3_cols.append('PDMEDYN')
            part3_cols += [c for c in part3_df.columns if c.startswith('NP3')]
            part3_df = part3_df[part3_cols]
            print(f"  ✓ Part III: {len(part3_df)} visits, {len([c for c in part3_df.columns if c.startswith('NP3')])} features")
        except Exception as e:
            print(f"  ⚠️  Could not load Part III: {e}")
        
        # Load Part IV (Motor Complications)
        try:
            part4_path = f"{self.base_dir}/{self.config['data'].updrs_part4}"
            print(f"Loading UPDRS Part IV from: {part4_path}")
            part4_df = pd.read_csv(part4_path)
            part4_cols = ['PATNO', 'EVENT_ID', 'INFODT'] + [c for c in part4_df.columns if c.startswith('NP4')]
            part4_df = part4_df[part4_cols]
            print(f"  ✓ Part IV: {len(part4_df)} visits, {len([c for c in part4_df.columns if c.startswith('NP4')])} features")
        except Exception as e:
            print(f"  ⚠️  Could not load Part IV: {e}")
        
        # Merge all parts
        updrs_df = None
        
        # Start with the first available part
        for df in [part1_df, part2_df, part3_df, part4_df]:
            if df is not None:
                if updrs_df is None:
                    updrs_df = df
                else:
                    updrs_df = updrs_df.merge(df, on=['PATNO', 'EVENT_ID'], how='outer', suffixes=('', '_dup'))
                    # Remove duplicate INFODT columns
                    updrs_df = updrs_df.loc[:, ~updrs_df.columns.str.endswith('_dup')]
        
        if updrs_df is None:
            raise ValueError("Could not load any UPDRS parts")
        
        # Compute time since baseline
        updrs_df = self.compute_time_since_baseline(updrs_df)
        
        print(f"✓ Merged UPDRS data: {len(updrs_df)} visits, {len(updrs_df.columns)-3} features")
        
        return updrs_df
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO', 'EVENT_ID', 'months_since_baseline',
            'NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT'  # At minimum, the totals
        ]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate UPDRS data"""
        # Check required columns
        if 'PATNO' not in df.columns or 'EVENT_ID' not in df.columns:
            raise ValueError("Missing PATNO or EVENT_ID columns")
        
        # Check for at least one UPDRS total
        totals = [c for c in df.columns if c in ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']]
        if len(totals) == 0:
            raise ValueError("No UPDRS total scores found")
        
        # Check time computation
        if 'months_since_baseline' not in df.columns:
            print("⚠️  Warning: months_since_baseline not computed")
        else:
            # Check for reasonable values
            time_range = df['months_since_baseline'].dropna()
            if len(time_range) > 0:
                if time_range.min() < -1 or time_range.max() > 200:
                    print(f"⚠️  Warning: Time range looks unusual: {time_range.min():.1f} - {time_range.max():.1f} months")
        
        print(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        print(f"  Available UPDRS totals: {', '.join(totals)}")
        
        return True


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing UPDRSLoader")
    print("=" * 80)
    
    # Add parent directories to path for config import
    v1_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
    from training.config import get_default_config
    
    config = get_default_config()
    
    # Create loader
    loader = UPDRSLoader("../../ppmi_pd", config)
    
    try:
        # Load data
        updrs_df = loader.load()
        
        # Validate
        loader.validate(updrs_df)
        
        # Get summary
        summary = loader.get_summary(updrs_df)
        print(f"\n✓ UPDRS loader working!")
        print(f"  Total visits: {summary['n_rows']}")
        print(f"  Unique patients: {updrs_df['PATNO'].nunique()}")
        print(f"  Features: {summary['n_columns']-3}")
        
        # Show totals distribution
        print(f"\nUPDRS Totals (mean ± std):")
        for total in ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']:
            if total in updrs_df.columns:
                mean_val = updrs_df[total].mean()
                std_val = updrs_df[total].std()
                n_val = updrs_df[total].notna().sum()
                print(f"  {total}: {mean_val:.1f} ± {std_val:.1f} (n={n_val})")
        
        # Show time distribution
        if 'months_since_baseline' in updrs_df.columns:
            print(f"\nTime distribution:")
            print(f"  Range: {updrs_df['months_since_baseline'].min():.1f} - {updrs_df['months_since_baseline'].max():.1f} months")
            print(f"  Mean: {updrs_df['months_since_baseline'].mean():.1f} months")
            
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("Update the paths in config.py to match your data location")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
