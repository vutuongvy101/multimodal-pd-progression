"""
Task 4: Clinical Assessments Loader
Load additional clinical assessments (MoCA, sleep, autonomic, QoL)
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


class ClinicalAssessmentsLoader(LongitudinalDataLoader):
    """
    Loads non-UPDRS clinical assessments:
    - MoCA (cognitive)
    - ESS (sleep)
    - SCOPA-AUT (autonomic)
    - Schwab & England (ADL)
    - Other scales as available
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
        Load clinical assessments and merge them
        
        Returns:
            DataFrame with PATNO, EVENT_ID, months_since_baseline, and clinical scores
        """
        clinical_dfs = []
        
        # Load MoCA (cognitive)
        try:
            moca_path = f"{self.base_dir}/{self.config['data'].moca}"
            print(f"Loading MoCA from: {moca_path}")
            moca_df = pd.read_csv(moca_path)
            # Usually has MOCA total score
            moca_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            if 'MCATOT' in moca_df.columns:
                moca_cols.append('MCATOT')  # MoCA total
            elif 'MOCA' in moca_df.columns:
                moca_cols.append('MOCA')
            moca_df = moca_df[[c for c in moca_cols if c in moca_df.columns]]
            # Rename to consistent MOCA
            if 'MCATOT' in moca_df.columns:
                moca_df = moca_df.rename(columns={'MCATOT': 'MOCA'})
            clinical_dfs.append(('MoCA', moca_df))
            print(f"  ✓ MoCA: {len(moca_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load MoCA: {e}")
        
        # Load ESS (Epworth Sleepiness Scale)
        try:
            ess_path = f"{self.base_dir}/Epworth_Sleepiness_Scale.csv"
            print(f"Loading ESS from: {ess_path}")
            ess_df = pd.read_csv(ess_path)
            ess_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            if 'ESS' in ess_df.columns:
                ess_cols.append('ESS')
            elif 'ESSSCORE' in ess_df.columns:
                ess_cols.append('ESSSCORE')
            ess_df = ess_df[[c for c in ess_cols if c in ess_df.columns]]
            # Rename to ESS
            if 'ESSSCORE' in ess_df.columns:
                ess_df = ess_df.rename(columns={'ESSSCORE': 'ESS'})
            clinical_dfs.append(('ESS', ess_df))
            print(f"  ✓ ESS: {len(ess_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load ESS: {e}")
        
        # Load SCOPA-AUT (autonomic)
        try:
            scopa_path = f"{self.base_dir}/SCOPA-AUT.csv"
            print(f"Loading SCOPA-AUT from: {scopa_path}")
            scopa_df = pd.read_csv(scopa_path)
            scopa_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            # Look for total score
            if 'SCAU_TOTAL' in scopa_df.columns:
                scopa_cols.append('SCAU_TOTAL')
            elif 'SCOPA_AUT' in scopa_df.columns:
                scopa_cols.append('SCOPA_AUT')
            scopa_df = scopa_df[[c for c in scopa_cols if c in scopa_df.columns]]
            # Rename to SCOPA_AUT
            if 'SCAU_TOTAL' in scopa_df.columns:
                scopa_df = scopa_df.rename(columns={'SCAU_TOTAL': 'SCOPA_AUT'})
            clinical_dfs.append(('SCOPA-AUT', scopa_df))
            print(f"  ✓ SCOPA-AUT: {len(scopa_df)} visits")
        except Exception as e:
            print(f"  ⚠️  Could not load SCOPA-AUT: {e}")
        
        # Load Schwab & England (ADL)
        try:
            se_path = f"{self.base_dir}/Schwab_and_England_ADL.csv"
            print(f"Loading Schwab & England from: {se_path}")
            se_df = pd.read_csv(se_path)
            se_cols = ['PATNO', 'EVENT_ID', 'INFODT']
            if 'SENGLAND' in se_df.columns:
                se_cols.append('SENGLAND')
            elif 'SCHWAB_ENGLAND' in se_df.columns:
                se_cols.append('SCHWAB_ENGLAND')
            se_df = se_df[[c for c in se_cols if c in se_df.columns]]
            # Rename to SCHWAB_ENGLAND
            if 'SENGLAND' in se_df.columns:
                se_df = se_df.rename(columns={'SENGLAND': 'SCHWAB_ENGLAND'})
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
        
        print(f"✓ Validation passed: {len(df)} visits across {df['PATNO'].nunique()} patients")
        print(f"  Available assessments: {', '.join(clinical_cols)}")
        
        return True


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing ClinicalAssessmentsLoader")
    print("=" * 80)
    
    # Add parent directories to path for config import
    v1_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
    from training.config import get_default_config
    
    config = get_default_config()
    
    # Create loader
    loader = ClinicalAssessmentsLoader("../../ppmi_pd", config)
    
    try:
        # Load data
        clinical_df = loader.load()
        
        # Validate
        loader.validate(clinical_df)
        
        # Get summary
        summary = loader.get_summary(clinical_df)
        print(f"\n✓ Clinical assessments loader working!")
        print(f"  Total visits: {summary['n_rows']}")
        print(f"  Unique patients: {clinical_df['PATNO'].nunique()}")
        print(f"  Features: {summary['n_columns']-3}")
        
        # Show distributions
        print(f"\nClinical assessments (mean ± std):")
        for col in clinical_df.columns:
            if col not in ['PATNO', 'EVENT_ID', 'INFODT', 'months_since_baseline', 'visit_date']:
                mean_val = clinical_df[col].mean()
                std_val = clinical_df[col].std()
                n_val = clinical_df[col].notna().sum()
                print(f"  {col}: {mean_val:.1f} ± {std_val:.1f} (n={n_val})")
            
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("Update the paths in config.py to match your data location")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
