"""
Task 1: Genetics Data Loader
Load and merge genetics files (consensus, PRS, PCs)
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

from base_loader import StaticDataLoader


class GeneticsLoader(StaticDataLoader):
    """
    Loads genetics data from three sources:
    1. Genetic consensus (LRRK2, GBA, SNCA, PRKN, APOE, etc.)
    2. PRS scores (GP2_PGS, META5_PGS, etc.)
    3. Principal components (PC1-PC10 for population structure)
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
        Load all genetics files and merge them
        
        Returns:
            DataFrame with PATNO + all genetics features
        """
        def resolve_path(file_path):
            """Resolve file path, handling relative paths that start with ../"""
            # If path is absolute, return as-is
            if os.path.isabs(file_path):
                return file_path
            
            # If path starts with ../, resolve relative to current working directory
            if file_path.startswith('../'):
                resolved = os.path.normpath(os.path.abspath(file_path))
            else:
                # Otherwise, resolve relative to base_dir
                resolved = os.path.normpath(os.path.join(self.base_dir, file_path))
            
            return resolved
        
        # Load genetic consensus
        consensus_path = resolve_path(self.config['data'].genetic_consensus)
        print(f"Loading genetic consensus from: {consensus_path}")
        if not os.path.exists(consensus_path):
            raise FileNotFoundError(f"Genetic consensus file not found: {consensus_path}\n"
                                  f"  Checked: {os.path.abspath(consensus_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        consensus_df = pd.read_csv(consensus_path)
        
        # Load PRS scores
        prs_path = resolve_path(self.config['data'].prs_scores)
        print(f"Loading PRS scores from: {prs_path}")
        if not os.path.exists(prs_path):
            raise FileNotFoundError(f"PRS scores file not found: {prs_path}\n"
                                  f"  Checked: {os.path.abspath(prs_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        prs_df = pd.read_csv(prs_path)
        
        # Load principal components
        pcs_path = resolve_path(self.config['data'].prs_pcs)
        print(f"Loading PCs from: {pcs_path}")
        if not os.path.exists(pcs_path):
            raise FileNotFoundError(f"Principal components file not found: {pcs_path}\n"
                                  f"  Checked: {os.path.abspath(pcs_path)}\n"
                                  f"  Base dir: {os.path.abspath(self.base_dir)}")
        pcs_df = pd.read_csv(pcs_path)
        
        # Select relevant columns from consensus
        consensus_cols = ['PATNO', 'LRRK2', 'GBA', 'SNCA', 'PRKN', 'PATHVAR_COUNT']
        if 'APOE' in consensus_df.columns:
            consensus_cols.append('APOE')
        consensus_df = consensus_df[consensus_cols]
        
        # Select PRS columns
        prs_cols = ['PATNO', 'GP2_PGS', 'META5_PGS']
        if 'META5_excl_LRRK2_GBA_PGS' in prs_df.columns:
            prs_cols.append('META5_excl_LRRK2_GBA_PGS')
        prs_df = prs_df[prs_cols]
        
        # Select PC columns (use baseline visit only)
        pcs_df = pcs_df[pcs_df['EVENT_ID'] == 'SC'].copy()
        pc_cols = ['PATNO'] + [f'Genetic_PRS_PC{i}' for i in range(1, 11)]
        pc_cols = [c for c in pc_cols if c in pcs_df.columns]
        pcs_df = pcs_df[pc_cols]
        
        # Merge all genetics data
        genetics_df = consensus_df.merge(prs_df, on='PATNO', how='outer')
        genetics_df = genetics_df.merge(pcs_df, on='PATNO', how='outer')
        
        print(f"✓ Loaded genetics data: {len(genetics_df)} patients, {len(genetics_df.columns)-1} features")
        
        return genetics_df
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO',
            'LRRK2', 'GBA', 'SNCA', 'PRKN', 'PATHVAR_COUNT', 'APOE',
            'GP2_PGS', 'META5_PGS', 'META5_excl_LRRK2_GBA_PGS',
            'Genetic_PRS_PC1', 'Genetic_PRS_PC2', 'Genetic_PRS_PC3', 'Genetic_PRS_PC4', 'Genetic_PRS_PC5',
            'Genetic_PRS_PC6', 'Genetic_PRS_PC7', 'Genetic_PRS_PC8', 'Genetic_PRS_PC9', 'Genetic_PRS_PC10'
        ]
    
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
        
        print(f"✓ Validation passed: {len(df)} unique patients")
        return True


# ============================================================================
# TESTING CODE - Run this file to test the loader
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing GeneticsLoader")
    print("=" * 80)
    
    # This is a test - you'll need to adjust paths to your actual data
    # Add parent directories to path for config import
    import os
    v1_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
    from training.config import get_default_config
    
    config = get_default_config()
    
    # Create loader
    loader = GeneticsLoader("../../ppmi_pd", config)
    
    try:
        # Load data
        genetics_df = loader.load()
        
        # Validate
        loader.validate(genetics_df)
        
        # Get summary
        summary = loader.get_summary(genetics_df)
        print(f"\n✓ Genetics loader working!")
        print(f"  Patients: {summary['n_rows']}")
        print(f"  Features: {summary['n_columns']-1}")
        print(f"  Columns: {', '.join(genetics_df.columns[:10].tolist())}...")
        
        # Show missingness
        print(f"\nMissingness by column:")
        missing = (genetics_df.isnull().sum() / len(genetics_df) * 100).sort_values(ascending=False)
        for col, pct in missing.head(10).items():
            print(f"  {col}: {pct:.1f}%")
            
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("Update the paths in config.py to match your data location")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
