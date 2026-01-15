"""
Task 2: Demographics Data Loader
Load patient demographics (age, sex, education, family history)
"""

import pandas as pd
import numpy as np
from typing import List
import sys
sys.path.append('..')
from base_loader import StaticDataLoader


class DemographicsLoader(StaticDataLoader):
    """
    Loads demographics from Participant_Status and other demographic files
    Includes: PATNO, SEX, RACE, EDUCYRS, ENROLL_AGE, family history
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
        Load demographics data
        
        Returns:
            DataFrame with PATNO + demographics features
        """
        # Load participant status
        status_path = f"{self.base_dir}/{self.config['data'].participant_status}"
        print(f"Loading participant status from: {status_path}")
        df = pd.read_csv(status_path)
        
        # Select relevant columns
        demo_cols = ['PATNO', 'ENROLL_AGE', 'COHORT', 'COHORT_DEFINITION']
        
        # Add optional columns if they exist
        optional_cols = ['SEX', 'RACE', 'EDUCYRS', 'ENROLL_STATUS', 
                        'ENRLHPSM', 'ENRLRBD', 'ENRLLRRK2', 'ENRLSNCA', 'ENRLGBA']
        for col in optional_cols:
            if col in df.columns:
                demo_cols.append(col)
        
        df = df[demo_cols].copy()
        
        # Process categorical variables
        if 'SEX' in df.columns:
            # Encode sex (assuming 0=female, 1=male or similar)
            df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')
        
        if 'RACE' in df.columns:
            # Encode race as numeric
            df['RACE'] = pd.to_numeric(df['RACE'], errors='coerce')
        
        if 'COHORT' in df.columns:
            # 1=PD, 2=HC, 3=SWEDD, etc.
            df['COHORT'] = pd.to_numeric(df['COHORT'], errors='coerce')
        
        # TODO: Add family history if available from separate file
        
        print(f"✓ Loaded demographics: {len(df)} patients, {len(df.columns)-1} features")
        
        return df
    
    def get_required_columns(self) -> List[str]:
        """Required columns in output"""
        return [
            'PATNO',
            'ENROLL_AGE',
            'SEX',
            'COHORT',
            'COHORT_DEFINITION'
        ]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate demographics data"""
        # Check PATNO exists
        if 'PATNO' not in df.columns:
            raise ValueError("Missing PATNO column")
        
        # Check for duplicates
        if df['PATNO'].duplicated().any():
            raise ValueError("Duplicate PATNOs found")
        
        # Check ENROLL_AGE is reasonable
        if 'ENROLL_AGE' in df.columns:
            age_range = df['ENROLL_AGE'].dropna()
            if len(age_range) > 0:
                if age_range.min() < 18 or age_range.max() > 100:
                    print(f"⚠️  Warning: Age range looks unusual: {age_range.min():.1f} - {age_range.max():.1f}")
        
        print(f"✓ Validation passed: {len(df)} unique patients")
        return True


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Testing DemographicsLoader")
    print("=" * 80)
    
    from training.config import get_default_config
    
    config = get_default_config()
    
    # Create loader
    loader = DemographicsLoader("../../ppmi_pd", config)
    
    try:
        # Load data
        demo_df = loader.load()
        
        # Validate
        loader.validate(demo_df)
        
        # Get summary
        summary = loader.get_summary(demo_df)
        print(f"\n✓ Demographics loader working!")
        print(f"  Patients: {summary['n_rows']}")
        print(f"  Features: {summary['n_columns']-1}")
        print(f"  Columns: {', '.join(demo_df.columns.tolist())}")
        
        # Show distributions
        if 'ENROLL_AGE' in demo_df.columns:
            print(f"\nAge distribution:")
            print(f"  Mean: {demo_df['ENROLL_AGE'].mean():.1f}")
            print(f"  Range: {demo_df['ENROLL_AGE'].min():.1f} - {demo_df['ENROLL_AGE'].max():.1f}")
        
        if 'COHORT' in demo_df.columns:
            print(f"\nCohort distribution:")
            print(demo_df['COHORT_DEFINITION'].value_counts())
            
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("Update the paths in config.py to match your data location")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
