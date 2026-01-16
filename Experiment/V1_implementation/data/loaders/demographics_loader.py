"""
Task 2: Demographics Data Loader
Load patient demographics (age, sex, education, family history)
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


class DemographicsLoader(StaticDataLoader):
    """
    Loads demographics from Participant_Status and other demographic files
    Includes: 'PATNO', 'ENROLL_AGE', 'SEX', 'HANDED', 'EDUCYRS',
        'AFICBERB', 'ASHKJEW', 'BASQUE',
        'HOWLIVE', 'GAYLES', 'HETERO', 'BISEXUAL', 'PANSEXUAL', 'ASEXUAL', 'OTHSEXUALITY',
        'HISPLAT', 'RAASIAN', 'RABLACK', 'RAHAWOPI', 'RAINDALS', 'RANOS', 'RAWHITE', 'RAUNKNOWN', 
        'ANYFAMPD', 'BIOMOM', 'BIOMOMPD', 'BIODAD', 'BIODADPD',
        'FULSIB', 'FULBRO', 'FULSIS', 'FULSIBPD', 'FULBROPD', 'FULSISPD',
        'HAFSIB', 'PAHAFSIB', 'MAHAFSIB', 'HAFSIBPD', 'MAHAFSIBPD',
        'PAHAFSIBPD', 'MAGPAR', 'MAGPARPD', 'MAGFATHPD', 'MAGMOTHPD', 'PAGPAR',
        'PAGPARPD', 'PAGFATHPD', 'PAGMOTHPD', 'MATAU', 'MATAUPD', 'PATAU',
        'PATAUPD', 'KIDSNUM', 'KIDSPD', 'DISFAMPD', 'MATCOUS', 'MATCOUSPD',
        'PATCOUS', 'PATCOUSPD']
    """
    
    def __init__(self, base_dir: str, config: dict):
        """
        Args:
            base_dir: Base directory for data files
            config: Configuration dict with file paths
        """
        super().__init__(base_dir)
        self.config = config

    def __filter_valid_participants__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter valid participants from participant_status DataFrame"""

        # Remove participants with null ENROLL_DATE
        df = df.dropna(subset=['ENROLL_DATE'])

        # Keep only participants with valid enrollment status
        valid_statuses = [
            'Complete', 
            'Enrolled', 
            'Withdraw Deceased', 
            'Withdrew'
        ]
        df = df[df['ENROLL_STATUS'].isin(valid_statuses)]

        # Exclude SWEDD cohort
        valid_cohorts = [
            'Healthy Control', 
            "Parkinson's Disease", 
            'Prodromal'
        ]
        df = df[df['COHORT_DEFINITION'].isin(valid_cohorts)]

        # # Renumber cohorts: 1=PD, 2=HC, 3=Prodromal
        # cohort_mapping = {
        #     "Parkinson's Disease": 1,
        #     'Healthy Control': 2,
        #     'Prodromal': 3
        # }
        # df['COHORT'] = df['COHORT_DEFINITION'].map(cohort_mapping)
    
        # Reset index for a clean dataframe
        df.reset_index(drop=True, inplace=True)

        # Select only certain columns for final output
        df = df[[
            'PATNO', 
            'COHORT', 
            'COHORT_DEFINITION', 
            'ENROLL_STATUS',
            'ENROLL_AGE',
        ]].copy()

        return df
        
    def load(self) -> pd.DataFrame:
        """
        Load demographics data
        
        Returns:
            DataFrame with PATNO + demographics features
        """
        def resolve_path(file_path):
            """Resolve file path, handling relative paths that start with ../"""
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        # Load participant status
        status_path = resolve_path(self.config['data'].participant_status)
        print(f"Loading participant status from: {status_path}")
        if not os.path.exists(status_path):
            raise FileNotFoundError(f"Participant status file not found: {status_path}\n"
                                  f"  Checked: {os.path.abspath(status_path)}")
        df = pd.read_csv(status_path)

        df = self.__filter_valid_participants__(df)

        # Select relevant columns
        demo_cols = ['PATNO', 'ENROLL_AGE', 'SEX', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS']
        
        # Add optional columns if they exist
        optional_cols = [   'HANDED', 
                            # Descent    
                            'AFICBERB', 'ASHKJEW', 'BASQUE', 
                            # Sexuality
                            'HOWLIVE', 'GAYLES', 'HETERO', 'BISEXUAL', 'PANSEXUAL', 'ASEXUAL', 'OTHSEXUALITY', 
                            # Ethnicity/Race
                            'HISPLAT', 'RAASIAN', 'RABLACK', 'RAHAWOPI', 'RAINDALS', 'RANOS', 'RAWHITE', 'RAUNKNOWN', 
                            'ANYFAMPD', 
                            # 1st degree family
                            'BIOMOM', 'BIOMOMPD', 'BIODAD', 'BIODADPD',
                            'FULSIB', 'FULBRO', 'FULSIS', 'FULSIBPD', 'FULBROPD', 'FULSISPD',
                            # 2nd degree family
                            'HAFSIB', 'PAHAFSIB', 'MAHAFSIB', 'HAFSIBPD', 'MAHAFSIBPD',
                            'PAHAFSIBPD', 'MAGPAR', 'MAGPARPD', 'MAGFATHPD', 'MAGMOTHPD', 'PAGPAR',
                            'PAGPARPD', 'PAGFATHPD', 'PAGMOTHPD', 'MATAU', 'MATAUPD', 'PATAU',
                            'PATAUPD', 'KIDSNUM', 'KIDSPD', 'DISFAMPD', 'MATCOUS', 'MATCOUSPD',
                            'PATCOUS', 'PATCOUSPD',
                            # Education years
                            'EDUCYRS'
        ]

        for col in optional_cols:
            if col in df.columns:
                demo_cols.append(col)
        
        df = df[demo_cols].copy()
        
        # Process categorical variables
        if 'COHORT' in df.columns:
            # 1=PD, 2=HC, 3=Prodromal
            df['COHORT'] = pd.to_numeric(df['COHORT'], errors='coerce')

        if 'SEX' in df.columns:
            # Encode sex (assuming 0=female, 1=male or similar)
            df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')
        
        # all attributes in optional_cols are binary (0/1) or numeric, so convert them
        for col in optional_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        

        
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
            'COHORT_DEFINITION',
            'ENROLL_STATUS'
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
    
    # Add parent directories to path for config import
    v1_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
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
