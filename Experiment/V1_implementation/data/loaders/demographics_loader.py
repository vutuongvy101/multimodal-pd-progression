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

    def __load_and_merge_data__(self, df: pd.DataFrame, config_key: str, 
                            merge_columns: List[str], 
                            how: str = 'left') -> pd.DataFrame:
        """
        Generic method to load and merge a CSV file with the main dataframe
        
        Args:
            df: Main dataframe to merge into
            config_key: Key in config['data'] (e.g., 'family_history', 'socio_economic')
            merge_columns: Columns to select from the CSV before merging
            how: Type of merge ('left', 'inner', 'outer')
        """
        def resolve_path(file_path):
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        file_path = resolve_path(getattr(self.config['data'], config_key))
        
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: {config_key} file not found: {file_path}")
            return df
        
        print(f"  Loading {config_key} from: {file_path}")
        additional_df = pd.read_csv(file_path)
        
        # Ensure PATNO exists
        if 'PATNO' not in additional_df.columns:
            print(f"⚠️  Warning: PATNO not found in {config_key}, skipping merge")
            return df
        
        # Handle multiple rows per patient (e.g., multiple visits)
        # Take first occurrence or aggregate as needed
        # Filter out PATNO since it's the groupby key, not an aggregation column
        merge_columns = [col for col in merge_columns if col in additional_df.columns and col != 'PATNO']
        additional_df = additional_df.groupby('PATNO')[merge_columns].first().reset_index()
        
        # Merge
        df = df.merge(
            additional_df,
            on='PATNO',
            how=how,
            suffixes=('', f'_{config_key}')
        )
        
        print(f"  ✓ Merged {config_key}: {additional_df.shape[0]} records")
        return df
    
    def __aggregate_by_patient__(self, df: pd.DataFrame, 
                             agg_columns: dict) -> pd.DataFrame:
        """
        Aggregate multiple rows per patient
        
        Args:
            df: DataFrame with multiple rows per patient
            agg_columns: Dict mapping column names to aggregation functions
                        e.g., {'AGE': 'first', 'SCORE': 'mean', 'COUNT': 'max'}
        """
        return df.groupby('PATNO').agg(agg_columns).reset_index()
    
    def __load_and_merge_age_at_visit__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and merge age at visit file without aggregation.
        Each row in age_at_visit (with different EVENT_ID) is joined to all demographic columns.
        Result: multiple rows per PATNO (one for each EVENT_ID/visit)
        
        Args:
            df: Main demographics dataframe
            
        Returns:
            Dataframe with age_at_visit rows, each containing all demographic columns
        """
        def resolve_path(file_path):
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        file_path = resolve_path(self.config['data'].age_at_visit)
        
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: age_at_visit file not found: {file_path}")
            return df
        
        print(f"  Loading age_at_visit from: {file_path}")
        age_df = pd.read_csv(file_path)
        
        if 'PATNO' not in age_df.columns:
            print(f"⚠️  Warning: PATNO not found in age_at_visit, skipping merge")
            return df
        
        # Select only PATNO and other age visit columns (do not aggregate)
        age_cols_to_keep = ['PATNO', 'EVENT_ID', 'AGE_AT_VISIT']
        age_cols_available = [col for col in age_cols_to_keep if col in age_df.columns]
        age_df = age_df[age_cols_available].copy()
        
        # Merge: each age_at_visit row gets all demographic columns from df
        # Result: multiple rows per PATNO (one for each EVENT_ID)
        df = df.merge(
            age_df,
            on='PATNO',
            how='left'
        )
        
        print(f"  ✓ Merged age_at_visit: {len(age_df)} visit records")
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

        # Merge Socioeconomic Status
        socio_cols = ['PATNO', 'EDUCYRS']
        df = self.__load_and_merge_data__(
            df, 
            'socio_economic', 
            merge_columns=socio_cols,
            how='left'
        )

        # Merge Demographics file (if has additional columns not in participant_status)
        demo_cols = [   'PATNO', 'SEX', 'HANDED', 
                        # Descent    
                        'AFICBERB', 'ASHKJEW', 'BASQUE', 
                        # Sexuality
                        'HOWLIVE', 'GAYLES', 'HETERO', 'BISEXUAL', 'PANSEXUAL', 'ASEXUAL', 'OTHSEXUALITY', 
                        # Ethnicity/Race
                        'HISPLAT', 'RAASIAN', 'RABLACK', 'RAHAWOPI', 'RAINDALS', 'RANOS', 'RAWHITE', 'RAUNKNOWN' ]  
        df = self.__load_and_merge_data__(
            df, 
            'demographics', 
            merge_columns=demo_cols,
            how='left'
        )
        
        # Merge Family History
        family_cols = [ 'PATNO', 'ANYFAMPD', 
                        # 1st degree family
                        'BIOMOM', 'BIOMOMPD', 'BIODAD', 'BIODADPD',
                        'FULSIB', 'FULBRO', 'FULSIS', 'FULSIBPD', 'FULBROPD', 'FULSISPD', 
                        'KIDSPD',
                        # 2nd degree family
                        'HAFSIB', 'PAHAFSIB', 'MAHAFSIB', 'HAFSIBPD', 'MAHAFSIBPD',
                        'PAHAFSIBPD', 'MAGPAR', 'MAGPARPD', 'MAGFATHPD', 'MAGMOTHPD', 'PAGPAR',
                        'PAGPARPD', 'PAGFATHPD', 'PAGMOTHPD', 'MATAU', 'MATAUPD', 'PATAU',
                        'PATAUPD', 'MATCOUS', 'MATCOUSPD',
                        'PATCOUS', 'PATCOUSPD',
                        'DISFAMPD']
        df = self.__load_and_merge_data__(
            df, 
            'family_history', 
            merge_columns=family_cols,
            how='left'
        )
        
        # Merge Age at Visit
        df = self.__load_and_merge_age_at_visit__(df)

        # Select relevant columns
        demo_cols = ['PATNO', 'ENROLL_AGE', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS']
        
        # Add optional columns if they exist
        optional_cols = [col for col in df.columns if col not in ['PATNO', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS']]
        df = df[['PATNO', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS'] + optional_cols].copy()

        for col in optional_cols:
            if col in df.columns:
                demo_cols.append(col)
        
        df = df[demo_cols].copy()
        
        # Check for duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            print(f"⚠️  Warning: Found duplicate columns: {duplicate_cols}")
            # Remove duplicates by keeping first occurrence
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
        
        # Process categorical variables
        if 'COHORT' in df.columns:
            # 1=PD, 2=HC, 4=Prodromal
            df['COHORT'] = pd.to_numeric(df['COHORT'], errors='coerce')

        if 'SEX' in df.columns:
            # Encode sex (assuming 0=female, 1=male or similar)
            df['SEX'] = pd.to_numeric(df['SEX'], errors='coerce')
        
        # all attributes in optional_cols are binary (0/1) or numeric, so convert them
        for col in optional_cols:
            if col in df.columns:
                try:
                    # Check if column is a Series (not duplicate column names)
                    if isinstance(df[col], pd.Series):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"⚠️  Warning: Could not convert column '{col}' to numeric: {e}")
                    continue
        
        print(f"✓ Loaded demographics: {df['PATNO'].nunique()} unique patients, {len(df.columns)-1} columns")
        
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
    

    # This will not work since Age_at_Visit introduces multiple rows per PATNO
    # def validate(self, df: pd.DataFrame) -> bool:
    #     """Validate demographics data"""
    #     # Check PATNO exists
    #     if 'PATNO' not in df.columns:
    #         raise ValueError("Missing PATNO column")
        
    #     # Check for duplicates
    #     if df['PATNO'].duplicated().any():
    #         raise ValueError("Duplicate PATNOs found")
        
    #     # Check ENROLL_AGE is reasonable
    #     if 'ENROLL_AGE' in df.columns:
    #         age_range = df['ENROLL_AGE'].dropna()
    #         if len(age_range) > 0:
    #             if age_range.min() < 18 or age_range.max() > 100:
    #                 print(f"⚠️  Warning: Age range looks unusual: {age_range.min():.1f} - {age_range.max():.1f}")
        
    #     print(f"✓ Validation passed: {len(df)} unique patients")
    #     return True

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate demographics data (handles multiple rows per PATNO from age_at_visit)"""
        if 'PATNO' not in df.columns:
            raise ValueError("Missing PATNO column")
        
        unique_patients = df['PATNO'].nunique()
        print(f"✓ Validation passed: {unique_patients} unique patients, {len(df)} total records")
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
        unique_patients = demo_df['PATNO'].nunique()
        print(f"\n✓ Demographics loader working!")
        print(f"  Unique Patients: {unique_patients}")
        print(f"  Records: {summary['n_rows']}")
        print(f"  Features: {summary['n_columns']-1}")
        print(f"  Columns: {', '.join(demo_df.columns.tolist())}")
        
        # Show distributions
        if 'ENROLL_AGE' in demo_df.columns:
            print(f"\nAge distribution:")
            print(f"  Mean: {demo_df['ENROLL_AGE'].mean():.1f}")
            print(f"  Range: {demo_df['ENROLL_AGE'].min():.1f} - {demo_df['ENROLL_AGE'].max():.1f}")
        
        if 'COHORT' in demo_df.columns:
            print(f"\nCohort distribution (unique patients):")
            # Count cohorts by unique PATNO (one patient counted once)
            unique_cohort = demo_df[['PATNO', 'COHORT_DEFINITION']].drop_duplicates(subset='PATNO')
            print(unique_cohort['COHORT_DEFINITION'].value_counts())
            
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("Update the paths in config.py to match your data location")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
