"""
Task 2: Demographics Data Loader
Load patient demographics (age, sex, education, family history)
"""

import pandas as pd
import numpy as np
from typing import List
import os

from ..base_loader import StaticDataLoader
from training.config import Config


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
            file_path: self.config.data.age_at_visit or self.config.data.socio_economic ...
            merge_columns: Columns to select from the CSV before merging
            how: Type of merge ('left', 'inner', 'outer')
        """
        def resolve_path(file_path):
            if os.path.isabs(file_path):
                return file_path
            if file_path.startswith('../'):
                return os.path.normpath(os.path.abspath(file_path))
            return os.path.normpath(os.path.join(self.base_dir, file_path))
        
        file_path = resolve_path(file_path)
        
        if not os.path.exists(file_path):
            print(f"⚠️  Warning file path not found: {file_path}")
            return df
        
        print(f"  Loading: {file_path}")
        additional_df = pd.read_csv(file_path)
        
        # Ensure PATNO exists
        if 'PATNO' not in additional_df.columns:
            print(f"⚠️  Warning: PATNO not found in {file_path}, skipping merge")
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
            suffixes=('', f'_{file_path}')
        )
        
        print(f"  ✓ Merged {file_path}: {additional_df.shape[0]} records")
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
    
        
    def load(self) -> pd.DataFrame:
        """
        Load demographics data from multiple sources
        
        Returns:
            DataFrame with PATNO + demographics features
        """
        # Load and filter participant status first
        df = self._load_and_filter_participants(
            include_columns=['COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS', 'ENROLL_AGE']
        )

        # Merge Socioeconomic Status
        socio_cols = ['PATNO'] + self.config.features.socioeconomic_features
        df = self.__load_and_merge_data__(
            df, 
            self.config.data.socio_economic,
            merge_columns=socio_cols,
            how='left'
        )

        # Merge Basic Demographics (sex, ethnicity, sexuality, descent)
        demo_cols = ['PATNO'] + self.config.features.basic_demographics_features
        df = self.__load_and_merge_data__(
            df, 
            self.config.data.demographics,
            merge_columns=demo_cols,
            how='left'
        )
        
        # Merge Family History
        family_cols = ['PATNO'] + self.config.features.family_history_features
        df = self.__load_and_merge_data__(
            df, 
            self.config.data.family_history,
            merge_columns=family_cols,
            how='left'
        )

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
        
        # Convert all other columns to numeric
        numeric_cols = [col for col in df.columns if col not in ['PATNO', 'COHORT', 'COHORT_DEFINITION', 'ENROLL_STATUS']]
        for col in numeric_cols:
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

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate demographics data"""
        if 'PATNO' not in df.columns:
            raise ValueError("Missing PATNO column")
        
        unique_patients = df['PATNO'].nunique()
        print(f"✓ Validation passed: {unique_patients} unique patients, {len(df)} total records")
        return True

