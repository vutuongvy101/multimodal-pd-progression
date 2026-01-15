"""
Base class for data loaders - defines the interface
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders
    Each loader is responsible for loading and processing one data source
    """
    
    def __init__(self, base_dir: str):
        """
        Args:
            base_dir: Base directory for all data files
        """
        self.base_dir = base_dir
        
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load and preprocess the data
        
        Returns:
            DataFrame with required columns
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get list of required columns that will be in the output
        
        Returns:
            List of column names
        """
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate that the loaded data meets requirements
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        pass
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with summary information
        """
        return {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'missing_pct': df.isnull().sum() / len(df) * 100
        }


class StaticDataLoader(BaseDataLoader):
    """Base class for loaders that provide static (patient-level) data"""
    
    def get_merge_key(self) -> List[str]:
        """Get columns to merge on (typically ['PATNO'])"""
        return ['PATNO']


class LongitudinalDataLoader(BaseDataLoader):
    """Base class for loaders that provide time-varying (visit-level) data"""
    
    def get_merge_key(self) -> List[str]:
        """Get columns to merge on (typically ['PATNO', 'EVENT_ID'])"""
        return ['PATNO', 'EVENT_ID']
    
    def compute_time_since_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute months_since_baseline for each visit
        
        Args:
            df: DataFrame with PATNO, EVENT_ID, and optionally INFODT
            
        Returns:
            DataFrame with added 'months_since_baseline' column
        """
        if 'INFODT' in df.columns:
            # Use actual dates
            df['visit_date'] = pd.to_datetime(df['INFODT'], errors='coerce')
            
            # Get baseline date for each patient
            baseline_dates = df[df['EVENT_ID'] == 'BL'].groupby('PATNO')['visit_date'].first()
            
            # Compute months since baseline
            df['months_since_baseline'] = df.apply(
                lambda row: (row['visit_date'] - baseline_dates.get(row['PATNO'])).days / 30.44
                if pd.notna(row['visit_date']) and row['PATNO'] in baseline_dates.index
                else None,
                axis=1
            )
        else:
            # Use EVENT_ID mapping
            event_mapping = {
                'SC': 0, 'BL': 0, 'V01': 0, 'V02': 3, 'V03': 6, 'V04': 12,
                'V05': 18, 'V06': 24, 'V07': 30, 'V08': 36, 'V09': 42, 'V10': 48,
                'V11': 54, 'V12': 60, 'V13': 72, 'V14': 84, 'V15': 96, 'V16': 108, 'V17': 120
            }
            df['months_since_baseline'] = df['EVENT_ID'].map(event_mapping)
        
        return df


if __name__ == "__main__":
    print("Base loader classes defined.")
    print("\nLoader hierarchy:")
    print("  BaseDataLoader (abstract)")
    print("    ├─ StaticDataLoader (patient-level data)")
    print("    └─ LongitudinalDataLoader (visit-level data)")
    print("\nEach specific loader should inherit from one of these.")
