"""
Base class for data loaders - defines the interface
"""

from abc import ABC, abstractmethod
import pandas as pd
import os
import warnings
from typing import Dict, List, Optional, Any

from training.config import Config


# ============================================================================
# Standalone utility functions for participant filtering
# Shared across BaseDataLoader and DataIntegrator
# ============================================================================

def filter_valid_participants(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is a standardized function used by all loaders to ensure consistent
    participant filtering from participant_status DataFrame
    
    Args:
        df: DataFrame from participant_status file
        
    Returns:
        Filtered DataFrame with valid participants only
    """
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
    
    return df


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders
    Each loader is responsible for loading and processing one data source
    """
    
    def __init__(self, base_dir: str, config: Config, valid_participants: Optional[pd.DataFrame] = None):
        """
        Args:
            base_dir: Base directory for all data files
            valid_participants: Optional pre-filtered participant DataFrame.
                              If provided, will be used instead of loading participant_status.
                              Should have at least 'PATNO' column.
        """
        self.base_dir = base_dir
        self._valid_participants_cache = valid_participants
        self.config = config  # Centralized config - accessible by all loaders
        
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
    
    def resolve_path(self, file_path: str) -> str:
        """
        Resolve file path, handling relative paths that start with ../
        
        Args:
            file_path: File path from config (can be relative or absolute)
            
        Returns:
            Resolved absolute path
            
        Examples:
            - Absolute path: returns as-is
            - Path starting with ../: resolves relative to current working directory
            - Other relative path: resolves relative to base_dir
        """
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
    
    def _load_and_filter_participants(self, include_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load participant_status file and filter to valid participants(cached).
        All loaders should use this method to ensure consistent participant filtering.

        Args:
            include_columns: Optional list of columns to include from participant_status.
        
        Returns:
            DataFrame with valid participants (PATNO + optionally include_columns)
        """
        if self._valid_participants_cache is None:
            if not hasattr(self, 'config') or self.config is None:
                raise ValueError("config must be provided during initialization")
            
            participant_status_path = self.resolve_path(self.config.data.participant_status)
            print(f"Loading participant status from: {participant_status_path}")
            
            if not os.path.exists(participant_status_path):
                raise FileNotFoundError(
                    f"Participant status file not found: {participant_status_path}\n"
                    f"  Checked: {os.path.abspath(participant_status_path)}"
                )
            
            df = pd.read_csv(participant_status_path)
            df = filter_valid_participants(df)
            self._valid_participants_cache = df.copy()
        else:
            df = self._valid_participants_cache.copy()
        
        # Select columns to return
        cols = ['PATNO'] + [c for c in (include_columns or []) if c in df.columns and c != 'PATNO']
        return df[cols].copy()


class StaticDataLoader(BaseDataLoader):
    """Base class for loaders that provide static (patient-level) data"""


class LongitudinalDataLoader(BaseDataLoader):

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """Load the raw longitudinal data (subclasses implement)."""
        raise NotImplementedError

    def load(self) -> pd.DataFrame:
        """
        Standard longitudinal pipeline:
        - load raw
        - filter to valid participants
        - compute months_since_baseline
        """
        df = self._load_raw()

        # Allow loaders to return empty/optional data without crashing the pipeline
        if df is None or len(df) == 0:
            return df if df is not None else pd.DataFrame()

        df = self.filter_by_valid_participants(df)

        if "INFODT" in df.columns:
            df = self.compute_time_since_baseline(df)

        return df

    
    def filter_by_valid_participants(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter longitudinal data to only include valid participants.
        Uses participant_status to determine valid PATNOs (or cached participants).
        
        Args:
            df: Longitudinal DataFrame with PATNO column
            
        Returns:
            DataFrame filtered to only include valid participants
        """
        # Get valid participant IDs (uses cache if available)
        # Uses self.config from base class - single source of truth
        valid_participants_df = self._load_and_filter_participants()
        
        valid_patnos = valid_participants_df['PATNO'].unique()
        
        # Filter to only valid participants
        filtered_df = df[df['PATNO'].isin(valid_patnos)].copy()
        
        print(f"  Filtered to {len(filtered_df)} visits from {filtered_df['PATNO'].nunique()} valid participants")
        
        return filtered_df
    
    def compute_time_since_baseline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute months_since_baseline for each visit
        
        Args:
            df: DataFrame with PATNO, EVENT_ID, and INFODT (required)
            
        Returns:
            DataFrame with added 'months_since_baseline' column
            
        Raises:
            ValueError: If INFODT column is missing
        """
        if 'INFODT' not in df.columns:
            raise ValueError(
                "INFODT column is required to compute months_since_baseline. "
                "The DataFrame must contain visit dates in the INFODT column."
            )
        
        # Use actual dates - try ISO format first (YYYY-MM-DD) to avoid warnings
        # PPMI data typically uses ISO format in medical databases
        # If format doesn't match, errors='coerce' will return NaT for non-matching values
        # Then fall back to automatic parsing for any remaining NaT values
        df['visit_date'] = pd.to_datetime(df['INFODT'], format='%Y-%m-%d', errors='coerce')
        # If ISO format didn't match, try automatic parsing for any remaining NaT values
        if df['visit_date'].isna().any():
            mask = df['visit_date'].isna()
            # Suppress warnings for automatic date parsing fallback
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Could not infer format.*', category=UserWarning)
                df.loc[mask, 'visit_date'] = pd.to_datetime(df.loc[mask, 'INFODT'], errors='coerce')
        
        # Get baseline date for each patient
        baseline_dates = df[df['EVENT_ID'] == 'BL'].groupby('PATNO')['visit_date'].first()
        
        # Compute months since baseline
        df['months_since_baseline'] = df.apply(
            lambda row: (row['visit_date'] - baseline_dates.get(row['PATNO'])).days / 30.44
            if pd.notna(row['visit_date']) and row['PATNO'] in baseline_dates.index
            else None,
            axis=1
        )

        # Guardrail: some records can have visit dates earlier than the recorded baseline date,
        # which yields negative months. For downstream modeling/tests we treat those as baseline (0).
        df['months_since_baseline'] = pd.to_numeric(df['months_since_baseline'], errors='coerce')
        df.loc[df['months_since_baseline'].notna(), 'months_since_baseline'] = df.loc[
            df['months_since_baseline'].notna(), 'months_since_baseline'
        ].clip(lower=0.0)
        
        return df
