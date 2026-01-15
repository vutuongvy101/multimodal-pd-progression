"""
Data Integrator - Combines all loaders into final dataset
Run this AFTER all individual loaders are working
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try relative imports first, then absolute
try:
    from data.loaders.genetics_loader import GeneticsLoader
    from data.loaders.demographics_loader import DemographicsLoader
    from data.loaders.updrs_loader import UPDRSLoader
    from data.loaders.clinical_loader import ClinicalAssessmentsLoader
    from data.loaders.medication_loader import MedicationLoader
except ImportError:
    # Fall back to direct imports if running from data/ directory
    from loaders.genetics_loader import GeneticsLoader
    from loaders.demographics_loader import DemographicsLoader
    from loaders.updrs_loader import UPDRSLoader
    from loaders.clinical_loader import ClinicalAssessmentsLoader
    from loaders.medication_loader import MedicationLoader


class DataIntegrator:
    """
    Integrates data from all loaders into final dataset
    Handles merging, missing data, slope computation
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dict
        """
        self.config = config
        self.base_dir = config['data'].base_dir
        
        # Initialize all loaders
        print("Initializing loaders...")
        self.genetics_loader = GeneticsLoader(self.base_dir, config)
        self.demographics_loader = DemographicsLoader(self.base_dir, config)
        self.updrs_loader = UPDRSLoader(self.base_dir, config)
        self.clinical_loader = ClinicalAssessmentsLoader(self.base_dir, config)
        self.medication_loader = MedicationLoader(self.base_dir, config)
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from all loaders
        
        Returns:
            Dictionary with 'static' and 'longitudinal' DataFrames
        """
        print("\n" + "=" * 80)
        print("LOADING ALL DATA")
        print("=" * 80)
        
        # Load static data (patient-level)
        print("\n--- Loading Static Data ---")
        genetics_df = self.genetics_loader.load()
        demographics_df = self.demographics_loader.load()
        
        # Merge static data
        static_df = demographics_df.merge(genetics_df, on='PATNO', how='outer')
        print(f"\n✓ Static data merged: {len(static_df)} patients, {len(static_df.columns)-1} features")
        
        # Load longitudinal data (visit-level)
        print("\n--- Loading Longitudinal Data ---")
        updrs_df = self.updrs_loader.load()
        
        try:
            clinical_df = self.clinical_loader.load()
        except:
            print("  ⚠️  Clinical assessments not available, continuing without them")
            clinical_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        try:
            medication_df = self.medication_loader.load()
        except:
            print("  ⚠️  Medication data not available, continuing without it")
            medication_df = pd.DataFrame(columns=['PATNO', 'EVENT_ID'])
        
        # Merge longitudinal data
        longitudinal_df = updrs_df.copy()
        
        if len(clinical_df) > 0:
            longitudinal_df = longitudinal_df.merge(
                clinical_df, 
                on=['PATNO', 'EVENT_ID'], 
                how='outer',
                suffixes=('', '_clinical')
            )
            # Keep the first months_since_baseline if there are duplicates
            if 'months_since_baseline_clinical' in longitudinal_df.columns:
                longitudinal_df['months_since_baseline'] = longitudinal_df['months_since_baseline'].fillna(
                    longitudinal_df['months_since_baseline_clinical']
                )
                longitudinal_df = longitudinal_df.drop('months_since_baseline_clinical', axis=1)
        
        if len(medication_df) > 0:
            longitudinal_df = longitudinal_df.merge(
                medication_df, 
                on=['PATNO', 'EVENT_ID'], 
                how='outer',
                suffixes=('', '_med')
            )
            if 'months_since_baseline_med' in longitudinal_df.columns:
                longitudinal_df['months_since_baseline'] = longitudinal_df['months_since_baseline'].fillna(
                    longitudinal_df['months_since_baseline_med']
                )
                longitudinal_df = longitudinal_df.drop('months_since_baseline_med', axis=1)
        
        print(f"\n✓ Longitudinal data merged: {len(longitudinal_df)} visits, {len(longitudinal_df.columns)-3} features")
        print(f"  Visits per patient: {len(longitudinal_df) / longitudinal_df['PATNO'].nunique():.1f} average")
        
        return {
            'static': static_df,
            'longitudinal': longitudinal_df
        }
    
    def compute_progression_slopes(self, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute empirical progression slopes for each patient
        
        Args:
            longitudinal_df: Longitudinal data with PATNO, months_since_baseline, and UPDRS totals
            
        Returns:
            DataFrame with PATNO and slope for each UPDRS total
        """
        print("\n--- Computing Progression Slopes ---")
        
        min_visits = self.config['training'].min_visits_for_slope
        slopes_data = []
        
        for patno, group in longitudinal_df.groupby('PATNO'):
            # Filter to visits with valid time
            valid_group = group.dropna(subset=['months_since_baseline'])
            
            if len(valid_group) < min_visits:
                continue
            
            times = valid_group['months_since_baseline'].values
            slope_entry = {'PATNO': patno, 'n_visits': len(valid_group)}
            
            # Compute slope for each UPDRS total
            for total in ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']:
                if total in valid_group.columns:
                    scores = valid_group[total].dropna()
                    times_for_total = valid_group.loc[scores.index, 'months_since_baseline'].values
                    
                    if len(scores) >= min_visits:
                        result = linregress(times_for_total, scores.values)
                        slope_entry[f'{total}_slope'] = result.slope
                        slope_entry[f'{total}_r'] = result.rvalue
                        slope_entry[f'{total}_p'] = result.pvalue
            
            if len(slope_entry) > 2:  # Has at least one slope
                slopes_data.append(slope_entry)
        
        slopes_df = pd.DataFrame(slopes_data)
        
        print(f"✓ Computed slopes for {len(slopes_df)} patients")
        for total in ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']:
            slope_col = f'{total}_slope'
            if slope_col in slopes_df.columns:
                n_slopes = slopes_df[slope_col].notna().sum()
                mean_slope = slopes_df[slope_col].mean()
                std_slope = slopes_df[slope_col].std()
                print(f"  {total}: {mean_slope:.3f} ± {std_slope:.3f} points/month (n={n_slopes})")
        
        return slopes_df
    
    def prepare_final_dataset(self) -> Dict:
        """
        Main integration pipeline
        
        Returns:
            Dictionary with 'static', 'longitudinal', 'slopes', and metadata
        """
        print("\n" + "=" * 80)
        print("DATA INTEGRATION PIPELINE")
        print("=" * 80)
        
        # Load all data
        data = self.load_all_data()
        
        # Compute slopes
        slopes_df = self.compute_progression_slopes(data['longitudinal'])
        
        # Filter to only patients with both static and longitudinal data
        patients_with_both = set(data['static']['PATNO']) & set(data['longitudinal']['PATNO'])
        print(f"\n--- Filtering to Complete Cases ---")
        print(f"  Patients with static data: {len(data['static'])}")
        print(f"  Patients with longitudinal data: {data['longitudinal']['PATNO'].nunique()}")
        print(f"  Patients with both: {len(patients_with_both)}")
        
        data['static'] = data['static'][data['static']['PATNO'].isin(patients_with_both)]
        data['longitudinal'] = data['longitudinal'][data['longitudinal']['PATNO'].isin(patients_with_both)]
        
        # Add slopes to output
        data['slopes'] = slopes_df
        
        # Create metadata
        data['metadata'] = {
            'n_patients': len(data['static']),
            'n_visits': len(data['longitudinal']),
            'n_with_slopes': len(slopes_df),
            'static_features': [c for c in data['static'].columns if c != 'PATNO'],
            'longitudinal_features': [c for c in data['longitudinal'].columns 
                                     if c not in ['PATNO', 'EVENT_ID', 'INFODT', 'visit_date']],
            'updrs_totals': [c for c in data['longitudinal'].columns if c in ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']]
        }
        
        print("\n" + "=" * 80)
        print("DATA INTEGRATION COMPLETE")
        print("=" * 80)
        print(f"\nFinal dataset:")
        print(f"  Patients: {data['metadata']['n_patients']}")
        print(f"  Total visits: {data['metadata']['n_visits']}")
        print(f"  Visits per patient: {data['metadata']['n_visits'] / data['metadata']['n_patients']:.1f}")
        print(f"  Patients with slopes: {data['metadata']['n_with_slopes']}")
        print(f"  Static features: {len(data['metadata']['static_features'])}")
        print(f"  Longitudinal features: {len(data['metadata']['longitudinal_features'])}")
        print(f"  UPDRS totals available: {', '.join(data['metadata']['updrs_totals'])}")
        
        return data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Ensure parent directory is in path (already done above, but keep for clarity)
    v1_dir = os.path.dirname(parent_dir)  # Go up one more level from data/ to V1_implementation/
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)
    from training.config import get_default_config
    
    print("=" * 80)
    print("DATA INTEGRATOR - Final Integration")
    print("=" * 80)
    
    config = get_default_config()
    
    # Create integrator
    integrator = DataIntegrator(config)
    
    try:
        # Run full pipeline
        final_data = integrator.prepare_final_dataset()
        
        # Save to CSV for inspection
        print("\n--- Saving Data ---")
        output_dir = config['data'].processed_data_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        final_data['static'].to_csv(f"{output_dir}/static_data.csv", index=False)
        final_data['longitudinal'].to_csv(f"{output_dir}/longitudinal_data.csv", index=False)
        final_data['slopes'].to_csv(f"{output_dir}/slopes_data.csv", index=False)
        
        print(f"  ✓ Saved to {output_dir}/")
        print(f"    - static_data.csv")
        print(f"    - longitudinal_data.csv")
        print(f"    - slopes_data.csv")
        
        print("\n✓ DATA INTEGRATION SUCCESSFUL!")
        print("\nNext steps:")
        print("  1. Inspect the saved CSV files")
        print("  2. Check for missing data patterns")
        print("  3. Proceed to dataset.py to create PyTorch DataLoaders")
        
    except Exception as e:
        print(f"\n❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Make sure all individual loaders work (test each separately)")
        print("  2. Check that file paths in config.py are correct")
        print("  3. Verify data format matches expected structure")
