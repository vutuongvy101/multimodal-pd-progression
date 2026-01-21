#!/usr/bin/env python
"""
Script to run the data integrator and display merged data
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from training.config_v2 import get_default_config
from data.data_integrator_v2 import DataIntegrator

def main():
    print("=" * 80)
    print("DATA INTEGRATOR V2 - MERGING ALL MODALITIES")
    print("=" * 80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = get_default_config()
    print(f"   ✓ Config loaded")
    print(f"   - Base directory: {config.data.base_dir}")
    
    # Initialize data integrator
    print("\n2. Initializing data integrator...")
    integrator = DataIntegrator(config, normalize_features=False)
    
    # Load all data
    print("\n3. Loading and merging all data...")
    data = integrator.prepare_final_dataset()
    
    # Display results
    print("\n" + "=" * 80)
    print("MERGED DATA SUMMARY")
    print("=" * 80)
    
    static_df = data['static']
    longitudinal_df = data['longitudinal']
    slopes_df = data['slopes']
    metadata = data['metadata']
    
    print(f"\nSTATIC DATA (patient-level):")
    print(f"  Patients: {len(static_df)}")
    print(f"  Features: {len(static_df.columns) - 1}")
    print(f"  Columns: {list(static_df.columns)[:5]}..." if len(static_df.columns) > 5 else f"  Columns: {list(static_df.columns)}")
    print(f"\n  First 3 patients (static features):")
    print(static_df.head(3))
    
    print(f"\nLONGITUDINAL DATA (visit-level):")
    print(f"  Total visits: {len(longitudinal_df)}")
    print(f"  Unique patients: {longitudinal_df['PATNO'].nunique()}")
    print(f"  Avg visits per patient: {len(longitudinal_df) / longitudinal_df['PATNO'].nunique():.1f}")
    print(f"  Features: {len(longitudinal_df.columns) - 3}")
    print(f"  Columns: {list(longitudinal_df.columns)[:8]}..." if len(longitudinal_df.columns) > 8 else f"  Columns: {list(longitudinal_df.columns)}")
    print(f"\n  First 5 visits:")
    print(longitudinal_df.head(5)[['PATNO', 'EVENT_ID', 'months_since_baseline'] + [c for c in longitudinal_df.columns if 'TOT' in c][:3]])
    
    print(f"\nPROGRESSION SLOPES (patient-level):")
    print(f"  Patients with slopes: {len(slopes_df)}")
    print(f"  Slope columns: {[c for c in slopes_df.columns if 'slope' in c]}")
    if len(slopes_df) > 0:
        print(f"\n  First 3 patients (slopes):")
        print(slopes_df.head(3)[[c for c in slopes_df.columns if 'PATNO' in c or 'slope' in c]])
    
    print(f"\nMETADATA:")
    for key, value in metadata.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
            if len(value) <= 10:
                print(f"    {value}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("DATA INTEGRATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
