#!/usr/bin/env python3
"""
Scan All PPMI PD Attributes and Generate Documentation

This script scans all CSV files in the updated PPMI PD structure
and generates comprehensive attribute documentation.
"""

import os
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

def get_all_csv_files(base_dir):
    """Find all CSV files recursively in a directory."""
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip hidden directories and __MACOSX
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
        for file in files:
            if file.endswith(('.csv', '.data')):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

def extract_attributes_from_csv(csv_path, max_rows=3):
    """Extract column names and sample data types from a CSV file."""
    try:
        # Read just the header and a few rows
        try:
            df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False, encoding='latin-1')
        
        attributes = {
            'file_path': csv_path,
            'file_name': os.path.basename(csv_path),
            'num_rows': None,
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'sample_values': {}
        }
        
        # Get sample values for each column
        for col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample_val = non_null.iloc[0]
                if isinstance(sample_val, str) and len(sample_val) > 80:
                    sample_val = sample_val[:80] + "..."
                attributes['sample_values'][col] = sample_val
            else:
                attributes['sample_values'][col] = None
        
        # Get row count (faster method)
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f) - 1  # Subtract header
            attributes['num_rows'] = max(0, num_lines)
        except Exception:
            try:
                with open(csv_path, 'r', encoding='latin-1') as f:
                    num_lines = sum(1 for _ in f) - 1
                attributes['num_rows'] = max(0, num_lines)
            except Exception:
                attributes['num_rows'] = "Unknown"
        
        return attributes
    except Exception as e:
        return {
            'file_path': csv_path,
            'file_name': os.path.basename(csv_path),
            'error': str(e)[:200]
        }

def load_data_dictionary(dict_path):
    """Load PPMI data dictionary for attribute descriptions."""
    try:
        try:
            df = pd.read_csv(dict_path, low_memory=False, encoding='utf-8')
        except:
            df = pd.read_csv(dict_path, low_memory=False, encoding='latin-1')
        lookup = {}
        for _, row in df.iterrows():
            mod_name = str(row.get('MOD_NAME', ''))
            itm_name = str(row.get('ITM_NAME', ''))
            dscr = str(row.get('DSCR', ''))
            itm_type = str(row.get('ITM_TYPE', ''))
            if mod_name and itm_name:
                key = (mod_name, itm_name)
                lookup[key] = {
                    'description': dscr,
                    'type': itm_type,
                    'page': str(row.get('PAG_NAME', '')),
                    'mapping_notes': str(row.get('MAPPING_NOTES', ''))
                }
        return lookup
    except Exception as e:
        print(f"Warning: Could not load data dictionary: {e}")
        return {}

def main():
    """Main function."""
    base_dir = os.path.join('..', '..', 'Database data', 'ppmi_pd')
    base_dir = os.path.abspath(base_dir)
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print("=" * 80)
    print("PPMI PD Attribute Scanning")
    print("=" * 80)
    print(f"Base directory: {base_dir}\n")
    
    # Load data dictionary
    print("Loading data dictionary...")
    dict_path = os.path.join(base_dir, 'Data_Dictionary_-__Annotated__23May2025.csv')
    if not os.path.exists(dict_path):
        dict_path = os.path.join(base_dir, 'Data___Databases', 'Data_Dictionary_-__Annotated__14Dec2025.csv')
    
    data_dict = load_data_dictionary(dict_path) if os.path.exists(dict_path) else {}
    print(f"Loaded {len(data_dict)} variable descriptions\n")
    
    # Find all CSV files
    print("Scanning for CSV files...")
    csv_files = get_all_csv_files(base_dir)
    print(f"Found {len(csv_files)} CSV/data files\n")
    
    # Organize by category
    categories = {}
    for csv_file in csv_files:
        rel_path = os.path.relpath(csv_file, base_dir)
        parts = rel_path.split(os.sep)
        
        if len(parts) > 1:
            category = parts[0]
        else:
            category = 'Root'
        
        if category not in categories:
            categories[category] = []
        categories[category].append(csv_file)
    
    # Extract attributes
    print("Extracting attributes from files...")
    print("-" * 80)
    all_attributes = {}
    
    for category, files in sorted(categories.items()):
        print(f"\n{category}: {len(files)} files")
        all_attributes[category] = {}
        
        for csv_file in files:
            rel_path = os.path.relpath(csv_file, base_dir)
            attrs = extract_attributes_from_csv(csv_file)
            
            # Try to enrich with data dictionary
            if data_dict:
                enriched_columns = {}
                for col in attrs.get('columns', []):
                    desc = None
                    for (mod, itm), info in data_dict.items():
                        if itm == col:
                            desc = info.get('description', '')
                            break
                    
                    enriched_columns[col] = {
                        'dtype': attrs.get('dtypes', {}).get(col, 'unknown'),
                        'description': desc or 'No description found',
                        'sample_value': attrs.get('sample_values', {}).get(col)
                    }
                attrs['enriched_columns'] = enriched_columns
            
            all_attributes[category][rel_path] = attrs
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(output_dir, 'all_attributes_extracted.json')
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_attributes, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {json_file}")
    
    # Generate summary statistics
    total_files = 0
    total_attributes = 0
    
    for category, files in all_attributes.items():
        for file_path, attrs in files.items():
            if 'error' not in attrs:
                total_files += 1
                total_attributes += attrs.get('num_columns', 0)
    
    print(f"\nSummary: {total_files} files, {total_attributes} total attributes")
    print(f"Scan completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_attributes

if __name__ == '__main__':
    main()
