#!/usr/bin/env python3
"""
Generate Updated Documentation from Extracted Attributes

This script reads the extracted attributes JSON and generates
comprehensive markdown documentation.
"""

import json
import os
import pandas as pd
from collections import defaultdict

def load_extracted_data(json_path):
    """Load the extracted attributes JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data_dictionary(dict_path):
    """Load PPMI data dictionary CSV for attribute descriptions.
    
    Returns a dictionary mapping ITM_NAME to DSCR (description).
    """
    description_lookup = {}
    try:
        # Try utf-8 first, then latin-1
        try:
            df = pd.read_csv(dict_path, low_memory=False, encoding='utf-8')
        except:
            df = pd.read_csv(dict_path, low_memory=False, encoding='latin-1')
        
        for _, row in df.iterrows():
            itm_name = str(row.get('ITM_NAME', '')).strip()
            dscr = str(row.get('DSCR', '')).strip()
            
            # Only add if ITM_NAME is not empty and DSCR exists
            if itm_name and itm_name != 'nan' and dscr and dscr != 'nan':
                # Handle duplicates: keep the first non-empty description found
                if itm_name not in description_lookup or not description_lookup[itm_name]:
                    description_lookup[itm_name] = dscr
        
        return description_lookup
    except Exception as e:
        print(f"Warning: Could not load data dictionary from {dict_path}: {e}")
        return {}

def generate_main_documentation(data):
    """Generate the main DATA_ATTRIBUTES_DOCUMENTATION.md file."""
    
    # Category descriptions
    category_descriptions = {
        'Subject_Demographics': 'Demographic and baseline participant information',
        'Participant_Status': 'Enrollment status and cohort information',
        'Imaging': 'Neuroimaging data (MRI, PET, DaTSCAN, DTI, etc.)',
        'Biosample_Inventory': 'Biological sample catalog and availability',
        'Genetic_Status': 'Genetic data, variants, and polygenic risk scores',
        'Roche_Smartphone_App': 'Smartphone-based monitoring and questionnaires',
        'Data___Databases': 'Data dictionaries and code lists',
        'Medical_History': 'Medical conditions, adverse events, physical exams, and clinical assessments',
        'Motor___MDS-UPDRS': 'Motor function assessments including MDS-UPDRS parts I-IV and gait data',
        'Non-motor_Assessments': 'Cognitive, sleep, mood, and other non-motor assessments',
        'Follow_Up_persons_w_Neurologic_Disease': 'FOUND study questionnaires and assessments',
        'PPMI_Online': 'Online assessments and questionnaires',
        'PPMI_Remote_Screening': 'Remote screening data',
        'Root': 'Root-level data files'
    }
    
    md = []
    md.append("# Complete Data Attributes Documentation")
    md.append("")
    md.append("This document provides a comprehensive catalog of all data attributes (columns/variables) across the **parkinsons_tele** and **ppmi_pd** datasets.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Table of Contents")
    md.append("")
    md.append("1. [Parkinson's Telemonitoring Dataset (parkinsons_tele)](#parkinsons-telemonitoring-dataset)")
    md.append("2. [PPMI PD Dataset (ppmi_pd)](#ppmi-pd-dataset)")
    
    # Generate TOC for categories
    for category in sorted(data.keys()):
        if category != 'Root':
            cat_name = category.replace('_', ' ').replace('___', ' & ')
            md.append(f"   - [{cat_name}](#{category.lower().replace('_', '-')})")
    
    md.append("")
    md.append("---")
    md.append("")
    
    # Parkinson's Telemonitoring section (keep existing)
    md.append("## Parkinson's Telemonitoring Dataset")
    md.append("")
    md.append("**Location:** `Database data/parkinsons_tele/`")
    md.append("")
    md.append("### Dataset: `parkinsons_updrs.data`")
    md.append("")
    md.append("**Description:** Oxford Parkinson's Disease Telemonitoring Dataset containing biomedical voice measurements from 42 people with early-stage Parkinson's disease. Contains 5,875 voice recordings with 22 attributes.")
    md.append("")
    md.append("**Attributes:**")
    md.append("")
    md.append("| Attribute Name | Type | Description |")
    md.append("|---------------|------|-------------|")
    md.append("| `subject#` | Integer | Unique identifier for each subject (1-42) |")
    md.append("| `age` | Integer | Subject age in years |")
    md.append("| `sex` | Integer | Subject gender: 0 = male, 1 = female |")
    md.append("| `test_time` | Float | Time since recruitment (days). Integer part = days since recruitment |")
    md.append("| `motor_UPDRS` | Float | Clinician's motor UPDRS score, linearly interpolated |")
    md.append("| `total_UPDRS` | Float | Clinician's total UPDRS score, linearly interpolated |")
    md.append("| `Jitter(%)` | Float | Percentage of variation in fundamental frequency (cycle-to-cycle pitch variation) |")
    md.append("| `Jitter(Abs)` | Float | Absolute jitter measure |")
    md.append("| `Jitter:RAP` | Float | Relative Average Perturbation (RAP) - jitter measure |")
    md.append("| `Jitter:PPQ5` | Float | Pitch Period Perturbation Quotient (5-point) - jitter measure |")
    md.append("| `Jitter:DDP` | Float | Difference of Differences of Periods - jitter measure |")
    md.append("| `Shimmer` | Float | Variation in amplitude (amplitude variation) |")
    md.append("| `Shimmer(dB)` | Float | Shimmer in decibels |")
    md.append("| `Shimmer:APQ3` | Float | Amplitude Perturbation Quotient (3-point) - shimmer measure |")
    md.append("| `Shimmer:APQ5` | Float | Amplitude Perturbation Quotient (5-point) - shimmer measure |")
    md.append("| `Shimmer:APQ11` | Float | Amplitude Perturbation Quotient (11-point) - shimmer measure |")
    md.append("| `Shimmer:DDA` | Float | Difference of Differences of Amplitudes - shimmer measure |")
    md.append("| `NHR` | Float | Noise-to-Harmonics Ratio - measure of ratio of noise to tonal components |")
    md.append("| `HNR` | Float | Harmonics-to-Noise Ratio - measure of ratio of tonal to noise components |")
    md.append("| `RPDE` | Float | Recurrence Period Density Entropy - nonlinear dynamical complexity measure (vocal fold stability) |")
    md.append("| `DFA` | Float | Detrended Fluctuation Analysis - signal fractal scaling exponent (breathiness/turbulent noise) |")
    md.append("| `PPE` | Float | Pitch Period Entropy - nonlinear measure of fundamental frequency variation (impaired pitch control, robust to vibrato) |")
    md.append("")
    md.append("**Key Relationships:**")
    md.append("- Each row = one voice recording")
    md.append("- ~200 recordings per patient")
    md.append("- Target variables: `motor_UPDRS` and `total_UPDRS`")
    md.append("- Predictor variables: 16 biomedical voice measures (Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE)")
    md.append("")
    md.append("---")
    md.append("")
    
    # PPMI PD Dataset section
    md.append("## PPMI PD Dataset")
    md.append("")
    md.append("**Location:** `Database data/ppmi_pd/`")
    md.append("")
    md.append("The PPMI (Parkinson's Progression Markers Initiative) dataset is a comprehensive longitudinal study with multiple data domains. Key identifier: **`PATNO`** (Participant Number) and **`EVENT_ID`** (Visit ID, e.g., BL=Baseline, V01-V24=Visits, R01-R24=Remote visits).")
    md.append("")
    md.append("**Updated Structure (December 2025):**")
    md.append("- **198 CSV files** across **13 major categories**")
    md.append("- **5,081 total attributes** across all datasets")
    md.append("- Comprehensive coverage of clinical, imaging, genetic, biosample, and digital health data")
    md.append("")
    md.append("---")
    md.append("")
    
    # Generate sections for each category
    for category in sorted(data.keys()):
        if category == 'Root':
            continue
            
        cat_name = category.replace('_', ' ').replace('___', ' & ')
        md.append(f"### {cat_name}")
        md.append("")
        md.append(f"**Location:** `ppmi_pd/{category}/`")
        md.append("")
        
        desc = category_descriptions.get(category, f"Data files in the {cat_name} category")
        md.append(f"**Description:** {desc}.")
        md.append("")
        
        files = data[category]
        md.append(f"**Number of Files:** {len(files)}")
        md.append("")
        
        # List key files
        md.append("**Key Files:**")
        md.append("")
        for file_path, file_data in sorted(files.items()):
            file_name = os.path.basename(file_path)
            if 'error' not in file_data:
                num_cols = file_data.get('num_columns', 0)
                num_rows = file_data.get('num_rows', 'Unknown')
                md.append(f"- `{file_name}` - {num_cols} attributes, {num_rows} rows")
            else:
                md.append(f"- `{file_name}` - Error reading file")
        
        md.append("")
        md.append("**Common Attributes (across files in this category):**")
        md.append("")
        
        # Find common attributes
        all_attrs = defaultdict(int)
        for file_path, file_data in files.items():
            if 'error' not in file_data:
                for col in file_data.get('columns', []):
                    all_attrs[col] += 1
        
        # Show most common attributes
        common_attrs = sorted(all_attrs.items(), key=lambda x: x[1], reverse=True)[:10]
        if common_attrs:
            md.append("| Attribute | Frequency | Description |")
            md.append("|-----------|-----------|-------------|")
            for attr, freq in common_attrs:
                if freq > 1:  # Only show if appears in multiple files
                    desc = "Common identifier or metadata field"
                    if attr == 'PATNO':
                        desc = "Participant ID (primary key)"
                    elif attr == 'EVENT_ID':
                        desc = "Visit identifier"
                    elif attr == 'REC_ID':
                        desc = "Record ID (unique identifier)"
                    elif attr == 'INFODT':
                        desc = "Information/assessment date"
                    md.append(f"| `{attr}` | {freq} files | {desc} |")
        
        md.append("")
        md.append("---")
        md.append("")
    
    # Add common identifiers section
    md.append("## Common Identifiers Across PPMI Datasets")
    md.append("")
    md.append("### Primary Keys:")
    md.append("- **`PATNO`** - Participant Number (public identifier, used across most tables)")
    md.append("- **`ALIAS_ID`** - Biorepository internal identifier (used in biosample tables, may need mapping to PATNO)")
    md.append("- **`REC_ID`** - Record ID (unique identifier for each row, often UUID format)")
    md.append("- **`EVENT_ID`** - Visit identifier (BL=Baseline, SC=Screening, V01-V24=Visits, R01-R24=Remote visits, U01-U02=Unscheduled, etc.)")
    md.append("- **`SUB_EVENT_ID`** - Sub-event identifier (for additional visit details)")
    md.append("")
    md.append("### Common Visit Codes (EVENT_ID):")
    md.append("- `SC` - Screening")
    md.append("- `BL` - Baseline")
    md.append("- `V01` through `V24` - Scheduled visits")
    md.append("- `R01` through `R24` - Remote visits")
    md.append("- `U01`, `U02` - Unscheduled visits")
    md.append("- `ST` - Symptomatic Therapy")
    md.append("- `PW` - Premature Withdrawal")
    md.append("- `FNL` - Final Visit")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Summary Statistics")
    md.append("")
    md.append("### parkinsons_tele:")
    md.append("- **1 dataset** with **22 attributes**")
    md.append("- **5,875 rows** (voice recordings)")
    md.append("- **42 participants**")
    md.append("")
    md.append("### ppmi_pd:")
    total_files = sum(len(files) for files in data.values())
    total_attrs = 0
    for category, files in data.items():
        for file_path, file_data in files.items():
            if 'error' not in file_data:
                total_attrs += file_data.get('num_columns', 0)
    
    md.append(f"- **{total_files} datasets** across **{len(data)} major categories**")
    md.append(f"- **{total_attrs} total attributes** across all tables")
    md.append("- **Thousands of participants** (varies by table)")
    md.append("- **Longitudinal data** spanning multiple visits per participant")
    md.append("")
    md.append("---")
    md.append("")
    md.append("*Last Updated: December 2025*")
    md.append("*Data Versions: 14Dec2025 (PPMI), 2009 (Parkinson's Telemonitoring)*")
    md.append("")
    
    return '\n'.join(md)

def generate_details_summary(data, data_dict_lookup=None):
    """Generate the DETAILS_ATTRIBUTES_SUMMARY.md file.
    
    Args:
        data: The extracted attributes JSON data
        data_dict_lookup: Dictionary mapping attribute names (ITM_NAME) to descriptions (DSCR)
    """
    if data_dict_lookup is None:
        data_dict_lookup = {}
    
    md = []
    md.append("# Complete Attributes Summary")
    md.append("")
    md.append("This file contains detailed attribute information extracted from all CSV files.")
    md.append("")
    
    # Calculate totals
    total_files = 0
    total_attributes = 0
    for category, files in data.items():
        for file_path, file_data in files.items():
            if 'error' not in file_data:
                total_files += 1
                total_attributes += file_data.get('num_columns', 0)
    
    md.append(f"**Total Files:** {total_files}")
    md.append(f"**Total Attributes:** {total_attributes}")
    md.append("")
    md.append("---")
    md.append("")
    
    # Generate detailed sections for each category
    for category in sorted(data.keys()):
        cat_name = category.replace('_', ' ').replace('___', ' & ')
        md.append(f"## {cat_name.upper()}")
        md.append("")
        
        files = data[category]
        for file_path, file_data in sorted(files.items()):
            file_name = os.path.basename(file_path)
            md.append(f"### {file_name}")
            md.append("")
            md.append(f"**Path:** `{file_path}`")
            md.append("")
            
            if 'error' in file_data:
                md.append(f"**Error:** {file_data['error']}")
                md.append("")
                continue
            
            num_rows = file_data.get('num_rows', 'Unknown')
            num_cols = file_data.get('num_columns', 0)
            md.append(f"**Rows:** {num_rows}  ")
            md.append(f"**Columns:** {num_cols}")
            md.append("")
            
            md.append("| Attribute | Type | Description | Sample Value |")
            md.append("|-----------|------|-------------|--------------|")
            
            columns = file_data.get('columns', [])
            dtypes = file_data.get('dtypes', {})
            sample_values = file_data.get('sample_values', {})
            enriched = file_data.get('enriched_columns', {})
            
            for col in columns:
                dtype = dtypes.get(col, 'unknown')
                sample = sample_values.get(col, 'N/A')
                desc = 'No description found'
                
                # First, try to get description from data dictionary lookup
                if col in data_dict_lookup:
                    desc = data_dict_lookup[col]
                # Fall back to enriched columns if data dictionary doesn't have it
                elif col in enriched:
                    desc = enriched[col].get('description', 'No description found')
                
                # Truncate long samples
                if isinstance(sample, str) and len(sample) > 50:
                    sample = sample[:50] + "..."
                elif sample is None:
                    sample = 'None'
                
                md.append(f"| `{col}` | {dtype} | {desc} | {sample} |")
            
            md.append("")
    
    return '\n'.join(md)

def main():
    """Main function."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'all_attributes_extracted.json')
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run scan_all_attributes.py first.")
        return
    
    # Load data dictionary for descriptions
    print("Loading data dictionary...")
    dict_path = os.path.join(script_dir, '..', '..', 'Database data', 'ppmi_pd', 
                             'Data___Databases', 'Data_Dictionary_-__Annotated__14Dec2025.csv')
    dict_path = os.path.abspath(dict_path)
    
    data_dict_lookup = {}
    if os.path.exists(dict_path):
        data_dict_lookup = load_data_dictionary(dict_path)
        print(f"Loaded {len(data_dict_lookup)} attribute descriptions from data dictionary")
    else:
        print(f"Warning: Data dictionary not found at {dict_path}")
        print("Proceeding without data dictionary lookup...")
    
    print("\nLoading extracted attributes...")
    data = load_extracted_data(json_path)
    
    print("Generating main documentation...")
    main_doc = generate_main_documentation(data)
    main_path = os.path.join(script_dir, 'DATA_ATTRIBUTES_DOCUMENTATION.md')
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(main_doc)
    print(f"Saved: {main_path}")
    
    print("Generating details summary...")
    details_doc = generate_details_summary(data, data_dict_lookup)
    details_path = os.path.join(script_dir, 'DETAILS_ATTRIBUTES_SUMMARY.md')
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write(details_doc)
    print(f"Saved: {details_path}")
    
    print("\nDocumentation generation complete!")

if __name__ == '__main__':
    main()
