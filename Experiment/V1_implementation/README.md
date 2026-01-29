# V1: Multimodal Longitudinal Transformer for Parkinson's Disease

A simplified, production-ready deep learning model for PD progression prediction using PPMI data.

---

## Table of Contents

1. [Overview](#overview)
2. [Why V1?](#why-v1)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Data Pipeline](#data-pipeline)
7. [Model Components](#model-components)
8. [Training](#training)
9. [Configuration](#configuration)
10. [UPDRS Structure](#updrs-structure)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps](#next-steps)

---

## Overview

**V1** is a Transformer-based model that predicts Parkinson's disease progression by:
- Fusing multimodal data (genetics, demographics, clinical assessments)
- Learning longitudinal patterns across patient visits
- Handling missing data explicitly
- Predicting both next-visit outcomes and long-term progression slopes

### Key Features

| Feature | Description |
|---------|-------------|
| **Longitudinal Fusion** | Transformer learns from sequences of visits |
| **Multimodal** | Combines genetics + demographics + motor + non-motor + medication |
| **Medication-Aware** | Explicit ON/OFF status and LEDD as features |
| **Missingness Handling** | Built-in masks for each modality |
| **Time-Aware** | Continuous time encoding (actual months, not indices) |
| **Multi-Objective** | Predicts next-visit UPDRS totals (NP1RTOT, NP2PTOT, NP3TOT, NP4TOT) + patient-level progression slopes (all 4 UPDRS totals) |

---

## Why V1?

V1 is a **strategic simplification** of a more complex architecture. The goal is to establish a working baseline before adding complexity.

### Comparison with Full Model

| Component | Complex Model | V1 (Simplified) | Benefit |
|-----------|--------------|-----------------|---------|
| Latent state | Variational (μ, σ) + KL | Direct embeddings | Easier training |
| Medication | Mixed-effects decomposition | Explicit features | Clear interpretation |
| Loss function | ELBO + survival + MSE | 2× MSE only | Simpler debugging |
| Training | VAE + multiple heads | Standard supervised | Faster convergence |

### What V1 Achieves

✅ Learns longitudinal patterns from irregular visits  
✅ Produces medication-aware predictions  
✅ Handles systematic missingness in clinical data  
✅ Provides a validated baseline for comparison  

### What V1 Defers (for V2+)

❌ Latent disease state separation from symptoms  
❌ Individual medication responsiveness modeling  
❌ Time-to-event / survival predictions  
❌ Uncertainty quantification  

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VISIT INDEX (master timeline)               │
│  PATNO | EVENT_ID | months_since_baseline | visit_order | Δt    │
│  Defines WHEN visits happen - independent of any modality       │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐    ┌─────────────────┐   ┌─────────────────┐
│  STATIC DATA    │    │  TIME-VARYING   │   │  MEDICATION     │
│  • Genetics     │    │  (per visit)    │   │  CONTEXT        │
│  • Demographics │    │  • Part I-IV    │   │  • LEDD         │
│                 │    │  • MoCA, ESS    │   │  • ON/OFF       │
└─────────────────┘    └─────────────────┘   └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                     MODALITY EMBEDDINGS (per modality)           │
│  Each modality: MLP([values, input_missingness_mask]) → embedding  │
│  Enabled modalities configurable: ['static','motor','nonmotor',  │
│                                    'medication']                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     VISIT TOKEN FORMATION                        │
│  v_t = Concat(enabled modality embeddings)                       │
│  v_t = Project(v_t) + SinusoidalTimeEncoding(months, delta_t)    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                   TRANSFORMER ENCODER                            │
│  4 layers, 8 heads, d_model=256                                  │
│  Attention mask for variable-length sequences                    │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   NEXT-VISIT HEAD       │     │   SLOPE HEAD            │
│   Input: h_t            │     │   Input: pool(H)        │
│   Output: UPDRS totals  │     │   Output: slopes        │
│   Loss: MSE with        │     │   Loss: MSE             │
│   LABEL AVAILABILITY    │     │                         │
│   MASK per target       │     │                         │
└─────────────────────────┘     └─────────────────────────┘

Total Loss = L_next_visit + λ × L_slope  (λ = 0.2)
```

### Two-Mask System

The model uses **two different types of masks** to handle PPMI-style irregular data:

```
┌─────────────────────────────────────────────────────────────────┐
│                    TWO SEPARATE MASKS                           │
├─────────────────────────────────────────────────────────────────┤
│  A) INPUT MISSINGNESS MASKS (per modality)                      │
│     • "Is feature X observed at visit t?"                       │
│     • Shape: [batch, seq_len, n_features] per modality          │
│     • Used by: modality embedding MLPs                          │
│     • 1 = missing, 0 = present                                  │
├─────────────────────────────────────────────────────────────────┤
│  B) LABEL AVAILABILITY MASKS (per target)                       │
│     • "Does visit t+1 have a valid UPDRS total for this task?"  │
│     • Shape: [batch, seq_len, n_targets]                        │
│     • Used by: loss computation only                            │
│     • 1 = label present, 0 = label missing                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: You do NOT filter timelines based on missing modalities. You keep all visits and mask out loss contributions where labels are missing.

**Example**:
- If your task is "predict next-visit motor" but motor at V2 is missing:
  - The V1→V2 prediction should NOT contribute to motor loss
  - But that same V1→V2 can still contribute to medication-related predictions
  - You mask the loss, not the timeline

### Modality Selection (Ablation Studies)

Modalities can be enabled/disabled via config for ablation experiments:

```python
# Train with all modalities (default)
config.model.enabled_modalities = ['static', 'motor', 'nonmotor', 'medication']

# Train with only static + motor (ablation)
config.model.enabled_modalities = ['static', 'motor']

# Train without medication context
config.model.enabled_modalities = ['static', 'motor', 'nonmotor']
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- PPMI data access

### 1. Activate Virtual Environment

You can use either **conda** or **pyenv (virtualenv)**.

#### Option A: conda

```bash
conda activate sri_hri
```

#### Option B: pyenv (virtualenv)

```bash
# Example (adjust python version as needed)
pyenv install -s 3.10.13

# Create and activate a dedicated virtualenv for this project
pyenv virtualenv 3.10.13 pd-v1
pyenv activate pd-v1

python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# From git/pdS22025/Experiment/V1_implementation
python -m pip install -r ../../requirements.txt
```

### 3. Install Package in Editable Mode (Optional but Recommended)

Install the V1 package in editable mode so that imports work properly and IDE support (like IntelliJ) can recognize the package structure:

```bash
# Make sure you're in the V1_implementation directory
cd V1_implementation

# Install in editable mode (use the same virtual environment)
pip install -e .
```

This will install the package in development mode, which means:
- Changes to the code are immediately reflected without reinstalling
- IDEs like IntelliJ/PyCharm can better resolve imports (especially `from training.config import ...`)
- The package is only installed in your current virtual environment

**Note**: If you're using IntelliJ/PyCharm, make sure your IDE is configured to use the Python interpreter from your active virtual environment (Settings → Project → Python Interpreter).

### 4. Configure Data Paths

Edit `training/config.py` to point to your PPMI data:

```python
@dataclass
class DataConfig:
    # Path configuration options:
    # 1. Use absolute paths (recommended)
    base_dir: str = "/full/path/to/ppmi_pd"
    genetic_consensus: str = "/full/path/to/genetics/iu_genetic_consensus_*.csv"
    
    # 2. Use paths relative to base_dir (without ../)
    base_dir: str = "../../ppmi_pd"
    participant_status: str = "Participant_Status_14Dec2025.csv"
    
    # 3. Use paths relative to where script is run (paths starting with ../)
    # These are resolved relative to current working directory
    genetic_consensus: str = "../genetics/iu_genetic_consensus_*.csv"
```

**Path Resolution Rules:**
- Paths starting with `/` are treated as absolute paths
- Paths starting with `../` are resolved relative to the **current working directory**
- Other paths are resolved relative to `base_dir`

**Recommended**: Use absolute paths or update paths to be relative to `base_dir` (without `../` prefix).

### 5. Test Components

Each components have according test files. Run test to understand component behavior.

```bash
cd V1_implementation

# Test configuration
python training/config.py

# Test model (requires PyTorch)
python models/v1_model.py
```

### 6. Train Model

**Recommended: Use the unified CLI entrypoint:**
```bash
# Default: 5-fold cross-validation
python -m training.main

# See Training section below for full documentation
```

---

## Project Structure

```
V1_implementation/
├── README.md                    # This file
├── setup.py                     # Package setup (for pip install -e .)
├── pyproject.toml               # Python project configuration
├── __init__.py                  # Package initialization
│
├── training/                   # Training & configuration
│   ├── config.py               # All hyperparameters & paths
│   ├── main.py                 # Unified CLI entrypoint (RECOMMENDED)
│   ├── train.py                # Training loop (V1Trainer class)
│   ├── kfold_trainer.py        # K-fold CV trainer class
│   ├── multi_modal_trainer.py  # Multi-modal ablation trainer
│   └── __init__.py
│
├── models/                      # Model architecture
│   ├── embeddings.py           # Modality embeddings + time encoding
│   ├── heads.py                # Prediction heads
│   ├── v1_model.py             # Complete V1 Transformer
│   └── __init__.py
│
├── data/                        # Data loading & processing
│   ├── base_loader.py          # Base classes
│   ├── data_integrator.py      # Combines all loaders + feature vector creation
│   ├── dataset.py              # PyTorch Dataset & DataLoader
│   ├── __init__.py
│   └── loaders/                # Individual data loaders
│       ├── genetics_loader.py
│       ├── demographics_loader.py
│       ├── updrs_loader.py
│       ├── non_motor_loader.py
│       ├── medication_loader.py
│       └── __init__.py
│
└── tests/                       # Test suite
    ├── __init__.py             # Test package initialization
    ├── conftest.py             # Pytest fixtures and shared configuration
    ├── test_loaders.py         # Tests for data loaders
    ├── test_models.py          # Tests for model components
    ├── test_training.py        # Tests for training configuration
    └── test_data.py            # Tests for data processing pipeline
```

### Key Files

- **Configuration**: `training/config.py` - All hyperparameters, paths, and settings
- **Model**: `models/v1_model.py` - Complete V1 Transformer architecture
- **Data Pipeline**: `data/data_integrator.py` - Combines all data loaders (includes feature scaling)
- **Training**: `training/train.py` - Training loop and model training
- **K-Fold Training**: `training/kfold_trainer.py` - K-fold cross-validation trainer
- **Dataset**: `data/dataset.py` - PyTorch Dataset & DataLoader utilities (includes k-fold dataloaders)
- **Tests**: `tests/` - Comprehensive test suite for all components

---

## Data Pipeline

### Overview

The data pipeline uses a **modular loader architecture** with a **master visit index** that defines the timeline independently of any modality. All loaders inherit from base classes that ensure consistent data processing.

**Key Design Principles:**
- ✅ **Visit Index as master timeline**: Timeline defined once, not implicitly by any modality
- ✅ **Standardized participant filtering**: All loaders use the same valid participant list
- ✅ **Type-based base classes**: `StaticDataLoader` (patient-level) vs `LongitudinalDataLoader` (visit-level)
- ✅ **Explicit modality alignment**: Each modality is aligned TO the visit_index
- ✅ **Two-mask system**: Input missingness (A) and label availability (B) are separate

### Visit Index (Master Timeline)

The `VisitIndexBuilder` creates the master timeline that all modalities align to:

```python
from data.visit_index_builder import VisitIndexBuilder

builder = VisitIndexBuilder(config)
visit_index = builder.build_from_sources(
    source_dfs=[updrs_df, non_motor_df, medication_df],
    valid_patnos=valid_patient_list
)
```

**visit_index columns:**

| Column | Description |
|--------|-------------|
| `PATNO` | Patient ID |
| `EVENT_ID` | Visit code (BL, V01, V02, R01, U01, etc.) |
| `visit_date` | Actual date (from INFODT) |
| `months_since_baseline` | Continuous time since baseline |
| `visit_order` | Integer (0, 1, 2, ...) within patient |
| `delta_months` | Time gap from previous visit |


**Key insight**: The visit_index is the UNION of all visit anchors across all sources, then filtered and sorted. This means:
- If a patient has a medication visit but no UPDRS that day, the visit still exists
- Missing modality data at a visit = mask it, don't filter the timeline

### Base Loader Classes

All loaders inherit from base classes defined in `data/base_loader.py`:

#### `StaticDataLoader` (Patient-Level Data)
- **Merge key**: `['PATNO']`
- **Use case**: One row per patient (genetics, demographics)
- **Examples**: `GeneticsLoader`, `DemographicsLoader`

#### `LongitudinalDataLoader` (Visit-Level Data)
- **Merge key**: `['PATNO', 'EVENT_ID']`
- **Use case**: Multiple rows per patient (one per visit)
- **Helper methods**:
  - `compute_time_since_baseline()`: Computes `months_since_baseline` from dates or `EVENT_ID`
  - `filter_by_valid_participants()`: Filters visits to only valid participants
- **Examples**: `UPDRSLoader`, `NonMotorAssessmentsLoader`, `MedicationLoader`, `AgeAtVisitLoader`

#### Standardized Participant Filtering

All loaders use `participant_status` as the source of truth for valid participants:

```python
# Base class methods available to all loaders:
self._load_and_filter_participants(config)  # Load and filter participant_status
```

**Filtering criteria**:
- ✅ Valid enrollment status: `Complete`, `Enrolled`, `Withdraw Deceased`, `Withdrew`
- ✅ Valid cohorts: `Healthy Control`, `Parkinson's Disease`, `Prodromal` (excludes SWEDD)
- ✅ Non-null `ENROLL_DATE`

### Individual Loaders

| Loader | Type | Description | Output Columns |
|--------|------|-------------|----------------|
| **DemographicsLoader** | Static | Demographics, socioeconomic, family history | `PATNO` + demographics features |
| **GeneticsLoader** | Static | Genetic consensus, PRS scores, principal components | `PATNO` + genetics features |
| **UPDRSLoader** | Longitudinal | UPDRS Parts I-IV (motor/non-motor assessments) | `PATNO`, `EVENT_ID`, `months_since_baseline` + UPDRS features |
| **NonMotorAssessmentsLoader** | Longitudinal | MoCA, ESS, SCOPA-AUT, Schwab & England | `PATNO`, `EVENT_ID`, `months_since_baseline` + non-motor scores |
| **MedicationLoader** | Longitudinal | LEDD, medication history, ON/OFF status | `PATNO`, `EVENT_ID`, `months_since_baseline` + medication features |
| **AgeAtVisitLoader** | Longitudinal | Age at each visit (varies by visit, not static) | `PATNO`, `EVENT_ID`, `months_since_baseline` + `AGE_AT_VISIT` |

#### Loader Pipeline Steps

**StaticDataLoader pattern** (`DemographicsLoader`, `GeneticsLoader`):
```python
1. Load and filter participant_status (using _load_and_filter_participants)
2. Load domain-specific data files
3. Merge on PATNO
4. Return: DataFrame with PATNO + features
```

**LongitudinalDataLoader pattern** (`UPDRSLoader`, `NonMotorAssessmentsLoader`, `MedicationLoader`, `AgeAtVisitLoader`):
```python
1. Load domain-specific data files
2. Filter to valid participants (using filter_by_valid_participants)
3. Compute months_since_baseline (using compute_time_since_baseline) if INFODT available
4. Return: DataFrame with PATNO, EVENT_ID, months_since_baseline + features
```

### Data Integration Pipeline

The `DataIntegrator` (`data/data_integrator.py`) orchestrates the complete pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Load All Data                                            │
├─────────────────────────────────────────────────────────────────┤
│ Static Data:                                                     │
│   demographics_df = demographics_loader.load()                  │
│   genetics_df = genetics_loader.load()                           │
│   → Merge on PATNO → static_df                                   │
│                                                                  │
│ Longitudinal Data:                                               │
│   updrs_df = updrs_loader.load()                                 │
│   non_motor_df = non_motor_loader.load()  (optional)              │
│   medication_df = medication_loader.load() (optional)           │
│   age_at_visit_df = age_at_visit_loader.load() (optional)       │
│   → Merge on [PATNO, EVENT_ID] → longitudinal_df                │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Compute Progression Slopes                               │
├─────────────────────────────────────────────────────────────────┤
│ For each patient with ≥min_visits visits:                        │
│   - Fit linear regression: score ~ months_since_baseline        │
│   - Fit linear regression: score ~ months_since_baseline        │
│   - Compute slope for NP1RTOT, NP2PTOT, NP3TOT, NP4TOT           │
│   - Store all slopes per patient (slopes_dict)                   │
│ → slopes_df (one row per patient with all slope columns)         │
│                                                                   │
│ Note: Model predicts ALL 4 slopes per patient (one per UPDRS total)│
│       Slopes are computed for all totals during data preparation  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Filter to Complete Cases                                 │
├─────────────────────────────────────────────────────────────────┤
│ Keep only patients present in BOTH:                              │
│   - static_df AND longitudinal_df                                │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT (prepare_final_dataset): Dictionary with                  │
├─────────────────────────────────────────────────────────────────┤
│ - 'static': DataFrame (patient-level features)                  │
│ - 'longitudinal': DataFrame (visit-level features)              │
│ - 'slopes': DataFrame (patient-level progression slopes for all totals)        │
│   Columns: PATNO, NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope   │
│ - 'metadata': Summary statistics                                │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Create Feature Vectors (REQUIRED)                       │
│ (create_feature_vectors() - creates missing value masks)        │
├─────────────────────────────────────────────────────────────────┤
│ Converts DataFrame format to dataset format with:                │
│   - Missing value masks (1=missing, 0=present)                  │
│   - Per-patient static data: {'values': array, 'mask': array}   │
│   - Per-patient longitudinal data: List of visit dicts          │
│   - Slopes dictionary: {PATNO: Dict[str, float]}                │
│     Keys: 'NP1RTOT_slope', 'NP2PTOT_slope', 'NP3TOT_slope', 'NP4TOT_slope'│
│     Model predicts all 4 slopes (one per UPDRS total)          │
│                                                                  │
│ This format is ready for PPMILongitudinalDataset                │
│ Note: Called automatically by create_dataloaders() if needed    │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Create PyTorch Datasets & DataLoaders                   │
│ (create_dataloaders() - handles train/val/test split)           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Automatically calls create_feature_vectors() if needed       │
│ 2. Splits patients into train/val/test                          │
│ 3. Creates PPMILongitudinalDataset for each split               │
│ 4. Returns DataLoaders ready for training                       │
└─────────────────────────────────────────────────────────────────┘
```

**Key Methods:**
- `prepare_final_dataset()` - Main pipeline: loads data, computes slopes, returns DataFrames
- `create_feature_vectors()` - Converts DataFrames to dataset format with missing value handling (REQUIRED for training)
- `create_missingness_masks()` - Creates explicit masks for missing values (used internally)
- `create_dataloaders()` - Creates train/val/test DataLoaders (automatically calls `create_feature_vectors()` if needed)
- `fit_scalers()` - Fit feature scalers on training data (for normalization)

### Feature Scaling/Normalization

The data pipeline includes **automatic feature normalization** using z-score (standardization) to handle features with different scales (e.g., UPDRS scores 0-132 vs. percentages 0-1 vs. age 30-80).

**How it works:**
- **FeatureScaler** class performs z-score normalization: `normalized = (value - mean) / std`
- Separate scalers are used for each modality: static, motor, UPDRS supplementary, non-motor, medication
- Scalers are **fitted on training data only** to prevent data leakage
- Missing values are handled correctly: only observed values are used for computing statistics, and missing values remain as 0 (with mask=1)

**Usage:**
```python
from data.data_integrator import DataIntegrator

# Enable normalization (default: True)
integrator = DataIntegrator(config, normalize_features=True)

# Prepare data (scalers will be fitted automatically on training split)
prepared_data = integrator.prepare_final_dataset()
train_loader, val_loader, test_loader = integrator.create_dataloaders(prepared_data, config)

# For k-fold CV, scalers are fitted per fold on training fold data only
# This is handled automatically by create_kfold_dataloaders()
```

**Disabling normalization:**
```python
# If you want raw feature values (not recommended for most cases)
integrator = DataIntegrator(config, normalize_features=False)
```

**Why normalize?**
- Features have very different scales (UPDRS scores: 0-132, LEDD: 0-2000, age: 30-90, percentages: 0-1)
- Neural networks train better with normalized features (stable gradients, faster convergence)
- Without normalization, large-scale features can dominate learning

### Running the Data Pipeline

**Test individual loaders:**
```bash
cd data/loaders
python genetics_loader.py      # Test genetics loader
python updrs_loader.py         # Test UPDRS loader
python demographics_loader.py  # Test demographics loader
```

**Run full integration:**
```bash
cd data
python data_integrator.py
```

**Output files:**
```
data/processed/
├── static_data.csv         # Patient-level features (PATNO + genetics + demographics)
├── longitudinal_data.csv   # Visit-level features (PATNO, EVENT_ID, months_since_baseline + all assessments)
└── slopes_data.csv         # Progression slopes (PATNO + NP1RTOT_slope, NP2PTOT_slope, etc.)
```

### Complete Pipeline Flow

```
Raw PPMI CSV Files
    │
    ├─→ participant_status.csv
    │       └─→ [STANDARDIZED FILTERING] ──→ Valid PATNOs
    │                                              │
    ├─→ DemographicsLoader ────────────────────────┤
    │   └─→ static_df (PATNO + demographics)       │
    │                                              │
    ├─→ GeneticsLoader ────────────────────────────└─→ static_df (merged)
    │   └─→ genetics_df (PATNO + genetics)       
    │                                            
    │                                            
    ├─→ UPDRSLoader ────────────────────────────┐
    │   └─→ updrs_df (PATNO, EVENT_ID, UPDRS)   │
    │                                           │
    ├─→ NonMotorAssessmentsLoader ──────────────┤
    │   └─→ non_motor_df (PATNO, EVENT_ID, ...)  │
    │                                           │
    ├─→ MedicationLoader ───────────────────────┤
    │   └─→ medication_df (PATNO, EVENT_ID, ...)├─→ longitudinal_df (merged)
    │                                           │
    └─→ AgeAtVisitLoader ───────────────────────┘
        └─→ age_at_visit_df (PATNO, EVENT_ID, AGE_AT_VISIT)
                        │
                        ▼
        ┌───────────────────────────────┐
        │   DataIntegrator              │
        │   prepare_final_dataset()     │
        ├───────────────────────────────┤
        │ 1. Merge static data          │
        │ 2. Merge longitudinal data    │
        │ 3. Compute slopes             │
        │ 4. Filter complete cases      │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   DataIntegrator              │
        │   create_feature_vectors()    │
        │   (REQUIRED for training)     │
        ├───────────────────────────────┤
        │ - Create missingness masks    │
        │ - Convert to dataset format   │
        │ - Per-patient data structure  │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   PPMILongitudinalDataset     │
        │   (PyTorch Dataset)           │
        ├───────────────────────────────┤
        │ - Feature extraction          │
        │ - Missingness masks           │
        │ - Sequence padding            │
        └───────────────────────────────┘
                        │
                        ▼
              Training DataLoader
```

### Using the Base Loader Classes

**Creating a new StaticDataLoader:**
```python
from data.base_loader import StaticDataLoader

class MyStaticLoader(StaticDataLoader):
    def load(self) -> pd.DataFrame:
        # Start with valid participants
        df = self._load_and_filter_participants(self.config)
        
        # Load your data
        my_data = pd.read_csv(...)
        
        # Merge
        df = df.merge(my_data, on='PATNO', how='inner')
        return df
```

**Creating a new LongitudinalDataLoader:**
```python
from data.base_loader import LongitudinalDataLoader

class MyLongitudinalLoader(LongitudinalDataLoader):
    def load(self) -> pd.DataFrame:
        # Load your longitudinal data
        df = pd.read_csv(...)
        
        # Filter to valid participants
        df = self.filter_by_valid_participants(df, self.config)
        
        # Compute time since baseline
        df = self.compute_time_since_baseline(df)
        
        return df
```

### Converting DataFrames to Dataset Format

For training or inference, you may need to convert DataFrame format to the structure expected by `PPMILongitudinalDataset`:

```python
from data.data_integrator import DataIntegrator

integrator = DataIntegrator(config)

# Option 1: Get DataFrames (for inspection, saving to CSV)
prepared_data = integrator.prepare_final_dataset()
# Returns: {'static': DataFrame, 'longitudinal': DataFrame, 'slopes': DataFrame, 'metadata': dict}

# Option 2: Convert directly to dataset format (with missing value handling)
feature_vectors = integrator.create_feature_vectors(prepared_data)
# Returns: {'static_data': Dict[PATNO, {...}], 'longitudinal_data': Dict[PATNO, [...]], 'slopes': Dict[PATNO, Dict[str, float]]}
# Note: slopes dict contains all computed slopes per patient (NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope)
#       Model predicts all 4 slopes per patient (one per UPDRS total)

# Option 3: Auto-convert (calls prepare_final_dataset internally)
feature_vectors = integrator.create_feature_vectors()
```

**Key differences:**
- `prepare_final_dataset()` → Returns **DataFrames** (good for CSV export, inspection)
- `create_feature_vectors()` → Returns **structured dicts** (good for PyTorch Dataset creation)

The `create_feature_vectors()` method automatically:
- Creates missing value masks (1 = missing, 0 = present)
- Converts NaN values to 0.0
- Structures data per-patient for `PPMILongitudinalDataset`

**Important**: `create_feature_vectors()` is **REQUIRED** for training, not optional. It must be called to convert DataFrames (with NaN values) into the masked format that `PPMILongitudinalDataset` expects. The `create_dataloaders()` function automatically calls `create_feature_vectors()` if you pass in DataFrame format.

### Creating DataLoaders for Training

The `create_dataloaders()` function handles the complete pipeline:

```python
from data.data_integrator import DataIntegrator
from data.dataset import create_dataloaders

# Option 1: Pass DataFrames (create_feature_vectors called automatically)
integrator = DataIntegrator(config)
prepared_data = integrator.prepare_final_dataset()  # Returns DataFrames
train_loader, val_loader, test_loader = create_dataloaders(prepared_data, config)

# Option 2: Pass feature vectors (already masked)
feature_vectors = integrator.create_feature_vectors(prepared_data)
train_loader, val_loader, test_loader = create_dataloaders(feature_vectors, config)
```

---

## Model Components

### 1. Modality Embeddings (`models/embeddings.py`)

Each modality is embedded with explicit missingness handling:

```python
# Input: values + mask
values = [x1, x2, x3]  # Actual values (0 if missing)
mask = [0, 1, 0]       # 1 if missing, 0 if observed

# Concatenate and embed
embedding = MLP([values, mask])  # Model learns to use mask
```

### 2. Time Encoding

Uses **continuous time** (actual months since baseline):

```python
time_encoding = SinusoidalEncoding(months_since_baseline)
# Not just visit indices: 0, 1, 2, 3...
# But actual timing: 0, 3, 6, 12, 18, 24, 36...
```

### 3. Transformer Encoder

- 4 layers, 8 attention heads
- Pre-LayerNorm architecture (more stable)
- Attention masks for variable-length sequences

### 4. Prediction Heads

**Next-Visit Head:**
- Input: Hidden state at time t
- Output: Predicted UPDRS totals at t+1
  - NP1RTOT (non-motor experiences, 0-52)
  - NP2PTOT (motor ADL, 0-52)
  - NP3TOT (motor examination, 0-132)
  - NP4TOT (motor complications, 0-24)
- Loss: MSE (summed across all totals)

**Slope Head:**
- Input: Pooled sequence representation
- Output: Patient-level progression slopes for all 4 UPDRS totals `[batch, 4]`
  - NP1RTOT_slope (non-motor progression)
  - NP2PTOT_slope (motor ADL progression)
  - NP3TOT_slope (motor examination progression, primary outcome)
  - NP4TOT_slope (complications progression)
- Target: Empirical slopes computed from ≥3 visits using linear regression for each UPDRS total
- Loss: Masked MSE (only valid slopes contribute to loss)
- Metrics: Per-target MAE, RMSE, and Spearman correlation with support counts

---

## Training

### Training with `main.py` (Recommended)

The `training/main.py` script provides a unified CLI entrypoint for all training workflows. It supports multiple training modes, automatic checkpointing, and multi-modal ablation studies.

#### Quick Start

```bash
# Default: 5-fold cross-validation training
python -m training.main

# With custom settings
python -m training.main --mode kfold --n-splits 5 --max-epochs 70 --batch-size 32
```

### Training Modes

The `--mode` flag determines the training strategy:

| Mode | Description | Use Case |
|------|-------------|----------|
| `kfold` | K-fold cross-validation (default) | Robust evaluation, model selection |
| `single` | Single train/val/test split | Quick experiments, fine-tuning |
| `multi_modal` | Train multiple models with different modality combinations | Ablation studies, modality comparison |

### Command-Line Arguments

```bash
python -m training.main [OPTIONS]

# Mode Selection
--mode {kfold,single,multi_modal}   # Training mode (default: kfold)
--config-version {v1,v2}             # Config version (default: v1)

# K-Fold CV Settings (for --mode kfold or multi_modal)
--n-splits N                         # Number of CV folds (default: 5)
--test-ratio RATIO                   # Test set ratio (default: 0.2)

# Single Split Settings (for --mode single)
--train-ratio RATIO                  # Train split ratio (default: 0.8)
--val-ratio RATIO                    # Val split ratio (default: 0.1)

# Hyperparameters (override config defaults)
--max-epochs N                       # Maximum epochs
--patience N                         # Early stopping patience
--batch-size N                       # Batch size
--learning-rate LR                   # Learning rate
--weight-decay WD                    # Weight decay

# Multi-Modal Training (for --mode multi_modal)
--modalities SPEC [SPEC ...]         # Modality configurations to train
--force-retrain                      # Force retrain even if checkpoint exists

# Fine-Tuning (for --mode single)
--checkpoint PATH                    # Path to checkpoint for fine-tuning
--resume-mode {model-only,full}      # Resume mode (default: model-only)
--freeze-backbone                    # Freeze embeddings + transformer

# Other Options
--device {cuda,mps,cpu}              # Device (default: auto-detect)
--num-workers N                      # DataLoader workers (default: 0)
--seed N                             # Random seed (default: 42)
--no-normalize                       # Disable feature normalization
--evaluate-test                      # Evaluate on test set after training
```

### Training Examples

#### 1. K-Fold Cross-Validation (Default)

**Best for:** Model evaluation, hyperparameter selection, robust performance estimates.

```bash
# Default 5-fold CV with 20% test set
python -m training.main

# Custom CV settings
python -m training.main \
  --mode kfold \
  --n-splits 3 \
  --test-ratio 0.15 \
  --max-epochs 70 \
  --patience 10 \
  --evaluate-test

# Using config v2
python -m training.main \
  --config-version v2 \
  --n-splits 5
```

**Output:**
- Per-fold checkpoints: `models/checkpoints/fold_1/`, `fold_2/`, etc.
- CV results: `models/checkpoints/kfold_results.json`
- Average validation loss across folds with standard deviation

#### 2. Single Train/Val/Test Split

**Best for:** Quick experiments, fine-tuning from a checkpoint, development.

```bash
# Standard single split
python -m training.main \
  --mode single \
  --train-ratio 0.8 \
  --val-ratio 0.1

# Fine-tuning from checkpoint
python -m training.main \
  --mode single \
  --checkpoint models/checkpoints/best_checkpoint.pt \
  --resume-mode model-only \
  --learning-rate 5e-5

# Head-only fine-tuning (freeze backbone)
python -m training.main \
  --mode single \
  --checkpoint models/checkpoints/best_checkpoint.pt \
  --freeze-backbone \
  --learning-rate 1e-4
```

#### 3. Multi-Modal Ablation Studies

**Best for:** Comparing different modality combinations, understanding which modalities matter most.

```bash
# Train with default modality sets
python -m training.main --mode multi_modal

# Train specific modality combinations
python -m training.main \
  --mode multi_modal \
  --modalities all static+motor motor_only static_only

# Use predefined sets + custom combinations
python -m training.main \
  --mode multi_modal \
  --modalities all static+motor static,motor,nonmotor

# Force retrain everything (ignore existing checkpoints)
python -m training.main \
  --mode multi_modal \
  --modalities all static+motor \
  --force-retrain
```

**Predefined Modality Sets:**
- `all`: All modalities (default)
- `static+motor`: Static + motor only
- `static+nonmotor`: Static + non-motor only
- `motor_only`: Motor only
- `static_only`: Static only
- `no_static`: All except static
- `no_motor`: All except motor
- And more...

**Custom Combinations:**
- Use comma-separated list: `static,motor,nonmotor`
- Automatically sorted and deduplicated

**Output Structure:**
```
models/checkpoints/
├── modalities_all/
│   ├── fold_1/
│   │   └── training_history.json  # Per-epoch training curves
│   ├── fold_2/
│   │   └── training_history.json
│   ├── kfold_results.json  # CV results with modality metadata
│   └── test_metrics.json  # Test set evaluation (if evaluated)
├── modalities_static+motor/
│   ├── kfold_results.json
│   └── test_metrics.json
└── ...
```

**Checkpoint Detection:**
- Automatically skips already-trained configurations
- Checks for `kfold_results.json` with `status='completed'`
- Use `--force-retrain` to retrain anyway

### Training Workflows

#### Workflow 1: Model Selection with K-Fold CV

```bash
# Step 1: Run k-fold CV to select best hyperparameters
python -m training.main \
  --mode kfold \
  --n-splits 5 \
  --max-epochs 100 \
  --learning-rate 1e-4

# Step 2: Review results in models/checkpoints/kfold_results.json
# Step 3: Adjust hyperparameters and retrain if needed
# Step 4: Use train_full mode (TODO) to train final model on all data
```

#### Workflow 2: Modality Ablation Study

```bash
# Step 1: Train all modality configurations
python -m training.main \
  --mode multi_modal \
  --modalities all static+motor static+nonmotor motor_only static_only

# Step 2: Review results in each modality folder (e.g., models/checkpoints/modalities_motor_only/kfold_results.json)
# Step 3: Use aggregate_summaries() method to compare all results, or manually review each folder
# Step 4: Identify best modality combination
# Step 4: Train final model with best configuration
python -m training.main \
  --mode kfold \
  --config-version v1
  # (manually edit config.model.enabled_modalities to best combination)
```

#### Workflow 3: Fine-Tuning Pipeline

```bash
# Step 1: Train base model with all modalities
python -m training.main --mode kfold --n-splits 5

# Step 2: Fine-tune best fold's model
python -m training.main \
  --mode single \
  --checkpoint models/checkpoints/fold_1/best_checkpoint.pt \
  --resume-mode model-only \
  --learning-rate 5e-5

# Step 3: Optional: Head-only fine-tuning
python -m training.main \
  --mode single \
  --checkpoint models/checkpoints/best_checkpoint.pt \
  --freeze-backbone \
  --learning-rate 1e-4
```

### Default Hyperparameters

```python
# Model
d_model = 256
n_heads = 8
n_layers = 4
dropout = 0.1

# Training
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
lambda_slope = 0.2      # Weight for slope loss
max_epochs = 100
early_stopping_patience = 15
```

All hyperparameters can be overridden via command-line arguments (see `--max-epochs`, `--learning-rate`, etc.).

### Training Programmatically (Advanced)

For programmatic access, you can still use the training classes directly:

```python
from models.v1_model import V1MultimodalTransformer
from training.train import V1Trainer
from training.kfold_trainer import KFoldTrainer
from training.multi_modal_trainer import MultiModalTrainer
from training.config import get_default_config
from data.data_integrator import DataIntegrator

config = get_default_config()

# Option 1: Single training
model = V1MultimodalTransformer(config)
trainer = V1Trainer(model, config, train_loader, val_loader, device='cuda')
trainer.train(max_epochs=100, early_stopping_patience=15)

# Option 2: K-fold CV
integrator = DataIntegrator(config, normalize_features=True)
prepared = integrator.prepare_final_dataset()
kfold_trainer = KFoldTrainer(config, prepared, n_splits=5)
results = kfold_trainer.train()

# Option 3: Multi-modal ablation
multi_modal_trainer = MultiModalTrainer(config, prepared)
results = multi_modal_trainer.train_multiple(
    modality_specs=['all', 'static+motor', 'motor_only'],
    force_retrain=False
)
```

### Expected Performance

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Next-visit MAE | < 5 points | On NP3TOT (primary motor outcome, 0-132 scale) |
| Next-visit MAE | < 3 points | On NP1RTOT, NP2PTOT (0-52 scale) |
| Next-visit MAE | < 2 points | On NP4TOT (0-24 scale) |
| Slope Spearman correlation | r > 0.5 | With empirical slopes (per-target, NP3TOT most important) |
| ON vs OFF gap | 5-10 points | NP3TOT medication effect (NP2PTOT also shows effect) |

### Uploading Checkpoints to Hugging Face

The `upload_checkpoints.py` script (located in `Experiment/`) allows you to upload all model checkpoints to Hugging Face Hub for easy sharing and version control.

#### Prerequisite

##### Install the Hugging Face CLI (if not installed)
```bash
brew install huggingface-cli
# curl -LsSf https://hf.co/cli/install.sh | bash
```

##### (optional) Login with your Hugging Face credentials
```bash
huggingface-cli login
# hf auth login
```

#### Usage

```bash
# UPLOAD
huggingface-cli upload SRI-HRI/PD-v1 modalities_age_at_visit+medication+motor+non_motor+static/  modalities_age_at_visit+medication+motor+non_motor+static/. --allow-patterns="*.pt"
# hf upload SRI-HRI/PD-v1 modalities_age_at_visit+medication+motor+non_motor+static/ modalities_age_at_visit+medication+motor+non_motor+static/ --include="*.pt"

# DELETE
# delete all .pt file
# hf repo-files delete SRI-HRI/PD-v1 "*.pt"

# DOWNLOAD 
# huggingface-cli download SRI-HRI/PD-v1 modalities_age_at_visit+medication+motor+non_motor+static/fold_7/best_checkpoint.pt --local-dir models/checkpoints/
```

## Metrics and Evaluation

The model computes comprehensive evaluation metrics that are saved in checkpoints and can be used for model comparison and analysis.

### Metrics Structure

All metrics are computed during validation and stored in checkpoints under `val_metrics` (current epoch) and `best_val_metrics` (best epoch).

#### Next-Visit Metrics (Per-Visit Regression)

**Per-Target Metrics** (one per UPDRS total: NP1RTOT, NP2PTOT, NP3TOT, NP4TOT):
- **MAE** (Mean Absolute Error): Average absolute difference between predictions and targets
- **RMSE** (Root Mean Squared Error): Square root of average squared differences
- **R²** (Coefficient of Determination): Proportion of variance explained
- **Pearson Correlation**: Linear correlation coefficient
- **Spearman Correlation**: Rank-based correlation (robust to non-linear relationships)
- **n_samples**: Number of evaluated samples (support count)

**Per-Δt Bucket Metrics** (grouped by time intervals between visits):
- Metrics computed separately for different time gaps:
  - `dt_0_6`: 0-6 months between visits
  - `dt_6_12`: 6-12 months between visits
  - `dt_12_24`: 12-24 months between visits
  - `dt_24_inf`: 24+ months between visits
- Each bucket includes: MAE, RMSE, and n_samples

**Overall Next-Visit Metrics**:
- **macro_avg_mae**: Mean of MAEs across all targets (equal weight)
- **macro_avg_rmse**: Mean of RMSEs across all targets
- **macro_avg_r2**: Mean of R² values across all targets
- **macro_avg_correlation**: Mean of Pearson correlations across all targets
- **macro_avg_spearman**: Mean of Spearman correlations across all targets
- **weighted_avg_mae**: MAE weighted by number of samples per target
- **weighted_avg_rmse**: RMSE weighted by number of samples per target
- **total_samples**: Total number of evaluated samples across all targets

#### Slope Metrics (Patient-Level Regression)

**Per-Target Slope Metrics** (one per UPDRS total slope: NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope):
- **MAE**: Mean absolute error for slope predictions
- **RMSE**: Root mean squared error for slope predictions
- **Spearman Correlation**: Rank-based correlation (important for progression trends)
- **n_samples**: Number of patients with valid slope labels (support count)

**Overall Slope Metrics**:
- **mae**: Macro-average MAE across all slope targets
- **rmse**: Macro-average RMSE across all slope targets
- **spearman**: Macro-average Spearman correlation across all slope targets
- **n_samples**: Total number of patients with valid slope labels

### Why Support Counts Matter

**Always check support counts** to avoid misleading metrics:
- A "good overall average" can hide that one target has almost no evaluated examples
- Low support counts indicate insufficient data for reliable evaluation
- Per-target metrics with support counts show where the model is actually being evaluated

**Example**:
```json
{
  "next_visit": {
    "NP1RTOT": {"mae": 2.5, "n_samples": 1500},
    "NP2PTOT": {"mae": 3.1, "n_samples": 1480},
    "NP3TOT": {"mae": 4.8, "n_samples": 1520},
    "NP4TOT": {"mae": 1.2, "n_samples": 200}  // ⚠️ Low support!
  }
}
```

In this example, NP4TOT has much lower support (200 vs ~1500), so its metrics are less reliable.

### Accessing Metrics

#### From Checkpoints

```python
import torch

# Load checkpoint
checkpoint = torch.load('models/checkpoints/best_checkpoint.pt')

# Access metrics
val_metrics = checkpoint['val_metrics']  # Current epoch metrics
best_val_metrics = checkpoint['best_val_metrics']  # Best epoch metrics

# Example: Get NP3TOT next-visit MAE
np3tot_mae = best_val_metrics['next_visit']['NP3TOT']['mae']
np3tot_n_samples = best_val_metrics['next_visit']['NP3TOT']['n_samples']

# Example: Get overall slope metrics
slope_mae = best_val_metrics['slope_overall']['mae']
slope_spearman = best_val_metrics['slope_overall']['spearman']
```

#### From K-Fold Results

K-fold training saves test metrics to JSON:

```python
import json

# Load k-fold results
with open('models/checkpoints/kfold_results.json', 'r') as f:
    results = json.load(f)

# Access test metrics (if --evaluate-test was used)
with open('models/checkpoints/test_metrics.json', 'r') as f:
    test_metrics = json.load(f)
    
# Test metrics structure
test_metrics = {
    'losses': {...},
    'metrics': {
        'next_visit': {...},
        'next_visit_overall': {...},
        'slope': {...},
        'slope_overall': {...}
    }
}
```

### Interpreting Metrics

#### Next-Visit Metrics

- **MAE < 5 points** on NP3TOT (0-132 scale) is considered good
- **MAE < 3 points** on NP1RTOT, NP2PTOT (0-52 scale) is considered good
- **MAE < 2 points** on NP4TOT (0-24 scale) is considered good
- **R² > 0.3** indicates meaningful predictive power
- **Correlation > 0.5** (Pearson or Spearman) shows strong relationship

#### Slope Metrics

- **Spearman correlation > 0.5** indicates good rank-order agreement with empirical slopes
- **MAE < 0.5 points/month** on NP3TOT_slope is considered good
- Slope metrics are particularly important for long-term progression tracking

#### Per-Δt Bucket Metrics

- Helps identify if model performance varies by time gap between visits
- Short-term predictions (0-6 months) may be more accurate than long-term (24+ months)
- Useful for understanding model limitations at different prediction horizons

### Metric Computation Details

Metrics are computed using:
- **Masked evaluation**: Only valid (non-NaN) targets contribute to metrics
- **Per-target computation**: Each UPDRS total is evaluated separately
- **Support-aware aggregation**: Overall metrics account for varying sample sizes

The metrics computation handles:
- Missing labels: NaN targets are excluded from evaluation
- Variable sequence lengths: Padding is masked out
- Per-target missingness: Different targets may have different availability

---

## Configuration

All settings are in `training/config.py`:

### Feature Configuration

```python
@dataclass
class FeatureConfig:
    # Static features
    static_features: List[str]      # Genetics + demographics
    
    # UPDRS by part
    part1_features: List[str]       # Non-motor (+ NP1RTOT)
    part2_features: List[str]       # Motor ADL (+ NP2PTOT)
    part3_features: List[str]       # Motor exam (+ NP3TOT)
    part4_features: List[str]       # Complications (+ NP4TOT)
    
    # Other
    other_nonmotor_features: List[str]  # MoCA, ESS, etc.
    medication_features: List[str]      # LEDD, PDMEDYN
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 20
    predict_totals: List[str] = ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']  # All UPDRS totals
```

### Data Configuration

```python
@dataclass
class DataConfig:
    base_dir: str = "../../ppmi_pd"
    # Update all file paths to match your data
```

---

## UPDRS Structure

The MDS-UPDRS has **4 parts**, each with its own total score:

| Part | Code | Total | Range | What It Measures |
|------|------|-------|-------|------------------|
| I | NP1* | NP1RTOT | 0-52 | Non-motor experiences |
| II | NP2* | NP2PTOT | 0-52 | Motor ADL (patient-reported) |
| III | NP3* | NP3TOT | 0-132 | Motor examination (clinician) |
| IV | NP4* | NP4TOT | 0-24 | Motor complications |

### Medication Sensitivity

| Part | ON vs OFF Difference | Notes |
|------|---------------------|-------|
| Part I (NP1RTOT) | Minimal (~0-2 pts) | Non-dopaminergic symptoms |
| Part II (NP2PTOT) | Moderate (~5-10 pts) | Patient-reported motor ADL |
| Part III (NP3TOT) | **Large (~10-20 pts)** | Main progression outcome, clinician-observed |
| Part IV (NP4TOT) | N/A | These ARE medication complications |

### V1 Implementation

V1 predicts **all four UPDRS totals**:
- **NP1RTOT**: Non-motor experiences (baseline + progression)
- **NP2PTOT**: Motor activities of daily living (patient perspective)
- **NP3TOT**: Motor examination (primary outcome, clinician-observed)
- **NP4TOT**: Motor complications (medication-related)

This comprehensive approach captures:
- Multi-domain progression (motor, non-motor, complications)
- Different perspectives (patient-reported vs clinician-observed)
- Medication effects (especially in NP3TOT and NP4TOT)

---

## Testing

### Test Suite Structure

The project includes a comprehensive test suite in the `tests/` directory:

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Pytest fixtures and shared configuration
├── test_models.py       # Tests for model components
├── test_training.py     # Tests for training configuration
├── test_data.py         # Tests for data processing pipeline
└── data_loaders/        # Tests for individual data loaders
    ├── __init__.py
    ├── test_base_loader.py        # Base loader + interface tests
    ├── test_demographics_loader.py
    ├── test_genetics_loader.py
    ├── test_updrs_loader.py
    ├── test_non_motor_loader.py
    └── test_medication_loader.py
```

### Running Tests

#### Prerequisites

Install dependencies (includes `pytest`). Optionally install `pytest-cov` for coverage:

```bash
# From V1_implementation directory
python -m pip install -r ../../requirements.txt

# Optional: coverage support
python -m pip install pytest-cov
```

#### Run All Tests

```bash
# From V1_implementation directory
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

#### Run Specific Test Files

```bash
# Test all data loaders
pytest tests/data_loaders/

# Test specific loader (e.g., demographics)
pytest tests/data_loaders/test_demographics_loader.py

# Test model components only
pytest tests/test_models.py

# Test training configuration only
pytest tests/test_training.py

# Test data processing only
pytest tests/test_data.py
```

#### Run Specific Test Classes or Methods

```bash
# Run specific test class
pytest tests/data_loaders/test_demographics_loader.py::TestDemographicsLoader
pytest tests/data_loaders/test_age_at_visit_loader.py::TestAgeAtVisitLoader

# Run specific test method
pytest tests/test_models.py::TestV1Model::test_model_forward
```

### Test Markers

Tests are marked for different conditions:

- `@pytest.mark.requires_data` - Tests that need actual data files (will skip if data not available)
- `@pytest.mark.slow` - Tests that take longer to run

Run tests by marker:

```bash
# Skip tests that require data
pytest tests/ -m "not requires_data"

# Run only slow tests
pytest tests/ -m "slow"
```

### Test Coverage

Generate coverage reports:

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# View report
open htmlcov/index.html  # On macOS
```

### Quick Component Testing

For quick testing of individual components during development, you can still use the `if __name__ == "__main__":` blocks in each module:

```bash
# Test configuration
python training/config.py

# Test embeddings
python models/embeddings.py

# Test data loaders
python data/loaders/demographics_loader.py
python data/loaders/genetics_loader.py

# Test model
python models/v1_model.py
```

**Note**: These quick tests are useful during development, but the formal test suite in `tests/` is recommended for comprehensive testing.

### Writing New Tests

When adding new features, add corresponding tests:

1. **Data Loaders**: Create a new file in `tests/data_loaders/` (e.g., `test_new_loader.py`)
2. **Model Components**: Add tests to `tests/test_models.py`
3. **Training Utilities**: Add tests to `tests/test_training.py`
4. **Data Processing**: Add tests to `tests/test_data.py`

#### Example Test Structure

```python
# tests/data_loaders/test_new_loader.py
"""
Tests for NewLoader
"""

import pytest
import pandas as pd


class TestNewLoader:
    """Test NewLoader"""
    
    @pytest.mark.requires_data
    def test_new_loader_load(self, test_config, skip_if_no_data):
        """Test NewLoader can load data"""
        from data.loaders.new_loader import NewLoader
        
        loader = NewLoader(test_config.data.base_dir, test_config)
        df = loader.load()
        
        assert isinstance(df, pd.DataFrame)
        assert 'PATNO' in df.columns
        assert len(df) > 0
```

### Continuous Integration

The test suite is designed to work with CI/CD systems. Example GitHub Actions workflow:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install pytest pytest-cov
      - run: pytest tests/ --cov=. --cov-report=xml
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Wrong data paths | Update `DataConfig` in `config.py` |
| `KeyError: column` | Column name mismatch | Check actual CSV columns |
| GPU out of memory | Batch/model too large | Reduce `batch_size` or `d_model` |
| Loss is NaN | Learning rate too high | Reduce to 1e-5, normalize features |
| All predictions similar | Low feature variance | Check data, increase `lambda_slope` |

### Quick Component Testing

For quick testing during development, you can use the `if __name__ == "__main__":` blocks:

```bash
# Test configuration
python training/config.py

# Test embeddings
python models/embeddings.py

# Test model
python models/v1_model.py

# Test dataset
python data/dataset.py
```

**For comprehensive testing, see the [Testing](#testing) section above.**

---

## Next Steps

### After V1 Works

1. **Evaluate thoroughly**: Check performance by subgroup (LRRK2/GBA carriers, etc.)
2. **Error analysis**: Where does the model fail?
3. **Feature importance**: What matters most?

## Quick Reference

### Activate Environment
```bash
conda activate sri_hri
```

### Test Configuration
```bash
python training/config.py
```

### Run Tests
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Test all data loaders
pytest tests/data_loaders/ -v

# Test specific loader (e.g., demographics)
pytest tests/data_loaders/test_demographics_loader.py -v
```

### Run Data Integration
```bash
python data/data_integrator.py
```

### Train Model
```bash
# Recommended: Use unified CLI
python -m training.main

# Or use legacy script
python training/train.py
```

### Key Files
- `training/config.py` - All settings
- `models/v1_model.py` - Model architecture
- `data/data_integrator.py` - Data pipeline
- `data/TASK_ASSIGNMENTS.md` - Parallel task guide

---

## License

[Add your license here]

## Contact

[Add contact information]

---

**Ready to start?** Follow the [Quick Start](#quick-start) section above! 🚀
