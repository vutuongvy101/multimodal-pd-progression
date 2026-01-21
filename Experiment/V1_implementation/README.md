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
| **Multi-Objective** | Predicts next-visit UPDRS totals (NP1RTOT, NP2PTOT, NP3TOT, NP4TOT) + patient-level progression slope (default: NP3TOT_slope) |

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

```bash
conda activate sri_hri
```

### 2. Install Dependencies

```bash
pip install torch numpy pandas scipy tqdm
```

### 3. Install Package in Editable Mode (Optional but Recommended)

Install the V1 package in editable mode so that imports work properly and IDE support (like IntelliJ) can recognize the package structure:

```bash
# Make sure you're in the V1_implementation directory
cd V1_implementation

# Install in editable mode (use the same conda environment)
pip install -e .
```

This will install the package in development mode, which means:
- Changes to the code are immediately reflected without reinstalling
- IDEs like IntelliJ/PyCharm can better resolve imports (especially `from training.config import ...`)
- The package is only installed in your current virtual environment

**Note**: If you're using IntelliJ/PyCharm, make sure your IDE is configured to use the Python interpreter from your conda environment (Settings → Project → Python Interpreter).

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

```bash
cd V1_implementation

# Test configuration
python training/config.py

# Test model (requires PyTorch)
python models/v1_model.py
```

### 6. Prepare Data

Data loading is modularized into 5 independent tasks (see [Data Pipeline](#data-pipeline)):

```bash
cd data
python data_integrator.py
```

### 7. Train Model

**Standard training:**
```bash
python training/train.py
```

**K-fold cross-validation (recommended for robust evaluation):**
```bash
python training/train_kfold.py
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
├── training/                    # Training & configuration
│   ├── config.py               # All hyperparameters & paths
│   ├── train.py                # Training loop
│   ├── train_kfold.py          # K-fold cross-validation training
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
- **K-Fold Training**: `training/train_kfold.py` - K-fold cross-validation training script
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
│   - Compute slope for NP1RTOT, NP2PTOT, NP3TOT, NP4TOT           │
│   - Store all slopes per patient (slopes_dict)                   │
│ → slopes_df (one row per patient with all slope columns)         │
│                                                                   │
│ Note: Model predicts ONE slope per patient (default: NP3TOT_slope)│
│       but slopes are computed for all totals during preparation   │
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
│     Keys: 'NP1RTOT_slope', 'NP2PTOT_slope', 'NP3TOT_slope', etc.│
│     Model uses NP3TOT_slope by default (primary target)         │
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
#       but model predicts one slope per patient (default: NP3TOT_slope)

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
- Output: One patient-level progression slope per patient (single scalar value)
- Target: Empirical slope computed from ≥3 visits using linear regression
  - **Default target**: `NP3TOT_slope` (primary motor progression, most clinically relevant)
  - Falls back to any available slope if NP3TOT_slope is missing
- Loss: MSE on the single slope prediction

**Note on Slope Computation:**
- During data preparation, slopes are computed for **all UPDRS totals** (NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope) and stored per patient
- However, the model is trained to predict **one slope per patient** (defaulting to NP3TOT_slope)
- This focuses the model on the most clinically important motor progression metric (NP3TOT)

---

## Training

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

### Training Loop

```python
from models.v1_model import V1MultimodalTransformer
from training.train import V1Trainer
from training.config import get_default_config

config = get_default_config()
model = V1MultimodalTransformer(config)

trainer = V1Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda'
)

trainer.train(max_epochs=100, early_stopping_patience=15)
```

### K-Fold Cross-Validation

The project supports **k-fold cross-validation** for robust model evaluation. This is especially useful for small datasets where a single train/val/test split might not be reliable.

**Key Features:**
- Proper scaler fitting per fold (fitted on training fold, applied to validation fold)
- Automatic patient splitting into k folds
- Optional held-out test set (separate from CV folds)
- Average performance metrics across folds

**Usage:**

```python
from models.v1_model import V1MultimodalTransformer
from training.train import V1Trainer
from training.train_kfold import train_kfold
from training.config import get_default_config
from data.data_integrator import DataIntegrator
from data.dataset import create_kfold_dataloaders

config = get_default_config()

# Option 1: Use the convenience function (recommended)
results = train_kfold(
    config=config,
    n_splits=5,        # 5-fold CV
    n_epochs=50,       # Epochs per fold
    device='cuda'
)

# Option 2: Manual setup for more control
integrator = DataIntegrator(config, normalize_features=True)
prepared_data = integrator.prepare_final_dataset()

# Create k-fold dataloaders (automatically handles scaler fitting per fold)
fold_dataloaders, test_loader = create_kfold_dataloaders(
    prepared_data=prepared_data,
    config=config,
    n_splits=5,
    test_ratio=0.2,    # Hold out 20% as test set
    random_seed=42
)

# Train on each fold
for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
    model = V1MultimodalTransformer(config)
    trainer = V1Trainer(model, config, train_loader, val_loader, device='cuda')
    trainer.train(max_epochs=50, early_stopping_patience=15)
    
    # Evaluate on validation fold
    val_metrics = trainer.validate()
    print(f"Fold {fold_idx + 1} Val Loss: {val_metrics['loss']:.4f}")
```

**Important Notes:**
- Scalers are fitted **per fold** on the training fold data only
- Test set is held out completely and scalers are fitted on all CV data (not test data) before applying to test set
- Each fold trains an independent model (you get k models)
- Use average metrics across folds to assess model performance
- For final evaluation, retrain on all training data (all folds combined) and evaluate on held-out test set

### Expected Performance

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Next-visit MAE | < 5 points | On NP3TOT (primary motor outcome, 0-132 scale) |
| Next-visit MAE | < 3 points | On NP1RTOT, NP2PTOT (0-52 scale) |
| Next-visit MAE | < 2 points | On NP4TOT (0-24 scale) |
| Slope correlation | r > 0.5 | With empirical slopes (NP3TOT most important) |
| ON vs OFF gap | 5-10 points | NP3TOT medication effect (NP2PTOT also shows effect) |

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

Install pytest and optional testing dependencies:

```bash
conda activate sri_hri
pip install pytest pytest-cov
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
