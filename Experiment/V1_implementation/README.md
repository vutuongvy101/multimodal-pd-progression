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
| **Multi-Objective** | Predicts next-visit UPDRS totals (NP1TOT, NP2TOT, NP3TOT, NP4TOT) + patient-level progression slopes |

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
│                        PATIENT DATA                              │
├─────────────────┬───────────────────────────────────────────────┤
│  STATIC         │  TIME-VARYING (per visit)                     │
│  • Genetics     │  • Part I (non-motor)                         │
│  • Demographics │  • Part II (motor ADL)                        │
│                 │  • Part III (motor exam)                      │
│                 │  • Part IV (complications)                    │
│                 │  • Other assessments (MoCA, ESS, etc.)        │
│                 │  • Medication context (LEDD, ON/OFF)          │
└─────────────────┴───────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                     MODALITY EMBEDDINGS                          │
│  Each modality: MLP([values, missing_mask]) → embedding          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     VISIT TOKEN FORMATION                        │
│  v_t = Concat(static_emb, motor_emb, nonmotor_emb, med_emb)     │
│  v_t = v_t + SinusoidalTimeEncoding(months_since_baseline)       │
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
│   (NP1TOT, NP2TOT,      │     │   (NP1TOT, NP2TOT,      │
│    NP3TOT, NP4TOT)      │     │    NP3TOT, NP4TOT)      │
│   Loss: MSE             │     │   Loss: MSE             │
└─────────────────────────┘     └─────────────────────────┘

Total Loss = L_next_visit + λ × L_slope  (λ = 0.2)
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

```bash
python training/train.py
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
│   ├── data_integrator.py      # Combines all loaders
│   ├── data_preparation.py     # Data preparation utilities
│   ├── dataset.py              # PyTorch Dataset & DataLoader
│   ├── __init__.py
│   └── loaders/                # Individual data loaders
│       ├── genetics_loader.py
│       ├── demographics_loader.py
│       ├── updrs_loader.py
│       ├── clinical_loader.py
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
- **Data Pipeline**: `data/data_integrator.py` - Combines all data loaders
- **Training**: `training/train.py` - Training loop and model training
- **Tests**: `tests/` - Comprehensive test suite for all components

---

## Data Pipeline

### Overview

The data pipeline uses a **modular loader architecture** with standardized participant filtering. All loaders inherit from base classes that ensure consistent data processing across the pipeline.

**Key Design Principles:**
- ✅ **Standardized participant filtering**: All loaders use the same valid participant list from `participant_status`
- ✅ **Type-based base classes**: `StaticDataLoader` (patient-level) vs `LongitudinalDataLoader` (visit-level)
- ✅ **Modular design**: Each loader is independent and can be tested separately
- ✅ **Consistent merge keys**: Static data uses `PATNO`, longitudinal data uses `[PATNO, EVENT_ID]`

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
- **Examples**: `UPDRSLoader`, `ClinicalAssessmentsLoader`, `MedicationLoader`, `AgeAtVisitLoader`

#### Standardized Participant Filtering

All loaders use `participant_status` as the source of truth for valid participants:

```python
# Base class methods available to all loaders:
self._load_and_filter_participants(config)  # Load and filter participant_status
self._filter_valid_participants(df)         # Apply filtering logic
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
| **ClinicalAssessmentsLoader** | Longitudinal | MoCA, ESS, SCOPA-AUT, Schwab & England | `PATNO`, `EVENT_ID`, `months_since_baseline` + clinical scores |
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

**LongitudinalDataLoader pattern** (`UPDRSLoader`, `ClinicalAssessmentsLoader`, `MedicationLoader`, `AgeAtVisitLoader`):
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
│   clinical_df = clinical_loader.load()  (optional)              │
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
│   - Extract slope for NP1TOT, NP2TOT, NP3TOT, NP4TOT           │
│ → slopes_df (one row per patient)                                │
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
│ OUTPUT: Dictionary with                                          │
├─────────────────────────────────────────────────────────────────┤
│ - 'static': Patient-level features                              │
│ - 'longitudinal': Visit-level features                          │
│ - 'slopes': Patient-level progression slopes                    │
│ - 'metadata': Summary statistics                                │
└─────────────────────────────────────────────────────────────────┘
```

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
└── slopes_data.csv         # Progression slopes (PATNO + NP1TOT_slope, NP2TOT_slope, etc.)
```

### Complete Pipeline Flow

```
Raw PPMI CSV Files
    │
    ├─→ participant_status.csv
    │       └─→ [STANDARDIZED FILTERING] ──→ Valid PATNOs
    │                                              │
    ├─→ DemographicsLoader ───────────────────────┘
    │   └─→ static_df (PATNO + demographics)
    │
    ├─→ GeneticsLoader ──────────────────────────┐
    │   └─→ genetics_df (PATNO + genetics)       │
    │                                            ├─→ static_df (merged)
    │                                            │
    ├─→ UPDRSLoader ────────────────────────────┐
    │   └─→ updrs_df (PATNO, EVENT_ID, UPDRS)   │
    │                                            │
    ├─→ ClinicalAssessmentsLoader ──────────────┤
    │   └─→ clinical_df (PATNO, EVENT_ID, ...)  │
    │                                            │
    ├─→ MedicationLoader ────────────────────────┤
    │   └─→ medication_df (PATNO, EVENT_ID, ...)├─→ longitudinal_df (merged)
    │                                            │
    └─→ AgeAtVisitLoader ─────────────────────────┘
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
  - NP1TOT (non-motor experiences, 0-52)
  - NP2TOT (motor ADL, 0-52)
  - NP3TOT (motor examination, 0-132)
  - NP4TOT (motor complications, 0-24)
- Loss: MSE (summed across all totals)

**Slope Head:**
- Input: Pooled sequence representation
- Output: Patient-level progression slopes for each UPDRS total
- Target: Empirical slopes computed from ≥3 visits (linear regression per total)
- Loss: MSE (summed across all totals)

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

### Expected Performance

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Next-visit MAE | < 5 points | On NP3TOT (primary outcome, 0-132 scale) |
| Next-visit MAE | < 3 points | On NP1TOT, NP2TOT, NP4TOT (0-52, 0-24 scales) |
| Slope correlation | r > 0.5 | With empirical slopes (NP3TOT most important) |
| ON vs OFF gap | 5-10 points | NP3TOT medication effect (NP2TOT also shows effect) |

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
    part1_features: List[str]       # Non-motor (+ NP1TOT)
    part2_features: List[str]       # Motor ADL (+ NP2TOT)
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
    predict_totals: List[str] = ['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT']  # All UPDRS totals
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
| I | NP1* | NP1TOT | 0-52 | Non-motor experiences |
| II | NP2* | NP2TOT | 0-52 | Motor ADL (patient-reported) |
| III | NP3* | NP3TOT | 0-132 | Motor examination (clinician) |
| IV | NP4* | NP4TOT | 0-24 | Motor complications |

### Medication Sensitivity

| Part | ON vs OFF Difference | Notes |
|------|---------------------|-------|
| Part I (NP1TOT) | Minimal (~0-2 pts) | Non-dopaminergic symptoms |
| Part II (NP2TOT) | Moderate (~5-10 pts) | Patient-reported motor ADL |
| Part III (NP3TOT) | **Large (~10-20 pts)** | Main progression outcome, clinician-observed |
| Part IV (NP4TOT) | N/A | These ARE medication complications |

### V1 Implementation

V1 predicts **all four UPDRS totals**:
- **NP1TOT**: Non-motor experiences (baseline + progression)
- **NP2TOT**: Motor activities of daily living (patient perspective)
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
    ├── test_clinical_loader.py
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
