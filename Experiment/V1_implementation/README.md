# V1: Multimodal Longitudinal Transformer for Parkinson's Disease

A Transformer-based deep learning model for predicting Parkinson's disease progression using PPMI longitudinal data.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Components](#model-components)
5. [Training Process](#training-process)
6. [Quick Start](#quick-start)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**V1** is a Transformer-based model that predicts Parkinson's disease progression by:
- Fusing multimodal data (genetics, demographics, clinical assessments, medication)
- Learning longitudinal patterns across irregular patient visits
- Handling missing data explicitly with two-mask system
- Predicting both next-visit outcomes and long-term progression slopes

### Key Features

| Feature | Description |
|---------|-------------|
| **Longitudinal Fusion** | Transformer learns from sequences of visits with variable timing |
| **Multimodal** | Combines static (genetics, demographics) + time-varying (motor, non-motor, medication, age) |
| **Missingness Handling** | Two-mask system: input missingness (Mask A) and label availability (Mask B) |
| **Time-Aware** | Continuous time encoding using actual months since baseline |
| **Multi-Objective** | Predicts next-visit UPDRS totals (4 targets) + patient-level progression slopes (4 targets) |
| **Feature Normalization** | Automatic z-score normalization per modality (fitted on training data only) |

### Prediction Targets

**Next-Visit Predictions** (per visit):
- `NP1RTOT`: Non-motor experiences of daily living (0-52)
- `NP2PTOT`: Motor activities of daily living (0-52)
- `NP3TOT`: Motor examination (0-132) - **primary outcome**
- `NP4TOT`: Motor complications (0-24)

**Slope Predictions** (per patient):
- `NP1RTOT_slope`: Non-motor progression rate (points/month)
- `NP2PTOT_slope`: Motor ADL progression rate (points/month)
- `NP3TOT_slope`: Motor examination progression rate (points/month) - **primary**
- `NP4TOT_slope`: Complications progression rate (points/month)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA (per patient)                    │
├─────────────────────────────────────────────────────────────────┤
│ Static: [batch, n_static] + mask                               │
│ Motor: [batch, seq_len, n_motor] + mask                        │
│ Non-motor: [batch, seq_len, n_nonmotor] + mask                 │
│ Medication: [batch, seq_len, n_med] + mask                      │
│ Age at visit: [batch, seq_len, n_age] + mask                   │
│ Time: [batch, seq_len] (months since baseline)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODALITY EMBEDDINGS (per enabled modality)        │
│  Static: MLP([values, mask]) → [batch, d_model]                │
│  Motor: MLP([values, mask]) → [batch, seq_len, d_model]        │
│  Non-motor: MLP([values, mask]) → [batch, seq_len, d_model]    │
│  Medication: MLP([values, mask]) → [batch, seq_len, d_model]   │
│  Age: MLP([values, mask]) → [batch, seq_len, d_model]           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISIT TOKEN FORMATION                       │
│  1. Concatenate enabled modality embeddings                     │
│  2. Project to d_model: Linear(d_model × n_modalities → d_model)│
│  3. Add time encoding: SinusoidalTimeEncoding(months)          │
│  Output: [batch, seq_len, d_model]                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSFORMER ENCODER                            │
│  - Layers: 4 (configurable)                                    │
│  - Heads: 8 (configurable)                                    │
│  - d_model: 256 (configurable)                                 │
│  - Pre-LayerNorm architecture                                 │
│  - Attention mask for variable-length sequences               │
│  Output: [batch, seq_len, d_model]                             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   NEXT-VISIT HEAD       │     │   SLOPE HEAD            │
│   Input: h_t            │     │   Input: pool(H)        │
│   MLP: [d_model → ... → 4]  │     │   Pooling: mean (masked)  │
│   Output: [batch, seq_len, 4]│     │   MLP: [d_model → ... → 4]│
│   Predicts: UPDRS totals│     │   Output: [batch, 4]    │
│   at next visit         │     │   Predicts: slopes     │
└─────────────────────────┘     └─────────────────────────┘
```

### Two-Mask System

The model uses **two separate mask types** to handle PPMI-style irregular data:

**Mask A: Input Missingness Masks** (per modality)
- **Purpose**: Indicate which features are observed vs. missing at each visit
- **Shape**: `[batch, seq_len, n_features]` (or `[batch, n_features]` for static)
- **Values**: `1 = missing`, `0 = present`
- **Usage**: Concatenated with feature values before embedding
- **Location**: Modality embedding MLPs

**Mask B: Label Availability Masks** (per target)
- **Purpose**: Indicate which target labels are available for loss computation
- **Shape**: `[batch, seq_len, n_targets]` for next-visit targets
- **Values**: `1 = label present`, `0 = label missing`
- **Usage**: Applied during loss computation only
- **Location**: Loss function (masks out invalid predictions)

**Key Insight**: The timeline includes ALL visits (even if some modalities are missing). Missing modalities are masked at the input level (Mask A), while missing labels are masked at the loss level (Mask B).

### Modality Selection

Modalities can be enabled/disabled via configuration for ablation studies:

```python
# All modalities (default)
config.model.enabled_modalities = ['static', 'motor', 'nonmotor', 'medication', 'age_at_visit']

# Ablation: static + motor only
config.model.enabled_modalities = ['static', 'motor']

# Ablation: without medication context
config.model.enabled_modalities = ['static', 'motor', 'nonmotor', 'age_at_visit']
```

Only enabled modalities are embedded and contribute to visit tokens.

---

## Data Pipeline

### Overview

The data pipeline transforms raw PPMI CSV files into training-ready PyTorch DataLoaders through these stages:

1. **Data Loading**: Individual loaders extract features from CSV files
2. **Visit Index Building**: Master timeline created independently of modalities
3. **Data Integration**: Merge static and longitudinal data, compute slopes
4. **Feature Engineering**: Create missingness masks, normalize features
5. **Dataset Creation**: Convert to PyTorch Dataset format
6. **DataLoader Creation**: Batch data with proper padding and masking

### Stage 1: Data Loading

Individual loaders extract domain-specific data from PPMI CSV files:

**Static Loaders** (patient-level, one row per patient):
- `GeneticsLoader`: Genetic variants, PRS scores, principal components
- `DemographicsLoader`: Demographics, socioeconomic status, family history

**Longitudinal Loaders** (visit-level, multiple rows per patient):
- `UPDRSLoader`: UPDRS Parts I-IV scores, supplementary motor assessments
- `NonMotorAssessmentsLoader`: MoCA, ESS, SCOPA-AUT, Schwab & England
- `MedicationLoader`: LEDD, medication history, ON/OFF status
- `AgeAtVisitLoader`: Age at each visit (varies by visit, not static)

**Base Classes**:
- `StaticDataLoader`: Base class for patient-level data (merge key: `['PATNO']`)
- `LongitudinalDataLoader`: Base class for visit-level data (merge key: `['PATNO', 'EVENT_ID']`)

All loaders:
- Filter to valid participants (using `participant_status.csv`)
- Handle missing values gracefully
- Return DataFrames with standardized columns

### Stage 2: Visit Index Building

The `VisitIndexBuilder` creates a **master timeline** that defines when visits occur, independent of any modality:

```python
visit_index = VisitIndexBuilder.build_from_sources(
    source_dfs=[updrs_df, non_motor_df, medication_df, age_at_visit_df],
    valid_patnos=valid_patient_list
)
```

**visit_index columns**:
- `PATNO`: Patient ID
- `EVENT_ID`: Visit code (BL, V01, V02, R01, U01, etc.)
- `visit_date`: Actual date (from INFODT)
- `months_since_baseline`: Continuous time since baseline
- `visit_order`: Integer (0, 1, 2, ...) within patient
- `delta_months`: Time gap from previous visit

**Key Insight**: The visit_index is the **UNION** of all visit anchors across all sources. If a patient has a medication visit but no UPDRS that day, the visit still exists in the timeline. Missing modality data at a visit = mask it, don't filter the timeline.

**Patient Filtering**: Patients with fewer than `min_visits` (default: 2) are filtered out at this stage.

### Stage 3: Data Integration

The `DataIntegrator` orchestrates the complete pipeline:

```python
integrator = DataIntegrator(config, normalize_features=True)
prepared_data = integrator.prepare_final_dataset()
```

**Steps**:
1. **Load all data**: Call each loader's `load()` method
2. **Merge static data**: Merge genetics + demographics on `PATNO`
3. **Merge longitudinal data**: Merge all visit-level data on `['PATNO', 'EVENT_ID']`
4. **Compute progression slopes**: For each patient with ≥3 visits, fit linear regression:
   - `score ~ months_since_baseline` for each UPDRS total (NP1RTOT, NP2PTOT, NP3TOT, NP4TOT)
   - Store slopes in `slopes_df` (one row per patient)
5. **Filter to complete cases**: Keep only patients present in both static and longitudinal data

**Progression Slope Computation Details**:
- **Method**: Ordinary Least Squares (OLS) using `scipy.stats.linregress`
- **Visits included**: 
  - All visits with valid `months_since_baseline` (non-NaN)
  - For each UPDRS total, only visits with non-NaN scores are included
  - Minimum visits required: `min_visits_for_slope = 3` (configurable)
  - Each total is computed independently (patient may have slope for some totals but not others)
- **Missing data handling**: 
  - Missing values are dropped per target using `dropna()` on the specific UPDRS total column
  - If a patient has valid scores at visits [0, 1, 3, 5] but missing at visit 2, only [0, 1, 3, 5] are used
- **Time units**: Months (continuous `months_since_baseline`)
- **Clipping/Standardization**: 
  - No clipping applied to slopes
  - No standardization applied to slopes
  - Slopes are used as-is from regression (can be negative, positive, or zero)
  - Units: points per month
- **Validation**: 
  - Skips computation if all visits are at the same time point (std(time) == 0)
  - Stores correlation coefficient (`r`) and p-value (`p`) for each slope

**Output**: Dictionary with:
- `'static'`: DataFrame (patient-level features)
- `'longitudinal'`: DataFrame (visit-level features)
- `'slopes'`: DataFrame (patient-level slopes: PATNO, NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope)
- `'metadata'`: Summary statistics

### Stage 4: Feature Engineering

The `FeatureEngineer` creates missingness masks and normalizes features:

```python
feature_vectors = integrator.create_feature_vectors(prepared_data)
```

**Steps**:
1. **Create missingness masks** (Mask A):
   - For each feature, `mask = 1` if value is NaN, `mask = 0` if present
   - Missing values are set to `0.0` (mask indicates missingness)
2. **Normalize features** (if `normalize_features=True`):
   - Z-score normalization: `normalized = (value - mean) / std`
   - Separate scalers per modality: static, motor, UPDRS supplementary, non-motor, medication, age_at_visit
   - **Critical**: Scalers are fitted on training data only (prevents data leakage)
3. **Structure data per-patient**:
   - Static: `{PATNO: {'values': array, 'mask': array}}`
   - Longitudinal: `{PATNO: [visit_dict_1, visit_dict_2, ...]}`

**Visit Dictionary Structure** (per visit):
```python
{
    'motor_values': np.ndarray,      # [n_motor_features]
    'motor_mask': np.ndarray,         # [n_motor_features], 1=missing
    'nonmotor_values': np.ndarray,    # [n_nonmotor_features]
    'nonmotor_mask': np.ndarray,      # [n_nonmotor_features], 1=missing
    'med_values': np.ndarray,         # [n_med_features]
    'med_mask': np.ndarray,           # [n_med_features], 1=missing
    'age_at_visit_values': np.ndarray, # [n_age_features]
    'age_at_visit_mask': np.ndarray,   # [n_age_features], 1=missing
    'time_months': float,              # Months since baseline
    'delta_months': float,             # Time gap from previous visit
    'visit_order': int,                # Visit index (0, 1, 2, ...)
    'updrs_totals': np.ndarray         # [4] - NP1RTOT, NP2PTOT, NP3TOT, NP4TOT (NaN if missing)
}
```

**Slopes Dictionary Structure**:
```python
{
    PATNO: {
        'NP1RTOT_slope': float,   # or NaN if not available
        'NP2PTOT_slope': float,   # or NaN if not available
        'NP3TOT_slope': float,    # or NaN if not available
        'NP4TOT_slope': float     # or NaN if not available
    }
}
```

### Stage 5: Dataset Creation

The `PPMILongitudinalDataset` converts feature vectors to PyTorch tensors:

```python
dataset = PPMILongitudinalDataset(
    patient_ids=patient_list,
    static_data=static_data,      # Dict[PATNO, {'values': array, 'mask': array}]
    longitudinal_data=longitudinal_data,  # Dict[PATNO, List[visit_dict]]
    slopes=slopes,                # Dict[PATNO, Dict[str, float]]
    max_seq_len=20
)
```

**Dataset `__getitem__` returns**:
- `static_values`: `[n_static]` tensor
- `static_mask`: `[n_static]` tensor (1=missing)
- `motor_values`: `[seq_len, n_motor]` tensor
- `motor_mask`: `[seq_len, n_motor]` tensor (1=missing)
- `nonmotor_values`: `[seq_len, n_nonmotor]` tensor
- `nonmotor_mask`: `[seq_len, n_nonmotor]` tensor (1=missing)
- `med_values`: `[seq_len, n_med]` tensor
- `med_mask`: `[seq_len, n_med]` tensor (1=missing)
- `age_at_visit_values`: `[seq_len, n_age]` tensor
- `age_at_visit_mask`: `[seq_len, n_age]` tensor (1=missing)
- `time_months`: `[seq_len]` tensor
- `next_visit_targets`: `[seq_len, 4]` tensor (UPDRS totals, NaN filled with 0)
- `next_visit_label_mask`: `[seq_len, 4]` tensor (1=label present, 0=missing) - **Mask B**
- `slope_targets`: `[4]` tensor (slopes, NaN filled with 0)
- `seq_len`: int (actual sequence length before padding)

### Stage 6: DataLoader Creation

The `collate_fn` handles variable-length sequences:

```python
train_loader, val_loader, test_loader = create_dataloaders(
    prepared_data, config,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)
```

**Collate Function**:
1. **Pad sequences** to max length in batch
2. **Create attention_mask**: `[batch, max_seq_len]`, `1 = valid visit`, `0 = padding`
3. **Stack all tensors** with proper padding

**DataLoader Output** (per batch):
- All input tensors: `[batch, max_seq_len, n_features]` (or `[batch, n_features]` for static)
- `attention_mask`: `[batch, max_seq_len]` (1=valid, 0=padding)
- `next_visit_targets`: `[batch, max_seq_len, 4]`
- `next_visit_label_mask`: `[batch, max_seq_len, 4]` (1=present, 0=missing) - **Mask B**
- `slope_targets`: `[batch, 4]`

### Feature Normalization Details

**Why Normalize?**
- Features have very different scales:
  - UPDRS scores: 0-132
  - LEDD: 0-2000
  - Age: 30-90
  - Percentages: 0-1
- Neural networks train better with normalized features (stable gradients, faster convergence)

**How It Works**:
1. **Separate scalers per modality**: static, motor, UPDRS supplementary, non-motor, medication, age_at_visit
2. **Fitted on training data only**: Prevents data leakage
3. **Z-score normalization**: `normalized = (value - mean) / std`
4. **Missing values**: Excluded from mean/std computation, remain as 0 (with mask=1)

**For K-Fold CV**:
- Scalers are fitted per fold on training fold data only
- Applied to validation fold (and test set)
- Handled automatically by `create_kfold_dataloaders()`

### Complete Pipeline Flow

```
Raw PPMI CSV Files
    │
    ├─→ participant_status.csv
    │       └─→ [Filter valid participants] ──→ Valid PATNOs
    │                                              │
    ├─→ DemographicsLoader ────────────────────────┤
    │   └─→ demographics_df (PATNO + demographics) │
    │                                              │
    ├─→ GeneticsLoader ────────────────────────────└─→ static_df (merged)
    │   └─→ genetics_df (PATNO + genetics)       
    │                                            
    ├─→ UPDRSLoader ────────────────────────────┐
    │   └─→ updrs_df (PATNO, EVENT_ID, UPDRS)   │
    │                                           │
    ├─→ NonMotorAssessmentsLoader ──────────────┤
    │   └─→ non_motor_df (PATNO, EVENT_ID, ...) │
    │                                           │
    ├─→ MedicationLoader ────────────────────────┤
    │   └─→ medication_df (PATNO, EVENT_ID, ...)├─→ longitudinal_df (merged)
    │                                           │
    └─→ AgeAtVisitLoader ────────────────────────┘
        └─→ age_at_visit_df (PATNO, EVENT_ID, AGE_AT_VISIT)
                        │
                        ▼
        ┌───────────────────────────────┐
        │   VisitIndexBuilder            │
        │   build_from_sources()         │
        ├───────────────────────────────┤
        │ 1. Union all visit anchors    │
        │ 2. Compute months_since_baseline│
        │ 3. Filter by min_visits       │
        └───────────────────────────────┘
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
        │   fit_scalers()               │
        │   (on training data only)     │
        ├───────────────────────────────┤
        │ Fit z-score scalers per      │
        │ modality on training split   │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   DataIntegrator              │
        │   create_feature_vectors()   │
        ├───────────────────────────────┤
        │ 1. Create missingness masks  │
        │ 2. Normalize features        │
        │ 3. Structure per-patient     │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   PPMILongitudinalDataset     │
        │   (PyTorch Dataset)           │
        ├───────────────────────────────┤
        │ Convert to tensors            │
        │ Extract targets & masks       │
        └───────────────────────────────┘
                        │
                        ▼
              DataLoader (with collate_fn)
              - Padding
              - Batching
              - Attention masks
```

---

## Model Components

### 1. Modality Embeddings (`models/embeddings.py`)

Each modality is embedded with explicit missingness handling:

**StaticFeatureEmbedding** (patient-level):
```python
Input: values [batch, n_static], mask [batch, n_static]
Process: MLP([values, mask])  # Concatenate values + mask
Output: [batch, d_model]
```

**VisitFeatureEmbedding** (visit-level):
```python
Input: values [batch, seq_len, n_features], mask [batch, seq_len, n_features]
Process: MLP([values, mask])  # Concatenate along feature dimension
Output: [batch, seq_len, d_model]
```

**Architecture**:
- Input: `n_features * 2` (values + mask concatenated)
- Hidden layers: Configurable MLP (default: `[128]`)
- Activation: GELU
- Normalization: LayerNorm after each hidden layer
- Output: `d_model` (default: 256)

### 2. Time Encoding (`models/embeddings.py`)

**SinusoidalTimeEncoding**: Continuous time encoding based on actual months since baseline

```python
time_encoding = SinusoidalTimeEncoding(d_model=256, max_time=120.0)
time_emb = time_encoding(time_months)  # [batch, seq_len] → [batch, seq_len, d_model]
```

**Formula**:
- Normalize time: `time_normalized = time / max_time`
- Sinusoidal encoding: `PE(pos, 2i) = sin(time_normalized * div_term[i])`
- `PE(pos, 2i+1) = cos(time_normalized * div_term[i])`

**Key**: Uses actual months (0, 3, 6, 12, 18, ...), not visit indices (0, 1, 2, 3, ...).

### 3. Visit Token Builder (`models/embeddings.py`)

**VisitTokenBuilder**: Combines enabled modality embeddings into visit tokens

```python
visit_builder = VisitTokenBuilder(d_model=256, enabled_modalities=['static', 'motor', ...])
visit_tokens = visit_builder(embeddings, seq_len)
```

**Process**:
1. Expand static embedding: `[batch, d_model]` → `[batch, seq_len, d_model]` (broadcast to all visits)
2. Concatenate all enabled modality embeddings: `[batch, seq_len, d_model * n_modalities]`
3. Project to `d_model`: `Linear(d_model * n_modalities → d_model)`
4. Add time encoding: `visit_tokens + time_emb`

**Output**: `[batch, seq_len, d_model]`

### 4. Transformer Encoder (`models/v1_model.py`)

**Architecture**:
- **Layers**: 4 (configurable)
- **Heads**: 8 (configurable)
- **d_model**: 256 (configurable)
- **dim_feedforward**: 1024 (configurable)
- **Activation**: GELU
- **Normalization**: Pre-LayerNorm (more stable)
- **Dropout**: 0.1 (configurable)

**Attention Masking**:
- `src_key_padding_mask`: `[batch, seq_len]`, `True = padding position` (masked out)
- Derived from `attention_mask` in forward pass

**Output**: `[batch, seq_len, d_model]` - contextualized visit representations

### 5. Prediction Heads (`models/heads.py`)

**NextVisitPredictionHead**:
```python
Input: hidden_states [batch, seq_len, d_model]
Process: MLP(hidden_states)  # Applied per position
Output: [batch, seq_len, 4]  # Predictions for next visit
```

**Architecture**:
- Hidden layers: `[128, 64]` (configurable)
- Activation: GELU
- Normalization: LayerNorm
- Output: 4 values (NP1RTOT, NP2PTOT, NP3TOT, NP4TOT)

**Usage**: Predicts UPDRS totals at visit `t+1` from hidden state at visit `t`.

**ProgressionSlopeHead**:
```python
Input: hidden_states [batch, seq_len, d_model], attention_mask [batch, seq_len]
Process:
  1. Pool sequence: mean(hidden_states, dim=1, masked by attention_mask)
  2. MLP(pooled)
Output: [batch, 4]  # One slope per UPDRS total
```

**Pooling Methods**:
- `mean`: Masked mean pooling (default)
- `last`: Last valid position
- `max`: Masked max pooling

**Architecture**:
- Hidden layers: `[128, 64]` (configurable)
- Activation: GELU
- Normalization: LayerNorm
- Output: 4 values (NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope)

**Usage**: Predicts patient-level progression slopes from entire sequence.

### 6. Loss Computation (`models/v1_model.py`)

**Multi-Task Loss**:
```python
total_loss = loss_next_visit + lambda_slope * loss_slope
```

**Next-Visit Loss**:
```python
# Shift predictions and targets
preds = predictions['next_visit'][:, :-1, :]  # [batch, seq_len-1, 4]
targets = targets['next_visit'][:, 1:, :]      # [batch, seq_len-1, 4]
label_mask = targets['next_visit_mask'][:, 1:, :]  # [batch, seq_len-1, 4]

# Combined mask: attention_mask AND label_mask
combined_mask = attention_mask[:, :-1].unsqueeze(-1) * label_mask

# Masked MSE
mse = (preds - targets) ** 2
loss = (mse * combined_mask).sum() / combined_mask.sum()
```

**Slope Loss**:
```python
preds = predictions['slope']  # [batch, 4]
targets = targets['slope']   # [batch, 4]

# Mask out NaN targets
valid_mask = ~torch.isnan(targets)
targets_masked = torch.where(valid_mask, targets, torch.zeros_like(targets))

# Masked MSE
mse = (preds - targets_masked) ** 2
loss = (mse * valid_mask).sum() / valid_mask.sum()
```

**Key**: Both losses use proper masking to exclude invalid predictions/targets.

---

## Training Process

### Training Workflow

The training process follows this workflow:

```
1. Load Configuration
   └─→ training/config.py: get_default_config()

2. Prepare Data
   └─→ DataIntegrator.prepare_final_dataset()
       └─→ Returns DataFrames (static, longitudinal, slopes)

3. Create DataLoaders
   └─→ create_dataloaders() or create_kfold_dataloaders()
       ├─→ Split patients into train/val/test
       ├─→ Fit scalers on training data only
       ├─→ Create feature vectors (with masks)
       ├─→ Create PPMILongitudinalDataset
       └─→ Return DataLoaders

4. Initialize Model
   └─→ V1MultimodalTransformer(config)
       ├─→ Create modality embeddings (for enabled modalities)
       ├─→ Create visit token builder
       ├─→ Create time encoding
       ├─→ Create transformer encoder
       └─→ Create prediction heads

5. Initialize Trainer
   └─→ V1Trainer(model, config, train_loader, val_loader, device)
       ├─→ AdamW optimizer
       ├─→ WarmupCosineAnnealingLR scheduler
       └─→ Training history tracking

6. Training Loop
   └─→ For each epoch:
       ├─→ Train epoch:
       │   ├─→ For each batch:
       │   │   ├─→ Forward pass: model(batch)
       │   │   ├─→ Compute loss: model.compute_loss(predictions, targets)
       │   │   ├─→ Backward pass: loss.backward()
       │   │   └─→ Optimizer step: optimizer.step()
       │   └─→ Average losses
       │
       ├─→ Validate:
       │   ├─→ For each batch:
       │   │   ├─→ Forward pass (no gradients)
       │   │   └─→ Accumulate predictions and targets
       │   ├─→ Compute comprehensive metrics
       │   └─→ Average losses
       │
       ├─→ Update scheduler: scheduler.step(val_loss)
       ├─→ Check for best model: Save if val_loss improved
       └─→ Early stopping: Stop if no improvement for N epochs

7. Save Checkpoint
   └─→ Save model, optimizer, scheduler, metrics, history
```

### Training Modes

**1. Single Train/Val/Test Split**:
```python
train_loader, val_loader, test_loader = create_dataloaders(
    prepared_data, config,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

trainer = V1Trainer(model, config, train_loader, val_loader, device='cuda')
trainer.train(max_epochs=100, early_stopping_patience=15)
```

**2. K-Fold Cross-Validation**:
```python
fold_dataloaders, test_loader = create_kfold_dataloaders(
    prepared_data, config, n_splits=5, test_ratio=0.2
)

for fold_idx, (train_loader, val_loader) in enumerate(fold_dataloaders):
    model = V1MultimodalTransformer(config)
    trainer = V1Trainer(model, config, train_loader, val_loader, device='cuda')
    trainer.train(max_epochs=100, early_stopping_patience=15)
    # Save fold checkpoint
```

**3. Multi-Modal Ablation**:
```python
modality_configs = [
    ['static', 'motor', 'nonmotor', 'medication', 'age_at_visit'],  # All
    ['static', 'motor'],  # Static + motor only
    ['static', 'motor', 'nonmotor'],  # Without medication
    # ... more combinations
]

for modalities in modality_configs:
    config.model.enabled_modalities = modalities
    # Train model with this modality combination
```

### Training Details

**Optimizer**: AdamW
- Learning rate: `1e-4` (configurable)
- Weight decay: `1e-5` (configurable)
- Gradient clipping: `max_norm=1.0`

**Scheduler**: WarmupCosineAnnealingLR
- Warmup epochs: `10% of max_epochs` (configurable)
- Cosine annealing: `target_lr → min_lr` (min_lr = target_lr * 0.1)
- Applied per epoch based on validation loss

**Loss Weights**:
- `lambda_slope = 0.2` (configurable)
- Total loss: `loss_next_visit + lambda_slope * loss_slope`

**Early Stopping**:
- Patience: `15 epochs` (configurable)
- Metric: Validation loss
- Saves best model checkpoint

**Checkpointing**:
- Saves after each epoch: `latest_checkpoint.pt`
- Saves best model: `best_checkpoint.pt`
- Includes: model state, optimizer state, scheduler state, metrics, training history

**Metrics Computed** (during validation):
- **Next-visit metrics** (per target: NP1RTOT, NP2PTOT, NP3TOT, NP4TOT):
  - MAE, RMSE, R², Pearson correlation, Spearman correlation
  - Per-Δt bucket metrics (0-6mo, 6-12mo, 12-24mo, 24+mo)
  - Overall macro/weighted averages
- **Slope metrics** (per target: NP1RTOT_slope, NP2PTOT_slope, NP3TOT_slope, NP4TOT_slope):
  - MAE, RMSE, Spearman correlation
  - Overall macro averages

### Using the CLI (`training/main.py`)

**Recommended**: Use the unified CLI entrypoint:

```bash
# Default: 5-fold cross-validation
python -m training.main

# Single split training
python -m training.main --mode single --train-ratio 0.8 --val-ratio 0.1

# K-fold CV with custom settings
python -m training.main --mode kfold --n-splits 5 --max-epochs 100 --batch-size 32

# Multi-modal ablation
python -m training.main --mode multi_modal --modalities all static+motor motor_only
```

See `python -m training.main --help` for all options.

---

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- PPMI data access

### 1. Activate Virtual Environment

```bash
# Option A: conda
conda activate sri_hri

# Option B: pyenv (virtualenv)
pyenv virtualenv 3.10.13 pd-v1
pyenv activate pd-v1
```

### 2. Install Dependencies

```bash
# From Experiment/V1_implementation
cd Experiment/V1_implementation
python -m pip install -r ../../requirements.txt
```

### 3. Install Package in Editable Mode

```bash
# From V1_implementation directory
pip install -e .
```

This enables:
- Proper import resolution (e.g., `from training.config import ...`)
- IDE support (IntelliJ/PyCharm)
- Immediate code changes without reinstalling

### 4. Configure Data Paths

Edit `training/config.py` to point to your PPMI data:

```python
@dataclass
class DataConfig:
    # Use absolute paths (recommended)
    base_dir: str = "/full/path/to/ppmi_pd"
    participant_status: str = "Participant_Status_14Dec2025.csv"
    genetic_consensus: str = "/full/path/to/genetics/iu_genetic_consensus_*.csv"
```

**Path Resolution Rules**:
- Paths starting with `/` are treated as absolute paths
- Paths starting with `../` are resolved relative to current working directory
- Other paths are resolved relative to `base_dir`

### 5. Test Components

```bash
# Test configuration
python training/config.py

# Test model architecture
python models/v1_model.py

# Test data pipeline
python data/data_integrator.py
```

### 6. Train Model

```bash
# Default: 5-fold cross-validation
python -m training.main

# With custom settings
python -m training.main --mode kfold --n-splits 5 --max-epochs 100 --batch-size 32
```

---

## Configuration

All settings are in `training/config.py`:

### Model Configuration

```python
@dataclass
class ModelConfig:
    d_model: int = 256                    # Embedding dimension
    n_heads: int = 8                      # Attention heads
    n_layers: int = 4                     # Transformer layers
    dim_feedforward: int = 1024           # FFN dimension
    dropout: float = 0.1                  # Dropout rate
    max_seq_len: int = 20                # Maximum sequence length
    max_time_months: float = 120.0        # Max time for encoding
    enabled_modalities: List[str] = field(default_factory=lambda: [
        'static', 'motor', 'nonmotor', 'medication', 'age_at_visit'
    ])
    predict_totals: List[str] = field(default_factory=lambda: [
        'NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT'
    ])
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 15
    lambda_slope: float = 0.2             # Slope loss weight
    warmup_epochs: int = 10                # LR warmup epochs
    warmup_ratio: float = 0.1             # Warmup as ratio of max_epochs
    min_lr_ratio: float = 0.1             # Min LR = target_lr * min_lr_ratio
```

### Data Configuration

```python
@dataclass
class DataConfig:
    base_dir: str = "../../ppmi_pd"
    participant_status: str = "Participant_Status_14Dec2025.csv"
    genetic_consensus: str = "Genetic_Status/iu_genetic_consensus_*.csv"
    # ... more file paths
    min_visits: int = 2                    # Minimum visits per patient
    normalize_features: bool = True        # Z-score normalization
```

### Feature Configuration

See `training/config.py` for complete feature lists:
- `static_features`: Genetics + demographics
- `motor_features`: UPDRS Parts I-IV + supplementary
- `non_motor_features`: MoCA, ESS, SCOPA-AUT, etc.
- `medication_features`: LEDD, ON/OFF status, etc.
- `age_at_visit_features`: Age at each visit

---

## Project Structure

```
V1_implementation/
├── README.md                    # This file
├── setup.py                     # Package setup
├── pyproject.toml               # Python project configuration
├── __init__.py                  # Package initialization
│
├── training/                   # Training & configuration
│   ├── config.py               # All hyperparameters & paths
│   ├── main.py                 # Unified CLI entrypoint (RECOMMENDED)
│   ├── train.py                # Training loop (V1Trainer class)
│   ├── kfold_trainer.py        # K-fold CV trainer class
│   ├── multi_modal_trainer.py  # Multi-modal ablation trainer
│   ├── metrics.py              # Comprehensive metrics computation
│   └── __init__.py
│
├── models/                      # Model architecture
│   ├── embeddings.py           # Modality embeddings + time encoding
│   ├── heads.py                # Prediction heads
│   ├── v1_model.py             # Complete V1 Transformer
│   └── __init__.py
│
├── data/                        # Data loading & processing
│   ├── base_loader.py          # Base classes (StaticDataLoader, LongitudinalDataLoader)
│   ├── data_integrator.py      # Main integration pipeline
│   ├── dataset.py              # PyTorch Dataset & DataLoader
│   ├── feature_engineer.py     # Feature normalization & masking
│   ├── label_engineer.py       # Slope computation
│   ├── visit_index_builder.py  # Master timeline creation
│   ├── __init__.py
│   └── loaders/                # Individual data loaders
│       ├── genetics_loader.py
│       ├── demographics_loader.py
│       ├── updrs_loader.py
│       ├── non_motor_loader.py
│       ├── medication_loader.py
│       ├── age_at_visit_loader.py
│       └── __init__.py
│
└── tests/                       # Test suite
    ├── __init__.py
    ├── conftest.py             # Pytest fixtures
    ├── test_loaders.py         # Tests for data loaders
    ├── test_models.py          # Tests for model components
    ├── test_training.py        # Tests for training
    └── test_data.py           # Tests for data processing
```

### Key Files

- **Configuration**: `training/config.py` - All hyperparameters, paths, and settings
- **Model**: `models/v1_model.py` - Complete V1 Transformer architecture
- **Data Pipeline**: `data/data_integrator.py` - Main integration pipeline
- **Training**: `training/train.py` - Training loop and V1Trainer class
- **Dataset**: `data/dataset.py` - PyTorch Dataset & DataLoader utilities
- **Tests**: `tests/` - Comprehensive test suite

---

## Testing

### Run All Tests

```bash
# From V1_implementation directory
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Run Specific Tests

```bash
# Test data loaders
pytest tests/data_loaders/ -v

# Test model components
pytest tests/test_models.py -v

# Test training
pytest tests/test_training.py -v
```

### Quick Component Testing

For quick testing during development:

```bash
# Test configuration
python training/config.py

# Test model
python models/v1_model.py

# Test data pipeline
python data/data_integrator.py
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Wrong data paths | Update `DataConfig` in `config.py` |
| `KeyError: column` | Column name mismatch | Check actual CSV columns |
| GPU out of memory | Batch/model too large | Reduce `batch_size` or `d_model` |
| Loss is NaN | Learning rate too high | Reduce to `1e-5`, check feature normalization |
| All predictions similar | Low feature variance | Check data, increase `lambda_slope` |
| Slow data loading | Processing all patients | Use `create_kfold_dataloaders()` which filters per fold |

### Data Pipeline Issues

**Problem**: "Scalers not fitted"
- **Solution**: Call `integrator.fit_scalers(prepared_data, train_patnos=train_ids)` before `create_feature_vectors()`

**Problem**: "Missing modality data"
- **Solution**: This is expected. Missing modalities are masked (Mask A). Ensure loaders handle missing data gracefully.

**Problem**: "No slopes computed"
- **Solution**: Patients need ≥3 visits with valid UPDRS totals to compute slopes. Check `min_visits` and data availability.

### Training Issues

**Problem**: "Validation loss not decreasing"
- **Solution**: 
  - Check learning rate (try `1e-5`)
  - Verify feature normalization is enabled
  - Check data quality (missing values, outliers)
  - Increase `lambda_slope` if slope loss dominates

**Problem**: "Out of memory"
- **Solution**:
  - Reduce `batch_size` (e.g., 16 or 8)
  - Reduce `max_seq_len` (e.g., 15)
  - Reduce `d_model` (e.g., 128)
  - Use gradient accumulation

---

## Next Steps

After V1 is working:

1. **Evaluate thoroughly**: Check performance by subgroup (LRRK2/GBA carriers, etc.)
2. **Error analysis**: Where does the model fail? Which patients/visits?
3. **Feature importance**: What matters most? (Ablation studies)
4. **Hyperparameter tuning**: Learning rate, architecture size, loss weights
5. **Model comparison**: Compare with baselines (linear regression, RNN, etc.)

---

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
pytest tests/ -v
```

### Run Data Integration
```bash
python data/data_integrator.py
```

### Train Model
```bash
# Recommended: Use unified CLI
python -m training.main

# Or programmatically
python training/train.py
```

### Key Files
- `training/config.py` - All settings
- `models/v1_model.py` - Model architecture
- `data/data_integrator.py` - Data pipeline
- `training/train.py` - Training loop

---

**Ready to start?** Follow the [Quick Start](#quick-start) section above! 🚀
