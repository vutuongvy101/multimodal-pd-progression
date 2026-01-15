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
11. [Troubleshooting](#troubleshooting)
12. [Next Steps](#next-steps)

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
| **Dual Objectives** | Predicts next-visit NP3TOT + patient-level progression slope |

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
│   Output: NP3TOT_{t+1}  │     │   Output: slope         │
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

### 3. Configure Data Paths

Edit `training/config.py` to point to your PPMI data:

```python
@dataclass
class DataConfig:
    base_dir: str = "../../ppmi_pd"  # Update this
    participant_status: str = "Participant_Status_14Dec2025.csv"
    genetic_consensus: str = "../genetics/iu_genetic_consensus_*.csv"
    # ... update other paths
```

### 4. Test Components

```bash
cd V1_implementation

# Test configuration
python training/config.py

# Test model (requires PyTorch)
python models/v1_model.py
```

### 5. Prepare Data

Data loading is modularized into 5 independent tasks (see [Data Pipeline](#data-pipeline)):

```bash
cd data
python data_integrator.py
```

### 6. Train Model

```bash
python training/train.py
```

---

## Project Structure

```
V1_implementation/
├── README.md                    # This file
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
└── data/                        # Data loading & processing
    ├── base_loader.py          # Base classes
    ├── data_integrator.py      # Combines all loaders
    ├── dataset.py              # PyTorch Dataset & DataLoader
    ├── TASK_ASSIGNMENTS.md     # Parallel task guide
    └── loaders/                # Individual data loaders
        ├── genetics_loader.py
        ├── demographics_loader.py
        ├── updrs_loader.py
        ├── clinical_loader.py
        └── medication_loader.py
```

---

## Data Pipeline

### Overview

Data loading is modularized into **5 independent tasks** that can be developed in parallel:

| Task | Loader | Data Source | Estimated Time |
|------|--------|-------------|----------------|
| 1 | `genetics_loader.py` | Genetics (PRS, variants, PCs) | 30-45 min |
| 2 | `demographics_loader.py` | Demographics (age, sex, etc.) | 20-30 min |
| 3 | `updrs_loader.py` | UPDRS Parts I-IV | 45-60 min |
| 4 | `clinical_loader.py` | MoCA, ESS, SCOPA-AUT | 30-45 min |
| 5 | `medication_loader.py` | LEDD, medication history | 20-30 min |

### Running Data Pipeline

**Individual loader testing:**
```bash
cd data/loaders
python genetics_loader.py      # Test genetics
python updrs_loader.py         # Test UPDRS
```

**Full integration:**
```bash
cd data
python data_integrator.py
```

**Output:**
```
data/processed/
├── static_data.csv         # Patient-level features
├── longitudinal_data.csv   # Visit-level features
└── slopes_data.csv         # Progression slopes
```

### Parallel Development

For teams, see `data/TASK_ASSIGNMENTS.md` for detailed division of work.

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
- Output: Predicted NP3TOT at t+1
- Loss: MSE

**Slope Head:**
- Input: Pooled sequence representation
- Output: Patient-level progression slope
- Target: Empirical slope from ≥3 visits
- Loss: MSE

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
| Next-visit MAE | < 5 points | On NP3TOT (0-132 scale) |
| Slope correlation | r > 0.5 | With empirical slopes |
| ON vs OFF gap | 5-10 points | Medication effect captured |

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
    predict_totals: List[str] = ['NP3TOT']  # What to predict
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
| Part I | Minimal (~0-2 pts) | Non-dopaminergic |
| Part II | Moderate (~5-10 pts) | Patient-reported |
| Part III | **Large (~10-20 pts)** | Main outcome |
| Part IV | N/A | These ARE medication effects |

### Recommendation

Start with predicting **NP3TOT only** (default). It's:
- Most widely used progression outcome
- Clinician-observed (standardized)
- Most medication-sensitive

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

### Testing Components

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

### Run Data Integration
```bash
cd data && python data_integrator.py
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
