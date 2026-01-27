# Disease Stage Prediction Guide

## Overview

Predict a patient's disease severity and expected UPDRS scores at their next doctor's visit using their historical visit data (minimum 4 visits).

## Quick Start

### Option 1: Sample Prediction (No data needed)
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode sample
```

### Option 2: Your Own Patient Data (CSV file)
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode batch \
    --patient-file patient_visits.csv \
    --output predictions.json
```

### Option 3: Interactive Input
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode interactive
```

---

## Features

### ✨ What You Get

1. **Next Visit UPDRS Predictions**
   - NP1RTOT (Non-motor experiences)
   - NP2PTOT (Non-motor examination)
   - NP3TOT (Motor examination)
   - NP4TOT (Motor complications)

2. **Disease Stage Classification**
   - Normal/No PD
   - Mild PD
   - Moderate PD
   - Severe PD
   - Very Severe PD

3. **Progression Analysis**
   - Disease progression rate (slope)
   - Direction: Deteriorating/Stable/Improving
   - Confidence intervals (95% CI)

4. **Current Severity Assessment**
   - Current disease stage
   - Current UPDRS scores
   - Motor score (indicator of severity)

---

## Data Format

### CSV Format (Batch Mode)

```csv
patient_id,visit_id,visit_date,months_since_baseline,NP1RTOT,NP2PTOT,NP3TOT,NP4TOT,age,medication_count
P001,1,2022-01-15,0,5,3,25,1,65,2
P001,2,2022-02-15,1,6,3,28,1,65,2
P001,3,2022-03-15,2,7,4,32,2,65,3
P001,4,2022-04-15,3,8,4,35,2,65,3
P002,1,2022-01-20,0,3,2,15,0,60,1
...
```

**Required columns:**
- `patient_id`: Unique patient identifier
- `NP1RTOT`, `NP2PTOT`, `NP3TOT`, `NP4TOT`: UPDRS scores
- `months_since_baseline`: Months since first visit (or similar timeline)

**Optional columns:**
- Any other features used during training (age, medications, etc.)
- Columns not matching feature names are automatically ignored

### Python Usage (Programmatic)

```python
import pandas as pd
from inference.predictor import PatientPredictor

# Load patient data
visits = pd.DataFrame({
    'NP1RTOT': [5, 6, 7, 8],
    'NP2PTOT': [3, 3, 4, 4],
    'NP3TOT': [25, 28, 32, 35],
    'NP4TOT': [1, 1, 2, 2],
    'months_since_baseline': [0, 1, 2, 3]
})

# Initialize predictor
predictor = PatientPredictor(
    checkpoint_path='models/fold_1/best_checkpoint.pt',
    config_version='v1',
    device='cuda'
)

# Make prediction
result = predictor.predict_next_visit(visits)

# Access results
print(f"Predicted Stage: {result['predicted_stage']}")
print(f"Predicted NP3TOT: {result['predicted_updrs']['NP3TOT']:.2f}")
print(f"Disease Slope: {result['slope']:.4f}")
```

---

## Understanding Results

### Disease Stages

Based on MDS-UPDRS Part III (Motor Score - NP3TOT):

| Stage | NP3TOT Range | Characteristics |
|-------|----------|---|
| Normal/No PD | 0-20 | No apparent parkinsonian features |
| Mild PD | 20-41 | Subtle or limited motor signs |
| Moderate PD | 41-58 | More obvious motor signs, functional limitations |
| Severe PD | 58-75 | Significant motor disability |
| Very Severe PD | 75+ | Advanced motor symptoms, significant disability |

### UPDRS Score Interpretation

- **NP1RTOT**: Non-motor experiences (cognitive/mood)
  - Range: 0-40
  - Higher = More non-motor symptoms

- **NP2PTOT**: Non-motor examination (sleep, autonomic, etc.)
  - Range: 0-30
  - Higher = More examination findings

- **NP3TOT**: Motor examination (primary severity indicator)
  - Range: 0-132
  - Higher = More motor impairment
  - **Most important for disease stage classification**

- **NP4TOT**: Motor complications (dyskinesias, fluctuations)
  - Range: 0-24
  - Higher = More complications

### Progression Rate (Slope)

- **Positive slope**: Disease deteriorating (scores increasing)
- **Negative slope**: Disease improving (scores decreasing)
- **Near-zero slope**: Stable disease

Example interpretation:
```
Slope = 0.5 points/visit
→ Patient's motor score increases ~0.5 points per visit
→ Disease is slowly deteriorating
```

---

## Working with Results

### Result Dictionary Structure

```python
{
    'predicted_updrs': {
        'NP1RTOT': 8.5,      # Predicted non-motor experiences
        'NP2PTOT': 4.2,      # Predicted non-motor examination
        'NP3TOT': 38.1,      # Predicted motor score (key metric)
        'NP4TOT': 2.3        # Predicted motor complications
    },
    'predicted_stage': 'Mild PD',  # Disease stage classification
    'slope': 0.45,                 # Disease progression rate
    'current_severity': {
        'stage': 'Mild PD',        # Current stage
        'motor_score': 35.0,       # Current motor score
        'updrs_scores': {...}      # Current UPDRS scores
    },
    'confidence': {
        'NP3TOT': (34.2, 42.0),    # 95% confidence interval
        ...
    },
    'stage_thresholds': {          # Reference table
        'Normal/No PD': (0, 20),
        'Mild PD': (20, 41),
        ...
    }
}
```

### Accessing Results in Python

```python
# Single prediction
result = predictor.predict_next_visit(visits)

# Get predicted stage
stage = result['predicted_stage']
print(f"Next stage: {stage}")

# Get predicted motor score
motor_score = result['predicted_updrs']['NP3TOT']
print(f"Predicted motor score: {motor_score:.1f}")

# Check progression
slope = result['slope']
if slope > 0.2:
    print("Disease is deteriorating - consider treatment adjustment")
elif slope < -0.1:
    print("Disease is improving")
else:
    print("Disease is stable")

# Access confidence interval
motor_ci = result['confidence']['NP3TOT']
print(f"Motor score: {motor_score:.1f} [95% CI: {motor_ci[0]:.1f}-{motor_ci[1]:.1f}]")
```

---

## Batch Predictions

### Predicting for Multiple Patients

```python
import pandas as pd
from inference.predictor import BatchPredictor

# Load data (patient_id, visit_id, UPDRS scores, etc.)
df = pd.read_csv('patient_visits.csv')

# Initialize batch predictor
batch_predictor = BatchPredictor(
    checkpoint_path='models/fold_1/best_checkpoint.pt',
    config_version='v1',
    device='cuda'
)

# Group by patient
patient_groups = {
    pid: group.reset_index(drop=True)
    for pid, group in df.groupby('patient_id')
}

# Predict for all patients
results = batch_predictor.predict_patients(
    patient_groups,
    save_results='predictions.json'
)

# Results are saved to predictions.json
# Each patient_id maps to their prediction
```

### Results JSON Format

```json
{
  "P001": {
    "predicted_updrs": {
      "NP1RTOT": 8.5,
      "NP2PTOT": 4.2,
      "NP3TOT": 38.1,
      "NP4TOT": 2.3
    },
    "predicted_stage": "Mild PD",
    "slope": 0.45,
    "current_severity": {...},
    "confidence": {...}
  },
  "P002": {
    ...
  }
}
```

---

## Minimum Data Requirements

- **Minimum visits**: 4
- **Recommended visits**: 4-10 (model trained on sequences up to 10)
- **Required columns**: NP1RTOT, NP2PTOT, NP3TOT, NP4TOT, months_since_baseline
- **Optional columns**: Any features used during training

---

## Tips for Best Results

1. **Use consistent data collection**
   - Same UPDRS assessment protocol
   - Similar time intervals between visits (ideally monthly)

2. **Ensure data quality**
   - Remove obvious data entry errors
   - Handle missing values appropriately
   - Verify UPDRS score ranges (0-max for each)

3. **Interpret conservatively**
   - Use confidence intervals
   - Don't over-interpret small changes
   - Consider confidence alongside point estimates

4. **Check for data issues**
   - If R² was low in training (< 0.4), predictions may be unreliable
   - If patient is very different from training population, predictions may not apply
   - Very recent patients may have limited historical data

---

## Troubleshooting

### Error: "Need at least 4 visits"
**Solution**: Ensure patient has at least 4 visit records before prediction

### Error: "Checkpoint not found"
**Solution**: Verify path to trained model checkpoint (e.g., `models/fold_1/best_checkpoint.pt`)

### Error: "Feature not found in DataFrame"
**Solution**: Check that CSV has required columns matching config features

### Predictions are all zeros or NaN
**Solution**: Check data format and ensure UPDRS scores are numeric (not strings)

### Out of memory error
**Solution**: Use `--device cpu` instead of cuda, or reduce batch size

---

## Customization

### Using Different Model Versions

```python
# For config_v2
predictor = PatientPredictor(
    checkpoint_path='path/to/checkpoint.pt',
    config_version='v2',  # ← Change this
    device='cuda'
)
```

### Custom Disease Stage Thresholds

```python
from inference.predictor import DiseaseStage

# Modify thresholds
DiseaseStage.THRESHOLDS = {
    DiseaseStage.MILD: (0, 30),      # Adjust values
    DiseaseStage.MODERATE: (30, 50),
    # ... etc
}
```

### Adding Your Own Features

Extend `_extract_features()` method in PatientPredictor to include additional modalities or custom preprocessing.

---

## Integration with Your Workflow

### Option A: Standalone Script
```bash
python inference/example_usage.py --checkpoint ... --mode batch --patient-file ...
```

### Option B: Python Module
```python
from inference.predictor import PatientPredictor

predictor = PatientPredictor(checkpoint_path, config_version, device)
result = predictor.predict_next_visit(patient_visits)
```

### Option C: Added to Main Script (Optional)
Edit `training/main.py` to add `predict` subcommand:
```bash
python training/main.py predict --checkpoint ... --patient-data ...
```

---

## Files Overview

```
inference/
├── __init__.py           # Module initialization
├── predictor.py          # Core prediction classes (PatientPredictor, BatchPredictor)
├── commands.py           # Integration with main.py (optional)
└── example_usage.py      # Standalone usage examples and demos
```

## Example Workflows

### Workflow 1: Quick Demo
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode sample
```

### Workflow 2: Single Patient Prediction
```python
import pandas as pd
from inference import PatientPredictor

# Your patient's visit data
visits = pd.read_csv('patient_123_visits.csv')

# Predict
predictor = PatientPredictor('models/fold_1/best_checkpoint.pt')
result = predictor.predict_next_visit(visits)

print(f"Predicted stage: {result['predicted_stage']}")
print(f"Motor score: {result['predicted_updrs']['NP3TOT']:.1f}")
```

### Workflow 3: Batch Processing
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode batch \
    --patient-file all_patients.csv \
    --output predictions.json
```

---

**Questions?** Check example_usage.py for more code examples.
