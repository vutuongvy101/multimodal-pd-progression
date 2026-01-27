# Disease Severity Prediction - Implementation Guide

## What Was Created

A complete modular inference system for predicting patient disease stage and severity at their next doctor's visit. Minimal edits to existing code - all new functionality in the `inference/` module.

---

## 📁 Files Created

```
inference/
├── __init__.py              # Module initialization
├── predictor.py             # Core prediction classes (PatientPredictor, BatchPredictor, DiseaseStage)
├── example_usage.py         # Usage examples and demos
├── commands.py              # Optional integration with main.py
└── README.md                # Complete documentation
```

---

## 🚀 Quick Start

### 1. Sample Prediction (No data needed)
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode sample
```

### 2. Batch Prediction (Your patient data in CSV)
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode batch \
    --patient-file patient_visits.csv \
    --output predictions.json
```

### 3. Interactive Prediction (Enter data manually)
```bash
python inference/example_usage.py \
    --checkpoint models/fold_1/best_checkpoint.pt \
    --mode interactive
```

---

## 📊 Usage Examples

### Example 1: Single Patient (Python)

```python
import pandas as pd
from inference.predictor import PatientPredictor, DiseaseStage

# Load patient's 4+ visits
visits = pd.DataFrame({
    'NP1RTOT': [5, 6, 7, 8],          # Non-motor experiences
    'NP2PTOT': [3, 3, 4, 4],          # Non-motor examination
    'NP3TOT': [25, 28, 32, 35],       # Motor (severity indicator)
    'NP4TOT': [1, 1, 2, 2],           # Motor complications
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

# Results
print(f"Current Stage: {result['current_severity']['stage']}")
print(f"Predicted Stage: {result['predicted_stage']}")
print(f"Predicted Motor Score (NP3TOT): {result['predicted_updrs']['NP3TOT']:.1f}")
print(f"Disease Progression: {result['slope']:.4f} points/visit")

# Interpretation
if result['predicted_stage'] in ['Severe PD', 'Very Severe PD']:
    print("⚠️  Advanced disease - Consider intensified treatment")
```

### Example 2: Multiple Patients (CSV Batch)

```python
import pandas as pd
from inference.predictor import BatchPredictor

# CSV file format:
# patient_id, visit_id, NP1RTOT, NP2PTOT, NP3TOT, NP4TOT, months_since_baseline
df = pd.read_csv('all_patients.csv')

# Group by patient
patients = {pid: group.reset_index(drop=True) 
            for pid, group in df.groupby('patient_id')}

# Batch predict
batch_predictor = BatchPredictor('models/fold_1/best_checkpoint.pt')
results = batch_predictor.predict_patients(patients, 'predictions.json')

# Results saved to predictions.json
# Each patient_id maps to their prediction
```

### Example 3: Disease Stage Reference

```python
from inference.predictor import DiseaseStage

# All stages
stages = DiseaseStage.get_all_stages()
# ['Normal/No PD', 'Mild PD', 'Moderate PD', 'Severe PD', 'Very Severe PD']

# Classify by score
score = 35.0  # NP3TOT (motor score)
stage = DiseaseStage.classify(score)
# Returns: 'Mild PD'

# Thresholds
print(DiseaseStage.THRESHOLDS)
# {
#     'Normal/No PD': (0, 20),
#     'Mild PD': (20, 41),
#     'Moderate PD': (41, 58),
#     'Severe PD': (58, 75),
#     'Very Severe PD': (75, 140)
# }
```

---

## 📋 Data Format

### CSV Input Format

```csv
patient_id,visit_id,visit_date,months_since_baseline,NP1RTOT,NP2PTOT,NP3TOT,NP4TOT,age,medication_count
P001,1,2022-01-15,0,5,3,25,1,65,2
P001,2,2022-02-15,1,6,3,28,1,65,2
P001,3,2022-03-15,2,7,4,32,2,65,3
P001,4,2022-04-15,3,8,4,35,2,65,3
```

**Required columns:**
- `patient_id`: Unique identifier
- `NP1RTOT`, `NP2PTOT`, `NP3TOT`, `NP4TOT`: UPDRS scores
- `months_since_baseline`: Timeline

**Minimum:** 4 visits per patient

---

## 📈 Output Results

### Single Patient Result

```python
{
    'predicted_updrs': {
        'NP1RTOT': 8.5,
        'NP2PTOT': 4.2,
        'NP3TOT': 38.1,      # Motor score
        'NP4TOT': 2.3
    },
    'predicted_stage': 'Mild PD',
    'slope': 0.45,            # +0.45 points/visit = deteriorating
    'current_severity': {
        'stage': 'Mild PD',
        'motor_score': 35.0
    },
    'confidence': {
        'NP3TOT': (34.2, 42.0)  # 95% CI
    }
}
```

### JSON Output (Batch Mode)

```json
{
  "P001": {
    "predicted_updrs": {...},
    "predicted_stage": "Mild PD",
    "slope": 0.45,
    "current_severity": {...}
  },
  "P002": {
    "predicted_updrs": {...},
    "predicted_stage": "Moderate PD",
    "slope": 0.72,
    "current_severity": {...}
  }
}
```

---

## 🎯 Disease Stages (Based on Motor Score - NP3TOT)

| Stage | Motor Score | Characteristics |
|-------|-------------|---|
| Normal/No PD | 0-20 | No PD |
| Mild PD | 20-41 | Subtle motor signs |
| Moderate PD | 41-58 | Obvious motor signs |
| Severe PD | 58-75 | Significant disability |
| Very Severe PD | 75+ | Advanced motor symptoms |

---

## 💡 Key Classes

### PatientPredictor
```python
from inference import PatientPredictor

predictor = PatientPredictor(
    checkpoint_path='path/to/model.pt',
    config_version='v1',    # or 'v2'
    device='cuda'           # or 'cpu'
)

# Single patient prediction
result = predictor.predict_next_visit(
    patient_visits,         # DataFrame with 4+ visits
    return_confidence=True,
    return_stage=True
)
```

### BatchPredictor
```python
from inference import BatchPredictor

batch_predictor = BatchPredictor(checkpoint_path, config_version, device)

# Multiple patients
results = batch_predictor.predict_patients(
    patient_data,           # Dict of patient_id -> DataFrame
    save_results='file.json'
)
```

### DiseaseStage
```python
from inference import DiseaseStage

# Classify score to stage
stage = DiseaseStage.classify(motor_score)

# Get all stages
all_stages = DiseaseStage.get_all_stages()

# Access thresholds
thresholds = DiseaseStage.THRESHOLDS
```

---

## 🔧 Minimal Code Changes

**No changes needed to existing training code!**

Just use the inference module:

```python
# Option 1: Standalone script
python inference/example_usage.py --checkpoint ... --mode batch --patient-file ...

# Option 2: Import in your code
from inference import PatientPredictor

predictor = PatientPredictor(checkpoint_path)
result = predictor.predict_next_visit(patient_visits)
```

---

## 📚 Documentation

Complete documentation available in:
- **README.md** - Full usage guide with examples
- **predictor.py** - Inline docstrings for all classes
- **example_usage.py** - Working code examples

---

## ✅ Checklist

- ✓ PatientPredictor class created
- ✓ BatchPredictor for multiple patients
- ✓ DiseaseStage classification system
- ✓ Disease severity thresholds (MDS-UPDRS based)
- ✓ Example usage script with 3 modes
- ✓ Complete documentation
- ✓ Confidence interval estimation
- ✓ Minimal dependencies on existing code
- ✓ Modular design (easy to extend)

---

## 🚦 Next Steps

1. **Test with sample data:**
   ```bash
   python inference/example_usage.py \
       --checkpoint models/fold_1/best_checkpoint.pt \
       --mode sample
   ```

2. **Prepare your patient data:**
   - CSV format with required columns
   - At least 4 visits per patient

3. **Run batch predictions:**
   ```bash
   python inference/example_usage.py \
       --checkpoint models/fold_1/best_checkpoint.pt \
       --mode batch \
       --patient-file patient_data.csv \
       --output results.json
   ```

4. **Integrate into your workflow:**
   - Use as standalone script, or
   - Import PatientPredictor class in your code

---

## 📞 Troubleshooting

| Error | Solution |
|-------|----------|
| "Need at least 4 visits" | Ensure ≥4 visit records per patient |
| "Checkpoint not found" | Verify correct path to .pt file |
| "Feature not found" | Check CSV columns match config features |
| "Out of memory" | Use `--device cpu` instead |

See README.md for more troubleshooting tips.

---

## Implementation Summary

**What you requested:**
- Predict disease severity/stage for next visit
- Use 4 historical patient records
- Modular design with minimal edits to existing code

**What was delivered:**
- ✅ `PatientPredictor` class for single patient predictions
- ✅ `BatchPredictor` class for multiple patient predictions
- ✅ `DiseaseStage` classification based on motor scores
- ✅ Support for 3 usage modes (sample/batch/interactive)
- ✅ Confidence interval estimation
- ✅ Complete example usage script
- ✅ Comprehensive documentation
- ✅ No changes needed to existing training code
- ✅ Production-ready modular design

**Ready to use!**
