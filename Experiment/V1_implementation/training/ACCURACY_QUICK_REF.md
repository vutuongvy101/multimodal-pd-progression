# Quick Reference: Accuracy Metrics

## What Changed?

Your model training now displays **R² (accuracy)** alongside loss for each epoch:

```
Epoch 10:
  Train Loss: 0.8234 (next: 0.7124, slope: 0.1110)
  Val Loss:   0.9456 (next: 0.8345, slope: 0.1111)
  Val R² (Accuracy): 0.6543          ← NEW: This shows model prediction accuracy
  LR: 5.00e-04
```

## Understanding R²

**R² (R-squared)** measures how well predictions match actual values on a scale from -∞ to 1.0:

| R² Value | Interpretation | Model Quality |
|----------|----------------|---------------|
| 1.0      | Perfect fit    | Excellent |
| 0.8-1.0  | Very strong    | Excellent |
| 0.6-0.8  | Strong         | Good |
| 0.4-0.6  | Moderate       | Fair |
| 0.0-0.4  | Weak           | Poor |
| < 0.0    | Very poor      | Worse than guessing mean |

## Where to Find Metrics

### 1. **During Training** (Console output)
```
Val R² (Accuracy): 0.6543
```

### 2. **After Training** (JSON file)
```
# training_history.json
{
  "val_loss": [...],
  "val_accuracy": [0.5234, 0.5567, 0.6123, ...],  ← R² per epoch
  "val_r2": [0.5234, 0.5567, 0.6123, ...]         ← Same data
}
```

### 3. **K-Fold Results**
```
# kfold_results.json
{
  "summary": {
    "mean_val_r2": 0.6234,
    "std_val_r2": 0.0245
  },
  "fold_results": [
    {"val_accuracy": 0.6543, ...},
    {"val_accuracy": 0.6012, ...},
    ...
  ]
}
```

## Visualizing Results

```bash
# Create plots of training metrics
python training/visualize_training.py \
    --history path/to/training_history.json \
    --output-dir ./plots
```

This generates:
- **training_metrics.png**: 4 plots (losses + R²)
- **kfold_results.png**: Fold comparison

## Decision Guide: Do I Need to Fine-Tune?

### Question 1: Is R² still improving?
```
Epoch 20: R² = 0.5890
Epoch 30: R² = 0.6234  ✓ Still improving
Epoch 40: R² = 0.6245  ✓ Still improving
Epoch 50: R² = 0.6246  ✗ Plateau (stopped improving)
```
→ If plateaued for 10+ epochs → consider fine-tuning

### Question 2: Is R² > 0.6?
```
Final Val R²: 0.6543  ✓ Good performance
```
→ Generally acceptable; decide based on requirements

### Question 3: How does R² compare across modality combinations?
```
All modalities:    R² = 0.6234  ← Higher
Motor only:        R² = 0.5234  ← Lower
```
→ Use best modality combination; fine-tune that

## Common Actions

### Run baseline training
```bash
python training/main.py --mode kfold --n-splits 5 --max-epochs 50
```

### Check if model needs fine-tuning
```bash
python training/visualize_training.py --history fold_1/training_history.json
```
Look at: Is the green line (R²) still rising at epoch 50?

### Try different modalities
```bash
python training/main.py --mode multi_modal --max-epochs 50
```
Compare R² scores across modalities; use best one for fine-tuning.

### Fine-tune best model
```bash
python training/main.py \
    --mode single \
    --checkpoint models/best_checkpoint.pt \
    --freeze-backbone \
    --learning-rate 0.00001 \
    --max-epochs 20
```

## Metrics Storage

| File | Location | Content |
|------|----------|---------|
| training_history.json | `{model_save_dir}/training_history.json` | Per-epoch metrics |
| kfold_results.json | `{model_save_dir}/kfold_results.json` | Fold aggregates |
| training_metrics.png | `{model_save_dir}/training_metrics.png` | Plot of metrics |
| kfold_results.png | `{model_save_dir}/kfold_results.png` | Fold comparison plot |

## Example: Reading Results Programmatically

```python
import json

# Load training history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Get final R²
final_r2 = history['val_accuracy'][-1]
best_r2 = max(history['val_accuracy'])
best_epoch = history['val_accuracy'].index(best_r2)

print(f"Best R²: {best_r2:.4f} at epoch {best_epoch}")
print(f"Final R²: {final_r2:.4f}")

# Load k-fold results
with open('kfold_results.json', 'r') as f:
    results = json.load(f)

mean_r2 = results['summary']['mean_val_r2']
std_r2 = results['summary']['std_val_r2']

print(f"Average R² across folds: {mean_r2:.4f} ± {std_r2:.4f}")
```

## Key Thresholds

- **Train = ✓ Ok**: Loss and R² converge
- **Val Loss high but R² good** = Possible overfit to training
- **R² stuck < 0.3 after 20 epochs** = Model architecture or data issue
- **R² improves to 0.7+** = Consider production-ready

---

For detailed information, see [ACCURACY_METRICS_README.md](ACCURACY_METRICS_README.md)
