# Training Metrics Enhancement

## Overview

Your training pipeline has been enhanced to track and display **model accuracy metrics** along with validation loss on each epoch. This makes it easier to visualize model capability and determine if fine-tuning is necessary.

## What's New

### 1. **Accuracy Tracking (R² Score)**

For regression models (predicting UPDRS scores), accuracy is measured using **R² (coefficient of determination)**:
- **R² = 1.0**: Perfect predictions
- **R² = 0.0**: Model performs as well as predicting the mean
- **R² < 0.0**: Model performs worse than the mean

The R² score is computed on the validation set for each epoch and displayed in the training log.

### 2. **Enhanced Training Output**

Each epoch now shows:
```
Epoch 5:
  Train Loss: 0.8234 (next: 0.7124, slope: 0.1110)
  Val Loss:   0.9456 (next: 0.8345, slope: 0.1111)
  Val R² (Accuracy): 0.6543          <-- NEW
  LR: 5.00e-04
```

### 3. **Updated Training History**

The `training_history.json` now includes:
- `val_accuracy`: R² scores per epoch
- `val_r2`: Same as above (two formats for flexibility)

### 4. **K-Fold Results Enhanced**

The k-fold cross-validation results now show:
```
Per-fold results:
  Fold 1: Val Loss = 0.9123 (Best: 0.8901 at epoch 8), R² = 0.6234
  Fold 2: Val Loss = 0.9456 (Best: 0.9012 at epoch 7), R² = 0.5987
  ...

Average Validation R² (Accuracy): 0.6110 ± 0.0245
```

### 5. **Multi-Modal Comparison Updated**

When comparing different modality combinations, the comparison now includes R² metrics:
```
All configurations (sorted by val loss):
  static+motor: 0.8234 (R² = 0.6543)
  all: 0.8456 (R² = 0.6234)
  motor_only: 0.9123 (R² = 0.5876)
```

## Files Modified

1. **`training/train.py`**
   - Updated `validate()` method to compute comprehensive metrics including R²
   - Modified `train()` loop to track and display accuracy per epoch
   - Training history now stores R² values

2. **`training/kfold_trainer.py`**
   - Updated fold result tracking to include accuracy metrics
   - Enhanced summary statistics to show mean/std R² across folds
   - Improved per-fold result display

3. **`training/multi_modal_trainer.py`**
   - Enhanced `compare_results()` to include R² metrics in comparisons

## New File

**`training/visualize_training.py`** - Visualization utilities

### Usage:

```bash
# Visualize single model training
python training/visualize_training.py --history path/to/training_history.json

# Visualize k-fold results
python training/visualize_training.py --kfold path/to/kfold_results.json

# Save plots to specific directory
python training/visualize_training.py --history path/to/training_history.json --output-dir ./plots

# Print summary without displaying plots
python training/visualize_training.py --history path/to/training_history.json --no-show
```

### Generated Plots:

1. **training_metrics.png**: 4-subplot figure showing:
   - Overall loss (train vs val)
   - Next-visit loss (train vs val)
   - Slope loss (train vs val)
   - Validation R² (accuracy) over epochs

2. **kfold_results.png**: 2-subplot figure showing:
   - Validation loss per fold with mean
   - Validation R² per fold with mean

## Interpreting Accuracy Metrics

### R² Score Interpretation:
- **R² > 0.8**: Excellent model performance
- **R² 0.6 - 0.8**: Good model performance
- **R² 0.4 - 0.6**: Fair model performance
- **R² < 0.4**: Poor model performance

### Deciding If Fine-Tuning is Necessary:

1. **Monitor convergence**: Does R² improve or plateau?
   - If plateau → consider fine-tuning or architectural changes
   - If still improving → continue training

2. **Check loss-accuracy alignment**: Do validation loss and R² move in opposite directions?
   - Yes → model is learning correctly
   - No → potential issue with loss function or targets

3. **Compare across modalities**: Which modality combination achieves highest R²?
   - Use multi-modal mode to find optimal feature set
   - Fine-tune the best configuration

## Example Training Command

```bash
# Standard k-fold training with accuracy tracking
python training/main.py --mode kfold --n-splits 5 --max-epochs 50

# After training, visualize results
python training/visualize_training.py \
    --history experiments/fold_1/training_history.json \
    --output-dir ./visualizations
```

## Technical Details

### Accuracy Computation (from `metrics.py`):

The R² score is computed per UPDRS target (NP1RTOT, NP2PTOT, NP3TOT, NP4TOT), then averaged:

$$R^2 = 1 - \frac{\sum(y_{true} - y_{pred})^2}{\sum(y_{true} - \bar{y})^2}$$

Where:
- $y_{true}$: actual next-visit scores
- $y_{pred}$: model predictions
- $\bar{y}$: mean of actual scores

## Backward Compatibility

All changes are **backward compatible**:
- Existing training scripts work without modification
- Old checkpoint files can still be loaded
- Training history JSON includes new fields but maintains old ones

## Next Steps

1. Run training with your preferred configuration
2. Monitor the `Val R² (Accuracy)` metric each epoch
3. After training, use visualization script to plot results
4. If R² plateaus early, consider:
   - Increasing learning rate
   - Using different modality combinations
   - Adjusting regularization parameters
   - Collecting more data

