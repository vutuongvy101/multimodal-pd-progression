"""
Baseline XGBoost models using the same prepared data & splits as the V1 transformer.

This module provides utilities to turn the longitudinal PyTorch datasets into
fixed-length tabular matrices, train one or more XGBoost regressors, and
evaluate metrics using the same helpers already used by the transformer code.

The idea is to use `create_dataloaders()` (or `create_kfold_dataloaders()`)
to obtain training/validation/test `PPMILongitudinalDataset` instances.  Those
datasets already have scalers fitted on the training set and therefore contain
normalized feature vectors.  We iterate them to build per-visit tabular rows
with static + current-visit features.  Labels correspond to the next visit's
UPDRS totals (matching the transformer training target).  A separate patient-
level table is also created for slope prediction.

Because XGBoost cannot handle missing targets, we train one regressor per
output dimension and carefully mask out rows where the corresponding label is
missing.  The metrics routines from `training/metrics.py` are reused by
converting numpy outputs to torch tensors.

Usage example (inside a training script):

    from data.dataset import create_dataloaders
    from models.xgboost_model import run_xgb_experiment

    train_loader, val_loader, test_loader = create_dataloaders(
        prepared_data, config, num_workers=4, train_ratio=0.8, val_ratio=0.1
    )

    results = run_xgb_experiment(
        train_dataset=train_loader.dataset,
        val_dataset=val_loader.dataset,
        test_dataset=test_loader.dataset,
        params={...},
    )

    print(results['next_visit_metrics'])
    print(results['slope_metrics'])


Notes
-----
* The tabular feature engineering here is intentionally simple: it concatenates
  static features, the current visit's modality vectors, and a handful of time
  features.  You can extend `build_next_visit_tabular()` and
  `build_slope_tabular()` with additional aggregates or domain-specific
elements as desired.
* The functions return metadata (attention masks / time deltas) so that the
  same call to `compute_comprehensive_metrics()` can be applied.

"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

import torch
from training.metrics import compute_comprehensive_metrics


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _flatten_visit(visit: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a 1‑D vector containing all modality values for a single visit."""
    parts: List[np.ndarray] = []
    for key in ['motor_values', 'nonmotor_values', 'med_values', 'age_at_visit_values']:
        if key in visit:
            arr = np.asarray(visit[key], dtype=np.float32)
            parts.append(arr.ravel())
    # time_months is a scalar
    parts.append(np.asarray([visit.get('time_months', 0.0)], dtype=np.float32))
    if 'updrs_totals' in visit:
        # nothing - we don't include target here
        pass
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)


def build_next_visit_tabular(
    dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build tabular input/target matrices from a longitudinal dataset.

    Each generated row corresponds to a visit *t* for which a "next visit"
    exists.  The feature vector is composed of static features plus the flattened
    modalities for visit *t*.  The target vector is the UPDRS totals recorded at
    visit *t+1*.

    Parameters
    ----------
    dataset: PPMILongitudinalDataset
        Any dataset produced by :func:`data.dataset.create_dataloaders` or the
        k-fold equivalent.  The scalers are assumed to have been fit already on
        the appropriate training data.

    Returns
    -------
    X: np.ndarray, shape (N, F)
        Feature matrix.
    y: np.ndarray, shape (N, K)
        Next-visit targets.  Missing labels are filled with 0 but accompanied by
        ``label_mask`` below.
    label_mask: np.ndarray, shape (N, K)
        1 where the corresponding entry in ``y`` is a valid target, 0 otherwise.
    delta_t: np.ndarray, shape (N,)
        Time difference (months) between the current visit and the next visit
        (useful for Δt bucket metrics).
    """
    rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    deltas: List[float] = []

    for patno in dataset.patient_ids:
        static_vec = dataset.static_data[patno]['values']
        visits = dataset.longitudinal_data[patno]
        n = len(visits)
        if n < 2:
            continue
        for t in range(n - 1):
            cur = visits[t]
            nxt = visits[t + 1]
            # feature row: static + flattened current visit
            feat = np.concatenate([static_vec, _flatten_visit(cur)])
            rows.append(feat)
            # target vector and mask
            yvec = np.asarray(nxt.get('updrs_totals', np.zeros(0)), dtype=np.float32)
            mask = (~np.isnan(yvec)).astype(np.float32)
            targets.append(np.nan_to_num(yvec, nan=0.0))
            masks.append(mask)
            # delta t
            deltas.append(float(nxt.get('time_months', 0.0) - cur.get('time_months', 0.0)))

    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X = np.vstack(rows)
    y = np.vstack(targets)
    label_mask = np.vstack(masks)
    delta_t = np.asarray(deltas, dtype=np.float32)

    return X, y, label_mask, delta_t


def build_slope_tabular(dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Build a patient-level table for predicting progression slopes.

    Uses the *last* available visit for each patient as the feature vector;
    static features plus the flattened final visit.  Patients without any slope
    information are excluded.

    Returns
    -------
    X: np.ndarray, shape (M, F)
    y: np.ndarray, shape (M, K)
    """
    rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for patno in dataset.patient_ids:
        slope_dict = dataset.slopes.get(patno, {})
        # slope_dict may be a float (legacy) or a dict of 4 entries
        if isinstance(slope_dict, dict):
            yvec = np.array([
                slope_dict.get('NP1RTOT_slope', np.nan),
                slope_dict.get('NP2PTOT_slope', np.nan),
                slope_dict.get('NP3TOT_slope', np.nan),
                slope_dict.get('NP4TOT_slope', np.nan),
            ], dtype=np.float32)
        else:
            # assume the legacy single NP3TOT_slope
            yvec = np.array([np.nan, np.nan, float(slope_dict), np.nan], dtype=np.float32)
        mask = ~np.isnan(yvec)
        if not mask.any():
            continue
        visits = dataset.longitudinal_data[patno]
        if len(visits) == 0:
            continue
        last = visits[-1]
        feat = np.concatenate([dataset.static_data[patno]['values'], _flatten_visit(last)])
        rows.append(feat)
        targets.append(np.nan_to_num(yvec, nan=0.0))

    if len(rows) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    return np.vstack(rows), np.vstack(targets)


# ---------------------------------------------------------------------------
# XGBoost training helpers
# ---------------------------------------------------------------------------

def train_xgb_multitarget(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    mask_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> List[xgb.Booster]:
    """Train one XGB regressor per output dimension.

    Rows where ``mask[...] == 0`` are excluded for the corresponding target.
    """
    params = params or {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    n_targets = y_train.shape[1]
    models: List[xgb.Booster] = []

    for k in range(n_targets):
        idx_train = mask_train[:, k] == 1
        dtrain = xgb.DMatrix(X_train[idx_train], label=y_train[idx_train, k])
        evals = [(dtrain, 'train')]
        dval = None
        if X_val is not None and y_val is not None and mask_val is not None:
            idx_val = mask_val[:, k] == 1
            dval = xgb.DMatrix(X_val[idx_val], label=y_val[idx_val, k])
            evals.append((dval, 'val'))
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if dval is not None else None,
            verbose_eval=False,
        )
        models.append(bst)

    return models


def predict_multimodel(models: List[xgb.Booster], X: np.ndarray) -> np.ndarray:
    """Run each booster on X and stack the results into an [N, K] array."""
    preds = [m.predict(xgb.DMatrix(X)) for m in models]
    return np.vstack(preds).T


# ---------------------------------------------------------------------------
# High-level experiment orchestration
# ---------------------------------------------------------------------------

def run_xgb_experiment(
    train_dataset,
    val_dataset,
    test_dataset,
    target_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> Dict[str, Any]:
    """Run a full XGBoost baseline: train/val/test and compute metrics.

    Returns a dict containing:
        - next_visit_metrics
        - slope_metrics
        - models_next_visit: list of boosters
        - models_slope: list of boosters
    """
    # build tabular data
    X_tr, y_tr, mask_tr, dt_tr = build_next_visit_tabular(train_dataset)
    X_val, y_val, mask_val, dt_val = build_next_visit_tabular(val_dataset)
    X_te, y_te, mask_te, dt_te = build_next_visit_tabular(test_dataset)

    print(f"XGBoost: training rows={X_tr.shape[0]}, features={X_tr.shape[1]}")

    # default names if caller did not provide them
    if target_names is None:
        target_names = []

    models_next = train_xgb_multitarget(
        X_tr, y_tr, mask_tr, X_val, y_val, mask_val,
        params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    # predictions
    y_pred_te = predict_multimodel(models_next, X_te)

    # build artificial sequence-of-length-2 tensors so that the "shift"
    # implemented in compute_comprehensive_metrics produces (pred, true)
    # pairs identical to our tabular rows.
    B, K = y_pred_te.shape
    preds_seq = torch.zeros((B, 2, K), dtype=torch.float32)
    targets_seq = torch.zeros((B, 2, K), dtype=torch.float32)
    mask_seq = torch.zeros((B, 2, K), dtype=torch.float32)
    time_seq = torch.zeros((B, 2), dtype=torch.float32)

    preds_seq[:, 0, :] = torch.from_numpy(y_pred_te)
    targets_seq[:, 1, :] = torch.from_numpy(y_te)
    mask_seq[:, 1, :] = torch.from_numpy(mask_te)
    # use delta_t as the time difference between position 0 and 1;
    # any constant offset cancels during shifting so this is sufficient
    time_seq[:, 0] = torch.from_numpy(dt_te)
    time_seq[:, 1] = torch.from_numpy(dt_te)

    next_visit_metrics = compute_comprehensive_metrics(
        {'next_visit': preds_seq, 'slope': torch.zeros((B, K))},
        {'next_visit': targets_seq, 'next_visit_mask': mask_seq, 'slope': torch.zeros((B, K))},
        attention_mask=torch.ones((B, 2)),
        target_names=target_names,
        time_months=time_seq,
    )

    # slope model
    Xs_tr, ys_tr = build_slope_tabular(train_dataset)
    Xs_te, ys_te = build_slope_tabular(test_dataset)
    slope_models = train_xgb_multitarget(Xs_tr, ys_tr, ~np.isnan(ys_tr))
    ys_pred = predict_multimodel(slope_models, Xs_te)
    # slope metrics use helper directly on numpy arrays
    slope_metrics = compute_comprehensive_metrics(
        {'next_visit': torch.zeros((ys_pred.shape[0], 1, ys_pred.shape[1])), 'slope': torch.from_numpy(ys_pred).float()},
        {'next_visit': torch.zeros((ys_pred.shape[0], 1, ys_pred.shape[1])), 'next_visit_mask': torch.zeros((ys_pred.shape[0],1,ys_pred.shape[1])), 'slope': torch.from_numpy(ys_te).float()},
        attention_mask=torch.ones((ys_pred.shape[0], 1)),
        target_names=target_names,
    )

    return {
        'next_visit_metrics': next_visit_metrics,
        'slope_metrics': slope_metrics,
        'models_next_visit': models_next,
        'models_slope': slope_models,
    }
