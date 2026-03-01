"""
Temporal XGBoost baseline with lagged features (visits t-1 and t-2).

This module is identical to ``models/xgboost_model`` except that the
``build_next_visit_tabular`` function is augmented to add lag1/lag2 versions of
every numeric feature (including time) plus delta_t1 and delta_t2.  Missing
lag values are left as NaN, which XGBoost can handle natively.

The training, validation, and evaluation pipeline remains unchanged from the
original XGBoost model; we simply feed the expanded feature matrix to the same
learning code.  This allows a fair comparison with the transformer models
which also see historical context implicitly.

Feature engineering logic:

* For each visit *t* we build a row containing:
    * static covariates (unchanged)
    * flattened modalities of visit *t* (as before)
    * flattened modalities of visit *t-1* (lag1) or NaNs if t==0
    * flattened modalities of visit *t-2* (lag2) or NaNs if t<2
    * delta_t1 = time_months[t] - time_months[t-1] (NaN if t==0)
    * delta_t2 = time_months[t] - time_months[t-2] (NaN if t<2)

* The target remains the UPDRS totals at visit *t+1*.

All other utilities (slope table, training helpers, prediction, evaluation)
are copied verbatim from ``xgboost_model.py`` to maintain identical behaviour.
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
    parts.append(np.asarray([visit.get('time_months', np.nan)], dtype=np.float32))
    # targets are not included
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)


def build_next_visit_tabular(
    dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build tabular matrices with lagged features.

    In addition to the columns produced by the base XGBoost model, this
    function appends lagged versions of every numeric column (lag1 and lag2)
    along with two delta-time features.  Rows corresponding to the first two
    visits for each patient will contain NaNs in the lag columns; this mimics
    the behaviour of a groupby-shift operation on a DataFrame.

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

        # storage for flattened visits to avoid repeated computation
        flat_cache: List[np.ndarray] = [None] * n
        for t in range(n - 1):
            cur = visits[t]
            nxt = visits[t + 1]

            if flat_cache[t] is None:
                flat_cache[t] = _flatten_visit(cur)
            curr_feat = flat_cache[t]

            # lag1 (t-1)
            if t - 1 >= 0:
                if flat_cache[t - 1] is None:
                    flat_cache[t - 1] = _flatten_visit(visits[t - 1])
                lag1 = flat_cache[t - 1]
                delta_t1 = cur.get('time_months', np.nan) - visits[t - 1].get('time_months', np.nan)
            else:
                lag1 = np.full_like(curr_feat, np.nan)
                delta_t1 = np.nan

            # lag2 (t-2)
            if t - 2 >= 0:
                if flat_cache[t - 2] is None:
                    flat_cache[t - 2] = _flatten_visit(visits[t - 2])
                lag2 = flat_cache[t - 2]
                delta_t2 = cur.get('time_months', np.nan) - visits[t - 2].get('time_months', np.nan)
            else:
                lag2 = np.full_like(curr_feat, np.nan)
                delta_t2 = np.nan

            # build feature vector
            feat = np.concatenate([
                static_vec,
                curr_feat,
                lag1,
                lag2,
                np.array([delta_t1, delta_t2], dtype=np.float32),
            ])

            rows.append(feat)
            yvec = np.asarray(nxt.get('updrs_totals', np.zeros(0)), dtype=np.float32)
            mask = (~np.isnan(yvec)).astype(np.float32)
            targets.append(np.nan_to_num(yvec, nan=0.0))
            masks.append(mask)
            deltas.append(float(nxt.get('time_months', np.nan) - cur.get('time_months', np.nan)))

    if len(rows) == 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    X = np.vstack(rows)
    y = np.vstack(targets)
    label_mask = np.vstack(masks)
    delta_t = np.asarray(deltas, dtype=np.float32)

    return X, y, label_mask, delta_t


# slope builder remains unchanged from base model

def build_slope_tabular(dataset) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for patno in dataset.patient_ids:
        slope_dict = dataset.slopes.get(patno, {})
        if isinstance(slope_dict, dict):
            yvec = np.array([
                slope_dict.get('NP1RTOT_slope', np.nan),
                slope_dict.get('NP2PTOT_slope', np.nan),
                slope_dict.get('NP3TOT_slope', np.nan),
                slope_dict.get('NP4TOT_slope', np.nan),
            ], dtype=np.float32)
        else:
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


# training helpers, prediction, and run_xgb_experiment are identical to base model


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
    preds = [m.predict(xgb.DMatrix(X)) for m in models]
    return np.vstack(preds).T


def run_xgb_experiment(
    train_dataset,
    val_dataset,
    test_dataset,
    target_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
) -> Dict[str, Any]:
    # import metrics lazily
    from training.metrics import compute_comprehensive_metrics

    X_tr, y_tr, mask_tr, dt_tr = build_next_visit_tabular(train_dataset)
    X_val, y_val, mask_val, dt_val = build_next_visit_tabular(val_dataset)
    X_te, y_te, mask_te, dt_te = build_next_visit_tabular(test_dataset)

    print(f"XGBoost-temporal: training rows={X_tr.shape[0]}, features={X_tr.shape[1]}")

    if target_names is None:
        target_names = []

    models_next = train_xgb_multitarget(
        X_tr, y_tr, mask_tr, X_val, y_val, mask_val,
        params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    y_pred_te = predict_multimodel(models_next, X_te)

    B, K = y_pred_te.shape
    preds_seq = torch.zeros((B, 2, K), dtype=torch.float32)
    targets_seq = torch.zeros((B, 2, K), dtype=torch.float32)
    mask_seq = torch.zeros((B, 2, K), dtype=torch.float32)
    time_seq = torch.zeros((B, 2), dtype=torch.float32)

    preds_seq[:, 0, :] = torch.from_numpy(y_pred_te)
    targets_seq[:, 1, :] = torch.from_numpy(y_te)
    mask_seq[:, 1, :] = torch.from_numpy(mask_te)
    time_seq[:, 0] = torch.from_numpy(dt_te)
    time_seq[:, 1] = torch.from_numpy(dt_te)

    next_visit_metrics = compute_comprehensive_metrics(
        {'next_visit': preds_seq, 'slope': torch.zeros((B, K))},
        {'next_visit': targets_seq, 'next_visit_mask': mask_seq, 'slope': torch.zeros((B, K))},
        attention_mask=torch.ones((B, 2)),
        target_names=target_names,
        time_months=time_seq,
    )

    Xs_tr, ys_tr = build_slope_tabular(train_dataset)
    Xs_te, ys_te = build_slope_tabular(test_dataset)
    slope_models = train_xgb_multitarget(Xs_tr, ys_tr, ~np.isnan(ys_tr))
    ys_pred = predict_multimodel(slope_models, Xs_te)
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
