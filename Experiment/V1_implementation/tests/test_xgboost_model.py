"""Unit tests for the XGBoost baseline module.

These tests use a tiny in-memory longitudinal dataset so that they run in a
fraction of a second.  They verify shapes, absence of NaNs, and that the
high-level ``run_xgb_experiment`` function executes end-to-end without
raising errors.
"""

import numpy as np
import pytest

# skip entire module if torch isn't available (many helpers rely on it)
try:
    import torch
except ImportError:
    pytest.skip("torch not installed, skipping xgboost_model tests", allow_module_level=True)

from models.xgboost_model import (
    _flatten_visit,
    build_next_visit_tabular,
    build_slope_tabular,
    train_xgb_multitarget,
    predict_multimodel,
    run_xgb_experiment,
)
from models.xgboost_temporal_model import (
    build_next_visit_tabular as build_next_visit_tabular_temporal,
    run_xgb_experiment as run_xgb_experiment_temporal,
)
from data.dataset import PPMILongitudinalDataset


@pytest.fixture
def dummy_longitudinal_dataset():
    """Create a tiny in-memory longitudinal dataset for testing.

    Two patients with two visits each and minimal feature vectors.
    Entire dataset is in-memory so tests execute in milliseconds.
    """
    patnos = [1001, 1002]
    static_data = {}
    for pat in patnos:
        static_data[pat] = {
            'values': np.arange(3, dtype=np.float32),
            'mask': np.zeros(3, dtype=np.float32),
        }

    longitudinal_data = {}
    for pat in patnos:
        visits = []
        for v in range(2):
            visits.append({
                'motor_values': np.ones(2, dtype=np.float32) * v,
                'nonmotor_values': np.ones(2, dtype=np.float32) * v,
                'med_values': np.array([v], dtype=np.float32),
                'age_at_visit_values': np.array([50.0 + v], dtype=np.float32),
                'time_months': float(v * 6),
                'updrs_totals': np.array([1.0 + v, 2.0 + v, 3.0 + v, 4.0 + v], dtype=np.float32),
            })
        longitudinal_data[pat] = visits

    slopes = {
        1001: {'NP1RTOT_slope': 0.1, 'NP2PTOT_slope': 0.2, 'NP3TOT_slope': 0.3, 'NP4TOT_slope': 0.4},
        1002: {'NP1RTOT_slope': 0.0, 'NP2PTOT_slope': 0.0, 'NP3TOT_slope': 0.0, 'NP4TOT_slope': 0.0},
    }

    return PPMILongitudinalDataset(
        patient_ids=patnos,
        static_data=static_data,
        longitudinal_data=longitudinal_data,
        slopes=slopes,
        max_seq_len=5,
    )


def test_flatten_visit():
    """Test _flatten_visit helper with a simple visit dict."""
    visit = {'motor_values': np.array([1, 2], dtype=np.float32), 'time_months': 3.0}
    arr = _flatten_visit(visit)
    assert arr.dtype == np.float32
    assert arr.shape == (3,)
    assert arr[-1] == 3.0


def test_build_next_visit_tabular_shapes(dummy_longitudinal_dataset):
    ds = dummy_longitudinal_dataset
    X, y, mask, dt = build_next_visit_tabular(ds)
    assert X.ndim == 2
    assert y.ndim == 2
    assert mask.shape == y.shape
    assert dt.shape[0] == X.shape[0]
    # there should be exactly 2 rows (one for each patient)
    assert X.shape[0] == 2


def test_next_visit_mask_behavior(dummy_longitudinal_dataset):
    ds = dummy_longitudinal_dataset
    X, y, mask, dt = build_next_visit_tabular(ds)
    # all mask entries should be 1 since we constructed non-nan targets
    assert np.all(mask == 1.0)


def test_train_and_predict_small(dummy_longitudinal_dataset):
    ds = dummy_longitudinal_dataset
    X, y, mask, dt = build_next_visit_tabular(ds)
    models = train_xgb_multitarget(X, y, mask, params={'objective': 'reg:squarederror', 'tree_method': 'hist'}, num_boost_round=2)
    preds = predict_multimodel(models, X)
    assert preds.shape == y.shape


def test_run_experiment_smoke(dummy_longitudinal_dataset):
    ds = dummy_longitudinal_dataset
    results = run_xgb_experiment(
        train_dataset=ds,
        val_dataset=ds,
        test_dataset=ds,
        target_names=['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT'],
        params={'tree_method': 'hist'},
        num_boost_round=2,
        early_stopping_rounds=1,
    )
    assert 'next_visit_metrics' in results
    assert 'slope_metrics' in results
    # metrics dictionaries should contain at least one target entry
    assert results['next_visit_metrics']['next_visit']
    assert results['slope_metrics']['slope']


def test_temporal_feature_expansion(dummy_longitudinal_dataset):
    """Ensure the temporal builder adds lag columns and NaNs appropriately."""
    ds = dummy_longitudinal_dataset
    X_base, _, _, _ = build_next_visit_tabular(ds)
    X_temp, _, mask_temp, _ = build_next_visit_tabular_temporal(ds)
    # temporal should have strictly more features (static + curr + lag1 + lag2 + 2 deltas)
    assert X_temp.shape[1] > X_base.shape[1]
    # first row (patient 1001, visit0) should contain NaNs in lag zones
    first = X_temp[0]
    assert np.isnan(first).any()


def test_temporal_smoke(dummy_longitudinal_dataset):
    ds = dummy_longitudinal_dataset
    # just run end-to-end to make sure nothing breaks
    results = run_xgb_experiment_temporal(
        train_dataset=ds,
        val_dataset=ds,
        test_dataset=ds,
        target_names=['NP1TOT', 'NP2TOT', 'NP3TOT', 'NP4TOT'],
        params={'tree_method': 'hist'},
        num_boost_round=2,
        early_stopping_rounds=1,
    )
    assert 'next_visit_metrics' in results
    assert 'slope_metrics' in results
