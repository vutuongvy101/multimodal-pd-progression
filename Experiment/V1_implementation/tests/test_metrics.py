import torch

from training.metrics import compute_comprehensive_metrics


def test_compute_comprehensive_metrics_basic():
    """
    Sanity check metrics for small synthetic tensors:
    - Two UPDRS targets (K=2)
    - Sequence length = 3 (effective comparisons at T-1 = 2)
    - All masks active
    """
    device = "cpu"

    # Batch=1, T=3, K=2
    preds_next = torch.tensor(
        [[[1.0, 2.0],
          [2.0, 4.0],
          [3.0, 6.0]]],
        device=device,
    )
    targets_next = torch.tensor(
        [[[1.0, 1.0],
          [2.0, 2.0],
          [3.0, 3.0]]],
        device=device,
    )
    # Shifted comparisons:
    #  t0 -> target t1, t1 -> target t2
    # For target0: preds [1,2], targets [2,3] => MAE=1.0, RMSE=1.0, corr=1.0, R2=-3.0
    # For target1: preds [2,4], targets [2,3] => MAE=0.5, RMSE~0.7071, corr=1.0, R2=-1.0

    attention_mask = torch.ones((1, 3), device=device)
    label_mask = torch.ones((1, 3, 2), device=device)
    # Slopes: [batch, n_targets] - one slope per UPDRS total
    slope_preds = torch.tensor([[0.5, 0.3]], device=device)  # [1, 2]
    slope_targets = torch.tensor([[1.0, 0.8]], device=device)  # [1, 2]

    predictions = {
        "next_visit": preds_next,
        "slope": slope_preds,
    }
    targets = {
        "next_visit": targets_next,
        "next_visit_mask": label_mask,
        "slope": slope_targets,
    }

    metrics = compute_comprehensive_metrics(
        predictions,
        targets,
        attention_mask,
        target_names=["NP1RTOT", "NP2PTOT"],
    )

    m0 = metrics["next_visit"]["NP1RTOT"]
    m1 = metrics["next_visit"]["NP2PTOT"]

    assert m0["n_samples"] == 2
    assert m1["n_samples"] == 2

    assert abs(m0["mae"] - 1.0) < 1e-5
    assert abs(m0["rmse"] - 1.0) < 1e-5
    assert abs(m0["correlation"] - 1.0) < 1e-5
    assert abs(m0["r2"] - (-3.0)) < 1e-5  # Negative R2 is allowed

    assert abs(m1["mae"] - 0.5) < 1e-5
    assert abs(m1["rmse"] - 0.70710677) < 1e-4
    assert abs(m1["correlation"] - 1.0) < 1e-5
    assert abs(m1["r2"] - (-1.0)) < 1e-5

    # Slope metrics: per-target slopes
    # NP1RTOT_slope: preds=0.5, target=1.0 -> MAE=0.5, RMSE=0.5
    # NP2PTOT_slope: preds=0.3, target=0.8 -> MAE=0.5, RMSE=0.5
    # Check per-target slope metrics
    assert "NP1RTOT_slope" in metrics["slope"]
    assert "NP2PTOT_slope" in metrics["slope"]
    
    ms0 = metrics["slope"]["NP1RTOT_slope"]
    ms1 = metrics["slope"]["NP2PTOT_slope"]
    
    assert ms0["n_samples"] == 1
    assert abs(ms0["mae"] - 0.5) < 1e-5
    assert abs(ms0["rmse"] - 0.5) < 1e-5
    
    assert ms1["n_samples"] == 1
    assert abs(ms1["mae"] - 0.5) < 1e-5
    assert abs(ms1["rmse"] - 0.5) < 1e-5
    
    # Check overall slope metrics (macro-average)
    ms_overall = metrics["slope_overall"]
    assert ms_overall["n_samples"] == 2
    assert abs(ms_overall["mae"] - 0.5) < 1e-5  # Mean of 0.5 and 0.5
    assert abs(ms_overall["rmse"] - 0.5) < 1e-5  # Mean of 0.5 and 0.5
