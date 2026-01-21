"""
Evaluation metrics for V1 model.

Provides:
 - Per-UPDRS-total metrics (MAE, RMSE, R^2, correlation)
 - Overall next-visit metrics (averaged across totals)
 - Slope metrics with correlation
"""

from typing import Dict, List

import numpy as np
import torch


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0.0 if undefined."""
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = x.std()
    y_std = y.std()
    if x_std == 0 or y_std == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2, returning 0.0 if undefined."""
    if y_true.size == 0:
        return 0.0
    y_mean = y_true.mean()
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_comprehensive_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    target_names: List[str],
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: dict with:
            - 'next_visit': [B, T, K] tensor
            - 'slope': [B] tensor
        targets: dict with:
            - 'next_visit': [B, T, K] tensor
            - 'next_visit_mask': [B, T, K] tensor (1=present, 0=missing) or None
            - 'slope': [B] tensor
        attention_mask: [B, T] tensor (1=valid, 0=padding)
        target_names: names for each UPDRS total in order of the last dimension

    Returns:
        Dictionary with:
          - 'next_visit': per-target metrics
          - 'next_visit_overall': aggregated metrics
          - 'slope': slope metrics
    """
    metrics: Dict[str, Dict] = {}

    # -----------------------------
    # Next-visit per-target metrics
    # -----------------------------
    next_visit_preds = predictions["next_visit"]  # [B, T, K]
    next_visit_targets = targets["next_visit"]  # [B, T, K]
    label_mask = targets.get("next_visit_mask")  # [B, T, K] or None

    # Shift for next-visit prediction (predict t+1 from t)
    preds_shifted = next_visit_preds[:, :-1, :]  # [B, T-1, K]
    targets_shifted = next_visit_targets[:, 1:, :]  # [B, T-1, K]

    if attention_mask is not None:
        time_mask = attention_mask[:, :-1].unsqueeze(-1)  # [B, T-1, 1]
    else:
        time_mask = torch.ones_like(preds_shifted[..., :1])

    if label_mask is not None:
        label_mask_shifted = label_mask[:, 1:, :]  # [B, T-1, K]
    else:
        label_mask_shifted = torch.ones_like(targets_shifted)

    combined_mask = time_mask * label_mask_shifted  # [B, T-1, K]

    per_target_metrics: Dict[str, Dict] = {}
    n_targets = preds_shifted.shape[-1]

    for target_idx in range(n_targets):
        name = target_names[target_idx] if target_idx < len(target_names) else f"target_{target_idx}"

        pred_values = preds_shifted[:, :, target_idx]  # [B, T-1]
        target_values = targets_shifted[:, :, target_idx]  # [B, T-1]
        mask = combined_mask[:, :, target_idx]  # [B, T-1]

        # Flatten and filter by mask
        valid_mask = (mask == 1)
        if valid_mask.sum().item() == 0:
            per_target_metrics[name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "correlation": float("nan"),
                "n_samples": 0,
            }
            continue

        valid_preds = pred_values[valid_mask].detach().cpu().numpy()
        valid_targets = target_values[valid_mask].detach().cpu().numpy()

        mae = float(np.mean(np.abs(valid_preds - valid_targets)))
        rmse = float(np.sqrt(np.mean((valid_preds - valid_targets) ** 2)))
        r2 = _safe_r2(valid_targets, valid_preds)
        corr = _safe_correlation(valid_targets, valid_preds)

        per_target_metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "correlation": corr,
            "n_samples": int(valid_preds.size),
        }

    metrics["next_visit"] = per_target_metrics

    # Overall metrics across all targets
    maes = [m["mae"] for m in per_target_metrics.values() if not np.isnan(m["mae"])]
    rmses = [m["rmse"] for m in per_target_metrics.values() if not np.isnan(m["rmse"])]
    r2s = [m["r2"] for m in per_target_metrics.values() if not np.isnan(m["r2"])]

    metrics["next_visit_overall"] = {
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
    }

    # -----------------
    # Slope metrics
    # -----------------
    slope_preds = predictions["slope"].detach().cpu().numpy()
    slope_targets = targets["slope"].detach().cpu().numpy()

    valid_mask = ~np.isnan(slope_targets)
    if valid_mask.sum() == 0:
        metrics["slope"] = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "correlation": float("nan"),
            "n_samples": 0,
        }
    else:
        v_preds = slope_preds[valid_mask]
        v_targets = slope_targets[valid_mask]
        slope_mae = float(np.mean(np.abs(v_preds - v_targets)))
        slope_rmse = float(np.sqrt(np.mean((v_preds - v_targets) ** 2)))
        slope_r2 = _safe_r2(v_targets, v_preds)
        slope_corr = _safe_correlation(v_targets, v_preds)

        metrics["slope"] = {
            "mae": slope_mae,
            "rmse": slope_rmse,
            "r2": slope_r2,
            "correlation": slope_corr,
            "n_samples": int(v_preds.size),
        }

    return metrics

