"""
Evaluation metrics for V1 model.

Provides:
 - Per-UPDRS-total metrics (MAE, RMSE, R^2, Pearson correlation, Spearman correlation)
 - Per-Δt bucket metrics for next-visit predictions
 - Overall next-visit metrics (macro-average, weighted-average)
 - Per-target slope metrics (MAE, RMSE, Spearman correlation)
 - Support counts (number of samples per target)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0.0 if undefined."""
    if x.size < 2 or y.size < 2:
        return 0.0
    x_std = x.std()
    y_std = y.std()
    if x_std == 0 or y_std == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation, returning 0.0 if undefined."""
    if x.size < 2 or y.size < 2:
        return 0.0
    try:
        rho, _ = spearmanr(x, y)
        if np.isnan(rho):
            return 0.0
        return float(rho)
    except:
        return 0.0


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
    time_months: Optional[torch.Tensor] = None,
    delta_t_buckets: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        predictions: dict with:
            - 'next_visit': [B, T, K] tensor
            - 'slope': [B, K] tensor (one slope per UPDRS total)
        targets: dict with:
            - 'next_visit': [B, T, K] tensor
            - 'next_visit_mask': [B, T, K] tensor (1=present, 0=missing) or None
            - 'slope': [B, K] tensor (one slope per UPDRS total)
        attention_mask: [B, T] tensor (1=valid, 0=padding)
        target_names: names for each UPDRS total in order of the last dimension
        time_months: [B, T] tensor with time in months since baseline (for Δt bucket computation)
        delta_t_buckets: List of (min, max) tuples for Δt buckets. Default: [(0, 6), (6, 12), (12, 24), (24, inf)]

    Returns:
        Dictionary with:
          - 'next_visit': per-target metrics (with per-Δt bucket metrics)
          - 'next_visit_overall': aggregated metrics (macro/weighted average)
          - 'slope': per-target slope metrics
          - 'slope_overall': overall slope metrics
    """
    # Default Δt buckets: 0-6, 6-12, 12-24, 24+ months
    if delta_t_buckets is None:
        delta_t_buckets = [(0, 6), (6, 12), (12, 24), (24, float('inf'))]
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

    # Compute delta time (Δt) if time_months is provided
    delta_t = None
    if time_months is not None:
        # Compute Δt = time[t+1] - time[t] for each transition
        time_shifted = time_months[:, :-1]  # [B, T-1]
        time_next = time_months[:, 1:]  # [B, T-1]
        delta_t = (time_next - time_shifted).detach().cpu().numpy()  # [B, T-1]

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
                "spearman": float("nan"),
                "n_samples": 0,
                "delta_t_buckets": {},
            }
            continue

        # Get valid predictions and targets
        valid_preds = pred_values[valid_mask].detach().cpu().numpy()
        valid_targets = target_values[valid_mask].detach().cpu().numpy()
        
        # Get corresponding delta_t values if available
        valid_delta_t = None
        if delta_t is not None:
            valid_delta_t = delta_t[valid_mask.cpu().numpy()]

        # Overall metrics for this target
        mae = float(np.mean(np.abs(valid_preds - valid_targets)))
        rmse = float(np.sqrt(np.mean((valid_preds - valid_targets) ** 2)))
        r2 = _safe_r2(valid_targets, valid_preds)
        corr = _safe_correlation(valid_targets, valid_preds)
        spearman_corr = _safe_spearman(valid_targets, valid_preds)

        # Per-Δt bucket metrics
        delta_t_bucket_metrics = {}
        if valid_delta_t is not None:
            for bucket_idx, (min_dt, max_dt) in enumerate(delta_t_buckets):
                bucket_mask = (valid_delta_t >= min_dt) & (valid_delta_t < max_dt)
                if bucket_mask.sum() == 0:
                    delta_t_bucket_metrics[f"dt_{min_dt}_{max_dt}"] = {
                        "mae": float("nan"),
                        "rmse": float("nan"),
                        "n_samples": 0,
                    }
                    continue
                
                bucket_preds = valid_preds[bucket_mask]
                bucket_targets = valid_targets[bucket_mask]
                
                bucket_mae = float(np.mean(np.abs(bucket_preds - bucket_targets)))
                bucket_rmse = float(np.sqrt(np.mean((bucket_preds - bucket_targets) ** 2)))
                
                delta_t_bucket_metrics[f"dt_{min_dt}_{max_dt}"] = {
                    "mae": bucket_mae,
                    "rmse": bucket_rmse,
                    "n_samples": int(bucket_mask.sum()),
                }

        per_target_metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "correlation": corr,
            "spearman": spearman_corr,
            "n_samples": int(valid_preds.size),
            "delta_t_buckets": delta_t_bucket_metrics,
        }

    metrics["next_visit"] = per_target_metrics

    # Overall metrics across all targets
    # Macro-average: mean of per-target metrics
    maes = [m["mae"] for m in per_target_metrics.values() if not np.isnan(m["mae"])]
    rmses = [m["rmse"] for m in per_target_metrics.values() if not np.isnan(m["rmse"])]
    r2s = [m["r2"] for m in per_target_metrics.values() if not np.isnan(m["r2"])]
    correlations = [m["correlation"] for m in per_target_metrics.values() if not np.isnan(m["correlation"])]
    spearmans = [m["spearman"] for m in per_target_metrics.values() if not np.isnan(m["spearman"])]
    
    # Weighted-average: weights proportional to number of samples (support)
    total_samples = sum(m["n_samples"] for m in per_target_metrics.values())
    weighted_mae = 0.0
    weighted_rmse = 0.0
    if total_samples > 0:
        for m in per_target_metrics.values():
            if not np.isnan(m["mae"]) and m["n_samples"] > 0:
                weight = m["n_samples"] / total_samples
                weighted_mae += m["mae"] * weight
                weighted_rmse += m["rmse"] * weight

    metrics["next_visit_overall"] = {
        "macro_avg_mae": float(np.mean(maes)) if maes else float("nan"),
        "macro_avg_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "macro_avg_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "macro_avg_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "macro_avg_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        "weighted_avg_mae": weighted_mae if total_samples > 0 else float("nan"),
        "weighted_avg_rmse": weighted_rmse if total_samples > 0 else float("nan"),
        "total_samples": int(total_samples),
    }

    # -----------------
    # Slope metrics (per-target)
    # -----------------
    # Model predicts all 4 UPDRS total slopes: [batch, n_targets]
    slope_preds = predictions["slope"].detach().cpu().numpy()  # [batch, n_targets]
    slope_targets = targets["slope"].detach().cpu().numpy()  # [batch, n_targets]

    per_target_slope_metrics = {}
    n_targets = slope_preds.shape[-1]
    
    # Compute per-target metrics
    for target_idx in range(n_targets):
        name = target_names[target_idx] if target_idx < len(target_names) else f"target_{target_idx}"
        slope_name = f"{name}_slope"
        
        pred_values = slope_preds[:, target_idx]  # [batch]
        target_values = slope_targets[:, target_idx]  # [batch]
        
        # Filter out NaN targets
        valid_mask = ~np.isnan(target_values)
        if valid_mask.sum() == 0:
            per_target_slope_metrics[slope_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "spearman": float("nan"),
                "n_samples": 0,
            }
            continue
        
        valid_preds = pred_values[valid_mask]
        valid_targets = target_values[valid_mask]
        
        slope_mae = float(np.mean(np.abs(valid_preds - valid_targets)))
        slope_rmse = float(np.sqrt(np.mean((valid_preds - valid_targets) ** 2)))
        slope_spearman = _safe_spearman(valid_targets, valid_preds)
        
        per_target_slope_metrics[slope_name] = {
            "mae": slope_mae,
            "rmse": slope_rmse,
            "spearman": slope_spearman,
            "n_samples": int(valid_preds.size),
        }
    
    metrics["slope"] = per_target_slope_metrics
    
    # Overall slope metrics (macro-average across targets)
    valid_slope_metrics = [m for m in per_target_slope_metrics.values() if m["n_samples"] > 0]
    if valid_slope_metrics:
        maes = [m["mae"] for m in valid_slope_metrics if not np.isnan(m["mae"])]
        rmses = [m["rmse"] for m in valid_slope_metrics if not np.isnan(m["rmse"])]
        spearmans = [m["spearman"] for m in valid_slope_metrics if not np.isnan(m["spearman"])]
        total_samples = sum(m["n_samples"] for m in valid_slope_metrics)
        
        metrics["slope_overall"] = {
            "mae": float(np.mean(maes)) if maes else float("nan"),
            "rmse": float(np.mean(rmses)) if rmses else float("nan"),
            "spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
            "n_samples": int(total_samples),
        }
    else:
        metrics["slope_overall"] = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "n_samples": 0,
        }

    return metrics

