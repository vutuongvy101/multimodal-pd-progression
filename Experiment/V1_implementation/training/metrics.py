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


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "r2": _safe_r2(y_true, y_pred),
        "correlation": _safe_correlation(y_true, y_pred),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def _compute_delta_t_bucket_metrics(
    valid_preds: np.ndarray,
    valid_targets: np.ndarray,
    valid_delta_t: np.ndarray,
    delta_t_buckets: List[Tuple[float, float]],
) -> Dict[str, Dict[str, float]]:
    bucket_metrics = {}
    for min_dt, max_dt in delta_t_buckets:
        bucket_mask = (valid_delta_t >= min_dt) & (valid_delta_t < max_dt)
        bucket_key = f"dt_{min_dt}_{max_dt}"
        
        if bucket_mask.sum() == 0:
            bucket_metrics[bucket_key] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "n_samples": 0,
            }
            continue
        
        bucket_preds = valid_preds[bucket_mask]
        bucket_targets = valid_targets[bucket_mask]
        bucket_basic_metrics = _compute_basic_metrics(bucket_targets, bucket_preds)
        
        bucket_metrics[bucket_key] = {
            "mae": bucket_basic_metrics["mae"],
            "rmse": bucket_basic_metrics["rmse"],
            "n_samples": int(bucket_mask.sum()),
        }
    return bucket_metrics


def _prepare_next_visit_data(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    time_months: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    next_visit_preds = predictions["next_visit"]
    next_visit_targets = targets["next_visit"]
    label_mask = targets.get("next_visit_mask")
    
    preds_shifted = next_visit_preds[:, :-1, :]
    targets_shifted = next_visit_targets[:, 1:, :]
    
    time_mask = attention_mask[:, :-1].unsqueeze(-1) if attention_mask is not None else torch.ones_like(preds_shifted[..., :1])
    label_mask_shifted = label_mask[:, 1:, :] if label_mask is not None else torch.ones_like(targets_shifted)
    combined_mask = time_mask * label_mask_shifted
    
    delta_t = None
    if time_months is not None:
        time_shifted = time_months[:, :-1]
        time_next = time_months[:, 1:]
        delta_t = (time_next - time_shifted).detach().cpu().numpy()
    
    return preds_shifted, targets_shifted, combined_mask, delta_t


def _compute_per_target_next_visit_metrics(
    preds_shifted: torch.Tensor,
    targets_shifted: torch.Tensor,
    combined_mask: torch.Tensor,
    target_names: List[str],
    delta_t: Optional[np.ndarray],
    delta_t_buckets: List[Tuple[float, float]],
) -> Dict[str, Dict]:
    per_target_metrics = {}
    n_targets = preds_shifted.shape[-1]
    
    for target_idx in range(n_targets):
        target_name = target_names[target_idx] if target_idx < len(target_names) else f"target_{target_idx}"
        
        pred_values = preds_shifted[:, :, target_idx]
        target_values = targets_shifted[:, :, target_idx]
        mask = combined_mask[:, :, target_idx]
        
        valid_mask = (mask == 1)
        if valid_mask.sum().item() == 0:
            per_target_metrics[target_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "r2": float("nan"),
                "correlation": float("nan"),
                "spearman": float("nan"),
                "n_samples": 0,
                "delta_t_buckets": {},
            }
            continue
        
        valid_preds = pred_values[valid_mask].detach().cpu().numpy()
        valid_targets = target_values[valid_mask].detach().cpu().numpy()
        
        basic_metrics = _compute_basic_metrics(valid_targets, valid_preds)
        
        delta_t_bucket_metrics = {}
        if delta_t is not None:
            valid_delta_t = delta_t[valid_mask.cpu().numpy()]
            delta_t_bucket_metrics = _compute_delta_t_bucket_metrics(
                valid_preds, valid_targets, valid_delta_t, delta_t_buckets
            )
        
        per_target_metrics[target_name] = {
            **basic_metrics,
            "n_samples": int(valid_preds.size),
            "delta_t_buckets": delta_t_bucket_metrics,
        }
    
    return per_target_metrics


def _compute_overall_metrics(per_target_metrics: Dict[str, Dict]) -> Dict[str, float]:
    def extract_valid_values(key: str) -> List[float]:
        return [m[key] for m in per_target_metrics.values() if not np.isnan(m[key])]
    
    maes = extract_valid_values("mae")
    rmses = extract_valid_values("rmse")
    r2s = extract_valid_values("r2")
    correlations = extract_valid_values("correlation")
    spearmans = extract_valid_values("spearman")
    
    total_samples = sum(m["n_samples"] for m in per_target_metrics.values())
    
    weighted_mae = 0.0
    weighted_rmse = 0.0
    if total_samples > 0:
        for m in per_target_metrics.values():
            if not np.isnan(m["mae"]) and m["n_samples"] > 0:
                weight = m["n_samples"] / total_samples
                weighted_mae += m["mae"] * weight
                weighted_rmse += m["rmse"] * weight
    
    return {
        "macro_avg_mae": float(np.mean(maes)) if maes else float("nan"),
        "macro_avg_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "macro_avg_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "macro_avg_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "macro_avg_spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        "weighted_avg_mae": weighted_mae if total_samples > 0 else float("nan"),
        "weighted_avg_rmse": weighted_rmse if total_samples > 0 else float("nan"),
        "total_samples": int(total_samples),
    }


def _compute_per_target_slope_metrics(
    slope_preds: np.ndarray,
    slope_targets: np.ndarray,
    target_names: List[str],
) -> Dict[str, Dict]:
    per_target_slope_metrics = {}
    n_targets = slope_preds.shape[-1]
    
    for target_idx in range(n_targets):
        target_name = target_names[target_idx] if target_idx < len(target_names) else f"target_{target_idx}"
        slope_name = f"{target_name}_slope"
        
        pred_values = slope_preds[:, target_idx]
        target_values = slope_targets[:, target_idx]
        
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
        
        basic_metrics = _compute_basic_metrics(valid_targets, valid_preds)
        
        per_target_slope_metrics[slope_name] = {
            "mae": basic_metrics["mae"],
            "rmse": basic_metrics["rmse"],
            "spearman": basic_metrics["spearman"],
            "n_samples": int(valid_preds.size),
        }
    
    return per_target_slope_metrics


def _compute_overall_slope_metrics(per_target_slope_metrics: Dict[str, Dict]) -> Dict[str, float]:
    valid_metrics = [m for m in per_target_slope_metrics.values() if m["n_samples"] > 0]
    
    if not valid_metrics:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "spearman": float("nan"),
            "n_samples": 0,
        }
    
    def extract_valid_values(key: str) -> List[float]:
        return [m[key] for m in valid_metrics if not np.isnan(m[key])]
    
    maes = extract_valid_values("mae")
    rmses = extract_valid_values("rmse")
    spearmans = extract_valid_values("spearman")
    total_samples = sum(m["n_samples"] for m in valid_metrics)
    
    return {
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "spearman": float(np.mean(spearmans)) if spearmans else float("nan"),
        "n_samples": int(total_samples),
    }


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
    if delta_t_buckets is None:
        delta_t_buckets = [(0, 6), (6, 12), (12, 24), (24, float('inf'))]
    
    preds_shifted, targets_shifted, combined_mask, delta_t = _prepare_next_visit_data(
        predictions, targets, attention_mask, time_months
    )
    
    per_target_metrics = _compute_per_target_next_visit_metrics(
        preds_shifted, targets_shifted, combined_mask, target_names, delta_t, delta_t_buckets
    )
    
    slope_preds = predictions["slope"].detach().cpu().numpy()
    slope_targets = targets["slope"].detach().cpu().numpy()
    
    per_target_slope_metrics = _compute_per_target_slope_metrics(
        slope_preds, slope_targets, target_names
    )
    
    return {
        "next_visit": per_target_metrics,
        "next_visit_overall": _compute_overall_metrics(per_target_metrics),
        "slope": per_target_slope_metrics,
        "slope_overall": _compute_overall_slope_metrics(per_target_slope_metrics),
    }

