import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# Import project modules
# -------------------------------------------------------------------------
# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.config import get_default_config
from data.data_integrator import DataIntegrator
from data.dataset import PPMILongitudinalDataset, collate_fn
from models.v1_model import V1MultimodalTransformer
from visualization.plotters import EvaluationPlotter
from visualization.io import PlotIO

# -------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------
CHECKPOINT_PATH = BASE_DIR / "models" / "checkpoints" / \
    "modalities_age_at_visit+medication+motor+non_motor+static" / "best_checkpoint.pt"

OUTPUT_DIR = BASE_DIR / "visualization" / "all_modalities" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of stochastic passes for MC-dropout
MC_SAMPLES = 20  # you can increase to 50 for smoother bands


# -------------------------------------------------------------------------
# 1. Load config, prepare data, build dataset/dataloader
# -------------------------------------------------------------------------
def build_full_dataset_and_loader(config):
    """
    Use DataIntegrator + PPMILongitudinalDataset to get all patients and
    their full visit sequences, with months_since_baseline and UPDRS totals.
    """
    print("=== Preparing data for inference ===")
    integrator = DataIntegrator(config, normalize_features=True)

    # Load & merge all raw CSVs into DataFrames
    prepared = integrator.prepare_final_dataset()  # dict with 'static', 'longitudinal', 'slopes', 'metadata', ...

    # Convert DataFrames → feature vectors with masks
    feature_vectors = integrator.create_feature_vectors(prepared)

    static_data = feature_vectors["static_data"]
    longitudinal_data = feature_vectors["longitudinal_data"]
    slopes = feature_vectors["slopes"]

    # Use all patients
    patient_ids = list(static_data.keys())
    print(f"Total patients used for uncertainty plot: {len(patient_ids)}")

    dataset = PPMILongitudinalDataset(
        patient_ids=patient_ids,
        static_data=static_data,
        longitudinal_data=longitudinal_data,
        slopes=slopes,
        max_seq_len=config.model.max_seq_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return dataset, loader


# -------------------------------------------------------------------------
# 2. Load trained model
# -------------------------------------------------------------------------
def load_trained_model(checkpoint_path, config):
    print(f"=== Loading model from {checkpoint_path} ===")
    model = V1MultimodalTransformer(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    print("✓ Model loaded")

    return model


# -------------------------------------------------------------------------
# 3. MC-dropout to estimate uncertainty for next-visit UPDRS predictions
# -------------------------------------------------------------------------
def collect_mc_dropout_predictions(model, loader, n_samples=MC_SAMPLES):
    """
    For each (patient, visit t) that predicts the next visit (t+1),
    collect multiple stochastic predictions via MC-dropout.

    Returns:
        pred_mean:   [N, 4]
        pred_std:    [N, 4]
        pred_lower:  [N, 4]
        pred_upper:  [N, 4]
        actuals:     [N, 4]   (UPDRS at visit t+1)
        time_months: [N]      (months_since_baseline for visit t+1)
    """
    print("=== Running MC-dropout inference ===")
    model.train()  # IMPORTANT: keep dropout ON
    # But we will still disable gradients
    pred_samples_per_visit = None
    actual_list = None
    time_list = None

    for s in range(n_samples):
        print(f"  MC sample {s + 1}/{n_samples}")
        # For each sample, we flatten in a consistent order over all batches
        sample_preds = []

        # We only need to build actuals/time once (on first sample)
        if s == 0:
            actual_list = []
            time_list = []

        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch_device = {
                    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass with dropout active
                outputs = model(
                    batch_device["static_values"],
                    batch_device["static_mask"],
                    batch_device["motor_values"],
                    batch_device["motor_mask"],
                    batch_device["nonmotor_values"],
                    batch_device["nonmotor_mask"],
                    batch_device["med_values"],
                    batch_device["med_mask"],
                    batch_device["age_at_visit_values"],
                    batch_device["age_at_visit_mask"],
                    batch_device["time_months"],
                    attention_mask=batch_device["attention_mask"],
                )

                # outputs["next_visit"]: [B, seq_len, 4]
                next_visit_preds = outputs["next_visit"].cpu()  # keep on CPU for aggregation

                # Batch tensors for alignment
                next_visit_targets = batch["next_visit_targets"].cpu()  # [B, seq_len, 4]
                time_months = batch["time_months"].cpu()                # [B, seq_len]
                attention_mask = batch["attention_mask"].cpu()          # [B, seq_len]

                B, L, _ = next_visit_preds.shape

                for b in range(B):
                    # True sequence length (non-padding) for this patient
                    seq_len = int(attention_mask[b].sum().item())
                    if seq_len < 2:
                        # Need at least two visits to form (t → t+1)
                        continue

                    # As in training: prediction at position t corresponds to target at t+1
                    # Use time of the NEXT visit (t+1) as x-axis
                    for t_idx in range(seq_len - 1):
                        pred_vec = next_visit_preds[b, t_idx]  # [4]
                        sample_preds.append(pred_vec.numpy())

                        if s == 0:
                            actual_vec = next_visit_targets[b, t_idx + 1]  # [4]
                            time_val = float(time_months[b, t_idx + 1].item())
                            actual_list.append(actual_vec.numpy())
                            time_list.append(time_val)

        sample_preds = np.stack(sample_preds, axis=0)  # [N, 4] for this MC sample

        if pred_samples_per_visit is None:
            # Initialize [N, n_samples, 4]
            N = sample_preds.shape[0]
            pred_samples_per_visit = np.zeros((N, n_samples, 4), dtype=np.float32)
        else:
            # Sanity-check N consistency
            if sample_preds.shape[0] != pred_samples_per_visit.shape[0]:
                raise RuntimeError("Inconsistent N across MC samples; check dataloader order.")

        pred_samples_per_visit[:, s, :] = sample_preds

    # Convert actuals/time to arrays
    actuals = np.stack(actual_list, axis=0)       # [N, 4]
    time_months = np.array(time_list, dtype=np.float32)  # [N]

    # Compute mean, std, and quantile bands across MC samples
    pred_mean = pred_samples_per_visit.mean(axis=1)                  # [N, 4]
    pred_std = pred_samples_per_visit.std(axis=1, ddof=1)            # [N, 4]
    pred_lower = np.percentile(pred_samples_per_visit, 2.5, axis=1)  # [N, 4]
    pred_upper = np.percentile(pred_samples_per_visit, 97.5, axis=1) # [N, 4]

    print(f"Total valid next-visit predictions: {pred_mean.shape[0]}")
    return pred_mean, pred_std, pred_lower, pred_upper, actuals, time_months

# -------------------------------------------------------------------------
# 4. Generate the 2×2 UPDRS plot with uncertainty bands
# -------------------------------------------------------------------------
def main():
    config = get_default_config()

    _, loader = build_full_dataset_and_loader(config)
    model = load_trained_model(str(CHECKPOINT_PATH), config)

    pred_mean, pred_std, pred_lower, pred_upper, actuals, time_months = \
        collect_mc_dropout_predictions(model, loader, n_samples=MC_SAMPLES)

    # Use the EvaluationPlotter helper
    io = PlotIO(str(OUTPUT_DIR), show=True, dpi=150)
    plotter = EvaluationPlotter(target_names=['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT'])

    # You can choose to use std-based bands OR quantile bands.
    # Example 1: std-based (mean ± 1.96 * std)
    plotter.plot_updrs_predictions_with_uncertainty(
        pred_mean=pred_mean,
        actuals=actuals,
        time_months=time_months,
        io=io,
        pred_std=pred_std,      # 95% ≈ mean ± 1.96*std inside the plotting function
        pred_lower=None,
        pred_upper=None,
        title="UPDRS Next-Visit Prediction (MC-dropout Uncertainty)",
        filename="updrs_uncertainty_mc_std.png",
    )

    # Example 2: quantile-based bands (2.5%–97.5%)
    plotter.plot_updrs_predictions_with_uncertainty(
        pred_mean=pred_mean,
        actuals=actuals,
        time_months=time_months,
        io=io,
        pred_std=None,
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        title="UPDRS Next-Visit Prediction (MC-dropout Quantile Band)",
        filename="updrs_uncertainty_mc_quantiles.png",
    )

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()