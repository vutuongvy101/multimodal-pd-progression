"""
Post-Training Model Evaluation Visualizations
Generates 4 main visualizations:
  A. Next-Visit Prediction: Predicted vs Actual UPDRS scatter plot over time
  B. Time-to-Visit Bucket Performance: MAE/RMSE across time intervals
  C. Patient-Level Trajectories: Spaghetti plot of patient progressions
  D. Distribution of Slope Errors: Histogram of slope prediction errors
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
import torch

from ..io import PlotIO


class EvaluationPlotter:
    """Post-training evaluation visualizations for regression predictions."""

    def __init__(self, target_names: Optional[list] = None):
        """
        Args:
            target_names: List of UPDRS target names, defaults to ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']
        """
        self.target_names = target_names or ['NP1RTOT', 'NP2PTOT', 'NP3TOT', 'NP4TOT']

    # ==================== HEAD 1: Next-Visit & Time-Bucket & Trajectories ====================

    def plot_next_visit_predictions(
        self,
        predictions: torch.Tensor,  # [total_visits, 4]
        actuals: torch.Tensor,  # [total_visits, 4]
        time_months: torch.Tensor,  # [total_visits]
        io: PlotIO,
        title: str = "Next-Visit Prediction Performance"
    ) -> None:
        """
        A. Next-Visit Prediction Scatter Plot
        
        Shows predicted vs actual UPDRS scores over time for all visits.
        
        Args:
            predictions: Model predictions [total_visits, 4]
            actuals: Ground truth values [total_visits, 4]
            time_months: Time in months since baseline [total_visits]
            io: PlotIO instance for saving
            title: Plot title
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        actual_np = actuals.cpu().numpy() if isinstance(actuals, torch.Tensor) else actuals
        time_np = time_months.cpu().numpy() if isinstance(time_months, torch.Tensor) else time_months

        # Create 2x2 subplot for 4 UPDRS totals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for target_idx in range(4):
            ax = axes[target_idx]

            # Get predictions and actuals for this UPDRS score
            pred = pred_np[:, target_idx]
            actual = actual_np[:, target_idx]

            # Filter out NaN values
            valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
            n_valid = valid_mask.sum()

            if n_valid == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.target_names[target_idx]} - No Data')
                continue

            time_valid = time_np[valid_mask]
            pred_valid = pred[valid_mask]
            actual_valid = actual[valid_mask]

            # Calculate metrics
            mae = np.mean(np.abs(pred_valid - actual_valid))
            rmse = np.sqrt(np.mean((pred_valid - actual_valid) ** 2))
            r2 = 1 - np.sum((actual_valid - pred_valid) ** 2) / np.sum((actual_valid - actual_valid.mean()) ** 2)

            # Scatter plot: color by time
            scatter = ax.scatter(time_valid, actual_valid, alpha=0.6, s=40, label='Actual', color='steelblue')
            ax.scatter(time_valid, pred_valid, alpha=0.6, s=40, label='Predicted', color='orange', marker='^')

            # Perfect prediction line
            min_val, max_val = min(actual_valid.min(), pred_valid.min()), max(actual_valid.max(), pred_valid.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='Perfect')

            ax.set_xlabel('Months Since Baseline', fontsize=10)
            ax.set_ylabel(f'{self.target_names[target_idx]} Score', fontsize=10)
            ax.set_title(
                f'{self.target_names[target_idx]} | MAE={mae:.2f} RMSE={rmse:.2f} R²={r2:.3f}',
                fontsize=11, fontweight='bold'
            )
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title} (n={len(time_np)} visits)", fontsize=14, fontweight='bold', y=0.995)
        io.save("next_visit_predictions.png")

    def plot_time_bucket_performance(
        self,
        predictions: torch.Tensor,  # [total_visits, 4]
        actuals: torch.Tensor,  # [total_visits, 4]
        time_months: torch.Tensor,  # [total_visits]
        io: PlotIO,
        time_buckets: Optional[list] = None,
        title: str = "Time-to-Visit Bucket Performance"
    ) -> None:
        """
        B. Time-to-Visit Bucket Performance
        
        Separate plots for different time intervals (Δt < 6, 6-12, 12-24, >24 months).
        Shows MAE and RMSE across time buckets for each UPDRS total.
        
        Args:
            predictions: Model predictions [total_visits, 4]
            actuals: Ground truth values [total_visits, 4]
            time_months: Time in months since baseline [total_visits]
            io: PlotIO instance for saving
            time_buckets: List of tuples (min, max, label) for time intervals
                         Default: [(0, 6, '<6mo'), (6, 12, '6-12mo'), (12, 24, '12-24mo'), (24, float('inf'), '>24mo')]
            title: Plot title
        """
        if time_buckets is None:
            time_buckets = [
                (0, 6, '<6 months'),
                (6, 12, '6-12 months'),
                (12, 24, '12-24 months'),
                (24, float('inf'), '>24 months')
            ]

        # Convert to numpy
        pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        actual_np = actuals.cpu().numpy() if isinstance(actuals, torch.Tensor) else actuals
        time_np = time_months.cpu().numpy() if isinstance(time_months, torch.Tensor) else time_months

        # Create 2x2 subplot for 4 UPDRS totals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for target_idx in range(4):
            ax = axes[target_idx]

            pred = pred_np[:, target_idx]
            actual = actual_np[:, target_idx]
            valid_mask = ~np.isnan(actual) & ~np.isnan(pred)

            if valid_mask.sum() == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.target_names[target_idx]} - No Data')
                continue

            time_valid = time_np[valid_mask]
            pred_valid = pred[valid_mask]
            actual_valid = actual[valid_mask]

            mae_list = []
            rmse_list = []
            bucket_labels = []

            for min_t, max_t, label in time_buckets:
                bucket_mask = (time_valid >= min_t) & (time_valid < max_t)
                if bucket_mask.sum() == 0:
                    continue

                mae = np.mean(np.abs(pred_valid[bucket_mask] - actual_valid[bucket_mask]))
                rmse = np.sqrt(np.mean((pred_valid[bucket_mask] - actual_valid[bucket_mask]) ** 2))

                mae_list.append(mae)
                rmse_list.append(rmse)
                bucket_labels.append(label)

            if not mae_list:
                ax.text(0.5, 0.5, 'No bucketed data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.target_names[target_idx]} - No Buckets')
                continue

            x = np.arange(len(bucket_labels))
            width = 0.35

            ax.bar(x - width / 2, mae_list, width, label='MAE', alpha=0.8, color='steelblue')
            ax.bar(x + width / 2, rmse_list, width, label='RMSE', alpha=0.8, color='orange')

            ax.set_xlabel('Time Interval', fontsize=10)
            ax.set_ylabel('Error', fontsize=10)
            ax.set_title(f'{self.target_names[target_idx]} - Error by Time Interval', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        total_valid = np.sum(~np.isnan(pred_np) & ~np.isnan(actual_np))
        fig.suptitle(f"{title} (n={int(total_valid)} visits)", fontsize=14, fontweight='bold', y=0.995)
        io.save("time_bucket_performance.png")

    def plot_patient_trajectories(
        self,
        predictions: torch.Tensor,  # [total_visits, 4]
        actuals: torch.Tensor,  # [total_visits, 4]
        patient_ids: np.ndarray,  # [total_visits] - patient identifier
        months_since_baseline: Optional[np.ndarray] = None,  # [total_visits] - time since baseline (months)
        io: Optional[PlotIO] = None,
        max_patients: int = 30,
        title: str = "Patient-Level UPDRS Trajectories"
    ) -> None:
        """
        C. Patient-Level Trajectories (Spaghetti Plot)
        
        Shows individual patient progressions with actual vs predicted UPDRS trajectories.
        Each line represents one patient's progression over visits.
        
        Args:
            predictions: Model predictions [total_visits, 4]
            actuals: Ground truth values [total_visits, 4]
            patient_ids: Patient identifier for each visit [total_visits]
            months_since_baseline: Time since baseline in months (default: index 0, 1, 2, ...)
            io: PlotIO instance for saving (optional)
            max_patients: Maximum number of unique patients to plot
            title: Plot title
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        actual_np = actuals.cpu().numpy() if isinstance(actuals, torch.Tensor) else actuals

        if months_since_baseline is None:
            months_since_baseline = np.arange(len(patient_ids))

        # Create 2x2 subplot for 4 UPDRS totals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        unique_patients = np.unique(patient_ids)
        patients_to_plot = unique_patients[:max_patients]

        for target_idx in range(4):
            ax = axes[target_idx]

            actual = actual_np[:, target_idx]
            pred = pred_np[:, target_idx]

            patient_count = 0
            for patient_id in patients_to_plot:
                patient_mask = patient_ids == patient_id
                pt_visits = months_since_baseline[patient_mask]
                pt_actual = actual[patient_mask]
                pt_pred = pred[patient_mask]

                # Filter NaN values
                valid_mask = ~np.isnan(pt_actual) & ~np.isnan(pt_pred)
                if valid_mask.sum() < 2:  # Need at least 2 points for a line
                    continue

                pt_visits_valid = pt_visits[valid_mask]
                pt_actual_valid = pt_actual[valid_mask]
                pt_pred_valid = pt_pred[valid_mask]

                # Sort by visit number
                sort_idx = np.argsort(pt_visits_valid)
                pt_visits_valid = pt_visits_valid[sort_idx]
                pt_actual_valid = pt_actual_valid[sort_idx]
                pt_pred_valid = pt_pred_valid[sort_idx]

                # Plot trajectories with transparency
                ax.plot(pt_visits_valid, pt_actual_valid, 'o-', alpha=0.3, linewidth=1.5, color='steelblue', markersize=4)
                ax.plot(pt_visits_valid, pt_pred_valid, 's--', alpha=0.3, linewidth=1.5, color='orange', markersize=4)

                patient_count += 1

            # Add legend (only once)
            if patient_count > 0:
                from matplotlib.lines import Line2D
                custom_lines = [
                    Line2D([0], [0], marker='o', color='steelblue', linewidth=1.5, markersize=4, label='Actual'),
                    Line2D([0], [0], marker='s', color='orange', linewidth=1.5, linestyle='--', markersize=4, label='Predicted')
                ]
                ax.legend(handles=custom_lines, fontsize=9)

            ax.set_xlabel('Months Since Baseline', fontsize=10)
            ax.set_ylabel(f'{self.target_names[target_idx]} Score', fontsize=10)
            ax.set_title(f'{self.target_names[target_idx]} - {patient_count} patients shown', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        total_patients = len(np.unique(patient_ids))
        total_visits = len(patient_ids)
        fig.suptitle(f"{title} (n={total_patients} patients, {total_visits} visits)", fontsize=14, fontweight='bold', y=0.995)
        if io:
            io.save("patient_trajectories.png")
        else:
            plt.tight_layout()
            plt.show()

    # ==================== E: Single Patient Prediction ====================

    def plot_single_patient_prediction(
        self,
        predictions: torch.Tensor,  # [total_visits, 4]
        actuals: torch.Tensor,  # [total_visits, 4]
        time_months: torch.Tensor,  # [total_visits]
        patno: np.ndarray,  # [total_visits] - PPMI participant number
        target_patno: int,
        io: Optional[PlotIO] = None,
        title: str = "Single Patient: Predicted vs Actual UPDRS Scores"
    ) -> None:
        """
        E. Single Patient Prediction Performance
        
        Detailed view of one patient's predicted vs actual UPDRS progression over time.
        Similar to Graph A but focused on individual patient trajectory.
        
        Args:
            predictions: Model predictions [total_visits, 4]
            actuals: Ground truth values [total_visits, 4]
            time_months: Time in months since baseline [total_visits]
            patno: PPMI participant number for each visit [total_visits]
            target_patno: The specific PATNO to visualize
            io: PlotIO instance for saving (optional)
            title: Plot title
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        actual_np = actuals.cpu().numpy() if isinstance(actuals, torch.Tensor) else actuals
        time_np = time_months.cpu().numpy() if isinstance(time_months, torch.Tensor) else time_months

        # Filter data for this patient (by PATNO)
        patient_mask = patno == target_patno
        if patient_mask.sum() == 0:
            print(f"Warning: PATNO {target_patno} not found in dataset")
            return

        pt_time = time_np[patient_mask]
        pt_pred = pred_np[patient_mask]
        pt_actual = actual_np[patient_mask]

        # Sort by time
        sort_idx = np.argsort(pt_time)
        pt_time = pt_time[sort_idx]
        pt_pred = pt_pred[sort_idx]
        pt_actual = pt_actual[sort_idx]

        # Create 2x2 subplot for 4 UPDRS totals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for target_idx in range(4):
            ax = axes[target_idx]

            pred = pt_pred[:, target_idx]
            actual = pt_actual[:, target_idx]

            # Filter out NaN values
            valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
            n_visits = valid_mask.sum()

            if n_visits == 0:
                ax.text(0.5, 0.5, 'No valid data for this patient', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.target_names[target_idx]} - No Data')
                continue

            time_valid = pt_time[valid_mask]
            pred_valid = pred[valid_mask]
            actual_valid = actual[valid_mask]

            # Calculate metrics for this patient
            mae = np.mean(np.abs(pred_valid - actual_valid))
            rmse = np.sqrt(np.mean((pred_valid - actual_valid) ** 2))
            
            # Avoid division by zero for R²
            ss_res = np.sum((actual_valid - pred_valid) ** 2)
            ss_tot = np.sum((actual_valid - actual_valid.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            # Plot with lines connecting visits
            ax.plot(time_valid, actual_valid, 'o-', linewidth=2.5, markersize=8, label='Actual', color='steelblue', alpha=0.8)
            ax.plot(time_valid, pred_valid, 's--', linewidth=2.5, markersize=8, label='Predicted', color='orange', alpha=0.8)

            # Perfect prediction line (if there's range)
            if len(actual_valid) > 0:
                min_val = min(actual_valid.min(), pred_valid.min())
                max_val = max(actual_valid.max(), pred_valid.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1.5, label='Perfect')

            ax.set_xlabel('Months Since Baseline', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{self.target_names[target_idx]} Score', fontsize=11, fontweight='bold')
            ax.set_title(
                f'{self.target_names[target_idx]} | MAE={mae:.2f} RMSE={rmse:.2f} R²={r2:.3f} | {n_visits} visits',
                fontsize=12, fontweight='bold'
            )
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title} (PATNO: {target_patno})", fontsize=15, fontweight='bold', y=0.995)
        if io:
            io.save(f"single_patient_PATNO_{target_patno}_predictions.png")
        else:
            plt.tight_layout()
            plt.show()

    # ==================== HEAD 2: Slope Error Distribution ====================

    def plot_slope_error_distributions(
        self,
        slope_predictions: torch.Tensor,  # [n_patients, 4]
        slope_actuals: torch.Tensor,  # [n_patients, 4]
        io: PlotIO,
        title: str = "Distribution of Slope Prediction Errors"
    ) -> None:
        """
        D. Distribution of Slope Errors
        
        Histograms of slope prediction errors (actual - predicted) for each UPDRS total.
        Shows whether errors are normally distributed or exhibit bias.
        
        Args:
            slope_predictions: Model slope predictions [n_patients, 4]
            slope_actuals: Ground truth slopes [n_patients, 4]
            io: PlotIO instance for saving
            title: Plot title
        """
        # Convert to numpy
        pred_np = slope_predictions.cpu().numpy() if isinstance(slope_predictions, torch.Tensor) else slope_predictions
        actual_np = slope_actuals.cpu().numpy() if isinstance(slope_actuals, torch.Tensor) else slope_actuals

        # Create 2x2 subplot for 4 UPDRS totals
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for target_idx in range(4):
            ax = axes[target_idx]

            pred = pred_np[:, target_idx]
            actual = actual_np[:, target_idx]

            # Filter out NaN values
            valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
            n_valid = valid_mask.sum()

            if n_valid == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{self.target_names[target_idx]} - No Data')
                continue

            errors = (actual[valid_mask] - pred[valid_mask])

            # Calculate statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            median_error = np.median(errors)

            # Histogram with KDE
            ax.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)

            # Overlay normal distribution
            x = np.linspace(errors.min(), errors.max(), 100)
            normal_dist = (1 / (std_error * np.sqrt(2 * np.pi))) * np.exp(-(x - mean_error) ** 2 / (2 * std_error ** 2))
            ax.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Fit')

            # Add reference lines
            ax.axvline(mean_error, color='green', linestyle='--', linewidth=2, label=f'Mean={mean_error:.3f}')
            ax.axvline(median_error, color='orange', linestyle='--', linewidth=2, label=f'Median={median_error:.3f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

            ax.set_xlabel('Slope Error (Actual - Predicted)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(
                f'{self.target_names[target_idx]} | μ={mean_error:.3f} σ={std_error:.3f} (n={n_valid})',
                fontsize=11, fontweight='bold'
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

        total_valid_patients = np.sum(~np.isnan(pred_np) & ~np.isnan(actual_np))
        fig.suptitle(f"{title} (n={int(total_valid_patients)} patients)", fontsize=14, fontweight='bold', y=0.995)
        io.save("slope_error_distributions.png")

    # ==================== Combined Evaluation ====================

    def plot_all_evaluations(
        self,
        predictions: torch.Tensor,  # [total_visits, 4]
        actuals: torch.Tensor,  # [total_visits, 4]
        time_months: torch.Tensor,  # [total_visits]
        patno: np.ndarray,  # [total_visits] - PPMI participant number
        slope_predictions: Optional[torch.Tensor] = None,  # [n_patients, 4]
        slope_actuals: Optional[torch.Tensor] = None,  # [n_patients, 4]
        visit_numbers: Optional[np.ndarray] = None,  # [total_visits]
        single_patno: Optional[int] = None,  # Specific PATNO to visualize
        io: Optional[PlotIO] = None,
        output_dir: str = "../V1_implementation/visualization/all_modalities/plots",
        show: bool = False,
        **kwargs
    ) -> None:
        """
        Generate all 4 evaluation plots in sequence.
        
        Args:
            predictions: Model predictions [total_visits, 4]
            actuals: Ground truth values [total_visits, 4]
            time_months: Time in months since baseline [total_visits]
            patno: PPMI participant number for each visit [total_visits]
            slope_predictions: Model slope predictions [n_patients, 4] (optional)
            slope_actuals: Ground truth slopes [n_patients, 4] (optional)
            visit_numbers: Visit sequence number (optional)
            single_patno: Specific PATNO to visualize in detail (optional, if None uses first patient)
            io: PlotIO instance (auto-created if None)
            output_dir: Directory to save plots
            show: Whether to display plots
            **kwargs: Additional arguments for individual plot methods
        """
        if io is None:
            io = PlotIO(output_dir, show=show, dpi=150)

        print("Generating evaluation plots...")

        # A. Next-Visit Predictions
        print("  A. Generating next-visit prediction plots...")
        self.plot_next_visit_predictions(predictions, actuals, time_months, io, **kwargs)

        # B. Time Bucket Performance
        print("  B. Generating time-bucket performance plots...")
        self.plot_time_bucket_performance(predictions, actuals, time_months, io, **kwargs)

        # C. Patient Trajectories
        print("  C. Generating patient trajectory plots...")
        self.plot_patient_trajectories(predictions, actuals, patno, time_months, io, **kwargs)

        # D. Slope Error Distributions (if available)
        if slope_predictions is not None and slope_actuals is not None:
            print("  D. Generating slope error distribution plots...")
            self.plot_slope_error_distributions(slope_predictions, slope_actuals, io, **kwargs)
        else:
            print("  D. Skipping slope plots (no slope data provided)")

        # E. Single Patient Prediction (optional)
        if single_patno is not None:
            print(f"  E. Generating single patient prediction plot (PATNO: {single_patno})...")
            self.plot_single_patient_prediction(predictions, actuals, time_months, patno, single_patno, io, **kwargs)
        else:
            # If not specified, use first patient in dataset
            unique_patno = np.unique(patno)
            if len(unique_patno) > 0:
                first_patno = unique_patno[0]
                print(f"  E. Generating single patient prediction plot (PATNO: {first_patno})...")
                self.plot_single_patient_prediction(predictions, actuals, time_months, patno, first_patno, io, **kwargs)

        print(f"✓ All evaluation plots saved to {io.output_dir}")


@torch.no_grad()
def evaluate_fold_checkpoint(
    fold_index: int,
    checkpoint_path: str,
    output_dir: str,
    device: Optional[str] = None,
    n_splits: int = 10,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    normalize_features: bool = True,
    single_patno: Optional[int] = None,
    debug_longitudinal_csv: Optional[str] = None,
    debug_longitudinal_max_rows: Optional[int] = None,
    load_predictions_path: Optional[str] = None,
    save_predictions_path: Optional[str] = None,
    select_patnos_output: Optional[str] = None,
    select_patnos_min_visits: int = 3,
) -> None:
    """
    Evaluate a fold checkpoint on its validation dataset and generate plots.

    Args:
        fold_index: 1-based fold index (e.g., 7 for fold_7)
        checkpoint_path: Path to best_checkpoint.pt
        output_dir: Where plots should be saved
        device: torch device string (defaults to config training device)
        n_splits: Number of folds used during training
        test_ratio: Test split ratio used during training
        random_seed: Random seed used for split
        normalize_features: Whether features were normalized during training
        single_patno: Optional PATNO for single-patient plot
        debug_longitudinal_csv: Optional CSV path to dump longitudinal data before splits
        debug_longitudinal_max_rows: Optional row limit for CSV dump
        load_predictions_path: Optional .npz with cached predictions/targets
        save_predictions_path: Optional .npz path to save predictions/targets
        select_patnos_output: Optional CSV path to write PATNO selection summary
        select_patnos_min_visits: Minimum visits for PATNO selection summary
    """
    from training.config import get_default_config
    from data.data_integrator import DataIntegrator
    from data.dataset import create_kfold_dataloaders
    from models.v1_model import V1MultimodalTransformer

    if load_predictions_path:
        cache = np.load(load_predictions_path, allow_pickle=False)
        predictions = cache["predictions"]
        actuals = cache["actuals"]
        time_months = cache["time_months"]
        patno = cache["patno"]
        slope_pred = cache["slope_pred"] if "slope_pred" in cache.files else None
        slope_actual = cache["slope_actual"] if "slope_actual" in cache.files else None
    else:
        if fold_index < 1 or fold_index > n_splits:
            raise ValueError(f"fold_index must be in [1, {n_splits}] but got {fold_index}")

        config = get_default_config()
        if device is None:
            device = config.training.device

        print("Preparing data for evaluation...")
        integrator = DataIntegrator(config, normalize_features=normalize_features)
        prepared = integrator.prepare_final_dataset()

        fold_dataloaders, _ = create_kfold_dataloaders(
            prepared_data=prepared,
            config=config,
            n_splits=n_splits,
            num_workers=0,
            test_ratio=test_ratio,
            random_seed=random_seed,
            debug_longitudinal_csv=debug_longitudinal_csv,
            debug_longitudinal_max_rows=debug_longitudinal_max_rows
        )

        _, val_loader = fold_dataloaders[fold_index - 1]

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = V1MultimodalTransformer(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        all_pred = []
        all_actual = []
        all_time = []
        all_patno = []
        all_slope_pred = []
        all_slope_actual = []

        for batch in val_loader:
            if "patno" not in batch:
                raise KeyError("Batch is missing 'patno'. Ensure dataset/collate_fn include PATNO.")

            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            preds = model(
                batch['static_values'], batch['static_mask'],
                batch['motor_values'], batch['motor_mask'],
                batch['nonmotor_values'], batch['nonmotor_mask'],
                batch['med_values'], batch['med_mask'],
                batch['age_at_visit_values'], batch['age_at_visit_mask'],
                batch['time_months'], batch['attention_mask']
            )

            pred_next = preds['next_visit'].detach().cpu()
            actual_next = batch['next_visit_targets'].detach().cpu()
            time_months = batch['time_months'].detach().cpu()
            attn = batch['attention_mask'].detach().cpu() > 0.5

            patno = batch['patno'].detach().cpu()
            patno_per_visit = patno.unsqueeze(1).expand(-1, pred_next.shape[1])

            pred_next = pred_next[attn]
            actual_next = actual_next[attn]
            time_months = time_months[attn]
            patno_per_visit = patno_per_visit[attn]

            all_pred.append(pred_next)
            all_actual.append(actual_next)
            all_time.append(time_months)
            all_patno.append(patno_per_visit)

            if 'slope_targets' in batch:
                all_slope_pred.append(preds['slope'].detach().cpu())
                all_slope_actual.append(batch['slope_targets'].detach().cpu())

        predictions = torch.cat(all_pred, dim=0)
        actuals = torch.cat(all_actual, dim=0)
        time_months = torch.cat(all_time, dim=0)
        patno = torch.cat(all_patno, dim=0).numpy()

        slope_pred = torch.cat(all_slope_pred, dim=0) if all_slope_pred else None
        slope_actual = torch.cat(all_slope_actual, dim=0) if all_slope_actual else None

        if save_predictions_path:
            payload = {
                "predictions": predictions.numpy(),
                "actuals": actuals.numpy(),
                "time_months": time_months.numpy(),
                "patno": patno,
            }
            if slope_pred is not None:
                payload["slope_pred"] = slope_pred.numpy()
            if slope_actual is not None:
                payload["slope_actual"] = slope_actual.numpy()
            np.savez(save_predictions_path, **payload)

    if select_patnos_output:
        from data.patient_selection import select_patnos_from_predictions

        actuals_np = actuals.numpy() if isinstance(actuals, torch.Tensor) else actuals
        patno_np = patno.numpy() if isinstance(patno, torch.Tensor) else patno
        out_csv = select_patnos_from_predictions(
            actuals=actuals_np,
            patno=patno_np,
            output_path=select_patnos_output,
            min_visits=select_patnos_min_visits,
        )
        print(f"Saved PATNO summary to {out_csv}")

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    io = PlotIO(str(out_path), show=False, dpi=150)

    plotter = EvaluationPlotter()

    # Always generate the global plots once
    plotter.plot_all_evaluations(
        predictions=predictions,
        actuals=actuals,
        time_months=time_months,
        patno=patno,
        slope_predictions=slope_pred,
        slope_actuals=slope_actual,
        single_patno=None,
        io=io,
        output_dir=str(out_path),
        show=False
    )

    if single_patno is not None:
        print(f"Generating single patient plot for PATNO: {single_patno}")
        plotter.plot_single_patient_prediction(
            predictions=predictions,
            actuals=actuals,
            time_months=time_months,
            patno=patno,
            target_patno=int(single_patno),
            io=io
        )


if __name__ == "__main__":
    """CLI entrypoint for fold evaluation."""
    # Fix for OpenMP conflict (common with NumPy/SciPy/Matplotlib)
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate fold checkpoint")
    parser.add_argument("--fold", type=int, default=7, help="Fold index (1-based) for evaluation")
    parser.add_argument("--checkpoint", default=None, help="Path to best_checkpoint.pt (defaults to fold path)")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of CV folds used during training")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio used during training")
    parser.add_argument("--device", default=None, help="Device to use (e.g., cuda, cpu)")
    parser.add_argument("--output-dir", default="../V1_implementation/visualization/all_modalities/plots", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Display plots on screen (default: save only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single-patno", type=int, default=None, help="PATNO for single-patient plot")
    parser.add_argument("--debug-longitudinal-csv", default=None,
                        help="Write longitudinal data CSV before split (for debugging)")
    parser.add_argument("--debug-longitudinal-max-rows", type=int, default=None,
                        help="Max rows to dump in longitudinal CSV")
    parser.add_argument("--load-predictions", default=None,
                        help="Load cached predictions/targets (.npz) to skip evaluation")
    parser.add_argument("--save-predictions", default=None,
                        help="Save predictions/targets (.npz) after evaluation")
    parser.add_argument("--select-patnos-output", default=None,
                        help="Write PATNO selection summary CSV from validation set")
    parser.add_argument("--select-patnos-min-visits", type=int, default=3,
                        help="Minimum visits for PATNO selection summary")
    parser.add_argument("--list-cached-patnos", action="store_true",
                        help="List unique PATNOs in cached predictions and exit")
    parser.add_argument("--no-normalize", action="store_true", help="Disable feature normalization")
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("EvaluationPlotter")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Show plots: {args.show}\n")

    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    default_ckpt = Path("models") / "checkpoints" / "modalities_age_at_visit+medication+motor+non_motor+static" / f"fold_{args.fold}" / "best_checkpoint.pt"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_ckpt

    print(f"Using fold: {args.fold}")
    print(f"Checkpoint: {checkpoint_path}")

    if args.list_cached_patnos:
        if not args.load_predictions:
            raise ValueError("--list-cached-patnos requires --load-predictions")
        cache = np.load(args.load_predictions, allow_pickle=False)
        if "patno" not in cache.files:
            raise KeyError("Cached predictions are missing 'patno'.")
        unique_patnos = np.unique(cache["patno"])
        print(f"Cached PATNOs (n={len(unique_patnos)}):")
        print(", ".join(str(int(p)) for p in unique_patnos))
        raise SystemExit(0)

    evaluate_fold_checkpoint(
        fold_index=args.fold,
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_path),
        device=args.device,
        n_splits=args.n_splits,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        normalize_features=not args.no_normalize,
        single_patno=args.single_patno,
        debug_longitudinal_csv=args.debug_longitudinal_csv,
        debug_longitudinal_max_rows=args.debug_longitudinal_max_rows,
        load_predictions_path=args.load_predictions,
        save_predictions_path=args.save_predictions,
        select_patnos_output=args.select_patnos_output,
        select_patnos_min_visits=args.select_patnos_min_visits
    )
    raise SystemExit(0)
