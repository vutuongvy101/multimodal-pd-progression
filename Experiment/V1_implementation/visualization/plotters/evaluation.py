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
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Dict, Tuple, Optional
import torch

from ..io import PlotIO
from ..config import VizConfig


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
        visit_numbers: Optional[np.ndarray] = None,  # [total_visits] - visit sequence number
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
            visit_numbers: Visit sequence number (default: index 0, 1, 2, ...)
            io: PlotIO instance for saving (optional)
            max_patients: Maximum number of unique patients to plot
            title: Plot title
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else predictions
        actual_np = actuals.cpu().numpy() if isinstance(actuals, torch.Tensor) else actuals

        if visit_numbers is None:
            visit_numbers = np.arange(len(patient_ids))

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
                pt_visits = visit_numbers[patient_mask]
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

            ax.set_xlabel('Visit Number', fontsize=10)
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
            io.save(f"single_patient_predictions.png")
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
        self.plot_patient_trajectories(predictions, actuals, patno, visit_numbers, io, **kwargs)

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


if __name__ == "__main__":
    """
    Demo mode: Generate example evaluation plots with synthetic data.
    
    Usage:
        python -m visualization.plotters.evaluation
    
    This creates sample predictions/actuals to demonstrate the visualization capabilities.
    """
    # Fix for OpenMP conflict (common with NumPy/SciPy/Matplotlib)
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Generate example evaluation plots")
    parser.add_argument("--output-dir", default="../V1_implementation/visualization/all_modalities/plots", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Display plots on screen (default: save only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("EvaluationPlotter - Demo Mode")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Show plots: {args.show}\n")

    # Generate synthetic data for demonstration
    n_visits = 500
    n_patients = 100

    # Synthetic next-visit predictions and actuals
    time_months = np.random.uniform(0, 60, n_visits)  # 0-60 months
    
    # Create realistic UPDRS-like data
    actuals_next = np.random.uniform(10, 80, (n_visits, 4))
    # Predictions with some error
    noise = np.random.normal(0, 5, (n_visits, 4))
    predictions_next = actuals_next + noise
    
    # Add some NaN values (missing data)
    nan_mask = np.random.random((n_visits, 4)) < 0.1
    actuals_next[nan_mask] = np.nan
    predictions_next[nan_mask] = np.nan

    # Patient IDs (using PATNO format - typical PPMI participant numbers)
    patient_ids = np.repeat(np.arange(n_patients), n_visits // n_patients + 1)[:n_visits]
    # Create realistic PATNO values (PPMI uses 4-5 digit numbers)
    patno = np.repeat(np.arange(3000, 3000 + n_patients), n_visits // n_patients + 1)[:n_visits]
    visit_numbers = np.tile(np.arange(n_visits // n_patients + 1), n_patients)[:n_visits]

    # Synthetic slope data
    slope_actuals = np.random.normal(0, 2, (n_patients, 4))  # Mean slope ≈ 0, std ≈ 2
    slope_predictions = slope_actuals + np.random.normal(0, 1, (n_patients, 4))  # With prediction error

    # Convert to tensors
    predictions_tensor = torch.from_numpy(predictions_next).float()
    actuals_tensor = torch.from_numpy(actuals_next).float()
    time_months_tensor = torch.from_numpy(time_months).float()
    slope_pred_tensor = torch.from_numpy(slope_predictions).float()
    slope_actual_tensor = torch.from_numpy(slope_actuals).float()

    # Create output directory with absolute path
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating directory: {output_path}\n")

    # Initialize plotter and IO
    io = PlotIO(str(output_path), show=args.show, dpi=150)
    plotter = EvaluationPlotter()

    # Generate all plots
    print("Generating demonstration plots with synthetic data...\n")
    plotter.plot_all_evaluations(
        predictions=predictions_tensor,
        actuals=actuals_tensor,
        time_months=time_months_tensor,
        patno=patno,
        slope_predictions=slope_pred_tensor,
        slope_actuals=slope_actual_tensor,
        visit_numbers=visit_numbers,
        single_patno=3000,  # Example: use first PATNO in demo
        io=io,
        output_dir=str(output_path),
        show=args.show
    )

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print(f"\nPlots saved to: {output_path}")
    print("\nGenerated files:")
    for png_file in sorted(output_path.glob("*.png")):
        print(f"  ✓ {png_file.name}")
    print("\nTo use with your own data, import and use:")
    print("  from visualization.plotters import EvaluationPlotter")
    print("  from visualization.io import PlotIO")
    print("  plotter = EvaluationPlotter()")
    print("  plotter.plot_all_evaluations(predictions, actuals, time_months, ...)")
