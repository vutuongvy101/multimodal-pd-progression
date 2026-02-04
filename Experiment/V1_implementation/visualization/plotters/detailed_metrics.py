import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..history import History
from ..io import PlotIO
from ..config import VizConfig


class DetailedMetricsPlotter:
    """Comprehensive epoch-by-epoch visualization of all training metrics."""

    def plot_single(self, h: History, io: PlotIO, cfg: VizConfig, title: str = "Detailed Metrics") -> None:
        """Create detailed visualizations showing all metrics over epochs."""
        n_epochs = h.n_epochs()
        if n_epochs == 0:
            return

        epochs = np.arange(1, n_epochs + 1)
        best_epoch = h.best_epoch()

        # 1. Comprehensive Loss Curves (all losses together)
        self._plot_all_losses(h, io, epochs, best_epoch, title)

        # 2. Learning Rate (detailed)
        self._plot_learning_rate_detailed(h, io, epochs, title)

        # 3. Next Visit Metrics per Target (MAE, RMSE, R2, Correlation, Spearman)
        self._plot_next_visit_metrics(h, io, epochs, best_epoch, cfg, title)

        # 4. Next Visit Overall Metrics
        self._plot_next_visit_overall(h, io, epochs, best_epoch, title)

        # 5. Slope Metrics per Target
        self._plot_slope_metrics_per_target(h, io, epochs, best_epoch, title)

        # 6. Slope Overall Metrics
        self._plot_slope_overall(h, io, epochs, best_epoch, title)

    def _plot_all_losses(self, h: History, io: PlotIO, epochs: np.ndarray, best_epoch: int, title: str) -> None:
        """Plot all loss types in subplots."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        loss_pairs = [
            ("train_loss", "val_loss", "Overall Loss"),
            ("train_loss_next_visit", "val_loss_next_visit", "Next Visit Loss"),
            ("train_loss_slope", "val_loss_slope", "Slope Loss"),
        ]

        for idx, (train_key, val_key, sub_title) in enumerate(loss_pairs):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            train_loss = h.series(train_key)
            val_loss = h.series(val_key)

            if train_loss is not None:
                ax.plot(epochs, train_loss, label=train_key, linewidth=2, alpha=0.8)
            if val_loss is not None:
                ax.plot(epochs, val_loss, label=val_key, linewidth=2, alpha=0.8)
                # Mark best epoch
                if 0 <= best_epoch < len(val_loss):
                    ax.scatter([best_epoch + 1], [float(val_loss[best_epoch])],
                               s=100, marker='x', color='red', zorder=5,
                               label=f'Best Epoch {best_epoch + 1}')

            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_title(sub_title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Learning rate in the 4th subplot
        ax_lr = fig.add_subplot(gs[1, 1])
        lr = h.series("learning_rate")
        if lr is not None:
            ax_lr.plot(epochs, lr, label="learning_rate", linewidth=2, color='purple', alpha=0.8)
            ax_lr.set_xlabel("Epoch", fontsize=11)
            ax_lr.set_ylabel("Learning Rate", fontsize=11)
            ax_lr.set_title("Learning Rate", fontsize=12, fontweight='bold')
            ax_lr.legend(fontsize=9)
            ax_lr.grid(True, alpha=0.3)
            ax_lr.set_yscale('log')

        fig.suptitle(f"{title} - All Losses & Learning Rate", fontsize=14, fontweight='bold', y=0.995)
        io.save("detailed_all_losses.png")

    def _plot_learning_rate_detailed(self, h: History, io: PlotIO, epochs: np.ndarray, title: str) -> None:
        """Detailed learning rate plot with linear and log scales."""
        lr = h.series("learning_rate")
        if lr is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        ax1.plot(epochs, lr, linewidth=2, color='purple', alpha=0.8)
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Learning Rate", fontsize=11)
        ax1.set_title("Learning Rate (Linear Scale)", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Log scale
        ax2.plot(epochs, lr, linewidth=2, color='purple', alpha=0.8)
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Learning Rate", fontsize=11)
        ax2.set_title("Learning Rate (Log Scale)", fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{title} - Learning Rate Detailed", fontsize=14, fontweight='bold')
        io.save("detailed_learning_rate.png")

    def _plot_next_visit_metrics(self, h: History, io: PlotIO, epochs: np.ndarray,
                                 best_epoch: int, cfg: VizConfig, title: str) -> None:
        """Plot next visit metrics (MAE, RMSE, R2, Correlation, Spearman) per target over epochs."""
        n_epochs = len(epochs)

        # Extract metrics for all targets
        targets = set()
        metrics_data = {}

        for e in range(n_epochs):
            vme = h.val_metrics_epoch(e)
            if not vme:
                continue
            nv = vme.get("next_visit", {})
            if not isinstance(nv, dict):
                continue

            for target, metrics in nv.items():
                if not isinstance(metrics, dict):
                    continue
                targets.add(target)
                if target not in metrics_data:
                    metrics_data[target] = {
                        'mae': [], 'rmse': [], 'r2': [],
                        'correlation': [], 'spearman': []
                    }

                metrics_data[target]['mae'].append(metrics.get('mae', np.nan))
                metrics_data[target]['rmse'].append(metrics.get('rmse', np.nan))
                metrics_data[target]['r2'].append(metrics.get('r2', np.nan))
                metrics_data[target]['correlation'].append(metrics.get('correlation', np.nan))
                metrics_data[target]['spearman'].append(metrics.get('spearman', np.nan))

        if not targets:
            return

        targets = sorted(targets)
        n_targets = len(targets)

        # Create subplots: one row per target, one column per metric type
        fig = plt.figure(figsize=(20, 5 * n_targets))
        gs = GridSpec(n_targets, 5, figure=fig, hspace=0.4, wspace=0.3)

        metric_keys = ['mae', 'rmse', 'r2', 'correlation', 'spearman']
        metric_titles = ['MAE', 'RMSE', 'R²', 'Correlation', 'Spearman']

        for target_idx, target in enumerate(targets):
            data = metrics_data[target]

            for metric_idx, (key, metric_title) in enumerate(zip(metric_keys, metric_titles)):
                ax = fig.add_subplot(gs[target_idx, metric_idx])
                values = np.array(data[key])
                valid_mask = np.isfinite(values)

                if valid_mask.any():
                    valid_epochs = epochs[valid_mask]
                    valid_values = values[valid_mask]
                    ax.plot(valid_epochs, valid_values, linewidth=2, alpha=0.8)

                    # Mark best epoch
                    if 0 <= best_epoch < len(values) and np.isfinite(values[best_epoch]):
                        ax.scatter([best_epoch + 1], [values[best_epoch]],
                                   s=100, marker='x', color='red', zorder=5,
                                   label=f'Best Epoch {best_epoch + 1}')
                        ax.legend(fontsize=9)

                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel(metric_title, fontsize=10)
                ax.set_title(f"{target} - {metric_title}", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title} - Next Visit Metrics per Target", fontsize=14, fontweight='bold', y=0.995)
        io.save("detailed_next_visit_metrics.png")

    def _plot_next_visit_overall(self, h: History, io: PlotIO, epochs: np.ndarray,
                                 best_epoch: int, title: str) -> None:
        """Plot next visit overall metrics over epochs."""
        n_epochs = len(epochs)

        metrics = {
            'macro_avg_mae': [], 'macro_avg_rmse': [], 'macro_avg_r2': [],
            'macro_avg_correlation': [], 'macro_avg_spearman': [],
            'weighted_avg_mae': [], 'weighted_avg_rmse': []
        }

        for e in range(n_epochs):
            vme = h.val_metrics_epoch(e)
            if not vme:
                continue
            nvo = vme.get("next_visit_overall", {})
            if not isinstance(nvo, dict):
                continue

            for key in metrics.keys():
                metrics[key].append(nvo.get(key, np.nan))

        if not any(np.isfinite(v).any() for v in metrics.values()):
            return

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        metric_data = [
            ('macro_avg_mae', 'Macro Avg MAE'),
            ('macro_avg_rmse', 'Macro Avg RMSE'),
            ('macro_avg_r2', 'Macro Avg R²'),
            ('macro_avg_correlation', 'Macro Avg Correlation'),
            ('macro_avg_spearman', 'Macro Avg Spearman'),
            ('weighted_avg_mae', 'Weighted Avg MAE'),
            ('weighted_avg_rmse', 'Weighted Avg RMSE'),
        ]

        for idx, (key, label) in enumerate(metric_data):
            if idx >= len(axes):
                break
            ax = axes[idx]
            values = np.array(metrics[key])
            valid_mask = np.isfinite(values)

            if valid_mask.any():
                valid_epochs = epochs[valid_mask]
                valid_values = values[valid_mask]
                ax.plot(valid_epochs, valid_values, linewidth=2, alpha=0.8)

                if 0 <= best_epoch < len(values) and np.isfinite(values[best_epoch]):
                    ax.scatter([best_epoch + 1], [values[best_epoch]],
                               s=100, marker='x', color='red', zorder=5,
                               label=f'Best Epoch {best_epoch + 1}')
                    ax.legend(fontsize=9)

            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        if len(metric_data) < len(axes):
            axes[-1].axis('off')

        fig.suptitle(f"{title} - Next Visit Overall Metrics", fontsize=14, fontweight='bold')
        io.save("detailed_next_visit_overall.png")

    def _plot_slope_metrics_per_target(self, h: History, io: PlotIO, epochs: np.ndarray,
                                       best_epoch: int, title: str) -> None:
        """Plot slope metrics per target over epochs."""
        n_epochs = len(epochs)

        targets = set()
        metrics_data = {}

        for e in range(n_epochs):
            vme = h.val_metrics_epoch(e)
            if not vme:
                continue
            slope = vme.get("slope", {})
            if not isinstance(slope, dict):
                continue

            for target_key, metrics in slope.items():
                if not isinstance(metrics, dict):
                    continue
                # Extract target name (e.g., "NP3TOT_slope" -> "NP3TOT")
                target = target_key.replace('_slope', '')
                targets.add(target)

                if target not in metrics_data:
                    metrics_data[target] = {'mae': [], 'rmse': [], 'spearman': []}

                metrics_data[target]['mae'].append(metrics.get('mae', np.nan))
                metrics_data[target]['rmse'].append(metrics.get('rmse', np.nan))
                metrics_data[target]['spearman'].append(metrics.get('spearman', np.nan))

        if not targets:
            return

        targets = sorted(targets)
        n_targets = len(targets)

        fig, axes = plt.subplots(n_targets, 3, figsize=(15, 4 * n_targets))
        if n_targets == 1:
            axes = axes.reshape(1, -1)

        metric_keys = ['mae', 'rmse', 'spearman']
        metric_titles = ['MAE', 'RMSE', 'Spearman']

        for target_idx, target in enumerate(targets):
            data = metrics_data[target]

            for metric_idx, (key, metric_title) in enumerate(zip(metric_keys, metric_titles)):
                ax = axes[target_idx, metric_idx]
                values = np.array(data[key])
                valid_mask = np.isfinite(values)

                if valid_mask.any():
                    valid_epochs = epochs[valid_mask]
                    valid_values = values[valid_mask]
                    ax.plot(valid_epochs, valid_values, linewidth=2, alpha=0.8)

                    if 0 <= best_epoch < len(values) and np.isfinite(values[best_epoch]):
                        ax.scatter([best_epoch + 1], [values[best_epoch]],
                                   s=100, marker='x', color='red', zorder=5,
                                   label=f'Best Epoch {best_epoch + 1}')
                        ax.legend(fontsize=9)

                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel(metric_title, fontsize=10)
                ax.set_title(f"{target} Slope - {metric_title}", fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title} - Slope Metrics per Target", fontsize=14, fontweight='bold')
        io.save("detailed_slope_per_target.png")

    def _plot_slope_overall(self, h: History, io: PlotIO, epochs: np.ndarray,
                            best_epoch: int, title: str) -> None:
        """Plot slope overall metrics over epochs."""
        n_epochs = len(epochs)

        metrics = {'mae': [], 'rmse': [], 'spearman': []}

        for e in range(n_epochs):
            vme = h.val_metrics_epoch(e)
            if not vme:
                continue
            so = vme.get("slope_overall", {})
            if not isinstance(so, dict):
                continue

            metrics['mae'].append(so.get('mae', np.nan))
            metrics['rmse'].append(so.get('rmse', np.nan))
            metrics['spearman'].append(so.get('spearman', np.nan))

        if not any(np.isfinite(v).any() for v in metrics.values()):
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metric_keys = ['mae', 'rmse', 'spearman']
        metric_titles = ['MAE', 'RMSE', 'Spearman']

        for idx, (key, metric_title) in enumerate(zip(metric_keys, metric_titles)):
            ax = axes[idx]
            values = np.array(metrics[key])
            valid_mask = np.isfinite(values)

            if valid_mask.any():
                valid_epochs = epochs[valid_mask]
                valid_values = values[valid_mask]
                ax.plot(valid_epochs, valid_values, linewidth=2, alpha=0.8)

                if 0 <= best_epoch < len(values) and np.isfinite(values[best_epoch]):
                    ax.scatter([best_epoch + 1], [values[best_epoch]],
                               s=100, marker='x', color='red', zorder=5,
                               label=f'Best Epoch {best_epoch + 1}')
                    ax.legend(fontsize=9)

            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric_title, fontsize=11)
            ax.set_title(f"Slope Overall - {metric_title}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title} - Slope Overall Metrics", fontsize=14, fontweight='bold')
        io.save("detailed_slope_overall.png")