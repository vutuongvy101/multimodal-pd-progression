import numpy as np
import matplotlib.pyplot as plt

from ..history import History
from ..io import PlotIO

class SlopeMetricsPlotter:
    def plot_single(self, h: History, io: PlotIO, title: str = "Slope metrics") -> None:
        n = h.n_epochs()
        if n == 0:
            return

        epochs, maes, rmses, spears = [], [], [], []

        for e in range(n):
            vme = h.val_metrics_epoch(e)
            if not vme:
                continue
            so = vme.get("slope_overall")
            if not isinstance(so, dict):
                continue

            epochs.append(e + 1)
            maes.append(so.get("mae", np.nan))
            rmses.append(so.get("rmse", np.nan))
            spears.append(so.get("spearman", np.nan))

        if not epochs:
            return

        maes = np.asarray(maes, dtype=float)
        rmses = np.asarray(rmses, dtype=float)
        spears = np.asarray(spears, dtype=float)

        plt.figure(figsize=(10, 4.5))
        if np.isfinite(maes).any():
            plt.plot(epochs, maes, label="slope_overall_mae")
        if np.isfinite(rmses).any():
            plt.plot(epochs, rmses, label="slope_overall_rmse")
        if np.isfinite(spears).any():
            plt.plot(epochs, spears, label="slope_overall_spearman")

        plt.xlabel("Epoch")
        plt.title(title)
        plt.legend()
        io.save("slope_metrics.png")
