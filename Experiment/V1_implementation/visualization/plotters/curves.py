import numpy as np
import matplotlib.pyplot as plt

from ..history import History
from ..collection import HistoryCollection
from ..io import PlotIO

class CurvesPlotter:
    SINGLE_KEYS = (
        ("train_loss", "val_loss"),
        ("train_loss_next_visit", "val_loss_next_visit"),
        ("train_loss_slope", "val_loss_slope"),
    )

    def plot_single(self, h: History, io: PlotIO, title: str = "Training curves") -> None:
        best = h.best_epoch()

        plt.figure(figsize=(10, 6))
        plotted = False

        for trk, vak in self.SINGLE_KEYS:
            tr = h.series(trk)
            va = h.series(vak)
            if tr is not None:
                plt.plot(np.arange(1, len(tr) + 1), tr, label=trk)
                plotted = True
            if va is not None:
                plt.plot(np.arange(1, len(va) + 1), va, label=vak)
                plotted = True

        if plotted:
            # best marker
            va_pref = h.series("val_loss_next_visit")
            if va_pref is None:
                va_pref = h.series("val_loss")
            if va_pref is not None and 0 <= best < len(va_pref):
                plt.scatter([best + 1], [float(va_pref[best])], s=60, marker="x", label=f"best_epoch={best+1}")

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(title)
            plt.legend()
            io.save("training_curves.png")
        else:
            plt.close()

        # LR separate
        lr = h.series("learning_rate")
        if lr is not None:
            plt.figure(figsize=(10, 3.5))
            plt.plot(np.arange(1, len(lr) + 1), lr, label="learning_rate")
            plt.xlabel("Epoch")
            plt.ylabel("LR")
            plt.title(title + " - Learning rate")
            plt.legend()
            io.save("learning_rate.png")

    def plot_cv(self, hc: HistoryCollection, io: PlotIO, title: str = "CV") -> None:
        def plot_band(key: str, filename: str):
            X = hc.stack_series(key)
            if X is None:
                return
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            epochs = np.arange(1, len(mean) + 1)

            plt.figure(figsize=(9, 4.5))
            plt.plot(epochs, mean, label=f"{key} (mean)")
            plt.fill_between(epochs, mean - std, mean + std, alpha=0.2, label="±1 std")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{title} - {key}")
            plt.legend()
            io.save(filename)

        plot_band("val_loss", "cv_val_loss.png")
        plot_band("val_loss_next_visit", "cv_val_loss_next_visit.png")
        plot_band("val_loss_slope", "cv_val_loss_slope.png")
