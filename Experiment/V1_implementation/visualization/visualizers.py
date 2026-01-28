import os
import numpy as np
import matplotlib.pyplot as plt

from .config import VizConfig
from .io import PlotIO
from .history import History
from .collection import HistoryCollection
from .plotters import CurvesPlotter, DeltaTBucketsPlotter, SlopeMetricsPlotter, DetailedMetricsPlotter


class SingleRunVisualizer:
    def __init__(self, cfg: VizConfig):
        self.cfg = cfg
        self.curves = CurvesPlotter()
        self.dt = DeltaTBucketsPlotter()
        self.slope = SlopeMetricsPlotter()
        self.detailed = DetailedMetricsPlotter()  # Add detailed plotter

    def run(self, history_path: str, output_dir: str, show: bool = True, title: str = "run") -> None:
        h = History.load(history_path)
        io = PlotIO(output_dir=output_dir, show=show, dpi=self.cfg.dpi)

        self.curves.plot_single(h, io, title=f"{title} - training")
        self.dt.plot_single(h, io, self.cfg, title=f"{title} - next_visit")
        self.slope.plot_single(h, io, title=f"{title} - slope")
        self.detailed.plot_single(h, io, self.cfg, title=f"{title} - detailed")  # Add detailed visualization


class KFoldVisualizer:
    def __init__(self, cfg: VizConfig):
        self.cfg = cfg
        self.curves = CurvesPlotter()
        self.dt = DeltaTBucketsPlotter()

    def run(self, base_dir: str, pattern: str, output_dir: str, show: bool = True, title: str = "kfold") -> None:
        glob_pattern = os.path.join(base_dir, pattern)
        hc = HistoryCollection.from_glob(glob_pattern)
        io = PlotIO(output_dir=output_dir, show=show, dpi=self.cfg.dpi)

        self.curves.plot_cv(hc, io, title=title)
        self.dt.plot_cv(hc, io, self.cfg, title=title)


class CompareVisualizer:
    def __init__(self, cfg: VizConfig):
        self.cfg = cfg

    def run(self, path1: str, path2: str, output_dir: str, show: bool = True,
            name1: str = "Model1", name2: str = "Model2") -> None:
        h1 = History.load(path1)
        h2 = History.load(path2)
        io = PlotIO(output_dir=output_dir, show=show, dpi=self.cfg.dpi)

        def overlay(key: str, filename: str):
            s1 = h1.series(key)
            s2 = h2.series(key)
            if s1 is None or s2 is None:
                return
            L = min(len(s1), len(s2))
            epochs = np.arange(1, L + 1)

            plt.figure(figsize=(9, 4.5))
            plt.plot(epochs, s1[:L], label=f"{name1}:{key}")
            plt.plot(epochs, s2[:L], label=f"{name2}:{key}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{key} comparison")
            plt.legend()
            io.save(filename)

        overlay("val_loss", "compare_val_loss.png")
        overlay("val_loss_next_visit", "compare_val_loss_next_visit.png")
        overlay("val_loss_slope", "compare_val_loss_slope.png")
