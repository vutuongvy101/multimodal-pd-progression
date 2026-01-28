import numpy as np
import matplotlib.pyplot as plt

from ..config import VizConfig
from ..history import History
from ..collection import HistoryCollection
from ..io import PlotIO

class DeltaTBucketsPlotter:
    def plot_single(self, h: History, io: PlotIO, cfg: VizConfig, title: str = "Δt bucket MAE") -> None:
        best = h.best_epoch()
        vme = h.val_metrics_epoch(best)
        if not vme:
            return

        target = h.pick_target(cfg.target)
        if target is None:
            return

        buckets = vme.get("next_visit", {}).get(target, {}).get("delta_t_buckets")
        if not isinstance(buckets, dict):
            return

        labels, maes = [], []
        for b in cfg.dt_buckets:
            v = buckets.get(b, {}).get("mae")
            if v is None:
                continue
            labels.append(b)
            maes.append(float(v))

        if not maes:
            return

        plt.figure(figsize=(8, 4))
        x = np.arange(len(maes))
        plt.bar(x, maes)
        plt.xticks(x, labels)
        plt.ylabel("MAE")
        plt.title(f"{title} @ best epoch ({target})")
        io.save("dt_bucket_mae.png")

    def plot_cv(self, hc: HistoryCollection, io: PlotIO, cfg: VizConfig, title: str = "CV") -> None:
        rows = []
        for h in hc.histories:
            best = h.best_epoch()
            vme = h.val_metrics_epoch(best)
            if not vme:
                continue
            target = h.pick_target(cfg.target)
            if target is None:
                continue
            buckets = vme.get("next_visit", {}).get(target, {}).get("delta_t_buckets")
            if not isinstance(buckets, dict):
                continue

            row = []
            ok = True
            for b in cfg.dt_buckets:
                v = buckets.get(b, {}).get("mae")
                if v is None:
                    ok = False
                    break
                row.append(float(v))
            if ok:
                rows.append(row)

        if not rows:
            return

        X = np.asarray(rows, dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        plt.figure(figsize=(8.5, 4))
        x = np.arange(len(cfg.dt_buckets))
        plt.bar(x, mean, yerr=std, capsize=5)
        plt.xticks(x, cfg.dt_buckets)
        plt.ylabel("MAE")
        plt.title(f"{title} - Δt bucket MAE @ best epoch (mean±std)")
        io.save("cv_dt_bucket_mae.png")
