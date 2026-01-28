import json, glob
import numpy as np
import matplotlib.pyplot as plt

# 1) load histories
paths = sorted(glob.glob("../models/checkpoints/modalities_static/fold_*/training_history.json"))
assert len(paths) > 0, "No training_history.json found. Check your path/pattern."
histories = []
for p in paths:
    with open(p, "r") as f:
        histories.append(json.load(f))

# helper: stack a per-epoch series across folds
def stack_series(key):
    arrs = [np.array(h[key], dtype=float) for h in histories if key in h]
    # pad/truncate if folds have different epoch lengths (simple safe handling)
    min_len = min(map(len, arrs))
    arrs = [a[:min_len] for a in arrs]
    return np.stack(arrs, axis=0)  # (n_folds, n_epochs)

def plot_mean_std(key, title=None, ylabel=None):
    X = stack_series(key)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    epochs = np.arange(1, len(mean)+1)

    plt.figure(figsize=(7,4))
    plt.plot(epochs, mean, label=f"{key} (mean)")
    plt.fill_between(epochs, mean-std, mean+std, alpha=0.2, label="±1 std")
    plt.title(title or key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel or key)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 2) plots you can do immediately
plot_mean_std("train_loss", title="Train loss (mean±std across folds)")
plot_mean_std("val_loss",   title="Val loss (mean±std across folds)")

plot_mean_std("train_loss_next_visit", title="Train loss - next visit")
plot_mean_std("val_loss_next_visit",   title="Val loss - next visit")

plot_mean_std("train_loss_slope", title="Train loss - slope")
plot_mean_std("val_loss_slope",   title="Val loss - slope")

plot_mean_std("learning_rate", title="Learning rate schedule", ylabel="LR")

# 3) Δt bucket MAE at BEST epoch (per fold), then aggregate
TARGET = "NP3TOT"  # change to your target
BUCKETS = ["dt_0_6", "dt_6_12", "dt_12_24", "dt_24_inf"]

def best_epoch_idx(h):
    # choose best by val_loss_next_visit if present, else val_loss
    key = "val_loss_next_visit" if "val_loss_next_visit" in h else "val_loss"
    return int(np.argmin(np.array(h[key], dtype=float)))

bucket_mae = []  # shape (n_folds, n_buckets)
for h in histories:
    e = best_epoch_idx(h)
    vm = h["val_metrics"][e]["next_visit"][TARGET]["delta_t_buckets"]
    bucket_mae.append([vm[b]["mae"] for b in BUCKETS])

bucket_mae = np.array(bucket_mae, dtype=float)
mean = bucket_mae.mean(axis=0)
std  = bucket_mae.std(axis=0)

plt.figure(figsize=(7,4))
x = np.arange(len(BUCKETS))
plt.bar(x, mean, yerr=std, capsize=5)
plt.xticks(x, BUCKETS)
plt.ylabel("MAE")
plt.title(f"Next-visit MAE by Δt bucket @ best epoch (mean±std across folds) - {TARGET}")
plt.tight_layout()
plt.show()
