"""
Run only the XGBoost baseline (flat features) using the same data
preparation pipeline as the V1 transformer. Saves metrics, raw tabular
arrays and trained booster models to the configured model save directory.

Usage: python training/run_xgb_only.py
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from training.config import get_default_config as get_default_config_v1
from data.data_integrator import DataIntegrator
from data.dataset import create_dataloaders

from models.xgboost_model import run_xgb_experiment, build_next_visit_tabular, build_slope_tabular


def parse_args():
    p = argparse.ArgumentParser(description="Run XGBoost baseline using V1 data pipeline")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-normalize", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = get_default_config_v1()
    integrator = DataIntegrator(cfg, normalize_features=not args.no_normalize)
    prepared = integrator.prepare_final_dataset()

    train_loader, val_loader, test_loader = create_dataloaders(
        prepared_data=prepared,
        config=cfg,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        random_seed=args.seed,
    )

    target_names = getattr(cfg.features, "all_updrs_totals", ["NP1RTOT", "NP2PTOT", "NP3TOT", "NP4TOT"]) 

    print("Running XGBoost baseline (flat features)...")
    results = run_xgb_experiment(
        train_dataset=train_loader.dataset,
        val_dataset=val_loader.dataset,
        test_dataset=test_loader.dataset,
        target_names=target_names,
    )

    out_dir = Path(cfg.data.model_save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics dicts
    np.savez(out_dir / "xgb_flat_metrics.npz",
             next_visit_metrics=results["next_visit_metrics"],
             slope_metrics=results["slope_metrics"])

    # Save raw tabular arrays for reproducible downstream analyses
    X_tr, y_tr, mask_tr, dt_tr = build_next_visit_tabular(train_loader.dataset)
    X_val, y_val, mask_val, dt_val = build_next_visit_tabular(val_loader.dataset)
    X_te, y_te, mask_te, dt_te = build_next_visit_tabular(test_loader.dataset)

    np.savez(out_dir / "xgb_flat_raw_tabular.npz",
             X_tr=X_tr, y_tr=y_tr, mask_tr=mask_tr, dt_tr=dt_tr,
             X_val=X_val, y_val=y_val, mask_val=mask_val, dt_val=dt_val,
             X_te=X_te, y_te=y_te, mask_te=mask_te, dt_te=dt_te)

    # Persist trained boosters (one file per target)
    for i, booster in enumerate(results.get("models_next_visit", [])):
        booster.save_model(str(out_dir / f"xgb_flat_next_visit_target{i}.json"))

    for i, booster in enumerate(results.get("models_slope", [])):
        booster.save_model(str(out_dir / f"xgb_flat_slope_target{i}.json"))

    print("Saved XGBoost metrics, raw tables and models to:", out_dir)


if __name__ == "__main__":
    main()
