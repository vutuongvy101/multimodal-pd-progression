"""
CLI entrypoint to train the V1 multimodal longitudinal transformer.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.data_integrator import DataIntegrator
from data.dataset import create_dataloaders
from models.v1_model import V1MultimodalTransformer
from training.config import get_default_config as get_default_config_v1
from training.config_v2 import get_default_config as get_default_config_v2
from training.metrics import compute_comprehensive_metrics
from training.train import V1Trainer
from models.xgboost_model import run_xgb_experiment as run_xgb_flat_experiment
from models.xgboost_temporal_model import run_xgb_experiment as run_xgb_temporal_experiment
from models.xgboost_model import (
    build_next_visit_tabular as build_next_visit_tabular_flat,
)
from models.xgboost_temporal_model import (
    build_next_visit_tabular as build_next_visit_tabular_temporal,
)
from pathlib import Path
from training.kfold_trainer import KFoldTrainer
from training.multi_modal_trainer import MultiModalTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V1 multimodal transformer")
    parser.add_argument(
        "--config-version",
        choices=["v1", "v2"],
        default="v1",
        help="Which config module to use (v1=training/config.py, v2=training/config_v2.py)",
    )
    parser.add_argument(
        "--mode",
        choices=["kfold", "single", "multi_modal"],
        default="kfold",
        help="Training mode: 'kfold' (default, k-fold CV), 'single' (single train/val/test split), "
             "or 'multi_modal' (train multiple models with different modality combinations)",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=None,
        help="For --mode multi_modal: List of modality specifications to train. "
             "Can use predefined sets (all, static+motor, etc.) or custom (static,motor,nonmotor). "
             "Example: --modalities all static+motor motor_only",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if model checkpoint already exists (for multi_modal mode)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for k-fold CV (default: 5, only used with --mode kfold)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2, only used with --mode kfold)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (only used with --mode single)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (only used with --mode single)",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable feature normalization inside DataIntegrator",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load for fine-tuning or resuming (only used with --mode single)",
    )
    parser.add_argument(
        "--only-xgb",
        action="store_true",
        help="Run only the XGBoost baseline on the V1 data splits and exit (single mode only)",
    )
    parser.add_argument(
        "--xgb-temporal",
        action="store_true",
        help="When used with --only-xgb, run the temporal (lagged) XGBoost instead of flat features",
    )
    parser.add_argument(
        "--xgb-outdir",
        type=str,
        default=None,
        help="Optional output directory to save XGBoost metrics/models (defaults to config.data.model_save_dir)",
    )
    parser.add_argument(
        "--resume-mode",
        choices=["model-only", "full"],
        default="model-only",
        help="How to use the checkpoint: "
             "'model-only' loads weights and starts a fresh optimizer; "
             "'full' resumes optimizer/scheduler/epoch state (only used with --mode single)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze embeddings + transformer and only fine-tune prediction heads (only used with --mode single)",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate on test set after training (for kfold mode, uses best fold model)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(version: str):
    if version == "v2":
        return get_default_config_v2()
    return get_default_config_v1()


def apply_overrides(config, args: argparse.Namespace):
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.patience is not None:
        config.training.early_stopping_patience = args.patience
    if args.device is not None:
        config.training.device = args.device


@torch.no_grad()
def evaluate_with_metrics(
        model: V1MultimodalTransformer,
        loader,
        lambda_slope: float,
        device: str,
        target_names,
) -> Dict[str, Dict]:
    """
    Evaluate model on a loader and compute both losses and rich metrics.
    """
    model.eval()
    epoch_losses = {"loss": 0.0, "loss_next_visit": 0.0, "loss_slope": 0.0}
    n_batches = len(loader)

    # For metrics, accumulate predictions/targets across batches
    all_predictions = {"next_visit": [], "slope": []}
    all_targets = {"next_visit": [], "next_visit_mask": [], "slope": []}
    all_attention_masks = []
    all_time_months = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        predictions = model(
            batch["static_values"],
            batch["static_mask"],
            batch["motor_values"],
            batch["motor_mask"],
            batch["nonmotor_values"],
            batch["nonmotor_mask"],
            batch["med_values"],
            batch["med_mask"],
            batch["age_at_visit_values"],
            batch["age_at_visit_mask"],
            batch["time_months"],
            batch["attention_mask"],
        )
        targets = {
            "next_visit": batch["next_visit_targets"],
            "next_visit_mask": batch.get("next_visit_label_mask"),
            "slope": batch["slope_targets"],
        }
        losses = model.compute_loss(predictions, targets, batch["attention_mask"], lambda_slope)
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()

        all_predictions["next_visit"].append(predictions["next_visit"].detach().cpu())
        all_predictions["slope"].append(predictions["slope"].detach().cpu())
        all_targets["next_visit"].append(targets["next_visit"].detach().cpu())
        if targets["next_visit_mask"] is not None:
            all_targets["next_visit_mask"].append(targets["next_visit_mask"].detach().cpu())
        all_targets["slope"].append(targets["slope"].detach().cpu())
        all_attention_masks.append(batch["attention_mask"].detach().cpu())
        all_time_months.append(batch["time_months"].detach().cpu())

    for key in epoch_losses:
        epoch_losses[key] /= max(n_batches, 1)

    predictions_cat = {
        "next_visit": torch.cat(all_predictions["next_visit"], dim=0),
        "slope": torch.cat(all_predictions["slope"], dim=0),
    }
    targets_cat = {
        "next_visit": torch.cat(all_targets["next_visit"], dim=0),
        "slope": torch.cat(all_targets["slope"], dim=0),
        "next_visit_mask": torch.cat(all_targets["next_visit_mask"], dim=0)
        if all_targets["next_visit_mask"]
        else None,
    }
    attention_mask_cat = torch.cat(all_attention_masks, dim=0)
    time_months_cat = torch.cat(all_time_months, dim=0)

    metrics = compute_comprehensive_metrics(
        predictions_cat,
        targets_cat,
        attention_mask_cat,
        target_names=target_names,
        time_months=time_months_cat,
    )

    return {"losses": epoch_losses, "metrics": metrics}


def train_single_split(config, prepared, args):
    """Train with single train/val/test split (legacy mode)"""
    device = args.device or config.training.device

    train_loader, val_loader, test_loader = create_dataloaders(
        prepared_data=prepared,
        config=config,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        random_seed=args.seed,
    )

    # If user requested to run only the XGBoost baseline, do so and exit.
    if getattr(args, 'only_xgb', False):
        out_dir = Path(args.xgb_outdir) if args.xgb_outdir else Path(config.data.model_save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_names = getattr(config.features, "all_updrs_totals", ["NP1RTOT", "NP2PTOT", "NP3TOT", "NP4TOT"])

        if getattr(args, 'xgb_temporal', False):
            print("Running XGBoost temporal baseline (lagged features)...")
            results = run_xgb_temporal_experiment(
                train_dataset=train_loader.dataset,
                val_dataset=val_loader.dataset,
                test_dataset=test_loader.dataset,
                target_names=target_names,
            )
            prefix = "xgb_temporal"
        else:
            print("Running XGBoost baseline (flat features)...")
            results = run_xgb_flat_experiment(
                train_dataset=train_loader.dataset,
                val_dataset=val_loader.dataset,
                test_dataset=test_loader.dataset,
                target_names=target_names,
            )
            prefix = "xgb_flat"

        # Save metrics and raw tabular arrays
        np.savez(out_dir / f"{prefix}_metrics.npz",
                 next_visit_metrics=results["next_visit_metrics"],
                 slope_metrics=results["slope_metrics"])

        # Choose appropriate builder for raw tabular arrays
        build_fn = build_next_visit_tabular_temporal if prefix == "xgb_temporal" else build_next_visit_tabular_flat
        X_tr, y_tr, mask_tr, dt_tr = build_fn(train_loader.dataset)
        X_val, y_val, mask_val, dt_val = build_fn(val_loader.dataset)
        X_te, y_te, mask_te, dt_te = build_fn(test_loader.dataset)

        np.savez(out_dir / f"{prefix}_raw_tabular.npz",
             X_tr=X_tr, y_tr=y_tr, mask_tr=mask_tr, dt_tr=dt_tr,
             X_val=X_val, y_val=y_val, mask_val=mask_val, dt_val=dt_val,
             X_te=X_te, y_te=y_te, mask_te=mask_te, dt_te=dt_te)

        for i, booster in enumerate(results.get("models_next_visit", [])):
            booster.save_model(str(out_dir / f"{prefix}_next_visit_target{i}.json"))
        for i, booster in enumerate(results.get("models_slope", [])):
            booster.save_model(str(out_dir / f"{prefix}_slope_target{i}.json"))

        # Build a small summary row and append to CSV
        def _extract_summary(metrics_dict: dict):
            maelist = []
            rmselist = []
            n_total = 0
            for v in metrics_dict.get('next_visit', {}).values():
                if isinstance(v, dict):
                    if 'mae' in v:
                        maelist.append(v.get('mae', np.nan))
                    if 'rmse' in v:
                        rmselist.append(v.get('rmse', np.nan))
                    n_total += int(v.get('n_samples', 0) or 0)
            mean_mae = float(np.nanmean(maelist)) if maelist else float('nan')
            mean_rmse = float(np.nanmean(rmselist)) if rmselist else float('nan')

            accs = []
            f1s = []
            # per-target values
            for v in metrics_dict.get('next_visit', {}).values():
                if isinstance(v, dict):
                    if 'accuracy' in v:
                        accs.append(v.get('accuracy', np.nan))
                    if 'f1' in v:
                        f1s.append(v.get('f1', np.nan))

            mean_acc = float(np.nanmean(accs)) if accs else float('nan')
            mean_f1 = float(np.nanmean(f1s)) if f1s else float('nan')

            return mean_mae, mean_rmse, mean_acc, mean_f1, n_total

        mean_mae, mean_rmse, mean_acc, mean_f1, n_total = _extract_summary(results.get('next_visit_metrics', {}))

        import csv
        summary_path = out_dir / 'xgb_comparison_summary.csv'
        write_header = not summary_path.exists()
        with open(summary_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(['Model', 'Variant', 'Mean_MAE', 'Mean_RMSE', 'Mean_Accuracy', 'Mean_F1', 'Total_Samples'])
            writer.writerow([
                'xgboost', prefix,
                f"{mean_mae:.4f}" if not np.isnan(mean_mae) else '',
                f"{mean_rmse:.4f}" if not np.isnan(mean_rmse) else '',
                f"{mean_acc:.4f}" if not np.isnan(mean_acc) else '',
                f"{mean_f1:.4f}" if not np.isnan(mean_f1) else '',
                int(n_total)
            ])

        print(f"Saved XGBoost results to {out_dir}")
        return

    print("\nBuilding model...")
    model = V1MultimodalTransformer(config)

    trainer = V1Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Optional: load checkpoint for fine-tuning or resume
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\nLoading checkpoint from {checkpoint_path} ({args.resume_mode})...")
        if args.resume_mode == "full":
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print("  ✓ Loaded model weights (fresh optimizer/scheduler)")

    # Optional: freeze backbone for head-only fine-tuning
    if args.freeze_backbone:
        print("\nFreezing backbone parameters (embeddings + transformer + time encoding)...")
        for module in [
            getattr(model, "static_embedding", None),
            getattr(model, "motor_embedding", None),
            getattr(model, "nonmotor_embedding", None),
            getattr(model, "med_embedding", None),
            getattr(model, "age_at_visit_embedding", None),
            getattr(model, "visit_builder", None),
            getattr(model, "time_encoding", None),
            getattr(model, "transformer", None),
        ]:
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad = False

    print("\nStarting training...")
    trainer.train(
        max_epochs=config.training.max_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
    )

    print("\nEvaluating on test set...")
    target_names = getattr(config.features, "all_updrs_totals", ["NP1RTOT", "NP2PTOT", "NP3TOT", "NP4TOT"])
    eval_results = evaluate_with_metrics(
        model,
        test_loader,
        config.training.lambda_slope,
        device,
        target_names=target_names,
    )
    losses = eval_results["losses"]
    metrics = eval_results["metrics"]

    print(
        f"Test Loss: {losses['loss']:.4f} "
        f"(next: {losses['loss_next_visit']:.4f}, "
        f"slope: {losses['loss_slope']:.4f})"
    )
    print("\nPer-UPDRS metrics (test set):")
    for name, m in metrics["next_visit"].items():
        print(
            f"  {name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
            f"R2={m['r2']:.4f}, Pearson={m['correlation']:.4f}, "
            f"Spearman={m.get('spearman', float('nan')):.4f}, n={m['n_samples']}"
        )

        if m.get('delta_t_buckets'):
            print(f"    Per-Δt buckets:")
            for bucket_name, bucket_metrics in m['delta_t_buckets'].items():
                if bucket_metrics['n_samples'] > 0:
                    print(
                        f"      {bucket_name}: MAE={bucket_metrics['mae']:.4f}, "
                        f"RMSE={bucket_metrics['rmse']:.4f}, n={bucket_metrics['n_samples']}"
                    )

    if "next_visit_overall" in metrics:
        overall = metrics["next_visit_overall"]
        print(f"\nNext-visit overall: Macro MAE={overall.get('macro_avg_mae', float('nan')):.4f}, "
              f"Weighted MAE={overall.get('weighted_avg_mae', float('nan')):.4f}, "
              f"Total samples={overall.get('total_samples', 0)}")

    print("\nPer-target slope metrics:")
    for name, m in metrics.get("slope", {}).items():
        if m.get('n_samples', 0) > 0:
            print(
                f"  {name}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
                f"Spearman={m.get('spearman', float('nan')):.4f}, n={m['n_samples']}"
            )

    if "slope_overall" in metrics:
        slope_overall = metrics["slope_overall"]
        print(
            f"\nSlope overall: MAE={slope_overall['mae']:.4f}, "
            f"RMSE={slope_overall['rmse']:.4f}, "
            f"Spearman={slope_overall.get('spearman', float('nan')):.4f}, "
            f"n={slope_overall['n_samples']}"
        )


def train_kfold_cv(config, prepared, args):
    """Train with k-fold cross-validation (default mode)"""
    device = args.device or config.training.device

    print("\nInitializing k-fold cross-validation trainer...")
    kfold_trainer = KFoldTrainer(
        config=config,
        prepared_data=prepared,
        n_splits=args.n_splits,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        device=device,
        num_workers=args.num_workers,
    )

    print("\nStarting k-fold cross-validation...")
    results = kfold_trainer.train(save_fold_checkpoints=True)

    if args.evaluate_test:
        print("\nEvaluating on test set...")
        test_results = kfold_trainer.evaluate_test()
        losses = test_results["losses"]
        metrics = test_results["metrics"]
        print(
            f"Test Loss: {losses['loss']:.4f} "
            f"(next: {losses['loss_next_visit']:.4f}, "
            f"slope: {losses['loss_slope']:.4f})"
        )


def train_multi_modal(config, prepared, args):
    """Train multiple models with different modality combinations"""
    device = args.device or config.training.device

    if args.modalities is None:
        default_modalities = [
            'all',
            'static+motor',
            'static+nonmotor',
            'motor+nonmotor',
            'motor_only',
            'static_only'
        ]
        print(f"\nNo modalities specified. Using default set: {default_modalities}")
        modality_specs = default_modalities
    else:
        modality_specs = args.modalities

    print("\nInitializing multi-modal trainer...")
    multi_modal_trainer = MultiModalTrainer(
        config=config,
        prepared_data=prepared,
        base_save_dir=config.data.model_save_dir,
        n_splits=args.n_splits,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        device=device,
        num_workers=args.num_workers,
    )

    print("\nStarting multi-modal training...")
    results = multi_modal_trainer.train_multiple(
        modality_specs=modality_specs,
        force_retrain=args.force_retrain
    )

    comparison = multi_modal_trainer.compare_results()
    print("\n" + "=" * 80)
    print("MODALITY COMPARISON")
    print("=" * 80)
    if comparison['best_config']:
        print(f"\nBest configuration: {comparison['best_config']}")
        print(f"  Validation Loss: {comparison['best_val_loss']:.4f}")
        print("\nAll configurations (sorted by val loss):")
        for cfg in comparison['configurations']:
            print(f"  {cfg['modality_key']}: {cfg['mean_val_loss']:.4f} ± {cfg['std_val_loss']:.4f}")
    print("=" * 80)

    if args.evaluate_test:
        print("\nEvaluating on test set...")
        test_results = multi_modal_trainer.evaluate_test_all()
        if test_results:
            print("\nTest evaluation summary:")
            for modality_key, result in test_results.items():
                if 'error' in result:
                    print(f"  {modality_key}: Error - {result['error']}")
                else:
                    losses = result.get('losses', {})
                    print(f"  {modality_key}: Test Loss = {losses.get('loss', 'N/A'):.4f}")


def main():
    args = parse_args()

    if args.mode == "multi_modal" and args.modalities is None:
        print("Note: --mode multi_modal specified but no --modalities given.")
        print("Will use default modality sets. Use --modalities to specify custom sets.")

    set_seed(args.seed)

    print("=" * 80)
    print(f"Loading config ({args.config_version})")
    print(f"Training mode: {args.mode}")
    print("=" * 80)
    config = load_config(args.config_version)
    apply_overrides(config, args)

    device = args.device or config.training.device
    print(f"Using device: {device}")

    print("\nPreparing data...")
    integrator = DataIntegrator(config, normalize_features=not args.no_normalize)
    prepared = integrator.prepare_final_dataset()

    if args.mode == "multi_modal":
        train_multi_modal(config, prepared, args)
    elif args.mode == "kfold":
        train_kfold_cv(config, prepared, args)
    else:
        train_single_split(config, prepared, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
