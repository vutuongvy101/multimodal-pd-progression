import json
from pathlib import Path


class DummyKFoldTrainer:
    """
    Stand-in for KFoldTrainer to avoid real training.
    Creates fold directories and writes a kfold_results.json in the configured save dir.
    """

    def __init__(
        self,
        config,
        prepared_data,
        n_splits=5,
        test_ratio=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
        modalities=None,
        modality_key=None,
    ):
        self.config = config
        self.save_dir = Path(config.data.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_splits = n_splits
        self.modalities = modalities
        self.modality_key = modality_key

    def train(self, save_fold_checkpoints: bool = True):
        # Create fold dirs and minimal checkpoint markers
        for i in range(1, self.n_splits + 1):
            fold_dir = self.save_dir / f"fold_{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "best_checkpoint.pt").write_bytes(b"best")

        summary = {
            "mean_val_loss": 0.5,
            "std_val_loss": 0.1,
            "mean_val_next_visit": 0.4,
            "std_val_next_visit": 0.1,
            "mean_val_slope": 0.6,
            "std_val_slope": 0.2,
            "mean_best_val_loss": 0.45,
            "std_best_val_loss": 0.08,
        }

        fold_results = [{"fold": i, "val_loss": 0.5, "best_val_loss": 0.45} for i in range(1, self.n_splits + 1)]

        # Write kfold_results.json where MultiModalTrainer expects it
        results_path = self.save_dir / "kfold_results.json"
        results_data = {
            "n_splits": self.n_splits,
            "fold_results": fold_results,
            "summary": summary,
            "status": "completed"
        }
        if self.modalities is not None:
            results_data["modalities"] = self.modalities
        if self.modality_key is not None:
            results_data["modality_key"] = self.modality_key
        
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

        return {"fold_results": fold_results, "summary": summary, "test_loader": None}


def test_multi_modal_trainer_creates_isolated_dirs_and_skips(tmp_path, test_config, monkeypatch):
    """
    Ensure MultiModalTrainer:
    - Creates separate directories per modality combination
    - Writes completion markers
    - Skips already-trained configurations
    """
    import training.multi_modal_trainer as mm_mod
    from training.multi_modal_trainer import MultiModalTrainer

    # Patch KFoldTrainer used inside MultiModalTrainer
    monkeypatch.setattr(mm_mod, "KFoldTrainer", DummyKFoldTrainer)

    test_config.data.model_save_dir = str(tmp_path / "checkpoints")

    prepared_data = {"static": None, "longitudinal": None, "slopes": None}
    trainer = MultiModalTrainer(
        config=test_config,
        prepared_data=prepared_data,
        base_save_dir=test_config.data.model_save_dir,
        n_splits=2,
        test_ratio=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
    )

    # Train two different configs
    results = trainer.train_multiple(["static_only", "motor_only"], force_retrain=False)

    assert "static" in results  # key is sorted modality string, e.g. "static"
    assert "motor" in results

    # Each config has isolated directory + results file
    static_dir = Path(test_config.data.model_save_dir) / "modalities_static"
    motor_dir = Path(test_config.data.model_save_dir) / "modalities_motor"
    assert static_dir.exists()
    assert motor_dir.exists()
    assert (static_dir / "kfold_results.json").exists()
    assert (motor_dir / "kfold_results.json").exists()

    # Re-running should skip
    result2 = trainer.train_modality_config(["static"], force_retrain=False)
    assert result2["status"] == "skipped"
