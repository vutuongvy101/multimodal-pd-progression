import json
from pathlib import Path

import torch


class _TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"x": torch.tensor([1.0])}


class DummyTrainer:
    """
    Stand-in for V1Trainer to avoid real training.
    Writes fold-specific checkpoints into config.data.model_save_dir.
    """

    def __init__(self, model, config, train_loader, val_loader, device="cpu"):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.current_epoch = 3
        self.best_val_loss = 0.1234
        self.training_history = {"train_loss": [1.0, 0.5, 0.25]}

        self.save_dir = Path(config.data.model_save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self, max_epochs: int, early_stopping_patience: int = 15):
        # no-op
        return

    def validate(self):
        return {"loss": 0.2, "loss_next_visit": 0.15, "loss_slope": 0.05}

    def save_checkpoint(self, is_best: bool = False):
        # Create the same filenames as real trainer
        (self.save_dir / "latest_checkpoint.pt").write_bytes(b"latest")
        if is_best:
            (self.save_dir / "best_checkpoint.pt").write_bytes(b"best")


def test_kfold_trainer_creates_fold_dirs_and_results(tmp_path, test_config, monkeypatch):
    """
    Ensure KFoldTrainer:
    - Creates fold-specific directories
    - Does not overwrite fold checkpoints (each fold has its own folder)
    - Writes kfold_results.json
    """
    from training.kfold_trainer import KFoldTrainer
    import training.kfold_trainer as kfold_mod

    # Patch trainer to dummy (fast, deterministic)
    monkeypatch.setattr(kfold_mod, "V1Trainer", DummyTrainer)

    # Make save dir temporary
    test_config.data.model_save_dir = str(tmp_path / "checkpoints")

    # Patch fold dataloaders to avoid sklearn and data pipeline
    tiny_loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=1)
    fold_dataloaders = [(tiny_loader, tiny_loader), (tiny_loader, tiny_loader)]
    test_loader = tiny_loader

    def _fake_create(self):
        return fold_dataloaders, test_loader

    monkeypatch.setattr(KFoldTrainer, "_create_fold_dataloaders", _fake_create)

    trainer = KFoldTrainer(
        config=test_config,
        prepared_data={"static": None, "longitudinal": None, "slopes": None},
        n_splits=2,
        test_ratio=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
    )

    results = trainer.train(save_fold_checkpoints=True)

    # Check kfold_results.json exists
    results_path = Path(test_config.data.model_save_dir) / "kfold_results.json"
    assert results_path.exists()
    saved = json.loads(results_path.read_text())
    assert saved["n_splits"] == 2
    assert len(saved["fold_results"]) == 2

    # Each fold has its own checkpoint folder and files
    fold1 = Path(test_config.data.model_save_dir) / "fold_1"
    fold2 = Path(test_config.data.model_save_dir) / "fold_2"
    assert fold1.exists()
    assert fold2.exists()
    assert (fold1 / "latest_checkpoint.pt").exists()
    assert (fold1 / "best_checkpoint.pt").exists()
    assert (fold2 / "latest_checkpoint.pt").exists()
    assert (fold2 / "best_checkpoint.pt").exists()
