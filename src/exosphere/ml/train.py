"""Training script for ML molecule classifier (Phase 5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import StatisticsError

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from exosphere.ml.model import (
    MoleculeCNN,
    save_checkpoint,
)

# Default paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path("data_cache/ml")
DEFAULT_MODEL_DIR = Path("models")


class Trainer:
    """Training loop for MoleculeCNN."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(outputs.cpu())
            all_targets.append(batch_y.cpu())

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        # Compute metrics
        preds = torch.cat(all_preds, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        metrics = self._compute_metrics(preds, targets)

        return avg_loss, metrics

    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        """Compute per-molecule metrics."""
        from sklearn.metrics import precision_score, recall_score, roc_auc_score

        metrics = {}
        n_classes = targets.shape[1]

        for i in range(n_classes):
            if len(np.unique(targets[:, i])) < 2:
                # Skip if only one class present
                continue

            try:
                auc = roc_auc_score(targets[:, i], preds[:, i])
                metrics[f"auc_{i}"] = float(auc)
            except ValueError:
                metrics[f"auc_{i}"] = 0.0

            # Precision/Recall at 0.5 threshold
            preds_binary = (preds[:, i] > 0.5).astype(int)
            try:
                prec = precision_score(targets[:, i], preds_binary, zero_division=0)
                rec = recall_score(targets[:, i], preds_binary, zero_division=0)
                metrics[f"prec_{i}"] = float(prec)
                metrics[f"rec_{i}"] = float(rec)
            except ValueError:
                metrics[f"prec_{i}"] = 0.0
                metrics[f"rec_{i}"] = 0.0

        # Overall
        try:
            metrics["auc_macro"] = float(
                np.mean([v for k, v in metrics.items() if k.startswith("auc_")])
            )
        except (ValueError, StatisticsError):
            metrics["auc_macro"] = 0.0

        return metrics

    def train(
        self,
        epochs: int,
        patience: int = 10,
        min_delta: float = 1e-4,
        save_path: Path | None = None,
        model_version: str = "CNN-v1",
        dataset_hash: str = "",
        seed: int = 42,
    ) -> dict:
        """Train with early stopping."""
        history = {"train_loss": [], "val_loss": [], "metrics": []}

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["metrics"].append(metrics)

            print(
                f"Epoch {epoch + 1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"AUC: {metrics.get('auc_macro', 0):.4f}"
            )

            # Learning rate scheduling
            if hasattr(self, "scheduler"):
                self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if save_path:
                    self.save_checkpoint(
                        save_path, model_version="CNN-v1", dataset_hash="", seed=42, epoch=epoch
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

    def save_checkpoint(
        self, path: Path, model_version: str, dataset_hash: str, seed: int, epoch: int
    ) -> None:
        save_checkpoint(
            self.model,
            self.optimizer,
            0,  # epoch handled externally
            {"val_loss": self.best_val_loss},
            dataset_hash,
            "CNN-v1",
            42,
            path,
        )


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with stratified split by atmosphere config."""
    n_samples = len(X)
    indices = np.arange(len(X))

    # Simple random split for now (can be improved with stratification by atmosphere config)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    n_test = int(n_samples * 0.1)
    n_val = int(n_samples * 0.2)
    n_train = n_samples - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_dataset = TensorDataset(
        torch.from_numpy(X[train_idx]).float(), torch.from_numpy(y[train_idx]).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X[val_idx]).float(), torch.from_numpy(y[val_idx]).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X[test_idx]).float(), torch.from_numpy(y[test_idx]).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate on test set with detailed per-molecule metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu())
            all_targets.append(batch_y.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    # Per-molecule metrics
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}
    for i, mol in enumerate(["H2O", "CO2", "CO", "CH4", "SO2"]):
        # y_true = targets[:, i]  # unused
        # y_pred = preds[:, i]  # unused

        try:
            auc = roc_auc_score(targets[:, i], preds[:, i])
        except ValueError:
            auc = 0.0

        try:
            ap = average_precision_score(targets[:, i], preds[:, i])
        except ValueError:
            ap = 0.0

        try:
            prec = precision_score(targets[:, i], (preds[:, i] > 0.5).astype(int), zero_division=0)
            rec = recall_score(targets[:, i], (preds[:, i] > 0.5).astype(int), zero_division=0)
        except ValueError:
            prec = rec = 0.0

        acc = accuracy_score(targets[:, i], (preds[:, i] > 0.5).astype(int))

        # Expected Calibration Error (ECE) - 10 bins
        confidences = preds[:, i]
        bin_boundaries = np.linspace(0, 1, 11)
        ece = 0.0
        for j in range(10):
            mask = (confidences >= bin_boundaries[j]) & (confidences < bin_boundaries[j + 1])
            if np.any(mask):
                bin_acc = np.mean(targets[mask, i] == (preds[mask, i] > 0.5).astype(int))
                bin_conf = np.mean(confidences[mask])
                ece += np.sum(mask) / len(confidences) * abs(bin_acc - bin_conf)

        metrics[mol] = {
            "auc": float(auc),
            "ap": float(ap),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc),
            "ece": float(ece),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML molecule classifier")
    parser.add_argument("--n-samples", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--data-dir", type=str, default="data_cache/ml")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--generate-data", action="store_true", help="Generate new dataset")
    args = parser.parse_args()

    # Set seeds
    import random

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Paths
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Generate or load dataset
    if args.generate_data or not (Path(args.data_dir) / "dataset.npz").exists():
        print("Generating dataset...")
        from exosphere.ml.dataset import DatasetConfig, generate_dataset, save_dataset

        config = DatasetConfig(n_samples=args.n_samples, seed=42)
        X, y, params, metadata = generate_dataset(config)
        save_dataset(X, y, [], [], DatasetConfig(n_samples=args.n_samples), Path(args.data_dir))
    else:
        print("Loading existing dataset...")
        from exosphere.ml.dataset import load_dataset

        X, y, _ = load_dataset(Path(args.data_dir))

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class balance: {y.mean(axis=0)}")

    # Content hash of the dataset file (provenance for the checkpoint).
    import hashlib

    dataset_file = Path(args.data_dir) / "dataset.npz"
    dataset_hash = ""
    if dataset_file.exists():
        digest = hashlib.sha256()
        with open(dataset_file, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        dataset_hash = digest.hexdigest()
    print(f"Dataset sha256: {dataset_hash[:16] if dataset_hash else 'unknown'}...")

    # Create model
    model = MoleculeCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(X, y, batch_size=args.batch_size)

    # Train
    trainer = Trainer(model, train_loader, val_loader, device, lr=args.lr)
    history = trainer.train(
        epochs=args.epochs,
        patience=10,
        save_path=Path(args.model_dir) / "best_model.pt",
        model_version="CNN-v1",
        dataset_hash=dataset_hash,
        seed=42,
    )

    # Save final model
    model_path = Path(args.model_dir) / "final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_channels": 2,
                "n_classes": 5,
                "base_channels": 32,
                "n_conv_blocks": 4,
            },
            "molecule_order": ["H2O", "CO2", "CO", "CH4", "SO2"],
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # Save training history
    history_path = Path(args.model_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"History saved to {history_path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    main()
