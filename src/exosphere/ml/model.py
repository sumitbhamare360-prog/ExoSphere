"""1D CNN multi-label classifier for molecule detection (Phase 5).

Architecture: 2 input channels (depth, uncertainty) → 4 conv blocks →
global avg pool → dense → 5 sigmoid outputs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from exosphere.ml.dataset import N_MOLECULES


class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + MaxPool block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class MoleculeCNN(nn.Module):
    """1D CNN for multi-label molecule classification.

    Input: (batch, 2, n_wavelength) - channels: [depth, uncertainty]
    Output: (batch, 5) - sigmoid scores for [H2O, CO2, CO, CH4, SO2]
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_classes: int = N_MOLECULES,
        base_channels: int = 32,
        n_conv_blocks: int = 4,
        dropout_conv: float = 0.1,
        dropout_fc: float = 0.3,
        fc_hidden: int = 128,
    ):
        super().__init__()

        self.n_classes = n_classes

        # Convolutional blocks
        channels = [n_channels] + [base_channels * (2**i) for i in range(n_conv_blocks)]
        self.conv_blocks = nn.ModuleList()
        for i in range(n_conv_blocks):
            self.conv_blocks.append(
                ConvBlock(
                    channels[i], channels[i + 1], kernel_size=3, pool_size=2, dropout=dropout_conv
                )
            )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        fc_in = channels[-1]
        self.fc1 = nn.Linear(fc_in, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, n_classes)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2, n_wavelength)
        for block in self.conv_blocks:
            x = block(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.view(x.size(0), -1)  # (batch, channels)

        # Fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract feature maps from each conv block for visualization."""
        feature_maps = []
        for block in self.conv_blocks:
            x = block(x)
            feature_maps.append(x)
        return feature_maps


def create_model(config: dict[str, Any] | None = None) -> MoleculeCNN:
    """Create model from configuration dictionary."""
    if config is None:
        config = {}

    model = MoleculeCNN(
        n_channels=config.get("n_channels", 2),
        n_classes=config.get("n_classes", 5),
        base_channels=config.get("base_channels", 32),
        n_conv_blocks=config.get("n_conv_blocks", 4),
        dropout_conv=config.get("dropout_conv", 0.1),
        dropout_fc=config.get("dropout_fc", 0.3),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    dataset_hash: str,
    model_version: str,
    seed: int,
    path: str | Path,
) -> None:
    """Save model checkpoint with full provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
            "dataset_hash": dataset_hash,
            "model_version": model_version,
            "seed": seed,
            "model_config": {
                "n_channels": 2,
                "n_classes": 5,
                "base_channels": 32,
                "n_conv_blocks": 4,
            },
            "molecule_order": ["H2O", "CO2", "CO", "CH4", "SO2"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        path,
    )


def load_checkpoint(
    path: str | Path, model: nn.Module | None = None, optimizer: torch.optim.Optimizer | None = None
) -> dict:
    """Load checkpoint, return checkpoint dict."""
    checkpoint = torch.load(path, map_location="cpu")

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device the model is on."""
    return next(model.parameters()).device


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Quick test
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

    model = create_model()
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 147)  # batch=4, channels=2, wavelength=147
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.4f}, {y.max():.4f}]")
    print("Model test passed!")
