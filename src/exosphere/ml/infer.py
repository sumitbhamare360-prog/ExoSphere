"""Inference module for ML molecule classifier (Phase 5).

predict(spectrum) -> MLResult with per-molecule scores labelled
"ML candidate score (not abundance, not a detection)".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exosphere.core.spectrum import Spectrum
from exosphere.ml.dataset import MOLECULES, load_target_grid
from exosphere.ml.model import MoleculeCNN

MODEL_VERSION = "CNN-v1"
TRAINING_GRID_WL = None  # Will be loaded lazily


@dataclass
class MLResult:
    """ML inference result with per-molecule candidate scores."""

    # Per-molecule scores (0-1)
    scores: dict[str, float]

    # Provenance
    model_version: str
    dataset_hash: str
    model_config_hash: str
    timestamp: str

    # Input spectrum info
    input_coverage_fraction: float
    input_wavelength_range: tuple[float, float]
    grid_match: bool
    warnings: list[str]

    def __post_init__(self):
        # Ensure scores are properly labelled
        self.scores = {k: float(v) for k, v in self.scores.items()}

    @property
    def labelled_scores(self) -> dict[str, str]:
        """Scores with required labelling."""
        return {
            k: f"{v:.4f} (ML candidate score, not abundance, not a detection)"
            for k, v in self.scores.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return {
            "scores": self.scores,
            "labelled_scores": self.labelled_scores,
            "model_version": self.model_version,
            "dataset_hash": self.dataset_hash,
            "model_config_hash": self.model_config_hash,
            "timestamp": self.timestamp,
            "input_coverage_fraction": self.input_coverage_fraction,
            "input_wavelength_range": self.input_wavelength_range,
            "grid_match": self.grid_match,
            "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary_text(self) -> str:
        lines = [
            f"ML Inference Result ({self.model_version})",
            f"Timestamp: {self.timestamp}",
            f"Dataset hash: {self.dataset_hash[:8]}...",
            f"Grid match: {self.grid_match}",
            f"Input coverage: {self.input_coverage_fraction:.2%}",
            f"Input range: {self.input_wavelength_range[0]:.3f}"
            f"-{self.input_wavelength_range[1]:.3f} µm",
            "",
            "Per-molecule ML candidate scores (NOT abundance, NOT a detection):",
        ]
        for mol, score in self.scores.items():
            lines.append(f"  {mol:4s}: {score:.4f}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


class MLClassifier:
    """ML molecule classifier for inference."""

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | str | None = None,
    ):
        """Load model from checkpoint."""
        self.device = (
            torch.device(device)
            if device
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        checkpoint = torch.load(model_path, map_location=self.device)

        # Recreate model from config
        model_config = checkpoint.get(
            "model_config",
            {"n_channels": 2, "n_classes": 5, "base_channels": 32, "n_conv_blocks": 4},
        )

        self.model = MoleculeCNN(
            n_channels=model_config.get("n_channels", 2),
            n_classes=model_config.get("n_classes", 5),
            base_channels=model_config.get("base_channels", 32),
            n_conv_blocks=model_config.get("n_conv_blocks", 4),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Store metadata
        self.model_version = checkpoint.get("model_version", "CNN-v1")
        self.dataset_hash = checkpoint.get("dataset_hash", "")
        self.model_config = checkpoint.get("model_config", {})
        self.checkpoint = checkpoint

        print(f"Loaded model {self.model_version} on {self.device}")

    @staticmethod
    def _get_training_grid() -> tuple[np.ndarray, np.ndarray]:
        """Get the training wavelength grid (WASP-39 b PRISM cleaned)."""
        global TRAINING_GRID_WL
        if TRAINING_GRID_WL is None:
            TRAINING_GRID_WL = load_target_grid()
        return TRAINING_GRID_WL

    def _interpolate_to_training_grid(
        self,
        spectrum: Spectrum,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Interpolate input spectrum to training grid.

        Returns:
            depth_interp: interpolated depth on training grid
            uncertainty_interp: interpolated uncertainty on training grid
            warnings: list of warnings
        """
        from scipy.interpolate import interp1d

        wl_train, edges_train = self._get_training_grid()
        wl_in = np.array(spectrum.wavelength)
        depth_in = np.array(spectrum.transmission)
        unc_in = np.array(spectrum.uncertainty)

        warnings_list = []

        # Check coverage
        input_min, input_max = wl_in.min(), wl_in.max()
        train_min, train_max = wl_train.min(), wl_train.max()

        coverage = max(0.0, min(input_max, train_max) - max(input_min, train_min)) / (
            train_max - train_min
        )

        if coverage < 0.9:
            warnings_list.append(
                f"Input spectrum covers only {coverage:.1%} of training grid "
                f"({input_min:.2f}-{input_max:.2f} vs {train_min:.2f}-{train_max:.2f} µm)"
            )

        # Check grid alignment
        if not np.allclose(wl_in, np.linspace(wl_in[0], wl_in[-1], len(wl_in)), rtol=1e-3):
            warnings_list.append("Input wavelength grid is not uniformly spaced")

        # Interpolate
        f_depth = interp1d(
            wl_in, depth_in, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        f_unc = interp1d(wl_in, unc_in, kind="linear", bounds_error=False, fill_value="extrapolate")

        depth_interp = f_depth(wl_train)
        unc_interp = f_unc(wl_train)

        return depth_interp, unc_interp, warnings_list

    def predict(self, spectrum: Spectrum) -> MLResult:
        """Run inference on a spectrum.

        Returns MLResult with per-molecule scores.
        """
        # Interpolate to training grid
        depth_interp, unc_interp, warnings = self._interpolate_to_training_grid(spectrum)

        # Prepare input tensor: (1, 2, n_wavelength)
        depth_tensor = torch.from_numpy(depth_interp.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        unc_tensor = torch.from_numpy(unc_interp.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        x = torch.cat([depth_tensor, unc_tensor], dim=1).to(self.device)  # (1, 2, n_wl)

        # Inference
        with torch.no_grad():
            outputs = self.model(x)
            scores = outputs.cpu().numpy().squeeze()  # (5,)

        # Build result
        scores_dict = {mol: float(score) for mol, score in zip(MOLECULES, scores, strict=False)}

        wl_train, _ = self._get_training_grid()

        result = MLResult(
            scores=scores_dict,
            model_version=self.model_version,
            dataset_hash=self.checkpoint.get("dataset_hash", ""),
            model_config_hash=hash(str(self.checkpoint.get("model_config", {}))),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            input_coverage_fraction=1.0,
            input_wavelength_range=(
                float(np.min(spectrum.wavelength)),
                float(np.max(spectrum.wavelength)),
            ),
            grid_match=len(warnings) == 0,
            warnings=warnings,
        )

        return result

    def predict_batch(self, spectra: list[Spectrum]) -> list[MLResult]:
        """Run inference on multiple spectra."""
        return [self.predict(spec) for spec in spectra]


def predict_from_checkpoint(
    checkpoint_path: str | Path,
    spectrum: Spectrum,
    device: torch.device | str | None = None,
) -> MLResult:
    """Convenience function for single inference."""
    classifier = MLClassifier(checkpoint_path, device)
    return classifier.predict(spectrum)


def predict_from_checkpoint_batch(
    checkpoint_path: str | Path,
    spectra: list[Spectrum],
    device: torch.device | str | None = None,
) -> list[MLResult]:
    """Batch inference from checkpoint."""
    classifier = MLClassifier(checkpoint_path, device)
    return classifier.predict_batch(spectra)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run ML inference on a spectrum")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--spectrum", type=str, required=True, help="Input spectrum .npz path")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load spectrum
    spectrum = Spectrum.load(args.spectrum)

    # Run inference
    result = predict_from_checkpoint(args.checkpoint, spectrum, device)

    # Print
    print(result.summary_text())

    # Save output
    if args.output:
        Path(args.output).write_text(result.to_json())
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    main()
