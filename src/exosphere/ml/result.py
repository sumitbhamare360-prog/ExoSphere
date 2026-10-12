"""Dependency-light result type for optional ML inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class MLResult:
    """ML candidate scores and the metadata needed to interpret them safely."""

    scores: dict[str, float]
    model_version: str
    dataset_hash: str
    model_config_hash: str
    timestamp: str
    input_coverage_fraction: float
    input_wavelength_range: tuple[float, float]
    grid_match: bool
    warnings: list[str]

    def __post_init__(self) -> None:
        self.scores = {key: float(value) for key, value in self.scores.items()}

    @property
    def labelled_scores(self) -> dict[str, str]:
        return {
            key: f"{value:.4f} (ML candidate score, not abundance, not a detection)"
            for key, value in self.scores.items()
        }

    def to_dict(self) -> dict[str, Any]:
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
        """Human-readable, explicitly non-detection-oriented score summary."""
        lines = [
            f"ML Inference Result ({self.model_version})",
            "Per-molecule ML candidate scores (NOT abundance, NOT a detection):",
        ]
        lines.extend(f"  {molecule:4s}: {score:.4f}" for molecule, score in self.scores.items())
        return "\n".join(lines)
