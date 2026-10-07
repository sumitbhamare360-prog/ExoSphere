"""Spectrum internal data contract (AGENTS.md section 4).

Units are explicit and fixed for the whole platform:
- wavelength / wavelength_bin_edges: micrometres (um)
- transmission / uncertainty: fractional transit depth ((Rp/Rs)^2), NOT ppm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from exosphere.core.provenance import Provenance

WAVELENGTH_UNIT = "um"
TRANSMISSION_UNIT = "fractional transit depth ((Rp/Rs)^2)"
UNCERTAINTY_UNIT = TRANSMISSION_UNIT

UNITS: dict[str, str] = {
    "wavelength": WAVELENGTH_UNIT,
    "transmission": TRANSMISSION_UNIT,
    "uncertainty": UNCERTAINTY_UNIT,
    "wavelength_bin_edges": WAVELENGTH_UNIT,
}


def _as_float_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _as_str_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return value


class Spectrum(BaseModel):
    """Calibrated/extracted 1D transit transmission spectrum."""

    wavelength: list[float] = Field(description=f"Wavelength grid in {WAVELENGTH_UNIT}, ascending.")
    transmission: list[float] = Field(description=f"Transit depth in {TRANSMISSION_UNIT}.")
    uncertainty: list[float] = Field(
        description=f"1-sigma uncertainty on transit depth in {UNCERTAINTY_UNIT}; > 0."
    )
    wavelength_bin_edges: list[float] = Field(
        description=f"Bin edges in {WAVELENGTH_UNIT}; length = len(wavelength) + 1."
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="Per-point quality flags; empty or length = len(wavelength).",
    )
    observation_id: str
    target_id: str
    instrument: str
    provenance: Provenance

    @model_validator(mode="before")
    @classmethod
    def _coerce_array_like(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key in (
            "wavelength",
            "transmission",
            "uncertainty",
            "wavelength_bin_edges",
        ):
            if key in data:
                data[key] = _as_float_list(data[key])
        if "quality_flags" in data:
            data["quality_flags"] = _as_str_list(data["quality_flags"])
        return data

    @model_validator(mode="after")
    def _validate_contract(self) -> Spectrum:
        n_points = len(self.wavelength)
        if len(self.transmission) != n_points or len(self.uncertainty) != n_points:
            raise ValueError(
                "wavelength, transmission and uncertainty must have equal lengths "
                f"(got {n_points}, {len(self.transmission)}, {len(self.uncertainty)})"
            )
        if len(self.wavelength_bin_edges) != n_points + 1:
            raise ValueError(
                "wavelength_bin_edges must have length len(wavelength) + 1 "
                f"(got {len(self.wavelength_bin_edges)}, expected {n_points + 1})"
            )
        if self.quality_flags and len(self.quality_flags) != n_points:
            raise ValueError(
                "quality_flags must be empty or have length len(wavelength) "
                f"(got {len(self.quality_flags)}, expected {n_points})"
            )
        if any(np.isnan(w) for w in self.wavelength):
            raise ValueError("wavelength must not contain NaN")
        diffs = np.diff(np.asarray(self.wavelength, dtype=np.float64))
        if diffs.size and not bool(np.all(diffs > 0)):
            raise ValueError("wavelength must be sorted in strictly ascending order")
        if any(u <= 0 for u in self.uncertainty):
            raise ValueError("uncertainty must be > 0 at every point")
        return self

    @classmethod
    def units(cls) -> dict[str, str]:
        """Explicit units of every numeric field."""
        return dict(UNITS)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Spectrum:
        return cls.model_validate(data)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Spectrum:
        return cls.model_validate_json(payload)

    def save(self, path: str | Path) -> Path:
        """Save to a compressed .npz archive (exact float round-trip)."""
        file_path = Path(path)
        if file_path.suffix != ".npz":
            file_path = Path(str(file_path) + ".npz")
        arrays: dict[str, Any] = {
            "wavelength": np.asarray(self.wavelength, dtype=np.float64),
            "transmission": np.asarray(self.transmission, dtype=np.float64),
            "uncertainty": np.asarray(self.uncertainty, dtype=np.float64),
            "wavelength_bin_edges": np.asarray(self.wavelength_bin_edges, dtype=np.float64),
            "quality_flags": np.asarray(self.quality_flags, dtype=str),
            "observation_id": np.asarray(self.observation_id),
            "target_id": np.asarray(self.target_id),
            "instrument": np.asarray(self.instrument),
            "provenance_json": np.asarray(self.provenance.model_dump_json()),
        }
        np.savez_compressed(file_path, **arrays)
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> Spectrum:
        """Load a spectrum written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                wavelength=archive["wavelength"].tolist(),
                transmission=archive["transmission"].tolist(),
                uncertainty=archive["uncertainty"].tolist(),
                wavelength_bin_edges=archive["wavelength_bin_edges"].tolist(),
                quality_flags=[str(flag) for flag in archive["quality_flags"].tolist()],
                observation_id=str(archive["observation_id"].item()),
                target_id=str(archive["target_id"].item()),
                instrument=str(archive["instrument"].item()),
                provenance=Provenance.model_validate_json(str(archive["provenance_json"].item())),
            )
