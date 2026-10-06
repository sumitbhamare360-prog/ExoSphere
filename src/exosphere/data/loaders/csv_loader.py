"""Local CSV -> Spectrum loader with explicit units.

The file carries no identity metadata, so observation_id, target_id and
instrument must be passed by the caller. Numeric columns are converted from the
declared source units to the internal convention (um, fractional depth).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.data.loaders.units import depth_to_fraction, wavelength_to_um


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    for candidate in frame.columns:
        if str(candidate).strip().lower() == name.lower():
            return frame[candidate]
    raise ValueError(
        f"column {name!r} not found in CSV; available columns are "
        f"{[str(column) for column in frame.columns]}"
    )


def _optional_column(frame: pd.DataFrame, name: str | None) -> pd.Series | None:
    if name is None:
        return None
    return _column(frame, name)


def load_csv(
    path: str | Path,
    *,
    provenance: Provenance,
    observation_id: str,
    target_id: str,
    instrument: str,
    wavelength_column: str = "wavelength",
    depth_column: str = "transmission",
    uncertainty_column: str = "uncertainty",
    bin_edges_column: str | None = None,
    quality_flags_column: str | None = None,
    wavelength_unit: str = "um",
    depth_unit: str = "fraction",
    uncertainty_unit: str | None = None,
) -> Spectrum:
    """Load a CSV of wavelength / transit depth / uncertainty into a Spectrum."""
    frame = pd.read_csv(path, comment="#")
    frame.columns = [str(column).strip() for column in frame.columns]

    wavelength = wavelength_to_um(
        _column(frame, wavelength_column).to_numpy(dtype=float), wavelength_unit
    )
    source_depth_unit = uncertainty_unit or depth_unit
    transmission = depth_to_fraction(
        _column(frame, depth_column).to_numpy(dtype=float), depth_unit
    )
    uncertainty = depth_to_fraction(
        _column(frame, uncertainty_column).to_numpy(dtype=float), source_depth_unit
    )

    edges_series = _optional_column(frame, bin_edges_column)
    if edges_series is not None:
        edges = wavelength_to_um(edges_series.to_numpy(dtype=float), wavelength_unit)
        bin_edges = [float(value) for value in edges]
    else:
        bin_edges = bin_edges_from_centers(wavelength)

    flags_series = _optional_column(frame, quality_flags_column)
    if flags_series is not None:
        quality_flags = [str(value) for value in flags_series.tolist()]
    else:
        quality_flags = ["OK"] * int(len(wavelength))

    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(transmission)):
        raise ValueError(f"{Path(path).name}: non-finite wavelength or transit depth")
    if not np.all(np.isfinite(uncertainty)) or not np.all(uncertainty > 0):
        raise ValueError(f"{Path(path).name}: uncertainty must be finite and > 0")

    return Spectrum(
        wavelength=[float(value) for value in wavelength],
        transmission=[float(value) for value in transmission],
        uncertainty=[float(value) for value in uncertainty],
        wavelength_bin_edges=bin_edges,
        quality_flags=quality_flags,
        observation_id=observation_id,
        target_id=target_id,
        instrument=instrument,
        provenance=provenance,
    )
