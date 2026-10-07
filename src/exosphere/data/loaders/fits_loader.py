"""x1d-style FITS table -> Spectrum loader.

V1 only ingests calibrated/extracted 1D spectra: a FITS binary table that
already contains wavelength, transit depth and uncertainty columns. A raw x1d
*flux* spectrum (WAVELENGTH/FLUX/ERROR with no depth column) cannot be
converted to a transit depth here and is rejected with a clear error instead of
having flux silently treated as depth.

Column units are read from the FITS header when present; the depth unit
defaults to "fraction" (pass "percent" or "ppm" if the file says otherwise).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.data.loaders.units import depth_to_fraction, wavelength_to_um

_WAVELENGTH_CANDIDATES = ("WAVELENGTH", "WAVE", "LAMBDA", "LAM")
_DEPTH_CANDIDATES = (
    "DEPTH",
    "TDEPTH",
    "TRANSIT_DEPTH",
    "PL_TRANDEP",
    "TRANSMIT",
    "TRANSMISSION",
)
_UNCERTAINTY_CANDIDATES = (
    "DEPTH_ERR",
    "DEPTH_ERROR",
    "TDEPTH_ERR",
    "TRANSIT_DEPTH_ERR",
    "DEPTH_UNC",
    "UNCERTAINTY",
    "ERR",
    "ERROR",
)
_BIN_EDGES_CANDIDATES = ("WAVELENGTH_BIN_EDGES", "BIN_EDGES", "EDGES")


def _find_table_hdu(hdul) -> fits.BinTableHDU:
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU) and hdu.columns and hdu.columns.names:
            return hdu
    raise ValueError("FITS file contains no binary table extension with columns")


def _resolve_column(hdu, requested: str | None, candidates, kind: str) -> str:
    names = [str(name) for name in hdu.columns.names]
    if requested is not None:
        for name in names:
            if name.lower() == requested.lower():
                return name
        raise ValueError(f"{kind} column {requested!r} not found; available columns are {names}")
    for candidate in candidates:
        for name in names:
            if name.lower() == candidate.lower():
                return name
    if kind == "transit depth":
        raise ValueError(
            "no transit-depth column found "
            f"(looked for {list(candidates)}); available columns are {names}. "
            "A flux-only x1d spectrum cannot be converted to a transit depth."
        )
    raise ValueError(
        f"{kind} column not found (looked for {list(candidates)}); available columns are {names}"
    )


def load_x1d_fits(
    path: str | Path,
    *,
    provenance: Provenance,
    observation_id: str,
    target_id: str,
    instrument: str,
    wavelength_column: str | None = None,
    depth_column: str | None = None,
    uncertainty_column: str | None = None,
    bin_edges_column: str | None = None,
    wavelength_unit: str | None = None,
    depth_unit: str = "fraction",
    uncertainty_unit: str | None = None,
) -> Spectrum:
    """Load an x1d-style FITS transmission table into a Spectrum."""
    file_path = Path(path)
    with fits.open(file_path) as hdul:
        hdu = _find_table_hdu(hdul)
        names = [str(name) for name in hdu.columns.names]

        wavelength_name = _resolve_column(
            hdu, wavelength_column, _WAVELENGTH_CANDIDATES, "wavelength"
        )
        depth_name = _resolve_column(hdu, depth_column, _DEPTH_CANDIDATES, "transit depth")
        uncertainty_name = _resolve_column(
            hdu, uncertainty_column, _UNCERTAINTY_CANDIDATES, "uncertainty"
        )

        wavelength_raw = hdu.data[wavelength_name]
        depth_raw = hdu.data[depth_name]
        uncertainty_raw = hdu.data[uncertainty_name]

        column_unit = hdu.columns[wavelength_name].unit
        effective_wavelength_unit = wavelength_unit or column_unit
        if not effective_wavelength_unit:
            raise ValueError(
                "wavelength unit is missing from the FITS header; pass wavelength_unit"
            )

        edges_name = None
        if bin_edges_column is not None:
            edges_name = _resolve_column(hdu, bin_edges_column, _BIN_EDGES_CANDIDATES, "bin edges")
        else:
            for candidate in _BIN_EDGES_CANDIDATES:
                for name in names:
                    if name.lower() == candidate.lower():
                        edges_name = name
        edges_raw = hdu.data[edges_name] if edges_name is not None else None

    wavelength = wavelength_to_um(_as_float(wavelength_raw), effective_wavelength_unit)
    transmission = depth_to_fraction(_as_float(depth_raw), depth_unit)
    uncertainty = depth_to_fraction(_as_float(uncertainty_raw), uncertainty_unit or depth_unit)

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(transmission)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )
    if not bool(np.any(valid)):
        raise ValueError(f"{file_path.name}: no usable transmission data points")
    if not bool(np.all(valid)):
        # drop rows with missing/invalid values rather than imputing them
        wavelength = wavelength[valid]
        transmission = transmission[valid]
        uncertainty = uncertainty[valid]
        if edges_raw is not None:
            edges_raw = None  # edges would no longer align; rebuild from centers

    if edges_raw is not None and len(edges_raw) == len(wavelength) + 1:
        bin_edges = [
            float(value)
            for value in wavelength_to_um(_as_float(edges_raw), effective_wavelength_unit)
        ]
    else:
        bin_edges = bin_edges_from_centers(wavelength)

    return Spectrum(
        wavelength=[float(value) for value in wavelength],
        transmission=[float(value) for value in transmission],
        uncertainty=[float(value) for value in uncertainty],
        wavelength_bin_edges=bin_edges,
        quality_flags=["OK"] * int(len(wavelength)),
        observation_id=observation_id,
        target_id=target_id,
        instrument=instrument,
        provenance=provenance,
    )


def _as_float(column) -> np.ndarray:
    return np.ma.asarray(column).astype(float).filled(np.nan)
