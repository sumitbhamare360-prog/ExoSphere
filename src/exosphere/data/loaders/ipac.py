"""NASA Exoplanet Archive spectrum (.tbl, IPAC format) -> Spectrum loader.

Source units (verified from the archive's Data Column Definitions page and from
the downloaded files themselves):
- CENTRALWAVELNG, BANDWIDTH: microns -> already um, no conversion.
- PL_TRANDEP and its +/- errors: percent -> divided by 100 to a fraction.
- Fallback when PL_TRANDEP is null: depth = PL_RATROR**2 (Rp/R* is dimensionless)
  with uncertainty propagated as 2*|Rp/R*|*sigma(Rp/R*).

Rows with a null wavelength, no usable depth, or no usable uncertainty are
dropped (never imputed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import ascii

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.data.loaders.units import depth_to_fraction, wavelength_to_um

_DEFAULT_WAVELENGTH_UNIT = "um"
_DEFAULT_DEPTH_UNIT = "percent"


def _keyword(meta: dict, name: str) -> str | None:
    entry = meta.get("keywords", {}).get(name)
    value = entry.get("value") if isinstance(entry, dict) else entry
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _column(table, name: str):
    for column in table.colnames:
        if column.upper() == name.upper():
            return table[column]
    return None


def _array(column) -> np.ndarray:
    return np.ma.asarray(column).astype(float)


def _is_masked(column, index: int) -> bool:
    return bool(np.ma.getmaskarray(column)[index])


def load_archive_tbl(
    path: str | Path,
    *,
    provenance: Provenance,
    observation_id: str | None = None,
    target_id: str | None = None,
    instrument: str | None = None,
) -> Spectrum:
    """Load a downloaded Exoplanet Archive spectrum file into a Spectrum."""
    file_path = Path(path)
    table = ascii.read(str(file_path), format="ipac")
    meta = table.meta

    wavelength_col = _column(table, "CENTRALWAVELNG")
    if wavelength_col is None:
        raise ValueError(
            f"{file_path.name}: no CENTRALWAVELNG column; columns are {table.colnames}"
        )
    width_col = _column(table, "BANDWIDTH")
    depth_col = _column(table, "PL_TRANDEP")
    err1_col = _column(table, "PL_TRANDEPERR1")
    err2_col = _column(table, "PL_TRANDEPERR2")
    lim_col = _column(table, "PL_TRANDEPLIM")
    ratio_col = _column(table, "PL_RATROR")
    ratio_err1_col = _column(table, "PL_RATRORERR1")
    ratio_err2_col = _column(table, "PL_RATRORERR2")

    if depth_col is None and ratio_col is None:
        raise ValueError(
            f"{file_path.name}: neither PL_TRANDEP nor PL_RATROR column present; "
            f"columns are {table.colnames}"
        )

    wavelength_unit = _DEFAULT_WAVELENGTH_UNIT
    if wavelength_col.unit:
        wavelength_unit = str(wavelength_col.unit)
    wavelength = wavelength_to_um(_array(wavelength_col), wavelength_unit)
    widths = _array(width_col) if width_col is not None else None
    wavelength_mask = np.ma.getmaskarray(wavelength_col)

    depth_scale = 1.0
    if depth_col is not None:
        depth_unit = str(depth_col.unit) if depth_col.unit else _DEFAULT_DEPTH_UNIT
        depth_scale = float(depth_to_fraction(np.array([1.0]), depth_unit)[0])

    depth_arr = _array(depth_col) if depth_col is not None else None
    err1_arr = _array(err1_col) if err1_col is not None else None
    err2_arr = _array(err2_col) if err2_col is not None else None
    lim_arr = _array(lim_col) if lim_col is not None else None
    ratio_arr = _array(ratio_col) if ratio_col is not None else None
    ratio_err1_arr = _array(ratio_err1_col) if ratio_err1_col is not None else None
    ratio_err2_arr = _array(ratio_err2_col) if ratio_err2_col is not None else None

    keep: list[int] = []
    transmissions: list[float] = []
    uncertainties: list[float] = []
    flags: list[str] = []

    for index in range(len(table)):
        if wavelength_mask[index] or not np.isfinite(wavelength[index]):
            continue

        depth_value: float | None = None
        uncertainty_value: float | None = None

        if depth_arr is not None and not _is_masked(depth_col, index):
            depth_value = float(depth_arr[index]) * depth_scale
            if (
                err1_col is not None
                and err2_col is not None
                and not _is_masked(err1_col, index)
                and not _is_masked(err2_col, index)
            ):
                upper = abs(float(err1_arr[index]))
                lower = abs(float(err2_arr[index]))
                uncertainty_value = (upper + lower) / 2.0 * depth_scale
        elif (
            ratio_arr is not None
            and ratio_err1_arr is not None
            and ratio_err2_arr is not None
            and not _is_masked(ratio_col, index)
            and not _is_masked(ratio_err1_col, index)
            and not _is_masked(ratio_err2_col, index)
        ):
            ratio = float(ratio_arr[index])
            sigma_ratio = (
                abs(float(ratio_err1_arr[index])) + abs(float(ratio_err2_arr[index]))
            ) / 2.0
            depth_value = ratio**2
            uncertainty_value = 2.0 * abs(ratio) * sigma_ratio

        if depth_value is None or uncertainty_value is None:
            continue
        if not (np.isfinite(depth_value) and np.isfinite(uncertainty_value)):
            continue
        if uncertainty_value <= 0:
            continue

        keep.append(index)
        transmissions.append(depth_value)
        uncertainties.append(uncertainty_value)
        upper_limit = bool(
            lim_arr is not None
            and not _is_masked(lim_col, index)
            and lim_arr[index] == 1
        )
        flags.append("UPPER_LIMIT" if upper_limit else "OK")

    if not keep:
        raise ValueError(f"{file_path.name}: no usable transmission data points")

    kept_wavelength = wavelength[keep]
    kept_widths = widths[keep] if widths is not None else None
    edges = bin_edges_from_centers(kept_wavelength, kept_widths)

    resolved_target = target_id or _keyword(meta, "PL_NAME")
    resolved_instrument = instrument or _keyword(meta, "INSTRUMENT")
    if not resolved_target:
        raise ValueError(
            f"{file_path.name}: PL_NAME header keyword missing; pass target_id"
        )
    if not resolved_instrument:
        raise ValueError(
            f"{file_path.name}: INSTRUMENT header keyword missing; pass instrument"
        )

    return Spectrum(
        wavelength=[float(value) for value in kept_wavelength],
        transmission=transmissions,
        uncertainty=uncertainties,
        wavelength_bin_edges=edges,
        quality_flags=flags,
        observation_id=observation_id or file_path.name,
        target_id=resolved_target,
        instrument=resolved_instrument,
        provenance=provenance,
    )
