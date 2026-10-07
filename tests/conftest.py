"""Shared fixtures and synthetic-spectrum helpers for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.quality.assess import load_quality_config

# Source values (microns / percent) taken from the NASA Exoplanet Archive
# download of the WASP-39 b NIRSpec PRISM transmission spectrum
# (spec_path 40/24/96/78/WASP_39_b_3.11466_5502_6.tbl, Carter et al. 2024).
PRISM_SOURCE_ROWS = [
    # central_wavelength_um, bandwidth_um, depth_percent, err1_percent, err2_percent, limit
    ("0.52132", "0.00850", "2.08997", "0.02571", "-0.02551", "0"),
    ("0.53004", "0.00893", "2.11727", "0.01765", "-0.01748", "0"),
    ("0.53919", "0.00938", "2.14191", "0.01276", "-0.01266", "0"),
    ("0.54883", "0.00989", "2.15292", "0.01013", "-0.01015", "0"),
]

PRISM_RATIO_ROWS = [
    # central_wavelength_um, bandwidth_um, Rp/R*, err1, err2
    ("0.52132", "0.00850", "0.14457", "0.00089", "-0.00088"),
    ("0.53004", "0.00893", "0.14551", "0.00061", "-0.00060"),
]


def make_ipac_tbl(
    rows=PRISM_SOURCE_ROWS,
    *,
    include_depth: bool = True,
    include_ratio: bool = False,
    null_depth_rows: tuple[int, ...] = (),
) -> str:
    """Build a small IPAC .tbl in the exact layout of the archive downloads."""
    lines = [
        "\\PL_NAME = WASP-39 b",
        "\\SPEC_TYPE = Transmission",
        "\\INSTRUMENT = Near Infrared Spectrograph (NIRSpec)",
        "\\FACILITY = NASA 6.5m James Webb Space Telescope (JWST) Satellite Mission",
        "\\NOTE = PRISM, Native resolution",
        "\\REFERENCE = Carter et al. 2024",
        "\\",
    ]
    names = ["CENTRALWAVELNG", "BANDWIDTH"]
    types = ["double", "double"]
    units = ["microns", "microns"]
    if include_depth:
        names += ["PL_TRANDEP", "PL_TRANDEPERR1", "PL_TRANDEPERR2", "PL_TRANDEPLIM"]
        types += ["double", "double", "double", "long"]
        units += ["%", "%", "%", ""]
    if include_ratio:
        names += ["PL_RATROR", "PL_RATRORERR1", "PL_RATRORERR2"]
        types += ["double", "double", "double"]
        units += ["", "", ""]
    widths = [len(name) for name in names]
    lines.append("|" + "|".join(names) + "|")
    lines.append("|" + "|".join(f"{t:>{w}s}" for t, w in zip(types, widths, strict=False)) + "|")
    lines.append("|" + "|".join(f"{u:>{w}s}" for u, w in zip(units, widths, strict=False)) + "|")
    lines.append("|" + "|".join("null".rjust(w) for w in widths) + "|")

    for index, row in enumerate(rows):
        cells = [row[0], row[1]]
        if include_depth:
            if index in null_depth_rows:
                cells += ["null", "null", "null", "0"]
            else:
                cells += [row[2], row[3], row[4], row[5]]
        if include_ratio:
            ratio_row = PRISM_RATIO_ROWS[index % len(PRISM_RATIO_ROWS)]
            cells += [ratio_row[2], ratio_row[3], ratio_row[4]]
        # data rows align with the header: single spaces where the header has '|'
        cells = [cell.rjust(w) for cell, w in zip(cells, widths, strict=False)]
        lines.append(" " + " ".join(cells))
    return "\n".join(lines) + "\n"


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(analysis_id="EXO-000001", planet="WASP-39 b")


@pytest.fixture
def ipac_tbl_path(tmp_path):
    path = tmp_path / "WASP_39_b_prism.tbl"
    path.write_text(make_ipac_tbl(), encoding="utf-8")
    return path


def make_synthetic_spectrum(
    wavelength,
    transmission,
    uncertainty,
    *,
    provenance: Provenance | None = None,
    quality_flags=None,
    observation_id: str = "synthetic-001",
    target_id: str = "SYNTH b",
    instrument: str = "NIRSpec PRISM",
    bin_width: float | None = None,
) -> Spectrum:
    """Build a Spectrum on a regular grid (bin edges from bin centers).

    ``bin_width`` is required only for a single-point spectrum, where no
    neighbor spacing exists to infer the bin size from.
    """
    wave = np.asarray(wavelength, dtype=np.float64)
    if wave.size == 1:
        if bin_width is None:
            raise ValueError("bin_width is required for a single-point spectrum")
        edges = [float(wave[0] - bin_width / 2), float(wave[0] + bin_width / 2)]
    else:
        edges = bin_edges_from_centers(wave)
    return Spectrum(
        wavelength=[float(v) for v in wave],
        transmission=[float(v) for v in transmission],
        uncertainty=[float(v) for v in uncertainty],
        wavelength_bin_edges=edges,
        quality_flags=list(quality_flags) if quality_flags is not None else [],
        observation_id=observation_id,
        target_id=target_id,
        instrument=instrument,
        provenance=provenance or Provenance(analysis_id="EXO-000001", planet="SYNTH b"),
    )


def make_feature_spectrum(
    lo: float = 0.6,
    hi: float = 5.3,
    n_points: int = 400,
    noise_sigma: float = 2e-5,
    baseline: float = 0.015,
    amplitude: float = 5e-4,
    seed: int = 7,
    molecules: tuple[str, ...] = ("H2O", "CO2", "CO", "CH4", "SO2"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic transmission spectrum with smooth molecular band bumps.

    Every selected molecule window gets a Gaussian bump centred in the window
    with width = window/4 and the given amplitude (default = 25 x noise), so
    the band S/N (90th percentile of deviation/sigma) is well above threshold.
    Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    wave = np.linspace(lo, hi, n_points)
    depth = np.full(n_points, baseline, dtype=np.float64)
    config = load_quality_config()
    for molecule in molecules:
        for window_lo, window_hi in config.molecule_bands[molecule]:
            center = (window_lo + window_hi) / 2.0
            width = max((window_hi - window_lo) / 4.0, 0.02)
            bump = amplitude * np.exp(-0.5 * ((wave - center) / width) ** 2)
            depth = depth + bump
    depth = depth + rng.normal(0.0, noise_sigma, n_points)
    sigma = np.full(n_points, noise_sigma, dtype=np.float64)
    return wave, depth, sigma
