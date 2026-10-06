"""Explicit unit conversions into the internal convention.

Internal convention (AGENTS.md section 4):
- wavelength: micrometres (um)
- transit depth and uncertainty: fractional transit depth ((Rp/Rs)^2)
"""

from __future__ import annotations

import astropy.units as u
import numpy as np

WAVELENGTH_ALIASES: dict[str, str] = {
    "um": "um",
    "micron": "um",
    "microns": "um",
    "micrometer": "um",
    "micrometers": "um",
    "nm": "nm",
    "nanometer": "nm",
    "nanometers": "nm",
    "angstrom": "Angstrom",
    "angstroms": "Angstrom",
    "a": "Angstrom",
    "m": "m",
    "meter": "m",
    "meters": "m",
}

DEPTH_FACTORS: dict[str, float] = {
    "fraction": 1.0,
    "frac": 1.0,
    "percent": 0.01,
    "percentage": 0.01,
    "pct": 0.01,
    "%": 0.01,
    "ppm": 1e-6,
    "parts_per_million": 1e-6,
}


def wavelength_to_um(values, unit: str) -> np.ndarray:
    """Convert a wavelength array to micrometres using an astropy-compatible unit."""
    key = str(unit).strip().lower()
    canonical = WAVELENGTH_ALIASES.get(key, str(unit).strip())
    array = np.asarray(values, dtype=float)
    try:
        result = (array * u.Unit(canonical)).to(u.um)
    except (ValueError, u.UnitsError) as exc:
        raise ValueError(
            f"unsupported wavelength unit {unit!r}; "
            f"use one of {sorted(set(WAVELENGTH_ALIASES))}"
        ) from exc
    return result.value


def depth_to_fraction(values, unit: str) -> np.ndarray:
    """Convert transit depth or its uncertainty to a fraction.

    fraction -> factor 1, percent -> 0.01, ppm -> 1e-6.
    """
    key = str(unit).strip().lower()
    if key not in DEPTH_FACTORS:
        raise ValueError(
            f"unsupported depth unit {unit!r}; "
            f"use one of {sorted(set(DEPTH_FACTORS))}"
        )
    return np.asarray(values, dtype=float) * DEPTH_FACTORS[key]
