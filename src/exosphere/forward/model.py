"""Mock forward model for ExoSphere (Phase 3).

This is a pure-Python stand-in for petitRADTRANS, used because real pRT
cannot be installed on Windows + Python 3.13 + numpy 2.x (see DECISIONS.md
item 20). The API matches the Phase 3 specification exactly.

When real pRT is available, replace this module with a pRT-backed
implementation keeping the same public API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from exosphere.core.spectrum import Spectrum

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

MOCK_VERSION = "mock-1.0.0"
OPACITY_MODE = "mock-gaussian-R1000"
R = 1000  # Mock spectral resolution (lambda / Delta lambda)
WAVE_MIN = 0.3  # um
WAVE_MAX = 30.0  # um

# Reference pressure for radius definition
REFERENCE_PRESSURE_BAR = 0.01  # 10 mbar

# Solar He/H2 ratio (by number)
HE_H2_RATIO = 0.157  # He/H2 ~ 0.157

# Molecular weights (g/mol)
MOL_WEIGHT = {
    "H2": 2.016,
    "He": 4.003,
    "H2O": 18.015,
    "CO2": 44.01,
    "CO": 28.01,
    "CH4": 16.04,
    "SO2": 64.06,
}

# Line centers for mock Gaussian opacities (um) - approximate band centers
LINE_CENTERS = {
    "H2O": [0.94, 1.13, 1.38, 1.87, 2.66],  # near-IR bands
    "CO2": [2.0, 2.7, 4.3],
    "CO": [2.3, 4.7],
    "CH4": [1.66, 2.3, 3.3],
    "SO2": [4.0, 7.7],  # 7.7 um outside PRISM range
}

# Line strengths (cm^2/g) - peak cross-sections at line centers
# Real values: H2O ~ 1-10 cm^2/g; CO2 4.3 um ~ 100 cm^2/g
# Mock scaled so tau crosses 1 across multiple pressure layers
LINE_STRENGTHS = {
    "H2O": [0.5, 0.4, 0.6, 0.8, 0.5],
    "CO2": [0.2, 0.1, 5.0],
    "CO": [0.3, 1.0],
    "CH4": [0.4, 0.3, 0.5],
    "SO2": [0.5, 0.2],
}

# Line widths (um) - approximate FWHM at R=1000
LINE_WIDTHS = {
    "H2O": [0.02, 0.02, 0.03, 0.04, 0.05],
    "CO2": [0.02, 0.03, 0.05],
    "CO": [0.02, 0.05],
    "CH4": [0.03, 0.03, 0.05],
    "SO2": [0.04, 0.08],
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelParams:
    """Free parameters for the forward model (AGENTS.md §2, row 'Free params')."""

    T: float  # K, isothermal temperature
    log_h2o: float  # log10(VMR)
    log_co2: float  # log10(VMR)
    log_co: float  # log10(VMR)
    log_ch4: float  # log10(VMR)
    log_so2: float  # log10(VMR)
    r_ref: float  # planet radius at reference pressure, in R_Jup
    log_p_cloud: float  # cloud-top pressure in bar (log10)

    def __post_init__(self) -> None:
        """Validate parameters on construction."""
        if self.T <= 0:
            raise ValueError(f"Temperature must be > 0 K, got {self.T}")
        vmr_sum = sum(self.vmr_dict().values())
        if vmr_sum >= 1.0:
            raise ValueError(f"Sum of VMRs must be < 1, got {vmr_sum:.6f}")
        if self.r_ref <= 0:
            raise ValueError(f"Reference radius must be > 0 R_Jup, got {self.r_ref}")

    def vmr_dict(self) -> dict[str, float]:
        """Return VMR dictionary for the 5 gases."""
        return {
            "H2O": 10**self.log_h2o,
            "CO2": 10**self.log_co2,
            "CO": 10**self.log_co,
            "CH4": 10**self.log_ch4,
            "SO2": 10**self.log_so2,
        }

    def validate(self) -> None:
        """Validate parameters; raise ValueError if invalid (delegates to __post_init__)."""
        # Re-run validation (will raise if invalid)
        self.__post_init__()


@dataclass(frozen=True)
class PlanetFixed:
    """Fixed planetary parameters from catalog."""

    gravity_m_s2: float  # surface gravity in m/s^2
    stellar_radius_rsun: float  # stellar radius in R_sun
    reference_pressure_bar: float = REFERENCE_PRESSURE_BAR  # bar

    def __post_init__(self) -> None:
        """Validate parameters on construction."""
        if self.gravity_m_s2 <= 0:
            raise ValueError(f"Gravity must be > 0, got {self.gravity_m_s2}")
        if self.stellar_radius_rsun <= 0:
            raise ValueError(f"Stellar radius must be > 0, got {self.stellar_radius_rsun}")

    def validate(self) -> None:
        """Validate parameters (delegates to __post_init__)."""
        self.__post_init__()


# ---------------------------------------------------------------------------
# Mock opacity / cross-section computation
# ---------------------------------------------------------------------------


def _gaussian_line(wl: np.ndarray, center: float, strength: float, width: float) -> np.ndarray:
    """Gaussian line profile: exp(-(wl - center)^2 / (2 * sigma^2))."""
    # Convert FWHM to sigma: FWHM = 2*sqrt(2*ln2)*sigma
    sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return strength * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


def _mock_cross_sections(wl: np.ndarray, vmr: dict[str, float]) -> np.ndarray:
    """Compute total absorption cross-section (cm^2/g) on wavelength grid.

    This is a simplified mock: sum of Gaussian lines for each molecule,
    scaled by VMR. Units are arbitrary but internally consistent.
    """
    xsec = np.zeros_like(wl, dtype=np.float64)
    for mol, abund in vmr.items():
        if mol not in LINE_CENTERS:
            continue
        if abund <= 0:
            continue
        centers = LINE_CENTERS[mol]
        strengths = LINE_STRENGTHS[mol]
        widths = LINE_WIDTHS[mol]
        for c, s, w in zip(centers, strengths, widths, strict=True):
            xsec += abund * _gaussian_line(wl, c, s, w)
    return xsec


def _h2_he_cia(wl: np.ndarray) -> np.ndarray:
    """Mock H2-H2 and H2-He collision-induced absorption (CIA).

    Very rough approximation: increases toward shorter wavelengths.
    """
    # CIA roughly ~ lambda^-4 in near-IR; very weak in PRISM range
    return 1e-6 * (1.0 / wl) ** 4


def _rayleigh_scattering(wl: np.ndarray) -> np.ndarray:
    """Mock H2 Rayleigh scattering cross-section (cm^2/g).

    Roughly ~ lambda^-4.
    """
    return 1e-5 * (1.0 / wl) ** 4


# ---------------------------------------------------------------------------
# Atmosphere structure
# ---------------------------------------------------------------------------


def _build_atmosphere(
    params: ModelParams,
    fixed: PlanetFixed,
    n_layers: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build 1D atmosphere: pressure, temperature, altitude, mean molecular weight.

    Returns:
        pressure: pressure at layer centers (bar), descending (top to bottom)
        temperature: isothermal T (K)
        altitude: altitude above reference radius (m)
        mmw: mean molecular weight (g/mol)
    """
    # Pressure grid: log-spaced from 1e-6 to 1e2 bar
    p_top = 1e-6
    p_bottom = 1e2
    pressure = np.logspace(np.log10(p_top), np.log10(p_bottom), n_layers)
    # pressure[0] = top (1e-6 bar), pressure[-1] = bottom (100 bar)

    # Isothermal temperature
    temperature = np.full_like(pressure, params.T, dtype=np.float64)

    # Compute mean molecular weight from VMRs
    vmr = params.vmr_dict()
    vmr_sum = sum(vmr.values())
    vmr_h2 = 1.0 - vmr_sum - vmr_sum * HE_H2_RATIO
    vmr_he = vmr_sum * HE_H2_RATIO
    if vmr_h2 < 0:
        raise ValueError(f"VMR sum too large: {vmr_sum:.6f}, no room for H2/He")

    mmw = (
        vmr_h2 * MOL_WEIGHT["H2"]
        + vmr_he * MOL_WEIGHT["He"]
        + sum(vmr[m] * MOL_WEIGHT[m] for m in vmr)
    )

    # Hydrostatic equilibrium: dp/dz = -rho * g
    # rho = P * mmw / (R_specific * T), R_specific = R_univ / mmw
    # dp/dz = -P * g / (R_specific * T) = -P * g * mmw / (R_univ * T)
    # dz = -dP / (P * g * mmw / (R_univ * T))
    # Integrate from reference pressure (where r = r_ref) upward and downward
    R_univ = 8.314462618  # J/(mol*K)
    g = fixed.gravity_m_s2

    # Reference pressure index
    p_ref = fixed.reference_pressure_bar
    ref_idx = np.argmin(np.abs(pressure - p_ref))

    altitude = np.zeros_like(pressure)
    # Upward (lower pressure, indices decreasing from ref_idx-1 to 0)
    # pressure[i] < pressure[i+1], so dp = p[i+1] - p[i] > 0
    # Going up: altitude increases (positive dz)
    for i in range(ref_idx - 1, -1, -1):
        dp = pressure[i + 1] - pressure[i]  # positive
        p_mid = 0.5 * (pressure[i] + pressure[i + 1])
        dz = dp * 1e5 / (p_mid * 1e5 * g * mmw / 1000.0 / R_univ / params.T)
        altitude[i] = altitude[i + 1] + dz
    # Downward (higher pressure, indices increasing from ref_idx+1)
    # pressure[i] > pressure[i-1], dp > 0
    # Going down: altitude decreases (negative dz)
    for i in range(ref_idx + 1, n_layers):
        dp = pressure[i] - pressure[i - 1]  # positive
        p_mid = 0.5 * (pressure[i] + pressure[i - 1])
        dz = -dp * 1e5 / (p_mid * 1e5 * g * mmw / 1000.0 / R_univ / params.T)
        altitude[i] = altitude[i - 1] + dz  # dz is negative

    return pressure, temperature, altitude, mmw


# ---------------------------------------------------------------------------
# Radiative transfer (mock)
# ---------------------------------------------------------------------------


def _compute_transmission_radius(
    wl: np.ndarray,
    params: ModelParams,
    fixed: PlanetFixed,
    pressure: np.ndarray,
    temperature: np.ndarray,
    altitude: np.ndarray,
    mmw: float,
) -> np.ndarray:
    """Compute effective transit radius Rp(λ) using mock opacities.

    This is a simplified integral: tau = integral kappa * rho * dz
    Effective radius where tau = 1 (slant optical depth ~ sqrt(2*pi*Rp*H) * vertical tau)
    """
    vmr = params.vmr_dict()
    n_layers = len(pressure)
    n_wl = len(wl)

    # Cross-sections at each layer (assuming constant VMR with altitude)
    xsec = _mock_cross_sections(wl, vmr)  # shape (n_wl,)
    cia = _h2_he_cia(wl)
    rayleigh = _rayleigh_scattering(wl)

    # Total opacity (vertical, per unit mass)
    kappa_total = xsec + cia + rayleigh  # (n_wl,)

    # Density at each layer: rho = P * mmw / (R_univ * T)
    R_univ = 8.314462618
    rho = pressure * 1e5 * (mmw / 1000.0) / (R_univ * temperature)  # kg/m^3

    # Cloud opacity: infinite below cloud top
    p_cloud = 10**params.log_p_cloud
    cloud_mask = pressure > p_cloud

    # Vertical optical depth per layer
    # dtau = kappa * rho * dz, where dz = altitude difference
    dz = np.abs(np.diff(altitude, prepend=altitude[0]))
    dtau = kappa_total[:, None] * rho[None, :] * dz[None, :]  # (n_wl, n_layers)

    # Apply cloud: set dtau = large value below cloud
    dtau[:, cloud_mask] = 1e6

    # Slant optical depth: tau_slant = sqrt(2*pi*Rp/H) * tau_vertical
    # Scale height H = R_univ * T / (g * mmw)
    H = R_univ * params.T / (fixed.gravity_m_s2 * mmw / 1000.0)
    r_ref_m = params.r_ref * 7.1492e7  # R_Jup to m
    slant_factor = np.sqrt(2.0 * np.pi * r_ref_m / H)
    tau_slant = np.cumsum(dtau * slant_factor, axis=1)

    # Find radius where tau_slant = 1 for each wavelength.
    # tau_slant is non-decreasing along layers (cumsum of non-negative dtau),
    # so the first crossing index per row vectorizes exactly like the former
    # per-wavelength `searchsorted(tau, 1.0, side="left")` loop.
    crossed_at_top = tau_slant[:, 0] >= 1.0
    ever_crosses = tau_slant[:, -1] >= 1.0
    idx = np.argmax(tau_slant >= 1.0, axis=1)
    # For rows that cross strictly inside, argmax == searchsorted-left in [1, n_layers).
    idx_safe = np.clip(idx, 1, n_layers - 1)
    t0 = tau_slant[np.arange(n_wl), idx_safe - 1]
    t1 = tau_slant[np.arange(n_wl), idx_safe]
    a0 = altitude[idx_safe - 1]
    a1 = altitude[idx_safe]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_frac = (np.log(1.0) - np.log(t0)) / (np.log(t1) - np.log(t0))
    lin_frac = 1.0 / t1
    use_log = (t1 > t0) & (t0 > 0)
    use_lin = (t1 > t0) & ~(t0 > 0)
    frac = np.where(use_log, log_frac, np.where(use_lin, lin_frac, 0.0))
    rp_inner = r_ref_m + a0 + frac * (a1 - a0)
    rp = np.where(
        crossed_at_top,
        r_ref_m + altitude[0],
        np.where(~ever_crosses, r_ref_m + altitude[-1], rp_inner),
    )

    return rp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_RADTRANS_CACHE: dict[tuple, Any] = {}


def get_radtrans(
    line_species: tuple[str, ...] = ("H2O", "CO2", "CO", "CH4", "SO2"),
    rayleigh_species: tuple[str, ...] = ("H2", "He"),
    continuum_opacities: tuple[str, ...] = ("H2-H2", "H2-He"),
    wlen_bords_micron: tuple[float, float] = (WAVE_MIN, WAVE_MAX),
    mode: str = "c-k",
) -> object:
    """Get or create a cached Radtrans-like object (mock).

    In real pRT this would be a `petitRADTRANS.Radtrans` instance.
    Here we return a simple namespace with the configuration.
    """
    key = (line_species, rayleigh_species, continuum_opacities, wlen_bords_micron, mode)
    if key not in _RADTRANS_CACHE:
        _RADTRANS_CACHE[key] = type(
            "MockRadtrans",
            (),
            {
                "line_species": line_species,
                "rayleigh_species": rayleigh_species,
                "continuum_opacities": continuum_opacities,
                "wlen_bords_micron": wlen_bords_micron,
                "mode": mode,
                "opacity_mode": OPACITY_MODE,
            },
        )()
    return _RADTRANS_CACHE[key]


def transmission_spectrum(
    params: ModelParams,
    fixed: PlanetFixed,
    wavelength_grid: np.ndarray | None = None,
    radtrans: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute transmission spectrum: (wavelength_um, transit_depth_fraction).

    Transit depth = (Rp(lambda) / R_star)^2.

    Args:
        params: ModelParams (free parameters)
        fixed: PlanetFixed (fixed parameters from catalog)
        wavelength_grid: optional custom grid (um); if None, uses native mock grid
        radtrans: optional cached Radtrans object (mock)

    Returns:
        (wavelength_um, transit_depth_fraction) both as np.ndarray
    """
    start = time.perf_counter()

    params.validate()
    fixed.validate()

    if wavelength_grid is None:
        # Native mock grid at R=1000
        n_points = int((WAVE_MAX - WAVE_MIN) * R)
        wl = np.linspace(WAVE_MIN, WAVE_MAX, n_points)
    else:
        wl = np.asarray(wavelength_grid, dtype=np.float64)

    # Ensure within mock range
    wl = wl[(wl >= WAVE_MIN) & (wl <= WAVE_MAX)]
    if len(wl) == 0:
        raise ValueError(f"Wavelength grid must overlap [{WAVE_MIN}, {WAVE_MAX}] um")

    # Build atmosphere
    pressure, temperature, altitude, mmw = _build_atmosphere(params, fixed)

    # Compute Rp(lambda)
    rp_m = _compute_transmission_radius(wl, params, fixed, pressure, temperature, altitude, mmw)

    # Stellar radius in meters
    rs_m = fixed.stellar_radius_rsun * 6.957e8

    # Transit depth = (Rp / Rs)^2
    depth = (rp_m / rs_m) ** 2

    elapsed = time.perf_counter() - start
    return wl, depth, elapsed


def to_instrument(
    model_wl: np.ndarray,
    model_depth: np.ndarray,
    target_spectrum: Spectrum,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin model onto target spectrum's wavelength grid using bin edges.

    Conserves flux in each bin: depth_bin = sum(depth_i * delta_wl_i) / sum(delta_wl_i)

    Args:
        model_wl: model wavelengths (um), assumed uniform or arbitrary
        model_depth: model transit depth (fractional)
        target_spectrum: Spectrum with wavelength_bin_edges

    Returns:
        (binned_wl, binned_depth) on target grid centers
    """
    model_wl = np.asarray(model_wl, dtype=np.float64)
    model_depth = np.asarray(model_depth, dtype=np.float64)
    target_edges = np.asarray(target_spectrum.wavelength_bin_edges, dtype=np.float64)

    if len(target_edges) < 2:
        raise ValueError("target_spectrum must have wavelength_bin_edges with length >= 2")

    n_bins = len(target_edges) - 1
    binned_depth = np.zeros(n_bins, dtype=np.float64)
    binned_wl = 0.5 * (target_edges[:-1] + target_edges[1:])

    for i in range(n_bins):
        lo, hi = target_edges[i], target_edges[i + 1]
        mask = (model_wl >= lo) & (model_wl < hi)
        if i == n_bins - 1:
            # Include right edge for last bin
            mask = (model_wl >= lo) & (model_wl <= hi)

        if np.any(mask):
            # Flux-conserving: weight by wavelength width of each model point's bin
            wl_in = model_wl[mask]
            d_in = model_depth[mask]
            # Use uniform model grid spacing as weights (approximate)
            if len(wl_in) == 1:
                dw = hi - lo
            else:
                dw = np.diff(wl_in, prepend=wl_in[0] - (wl_in[1] - wl_in[0]))
                # Clip to bin boundaries
                dw = np.clip(dw, 0, hi - lo)
            binned_depth[i] = np.sum(d_in * dw) / np.sum(dw)
        else:
            # No model points in this bin: interpolate from neighbors
            binned_depth[i] = np.interp(binned_wl[i], model_wl, model_depth)

    return binned_wl, binned_depth


def compute_model_spectrum(
    params: ModelParams,
    fixed: PlanetFixed,
    target_spectrum: Spectrum | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """High-level: compute model, optionally binned to target.

    Returns:
        (wavelength_um, depth_fraction, metadata_dict)
    """
    radtrans = get_radtrans()
    wl, depth, elapsed = transmission_spectrum(params, fixed, radtrans=radtrans)

    if target_spectrum is not None:
        wl, depth = to_instrument(wl, depth, target_spectrum)

    meta = {
        "model_version": MOCK_VERSION,
        "opacity_mode": OPACITY_MODE,
        "R": R,
        "wave_range_um": (float(wl[0]), float(wl[-1])),
        "n_points": int(len(wl)),
        "compute_time_s": elapsed,
        "params": {
            "T": params.T,
            "log_h2o": params.log_h2o,
            "log_co2": params.log_co2,
            "log_co": params.log_co,
            "log_ch4": params.log_ch4,
            "log_so2": params.log_so2,
            "r_ref": params.r_ref,
            "log_p_cloud": params.log_p_cloud,
        },
        "fixed": {
            "gravity_m_s2": fixed.gravity_m_s2,
            "stellar_radius_rsun": fixed.stellar_radius_rsun,
            "reference_pressure_bar": fixed.reference_pressure_bar,
        },
    }
    return wl, depth, meta
