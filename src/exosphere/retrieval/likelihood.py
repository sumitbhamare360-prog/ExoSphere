"""Gaussian log-likelihood for retrieval using Spectrum.uncertainty.

Evaluates the forward model on the observed wavelength grid via
forward.to_instrument.
"""

from __future__ import annotations

import numpy as np

from exosphere.core.spectrum import Spectrum
from exosphere.forward.model import ModelParams, PlanetFixed, compute_model_spectrum


def log_likelihood(
    params: np.ndarray,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    error_inflation: float = 0.0,
) -> float:
    """Compute Gaussian log-likelihood for a given parameter set.

    log L = -0.5 * sum((d_obs - d_model)^2 / sigma^2) - 0.5 * sum(log(2*pi*sigma^2))

    where sigma^2 = uncertainty^2 * (1 + error_inflation)^2

    Args:
        params: Physical parameters
            [T, log_h2o, log_co2, log_co, log_ch4, log_so2, r_ref, log_p_cloud]
        spectrum: Observed Spectrum (wavelength, transmission, uncertainty, bin_edges)
        fixed: PlanetFixed parameters (gravity, stellar_radius, reference_pressure)
        error_inflation: Optional multiplicative inflation of uncertainties (default 0)

    Returns:
        Log-likelihood value (float), or -inf if model evaluation fails
    """
    try:
        model_params = ModelParams(
            T=float(params[0]),
            log_h2o=float(params[1]),
            log_co2=float(params[2]),
            log_co=float(params[3]),
            log_ch4=float(params[4]),
            log_so2=float(params[5]),
            r_ref=float(params[6]),
            log_p_cloud=float(params[7]),
        )

        # Compute model spectrum on observed wavelength grid
        # compute_model_spectrum returns (wl, depth, meta) binned to target spectrum
        model_wl, model_depth, _ = compute_model_spectrum(
            model_params, fixed, target_spectrum=spectrum
        )

        # Observed data
        obs_depth = np.asarray(spectrum.transmission, dtype=np.float64)
        obs_unc = np.asarray(spectrum.uncertainty, dtype=np.float64)

        # Check alignment
        if len(model_wl) != len(obs_depth):
            return -np.inf

        # Inflated uncertainties
        sigma = obs_unc * (1.0 + error_inflation)
        if np.any(sigma <= 0):
            return -np.inf

        # Gaussian log-likelihood
        residuals = obs_depth - model_depth
        chi2 = np.sum((residuals / sigma) ** 2)
        log_norm = np.sum(np.log(2.0 * np.pi * sigma**2))

        return -0.5 * (chi2 + log_norm)

    except (ValueError, ZeroDivisionError, FloatingPointError):
        return -np.inf


def log_likelihood_with_inflation(
    params: np.ndarray,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    log_error_inflation: float,
) -> float:
    """Log-likelihood with free error-inflation parameter (log space).

    Args:
        params: Physical parameters (first 8 elements) + log_error_inflation (last element)
        spectrum: Observed Spectrum
        fixed: PlanetFixed parameters
        log_error_inflation: Log of error inflation factor

    Returns:
        Log-likelihood value
    """
    error_inflation = np.exp(log_error_inflation)
    return log_likelihood(params[:8], spectrum, fixed, error_inflation=error_inflation)


def chi2(
    params: np.ndarray,
    spectrum: Spectrum,
    fixed: PlanetFixed,
) -> float:
    """Compute chi-squared for a parameter set (no normalization).

    Useful for goodness-of-fit assessment.

    Args:
        params: Physical parameters
        spectrum: Observed Spectrum
        fixed: PlanetFixed parameters

    Returns:
        Chi-squared value
    """
    model_params = ModelParams(
        T=float(params[0]),
        log_h2o=float(params[1]),
        log_co2=float(params[2]),
        log_co=float(params[3]),
        log_ch4=float(params[4]),
        log_so2=float(params[5]),
        r_ref=float(params[6]),
        log_p_cloud=float(params[7]),
    )

    model_wl, model_depth, _ = compute_model_spectrum(model_params, fixed, target_spectrum=spectrum)

    obs_depth = np.asarray(spectrum.transmission, dtype=np.float64)
    obs_unc = np.asarray(spectrum.uncertainty, dtype=np.float64)

    if len(model_wl) != len(obs_depth):
        return np.inf

    residuals = obs_depth - model_depth
    return np.sum((residuals / obs_unc) ** 2)
