"""Prior definitions for Bayesian retrieval (AGENTS.md §2, §8).

All priors are defined as unit-cube -> parameter transforms, compatible with
both dynesty and JAXNS. The transform takes a vector in [0, 1]^n and returns
physical parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Parameter names in order matching ModelParams
PARAM_NAMES = [
    "T",
    "log_h2o",
    "log_co2",
    "log_co",
    "log_ch4",
    "log_so2",
    "r_ref",
    "log_p_cloud",
]


@dataclass(frozen=True)
class PriorConfig:
    """Configuration for prior bounds (unit cube -> physical)."""

    # Temperature: uniform in [T_min, T_max]
    T_min: float = 300.0
    T_max: float = 2500.0

    # Log VMRs: uniform in [log_min, log_max]
    log_vmr_min: float = -12.0
    log_vmr_max: float = -1.0

    # Reference radius: uniform within +/- fraction of catalog value
    r_ref_frac: float = 0.30  # +/- 30%

    # Cloud-top pressure: uniform in log space [log_min, log_max] (bar)
    log_p_cloud_min: float = -6.0
    log_p_cloud_max: float = 2.0


DEFAULT_PRIOR_CONFIG = PriorConfig()


def unit_to_physical(
    u: np.ndarray, catalog_r_ref: float, cfg: PriorConfig = DEFAULT_PRIOR_CONFIG
) -> np.ndarray:
    """Transform unit-cube vector to physical parameters.

    Args:
        u: array of shape (8,) in [0, 1]
        catalog_r_ref: catalog reference radius (R_Jup)
        cfg: prior configuration

    Returns:
        Physical parameters array
            [T, log_h2o, log_co2, log_co, log_ch4, log_so2, r_ref, log_p_cloud]
    """
    u = np.asarray(u, dtype=np.float64)
    # Clip to valid range (dynesty may probe slightly outside during initialization)
    u = np.clip(u, 0.0, 1.0)
    assert u.shape == (8,), f"Expected 8 parameters, got {u.shape}"

    params = np.zeros(8, dtype=np.float64)

    # T: uniform
    params[0] = cfg.T_min + u[0] * (cfg.T_max - cfg.T_min)

    # log VMRs: uniform in log space
    vmr_range = cfg.log_vmr_max - cfg.log_vmr_min
    params[1] = cfg.log_vmr_min + u[1] * vmr_range  # log_h2o
    params[2] = cfg.log_vmr_min + u[2] * vmr_range  # log_co2
    params[3] = cfg.log_vmr_min + u[3] * vmr_range  # log_co
    params[4] = cfg.log_vmr_min + u[4] * vmr_range  # log_ch4
    params[5] = cfg.log_vmr_min + u[5] * vmr_range  # log_so2

    # r_ref: uniform within +/- frac of catalog
    params[6] = catalog_r_ref * (1.0 - cfg.r_ref_frac) + u[6] * (
        2.0 * cfg.r_ref_frac * catalog_r_ref
    )

    # log_p_cloud: uniform in log space
    params[7] = cfg.log_p_cloud_min + u[7] * (cfg.log_p_cloud_max - cfg.log_p_cloud_min)

    return params


def physical_to_unit(
    params: np.ndarray, catalog_r_ref: float, cfg: PriorConfig = DEFAULT_PRIOR_CONFIG
) -> np.ndarray:
    """Inverse transform: physical parameters to unit cube.

    Args:
        params: physical parameters array
            [T, log_h2o, log_co2, log_co, log_ch4, log_so2, r_ref, log_p_cloud]
        catalog_r_ref: catalog reference radius (R_Jup)
        cfg: prior configuration

    Returns:
        Unit cube vector in [0, 1]^8
    """
    params = np.asarray(params, dtype=np.float64)
    assert params.shape == (8,), f"Expected 8 parameters, got {params.shape}"

    u = np.zeros(8, dtype=np.float64)
    cfg = cfg or DEFAULT_PRIOR_CONFIG

    u[0] = (params[0] - cfg.T_min) / (cfg.T_max - cfg.T_min)

    vmr_range = cfg.log_vmr_max - cfg.log_vmr_min
    u[1] = (params[1] - cfg.log_vmr_min) / vmr_range
    u[2] = (params[2] - cfg.log_vmr_min) / vmr_range
    u[3] = (params[3] - cfg.log_vmr_min) / vmr_range
    u[4] = (params[4] - cfg.log_vmr_min) / vmr_range
    u[5] = (params[5] - cfg.log_vmr_min) / vmr_range

    u[6] = (params[6] - catalog_r_ref * (1.0 - cfg.r_ref_frac)) / (
        2.0 * cfg.r_ref_frac * catalog_r_ref
    )

    u[7] = (params[7] - cfg.log_p_cloud_min) / (cfg.log_p_cloud_max - cfg.log_p_cloud_min)

    return np.clip(u, 0.0, 1.0)


def check_vmr_sum(params: np.ndarray) -> bool:
    """Check if sum of VMRs < 1 (prior constraint).

    Args:
        params: physical parameters array

    Returns:
        True if sum(VMR) < 1, False otherwise
    """
    vmr_sum = (
        10 ** params[1]  # H2O
        + 10 ** params[2]  # CO2
        + 10 ** params[3]  # CO
        + 10 ** params[4]  # CH4
        + 10 ** params[5]  # SO2
    )
    return vmr_sum < 1.0


def log_prior(
    params: np.ndarray, catalog_r_ref: float, cfg: PriorConfig = DEFAULT_PRIOR_CONFIG
) -> float:
    """Log-prior density (uniform in the specified bounds).

    Returns -inf if outside bounds or sum(VMR) >= 1.

    Args:
        params: physical parameters
        catalog_r_ref: catalog reference radius
        cfg: prior configuration

    Returns:
        Log-prior density (constant within bounds, -inf outside)
    """
    # Check bounds
    cfg = cfg or DEFAULT_PRIOR_CONFIG
    if not (cfg.T_min <= params[0] <= cfg.T_max):
        return -np.inf
    if not all(cfg.log_vmr_min <= params[i] <= cfg.log_vmr_max for i in range(1, 6)):
        return -np.inf
    r_min = catalog_r_ref * (1.0 - cfg.r_ref_frac)
    r_max = catalog_r_ref * (1.0 + cfg.r_ref_frac)
    if not (r_min <= params[6] <= r_max):
        return -np.inf
    if not (cfg.log_p_cloud_min <= params[7] <= cfg.log_p_cloud_max):
        return -np.inf

    # Check VMR sum constraint
    if not check_vmr_sum(params):
        return -np.inf

    # Uniform prior: constant log-density
    # Volume = (T_max - T_min) * (vmr_range)^5 * (2*frac*R_ref) * (log_p_cloud_range)
    vmr_range = cfg.log_vmr_max - cfg.log_vmr_min
    log_vol = (
        np.log(cfg.T_max - cfg.T_min)
        + 5 * np.log(vmr_range)
        + np.log(2.0 * cfg.r_ref_frac * catalog_r_ref)
        + np.log(cfg.log_p_cloud_max - cfg.log_p_cloud_min)
    )
    return -log_vol


def make_prior_config_from_config(cfg_dict: dict[str, Any]) -> PriorConfig:
    """Create PriorConfig from a configuration dictionary."""
    return PriorConfig(
        T_min=cfg_dict.get("T_min", 300.0),
        T_max=cfg_dict.get("T_max", 2500.0),
        log_vmr_min=cfg_dict.get("log_vmr_min", -12.0),
        log_vmr_max=cfg_dict.get("log_vmr_max", -1.0),
        r_ref_frac=cfg_dict.get("r_ref_frac", 0.30),
        log_p_cloud_min=cfg_dict.get("log_p_cloud_min", -6.0),
        log_p_cloud_max=cfg_dict.get("log_p_cloud_max", 2.0),
    )


# JAXNS-specific prior building (when needed)
def build_jaxns_prior_model(catalog_r_ref: float, cfg: PriorConfig = DEFAULT_PRIOR_CONFIG):
    """Build a JAXNS prior_model callable.

    Returns a callable that takes a PRNGKey and returns a dict of physical parameters.
    The caller wraps this in jaxns.Model(prior_model).

    Note: This requires tensorflow_probability.substrates.jax.distributions.
    """
    try:
        import tensorflow_probability.substrates.jax.distributions as tfd
    except ImportError as e:
        raise RuntimeError("tensorflow_probability[jax] required for JAXNS priors") from e

    cfg = cfg or DEFAULT_PRIOR_CONFIG

    def prior_model(key):
        import jax.random as jr

        keys = jr.split(key, 8)
        params = {}

        # T
        params["T"] = tfd.Uniform(low=300.0, high=2500.0).sample(seed=keys[0])

        # Log VMRs
        for i, name in enumerate(["log_h2o", "log_co2", "log_co", "log_ch4", "log_so2"]):
            params[name] = tfd.Uniform(low=-12.0, high=-1.0).sample(seed=keys[i + 1])

        # r_ref
        params["r_ref"] = tfd.Uniform(
            low=0.7 * 1.27,  # catalog_r_ref * (1 - 0.3)
            high=1.3 * 1.27,  # catalog_r_ref * (1 + 0.3)
        ).sample(seed=keys[6])

        # log_p_cloud
        params["log_p_cloud"] = tfd.Uniform(low=-6.0, high=2.0).sample(seed=keys[7])

        return params

    return prior_model


# Parameter names for convenience
PARAM_NAMES_TUPLE = tuple(PARAM_NAMES)
N_PARAMS = len(PARAM_NAMES)
