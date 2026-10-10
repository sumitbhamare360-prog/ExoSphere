"""Retrieval samplers: dynesty (default) and JAXNS (primary, if compatible).

Common interface: run(spectrum, fixed, config, seed) -> RetrievalResult

Note: JAXNS may not be compatible with the numpy-based forward model.
If JAXNS cannot call the forward model, dynesty is used as fallback.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exosphere.core.spectrum import Spectrum
from exosphere.forward.model import PlanetFixed
from exosphere.retrieval.likelihood import log_likelihood
from exosphere.retrieval.priors import (
    DEFAULT_PRIOR_CONFIG,
    PriorConfig,
    log_prior,
    unit_to_physical,
)
from exosphere.retrieval.results import RetrievalResult


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for nested samplers."""

    # Sampler choice: "dynesty" or "jaxns"
    sampler: str = "dynesty"

    # dynesty proposal method: "auto" (uniform multi-ellipsoid) or "rslice".
    # "rslice" (slice sampling from live points) is the default: uniform
    # proposals suffer acceptance collapse on our vague 8-D priors, while
    # slice sampling converges (DECISIONS.md Phase 9a). Still seeded.
    sample: str = "rslice"

    # Number of live points
    n_live: int = 500

    # Termination criterion: dlogz
    dlogz: float = 0.01

    # Maximum number of iterations (safety)
    max_iter: int | None = None

    # Maximum number of likelihood calls (safety cap, dynesty 3.x)
    maxcall: int | None = None

    # Checkpoint file for dynesty resume (Phase 9 hardening: survives kills).
    # When set and the file exists, the run resumes instead of restarting.
    checkpoint_path: str | None = None

    # Random seed
    seed: int = 42

    # Error inflation: 0 = off, >0 = fixed, "free" = fit as parameter
    error_inflation: str | float = 0.0

    # Number of parallel workers (dynesty only, via pool)
    n_workers: int = 1

    # Prior configuration
    prior: PriorConfig | None = None


DEFAULT_SAMPLER_CONFIG = SamplerConfig()


class PriorTransform:
    """Picklable unit-cube -> physical transform (module level for checkpoints)."""

    def __init__(self, catalog_r_ref: float, prior_cfg: PriorConfig) -> None:
        self.catalog_r_ref = catalog_r_ref
        self.prior_cfg = prior_cfg

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return unit_to_physical(np.asarray(u, dtype=float), self.catalog_r_ref, self.prior_cfg)


def _make_prior_transform(catalog_r_ref: float, prior_cfg: PriorConfig):
    """Create a prior transform function for dynesty (unit cube -> physical)."""
    return PriorTransform(catalog_r_ref, prior_cfg)


class LikelihoodFunction:
    """Picklable dynesty likelihood (module level so sampler checkpoints pickle).

    dynesty calls this with PHYSICAL parameters (it applies our
    prior_transform to unit-cube proposals itself), so no
    unit-to-physical transform happens here (doing so double-transforms
    and silently corrupts every retrieval).
    """

    def __init__(
        self,
        spectrum: Spectrum,
        fixed: PlanetFixed,
        catalog_r_ref: float,
        prior_cfg: PriorConfig,
        error_inflation: float,
    ) -> None:
        self.spectrum = spectrum
        self.fixed = fixed
        self.catalog_r_ref = catalog_r_ref
        self.prior_cfg = prior_cfg
        self.error_inflation = error_inflation

    def __call__(self, physical_params: np.ndarray) -> float:
        physical_params = np.asarray(physical_params, dtype=float)
        # Check prior
        lp = log_prior(physical_params, self.catalog_r_ref, self.prior_cfg)
        if not np.isfinite(lp):
            return -np.inf

        # Compute log-likelihood
        return log_likelihood(
            physical_params, self.spectrum, self.fixed, error_inflation=self.error_inflation
        )


def _make_log_likelihood(
    spectrum: Spectrum,
    fixed: PlanetFixed,
    catalog_r_ref: float,
    prior_cfg: PriorConfig,
    error_inflation: float,
) -> Callable[[np.ndarray], float]:
    """Create a log-likelihood function for dynesty (see LikelihoodFunction)."""
    return LikelihoodFunction(spectrum, fixed, catalog_r_ref, prior_cfg, error_inflation)


def run_dynesty(
    spectrum: Spectrum,
    fixed: PlanetFixed,
    config: SamplerConfig,
    seed: int,
) -> RetrievalResult:
    """Run dynesty nested sampling.

    Args:
        spectrum: Observed Spectrum
        fixed: PlanetFixed parameters
        config: SamplerConfig
        seed: Random seed

    Returns:
        RetrievalResult with posterior samples, weights, logZ, etc.
    """
    import dynesty

    # Use catalog r_ref = 1.27 (WASP-39 b default)
    prior_cfg = config.prior or DEFAULT_PRIOR_CONFIG
    error_inflation = float(config.error_inflation) if config.error_inflation != "free" else 0.0

    prior_transform = _make_prior_transform(1.27, prior_cfg)
    log_likelihood_fn = _make_log_likelihood(spectrum, fixed, 1.27, prior_cfg, error_inflation)

    ndim = 8
    sampler = dynesty.NestedSampler(
        log_likelihood_fn,
        prior_transform,
        ndim,
        nlive=config.n_live,
        sample=config.sample,
        rstate=np.random.default_rng(seed),
    )

    start_time = time.time()
    # Static nested-sampling run; maxiter/maxcall bound it, dlogz stops it.
    checkpoint = config.checkpoint_path or None
    sampler.run_nested(
        maxiter=config.max_iter,
        maxcall=config.maxcall,
        dlogz=config.dlogz,
        print_progress=True,
        checkpoint_file=checkpoint,
        resume=bool(checkpoint and Path(checkpoint).exists()),
    )
    elapsed = time.time() - start_time

    results = sampler.results
    np.exp(results.logwt - results.logz[-1])
    results.logz[-1]
    results.logzerr[-1]

    return RetrievalResult.from_dynesty(
        results=results,
        spectrum=spectrum,
        fixed=fixed,
        sampler_config=config,
        seed=seed,
        runtime_s=elapsed,
    )


def run_jaxns(
    spectrum: Spectrum,
    fixed: PlanetFixed,
    config: SamplerConfig,
    seed: int,
) -> RetrievalResult:
    """Run JAXNS nested sampling.

    Note: JAXNS requires JAX-traceable forward model. Our forward model is
    numpy-based and not JAX-traceable. This function raises an error to
    trigger fallback to dynesty.
    """
    raise RuntimeError(
        "JAXNS requires JAX-traceable forward model. "
        "Current forward model is numpy-based and not JAX-traceable. "
        "Use dynesty sampler instead. "
        "To enable JAXNS, implement a JAX-traceable forward model."
    )


def run_retrieval(
    spectrum: Spectrum,
    fixed: PlanetFixed,
    config: SamplerConfig | None = None,
    seed: int | None = None,
) -> RetrievalResult:
    """Run retrieval with specified sampler (dynesty or JAXNS).

    Args:
        spectrum: Observed Spectrum
        fixed: PlanetFixed parameters
        config: SamplerConfig
        seed: Random seed (overrides config.seed if provided)

    Returns:
        RetrievalResult
    """
    if config is None:
        config = SamplerConfig()

    if seed is not None:
        config = SamplerConfig(
            sampler=config.sampler,
            n_live=config.n_live,
            dlogz=config.dlogz,
            max_iter=config.max_iter,
            seed=seed,
            error_inflation=config.error_inflation,
            n_workers=config.n_workers,
            prior=config.prior,
        )

    if config.sampler == "jaxns":
        # JAXNS not compatible with numpy forward model - fall back
        import warnings

        warnings.warn(
            "JAXNS requested but not compatible with numpy forward model. "
            "Falling back to dynesty. To use JAXNS, implement a JAX-traceable forward model.",
            UserWarning,
            stacklevel=2,
        )
        config = SamplerConfig(
            sampler="dynesty",
            n_live=config.n_live,
            dlogz=config.dlogz,
            max_iter=config.max_iter,
            seed=config.seed,
            error_inflation=config.error_inflation,
            n_workers=config.n_workers,
            prior=config.prior,
        )

    if config.sampler == "dynesty":
        return run_dynesty(spectrum, fixed, config, config.seed)

    raise ValueError(f"Unknown sampler: {config.sampler}")


def run(
    spectrum: Spectrum,
    fixed: PlanetFixed,
    config: SamplerConfig | None = None,
    seed: int | None = None,
) -> RetrievalResult:
    """Run retrieval with default config if not provided."""
    if config is None:
        config = SamplerConfig()
    return run_retrieval(spectrum, fixed, config, seed)
