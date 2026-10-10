"""Per-molecule detection evidence via nested-model comparison.

Computes ln Bayes factor by comparing full model vs. model with one molecule removed.
Uses nested sampling evidence (logZ) from dynesty.

Reference: Benneke & Seager (2013) "Atmospheric Retrieval for Super-Earths"
ln B > 3: substantial evidence; >5: strong; >10: very strong
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from exosphere.core.spectrum import Spectrum
from exosphere.forward.model import PlanetFixed
from exosphere.retrieval.priors import DEFAULT_PRIOR_CONFIG, PriorConfig, unit_to_physical
from exosphere.retrieval.results import RetrievalResult
from exosphere.retrieval.samplers import SamplerConfig

MOLECULE_INDICES = {
    "H2O": 1,
    "CO2": 2,
    "CO": 3,
    "CH4": 4,
    "SO2": 5,
}


class ReducedPriorTransform:
    """Fixed-molecule prior transform (module level so checkpoints pickle)."""

    def __init__(self, catalog_r_ref, prior_cfg, fixed_mol_idx, fixed_value):
        self.catalog_r_ref = catalog_r_ref
        self.prior_cfg = prior_cfg
        self.fixed_mol_idx = fixed_mol_idx
        self.fixed_value = fixed_value

    def __call__(self, u):
        params = unit_to_physical(
            np.asarray(u, dtype=float), self.catalog_r_ref, self.prior_cfg
        )
        params[self.fixed_mol_idx] = self.fixed_value
        return params


class ReducedLikelihoodFunction:
    """Reduced-model likelihood (module level so checkpoints pickle)."""

    def __init__(
        self, spectrum, fixed, catalog_r_ref, prior_cfg, error_inflation,
        fixed_mol_idx, fixed_value,
    ):
        self.spectrum = spectrum
        self.fixed = fixed
        self.catalog_r_ref = catalog_r_ref
        self.prior_cfg = prior_cfg
        self.error_inflation = error_inflation
        self.fixed_mol_idx = fixed_mol_idx
        self.fixed_value = fixed_value

    def __call__(self, physical_params):
        # dynesty passes physical parameters (prior_transform applied
        # by the sampler); only fix the molecule, never re-transform.
        from exosphere.retrieval.likelihood import log_likelihood
        from exosphere.retrieval.priors import log_prior

        params = np.asarray(physical_params, dtype=float).copy()
        params[self.fixed_mol_idx] = self.fixed_value
        lp = log_prior(params, self.catalog_r_ref, self.prior_cfg)
        if not np.isfinite(lp):
            return -np.inf
        return log_likelihood(
            params, self.spectrum, self.fixed, error_inflation=self.error_inflation
        )


def compute_bayes_factor(
    full_result: RetrievalResult,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    molecule: str,
    sampler_config: SamplerConfig,
    seed: int,
    prior_cfg: PriorConfig | None = None,
) -> float:
    """Compute ln Bayes factor for a molecule by nested-model comparison.

    Runs retrieval with the specified molecule removed (log VMR -> -inf),
    compares logZ with full model.

    ln B = logZ(full) - logZ(reduced)

    Args:
        full_result: RetrievalResult from full model
        spectrum: Observed Spectrum
        fixed: PlanetFixed parameters
        molecule: Molecule name to test ("H2O", "CO2", "CO", "CH4", "SO2")
        sampler_config: SamplerConfig for the nested run
        seed: Random seed
        prior_cfg: Optional prior config override

    Returns:
        ln Bayes factor (positive favors full model with molecule)
    """
    if molecule not in MOLECULE_INDICES:
        raise ValueError(f"Unknown molecule: {molecule}")

    mol_idx = MOLECULE_INDICES[molecule]

    # Create reduced prior config: force the molecule's VMR to be negligible
    # We do this by setting a very low prior upper bound for that molecule
    # Actually, the simplest approach: modify the prior transform to fix that parameter to -12
    reduced_prior = PriorConfig(
        T_min=sampler_config.prior.T_min if sampler_config.prior else 300.0,
        T_max=sampler_config.prior.T_max if sampler_config.prior else 2500.0,
        log_vmr_min=sampler_config.prior.log_vmr_min if sampler_config.prior else -12.0,
        log_vmr_max=sampler_config.prior.log_vmr_max if sampler_config.prior else -1.0,
        r_ref_frac=sampler_config.prior.r_ref_frac if sampler_config.prior else 0.30,
        log_p_cloud_min=sampler_config.prior.log_p_cloud_min if sampler_config.prior else -6.0,
        log_p_cloud_max=sampler_config.prior.log_p_cloud_max if sampler_config.prior else 2.0,
    )

    # For the reduced model, we fix the molecule's log VMR to -12 (negligible)
    # via a custom prior transform; the sampler then explores the reduced model.
    def make_reduced_prior_transform(catalog_r_ref, prior_cfg, fixed_mol_idx, fixed_value):
        return ReducedPriorTransform(catalog_r_ref, prior_cfg, fixed_mol_idx, fixed_value)

    reduced_prior = make_reduced_prior_transform(1.27, DEFAULT_PRIOR_CONFIG, mol_idx, -12.0)

    def make_reduced_log_likelihood(
        spectrum, fixed, catalog_r_ref, prior_cfg, error_inflation, fixed_mol_idx, fixed_value
    ):
        return ReducedLikelihoodFunction(
            spectrum, fixed, catalog_r_ref, prior_cfg, error_inflation,
            fixed_mol_idx, fixed_value,
        )

    # Run reduced model
    try:
        import dynesty

        ndim = 8
        checkpoint = sampler_config.checkpoint_path
        if checkpoint:
            Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)

        # As in the full retrieval, dynesty requires an explicit restore:
        # ``resume=True`` on a newly constructed sampler restarts it.
        if checkpoint and Path(checkpoint).exists():
            reduced_sampler = dynesty.NestedSampler.restore(checkpoint)
        else:
            reduced_sampler = dynesty.NestedSampler(
                make_reduced_log_likelihood(
                    spectrum, fixed, 1.27, DEFAULT_PRIOR_CONFIG, 0.0, mol_idx, -12.0
                ),
                reduced_prior,
                ndim,
                nlive=sampler_config.n_live,
                sample=sampler_config.sample,
                rstate=np.random.default_rng(seed),
            )

        reduced_sampler.run_nested(
            maxiter=sampler_config.max_iter,
            maxcall=sampler_config.maxcall,
            dlogz=sampler_config.dlogz,
            print_progress=False,
            checkpoint_file=checkpoint,
            resume=bool(checkpoint and Path(checkpoint).exists()),
        )

        reduced_logz = reduced_sampler.results.logz[-1]
        reduced_logz_err = reduced_sampler.results.logzerr[-1]

        ln_B = full_result.logz - reduced_logz
        ln_B_err = np.sqrt(full_result.logz_err**2 + reduced_logz_err**2)

        res = reduced_sampler.results
        # ``results.ncall`` holds calls *per nested-sampling iteration*, not
        # a running total.  Looking at its final element hid maxcall stops.
        ncall = int(np.sum(res.ncall)) if hasattr(res.ncall, "__len__") else int(res.ncall)
        diagnostics = {
            "niter": int(res.niter),
            "ncall": ncall,
            "capped": bool(
                (sampler_config.maxcall is not None and ncall >= sampler_config.maxcall)
                or (
                    sampler_config.max_iter is not None
                    and int(res.niter) >= sampler_config.max_iter
                )
            ),
            "logz_reduced": float(reduced_logz),
            "logz_reduced_err": float(reduced_logz_err),
            "logz_full": float(full_result.logz),
        }
        return ln_B, ln_B_err, diagnostics

    except Exception as e:
        warnings.warn(f"Failed to compute Bayes factor for {molecule}: {e}", stacklevel=2)
        return 0.0, np.inf, {"failed": True}


def compute_all_bayes_factors(
    full_result: RetrievalResult,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    sampler_config: SamplerConfig,
    seed: int,
    molecules: list[str] | None = None,
    checkpoint_dir: str | None = None,
) -> dict[str, tuple[float, float, dict]]:
    """Compute ln Bayes factors for all specified molecules.

    Returns dict: molecule -> (ln_B, ln_B_err, diagnostics). Diagnostics
    carry ncall/niter, whether caps stopped the run, and the reduced logZ.
    """
    if molecules is None:
        molecules = list(MOLECULE_INDICES.keys())

    results = {}
    for molecule in molecules:
        try:
            nested_config = sampler_config
            if checkpoint_dir is not None:
                import dataclasses

                nested_config = dataclasses.replace(
                    sampler_config,
                    checkpoint_path=str(Path(checkpoint_dir) / f"nested_{molecule}.pkl"),
                )
            bf, bf_err, diagnostics = compute_bayes_factor(
                full_result, spectrum, fixed, molecule, nested_config, seed
            )
            results[molecule] = (bf, bf_err, diagnostics)
        except Exception as e:
            warnings.warn(f"Failed to compute Bayes factor for {molecule}: {e}", stacklevel=2)
            results[molecule] = (0.0, np.inf, {"failed": True})

    return results


def detection_sigma(ln_B: float) -> float:
    """Convert ln Bayes factor to approximate sigma (Gaussian significance).

    Using Benneke & Seager (2013) approximation:
    ln B ~ 0.5 * sigma^2 for strong detections
    So sigma ~ sqrt(2 * ln B)
    """
    if ln_B <= 0:
        return 0.0
    return np.sqrt(2.0 * ln_B)


def compute_upper_limits(
    result: RetrievalResult,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    sampler_config: SamplerConfig,
    seed: int,
    molecules: list[str] | None = None,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compute posterior upper limits for non-detected molecules.

    For each molecule, find the VMR value at the specified confidence level
    of the marginal posterior. If the molecule is not detected (ln B < 3),
    report the 95% upper limit on log VMR.

    Args:
        result: Full RetrievalResult
        spectrum: Observed Spectrum
        fixed: PlanetFixed
        sampler_config: SamplerConfig
        seed: Random seed
        molecules: List of molecules to compute limits for
        confidence: Confidence level (0.95 default)

    Returns:
        Dict: molecule -> upper limit on log VMR
    """
    if molecules is None:
        molecules = list(MOLECULE_INDICES.keys())

    limits = {}
    for molecule in molecules:
        if molecule not in MOLECULE_INDICES:
            continue
        idx = MOLECULE_INDICES[molecule]
        samples = result.samples[:, idx]
        weights = result.weights

        # Weighted percentile
        def weighted_percentile(x, w, q):
            idx = np.argsort(x)
            x_sorted = x[idx]
            w_sorted = w[idx]
            cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
            return np.interp(q, cdf, x_sorted)

        upper = weighted_percentile(samples, weights, confidence)
        limits[molecule] = float(upper)

    return limits


def detection_summary(
    full_result: RetrievalResult,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    sampler_config: SamplerConfig,
    seed: int,
    molecules: list[str] | None = None,
    checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    """Run full detection analysis: Bayes factors + upper limits + sigma.

    Returns a summary dictionary suitable for JSON serialization.
    """
    bfs = compute_all_bayes_factors(
        full_result, spectrum, fixed, sampler_config, seed, molecules,
        checkpoint_dir=checkpoint_dir,
    )
    compute_upper_limits(full_result, spectrum, fixed, sampler_config, seed, molecules)

    summary = {}
    for molecule in molecules or list(MOLECULE_INDICES.keys()):
        bf, bf_err, diagnostics = bfs.get(molecule, (0.0, np.inf, {"failed": True}))
        sigma = detection_sigma(bf)
        limit = full_result.upper_limits.get(molecule, None)

        status = "detected" if bf > 3 else ("tentative" if bf > 1 else "not detected")

        summary[molecule] = {
            "ln_B": float(bf),
            "ln_B_err": float(bf_err) if np.isfinite(bf_err) else None,
            "sigma": float(sigma),
            "status": status,
            "upper_limit_log_vmr": limit,
            "diagnostics": diagnostics,
        }

    return summary


def update_result_with_detection(
    result: RetrievalResult,
    spectrum: Spectrum,
    fixed: PlanetFixed,
    sampler_config: SamplerConfig,
    seed: int,
) -> RetrievalResult:
    """Update a RetrievalResult with detection info (modifies in place)."""
    summary = detection_summary(result, spectrum, fixed, sampler_config, seed)
    for molecule, info in summary.items():
        result.bayes_factors[molecule] = info["ln_B"]
        if info["upper_limit_log_vmr"] is not None:
            result.upper_limits[molecule] = info["upper_limit_log_vmr"]
    return result
