"""RetrievalResult: posterior samples, log-evidence, credible intervals, serialization.

Writes retrieval config and seed into Provenance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.forward.model import PlanetFixed

if TYPE_CHECKING:
    from exosphere.retrieval.samplers import SamplerConfig


@dataclass
class RetrievalResult:
    """Complete retrieval result with posterior, evidence, and metadata."""

    # Core results
    samples: np.ndarray  # (n_samples, n_params) physical parameters
    weights: np.ndarray  # (n_samples,) normalized posterior weights
    logz: float  # log evidence
    logz_err: float  # log evidence uncertainty

    # Best fit and credible intervals
    best_fit: np.ndarray  # (8,) maximum likelihood or MAP
    median: np.ndarray  # (8,) posterior median
    ci_68: np.ndarray  # (8, 2) 68% credible interval [lo, hi]
    ci_95: np.ndarray  # (8, 2) 95% credible interval [lo, hi]

    # Model comparison
    logz_full: float  # log evidence for full model
    logz_err_full: float

    # Metadata
    param_names: list[str] = field(
        default_factory=lambda: [
            "T",
            "log_h2o",
            "log_co2",
            "log_co",
            "log_ch4",
            "log_so2",
            "r_ref",
            "log_p_cloud",
        ]
    )
    n_samples: int = 0
    n_live: int = 0
    dlogz: float = 0.0
    runtime_s: float = 0.0
    sampler: str = "dynesty"
    seed: int = 42
    error_inflation: float = 0.0

    # Provenance and config
    provenance: Provenance | None = None
    retrieval_config: dict = field(default_factory=dict)
    forward_model_version: str = "mock-1.0.0"
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    # Per-molecule detection info (filled by detection.py)
    bayes_factors: dict[str, float] = field(default_factory=dict)
    upper_limits: dict[str, float] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @classmethod
    def from_dynesty(
        cls,
        results,
        spectrum: Spectrum,
        fixed: PlanetFixed,
        sampler_config: SamplerConfig,
        seed: int,
        runtime_s: float,
    ) -> RetrievalResult:
        """Create RetrievalResult from dynesty results."""
        import dynesty.utils as dyutils

        res = results
        samples = res.samples
        weights = np.exp(res.logwt - res.logz[-1])
        res.logz[-1]
        res.logzerr[-1]

        # Best fit (max likelihood)
        logl = res.logl
        best_idx = np.argmax(logl)
        best_fit = res.samples[best_idx]

        # Posterior statistics
        median = dyutils.resample_equal(samples, weights).mean(axis=0)

        # Credible intervals
        def weighted_quantile(x, w, q):
            """Weighted quantile."""
            idx = np.argsort(x)
            x_sorted = x[idx]
            w_sorted = w[idx]
            cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
            return np.interp(q, cdf, x_sorted)

        ci_68 = np.array(
            [
                [
                    weighted_quantile(samples[:, i], weights, 0.16),
                    weighted_quantile(samples[:, i], weights, 0.84),
                ]
                for i in range(samples.shape[1])
            ]
        )
        ci_95 = np.array(
            [
                [
                    weighted_quantile(samples[:, i], weights, 0.025),
                    weighted_quantile(samples[:, i], weights, 0.975),
                ]
                for i in range(samples.shape[1])
            ]
        )

        # Build provenance
        provenance = Provenance(
            analysis_id=f"EXO-{int(time.time()) % 1000000:06d}",
            planet="WASP-39 b",
            observation_id=spectrum.observation_id,
            telescope="JWST",
            instrument=spectrum.instrument,
            source_archive="NASA Exoplanet Archive",
            retrieval_model_version="mock-1.0.0",
            retrieval_parameters={
                "sampler": "dynesty",
                "n_live": 500,
                "dlogz": 0.01,
                "seed": 42,
                "error_inflation": 0.0,
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        return cls(
            samples=samples,
            weights=weights,
            logz=res.logz[-1],
            logz_err=res.logzerr[-1],
            best_fit=best_fit,
            median=median,
            ci_68=ci_68,
            ci_95=ci_95,
            logz_full=res.logz[-1],
            logz_err_full=res.logzerr[-1],
            param_names=[
                "T",
                "log_h2o",
                "log_co2",
                "log_co",
                "log_ch4",
                "log_so2",
                "r_ref",
                "log_p_cloud",
            ],
            n_samples=int(np.sum(weights > 0)),
            n_live=500,
            dlogz=0.01,
            runtime_s=0.0,
            sampler="dynesty",
            seed=42,
            error_inflation=0.0,
            provenance=provenance,
            retrieval_config={},
        )

    @classmethod
    def from_jaxns(cls, *args, **kwargs) -> RetrievalResult:
        """Create RetrievalResult from JAXNS results."""
        raise NotImplementedError("JAXNS results not yet implemented")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (JSON-compatible)."""
        return {
            "samples": self.samples.tolist(),
            "weights": self.weights.tolist(),
            "logz": float(self.logz),
            "logz_err": float(self.logz_err),
            "best_fit": self.best_fit.tolist(),
            "median": self.median.tolist(),
            "ci_68": self.ci_68.tolist(),
            "ci_95": self.ci_95.tolist(),
            "logz_full": float(self.logz_full),
            "logz_err_full": float(self.logz_err_full),
            "param_names": self.param_names,
            "n_samples": int(self.n_samples),
            "n_live": int(self.n_live),
            "dlogz": float(self.dlogz),
            "runtime_s": float(self.runtime_s),
            "sampler": self.sampler,
            "seed": int(self.seed),
            "error_inflation": float(self.error_inflation),
            "bayes_factors": self.bayes_factors,
            "upper_limits": self.upper_limits,
            "provenance": self.provenance.model_dump() if self.provenance else None,
            "retrieval_config": self.retrieval_config,
            "forward_model_version": self.forward_model_version,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> RetrievalResult:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        # Reconstruct arrays
        data["samples"] = np.array(data["samples"])
        data["weights"] = np.array(data["weights"])
        data["best_fit"] = np.array(data["best_fit"])
        data["median"] = np.array(data["median"])
        data["ci_68"] = np.array(data["ci_68"])
        data["ci_95"] = np.array(data["ci_95"])
        # Provenance
        if data.get("provenance"):
            data["provenance"] = Provenance.model_validate(data["provenance"])
        return cls(**data)

    def save_npz(self, path: str | Path) -> Path:
        """Save to compressed .npz archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            samples=self.samples,
            weights=self.weights,
            logz=np.array([self.logz]),
            logz_err=np.array([self.logz_err]),
            best_fit=self.best_fit,
            median=self.median,
            ci_68=self.ci_68,
            ci_95=self.ci_95,
            param_names=np.array(self.param_names, dtype=object),
            metadata=np.array(
                [
                    self.n_samples,
                    self.n_live,
                    self.dlogz,
                    self.runtime_s,
                    self.seed,
                    self.error_inflation,
                ],
                dtype=object,
            ),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> RetrievalResult:
        """Load from .npz archive."""
        data = np.load(path, allow_pickle=True)
        return cls(
            samples=data["samples"],
            weights=data["weights"],
            logz=float(data["logz"][0]),
            logz_err=float(data["logz_err"][0]),
            best_fit=data["best_fit"],
            median=data["median"],
            ci_68=data["ci_68"],
            ci_95=data.ci_95,
            logz_full=float(data["logz"][0]),
            logz_err_full=float(data["logz_err"][0]),
            param_names=data["param_names"].tolist(),
            n_samples=int(data["metadata"][0]),
            n_live=int(data["metadata"][1]),
            dlogz=float(data["metadata"][2]),
            runtime_s=float(data["metadata"][3]),
            sampler="dynesty",
            seed=int(data["metadata"][4]),
            error_inflation=float(data["metadata"][5]),
        )

    def summary_text(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Retrieval summary ({self.sampler}, seed={self.seed})",
            f"  logZ = {self.logz:.3f} +/- {self.logz_err:.3f}",
            f"  n_live = {self.n_live}, n_samples = {self.n_samples}",
            f"  runtime = {self.runtime_s:.1f} s",
            "  Parameter estimates (median, 68% CI):",
        ]
        for i, name in enumerate(self.param_names):
            lo, hi = self.ci_68[i]
            lines.append(f"    {name:>10} = {self.median[i]:.4f} [{lo:.4f}, {hi:.4f}]")

        if self.bayes_factors:
            lines.append("  Bayes factors (ln B):")
            for mol, bf in self.bayes_factors.items():
                lines.append(f"    {mol}: ln B = {bf:.2f}")

        if self.upper_limits:
            lines.append("  95% upper limits (log VMR):")
            for mol, lim in self.upper_limits.items():
                lines.append(f"    {mol}: < {lim:.2f}")

        return "\n".join(lines)

    def corner_data(self) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Data for corner plot: (samples, labels, weights)."""
        return self.samples, self.param_names, self.weights

    def best_fit_spectrum(self, fixed: PlanetFixed) -> tuple[np.ndarray, np.ndarray]:
        """Compute best-fit model spectrum."""
        from exosphere.forward.model import ModelParams, compute_model_spectrum

        params = ModelParams(
            T=self.best_fit[0],
            log_h2o=self.best_fit[1],
            log_co2=self.best_fit[2],
            log_co=self.best_fit[3],
            log_ch4=self.best_fit[4],
            log_so2=self.best_fit[5],
            r_ref=self.best_fit[6],
            log_p_cloud=self.best_fit[7],
        )
        wl, depth, _ = compute_model_spectrum(params, fixed)
        return wl, depth

    def credible_band_spectrum(
        self, fixed: PlanetFixed, n_draws: int = 100
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute credible band: median, 16th, 84th percentiles of model spectra.

        Draws from posterior to compute model spectra.
        """
        from exosphere.forward.model import ModelParams, compute_model_spectrum

        # Resample according to weights
        n_eff = int(1.0 / np.sum(self.weights**2))
        n_draws = min(n_draws, n_eff)
        indices = np.random.choice(len(self.samples), size=n_draws, p=self.weights, replace=False)

        spectra = []
        for idx in indices:
            params = ModelParams(
                T=self.samples[idx, 0],
                log_h2o=self.samples[idx, 1],
                log_co2=self.samples[idx, 2],
                log_co=self.samples[idx, 3],
                log_ch4=self.samples[idx, 4],
                log_so2=self.samples[idx, 5],
                r_ref=self.samples[idx, 6],
                log_p_cloud=self.samples[idx, 7],
            )
            _, depth, _ = compute_model_spectrum(params, fixed)
            spectra.append(depth)

        spectra = np.array(spectra)  # (n_draws, n_wl)
        median = np.median(spectra, axis=0)
        lo = np.percentile(spectra, 16, axis=0)
        hi = np.percentile(spectra, 84, axis=0)
        return median, lo, hi
