"""Data-quality assessment: suitability + per-molecule rating (AGENTS.md section 6).

Runs BEFORE retrieval. Produces a JSON-serializable :class:`QualityReport`
describing S/N, coverage, bad points, outliers, uncertainty sanity, resolving
power, an overall GOOD/LIMITED/POOR suitability and a per-molecule
GOOD/LIMITED/POOR rating from band coverage + in-band S/N.

S/N method (per-point):
    The "local continuum" of the transit depth is a running median over
    ``continuum_window_points`` neighbouring bins (NaN-safe; scipy median
    filter). The point deviation is ``|depth_i - continuum_i|`` and the point
    S/N is that deviation divided by the point uncertainty ``sigma_i``.
    The report contains the MEDIAN point S/N over all usable points, and a
    per-band S/N = 90th percentile of the point S/N inside each molecule's
    band windows (robust visibility of that band's strongest structure).
    All thresholds live in ``config/molecule_bands.yaml`` (DECISIONS.md).

Outlier method: robust MAD clip of each point against the linear prediction
from its two direct neighbours (see :func:`mad_outlier_mask`), so isolated
spikes are flagged while smooth spectral features survive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field
from scipy.ndimage import median_filter

from exosphere.core.config import PROJECT_ROOT
from exosphere.core.spectrum import Spectrum

Rating = Literal["GOOD", "LIMITED", "POOR"]

DEFAULT_QUALITY_CONFIG_PATH = PROJECT_ROOT / "config" / "molecule_bands.yaml"
_RATING_ORDER: dict[str, int] = {"POOR": 0, "LIMITED": 1, "GOOD": 2}
_MAD_CONSTANT = 1.4826  # 1/Phi^-1(0.75): MAD -> sigma for Gaussian data


def worst_rating(*ratings: str) -> Rating:
    """Overall rating = the worst of the given ratings."""
    return min(ratings, key=lambda r: _RATING_ORDER[r])  # type: ignore[return-value]


def rating_from_threshold(value: float, good: float, poor: float) -> Rating:
    """Map a bigger-is-better metric to GOOD/LIMITED/POOR."""
    if value >= good:
        return "GOOD"
    if value >= poor:
        return "LIMITED"
    return "POOR"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    """Molecule bands + quality thresholds loaded from the YAML config."""

    molecule_bands: dict[str, tuple[tuple[float, float], ...]]
    wavelength_range_um: tuple[float, float]
    thresholds: dict[str, float]
    continuum_window_points: int
    outlier_n_sigma: float
    path: Path
    version: str

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> QualityConfig:
        config_path = Path(path) if path is not None else DEFAULT_QUALITY_CONFIG_PATH
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "molecules" not in raw or "quality" not in raw:
            raise ValueError(f"invalid quality config at {config_path}")
        molecules: dict[str, tuple[tuple[float, float], ...]] = {}
        for name, windows in raw["molecules"].items():
            bands = tuple((float(lo), float(hi)) for lo, hi in windows)
            if any(lo >= hi for lo, hi in bands):
                raise ValueError(f"band window lo < hi required for {name}")
            molecules[str(name)] = bands
        quality = raw["quality"]
        thresholds: dict[str, float] = {}
        thresholds.update({f"coverage_{k}": float(v) for k, v in quality["overall"].items()})
        thresholds.update({f"molecule_{k}": float(v) for k, v in quality["molecule"].items()})
        thresholds.update({f"uncertainty_{k}": float(v) for k, v in quality["uncertainty"].items()})
        lo, hi = raw["wavelength_range_um"]
        return cls(
            molecule_bands=molecules,
            wavelength_range_um=(float(lo), float(hi)),
            thresholds=thresholds,
            continuum_window_points=int(quality["continuum_window_points"]),
            outlier_n_sigma=float(quality["outlier_n_sigma"]),
            path=config_path,
            version=str(raw.get("version", "unknown")),
        )

    @property
    def nominal_range(self) -> tuple[float, float]:
        return self.wavelength_range_um


_DEFAULT_CONFIG: QualityConfig | None = None


def load_quality_config(path: str | Path | None = None, *, refresh: bool = False) -> QualityConfig:
    """Load (and cache) the quality/molecule-bands configuration."""
    global _DEFAULT_CONFIG
    if path is not None:
        return QualityConfig.from_yaml(path)
    if _DEFAULT_CONFIG is None or refresh:
        _DEFAULT_CONFIG = QualityConfig.from_yaml()
    return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Shared statistics (also used by exosphere.preprocess.clean)
# ---------------------------------------------------------------------------


def running_median(values: np.ndarray, window_points: int) -> np.ndarray:
    """NaN-safe running median; non-finite inputs give NaN outputs.

    The window is forced odd and clipped to the array length (>= 1).
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return v.copy()
    window = max(1, int(window_points))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(1, window)
    out = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(v)
    if not bool(np.any(finite)):
        return out
    if int(finite.sum()) < window:
        out[finite] = np.median(v[finite])
        return out
    out[finite] = median_filter(v[finite], size=window, mode="nearest")
    return out


def mad_outlier_mask(
    wavelength: np.ndarray,
    depth: np.ndarray,
    sigma: np.ndarray,
    n_sigma: float,
    max_gap_factor: float = 3.0,
) -> np.ndarray:
    """Robust MAD-based spike/outlier mask (used by assess and preprocess.clean).

    For every interior point the local linear prediction from its two direct
    neighbours is subtracted:

        r_i = d_i - (d_{i-1} + d_{i+1}) / 2
        sigma_eff_i = sqrt(sigma_i^2 + (sigma_{i-1}^2 + sigma_{i+1}^2)/4)
        z_i = r_i / sigma_eff_i

    A point that is smooth relative to its neighbours (a real spectral feature)
    has z ~ 0 no matter how strong the feature is; an isolated spike does not.
    Outliers are ``|z - median(z)| > n_sigma * 1.4826 * MAD(z)`` computed over
    the interior points, so the threshold itself is robust. If the MAD collapses
    to zero the nominal sigma (robust scale 1) is used instead. Endpoints and
    points whose neighbour spacing exceeds ``max_gap_factor`` x the median
    sampling gap (e.g. across a wavelength gap) are never flagged, since they
    have no reliable local prediction.
    """
    w = np.asarray(wavelength, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    n = d.size
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return mask
    finite = np.isfinite(w) & np.isfinite(d) & np.isfinite(s) & (s > 0)
    local_ok = finite[:-2] & finite[1:-1] & finite[2:]

    gaps = np.diff(w)
    positive_gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if positive_gaps.size:
        median_gap = float(np.median(positive_gaps))
        gap_l = w[1:-1] - w[:-2]
        gap_r = w[2:] - w[1:-1]
        local_ok = (
            local_ok
            & (gap_l > 0)
            & (gap_r > 0)
            & (gap_l <= max_gap_factor * median_gap)
            & (gap_r <= max_gap_factor * median_gap)
        )
    if not bool(np.any(local_ok)):
        return mask

    residual = d[1:-1] - 0.5 * (d[:-2] + d[2:])
    sigma_eff = np.sqrt(np.square(s[1:-1]) + 0.25 * (np.square(s[:-2]) + np.square(s[2:])))
    z = np.full(n - 2, np.nan, dtype=np.float64)
    valid = local_ok & (sigma_eff > 0)
    z[valid] = residual[valid] / sigma_eff[valid]
    z_valid = z[valid]
    centre = float(np.median(z_valid))
    mad = float(np.median(np.abs(z_valid - centre)))
    # The threshold is never tighter than n_sigma in units of the QUOTED
    # uncertainty: MAD only raises it when the data scatter exceeds sigma, but
    # must not shrink it for spectra that are smoother than their error bars
    # (otherwise a noiseless smooth spectrum would be over-clipped).
    robust_sigma = max(_MAD_CONSTANT * mad, 1.0)
    if not np.isfinite(robust_sigma):
        robust_sigma = 1.0
    local_outliers = np.abs(z - centre) > float(n_sigma) * robust_sigma
    local_outliers[~valid] = False
    mask[1:-1] = local_outliers
    return mask


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class BandQuality(BaseModel):
    """Quality of one band window of one molecule."""

    wavelength_range_um: tuple[float, float]
    coverage_fraction: float
    n_points: int
    snr: float | None = None  # 90th percentile of point S/N inside the window


class MoleculeQuality(BaseModel):
    molecule: str
    rating: Rating
    coverage_fraction: float
    n_points: int
    snr: float | None = None  # 90th percentile of point S/N across all windows
    bands: list[BandQuality] = Field(default_factory=list)


class UncertaintySanity(BaseModel):
    median: float | None
    non_positive_count: int
    nan_count: int
    huge_count: int
    tiny_count: int

    @property
    def bad_count(self) -> int:
        return self.non_positive_count + self.nan_count + self.huge_count + self.tiny_count


class QualityReport(BaseModel):
    """JSON-serializable data-quality report (AGENTS.md section 6)."""

    observation_id: str
    target_id: str
    instrument: str
    n_points: int
    wavelength_min_um: float | None
    wavelength_max_um: float | None
    wavelength_coverage_fraction: float
    median_snr: float | None
    max_band_snr: float | None = None
    band_snr: dict[str, float | None] = Field(default_factory=dict)
    flagged_fraction: float
    nan_fraction: float
    bad_point_fraction: float
    outlier_count: int
    outlier_fraction: float
    uncertainty: UncertaintySanity
    effective_resolving_power: float | None
    suitability: Rating
    molecule_ratings: dict[str, Rating] = Field(default_factory=dict)
    molecules: list[MoleculeQuality] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    config_version: str
    method: dict[str, str] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> QualityReport:
        return cls.model_validate_json(payload)

    def summary_text(self) -> str:
        """Human-readable multi-line summary (used by the CLI script)."""
        lines = [
            f"target={self.target_id} observation={self.observation_id} "
            f"instrument={self.instrument}",
            f"points={self.n_points} range="
            f"{self.wavelength_min_um}-{self.wavelength_max_um} um "
            f"coverage={self.wavelength_coverage_fraction:.3f} "
            f"R_eff={self.effective_resolving_power:.1f}"
            if self.wavelength_min_um is not None and self.effective_resolving_power
            else f"points={self.n_points}",
            f"median S/N={self.median_snr:.2f}"
            if self.median_snr is not None
            else "median S/N=n/a",
            f"flagged={self.flagged_fraction:.3f} nan={self.nan_fraction:.3f} "
            f"outliers={self.outlier_count} ({self.outlier_fraction:.3f})",
            f"uncertainty: median={self.uncertainty.median} "
            f"non-positive={self.uncertainty.non_positive_count} "
            f"nan={self.uncertainty.nan_count} huge={self.uncertainty.huge_count} "
            f"tiny={self.uncertainty.tiny_count}",
            f"overall suitability: {self.suitability}",
            "per-molecule:",
        ]
        for molecule in self.molecules:
            snr = "n/a" if molecule.snr is None else f"{molecule.snr:.2f}"
            lines.append(
                f"  {molecule.molecule:<4} {molecule.rating:<7} "
                f"coverage={molecule.coverage_fraction:.3f} points={molecule.n_points} "
                f"band S/N={snr}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


# Sentinel quality-flag values that mean "this point is fine". Anything else
# counts as flagged. The Phase 1 archive loaders write "OK" for good points.
_PASS_FLAGS = {"", "ok"}


def is_flagged(flag: str) -> bool:
    """True when a quality flag marks the point as bad."""
    return str(flag).strip().lower() not in _PASS_FLAGS


def _usable_mask(spectrum: Spectrum) -> np.ndarray:
    """Points usable for S/N and coverage: finite depth, sigma > 0, unflagged."""
    depth = np.asarray(spectrum.transmission, dtype=np.float64)
    sigma = np.asarray(spectrum.uncertainty, dtype=np.float64)
    flags = spectrum.quality_flags
    flagged = (
        np.array([is_flagged(flag) for flag in flags], dtype=bool)
        if flags
        else np.zeros(depth.shape, dtype=bool)
    )
    return np.isfinite(depth) & np.isfinite(sigma) & (sigma > 0) & ~flagged


def _band_coverage(
    bin_lo: np.ndarray, bin_hi: np.ndarray, usable: np.ndarray, window: tuple[float, float]
) -> float:
    """Fraction of a band window covered by usable bin widths."""
    lo, hi = window
    width = hi - lo
    if width <= 0:
        return 0.0
    overlap = np.clip(np.minimum(bin_hi, hi) - np.maximum(bin_lo, lo), 0.0, None)
    covered = float(np.sum(overlap[usable]))
    return float(min(1.0, max(0.0, covered / width)))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _band_snr(values: np.ndarray) -> float | None:
    """Band S/N = 90th percentile of the point S/N inside a band window.

    A robust estimate of the visibility of the band's strongest structure: the
    median over the whole window would be dominated by featureless points, while
    a bare maximum would be a single noise fluctuation.
    """
    if values.size == 0:
        return None
    return float(np.percentile(values, 90))


def assess(spectrum: Spectrum, config: QualityConfig | None = None) -> QualityReport:
    """Assess data quality of a spectrum (never mutates it)."""
    cfg = config if config is not None else load_quality_config()
    t = cfg.thresholds

    depth = np.asarray(spectrum.transmission, dtype=np.float64)
    sigma = np.asarray(spectrum.uncertainty, dtype=np.float64)
    wave = np.asarray(spectrum.wavelength, dtype=np.float64)
    edges = np.asarray(spectrum.wavelength_bin_edges, dtype=np.float64)
    n_points = int(wave.size)

    usable = _usable_mask(spectrum)
    n_usable = int(usable.sum())
    n_flagged = (
        int(sum(1 for flag in spectrum.quality_flags if is_flagged(flag)))
        if spectrum.quality_flags
        else 0
    )
    n_nan = int(np.sum(~np.isfinite(depth) | ~np.isfinite(sigma)))
    nan_fraction = float(n_nan / n_points) if n_points else 0.0
    flagged_fraction = float(n_flagged / n_points) if n_points else 0.0

    # --- S/N against the local continuum --------------------------------
    continuum = running_median(depth, cfg.continuum_window_points)
    deviation = np.abs(depth - continuum)
    with np.errstate(divide="ignore", invalid="ignore"):
        point_snr = np.where(usable, deviation / np.where(usable, sigma, np.nan), np.nan)
    usable_snr = point_snr[usable]
    median_snr = float(np.median(usable_snr)) if usable_snr.size else None

    # --- outliers (same robust MAD rule as preprocess.clean) -------------
    outlier_mask = mad_outlier_mask(wave, depth, sigma, cfg.outlier_n_sigma)
    outliers_usable = int(np.sum(outlier_mask & usable))
    outlier_fraction = float(outliers_usable / n_usable) if n_usable else 0.0

    # --- uncertainty sanity ---------------------------------------------
    finite_depth = depth[np.isfinite(depth)]
    median_depth = float(np.median(finite_depth)) if finite_depth.size else None
    finite_sigma = sigma[np.isfinite(sigma)]
    median_sigma = float(np.median(finite_sigma)) if finite_sigma.size else None
    huge_limit = (
        float(t["uncertainty_max_frac_of_depth"]) * abs(median_depth)
        if median_depth is not None and median_depth != 0
        else None
    )
    tiny_limit = float(t["uncertainty_min_fractional"])
    uncertainty = UncertaintySanity(
        median=_finite_or_none(median_sigma) if median_sigma is not None else None,
        non_positive_count=int(np.sum(np.isfinite(sigma) & (sigma <= 0))),
        nan_count=int(np.sum(~np.isfinite(sigma))),
        huge_count=(
            int(np.sum(np.isfinite(sigma) & (sigma > huge_limit)))
            if huge_limit is not None and huge_limit > 0
            else 0
        ),
        tiny_count=int(np.sum(np.isfinite(sigma) & (sigma > 0) & (sigma < tiny_limit))),
    )
    uncertainty_bad_fraction = float(uncertainty.bad_count / n_points) if n_points else 1.0

    # --- overall wavelength coverage ------------------------------------
    nominal_lo, nominal_hi = cfg.nominal_range
    # Coverage bins come from the local point spacing, not from
    # wavelength_bin_edges: the edges must tile the full range (N+1 values), so
    # a bin spanning an internal wavelength gap would otherwise claim coverage
    # of the gap itself.
    if n_points >= 2:
        gaps = np.diff(wave)
        point_widths = np.empty(n_points, dtype=np.float64)
        point_widths[0] = gaps[0]
        point_widths[-1] = gaps[-1]
        if n_points > 2:
            point_widths[1:-1] = np.minimum(gaps[:-1], gaps[1:])
    else:
        point_widths = np.array(
            [float(edges[1] - edges[0]) if edges.size == 2 else 1.0],
            dtype=np.float64,
        )
    bin_lo = wave - point_widths / 2.0
    bin_hi = wave + point_widths / 2.0
    overall_coverage = _band_coverage(bin_lo, bin_hi, usable, (nominal_lo, nominal_hi))
    if n_points == 0:
        overall_coverage = 0.0

    # --- effective resolving power R = lambda / dlambda (bin widths) -----
    edge_widths = edges[1:] - edges[:-1] if edges.size >= 2 else point_widths
    valid_r = (edge_widths > 0) & np.isfinite(edge_widths) & (wave > 0)
    resolving_power = (
        float(np.median(wave[valid_r] / edge_widths[valid_r])) if bool(np.any(valid_r)) else None
    )

    # --- per-molecule rating --------------------------------------------
    molecules: list[MoleculeQuality] = []
    band_snr: dict[str, float | None] = {}
    for molecule, windows in cfg.molecule_bands.items():
        band_qualities: list[BandQuality] = []
        band_points: list[np.ndarray] = []
        for window in windows:
            in_band = usable & (wave >= window[0]) & (wave <= window[1])
            band_points.append(in_band)
            n_in_band = int(in_band.sum())
            band_snr_values = point_snr[in_band]
            band_snr_values = band_snr_values[np.isfinite(band_snr_values)]
            band_qualities.append(
                BandQuality(
                    wavelength_range_um=window,
                    coverage_fraction=_band_coverage(bin_lo, bin_hi, usable, window),
                    n_points=n_in_band,
                    snr=_band_snr(band_snr_values),
                )
            )
        # Coverage = usable width over ALL band windows (sum of per-window
        # covered width / sum of window widths) so narrow windows are weighted
        # by their bandwidth, not by count.
        total_width = sum(window[1] - window[0] for window in windows)
        covered_width = sum(
            b.coverage_fraction * (window[1] - window[0])
            for b, window in zip(band_qualities, windows, strict=True)
        )
        coverage = float(covered_width / total_width) if total_width > 0 else 0.0
        all_band_points = np.logical_or.reduce(band_points) if band_points else usable & False
        in_band_snr = point_snr[all_band_points]
        in_band_snr = in_band_snr[np.isfinite(in_band_snr)]
        n_points_in_bands = int(np.sum(all_band_points))
        molecule_snr = _band_snr(in_band_snr)
        band_snr[molecule] = molecule_snr

        min_points_limited = int(t["molecule_min_points_limited"])
        min_points_good = int(t["molecule_min_points_good"])
        if (
            coverage < float(t["molecule_coverage_limited"])
            or n_points_in_bands < min_points_limited
            or molecule_snr is None
            or molecule_snr < float(t["molecule_snr_limited"])
        ):
            rating: Rating = "POOR"
        elif (
            coverage < float(t["molecule_coverage_good"])
            or molecule_snr < float(t["molecule_snr_good"])
            or n_points_in_bands < min_points_good
        ):
            rating = "LIMITED"
        else:
            rating = "GOOD"
        molecules.append(
            MoleculeQuality(
                molecule=molecule,
                rating=rating,
                coverage_fraction=coverage,
                n_points=n_points_in_bands,
                snr=molecule_snr,
                bands=band_qualities,
            )
        )

    # --- overall suitability (worst metric wins) ------------------------
    best_band_snr = max((value for value in band_snr.values() if value is not None), default=None)
    bad_point_fraction = float(min(1.0, flagged_fraction + nan_fraction))
    metric_ratings = [
        rating_from_threshold(
            overall_coverage,
            float(t["coverage_wavelength_coverage_good"]),
            float(t["coverage_wavelength_coverage_poor"]),
        ),
        rating_from_threshold(
            best_band_snr if best_band_snr is not None else 0.0,
            float(t["coverage_snr_good"]),
            float(t["coverage_snr_poor"]),
        ),
        rating_from_threshold(
            1.0 - bad_point_fraction,
            float(t["coverage_bad_point_fraction_good"]),
            float(t["coverage_bad_point_fraction_poor"]),
        ),
        rating_from_threshold(
            1.0 - outlier_fraction,
            float(t["coverage_outlier_fraction_good"]),
            float(t["coverage_outlier_fraction_poor"]),
        ),
        rating_from_threshold(
            1.0 - uncertainty_bad_fraction,
            1.0 - float(t["coverage_bad_point_fraction_good"]),
            1.0 - float(t["coverage_bad_point_fraction_poor"]),
        ),
        rating_from_threshold(
            float(n_points),
            float(t["coverage_min_points_good"]),
            float(t["coverage_min_points_poor"]),
        ),
    ]
    suitability = worst_rating(*metric_ratings)

    return QualityReport(
        observation_id=spectrum.observation_id,
        target_id=spectrum.target_id,
        instrument=spectrum.instrument,
        n_points=n_points,
        wavelength_min_um=float(wave[0]) if n_points else None,
        wavelength_max_um=float(wave[-1]) if n_points else None,
        wavelength_coverage_fraction=float(overall_coverage),
        median_snr=_finite_or_none(median_snr) if median_snr is not None else None,
        max_band_snr=_finite_or_none(best_band_snr) if best_band_snr is not None else None,
        band_snr=band_snr,
        flagged_fraction=flagged_fraction,
        nan_fraction=nan_fraction,
        bad_point_fraction=float(bad_point_fraction),
        outlier_count=outliers_usable,
        outlier_fraction=outlier_fraction,
        uncertainty=uncertainty,
        effective_resolving_power=_finite_or_none(resolving_power)
        if resolving_power is not None
        else None,
        suitability=suitability,
        molecule_ratings={m.molecule: m.rating for m in molecules},
        molecules=molecules,
        thresholds=dict(cfg.thresholds),
        config_version=cfg.version,
        method={
            "snr_point": "point S/N = |depth - running_median(depth)| / sigma; "
            f"window {cfg.continuum_window_points} pts",
            "snr_band": "band S/N = 90th percentile of point S/N across the "
            "molecule's windows (robust peak-feature visibility)",
            "outliers": f"MAD spike-clip at {cfg.outlier_n_sigma} sigma: point vs "
            "linear prediction from its two direct neighbours",
            "coverage": "usable bin widths inside the window / window width",
            "suitability": "worst metric across coverage, best-band S/N, bad "
            "points, outliers, uncertainty sanity, point count",
            "molecule_rating": "data capability in the molecule's windows "
            "(coverage + structure above noise) - NOT molecule presence",
        },
    )


def report_to_json(report: QualityReport, path: str | Path) -> Path:
    """Write a QualityReport to a JSON file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = json.loads(report.to_json())
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path
