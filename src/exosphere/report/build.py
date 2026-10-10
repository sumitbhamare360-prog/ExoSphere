"""Scientific report generator (Phase 9a).

``build_report(analysis_id, format="html"|"pdf")`` renders a self-contained
scientific report from the *stored* analysis products (database rows + spectrum
/ posterior files). It performs no new science except cheap plotting work:
best-fit / credible-band model spectra evaluated from the stored posterior,
chi-square arithmetic, and matplotlib figures embedded as base64 PNGs.

Report version: ``report-1.0.0`` (see ``REPORT_VERSION``).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from exosphere.api.db import (
    Analysis,
    DetectionResult,
    Posterior,
    Report,
    async_session_maker,
)
from exosphere.api.db import (
    MLResult as DBMLResult,
)
from exosphere.api.db import (
    QualityReport as DBQualityReport,
)
from exosphere.core.config import load_config
from exosphere.core.spectrum import Spectrum
from exosphere.forward.model import PlanetFixed
from exosphere.retrieval.results import RetrievalResult

REPORT_VERSION = "report-1.0.0"
REPORT_TYPE = "scientific"
TEMPLATE_NAME = "report.html"

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]

# Interpretation thresholds (documented in DECISIONS.md item 41).
LN_B_SUPPORTED = 5.0  # ln B >= 5: strong support
LN_B_WEAK = 3.0  # 3 <= ln B < 5: weak support

# Display metadata for the 8 retrieval parameters: (label, unit).
PARAM_DISPLAY: dict[str, tuple[str, str]] = {
    "T": ("Isothermal atmosphere temperature", "K"),
    "log_h2o": ("H2O log10 volume mixing ratio", "dex"),
    "log_co2": ("CO2 log10 volume mixing ratio", "dex"),
    "log_co": ("CO log10 volume mixing ratio", "dex"),
    "log_ch4": ("CH4 log10 volume mixing ratio", "dex"),
    "log_so2": ("SO2 log10 volume mixing ratio", "dex"),
    "r_ref": ("Reference radius at 0.01 bar", "R_Jup"),
    "log_p_cloud": ("Cloud-top pressure, log10", "log10(bar)"),
}

PARAM_ORDER = ["T", "log_h2o", "log_co2", "log_co", "log_ch4", "log_so2", "r_ref", "log_p_cloud"]

SECTION_IDS = [
    "sec-title",
    "sec-data",
    "sec-quality",
    "sec-preprocess",
    "sec-ml",
    "sec-setup",
    "sec-results",
    "sec-interpretation",
    "sec-twin",
    "sec-provenance",
]

_ML_SCORE_LABEL = "ML candidate score (dimensionless 0-1; not an abundance, not a detection)"


class ReportError(Exception):
    """Base class for report errors."""


class ReportNotFoundError(ReportError):
    """Raised when the analysis (or a required product) is not stored."""


class ReportWordingError(ReportError):
    """Raised when the rendered report violates the strict wording rules."""


class PDFUnavailableError(ReportError):
    """Raised when PDF rendering was requested but is unavailable here."""


# ---------------------------------------------------------------------------
# Wording guard
# ---------------------------------------------------------------------------

# Global bans: never allowed anywhere in the document.
_BANNED_GLOBAL = [
    r"\bhabit\w*",
    r"\bbiosignatur\w*",
    r"\blife\b",
    r"\baliens?\b",
]

# Banned inside the ML section and the generated summary: existence/detection
# claims must never be driven by ML candidate scores.
_BANNED_ML_DRIVEN = [
    r"\bis present\b",
    r"\bare present\b",
    r"\bexists?\b",
    r"\bdetected\b",
    r"\bdiscovered\b",
    r"\bconfirmed\b",
    r"\bfound\b",
    r"\bpresence\b",
]

_BANNED_GLOBAL_RX = [re.compile(p, re.IGNORECASE) for p in _BANNED_GLOBAL]
_BANNED_ML_RX = [re.compile(p, re.IGNORECASE) for p in _BANNED_ML_DRIVEN]


def _section_html(html: str, section_id: str) -> str:
    """Extract the inner HTML of a ``<section id=...>`` block."""
    match = re.search(
        rf'<section[^>]*id="{section_id}"[^>]*>(.*?)</section>', html, re.DOTALL | re.IGNORECASE
    )
    return match.group(1) if match else ""


def check_wording(html: str) -> list[str]:
    """Check the strict wording rules; return a list of violations (empty if clean).

    - Global bans (life/habitability claims) apply to the whole document.
    - Existence/detection language is banned in the ML section and in the
      generated summary, where only support/consistency language is allowed.
    - Every ML section must label its numbers with the mandatory score label.
    """
    violations: list[str] = []
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    for rx in _BANNED_GLOBAL_RX:
        match = rx.search(text)
        if match:
            violations.append(f"global ban violated: {match.group(0)!r}")
    for section_id in ("sec-ml", "sec-title"):
        section_text = re.sub(r"<[^>]+>", " ", _section_html(html, section_id))
        section_text = re.sub(r"\s+", " ", section_text)
        for rx in _BANNED_ML_RX:
            match = rx.search(section_text)
            if match:
                violations.append(f"{section_id} ban violated: {match.group(0)!r}")
    ml_section = _section_html(html, "sec-ml")
    if "not run" not in ml_section.lower():
        label_count = ml_section.count("ML candidate score")
        if label_count < len(MOLECULES) + 1:
            violations.append(
                f"ML section labels every number: found {label_count} "
                f"'ML candidate score' labels, need >= {len(MOLECULES) + 1}"
            )
    return violations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 4) -> str:
    """Compact number formatting for the report (units are shown separately)."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    if number != 0.0 and (abs(number) >= 1e6 or abs(number) < 1e-3):
        return f"{number:.3e}"
    return f"{number:.{digits}g}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fig_to_data_uri(fig: Any) -> str:
    """Render a matplotlib figure to a base64 PNG data URI."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=80, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def _molecule_bands() -> dict[str, list[tuple[float, float]]]:
    """Band windows from the auditable quality config (read-only)."""
    from exosphere.quality.assess import load_quality_config

    config = load_quality_config()
    bands: dict[str, list[tuple[float, float]]] = {}
    for molecule in MOLECULES:
        windows = config.molecule_bands.get(molecule, [])
        bands[molecule] = [(float(lo), float(hi)) for lo, hi in windows]
    return bands


# ---------------------------------------------------------------------------
# Data gathering (stored products only)
# ---------------------------------------------------------------------------


async def _load_stored(analysis_id: str) -> dict[str, Any]:
    """Load every stored product for the analysis (DB rows + files)."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(Analysis)
            .options(
                selectinload(Analysis.planet),
                selectinload(Analysis.observation),
                selectinload(Analysis.spectrum_file),
            )
            .where(Analysis.analysis_id == analysis_id)
        )
        analysis = result.scalar_one_or_none()
        if analysis is None:
            raise ReportNotFoundError(f"analysis {analysis_id} not found")

        stored: dict[str, Any] = {"analysis": analysis}
        stored["planet"] = analysis.planet
        stored["observation"] = analysis.observation
        stored["spectrum_file"] = analysis.spectrum_file
        stored["provenance"] = dict(analysis.provenance_json or {})
        stored["config"] = dict(analysis.config_json or {})

        async def latest(model: Any, order_col: Any = None) -> Any:
            query = select(model).where(model.analysis_id == analysis.id)
            query = query.order_by(desc(order_col if order_col is not None else model.id))
            rows = await session.execute(query)
            return rows.scalars().first()

        stored["quality_row"] = await latest(DBQualityReport)
        stored["ml_row"] = await latest(DBMLResult)
        stored["posterior_row"] = await latest(Posterior)
        det_result = await session.execute(
            select(DetectionResult)
            .where(DetectionResult.analysis_id == analysis.id)
            .order_by(desc(DetectionResult.id))
        )
        stored["detection_rows"] = list(det_result.scalars().all())

    # Spectrum file (cleaned product linked on the analysis, else the
    # observation source file).
    spec_path: str | None = None
    if stored["spectrum_file"] is not None:
        spec_path = stored["spectrum_file"].file_path
    observation = stored["observation"]
    if spec_path is None and observation is not None and observation.spectrum_file_path:
        spec_path = observation.spectrum_file_path
    spectrum: Spectrum | None = None
    if spec_path and Path(spec_path).exists():
        spectrum = Spectrum.load(spec_path)
    stored["spectrum"] = spectrum
    stored["spectrum_path"] = spec_path

    # Preprocess log (written by the pipeline since Phase 9a; older runs lack it).
    config = load_config()
    log_path = config.data_cache_dir / "preprocess_logs" / f"{analysis_id}.json"
    stored["preprocess_log"] = None
    if log_path.exists():
        import json

        stored["preprocess_log"] = json.loads(log_path.read_text(encoding="utf-8"))

    # Posterior numpy archive (holds samples + 95% intervals).
    stored["retrieval_result"] = None
    posterior_row = stored["posterior_row"]
    if posterior_row is not None and posterior_row.file_path:
        npz_path = Path(posterior_row.file_path)
        if npz_path.exists():
            stored["retrieval_result"] = RetrievalResult.load_npz(npz_path)

    return stored


def _fixed_from_planet(planet: Any) -> PlanetFixed | None:
    if planet is None:
        return None
    gravity = getattr(planet, "surface_gravity_m_s2", None)
    stellar_radius = getattr(planet, "st_rad", None)
    if gravity is None or stellar_radius is None:
        return None
    return PlanetFixed(
        gravity_m_s2=float(gravity),
        stellar_radius_rsun=float(stellar_radius),
        reference_pressure_bar=0.01,
    )


# ---------------------------------------------------------------------------
# Figures (cheap plotting from stored products only)
# ---------------------------------------------------------------------------

_MOLECULE_COLORS = {
    "H2O": "#3b82f6",
    "CO2": "#ef4444",
    "CO": "#f97316",
    "CH4": "#8b5cf6",
    "SO2": "#f59e0b",
}


def _figure_spectrum(
    spectrum: Spectrum,
    model: dict[str, Any],
    bands: dict[str, list[tuple[float, float]]],
) -> Any:
    """Observed spectrum with best fit, 68% band and molecule-band shading."""
    import matplotlib.pyplot as plt

    wl = np.asarray(spectrum.wavelength, dtype=float)
    depth = np.asarray(spectrum.transmission, dtype=float)
    sigma = np.asarray(spectrum.uncertainty, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for molecule in MOLECULES:
        for lo, hi in bands.get(molecule, []):
            ax.axvspan(lo, hi, color=_MOLECULE_COLORS[molecule], alpha=0.08, linewidth=0)
    ax.errorbar(
        wl, depth * 1e6, yerr=sigma * 1e6, fmt=".", ms=2, color="black",
        ecolor="gray", elinewidth=0.6, alpha=0.8, label="Observed transit depth",
    )
    if model.get("wavelength") is not None:
        mw = np.asarray(model["wavelength"], dtype=float)
        ax.plot(mw, np.asarray(model["best"], dtype=float) * 1e6, color="#6366f1",
                linewidth=1.2, label="Best-fit model")
        if model.get("lo") is not None:
            ax.fill_between(
                mw,
                np.asarray(model["lo"], dtype=float) * 1e6,
                np.asarray(model["hi"], dtype=float) * 1e6,
                color="#6366f1", alpha=0.2, label="68% credible band",
            )
    else:
        ax.text(0.02, 0.95, "Best-fit model: not run", transform=ax.transAxes,
                fontsize=9, verticalalignment="top")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Transit depth (ppm)")
    ax.set_title("Transmission spectrum with best-fit model")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _figure_corner(result: RetrievalResult) -> Any:
    """Compact corner plot: 1-D marginals on the diagonal, scatter below."""
    import matplotlib.pyplot as plt

    samples, names, weights = result.corner_data()
    n = len(names)
    order = np.argsort(np.asarray(weights, dtype=float))
    if len(order) > 3000:
        step = max(1, len(order) // 3000)
        order = order[::step]
    thin = np.asarray(samples, dtype=float)[order]

    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(thin[:, i], bins=30, color="#6366f1", alpha=0.8)
                ax.set_title(names[i], fontsize=8)
            elif i > j:
                ax.scatter(thin[:, j], thin[:, i], s=1, alpha=0.3, color="#1f2937")
            else:
                ax.axis("off")
                continue
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=7)
            if j == 0 and i != 0:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Posterior samples (thinned for display)")
    fig.tight_layout()
    return fig


def _figure_residuals(
    spectrum: Spectrum, model_depth: np.ndarray | None, chi2: float | None, dof: int | None
) -> Any:
    """Normalized residuals (observed minus best fit, in units of sigma)."""
    import matplotlib.pyplot as plt

    wl = np.asarray(spectrum.wavelength, dtype=float)
    depth = np.asarray(spectrum.transmission, dtype=float)
    sigma = np.asarray(spectrum.uncertainty, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    if model_depth is not None:
        resid = (depth - np.asarray(model_depth, dtype=float)) / sigma
        ax.scatter(wl, resid, s=6, color="#1f2937", alpha=0.8)
        for level, style in ((1.0, "--"), (3.0, ":")):
            ax.axhline(level, color="gray", linestyle=style, linewidth=0.8)
            ax.axhline(-level, color="gray", linestyle=style, linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        title = "Normalized residuals (sigma units)"
        if chi2 is not None and dof:
            title += f" — chi2 = {chi2:.1f}, dof = {dof}, reduced chi2 = {chi2 / dof:.2f}"
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "Residuals: retrieval not run", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Residual (sigma)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


def _quality_section(quality_row: Any) -> dict[str, Any]:
    if quality_row is None:
        return {"ran": False}
    report = dict(quality_row.report_json or {})
    molecules = report.get("molecules", []) or []
    per_molecule = []
    for entry in molecules:
        per_molecule.append(
            {
                "molecule": entry.get("molecule", "?"),
                "rating": entry.get("rating", "POOR"),
                "coverage_fraction": entry.get("coverage_fraction"),
                "n_points": entry.get("n_points"),
                "snr": entry.get("snr"),
            }
        )
    uncertainty = report.get("uncertainty", {}) or {}
    return {
        "ran": True,
        "suitability": report.get("suitability", "POOR"),
        "n_points": report.get("n_points"),
        "wavelength_min_um": report.get("wavelength_min_um"),
        "wavelength_max_um": report.get("wavelength_max_um"),
        "wavelength_coverage_fraction": report.get("wavelength_coverage_fraction"),
        "median_snr": report.get("median_snr"),
        "max_band_snr": report.get("max_band_snr"),
        "band_snr": report.get("band_snr", {}) or {},
        "flagged_fraction": report.get("flagged_fraction"),
        "nan_fraction": report.get("nan_fraction"),
        "outlier_count": report.get("outlier_count"),
        "outlier_fraction": report.get("outlier_fraction"),
        "uncertainty": {
            "median": uncertainty.get("median"),
            "non_positive_count": uncertainty.get("non_positive_count"),
            "nan_count": uncertainty.get("nan_count"),
            "huge_count": uncertainty.get("huge_count"),
            "tiny_count": uncertainty.get("tiny_count"),
        },
        "effective_resolving_power": report.get("effective_resolving_power"),
        "per_molecule": per_molecule,
        "molecule_ratings": report.get("molecule_ratings", {}) or {},
        "config_version": report.get("config_version"),
    }


def _preprocess_section(log: dict[str, Any] | None) -> dict[str, Any]:
    if log is None:
        return {"ran": False}
    removed = log.get("removed", []) or []
    counts: dict[str, int] = {}
    for item in removed:
        reason = item.get("reason", "unknown") if isinstance(item, dict) else "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    rebin = log.get("rebin")
    return {
        "ran": True,
        "version": log.get("preprocessing_version"),
        "n_input": log.get("n_input"),
        "n_output": log.get("n_output"),
        "parameters": log.get("parameters", {}) or {},
        "removal_counts": counts,
        "n_removed": len(removed),
        "rebin": rebin,
    }


def _ml_section(ml_row: Any) -> dict[str, Any]:
    if ml_row is None:
        return {"ran": False}
    scores = dict(ml_row.scores_json or {})
    rows = [
        {"molecule": molecule, "score": scores.get(molecule)}
        for molecule in MOLECULES
    ]
    return {
        "ran": True,
        "scores": rows,
        "score_label": _ML_SCORE_LABEL,
        "model_version": ml_row.model_version,
        "dataset_hash": ml_row.dataset_hash,
        "model_config_hash": ml_row.model_config_hash,
    }


def _retrieval_section(
    posterior_row: Any, result: RetrievalResult | None, seed: int
) -> dict[str, Any]:
    if posterior_row is None:
        return {"ran": False}
    summary = dict(posterior_row.summary_json or {})
    median = summary.get("median", {}) or {}
    ci_68 = summary.get("ci_68", {}) or {}
    best_fit = summary.get("best_fit", {}) or {}
    ci_95: dict[str, list[float]] = {}
    if result is not None:
        for i, name in enumerate(result.param_names):
            ci_95[name] = [float(result.ci_95[i, 0]), float(result.ci_95[i, 1])]
    rows = []
    names = summary.get("param_names") or PARAM_ORDER
    for name in names:
        label, unit = PARAM_DISPLAY.get(name, (name, ""))
        interval68 = ci_68.get(name)
        interval95 = ci_95.get(name)
        rows.append(
            {
                "name": name,
                "label": label,
                "unit": unit,
                "median": median.get(name),
                "ci68_lo": interval68[0] if interval68 else None,
                "ci68_hi": interval68[1] if interval68 else None,
                "ci95_lo": interval95[0] if interval95 else None,
                "ci95_hi": interval95[1] if interval95 else None,
                "best_fit": best_fit.get(name),
            }
        )
    return {
        "ran": True,
        "rows": rows,
        "logz": posterior_row.logz,
        "logz_err": posterior_row.logz_err,
        "n_samples": posterior_row.n_samples,
        "n_live": posterior_row.n_live,
        "dlogz": posterior_row.dlogz,
        "seed": seed,
        "sampler": "dynesty",
        "n_params": 8,
    }


def _detection_section(
    detection_rows: list[Any], quality_ratings: dict[str, str]
) -> dict[str, Any]:
    if not detection_rows:
        return {"ran": False}
    by_molecule = {row.molecule: row for row in detection_rows}
    rows = []
    for molecule in MOLECULES:
        row = by_molecule.get(molecule)
        if row is None:
            rows.append({"molecule": molecule, "ran": False})
            continue
        ln_b = row.ln_bayes_factor
        rating = quality_ratings.get(molecule, "POOR")
        if rating == "POOR":
            verdict = "not constrained (POOR data quality)"
            verdict_class = "not-constrained"
        elif ln_b is None:
            verdict = "not assessed (no Bayes factor stored)"
            verdict_class = "not-constrained"
        elif ln_b >= LN_B_SUPPORTED:
            verdict = f"supported (ln B = {ln_b:.2f} >= {LN_B_SUPPORTED:.0f})"
            verdict_class = "supported"
        elif ln_b >= LN_B_WEAK:
            verdict = (
                f"weakly supported ({LN_B_WEAK:.0f} <= ln B = {ln_b:.2f} "
                f"< {LN_B_SUPPORTED:.0f})"
            )
            verdict_class = "weak"
        else:
            verdict = f"not constrained (ln B = {ln_b:.2f} < {LN_B_WEAK:.0f})"
            verdict_class = "not-constrained"
        rows.append(
            {
                "molecule": molecule,
                "ran": True,
                "ln_b": ln_b,
                "ln_b_err": row.ln_bayes_factor_err,
                "sigma": row.sigma_equivalent,
                "status": row.status,
                "upper_limit": row.upper_limit_log_vmr,
                "quality_rating": rating,
                "verdict": verdict,
                "verdict_class": verdict_class,
            }
        )
    return {"ran": True, "rows": rows}


def _interpretation(
    detection: dict[str, Any], quality: dict[str, Any], retrieval: dict[str, Any]
) -> dict[str, Any]:
    """Guardrailed support classes from fixed thresholds (DECISIONS.md 41)."""
    if not detection.get("ran"):
        return {
            "assessed": False,
            "supported": [],
            "weak": [],
            "not_constrained": list(MOLECULES),
            "note": "Molecule-by-molecule model comparison was not run for this analysis, "
            "so no support class is assigned. Upper limits from the posterior, if any, "
            "are listed in the Results section and are not detections.",
        }
    supported = [r["molecule"] for r in detection["rows"]
                 if r.get("verdict_class") == "supported"]
    weak = [r["molecule"] for r in detection["rows"]
            if r.get("verdict_class") == "weak"]
    not_constrained = [r["molecule"] for r in detection["rows"]
                       if r.get("verdict_class") == "not-constrained"]
    missing = [m for m in MOLECULES
               if m not in {r["molecule"] for r in detection["rows"] if r.get("ran")}]
    return {
        "assessed": True,
        "supported": supported,
        "weak": weak,
        "not_constrained": sorted(set(not_constrained) | set(missing)),
        "note": None,
    }


def _support_sentence(molecules: list[str], verb: str) -> str:
    if not molecules:
        return ""
    names = ", ".join(molecules)
    return f"The data {verb} {names} absorption features at the documented thresholds."


def _summary_paragraph(ctx: dict[str, Any]) -> str:
    """One-paragraph templated summary from real numbers (no free-form prose)."""
    data = ctx["data"]
    quality = ctx["quality"]
    retrieval = ctx["retrieval"]
    interp = ctx["interpretation"]
    ml = ctx["ml"]

    parts = [
        f"Analysis {ctx['analysis_id']} of {ctx['planet_name']} "
        f"({data['telescope']}/{data['instrument']}, "
        f"{data['n_points']} points over {fmt(data['wavelength_min_um'])}-"
        f"{fmt(data['wavelength_max_um'])} um): "
        f"data quality {quality.get('suitability', 'not assessed')}."
    ]
    if retrieval.get("ran"):
        t_row = next((r for r in retrieval["rows"] if r["name"] == "T"), None)
        t_text = (
            f"T = {fmt(t_row['median'])} [{fmt(t_row['ci68_lo'])}, "
            f"{fmt(t_row['ci68_hi'])}] K (median, 68% CI)"
            if t_row else "temperature not available"
        )
        parts.append(
            f"Retrieval ({retrieval['sampler']}, n_live = {retrieval['n_live']}, "
            f"seed = {retrieval['seed']}) gives {t_text} "
            f"with log-evidence {fmt(retrieval['logz'])} +- {fmt(retrieval['logz_err'])}."
        )
    else:
        parts.append("Retrieval was not run for this analysis.")
    if interp.get("assessed"):
        if interp["supported"]:
            parts.append(_support_sentence(interp["supported"], "support"))
        if interp["weak"]:
            parts.append(_support_sentence(interp["weak"], "weakly support"))
        if interp["not_constrained"]:
            parts.append(_support_sentence(interp["not_constrained"], "do not support"))
    else:
        parts.append("Molecule-by-molecule support was not assessed (detection not run).")
    if ml.get("ran"):
        scored = [(r["molecule"], r["score"]) for r in ml["scores"] if r["score"] is not None]
        if scored:
            scored.sort(key=lambda item: item[1], reverse=True)
            if ml.get("dataset_hash") in (None, "", "unknown"):
                parts.append(
                    "ML candidate scores are all "
                    f"{scored[0][1]:.2f} (ML candidate score; no trained model "
                    "was available, so scores are placeholders)."
                )
            else:
                top = ", ".join(f"{name} ({value:.2f})" for name, value in scored[:2])
                parts.append(
                    "ML candidate scores (ML candidate score; not abundances or "
                    f"detections) range from {scored[-1][1]:.2f} to {scored[0][1]:.2f}, "
                    f"highest for {top}."
                )
    else:
        parts.append("ML scoring was not run for this analysis.")
    return " ".join(parts)


def _build_context(stored: dict[str, Any]) -> dict[str, Any]:
    analysis = stored["analysis"]
    planet = stored["planet"]
    observation = stored["observation"]
    spectrum_file = stored["spectrum_file"]
    spectrum: Spectrum | None = stored["spectrum"]
    provenance: dict[str, Any] = stored["provenance"]
    config: dict[str, Any] = stored["config"]
    posterior_row = stored["posterior_row"]
    result: RetrievalResult | None = stored["retrieval_result"]
    fixed = _fixed_from_planet(planet)

    planet_name = planet.name if planet is not None else "unknown"
    telescope = observation.telescope if observation is not None else "unknown"
    instrument = observation.instrument if observation is not None else "unknown"

    # --- model spectra on the observed grid (cheap plotting only) ---
    model: dict[str, Any] = {"wavelength": None, "best": None, "lo": None, "hi": None}
    chi2: float | None = None
    dof: int | None = None
    if result is not None and spectrum is not None and fixed is not None:
        from exosphere.forward.model import to_instrument

        native_wl, native_best = result.best_fit_spectrum(fixed)
        mw, mb = to_instrument(np.asarray(native_wl), np.asarray(native_best), spectrum)
        med, lo, hi = result.credible_band_spectrum(fixed, n_draws=64, seed=analysis.seed)
        _, med_b = to_instrument(np.asarray(native_wl), np.asarray(med), spectrum)
        _, lo_b = to_instrument(np.asarray(native_wl), np.asarray(lo), spectrum)
        _, hi_b = to_instrument(np.asarray(native_wl), np.asarray(hi), spectrum)
        model = {"wavelength": mw.tolist(), "best": mb.tolist(),
                 "lo": lo_b.tolist(), "hi": hi_b.tolist()}
        obs = np.asarray(spectrum.transmission, dtype=float)
        sig = np.asarray(spectrum.uncertainty, dtype=float)
        if np.all(sig > 0) and len(obs) == len(mb):
            chi2 = float(np.sum(((obs - mb) / sig) ** 2))
            dof = int(len(obs) - result.n_params)

    quality = _quality_section(stored["quality_row"])
    preprocess = _preprocess_section(stored["preprocess_log"])
    ml = _ml_section(stored["ml_row"])
    retrieval = _retrieval_section(posterior_row, result, analysis.seed)
    detection = _detection_section(
        stored["detection_rows"], quality.get("molecule_ratings", {})
    )
    interpretation = _interpretation(detection, quality, retrieval)

    wl = np.asarray(spectrum.wavelength, dtype=float) if spectrum is not None else None
    spec_hash = None
    spec_path = stored["spectrum_path"]
    if spec_path and Path(spec_path).exists():
        spec_hash = _sha256_file(Path(spec_path))

    data = {
        "source_archive": observation.source_archive if observation is not None else "unknown",
        "observation_id": observation.observation_id if observation is not None else "unknown",
        "spectrum_id": spectrum.observation_id if spectrum is not None else "unknown",
        "planet_name": planet_name,
        "telescope": telescope,
        "instrument": instrument,
        "units_wavelength": "µm",
        "units_depth": "fractional transit depth (Rp/Rs)^2",
        "n_points": len(wl) if wl is not None else None,
        "wavelength_min_um": float(wl.min()) if wl is not None else None,
        "wavelength_max_um": float(wl.max()) if wl is not None else None,
        "coverage_fraction": quality.get("wavelength_coverage_fraction"),
        "input_hash": spec_hash or provenance.get("input_data_hash") or "not recorded",
        "cleaned_hash": spectrum_file.file_hash if spectrum_file is not None else "not recorded",
    }

    # Prior bounds table (r_ref bounds need the catalog radius).
    catalog_r_jup = None
    if planet is not None and planet.pl_rade:
        catalog_r_jup = float(planet.pl_rade) / 11.209
    priors = [
        {"param": "T", "form": "uniform", "bounds": "[300, 2500] K"},
        {
            "param": "log H2O/CO2/CO/CH4/SO2",
            "form": "uniform in log10(VMR)",
            "bounds": "[-12, -1] dex",
        },
        {
            "param": "r_ref",
            "form": "uniform, +/- 30% of catalog radius",
            "bounds": (
                f"[{0.7 * catalog_r_jup:.3f}, {1.3 * catalog_r_jup:.3f}] R_Jup"
                if catalog_r_jup else "catalog-dependent (R_Jup)"
            ),
        },
        {"param": "log_p_cloud", "form": "uniform in log10(bar)", "bounds": "[-6, 2] log10(bar)"},
    ]

    setup = {
        "ran": retrieval.get("ran", False),
        "forward_model": "Isothermal atmosphere, constant-with-altitude volume mixing "
        "ratios, H2/He bulk background, grey cloud deck (cloud-top pressure).",
        "forward_model_version": (
            provenance.get("retrieval_model_version") or "not recorded"
        ),
        "gravity_m_s2": getattr(planet, "surface_gravity_m_s2", None),
        "stellar_radius_rsun": getattr(planet, "st_rad", None),
        "reference_pressure_bar": 0.01,
        "priors": priors,
        "sampler": retrieval.get("sampler", "dynesty"),
        "n_live": retrieval.get("n_live"),
        "dlogz": retrieval.get("dlogz"),
        "max_iter": config.get("max_iter"),
        "maxcall": config.get("maxcall"),
        "seed": analysis.seed,
        "error_inflation": config.get("error_inflation") or 0.0,
    }

    provenance_rows = [
        ("Analysis ID", analysis.analysis_id, ""),
        ("Planet", planet_name, ""),
        ("Observation ID", data["observation_id"], ""),
        ("Telescope", telescope, ""),
        ("Instrument", instrument, ""),
        ("Source archive", data["source_archive"], ""),
        ("Input data hash", provenance.get("input_data_hash") or "not recorded", ""),
        ("Cleaned spectrum hash", data["cleaned_hash"], "sha256"),
        ("Preprocessing version", provenance.get("preprocessing_version") or "not recorded", ""),
        ("ML model version", provenance.get("ml_model_version")
         or (ml.get("model_version") if ml.get("ran") else "not run"), ""),
        ("Retrieval model version", provenance.get("retrieval_model_version")
         or "not recorded", ""),
        ("Seed", str(analysis.seed), ""),
        ("Report version", REPORT_VERSION, ""),
        ("Report generated (UTC)", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"), ""),
    ]

    ctx = {
        "analysis_id": analysis.analysis_id,
        "planet_name": planet_name,
        "report_version": REPORT_VERSION,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "data": data,
        "quality": quality,
        "preprocess": preprocess,
        "ml": ml,
        "setup": setup,
        "retrieval": retrieval,
        "detection": detection,
        "interpretation": interpretation,
        "model": model,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": (chi2 / dof) if (chi2 is not None and dof) else None,
        "n_data_points": len(wl) if wl is not None else None,
        "n_params": 8 if retrieval.get("ran") else 0,
        "provenance_rows": provenance_rows,
        "bands": _molecule_bands(),
        "fmt": fmt,
    }
    ctx["summary"] = _summary_paragraph(ctx)
    return ctx


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _template() -> Any:
    template_path = Path(__file__).resolve().parent / "templates" / TEMPLATE_NAME
    return Environment(autoescape=True).from_string(
        template_path.read_text(encoding="utf-8")
    )


def render_html(ctx: dict[str, Any]) -> str:
    """Render the report HTML and enforce the wording rules."""
    _use_agg()
    import matplotlib.pyplot as plt

    spectrum = ctx["_spectrum"]
    result = ctx["_result"]
    figures: dict[str, str] = {}
    try:
        if spectrum is not None:
            figures["spectrum"] = _fig_to_data_uri(
                _figure_spectrum(spectrum, ctx["model"], ctx["bands"])
            )
            if ctx["model"]["best"] is not None:
                figures["residuals"] = _fig_to_data_uri(
                    _figure_residuals(
                        spectrum,
                        np.asarray(ctx["model"]["best"]),
                        ctx["chi2"],
                        ctx["dof"],
                    )
                )
        if result is not None:
            figures["corner"] = _fig_to_data_uri(_figure_corner(result))
    finally:
        plt.close("all")
    html = _template().render(
        ctx={key: value for key, value in ctx.items() if not key.startswith("_")},
        fmt=fmt,
        figures=figures,
    )
    violations = check_wording(html)
    if violations:
        raise ReportWordingError("; ".join(violations))
    return html


def _render_pdf_bytes(html: str) -> bytes:
    try:
        from weasyprint import HTML
    except Exception as error:
        raise PDFUnavailableError(
            "PDF export needs the WeasyPrint system libraries (Pango), which are "
            f"not installed in this environment: {error}"
        ) from error
    try:
        return bytes(HTML(string=html).write_pdf())
    except Exception as error:
        raise PDFUnavailableError(f"WeasyPrint failed to render the PDF: {error}") from error


async def build_report_async(analysis_id: str, format: str = "html") -> Path:
    """Build the report for a stored analysis and record it in the reports table."""
    if format not in ("html", "pdf"):
        raise ValueError(f"format must be 'html' or 'pdf', got {format!r}")
    stored = await _load_stored(analysis_id)
    if stored["spectrum"] is None:
        raise ReportNotFoundError(f"no spectrum file stored for analysis {analysis_id}")
    ctx = _build_context(stored)
    # Figures need the live objects; keep them out of the Jinja context itself.
    ctx["_spectrum"] = stored["spectrum"]
    ctx["_result"] = stored["retrieval_result"]
    html = render_html(ctx)

    config = load_config()
    reports_dir = config.data_cache_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if format == "html":
        payload = html.encode("utf-8")
        suffix = "html"
    else:
        payload = _render_pdf_bytes(html)
        suffix = "pdf"
    out_path = reports_dir / f"{analysis_id}_report.{suffix}"
    out_path.write_bytes(payload)
    file_hash = _sha256_file(out_path)

    async with async_session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(Analysis).where(Analysis.analysis_id == analysis_id)
            )
            analysis = result.scalar_one_or_none()
            if analysis is None:  # pragma: no cover - raced deletion
                raise ReportNotFoundError(f"analysis {analysis_id} not found")
            old = await session.execute(
                select(Report).where(
                    Report.analysis_id == analysis.id,
                    Report.report_type == REPORT_TYPE,
                )
            )
            for row in old.scalars().all():
                await session.delete(row)
            session.add(
                Report(
                    analysis_id=analysis.id,
                    report_type=REPORT_TYPE,
                    file_path=str(out_path),
                    file_hash=file_hash,
                )
            )
            analysis.report_file_path = str(out_path)
    return out_path


def build_report(analysis_id: str, format: str = "html") -> Path:
    """Synchronous wrapper (CLI/tests); the API awaits :func:`build_report_async`."""
    return asyncio.run(build_report_async(analysis_id, format=format))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m exosphere.report.build <analysis_id>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Build an ExoSphere scientific report.")
    parser.add_argument("analysis_id", help="Stored analysis id, e.g. EXO-000001")
    parser.add_argument("--format", choices=["html", "pdf"], default="html")
    args = parser.parse_args(argv)
    try:
        path = build_report(args.analysis_id, format=args.format)
    except PDFUnavailableError as error:
        parser.exit(2, f"PDF export unavailable: {error}\n")
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
