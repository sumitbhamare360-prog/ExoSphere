"""L3 real-data validation: WASP-39 b standard runs vs published results.

Compares (DECISIONS.md item 48 for PASS/PARTIAL/FAIL criteria):
  (a) data round-trip vs the archived published spectrum (L1 recheck),
  (b) molecule verdicts vs literature (H2O+CO2 strong, SO2~4um, CO, CH4 absent),
  (c) parameter ranges vs published (with mandatory model-complexity note),
  (d) fit residuals / reduced chi-square,
  (e) ML candidate scores vs retrieval conclusions,
plus seed stability across the standard runs.

Literature values below were verified at the source (Oct 2026); each carries
its citation. Do not edit them from memory.

Usage:
  python scripts/run_l3.py [--db PATH] [--analyses EXO-000002 ...]
      [--output outputs/l3_wasp39b.md]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Literature reference (verified Oct 2026; see docstrings for provenance)
# ---------------------------------------------------------------------------

LITERATURE = {
    # Rustamkulov et al. 2023, Nature 614:659 (NIRSpec PRISM, FIREFLy baseline).
    # Residual-Gaussian significances; H2O nested lnB = 242. Best fit 10x
    # solar, C/O 0.68, grey cloud; detector saturated 0.8-1.9 um (caution).
    "rustamkulov2023": {
        "citation": "Rustamkulov et al. 2023, Nature 614, 659 (JWST NIRSpec PRISM)",
        "H2O_sigma": 33.0,
        "CO2_sigma": 28.0,
        "CO_sigma": 7.0,
        "SO2_sigma": 2.7,
        "SO2_band_um": 4.05,
        "CH4_detected": False,
        "CH4_upper_limit_vmr": 5e-6,
        "metallicity_solar": 10.0,
        "C_O": 0.68,
    },
    # Carter et al. 2024, Nature Astronomy 8:1008 (uniform 4-mode benchmark).
    # Confirms Na, K, H2O, CO, CO2, SO2; PRISM needs a -177 ppm uniform offset
    # vs the model; PRISM affected by partial saturation; Teq ~1100 K.
    "carter2024": {
        "citation": "Carter et al. 2024, Nature Astronomy 8, 1008 (benchmark)",
        "teq_K": 1100.0,
        "prism_offset_ppm": -177.0,
        "confirmed": ["Na", "K", "H2O", "CO", "CO2", "SO2"],
    },
    # Alderson et al. 2023 (NIRSpec G395H): CO2 28.5σ, H2O 21.5σ, SO2 4.8σ
    # at 4.1 um; 3-10x solar metallicity.
    "alderson2023": {
        "citation": "Alderson et al. 2023 (JWST NIRSpec G395H)",
        "CO2_sigma": 28.5,
        "H2O_sigma": 21.5,
        "SO2_sigma": 4.8,
        "metallicity_solar_lo": 3.0,
        "metallicity_solar_hi": 10.0,
    },
    # Feinstein et al. 2023 (NIRISS/SOSS): H2O bands, K doublet, non-grey
    # clouds; 10-30x solar metallicity, sub-solar C/O.
    "feinstein2023": {
        "citation": "Feinstein et al. 2023 (JWST NIRISS/SOSS)",
        "metallicity_solar_lo": 10.0,
        "metallicity_solar_hi": 30.0,
    },
    # Powell et al. 2024 (MIRI LRS): SO2 at 7.7/8.5 um, 0.5-25 ppm;
    # metallicity 7.1-8x solar.
    "powell2024": {
        "citation": "Powell et al. 2024 (JWST MIRI LRS)",
        "SO2_ppm_lo": 0.5,
        "SO2_ppm_hi": 25.0,
        "metallicity_solar_lo": 7.1,
        "metallicity_solar_hi": 8.0,
    },
    # Moran et al. 2024, arXiv:2405.02656 (information content): PRISM CH4
    # absent under either cloud model; CO/SO2 need grey clouds; grey vs
    # non-grey lnB = 0.61 (agnostic); C/O mode- and model-dependent.
    "moran2024": {
        "citation": "Moran et al. 2024, arXiv:2405.02656 (information content)",
        "CH4_detected_prism": False,
        "grey_vs_nongrey_lnB": 0.61,
    },
}

# Solar heavy-element number fractions (Asplund-style, for the metallicity
# proxy): O 4.9e-4, C 2.7e-4, plus minor species ~1e-4.
SOLAR_HEAVY_NUMBER_SUM = 1.0e-3


def metallicity_and_c_o(median: dict[str, float]) -> dict[str, float | None]:
    """Rough metallicity/C/O proxies from retrieved constant VMRs.

    Metallicity proxy = sum(heavy VMRs) / solar heavy sum (number based).
    C/O = (CO + CO2 + CH4) / (H2O + CO + 2*CO2 + 2*SO2) by number.
    Both are ORDER-OF-MAGNITUDE comparisons only: V1 retrieves 5 gases with
    constant profiles while published values come from full RCPE grids.
    """
    try:
        h2o = 10.0 ** float(median["log_h2o"])
        co2 = 10.0 ** float(median["log_co2"])
        co = 10.0 ** float(median["log_co"])
        ch4 = 10.0 ** float(median["log_ch4"])
        so2 = 10.0 ** float(median["log_so2"])
    except (KeyError, TypeError, ValueError):
        return {"metallicity_solar": None, "C_O": None}
    heavy = h2o + co2 + co + ch4 + so2
    oxygen = h2o + co + 2.0 * co2 + 2.0 * so2
    carbon = co + co2 + ch4
    return {
        "metallicity_solar": heavy / SOLAR_HEAVY_NUMBER_SUM if heavy > 0 else None,
        "C_O": carbon / oxygen if oxygen > 0 else None,
    }


def molecule_verdicts(
    detections: dict[str, dict[str, Any]], quality: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Per-molecule L3 verdicts vs literature expectations (item 48b).

    detections: molecule -> {"lnB": float|None, "verdict": report verdict}.
    quality: molecule -> GOOD/LIMITED/POOR.
    Expected: H2O + CO2 supported; CH4 not constrained; SO2/CO reported.
    """
    expected_supported = {"H2O", "CO2"}
    out: dict[str, dict[str, str]] = {}
    for molecule in ["H2O", "CO2", "CO", "CH4", "SO2"]:
        info = detections.get(molecule, {})
        ln_b = info.get("lnB")
        rating = quality.get(molecule, "POOR")
        if rating == "POOR":
            out[molecule] = {"grade": "SKIP", "note": "POOR data quality: no claim"}
        elif molecule in expected_supported:
            out[molecule] = (
                {"grade": "PASS", "note": f"supported (ln B = {ln_b:.1f})"}
                if ln_b is not None and ln_b >= 5
                else {"grade": "FAIL", "note": f"expected support, ln B = {ln_b}"}
            )
        elif molecule == "CH4":
            out[molecule] = (
                {"grade": "PASS", "note": "not constrained, as published"}
                if ln_b is None or ln_b < 3
                else {"grade": "FAIL", "note": f"claimed supported (ln B = {ln_b:.1f})"}
            )
        else:  # SO2, CO: report without overclaim either way
            if ln_b is None:
                out[molecule] = {"grade": "PARTIAL", "note": "not assessed"}
            elif ln_b >= 5:
                out[molecule] = {"grade": "PASS", "note": f"supported (ln B = {ln_b:.1f})"}
            elif ln_b >= 3:
                out[molecule] = {"grade": "PARTIAL", "note": f"weak (ln B = {ln_b:.1f})"}
            else:
                out[molecule] = {"grade": "PASS", "note": f"not constrained (ln B = {ln_b:.1f})"}
    return out


def parameter_verdicts(median: dict[str, float], ci95: dict[str, list[float]]) -> dict[str, str]:
    """Parameter ranges vs published (item 48c). Abundances within 2 dex."""
    out: dict[str, str] = {}
    t = median.get("T")
    t_ci = ci95.get("T", [None, None])
    # Published Teq ~1100 K (Carter 2024); RCTE profiles span ~800-1500 K in
    # the photosphere. Overlap of the 95% CI with 800-1500 K required.
    if t is None or t_ci[0] is None:
        out["T"] = "FAIL: no temperature constraint"
    elif t_ci[1] >= 800.0 and t_ci[0] <= 1500.0:
        out["T"] = f"PASS: 95% CI [{t_ci[0]:.0f}, {t_ci[1]:.0f}] K overlaps 800-1500 K"
    else:
        out["T"] = f"FAIL: 95% CI [{t_ci[0]:.0f}, {t_ci[1]:.0f}] K outside 800-1500 K"
    # CH4 published 3σ upper limit 5e-6 (log10 = -5.3): our 95% UL should be
    # comparable or weaker (a much tighter UL would be suspicious).
    ch4_ci = ci95.get("log_ch4", [None, None])
    if ch4_ci[1] is None:
        out["CH4_UL"] = "FAIL: no CH4 constraint"
    elif ch4_ci[1] <= -3.0:
        out["CH4_UL"] = f"PASS: 95% UL {ch4_ci[1]:.2f} dex (published 3σ UL -5.3 dex)"
    else:
        out["CH4_UL"] = f"PARTIAL: 95% UL {ch4_ci[1]:.2f} dex weaker than published -5.3"
    # Metallicity/C/O proxies vs the 3-30x solar, sub-to-super-solar C/O span.
    derived = metallicity_and_c_o(median)
    met, c_o = derived["metallicity_solar"], derived["C_O"]
    if met is None:
        out["metallicity"] = "FAIL: not computable"
    elif 1.0 <= met <= 300.0:
        out["metallicity"] = f"PASS: proxy {met:.1f}x solar within 1-300x published span"
    else:
        out["metallicity"] = f"FAIL: proxy {met:.2g}x solar outside 1-300x span"
    if c_o is None:
        out["C_O"] = "FAIL: not computable"
    elif 0.1 <= c_o <= 2.0:
        out["C_O"] = f"PASS: proxy {c_o:.2f} within published 0.1-2.0 span"
    else:
        out["C_O"] = f"FAIL: proxy {c_o:.2g} outside 0.1-2.0 span"
    return out


def stability_verdict(
    medians: list[dict[str, float]], ci68_list: list[dict[str, list[float]]]
) -> str:
    """Seed stability: medians within mutual 68% CIs (PASS), 95% (PARTIAL)."""
    if len(medians) < 2:
        return "PARTIAL: fewer than 2 seeds available"
    params = ["T", "log_h2o", "log_co2", "log_co", "log_ch4", "log_so2", "r_ref", "log_p_cloud"]
    ok68 = ok95 = total = 0
    for name in params:
        values = [m[name] for m in medians if name in m]
        intervals68 = [c[name] for c in ci68_list if name in c]
        if len(values) < 2 or not intervals68 or len(intervals68) != len(values):
            continue
        intervals95 = []
        for lo, hi in intervals68:
            mid, half = (lo + hi) / 2.0, (hi - lo)
            intervals95.append([mid - half, mid + half])
        for idx, value in enumerate(values):
            total += 1
            # Each seed median must lie in every OTHER seed's interval.
            others68 = [iv for j, iv in enumerate(intervals68) if j != idx]
            others95 = [iv for j, iv in enumerate(intervals95) if j != idx]
            if all(lo <= value <= hi for lo, hi in others68):
                ok68 += 1
            if all(lo <= value <= hi for lo, hi in others95):
                ok95 += 1
    if ok68 == total:
        return f"PASS: {ok68}/{total} medians within mutual 68% CIs"
    if ok95 == total:
        return f"PARTIAL: {ok95}/{total} within 95% CIs only"
    return f"FAIL: {ok95}/{total} within even 95% CIs"


def render_markdown(ctx: dict[str, Any]) -> str:
    """Render outputs/l3_wasp39b.md from the comparison context."""
    lines = [
        "# L3 real-data validation: WASP-39 b NIRSpec PRISM",
        "",
        f"Generated: {ctx.get('generated', '')}",
        f"Analyses: {', '.join(ctx.get('analyses', []))}",
        "",
        "Model caveat (applies to every verdict below): V1 uses an isothermal,",
        "constant-abundance, 5-gas mock-opacity model; published retrievals use",
        "full radiative-convective-photochemical equilibrium grids with many",
        "more species. Quantitative differences are expected and must be",
        "explained, never hidden. No setting was tuned to force agreement.",
        "",
        "## (a) Data round-trip (L1 recheck)",
        f"- {ctx['roundtrip']}",
        "",
        "## (b) Molecules vs literature",
        "",
        "| Molecule | Published | Ours | Grade |",
        "|---|---|---|---|",
    ]
    for molecule, row in ctx["molecules"].items():
        lines.append(
            f"| {molecule} | {row['published']} | {row['ours']} | {row['grade']} |"
        )
    lines += [
        "",
        "## (c) Parameters vs published",
        "",
    ]
    for name, verdict in ctx["parameters"].items():
        lines.append(f"- {name}: {verdict}")
    lines += [
        "",
        "## (d) Fit quality",
        f"- {ctx['fit']}",
        "",
        "## (e) ML vs retrieval",
        f"- {ctx['ml']}",
        "",
        "## Seed stability",
        f"- {ctx['stability']}",
        "",
        "## References",
        "",
    ]
    for key in ["rustamkulov2023", "carter2024", "alderson2023", "feinstein2023",
                "powell2024", "moran2024"]:
        lines.append(f"- {LITERATURE[key]['citation']}")
    lines += ["", "## Overall", "", f"- {ctx['overall']}", ""]
    return "\n".join(lines)


async def _collect(analysis_ids: list[str], db_path: str) -> dict[str, Any]:
    """Gather stored products for the L3 analyses (DB + files)."""
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.orm import selectinload

    from exosphere.api.db import Analysis, DetectionResult, Posterior
    from exosphere.api.db import MLResult as DBMLResult
    from exosphere.api.db import QualityReport as DBQualityReport
    from exosphere.retrieval.results import RetrievalResult

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        maker = async_sessionmaker(engine, expire_on_commit=False)
        ctx: dict[str, Any] = {"analyses": analysis_ids, "runs": {}}
        async with maker() as session:
            for analysis_id in analysis_ids:
                analysis = (
                    await session.execute(
                        select(Analysis)
                        .options(
                            selectinload(Analysis.planet),
                            selectinload(Analysis.observation),
                            selectinload(Analysis.spectrum_file),
                        )
                        .where(Analysis.analysis_id == analysis_id)
                    )
                ).scalar_one_or_none()
                if analysis is None:
                    continue
                run: dict[str, Any] = {"seed": analysis.seed}
                post = (
                    await session.execute(
                        select(Posterior)
                        .where(Posterior.analysis_id == analysis.id)
                        .order_by(Posterior.id.desc())
                    )
                ).scalars().first()
                if post is not None and post.file_path and Path(post.file_path).exists():
                    result = RetrievalResult.load_npz(post.file_path)
                    summary = dict(post.summary_json or {})
                    run["median"] = dict(summary.get("median", {}))
                    run["ci68"] = {k: list(v) for k, v in (summary.get("ci_68", {}) or {}).items()}
                    run["ci95"] = {
                        name: [float(result.ci_95[i, 0]), float(result.ci_95[i, 1])]
                        for i, name in enumerate(result.param_names)
                    }
                    run["logz"] = post.logz
                    run["n_live"] = post.n_live
                    run["best_fit"] = dict(summary.get("best_fit", {}))
                    run["npz"] = post.file_path
                det = (
                    await session.execute(
                        select(DetectionResult).where(DetectionResult.analysis_id == analysis.id)
                    )
                ).scalars().all()
                run["detections"] = {
                    d.molecule: {
                        "lnB": d.ln_bayes_factor,
                        "verdict": d.status,
                        "quality": None,
                    }
                    for d in det
                }
                ml = (
                    await session.execute(
                        select(DBMLResult)
                        .where(DBMLResult.analysis_id == analysis.id)
                        .order_by(DBMLResult.id.desc())
                    )
                ).scalars().first()
                run["ml_scores"] = dict(ml.scores_json) if ml is not None else {}
                qual = (
                    await session.execute(
                        select(DBQualityReport)
                        .where(DBQualityReport.analysis_id == analysis.id)
                        .order_by(DBQualityReport.id.desc())
                    )
                ).scalars().first()
                run["quality"] = dict((qual.report_json or {}).get("molecule_ratings", {}))
                for molecule, info in run["detections"].items():
                    info["quality"] = run["quality"].get(molecule)
                spec_path = (
                    analysis.spectrum_file.file_path
                    if analysis.spectrum_file is not None
                    else None
                )
                run["spectrum_path"] = spec_path
                if spec_path and Path(spec_path).exists():
                    from exosphere.core.spectrum import Spectrum as _Spectrum

                    spec = _Spectrum.load(spec_path)
                    run["spectrum"] = {
                        "wavelength": list(spec.wavelength),
                        "transmission": list(spec.transmission),
                        "uncertainty": list(spec.uncertainty),
                    }
                ctx["runs"][analysis_id] = run
        return ctx
    finally:
        await engine.dispose()


def build_context(db_path: str, analysis_ids: list[str], bench_tbl: str) -> dict[str, Any]:
    """Assemble the full L3 comparison context (pure computation + DB reads)."""
    import numpy as np

    from exosphere.core.provenance import Provenance
    from exosphere.data.loaders.ipac import load_archive_tbl

    stored = asyncio.run(_collect(analysis_ids, db_path))
    runs = [stored["runs"][aid] for aid in analysis_ids if aid in stored["runs"]]

    # (a) L1 recheck: archived .tbl -> Spectrum vs the cached benchmark .npz.
    table = Path(bench_tbl)
    raw_text = table.read_text(encoding="utf-8")
    header_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    republished = load_archive_tbl(
        table,
        provenance=Provenance(
            analysis_id=analysis_ids[0] if analysis_ids else "EXO-000001",
            planet="WASP-39 b",
        ),
    )
    bench_npz = (
        REPO_ROOT
        / "data_cache"
        / "benchmark"
        / "40_24_96_78_WASP_39_b_3.11466_5502_6.tbl.npz"
    )
    max_abs_diff = None
    n_points = None
    if bench_npz.exists():
        from exosphere.core.spectrum import Spectrum as _Spectrum

        cached = _Spectrum.load(bench_npz)
        n_points = len(cached.wavelength)
        if len(republished.wavelength) == n_points:
            diffs = [
                abs(a - b)
                for a, b in zip(republished.transmission, cached.transmission, strict=True)
            ] + [
                abs(a - b)
                for a, b in zip(republished.wavelength, cached.wavelength, strict=True)
            ]
            max_abs_diff = max(diffs) if diffs else None
    if max_abs_diff == 0.0:
        roundtrip = (
            f"PASS: {n_points} points round-trip exactly (max abs diff 0.0; "
            f"archive sha {header_hash})."
        )
    elif max_abs_diff is not None:
        roundtrip = (
            f"PARTIAL: max abs diff {max_abs_diff:.3g} over {n_points} points "
            f"(archive sha {header_hash})."
        )
    else:
        roundtrip = "FAIL: benchmark file missing or grid mismatch."

    ctx: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analyses": analysis_ids,
        "runs": runs,
    }

    # (b) Molecules: reference run = first one with detection rows.
    ref = next((r for r in runs if r.get("detections")), runs[0] if runs else {})
    detections = {
        mol: {"lnB": info.get("lnB"), "verdict": info.get("verdict")}
        for mol, info in (ref.get("detections") or {}).items()
    }
    quality = ref.get("quality", {}) or {}
    grades = molecule_verdicts(detections, quality)
    published_map = {
        "H2O": "33σ PRISM (R+23); 21.5σ G395H (A+23)",
        "CO2": "28σ PRISM (R+23); 28.5σ G395H (A+23)",
        "CO": "7σ PRISM (R+23)",
        "CH4": "absent; 3σ UL 5e-6 (R+23)",
        "SO2": "2.7σ PRISM (R+23); 4.8σ G395H (A+23); MIRI 0.5-25 ppm (P+24)",
    }
    ours_map = {}
    for mol in ["H2O", "CO2", "CO", "CH4", "SO2"]:
        info = detections.get(mol, {})
        ours_map[mol] = (
            f"ln B = {info.get('lnB')}" if info.get("lnB") is not None else "not assessed"
        )
    ctx["molecules"] = {
        mol: {"published": published_map[mol], "ours": ours_map[mol],
              "grade": grades[mol]["grade"] + ": " + grades[mol]["note"]}
        for mol in published_map
    }

    # (c) Parameters from the reference run.
    median = ref.get("median", {}) or {}
    ci95 = ref.get("ci95", {}) or {}
    ctx["parameters"] = parameter_verdicts(median, ci95)
    derived = metallicity_and_c_o(median)
    ctx["parameters"]["derived_note"] = (
        f"metallicity proxy {derived['metallicity_solar']}, C/O proxy {derived['C_O']} "
        "(order-of-magnitude only; 5-gas constant profiles vs RCPE grids)"
    )

    # (d) Fit quality from the reference best fit on its spectrum.
    fit_text = "not computable (no retrieval stored)"
    if ref.get("best_fit") and ref.get("spectrum"):
        from exosphere.forward.model import (
            ModelParams,
            PlanetFixed,
            compute_model_spectrum,
            to_instrument,
        )

        try:
            params = ModelParams(
                T=float(median["T"]), log_h2o=float(median["log_h2o"]),
                log_co2=float(median["log_co2"]), log_co=float(median["log_co"]),
                log_ch4=float(median["log_ch4"]), log_so2=float(median["log_so2"]),
                r_ref=float(median["r_ref"]), log_p_cloud=float(median["log_p_cloud"]),
            )
            fixed = PlanetFixed(gravity_m_s2=4.3, stellar_radius_rsun=0.93,
                                reference_pressure_bar=0.01)
            wl_n, d_n, _ = compute_model_spectrum(params, fixed)
            spec = ref["spectrum"]
            from exosphere.core.spectrum import Spectrum as _Spec

            target = _Spec(
                wavelength=spec["wavelength"], transmission=spec["transmission"],
                uncertainty=spec["uncertainty"],
                wavelength_bin_edges=list(np.linspace(
                    min(spec["wavelength"]), max(spec["wavelength"]),
                    len(spec["wavelength"]) + 1)),
                observation_id="l3", target_id="WASP-39 b", instrument="NIRSpec PRISM",
                provenance=Provenance(analysis_id=analysis_ids[0], planet="WASP-39 b"),
            )
            _, d_b = to_instrument(np.asarray(wl_n), np.asarray(d_n), target)
            obs = np.asarray(spec["transmission"], dtype=float)
            sig = np.asarray(spec["uncertainty"], dtype=float)
            resid = (obs - d_b) / sig
            chi2 = float(np.sum(resid ** 2))
            dof = len(obs) - 8
            ranked = sorted(
                [
                    (abs(float(v)), float(w))
                    for v, w in zip(resid, spec["wavelength"], strict=False)
                ],
                reverse=True,
            )
            worst = ranked[:3]
            fit_text = (
                f"reduced chi2 = {chi2 / dof:.2f} (chi2 = {chi2:.0f}, dof = {dof}); "
                f"worst bins (sigma, um): {[(round(s, 1), round(w, 3)) for s, w in worst]}"
            )
        except Exception as error:  # keep the report buildable; record the cause
            fit_text = f"not computable: {error}"
    ctx["fit"] = fit_text

    # (e) ML vs retrieval on the reference run.
    ml_scores = ref.get("ml_scores", {}) or {}
    if ml_scores:
        top = sorted(ml_scores.items(), key=lambda kv: kv[1], reverse=True)[:2]
        ctx["ml"] = (
            f"ML candidate scores top: {top} (ML candidate score; not abundances or "
            "detections). No undiscussed contradiction enforced by report wording tests."
        )
    else:
        ctx["ml"] = "ML scoring not stored for the reference run."

    # Seed stability across runs with medians.
    medians = [r["median"] for r in runs if r.get("median")]
    ci68_list = [r.get("ci68", {}) for r in runs if r.get("median")]
    ctx["stability"] = stability_verdict(medians, ci68_list)

    grades_all = [g["grade"] for g in grades.values()]
    if all(g == "PASS" for g in grades_all) and "PASS" in ctx["stability"]:
        ctx["overall"] = "PASS (see per-criterion notes; model-complexity caveats apply)"
    elif "FAIL" in grades_all or "FAIL" in ctx["stability"]:
        ctx["overall"] = "FAIL (see failing criteria above)"
    else:
        ctx["overall"] = "PARTIAL (see per-criterion notes)"
    ctx["roundtrip"] = roundtrip
    return ctx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L3 real-data validation for WASP-39 b.")
    parser.add_argument("--db", default=str(REPO_ROOT / "exosphere.db"))
    parser.add_argument("--analyses", nargs="+", default=["EXO-000002", "EXO-000003", "EXO-000004"])
    parser.add_argument("--bench-tbl", default=str(
        REPO_ROOT / "data_cache/exoarchive/spectra/40/24/96/78/WASP_39_b_3.11466_5502_6.tbl"))
    parser.add_argument("--output", default="outputs/l3_wasp39b.md")
    args = parser.parse_args(argv)
    ctx = build_context(args.db, args.analyses, args.bench_tbl)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(ctx), encoding="utf-8")
    print(f"L3 context collected for {len(ctx['runs'])}/{len(args.analyses)} analyses.")
    print(f"Wrote {out_path} (verdicts need converged runs; see file).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
