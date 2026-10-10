"""Phase 9a tests: scientific report builder, wording rules, report API."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Point the API at a throwaway database BEFORE importing exosphere.api modules.
_TMPDIR = tempfile.mkdtemp(prefix="exosphere-report-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMPDIR) / 'report_test.db'}"

from exosphere.core.config import Config  # noqa: E402
from exosphere.report.build import (  # noqa: E402
    SECTION_IDS,
    PDFUnavailableError,
    ReportNotFoundError,
    build_report,
    check_wording,
    main,
)

PARAM_NAMES = ["T", "log_h2o", "log_co2", "log_co", "log_ch4", "log_so2", "r_ref",
             "log_p_cloud"]


@pytest.fixture(scope="module", autouse=True)
def _isolated_database():
    """Wipe the shared test database once: all DB modules share one engine."""
    from conftest import reset_database

    reset_database()
CENTER = np.array([1100.0, -3.5, -4.2, -5.0, -6.0, -7.0, 1.27, -1.0])
SCALES = np.array([80.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.03, 0.4])


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    idx = np.argsort(x)
    cdf = np.cumsum(w[idx]) / np.sum(w)
    return float(np.interp(q, cdf, x[idx]))


def _seed_db(tmp_path: Path, analysis_id: str, *, minimal: bool = False) -> None:
    """Seed a complete fake stored analysis (or a minimal spectrum-only one)."""
    from tests.conftest import make_feature_spectrum, make_synthetic_spectrum

    from exosphere.api.db import (
        Analysis,
        AnalysisStage,
        AnalysisStatus,
        DetectionResult,
        Observation,
        Planet,
        Posterior,
        SpectrumFile,
        async_session_maker,
        init_db,
    )
    from exosphere.api.db import MLResult as DBMLResult
    from exosphere.api.db import QualityReport as DBQualityReport
    from exosphere.core.provenance import Provenance
    from exosphere.core.spectrum import Spectrum
    from exosphere.preprocess.clean import CleanOptions, clean
    from exosphere.quality.assess import assess
    from exosphere.retrieval.results import RetrievalResult

    wave, depth, sigma = make_feature_spectrum(n_points=200, seed=11)
    planet_name = f"FAKE-{analysis_id}"
    obs_id = f"fake-obs-{analysis_id}"
    raw = make_synthetic_spectrum(wave, depth, sigma, observation_id=obs_id, target_id=planet_name)
    quality = assess(raw)
    cleaned, log = clean(raw, CleanOptions())
    # Preprocess log lives outside the DB (data_cache/preprocess_logs/<id>.json);
    # the report builder resolves it through load_config (redirected in tests).
    log_dir = tmp_path / "preprocess_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{analysis_id}.json").write_text(log.to_json(), encoding="utf-8")

    async def _seed():
        await init_db()
        async with async_session_maker() as session:
            async with session.begin():
                planet = Planet(
                    name=planet_name,
                    pl_rade=14.23,
                    pl_bmasse=89.0,
                    pl_eqt=1166.0,
                    pl_orbper=4.055,
                    pl_orbsmax=0.0486,
                    pl_orbeccen=0.0,
                    pl_orbincl=87.83,
                    pl_ratdor=11.4,
                    st_rad=0.93,
                    st_teff=5400.0,
                    surface_gravity_m_s2=4.3,
                )
                session.add(planet)
                await session.flush()
                obs = Observation(
                    planet_id=planet.id,
                    observation_id=obs_id,
                    target_id=planet_name,
                    instrument="NIRSpec PRISM",
                    telescope="JWST",
                    source_archive="NASA Exoplanet Archive",
                )
                session.add(obs)
                await session.flush()
                spec = Spectrum(
                    wavelength=list(cleaned.wavelength),
                    transmission=list(cleaned.transmission),
                    uncertainty=list(cleaned.uncertainty),
                    wavelength_bin_edges=list(cleaned.wavelength_bin_edges),
                    observation_id=obs_id,
                    target_id=planet_name,
                    instrument="NIRSpec PRISM",
                    provenance=Provenance(analysis_id=analysis_id, planet=planet_name),
                )
                spec_path = tmp_path / "cleaned.npz"
                spec.save(spec_path)
                spec_file = SpectrumFile(
                    observation_id=obs.id,
                    file_path=str(spec_path),
                    file_hash="deadbeef",
                    wavelength_min_um=float(wave.min()),
                    wavelength_max_um=float(wave.max()),
                    num_points=len(wave),
                )
                session.add(spec_file)
                await session.flush()
                analysis = Analysis(
                    analysis_id=analysis_id,
                    planet_id=planet.id,
                    observation_id=obs.id,
                    status=AnalysisStatus.COMPLETED,
                    stage=AnalysisStage.COMPLETED,
                    progress=1.0,
                    seed=42,
                    spectrum_file_id=spec_file.id,
                    provenance_json=Provenance(
                        analysis_id=analysis_id,
                        planet=planet_name,
                        observation_id=obs_id,
                        telescope="JWST",
                        instrument="NIRSpec PRISM",
                        source_archive="NASA Exoplanet Archive",
                        preprocessing_version="preprocess-1.0.0",
                        ml_model_version="CNN-v1",
                        retrieval_model_version="mock-1.0.0",
                    ).model_dump(mode="json"),
                )
                session.add(analysis)
                await session.flush()
                if minimal:
                    return
                session.add(
                    DBQualityReport(
                        analysis_id=analysis.id, report_json=quality.model_dump(mode="json")
                    )
                )
                session.add(
                    DBMLResult(
                        analysis_id=analysis.id,
                        scores_json={"H2O": 0.9, "CO2": 0.8, "CO": 0.2, "CH4": 0.1, "SO2": 0.4},
                        model_version="CNN-v1",
                        dataset_hash="abc123",
                        model_config_hash="cfg1",
                    )
                )
                rng = np.random.default_rng(7)
                samples = CENTER + SCALES * rng.normal(size=(400, 8))
                weights = np.full(400, 1.0 / 400)
                median = np.array(
                    [_weighted_quantile(samples[:, i], weights, 0.5) for i in range(8)]
                )
                ci_68 = np.array(
                    [
                        [
                            _weighted_quantile(samples[:, i], weights, 0.16),
                            _weighted_quantile(samples[:, i], weights, 0.84),
                        ]
                        for i in range(8)
                    ]
                )
                ci_95 = np.array(
                    [
                        [
                            _weighted_quantile(samples[:, i], weights, 0.025),
                            _weighted_quantile(samples[:, i], weights, 0.975),
                        ]
                        for i in range(8)
                    ]
                )
                result = RetrievalResult(
                    samples=samples,
                    weights=weights,
                    logz=-123.4,
                    logz_err=0.5,
                    best_fit=CENTER.copy(),
                    median=median,
                    ci_68=ci_68,
                    ci_95=ci_95,
                    logz_full=-123.4,
                    logz_err_full=0.5,
                    param_names=list(PARAM_NAMES),
                    n_samples=400,
                    n_live=50,
                    dlogz=0.5,
                    runtime_s=1.0,
                    sampler="dynesty",
                    seed=42,
                )
                npz_path = tmp_path / "posterior.npz"
                result.save_npz(npz_path)
                session.add(
                    Posterior(
                        analysis_id=analysis.id,
                        file_path=str(npz_path),
                        file_hash="feedface",
                        logz=-123.4,
                        logz_err=0.5,
                        n_samples=400,
                        n_live=50,
                        dlogz=0.5,
                        summary_json={
                            "median": {
                                name: float(value)
                                for name, value in zip(PARAM_NAMES, median, strict=True)
                            },
                            "ci_68": {
                                name: [float(lo), float(hi)]
                                for name, (lo, hi) in zip(PARAM_NAMES, ci_68, strict=True)
                            },
                            "best_fit": {
                                name: float(value)
                                for name, value in zip(PARAM_NAMES, CENTER, strict=True)
                            },
                            "param_names": list(PARAM_NAMES),
                        },
                    )
                )
                detections = [
                    ("H2O", 12.5, 0.6, "detected", None),
                    ("CO2", 4.2, 0.7, "tentative", None),
                    ("CO", 0.5, 0.8, "not_detected", -5.5),
                    ("CH4", -0.3, 0.9, "not_detected", -6.2),
                    ("SO2", 3.5, 0.7, "tentative", None),
                ]
                for mol, ln_b, err, status, upper in detections:
                    session.add(
                        DetectionResult(
                            analysis_id=analysis.id,
                            molecule=mol,
                            ln_bayes_factor=ln_b,
                            ln_bayes_factor_err=err,
                            sigma_equivalent=float(np.sqrt(2.0 * ln_b)) if ln_b > 0 else 0.0,
                            status=status,
                            upper_limit_log_vmr=upper,
                            details_json={},
                        )
                    )

    asyncio.run(_seed())


@pytest.fixture()
def redirect_cache(tmp_path, monkeypatch):
    """Point report inputs/outputs (preprocess logs, report files) at tmp_path."""
    import exosphere.report.build as build_module

    monkeypatch.setattr(
        build_module,
        "load_config",
        lambda: Config(data_cache_dir=tmp_path, prt_input_data_path=None),
    )
    return tmp_path


def _section_ids(html: str) -> list[str]:
    return re.findall(r'<section[^>]*id="([^"]+)"', html)


def test_build_full_html(tmp_path, redirect_cache):
    _seed_db(tmp_path, "EXO-000001")
    out = build_report("EXO-000001")
    assert out.suffix == ".html"
    assert out.exists()
    html = out.read_text(encoding="utf-8")

    # Section order snapshot: exactly the 10 sections, in phase order.
    assert _section_ids(html) == SECTION_IDS
    assert len(SECTION_IDS) == 10

    # Wording rules hold on the generated document.
    assert check_wording(html) == []

    # ML numbers are labelled; units appear in the results.
    assert html.count("ML candidate score") >= 6
    for unit in ("µm", "dex", "R_Jup", "K", "log10(bar)", "m/s^2"):
        assert unit in html, f"unit {unit} missing"

    # Figures embedded; provenance recorded.
    assert html.count("data:image/png;base64") == 3
    assert "EXO-000001" in html and "FAKE-EXO-000001" in html
    assert "report-1.0.0" in html

    # Guardrail verdicts from the seeded evidence.
    assert "supported (ln B = 12.50" in html  # H2O
    assert "weakly supported" in html  # CO2, SO2
    assert "not constrained" in html  # CO, CH4

    # Stored in the reports table + analysis pointer.
    from sqlalchemy import select

    from exosphere.api.db import Analysis, Report, async_session_maker

    async def _check():
        async with async_session_maker() as session:
            result = await session.execute(
                select(Analysis).where(Analysis.analysis_id == "EXO-000001")
            )
            analysis = result.scalar_one()
            assert analysis.report_file_path == str(out)
            rep = await session.execute(
                select(Report).where(Report.analysis_id == analysis.id)
            )
            rows = rep.scalars().all()
            assert len(rows) == 1
            assert rows[0].report_type == "scientific"
            assert rows[0].file_path == str(out)
            assert len(rows[0].file_hash) == 64

    asyncio.run(_check())


def test_missing_parts_graceful(tmp_path, redirect_cache):
    _seed_db(tmp_path, "EXO-000002", minimal=True)
    out = build_report("EXO-000002")
    html = out.read_text(encoding="utf-8")
    assert _section_ids(html) == SECTION_IDS
    assert check_wording(html) == []
    # Every optional stage degrades to an explicit "not run" section.
    assert html.count("not run") >= 5
    assert "ML scoring was not run" in html
    assert "Retrieval was not run" in html
    # Data-only spectrum figure still renders.
    assert html.count("data:image/png;base64") == 1


def test_partial_detection_graceful(tmp_path, redirect_cache):
    """A molecule without a stored comparison renders 'not run', not empty cells."""
    from sqlalchemy import select

    from exosphere.api.db import Analysis, DetectionResult, async_session_maker

    _seed_db(tmp_path, "EXO-000007")

    async def _drop():
        async with async_session_maker() as session:
            async with session.begin():
                result = await session.execute(
                    select(Analysis).where(Analysis.analysis_id == "EXO-000007")
                )
                analysis = result.scalar_one()
                rows = await session.execute(
                    select(DetectionResult).where(
                        DetectionResult.analysis_id == analysis.id,
                        DetectionResult.molecule == "SO2",
                    )
                )
                for row in rows.scalars().all():
                    await session.delete(row)

    asyncio.run(_drop())
    out = build_report("EXO-000007")
    html = out.read_text(encoding="utf-8")
    assert "model comparison not run" in html
    assert check_wording(html) == []


def test_unknown_analysis_raises(tmp_path, redirect_cache):
    with pytest.raises(ReportNotFoundError):
        build_report("EXO-999999")


def test_poor_quality_overrides_evidence(tmp_path, redirect_cache, monkeypatch):
    """AGENTS.md rule 3: POOR data quality suppresses the molecule claim."""

    _seed_db(tmp_path, "EXO-000003")

    from sqlalchemy import select

    from exosphere.api.db import Analysis, async_session_maker
    from exosphere.api.db import QualityReport as DBQualityReport

    async def _spoil():
        async with async_session_maker() as session:
            async with session.begin():
                result = await session.execute(
                    select(Analysis).where(Analysis.analysis_id == "EXO-000003")
                )
                analysis = result.scalar_one()
                poor = {
                    "suitability": "LIMITED",
                    "n_points": 200,
                    "wavelength_min_um": 0.6,
                    "wavelength_max_um": 5.3,
                    "wavelength_coverage_fraction": 0.9,
                    "median_snr": 2.0,
                    "max_band_snr": 6.0,
                    "band_snr": {},
                    "flagged_fraction": 0.0,
                    "nan_fraction": 0.0,
                    "outlier_count": 0,
                    "outlier_fraction": 0.0,
                    "uncertainty": {
                        "median": 1e-4,
                        "non_positive_count": 0,
                        "nan_count": 0,
                        "huge_count": 0,
                        "tiny_count": 0,
                    },
                    "effective_resolving_power": 100.0,
                    "molecule_ratings": {"H2O": "POOR", "CO2": "GOOD", "CO": "GOOD",
                                         "CH4": "GOOD", "SO2": "GOOD"},
                    "molecules": [],
                    "thresholds": {},
                    "config_version": "test",
                    "method": {},
                }
                session.add(DBQualityReport(analysis_id=analysis.id, report_json=poor))

    asyncio.run(_spoil())
    out = build_report("EXO-000003")
    html = out.read_text(encoding="utf-8")
    # H2O has ln B = 12.5 but POOR data quality: no support claim.
    assert "not constrained (POOR data quality)" in html
    assert "supported (ln B = 12.50" not in html
    assert check_wording(html) == []


def test_wording_rules_unit():
    def wrap(section: str, body: str) -> str:
        sections = "".join(
            f'<section id="{sid}"><h2>{sid}</h2>'
            f'<p>{body if sid == section else "neutral text."}</p></section>'
            for sid in SECTION_IDS
        )
        return "<html><body>" + sections + "</body></html>"

    # Banned ML-driven claims are caught in the ML section and the summary.
    assert check_wording(wrap("sec-ml", "H2O is present with score 0.9 (ML candidate score)."))
    assert check_wording(wrap("sec-ml", "CO2 was detected at 0.8 (ML candidate score)."))
    assert check_wording(wrap("sec-title", "The data show H2O exists."))
    # Global bans apply everywhere.
    assert check_wording(wrap("sec-results", "Implications for habitability are discussed."))
    assert check_wording(wrap("sec-results", "No life claims are made."))
    # Clean text passes (with the mandatory per-number labels).
    clean_ml = " ".join(f"{m} 0.1 (ML candidate score)" for m in ["H2O", "CO2", "CO", "CH4", "SO2"])
    assert check_wording(wrap("sec-ml", "Scores: " + clean_ml + " (ML candidate score).")) == []
    # Missing ML labels are caught.
    assert any("labels every number" in v for v in check_wording(wrap("sec-ml", "plain numbers")))


def test_pdf_unavailable_without_system_libs(tmp_path, redirect_cache, monkeypatch):
    """PDF export must fail with a clear error when Pango is missing."""
    import sys

    _seed_db(tmp_path, "EXO-000004")
    monkeypatch.setitem(sys.modules, "weasyprint", None)
    with pytest.raises(PDFUnavailableError):
        build_report("EXO-000004", format="pdf")


def test_cli_builds_html(tmp_path, redirect_cache, capsys):
    _seed_db(tmp_path, "EXO-000005")
    assert main(["EXO-000005", "--format", "html"]) == 0
    out, _ = capsys.readouterr()
    assert out.strip().endswith("EXO-000005_report.html")
    assert Path(out.strip()).exists()


def test_report_api(tmp_path, redirect_cache, monkeypatch):
    import sys

    _seed_db(tmp_path, "EXO-000006")
    from fastapi.testclient import TestClient

    from exosphere.api.app import app

    with TestClient(app) as client:
        # Download before build -> 404.
        resp = client.get("/analyses/EXO-000006/report")
        assert resp.status_code == 404
        # Build via POST.
        resp = client.post("/analyses/EXO-000006/report", json={"format": "html"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["analysis_id"] == "EXO-000006"
        assert body["report_version"] == "report-1.0.0"
        assert len(body["file_hash"]) == 64
        # Download the built file.
        resp = client.get("/analyses/EXO-000006/report?format=html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "EXO-000006" in resp.text
        # Unknown analysis -> 404.
        resp = client.post("/analyses/EXO-999999/report", json={"format": "html"})
        assert resp.status_code == 404
        assert client.get("/analyses/EXO-999999/report").status_code == 404
        # PDF requested without system libs -> 501.
        monkeypatch.setitem(sys.modules, "weasyprint", None)
        resp = client.post("/analyses/EXO-000006/report", json={"format": "pdf"})
        assert resp.status_code == 501
