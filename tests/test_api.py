"""Phase 6 end-to-end test: WASP-39 b through the API.

quality -> preprocess -> ML -> retrieval (small settings) -> report,
asserting every result endpoint serves real stored content. Detection runs
are covered by a small synthetic unit test (nested runs are too slow for CI).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Point the API at a throwaway database BEFORE importing exosphere.api modules.
_TMPDIR = tempfile.mkdtemp(prefix="exosphere-api-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMPDIR) / 'api_test.db'}"

BENCH_NPZ = (
    Path(__file__).resolve().parents[1]
    / "data_cache"
    / "benchmark"
    / "40_24_96_78_WASP_39_b_3.11466_5502_6.tbl.npz"
)


def _seed_planet_observation() -> None:
    from exosphere.api.db import Observation, Planet, async_session_maker, init_db

    async def _seed():
        await init_db()
        async with async_session_maker() as session:
            async with session.begin():
                planet = Planet(
                    name="WASP-39 b",
                    pl_rade=14.23,
                    pl_bmasse=89.0,
                    pl_eqt=1166.0,
                    pl_orbper=4.055,
                    pl_orbsmax=0.0486,
                    pl_orbeccen=0.0,
                    pl_orbincl=87.83,
                    pl_ratdor=11.4,
                    pl_ratror=0.146,
                    st_rad=0.93,
                    st_teff=5400.0,
                    surface_gravity_m_s2=4.3,
                )
                session.add(planet)
                await session.flush()
                session.add(
                    Observation(
                        planet_id=planet.id,
                        observation_id="40/24/96/78/WASP_39_b_3.11466_5502_6.tbl",
                        target_id="WASP-39 b",
                        instrument="NIRSpec PRISM",
                        telescope="JWST",
                        source_archive="NASA Exoplanet Archive",
                        wavelength_min_um=0.52132,
                        wavelength_max_um=5.34412,
                        num_points=147,
                        spectrum_file_path=str(BENCH_NPZ),
                        spectrum_file_hash="3359e371ec0364d381be808534ab2b9e1ca92e793f35d02b8daf20266afb65e3",
                    )
                )

    asyncio.run(_seed())


@pytest.mark.slow
def test_end_to_end_wasp39b(tmp_path, monkeypatch):
    """POST /analyses -> all result endpoints serve real content."""
    import exosphere.report.build as build_module
    from exosphere.core.config import Config

    _seed_planet_observation()
    # Keep report artifacts out of the real data_cache.
    monkeypatch.setattr(
        build_module,
        "load_config",
        lambda: Config(data_cache_dir=tmp_path, prt_input_data_path=None),
    )

    from fastapi.testclient import TestClient

    from exosphere.api.app import app

    with TestClient(app) as client:
        resp = client.post(
            "/analyses",
            json={
                "planet_name": "WASP-39 b",
                "n_live": 50,
                "dlogz": 1.0,
                "max_iter": 2000,
                "maxcall": 8000,
                "seed": 42,
                "run_ml": True,
                "run_detection": False,
                "run_quality": True,
                "run_preprocess": True,
            },
        )
        assert resp.status_code == 201, resp.text
        analysis_id = resp.json()["analysis_id"]

        # Status is completed (TestClient runs background tasks inline).
        status = client.get(f"/analyses/{analysis_id}/status").json()
        assert status["status"] == "completed", status
        assert status["progress"] == 1.0

        quality = client.get(f"/analyses/{analysis_id}/quality")
        assert quality.status_code == 200, quality.text
        assert quality.json()["overall_suitability"] in ("GOOD", "LIMITED", "POOR")
        assert quality.json()["molecule_ratings"]

        ml = client.get(f"/analyses/{analysis_id}/ml")
        assert ml.status_code == 200, ml.text
        assert set(ml.json()["scores"]) == {"H2O", "CO2", "CO", "CH4", "SO2"}

        retrieval = client.get(f"/analyses/{analysis_id}/retrieval")
        assert retrieval.status_code == 200, retrieval.text
        body = retrieval.json()
        assert body["median"] and body["ci_68"] and body["best_fit"]
        assert body["seed"] == 42

        posterior = client.get(f"/analyses/{analysis_id}/posterior")
        assert posterior.status_code == 200, posterior.text
        assert len(posterior.json()["samples"]) > 0
        assert len(posterior.json()["samples"][0]) == 8

        model = client.get(f"/analyses/{analysis_id}/model")
        assert model.status_code == 200, model.text
        assert len(model.json()["best_fit_depth"]) == 147
        assert len(model.json()["ci_lo"]) == 147

        spectrum = client.get(f"/analyses/{analysis_id}/spectrum?cleaned=true")
        assert spectrum.status_code == 200, spectrum.text
        assert len(spectrum.json()["wavelength_um"]) == 147

        provenance = client.get(f"/analyses/{analysis_id}/provenance")
        assert provenance.status_code == 200, provenance.text
        prov = provenance.json()
        assert prov["planet"] == "WASP-39 b"
        assert prov["telescope"] == "JWST"
        assert prov["instrument"] == "NIRSpec PRISM"
        assert prov["source_archive"] == "NASA Exoplanet Archive"

        # The report builder resolves the preprocess log through load_config
        # (redirected to tmp_path here); stage the pipeline-written file.
        from exosphere.core.config import load_config as real_load_config

        real_log = real_load_config().data_cache_dir / "preprocess_logs" / f"{analysis_id}.json"
        assert real_log.exists(), "pipeline did not persist the PreprocessLog"
        staged_dir = tmp_path / "preprocess_logs"
        staged_dir.mkdir(parents=True, exist_ok=True)
        staged_dir.joinpath(real_log.name).write_bytes(real_log.read_bytes())

        report = client.post(f"/analyses/{analysis_id}/report", json={"format": "html"})
        assert report.status_code == 200, report.text
        download = client.get(f"/analyses/{analysis_id}/report?format=html")
        assert download.status_code == 200
        assert "ExoSphere scientific report" in download.text
        # Preprocessing section is populated from the pipeline-written log.
        assert "No preprocessing log is stored" not in download.text


@pytest.mark.slow
def test_detection_bayes_factor_synthetic():
    """Nested-model comparison runs on the fixed reduced-likelihood path."""
    from tests.conftest import make_synthetic_spectrum

    from exosphere.forward.model import PlanetFixed
    from exosphere.retrieval.detection import compute_bayes_factor
    from exosphere.retrieval.results import RetrievalResult
    from exosphere.retrieval.samplers import SamplerConfig, run_dynesty

    wave = np.linspace(0.6, 5.3, 40)
    spectrum = make_synthetic_spectrum(
        wave, np.full(40, 0.021) + 1e-4 * np.sin(wave * 8), np.full(40, 5e-5)
    )
    fixed = PlanetFixed(gravity_m_s2=4.3, stellar_radius_rsun=0.93)
    config = SamplerConfig(n_live=10, dlogz=1.0, max_iter=300, maxcall=1500, seed=7)
    full = run_dynesty(spectrum, fixed, config, 7)
    assert isinstance(full, RetrievalResult)

    ln_b, ln_b_err, diagnostics = compute_bayes_factor(full, spectrum, fixed, "H2O", config, 7)
    assert np.isfinite(ln_b)
    assert ln_b_err is None or np.isfinite(ln_b_err) or np.isinf(ln_b_err)
    assert diagnostics["ncall"] > 0
    assert diagnostics["ncall"] >= diagnostics["niter"]
    assert diagnostics["capped"] is True
