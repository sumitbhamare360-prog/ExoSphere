"""Phase 8 tests: twin parameter mappings, builder, and the /twin endpoint."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Point the API at a throwaway database BEFORE importing exosphere.api modules.
_TMPDIR = tempfile.mkdtemp(prefix="exosphere-twin-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMPDIR) / 'twin_test.db'}"

from exosphere import twin  # noqa: E402

WASP39_CATALOG = {
    "pl_rade": 14.23,  # 1.27 R_Jup in Earth radii
    "pl_bmasse": 89.0,  # ~0.28 M_Jup in Earth masses
    "st_rad": 0.93,
    "st_teff": 5400.0,
    "pl_orbsmax": 0.0486,
    "pl_orbper": 4.055,
    "pl_orbincl": 87.83,
    "pl_orbeccen": 0.0,
    "pl_eqt": 1166.0,
    "pl_ratdor": 11.4,
    "surface_gravity_m_s2": None,
}

RETRIEVAL = {
    "median": {
        "T": 1100.0,
        "log_h2o": -3.5,
        "log_co2": -4.0,
        "log_co": -4.5,
        "log_ch4": -5.0,
        "log_so2": -6.0,
        "r_ref": 1.27,
        "log_p_cloud": -1.0,
    },
    "ci_68": {
        "T": [1000.0, 1200.0],
        "log_h2o": [-3.8, -3.2],
        "log_co2": [-4.4, -3.6],
        "log_co": [-5.0, -4.0],
        "log_ch4": [-6.0, -4.5],
        "log_so2": [-7.0, -5.5],
        "r_ref": [1.22, 1.32],
        "log_p_cloud": [-1.5, -0.5],
    },
    "best_fit": {
        "T": 1100.0,
        "log_h2o": -3.5,
        "log_co2": -4.0,
        "log_co": -4.5,
        "log_ch4": -5.0,
        "log_so2": -6.0,
        "r_ref": 1.27,
        "log_p_cloud": -1.0,
    },
}

BANDS = {
    "H2O": [(0.93, 0.98), (1.3, 1.5)],
    "CO2": [(4.2, 4.4)],
    "CO": [(4.5, 4.9)],
    "CH4": [(3.2, 3.5)],
    "SO2": [(3.9, 4.2)],
}


def _by_name(twin_obj: dict, name: str) -> dict:
    for param in twin_obj["parameters"]:
        if param["name"] == name:
            return param
    raise AssertionError(f"parameter {name!r} missing")


def test_scale_height_matches_formula():
    # H = kB*T / (mu*mH*g) = 1.380649e-23*1000 / (2.3*1.6735575e-27*9.81)
    expected = 1.380649e-23 * 1000.0 / (2.3 * 1.6735575e-27 * 9.81)
    assert twin.scale_height_m(1000.0, 2.3, 9.81) == pytest.approx(expected, rel=1e-12)
    assert twin.scale_height_m(1000.0, 2.3, 9.81) == pytest.approx(365900.0, rel=1e-3)


def test_scale_height_rejects_bad_inputs():
    with pytest.raises(ValueError):
        twin.scale_height_m(0.0, 2.3, 9.81)
    with pytest.raises(ValueError):
        twin.scale_height_m(1000.0, 0.0, 9.81)
    with pytest.raises(ValueError):
        twin.scale_height_m(1000.0, 2.3, -1.0)


def test_mmw_matches_forward_model():
    from exosphere.forward.model import ModelParams, PlanetFixed, _build_atmosphere

    params = ModelParams(
        T=1100.0,
        log_h2o=-3.5,
        log_co2=-4.0,
        log_co=-4.5,
        log_ch4=-5.0,
        log_so2=-6.0,
        r_ref=1.27,
        log_p_cloud=-1.0,
    )
    fixed = PlanetFixed(gravity_m_s2=4.3, stellar_radius_rsun=0.93)
    _, _, _, mmw_forward = _build_atmosphere(params, fixed)
    mmw_twin = twin.mean_molecular_weight_g_mol(params.vmr_dict())
    assert mmw_twin == pytest.approx(mmw_forward, rel=1e-12)


def test_mmw_rejects_impossible_composition():
    with pytest.raises(ValueError):
        twin.mean_molecular_weight_g_mol({"H2O": 0.6, "CO2": 0.5})


def test_transit_depth_formula():
    rp = 1.27 * twin.R_JUP_M
    rs = 0.93 * twin.R_SUN_M
    assert twin.transit_depth_fraction(rp, rs) == pytest.approx((rp / rs) ** 2, rel=1e-12)
    with pytest.raises(ValueError):
        twin.transit_depth_fraction(rp, 0.0)


def test_semi_major_axis_sources():
    assert twin.semi_major_axis_stellar_radii(11.4, 0.0486, 0.93) == pytest.approx(11.4)
    derived = twin.semi_major_axis_stellar_radii(None, 0.0486, 0.93)
    assert derived == pytest.approx(0.0486 * twin.AU_M / (0.93 * twin.R_SUN_M), rel=1e-12)
    assert twin.semi_major_axis_stellar_radii(None, None, 0.93) is None


def test_build_full_catalog_plus_retrieval():
    wl = np.linspace(0.6, 5.3, 60)
    depth = 0.02 + 0.001 * np.exp(-0.5 * ((wl - 1.4) / 0.1) ** 2)
    out = twin.build_twin_parameters(
        analysis_id="EXO-000001",
        planet_name="WASP-39 b",
        catalog=dict(WASP39_CATALOG),
        retrieval={k: dict(v) for k, v in RETRIEVAL.items()},
        spectrum={"wavelength": wl.tolist(), "transmission": depth.tolist()},
        bands=BANDS,
        bands_version="test-1.0",
    )
    assert out["analysis_id"] == "EXO-000001"
    assert len(out["molecules"]) == 5

    r = _by_name(out, "planet_radius")
    assert r["source"] == "inferred" and r["unit"] == "R_Jup"
    assert r["value"] == pytest.approx(1.27)
    assert r["ci_68"] == [1.22, 1.32]

    assert _by_name(out, "stellar_teff")["source"] == "measured"
    assert _by_name(out, "surface_gravity")["source"] == "derived"
    assert _by_name(out, "eccentricity")["source"] == "measured"

    cloud = _by_name(out, "cloud_top_pressure")
    assert cloud["source"] == "inferred"
    assert cloud["constrained"] is True  # CI width 1.0 dex < 2.0

    h = _by_name(out, "scale_height")
    assert h["source"] == "derived" and h["unit"] == "km" and h["value"] > 0

    obs = _by_name(out, "transit_depth_observed")
    assert obs["source"] == "measured"
    assert obs["value"] == pytest.approx(float(np.median(depth)))

    contribs = {m["molecule"]: m for m in out["molecules"]}
    assert contribs["H2O"]["contribution_fraction"] > 0.5
    assert all(0.0 <= m["contribution_fraction"] <= 1.0 for m in out["molecules"])
    assert all(m["in_band"] for m in out["molecules"])


def test_absent_molecule_has_zero_contribution():
    best_fit = dict(RETRIEVAL["best_fit"])
    best_fit["log_ch4"] = -12.0  # negligible abundance
    wl = np.linspace(0.6, 5.3, 40)
    contribs = twin.molecule_contributions(best_fit, 4.3, 0.93, wl, BANDS, "test-1.0")
    assert contribs["CH4"] == 0.0


def test_build_empty_catalog_falls_back_to_assumed():
    out = twin.build_twin_parameters(
        analysis_id="EXO-000002",
        planet_name="Unknown b",
        catalog={},
        retrieval=None,
        spectrum=None,
        bands=BANDS,
        bands_version="test-1.0",
    )
    by_name = {p["name"]: p for p in out["parameters"]}
    assert by_name["planet_radius"]["source"] == "assumed"
    assert by_name["stellar_teff"]["value"] == pytest.approx(5778.0)
    assert by_name["inclination"]["value"] == pytest.approx(90.0)
    assert by_name["eccentricity"]["value"] == pytest.approx(0.0)
    assert by_name["atmosphere_temperature"]["source"] == "assumed"
    assert by_name["cloud_top_pressure"]["source"] == "assumed"
    assert by_name["transit_depth_observed"]["value"] is None
    assert all(m["contribution_fraction"] == 0.0 for m in out["molecules"])
    assert out["meta"]["has_retrieval"] is False


def test_build_wide_cloud_ci_is_unconstrained():
    retrieval = {k: dict(v) for k, v in RETRIEVAL.items()}
    retrieval["ci_68"] = dict(retrieval["ci_68"])
    retrieval["ci_68"]["log_p_cloud"] = [-4.0, 1.0]  # 5 dex wide
    out = twin.build_twin_parameters(
        analysis_id="EXO-000003",
        planet_name="WASP-39 b",
        catalog=dict(WASP39_CATALOG),
        retrieval=retrieval,
        spectrum=None,
        bands=BANDS,
        bands_version="test-1.0",
    )
    cloud = _by_name(out, "cloud_top_pressure")
    assert cloud["source"] == "inferred"
    assert cloud["constrained"] is False


def test_contribution_cache_returns_consistent_values():
    twin._CONTRIBUTION_CACHE.clear()
    wl = np.linspace(0.6, 5.3, 40)
    first = twin.molecule_contributions(RETRIEVAL["best_fit"], 4.3, 0.93, wl, BANDS, "test-1.0")
    assert len(twin._CONTRIBUTION_CACHE) == 1
    second = twin.molecule_contributions(RETRIEVAL["best_fit"], 4.3, 0.93, wl, BANDS, "test-1.0")
    assert first == second
    assert len(twin._CONTRIBUTION_CACHE) == 1


def test_sources_are_valid_tags():
    out = twin.build_twin_parameters(
        analysis_id="EXO-000004",
        planet_name="WASP-39 b",
        catalog=dict(WASP39_CATALOG),
        retrieval={k: dict(v) for k, v in RETRIEVAL.items()},
        spectrum=None,
        bands=BANDS,
        bands_version="test-1.0",
    )
    valid = {"measured", "inferred", "derived", "assumed"}
    for param in out["parameters"]:
        assert param["source"] in valid, param
        assert param["unit"] != "", param


# ---------------------------------------------------------------------------
# Endpoint tests (seeded temp database)
# ---------------------------------------------------------------------------


def _seed_db(tmp_path: Path):
    """Create planet + observation + analysis + spectrum + posterior rows."""
    import asyncio

    from exosphere.api.db import (
        Analysis,
        AnalysisStage,
        AnalysisStatus,
        Observation,
        Planet,
        Posterior,
        SpectrumFile,
        async_session_maker,
        init_db,
    )
    from exosphere.core.provenance import Provenance
    from exosphere.core.spectrum import Spectrum

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
                    st_rad=0.93,
                    st_teff=5400.0,
                    surface_gravity_m_s2=4.3,
                )
                session.add(planet)
                await session.flush()
                obs = Observation(
                    planet_id=planet.id,
                    observation_id="bench-prism",
                    target_id="WASP-39 b",
                    instrument="NIRSpec PRISM",
                    telescope="JWST",
                    source_archive="archive",
                )
                session.add(obs)
                await session.flush()
                wl = np.linspace(0.6, 5.3, 60).tolist()
                spec = Spectrum(
                    wavelength=wl,
                    transmission=[0.02] * 60,
                    uncertainty=[0.0001] * 60,
                    wavelength_bin_edges=[0.6 + i * (4.7 / 60) for i in range(61)],
                    observation_id="bench-prism",
                    target_id="WASP-39 b",
                    instrument="NIRSpec PRISM",
                    provenance=Provenance(analysis_id="EXO-000001", planet="WASP-39 b"),
                )
                spec_path = tmp_path / "bench.npz"
                spec.save(spec_path)
                spec_file = SpectrumFile(
                    observation_id=obs.id,
                    file_path=str(spec_path),
                    file_hash="abc",
                    wavelength_min_um=0.6,
                    wavelength_max_um=5.3,
                    num_points=60,
                )
                session.add(spec_file)
                await session.flush()
                analysis = Analysis(
                    analysis_id="EXO-000001",
                    planet_id=planet.id,
                    observation_id=obs.id,
                    status=AnalysisStatus.COMPLETED,
                    stage=AnalysisStage.COMPLETED,
                    progress=1.0,
                    seed=42,
                    spectrum_file_id=spec_file.id,
                )
                session.add(analysis)
                await session.flush()
                posterior = Posterior(
                    analysis_id=analysis.id,
                    file_path="",
                    file_hash="",
                    logz=-100.0,
                    logz_err=1.0,
                    n_samples=100,
                    n_live=50,
                    dlogz=0.5,
                    summary_json={
                        "median": RETRIEVAL["median"],
                        "ci_68": RETRIEVAL["ci_68"],
                        "best_fit": RETRIEVAL["best_fit"],
                        "param_names": list(RETRIEVAL["median"]),
                    },
                )
                session.add(posterior)

    asyncio.run(_seed())


def test_twin_endpoint(tmp_path):
    _seed_db(tmp_path)
    from fastapi.testclient import TestClient

    from exosphere.api.app import app

    with TestClient(app) as client:
        resp = client.get("/analyses/EXO-000001/twin")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_id"] == "EXO-000001"
    assert body["planet_name"] == "WASP-39 b"
    names = {p["name"] for p in body["parameters"]}
    assert {
        "planet_radius",
        "surface_gravity",
        "stellar_radius",
        "stellar_teff",
        "semi_major_axis",
        "orbital_period",
        "inclination",
        "eccentricity",
        "atmosphere_temperature",
        "cloud_top_pressure",
        "mean_molecular_weight",
        "scale_height",
        "transit_depth_observed",
        "transit_depth_model",
    } <= names
    by_name = {p["name"]: p for p in body["parameters"]}
    assert by_name["planet_radius"]["source"] == "inferred"
    assert by_name["planet_radius"]["ci_68"] == [1.22, 1.32]
    assert by_name["stellar_teff"]["source"] == "measured"
    assert len(body["molecules"]) == 5


def test_twin_endpoint_404():
    from fastapi.testclient import TestClient

    from exosphere.api.app import app

    with TestClient(app) as client:
        resp = client.get("/analyses/EXO-999999/twin")
    assert resp.status_code == 404


def test_molecule_bands_endpoint():
    from fastapi.testclient import TestClient

    from exosphere.api.app import app

    with TestClient(app) as client:
        resp = client.get("/config/molecule-bands")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["molecules"]) == {"H2O", "CO2", "CO", "CH4", "SO2"}
    assert body["wavelength_range_um"] == [0.6, 5.3]
