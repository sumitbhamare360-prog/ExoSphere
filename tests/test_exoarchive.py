"""Unit tests for data/exoarchive.py with all network calls mocked."""

import pytest
import requests

from exosphere.core.provenance import Provenance
from exosphere.data import exoarchive
from exosphere.data.exoarchive import (
    ExoplanetArchiveError,
    get_planet_params,
    get_spectrum,
    list_literature_spectra,
    tap_query_csv,
)

PSCOMPPARS_CSV = (
    "pl_name,pl_rade,pl_bmasse,pl_bmassprov,pl_eqt,pl_insol,pl_orbper,pl_orbsmax,"
    "pl_orbeccen,pl_orbincl,pl_tranmid,pl_ratror,pl_ratdor,st_rad,st_teff,st_mass,"
    "st_logg,st_met,st_spectype\n"
    "WASP-39 b,1.275,89.8,Masses or Mass*sin(i),740.0,660.0,4.055259,0.04751,,87.83,"
    "2457082.72735,0.1451,11.4,,5950.0,0.93,,0.0,G6V\n"
)

SPECTRA_CSV = (
    "pl_name,spec_type,authors,num_datapoints,instrument,facility,minwavelng,"
    "maxwavelng,note,bibcode,spec_path\n"
    "WASP-39 b,Transmission,Carter et al. 2024,147,Near Infrared Spectrograph (NIRSpec),"
    "NASA 6.5m James Webb Space Telescope (JWST) Satellite Mission,0.5213,5.3441,"
    '"PRISM, Native resolution",2024NatAs...8.1008C,40/24/96/78/WASP_39_b_5502_6.tbl\n'
    "WASP-39 b,Transmission,Feinstein et al. 2023,331,Near Infrared Imager and Slitless "
    "Spectrograph (NIRISS),NASA 6.5m James Webb Space Telescope (JWST) Satellite Mission,"
    "0.6310,2.7965,,2023Natur.614..670F,74/34/80/11/WASP_39_b_5078_1.tbl\n"
    "WASP-39 b,Eclipse,Kammer et al. 2015,2,Infrared Array Camera (IRAC),"
    "Spitzer Space Telescope satellite,3.6000,4.5000,Table 3,2015ApJ...810..118K,"
    "45/23/09/89/WASP_39_b_4121_1.tbl\n"
)

FIREFLY_HTML = (
    "<html><body onload="
    "\"FF_InitPage ('/work/TMP_abc_1/atmospheres/tab1', '/workspace/TMP_abc_1', "
    "'/exodata/FDL', 'Exoplanet Archive (FF v1.0)', 'tab1Tab', '0', 'ops')\">"
    "</body></html>"
)

TBL_BYTES = None  # filled by fixture


class FakeResponse:
    def __init__(self, *, text="", content=None, status_code=200):
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


@pytest.fixture
def tbl_bytes(ipac_tbl_path):
    return ipac_tbl_path.read_bytes()


def _fake_http(monkeypatch, tbl_bytes=None):
    """Route fake responses; return (calls, fake)."""
    calls: list[tuple[str, dict | None]] = []

    def fake_get(url, *, params=None):
        calls.append((url, params))
        if url == exoarchive.TAP_URL:
            query = (params or {}).get("query", "")
            if "FROM pscomppars" in query:
                return FakeResponse(text=PSCOMPPARS_CSV)
            if "FROM spectra" in query:
                return FakeResponse(text=SPECTRA_CSV)
            return FakeResponse(text="ERROR  unknown table")
        if url == exoarchive.SPECTRA_APP_URL:
            return FakeResponse(text=FIREFLY_HTML)
        if "/atmospheres/tab1/data/" in url:
            assert tbl_bytes is not None
            return FakeResponse(content=tbl_bytes)
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(exoarchive, "_http_get", fake_get)
    return calls, fake_get


def test_get_planet_params_parses_and_derives_gravity(tmp_path, monkeypatch):
    calls, _ = _fake_http(monkeypatch)
    params = get_planet_params("WASP-39 b", cache_dir=tmp_path)

    assert params.planet_name == "WASP-39 b"
    assert params.planet_radius_earth == pytest.approx(1.275)
    assert params.planet_mass_earth == pytest.approx(89.8)
    assert params.equilibrium_temperature_k == 740.0
    assert params.orbital_period_days == pytest.approx(4.055259)
    assert params.stellar_radius_solar is None  # empty archive cell -> None
    assert params.stellar_teff_k == 5950.0
    assert params.stellar_spectype == "G6V"
    assert params.eccentricity is None

    # derived surface gravity: G*M/R^2 with M, R in Earth units
    expected = 6.67430e-11 * (89.8 * 5.9722e24) / (1.275 * 6.371e6) ** 2
    assert params.surface_gravity_m_s2 == pytest.approx(expected, rel=1e-12)

    # response was cached; the second call must not hit the network
    assert len(calls) == 1
    assert list(tmp_path.glob("tap_*.csv"))
    again = get_planet_params("WASP-39 b", cache_dir=tmp_path)
    assert again == params
    assert len(calls) == 1


def test_get_planet_params_unknown_planet_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exoarchive,
        "_http_get",
        lambda url, params=None: FakeResponse(text="pl_name\n"),
    )
    with pytest.raises(ExoplanetArchiveError, match="no pscomppars row"):
        get_planet_params("Not-A-Planet b", cache_dir=tmp_path)


def test_tap_error_response_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exoarchive,
        "_http_get",
        lambda url, params=None: FakeResponse(text="ERROR 504 Upstream Request Timeout"),
    )
    with pytest.raises(ExoplanetArchiveError, match="TAP query failed"):
        tap_query_csv("SELECT 1 FROM pscomppars", cache_dir=tmp_path)


def test_list_literature_spectra_parses_and_filters(tmp_path, monkeypatch):
    _fake_http(monkeypatch)
    spectra = list_literature_spectra("WASP-39 b", cache_dir=tmp_path)
    assert len(spectra) == 3
    assert spectra[0].spectrum_id == spectra[0].spec_path
    assert spectra[0].spectrum_type == "Transmission"
    assert "PRISM" in (spectra[0].note or "")
    assert spectra[0].min_wavelength_um == pytest.approx(0.5213)

    transmission = list_literature_spectra(
        "WASP-39 b", spec_type="Transmission", cache_dir=tmp_path
    )
    assert len(transmission) == 2
    assert all(s.spectrum_type == "Transmission" for s in transmission)


def test_list_literature_spectra_empty_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exoarchive,
        "_http_get",
        lambda url, params=None: FakeResponse(
            text=(
                "pl_name,spec_type,authors,num_datapoints,instrument,facility,"
                "minwavelng,maxwavelng,note,bibcode,spec_path\n"
            )
        ),
    )
    assert list_literature_spectra("Nobody b", cache_dir=tmp_path) == []


def test_get_spectrum_downloads_and_caches(tmp_path, monkeypatch, tbl_bytes):
    calls, _ = _fake_http(monkeypatch, tbl_bytes=tbl_bytes)
    provenance = Provenance(analysis_id="EXO-000001")
    spec_path = "40/24/96/78/WASP_39_b_5502_6.tbl"

    spectrum = get_spectrum(spec_path, provenance=provenance, cache_dir=tmp_path)
    assert spectrum.target_id == "WASP-39 b"
    assert spectrum.observation_id == spec_path
    assert len(spectrum.wavelength) == 4
    page_calls = [url for url, _ in calls if url == exoarchive.SPECTRA_APP_URL]
    file_calls = [url for url, _ in calls if "/atmospheres/tab1/data/" in url]
    assert len(page_calls) == 1
    assert len(file_calls) == 1
    assert file_calls[0].endswith(spec_path)

    # cached file is reused: no further network traffic
    again = get_spectrum(spec_path, provenance=provenance, cache_dir=tmp_path)
    assert again == spectrum
    assert len(calls) == 2


def test_get_spectrum_invalid_reference_raises(tmp_path, monkeypatch):
    _fake_http(monkeypatch)
    provenance = Provenance(analysis_id="EXO-000001")
    with pytest.raises(ExoplanetArchiveError, match="spec_path"):
        get_spectrum({}, provenance=provenance, cache_dir=tmp_path)


def test_http_get_wraps_connection_errors(monkeypatch):
    def boom(url, params=None, timeout=None):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(exoarchive.requests, "get", boom)
    with pytest.raises(ExoplanetArchiveError, match="GET .* failed"):
        exoarchive._http_get("https://example.invalid/")
