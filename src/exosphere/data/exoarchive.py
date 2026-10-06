"""NASA Exoplanet Archive access (TAP queries + literature spectrum downloads).

Table/column names are taken from the archive's live TAP service (TAP_SCHEMA
inspection), not guessed:
- ``pscomppars``: planetary and stellar composite parameters.
- ``spectra``: the Atmospheric Spectroscopy table (panel 1 metadata only; the
  per-point data columns are not exposed through TAP).

Because the ``spectra`` table has no id column, the stable spectrum identifier
used throughout ExoSphere is its ``spec_path`` (unique file path).
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pandas as pd
import requests
from pydantic import BaseModel

from exosphere.core.config import load_config
from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders import load_archive_tbl

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
SPECTRA_APP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/atmospheres/nph-firefly?atmospheres"
)
APP_HOST = "https://exoplanetarchive.ipac.caltech.edu"
_REQUEST_TIMEOUT_S = 60

_PSCOMPPARS_COLUMNS = [
    "pl_name",
    "pl_rade",
    "pl_bmasse",
    "pl_bmassprov",
    "pl_eqt",
    "pl_insol",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_orbincl",
    "pl_tranmid",
    "pl_ratror",
    "pl_ratdor",
    "st_rad",
    "st_teff",
    "st_mass",
    "st_logg",
    "st_met",
    "st_spectype",
]

_SPECTRA_COLUMNS = [
    "pl_name",
    "spec_type",
    "authors",
    "num_datapoints",
    "instrument",
    "facility",
    "minwavelng",
    "maxwavelng",
    "note",
    "bibcode",
    "spec_path",
]

_G = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2018)
_M_EARTH_KG = 5.9722e24
_R_EARTH_M = 6.371e6

_INIT_PAGE_RE = re.compile(r"FF_InitPage\s*\(\s*'([^']+)'\s*,\s*'([^']+)'")


class ExoplanetArchiveError(RuntimeError):
    """Raised when the archive cannot be queried or a spectrum cannot be fetched."""


def _http_get(url: str, *, params: dict | None = None) -> requests.Response:
    try:
        response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExoplanetArchiveError(f"GET {url} failed: {exc}") from exc
    return response


def default_cache_dir() -> Path:
    return load_config().data_cache_dir / "exoarchive"


def tap_query_csv(
    query: str, *, cache_dir: Path | None = None, refresh: bool = False
) -> str:
    """Run an ADQL query through the TAP sync endpoint, caching the CSV reply."""
    directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
    cache_path = directory / f"tap_{key}.csv"
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")

    response = _http_get(TAP_URL, params={"query": query, "format": "csv"})
    text = response.text
    if text.lstrip().upper().startswith("ERROR"):
        raise ExoplanetArchiveError(f"TAP query failed: {text.strip()[:300]}")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _escape(value: str) -> str:
    return value.strip().replace("'", "''")


def _frame(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def _opt_number(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _opt_text(row: pd.Series, column: str) -> str | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


class PlanetParams(BaseModel):
    """Planet + host-star parameters from the pscomppars table (units in names)."""

    planet_name: str
    planet_radius_earth: float | None = None  # pl_rade [Earth radius]
    planet_mass_earth: float | None = None  # pl_bmasse [Earth mass]
    mass_provenance: str | None = None  # pl_bmassprov
    equilibrium_temperature_k: float | None = None  # pl_eqt [K]
    insolation_earth: float | None = None  # pl_insol [Earth flux]
    orbital_period_days: float | None = None  # pl_orbper [days]
    semi_major_axis_au: float | None = None  # pl_orbsmax [au]
    eccentricity: float | None = None  # pl_orbeccen
    inclination_deg: float | None = None  # pl_orbincl [deg]
    transit_midpoint_days: float | None = None  # pl_tranmid [days]
    radius_ratio: float | None = None  # pl_ratror (Rp/R*)
    axis_ratio: float | None = None  # pl_ratdor (a/R*)
    stellar_radius_solar: float | None = None  # st_rad [solar radius]
    stellar_teff_k: float | None = None  # st_teff [K]
    stellar_mass_solar: float | None = None  # st_mass [solar mass]
    stellar_logg_cgs: float | None = None  # st_logg [log10(cm/s^2)]
    stellar_metallicity_dex: float | None = None  # st_met [dex]
    stellar_spectype: str | None = None  # st_spectype
    surface_gravity_m_s2: float | None = None  # derived: G*M/R^2 (not an archive column)
    source: str = "NASA Exoplanet Archive TAP table pscomppars"

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


def get_planet_params(
    name: str, *, cache_dir: Path | None = None, refresh: bool = False
) -> PlanetParams:
    """Fetch planet and host-star parameters for ``name`` (exact archive name)."""
    query = (
        f"SELECT {', '.join(_PSCOMPPARS_COLUMNS)} FROM pscomppars "
        f"WHERE pl_name = '{_escape(name)}'"
    )
    text = tap_query_csv(query, cache_dir=cache_dir, refresh=refresh)
    frame = _frame(text)
    if frame.empty:
        raise ExoplanetArchiveError(f"no pscomppars row for planet {name!r}")
    row = frame.iloc[0]

    params = PlanetParams(
        planet_name=str(row.get("pl_name") or name),
        planet_radius_earth=_opt_number(row, "pl_rade"),
        planet_mass_earth=_opt_number(row, "pl_bmasse"),
        mass_provenance=_opt_text(row, "pl_bmassprov"),
        equilibrium_temperature_k=_opt_number(row, "pl_eqt"),
        insolation_earth=_opt_number(row, "pl_insol"),
        orbital_period_days=_opt_number(row, "pl_orbper"),
        semi_major_axis_au=_opt_number(row, "pl_orbsmax"),
        eccentricity=_opt_number(row, "pl_orbeccen"),
        inclination_deg=_opt_number(row, "pl_orbincl"),
        transit_midpoint_days=_opt_number(row, "pl_tranmid"),
        radius_ratio=_opt_number(row, "pl_ratror"),
        axis_ratio=_opt_number(row, "pl_ratdor"),
        stellar_radius_solar=_opt_number(row, "st_rad"),
        stellar_teff_k=_opt_number(row, "st_teff"),
        stellar_mass_solar=_opt_number(row, "st_mass"),
        stellar_logg_cgs=_opt_number(row, "st_logg"),
        stellar_metallicity_dex=_opt_number(row, "st_met"),
        stellar_spectype=_opt_text(row, "st_spectype"),
    )
    if params.planet_mass_earth is not None and params.planet_radius_earth is not None:
        mass_kg = params.planet_mass_earth * _M_EARTH_KG
        radius_m = params.planet_radius_earth * _R_EARTH_M
        params.surface_gravity_m_s2 = _G * mass_kg / radius_m**2
    return params


class LiteratureSpectrum(BaseModel):
    """One row of the Atmospheric Spectroscopy (``spectra``) table."""

    spectrum_id: str  # the row's spec_path; stable unique identifier
    spec_path: str
    planet_name: str
    spectrum_type: str  # Transmission | Eclipse | Direct Imaging
    reference: str | None = None  # authors
    bibcode: str | None = None
    instrument: str | None = None
    facility: str | None = None
    min_wavelength_um: float | None = None
    max_wavelength_um: float | None = None
    num_datapoints: int | None = None
    note: str | None = None


def list_literature_spectra(
    name: str,
    *,
    spec_type: str | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> list[LiteratureSpectrum]:
    """List literature spectra available for ``name`` (optionally one type)."""
    query = (
        f"SELECT {', '.join(_SPECTRA_COLUMNS)} FROM spectra "
        f"WHERE pl_name = '{_escape(name)}'"
    )
    text = tap_query_csv(query, cache_dir=cache_dir, refresh=refresh)
    frame = _frame(text)
    spectra: list[LiteratureSpectrum] = []
    for _, row in frame.iterrows():
        spec_path = _opt_text(row, "spec_path")
        spectrum_type = _opt_text(row, "spec_type")
        if spec_path is None or spectrum_type is None:
            continue
        spectra.append(
            LiteratureSpectrum(
                spectrum_id=spec_path,
                spec_path=spec_path,
                planet_name=str(row.get("pl_name") or name),
                spectrum_type=spectrum_type,
                reference=_opt_text(row, "authors"),
                bibcode=_opt_text(row, "bibcode"),
                instrument=_opt_text(row, "instrument"),
                facility=_opt_text(row, "facility"),
                min_wavelength_um=_opt_number(row, "minwavelng"),
                max_wavelength_um=_opt_number(row, "maxwavelng"),
                num_datapoints=(
                    int(row["num_datapoints"]) if _opt_number(row, "num_datapoints") else None
                ),
                note=_opt_text(row, "note"),
            )
        )
    if spec_type is not None:
        wanted = spec_type.strip().lower()
        spectra = [s for s in spectra if s.spectrum_type.lower() == wanted]
    return spectra


def _spec_path(spectrum: str | dict | LiteratureSpectrum) -> str:
    if isinstance(spectrum, LiteratureSpectrum):
        return spectrum.spec_path
    if isinstance(spectrum, dict):
        for key in ("spec_path", "spectrum_id"):
            value = spectrum.get(key)
            if value:
                return str(value)
        raise ExoplanetArchiveError(
            f"spectrum dict must contain 'spec_path' or 'spectrum_id'; got {sorted(spectrum)}"
        )
    if isinstance(spectrum, str) and spectrum.strip():
        return spectrum.strip()
    raise ExoplanetArchiveError(f"invalid spectrum reference: {spectrum!r}")


def cached_spectrum_path(spec_path: str, cache_dir: Path) -> Path:
    """Local cache path mirroring the archive-relative ``spec_path``."""
    parts = [part for part in spec_path.replace("\\", "/").split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ExoplanetArchiveError(f"invalid spec_path {spec_path!r}")
    return cache_dir / "spectra" / Path(*parts)


def firefly_data_url(spec_path: str) -> str:
    """Resolve the download URL the archive's own UI uses for spectrum files.

    The spectra web app publishes its base path in the FF_InitPage call of the
    page HTML; the file URL is ``<base>/atmospheres/tab1/data/<spec_path>``.
    """
    response = _http_get(SPECTRA_APP_URL)
    match = _INIT_PAGE_RE.search(response.text)
    if not match:
        raise ExoplanetArchiveError(
            "could not determine the spectra file base URL from the archive page"
        )
    base = match.group(2)
    return f"{APP_HOST}{base}/atmospheres/tab1/data/{spec_path}"


def get_spectrum(
    spectrum: str | dict | LiteratureSpectrum,
    *,
    provenance: Provenance,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> Spectrum:
    """Download (or reuse a cached copy of) a literature spectrum and parse it."""
    directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    spec_path = _spec_path(spectrum)
    cache_path = cached_spectrum_path(spec_path, directory)
    if refresh or not cache_path.exists():
        url = firefly_data_url(spec_path)
        payload = _http_get(url).content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
    return load_archive_tbl(
        cache_path,
        provenance=provenance,
        observation_id=spec_path,
    )


def sha256_of_file(path: str | Path) -> str:
    """SHA256 checksum of a file, as lowercase hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ExoplanetArchiveError",
    "LiteratureSpectrum",
    "PlanetParams",
    "TAP_URL",
    "cached_spectrum_path",
    "firefly_data_url",
    "get_planet_params",
    "get_spectrum",
    "list_literature_spectra",
    "sha256_of_file",
    "tap_query_csv",
]
