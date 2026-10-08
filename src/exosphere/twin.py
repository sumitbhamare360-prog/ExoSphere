"""Scientific digital twin parameters (Phase 8).

Builds the "twin parameters" object served by ``GET /analyses/{id}/twin``.
Every value carries a source tag so the UI can label it honestly:

- ``measured`` — read from the catalog (NASA Exoplanet Archive) or the data.
- ``inferred`` — posterior median from the Bayesian retrieval (+ 68% CI).
- ``derived``  — computed from measured/inferred values (formula documented).
- ``assumed``  — default used because the data are missing (value documented).

No geography is produced anywhere here: the twin is orbital + bulk-planet
geometry plus a uniform model atmosphere shell.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np

from exosphere.forward.model import MOL_WEIGHT

# ---------------------------------------------------------------------------
# Physical constants (documented for provenance)
# ---------------------------------------------------------------------------

K_BOLTZMANN_J_K = 1.380649e-23  # Boltzmann constant [J/K]
M_H_KG = 1.6735575e-27  # hydrogen atom mass [kg]
R_JUP_M = 7.1492e7  # Jupiter radius [m]
R_SUN_M = 6.957e8  # solar radius [m]
R_EARTH_M = 6.371e6  # Earth radius [m]
AU_M = 1.495978707e11  # astronomical unit [m]
R_JUP_PER_R_EARTH = 11.209  # Jupiter radii per Earth radius
M_JUP_PER_M_EARTH = 317.828  # Jupiter masses per Earth mass
M_EARTH_KG = 5.9722e24  # Earth mass [kg]
G_SI = 6.67430e-11  # gravitational constant [m^3 kg^-1 s^-2]

MOLECULES: tuple[str, ...] = ("H2O", "CO2", "CO", "CH4", "SO2")

# ---------------------------------------------------------------------------
# Documented fallback defaults (used only when data are missing; tagged assumed)
# ---------------------------------------------------------------------------

ASSUMED_PLANET_RADIUS_R_JUP = 1.0
ASSUMED_GRAVITY_M_S2 = 24.79  # Jupiter's surface gravity; generic gas-giant fallback
ASSUMED_STELLAR_RADIUS_R_SUN = 1.0
ASSUMED_STELLAR_TEFF_K = 5778.0
ASSUMED_INCLINATION_DEG = 90.0  # edge-on; required for a transit to occur
ASSUMED_ECCENTRICITY = 0.0  # circular orbit
ASSUMED_ATMOSPHERE_T_K = 1000.0
ASSUMED_LOG_P_CLOUD = 2.0  # 100 bar: deep enough to be effectively clear
ASSUMED_TRACE_LOG_VMR = -12.0  # negligible abundance for MMW fallback
# A cloud-top pressure is "constrained" when the retrieval actually ran and the
# 68% credible interval is narrower than this (dex). Wider => hatched/uncertain.
CLOUD_CONSTRAINED_CI_WIDTH_DEX = 2.0

SourceTag = Literal["measured", "inferred", "derived", "assumed"]


# ---------------------------------------------------------------------------
# Pure mapping functions (unit-tested in tests/test_twin.py)
# ---------------------------------------------------------------------------


def mean_molecular_weight_g_mol(vmr: dict[str, float]) -> float:
    """Mean molecular weight [g/mol] for constant-with-altitude mixing ratios.

    Same formula as ``forward.model._build_atmosphere``: trace gases plus an
    H2/He background, so the twin matches the forward model exactly.
    """
    from exosphere.forward.model import HE_H2_RATIO

    vmr_sum = sum(vmr.values())
    vmr_h2 = 1.0 - vmr_sum - vmr_sum * HE_H2_RATIO
    vmr_he = vmr_sum * HE_H2_RATIO
    if vmr_h2 < 0:
        raise ValueError(f"VMR sum too large: {vmr_sum:.6f}, no room for H2/He")
    return (
        vmr_h2 * MOL_WEIGHT["H2"]
        + vmr_he * MOL_WEIGHT["He"]
        + sum(vmr[m] * MOL_WEIGHT[m] for m in vmr)
    )


def scale_height_m(t_k: float, mmw_g_mol: float, gravity_m_s2: float) -> float:
    """Pressure scale height [m]: H = k_B * T / (mu * m_H * g).

    Isothermal, hydrostatic atmosphere with mean particle mass mu*m_H.
    """
    if t_k <= 0:
        raise ValueError(f"temperature must be > 0 K, got {t_k}")
    if mmw_g_mol <= 0:
        raise ValueError(f"mean molecular weight must be > 0, got {mmw_g_mol}")
    if gravity_m_s2 <= 0:
        raise ValueError(f"gravity must be > 0, got {gravity_m_s2}")
    return K_BOLTZMANN_J_K * t_k / (mmw_g_mol * M_H_KG * gravity_m_s2)


def transit_depth_fraction(planet_radius_m: float, stellar_radius_m: float) -> float:
    """Transit depth (Rp/Rs)^2 from radii in metres."""
    if stellar_radius_m <= 0:
        raise ValueError("stellar radius must be > 0")
    return (planet_radius_m / stellar_radius_m) ** 2


def surface_gravity_m_s2(mass_earth: float, radius_earth: float) -> float:
    """Surface gravity [m/s^2] from mass/radius in Earth units: g = G*M/R^2."""
    mass_kg = mass_earth * M_EARTH_KG
    radius_m = radius_earth * R_EARTH_M
    return G_SI * mass_kg / radius_m**2


def semi_major_axis_stellar_radii(
    axis_ratio: float | None,
    semi_major_axis_au: float | None,
    stellar_radius_rsun: float | None,
) -> float | None:
    """Semi-major axis in stellar radii.

    Uses the catalog a/R* directly when present; otherwise derives it from
    a[AU]/R*[R_sun]. Returns None when neither is available.
    """
    if axis_ratio is not None:
        return float(axis_ratio)
    if semi_major_axis_au is not None and stellar_radius_rsun:
        return float(semi_major_axis_au * AU_M / (stellar_radius_rsun * R_SUN_M))
    return None


# ---------------------------------------------------------------------------
# Per-molecule modelled contribution (cached; uses the forward model)
# ---------------------------------------------------------------------------

_CONTRIBUTION_CACHE: dict[str, dict[str, float]] = {}
_CONTRIBUTION_CACHE_MAX = 128


def _contribution_cache_key(
    params_tuple: tuple[float, ...], wl: np.ndarray, bands_version: str
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(params_tuple, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(wl, dtype=np.float64).tobytes())
    digest.update(bands_version.encode("utf-8"))
    return digest.hexdigest()


def molecule_contributions(
    best_fit: dict[str, float],
    gravity_m_s2: float,
    stellar_radius_rsun: float,
    wavelength_um: np.ndarray,
    bands: dict[str, list[tuple[float, float]]],
    bands_version: str = "molecule-bands-1.0.0",
) -> dict[str, float]:
    """Fraction of each molecule's in-band spectral signal from the best-fit model.

    For molecule M: contribution = sum_inband(|d_full - d_noM|)
    / sum_inband(|d_full - median(d_full)|), clipped to [0, 1].
    ``d_noM`` is the best-fit model with M's log VMR set to -12 (negligible).
    A zero denominator (featureless model in the band) yields 0.0.

    This is model-derived structure attribution, NOT a geographic map and NOT
    a detection claim: overlapping absorbers (e.g. H2O wings in CH4 windows)
    legitimately raise a window's score (see DECISIONS.md 14).
    """
    from exosphere.forward.model import ModelParams, PlanetFixed, transmission_spectrum

    wl = np.asarray(wavelength_um, dtype=np.float64)
    params_tuple = (
        best_fit["T"],
        best_fit["log_h2o"],
        best_fit["log_co2"],
        best_fit["log_co"],
        best_fit["log_ch4"],
        best_fit["log_so2"],
        best_fit["r_ref"],
        best_fit["log_p_cloud"],
    )
    key = _contribution_cache_key(params_tuple, wl, bands_version)
    cached = _CONTRIBUTION_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    fixed = PlanetFixed(
        gravity_m_s2=gravity_m_s2,
        stellar_radius_rsun=stellar_radius_rsun,
    )
    full = ModelParams(
        T=best_fit["T"],
        log_h2o=best_fit["log_h2o"],
        log_co2=best_fit["log_co2"],
        log_co=best_fit["log_co"],
        log_ch4=best_fit["log_ch4"],
        log_so2=best_fit["log_so2"],
        r_ref=best_fit["r_ref"],
        log_p_cloud=best_fit["log_p_cloud"],
    )
    _, d_full, _ = transmission_spectrum(full, fixed, wavelength_grid=wl)
    continuum = float(np.median(d_full))

    contributions: dict[str, float] = {}
    for mol in MOLECULES:
        removed = ModelParams(
            T=best_fit["T"],
            log_h2o=best_fit["log_h2o"] if mol != "H2O" else -12.0,
            log_co2=best_fit["log_co2"] if mol != "CO2" else -12.0,
            log_co=best_fit["log_co"] if mol != "CO" else -12.0,
            log_ch4=best_fit["log_ch4"] if mol != "CH4" else -12.0,
            log_so2=best_fit["log_so2"] if mol != "SO2" else -12.0,
            r_ref=best_fit["r_ref"],
            log_p_cloud=best_fit["log_p_cloud"],
        )
        _, d_removed, _ = transmission_spectrum(removed, fixed, wavelength_grid=wl)
        mask = np.zeros(wl.shape, dtype=bool)
        for lo, hi in bands.get(mol, []):
            mask |= (wl >= lo) & (wl <= hi)
        if not bool(np.any(mask)):
            contributions[mol] = 0.0
            continue
        numerator = float(np.sum(np.abs(d_full[mask] - d_removed[mask])))
        denominator = float(np.sum(np.abs(d_full[mask] - continuum)))
        if denominator <= 0:
            contributions[mol] = 0.0
        else:
            contributions[mol] = float(min(1.0, max(0.0, numerator / denominator)))

    if len(_CONTRIBUTION_CACHE) >= _CONTRIBUTION_CACHE_MAX:
        _CONTRIBUTION_CACHE.pop(next(iter(_CONTRIBUTION_CACHE)))
    _CONTRIBUTION_CACHE[key] = dict(contributions)
    return contributions


def _bands_covering(
    bands: dict[str, list[tuple[float, float]]],
    wl_min: float,
    wl_max: float,
) -> dict[str, bool]:
    """Whether each molecule has any band window overlapping [wl_min, wl_max]."""
    covered: dict[str, bool] = {}
    for mol in MOLECULES:
        covered[mol] = any(
            hi >= wl_min and lo <= wl_max for lo, hi in bands.get(mol, [])
        )
    return covered


# ---------------------------------------------------------------------------
# Twin parameters builder
# ---------------------------------------------------------------------------


def _param(
    name: str,
    label: str,
    value: float | None,
    unit: str,
    source: SourceTag,
    ci_68: list[float] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "value": value,
        "unit": unit,
        "source": source,
        "ci_68": ci_68,
        "note": note,
    }


def build_twin_parameters(
    *,
    analysis_id: str,
    planet_name: str,
    catalog: dict[str, Any],
    retrieval: dict[str, Any] | None = None,
    spectrum: dict[str, Any] | None = None,
    bands: dict[str, list[tuple[float, float]]] | None = None,
    bands_version: str = "molecule-bands-1.0.0",
) -> dict[str, Any]:
    """Assemble the source-tagged twin parameters object.

    Args:
        analysis_id: EXO-XXXXXX id.
        planet_name: planet name.
        catalog: catalog fields (pl_rade, pl_bmasse, st_rad, st_teff,
            pl_orbsmax, pl_orbper, pl_orbincl, pl_orbeccen, pl_eqt, pl_ratdor,
            surface_gravity_m_s2). Missing keys / None = missing data.
        retrieval: optional ``{"median": {param: value},
            "ci_68": {param: [lo, hi]}, "best_fit": {param: value}}``
            with the 8 forward-model parameter names.
        spectrum: optional ``{"wavelength": [...], "transmission": [...]}``.
        bands: molecule band windows; defaults to config/molecule_bands.yaml.
        bands_version: band config version string (cache key + provenance).
    """
    from exosphere.quality.assess import load_quality_config

    if bands is None:
        cfg = load_quality_config()
        bands = {mol: [tuple(w) for w in wins] for mol, wins in cfg.molecule_bands.items()}
        bands_version = cfg.version

    retrieval = retrieval or {}
    median: dict[str, float] = retrieval.get("median", {})
    ci_68: dict[str, list[float]] = retrieval.get("ci_68", {})
    best_fit: dict[str, float] = retrieval.get("best_fit", {})
    has_retrieval = bool(median)

    def ci_of(param: str) -> list[float] | None:
        ci = ci_68.get(param)
        if ci is None:
            return None
        return [float(ci[0]), float(ci[1])]

    params: list[dict[str, Any]] = []

    # --- planet radius [R_Jup]: retrieval wins, else catalog, else assumed ---
    if "r_ref" in median:
        r_jup = float(median["r_ref"])
        r_source: SourceTag = "inferred"
        r_note: str | None = "retrieval posterior median"
        r_ci = ci_of("r_ref")
    elif catalog.get("pl_rade") is not None:
        r_jup = float(catalog["pl_rade"]) / R_JUP_PER_R_EARTH
        r_source = "measured"
        r_note = "catalog pl_rade / 11.209"
        r_ci = None
    else:
        r_jup = ASSUMED_PLANET_RADIUS_R_JUP
        r_source = "assumed"
        r_note = f"default {ASSUMED_PLANET_RADIUS_R_JUP} R_Jup (no catalog radius)"
        r_ci = None
    params.append(_param("planet_radius", "Planet radius", r_jup, "R_Jup", r_source, r_ci, r_note))

    # --- planet mass [M_Jup]: catalog only ---
    if catalog.get("pl_bmasse") is not None:
        m_jup = float(catalog["pl_bmasse"]) / M_JUP_PER_M_EARTH
        m_source: SourceTag = "measured"
        m_note = "catalog pl_bmasse / 317.828"
    else:
        m_jup = None
        m_source = "assumed"
        m_note = "no catalog mass"
    params.append(_param("planet_mass", "Planet mass", m_jup, "M_Jup", m_source, None, m_note))

    # --- surface gravity [m/s^2]: derived from catalog M+R, else assumed ---
    if catalog.get("pl_bmasse") is not None and catalog.get("pl_rade") is not None:
        g = surface_gravity_m_s2(float(catalog["pl_bmasse"]), float(catalog["pl_rade"]))
        g_source: SourceTag = "derived"
        g_note: str | None = "g = G*M/R^2 from catalog mass and radius"
    elif catalog.get("surface_gravity_m_s2") is not None:
        g = float(catalog["surface_gravity_m_s2"])
        g_source = "derived"
        g_note = "stored catalog-derived gravity"
    else:
        g = ASSUMED_GRAVITY_M_S2
        g_source = "assumed"
        g_note = f"Jupiter fallback {ASSUMED_GRAVITY_M_S2} m/s^2 (no catalog mass/radius)"
    params.append(_param("surface_gravity", "Surface gravity", g, "m/s^2", g_source, None, g_note))

    # --- star ---
    st_rad = catalog.get("st_rad")
    params.append(
        _param(
            "stellar_radius",
            "Stellar radius",
            float(st_rad) if st_rad is not None else ASSUMED_STELLAR_RADIUS_R_SUN,
            "R_Sun",
            "measured" if st_rad is not None else "assumed",
            None,
            "catalog st_rad" if st_rad is not None else "solar fallback (no catalog value)",
        )
    )
    st_teff = catalog.get("st_teff")
    params.append(
        _param(
            "stellar_teff",
            "Stellar effective temperature",
            float(st_teff) if st_teff is not None else ASSUMED_STELLAR_TEFF_K,
            "K",
            "measured" if st_teff is not None else "assumed",
            None,
            "catalog st_teff" if st_teff is not None else "solar fallback (no catalog value)",
        )
    )

    # --- orbit ---
    sma_au = catalog.get("pl_orbsmax")
    params.append(
        _param(
            "semi_major_axis",
            "Semi-major axis",
            float(sma_au) if sma_au is not None else None,
            "AU",
            "measured" if sma_au is not None else "assumed",
            None,
            "catalog pl_orbsmax" if sma_au is not None else "not in catalog",
        )
    )
    a_rs = semi_major_axis_stellar_radii(
        catalog.get("pl_ratdor"),
        sma_au,
        st_rad,
    )
    params.append(
        _param(
            "semi_major_axis_stellar_radii",
            "Semi-major axis",
            a_rs,
            "R_*",
            "measured"
            if catalog.get("pl_ratdor") is not None
            else ("derived" if a_rs is not None else "assumed"),
            None,
            "catalog pl_ratdor (a/R*)"
            if catalog.get("pl_ratdor") is not None
            else ("a[AU]/R*[R_Sun]" if a_rs is not None else "not in catalog"),
        )
    )
    period = catalog.get("pl_orbper")
    params.append(
        _param(
            "orbital_period",
            "Orbital period",
            float(period) if period is not None else None,
            "days",
            "measured" if period is not None else "assumed",
            None,
            "catalog pl_orbper" if period is not None else "not in catalog",
        )
    )
    incl = catalog.get("pl_orbincl")
    params.append(
        _param(
            "inclination",
            "Orbital inclination",
            float(incl) if incl is not None else ASSUMED_INCLINATION_DEG,
            "deg",
            "measured" if incl is not None else "assumed",
            None,
            "catalog pl_orbincl"
            if incl is not None
            else f"edge-on default {ASSUMED_INCLINATION_DEG} deg (a transit must occur)",
        )
    )
    ecc = catalog.get("pl_orbeccen")
    params.append(
        _param(
            "eccentricity",
            "Orbital eccentricity",
            float(ecc) if ecc is not None else ASSUMED_ECCENTRICITY,
            "dimensionless",
            "measured" if ecc is not None else "assumed",
            None,
            "catalog pl_orbeccen"
            if ecc is not None
            else f"circular default {ASSUMED_ECCENTRICITY} (no catalog value)",
        )
    )
    teq = catalog.get("pl_eqt")
    params.append(
        _param(
            "equilibrium_temperature",
            "Equilibrium temperature",
            float(teq) if teq is not None else None,
            "K",
            "measured" if teq is not None else "assumed",
            None,
            "catalog pl_eqt" if teq is not None else "not in catalog",
        )
    )

    # --- retrieved atmosphere temperature [K] ---
    if "T" in median:
        t_k = float(median["T"])
        t_source: SourceTag = "inferred"
        t_note: str | None = "retrieval posterior median"
        t_ci = ci_of("T")
    elif teq is not None:
        t_k = float(teq)
        t_source = "measured"
        t_note = "catalog equilibrium temperature used as atmosphere-temperature fallback"
        t_ci = None
    else:
        t_k = ASSUMED_ATMOSPHERE_T_K
        t_source = "assumed"
        t_note = f"default {ASSUMED_ATMOSPHERE_T_K} K (no retrieval, no catalog Teq)"
        t_ci = None
    params.append(
        _param("atmosphere_temperature", "Atmosphere temperature", t_k, "K", t_source, t_ci, t_note)
    )

    # --- cloud-top pressure [log10(bar)] ---
    if "log_p_cloud" in median:
        lp = float(median["log_p_cloud"])
        lp_ci = ci_of("log_p_cloud")
        lp_source: SourceTag = "inferred"
        if lp_ci is not None and (lp_ci[1] - lp_ci[0]) < CLOUD_CONSTRAINED_CI_WIDTH_DEX:
            lp_note: str | None = "retrieval posterior median (constrained)"
            cloud_constrained = True
        else:
            lp_note = "retrieval posterior median (unconstrained: wide CI)"
            cloud_constrained = False
    else:
        lp = ASSUMED_LOG_P_CLOUD
        lp_ci = None
        lp_source = "assumed"
        lp_note = f"deep default {ASSUMED_LOG_P_CLOUD} (effectively clear; no retrieval)"
        cloud_constrained = False
    cloud_param = _param(
        "cloud_top_pressure", "Cloud-top pressure", lp, "log10(bar)", lp_source, lp_ci, lp_note
    )
    cloud_param["constrained"] = cloud_constrained
    params.append(cloud_param)

    # --- composition -> MMW [g/mol]: derived ---
    if best_fit:
        vmr = {
            "H2O": 10.0 ** float(best_fit.get("log_h2o", ASSUMED_TRACE_LOG_VMR)),
            "CO2": 10.0 ** float(best_fit.get("log_co2", ASSUMED_TRACE_LOG_VMR)),
            "CO": 10.0 ** float(best_fit.get("log_co", ASSUMED_TRACE_LOG_VMR)),
            "CH4": 10.0 ** float(best_fit.get("log_ch4", ASSUMED_TRACE_LOG_VMR)),
            "SO2": 10.0 ** float(best_fit.get("log_so2", ASSUMED_TRACE_LOG_VMR)),
        }
        mmw_note = "from retrieval best-fit composition (H2/He background)"
    else:
        vmr = {m: 10.0**ASSUMED_TRACE_LOG_VMR for m in MOLECULES}
        mmw_note = "trace-gas composition assumed (no retrieval)"
    mmw = mean_molecular_weight_g_mol(vmr)
    mmw_param = _param(
        "mean_molecular_weight",
        "Mean molecular weight",
        mmw,
        "g/mol",
        "derived",
        None,
        mmw_note,
    )
    params.append(mmw_param)

    # --- scale height [km]: H = kB*T/(mu*mH*g), derived ---
    h_m = scale_height_m(t_k, mmw, g)
    h_sources = f"T({t_source}) mu({ 'retrieval' if best_fit else 'assumed'}) g({g_source})"
    params.append(
        _param(
            "scale_height",
            "Atmospheric scale height",
            h_m / 1000.0,
            "km",
            "derived",
            None,
            f"H = kB*T/(mu*mH*g); inputs: {h_sources}",
        )
    )

    # --- transit depths ---
    obs_depth: float | None = None
    wl_obs: np.ndarray | None = None
    if spectrum is not None:
        wl_obs = np.asarray(spectrum["wavelength"], dtype=np.float64)
        obs_depth = float(np.median(np.asarray(spectrum["transmission"], dtype=np.float64)))
    params.append(
        _param(
            "transit_depth_observed",
            "Observed transit depth",
            obs_depth,
            "fractional (Rp/Rs)^2",
            "measured" if obs_depth is not None else "assumed",
            None,
            "median of observed transmission spectrum"
            if obs_depth is not None
            else "no spectrum available",
        )
    )
    st_rad_eff = float(st_rad) if st_rad is not None else ASSUMED_STELLAR_RADIUS_R_SUN
    model_depth = transit_depth_fraction(r_jup * R_JUP_M, st_rad_eff * R_SUN_M)
    params.append(
        _param(
            "transit_depth_model",
            "Model transit depth",
            model_depth,
            "fractional (Rp/Rs)^2",
            "derived",
            None,
            "(Rp/Rs)^2 from twin planet/star radii",
        )
    )

    # --- per-molecule modelled contributions ---
    molecules: list[dict[str, Any]] = []
    if wl_obs is not None:
        wl_min, wl_max = float(wl_obs.min()), float(wl_obs.max())
    else:
        wl_min, wl_max = 0.6, 5.3
    coverage = _bands_covering(bands, wl_min, wl_max)
    if best_fit and wl_obs is not None:
        contribs = molecule_contributions(
            best_fit, g, st_rad_eff, wl_obs, bands, bands_version
        )
        for mol in MOLECULES:
            molecules.append(
                {
                    "molecule": mol,
                    "contribution_fraction": contribs[mol],
                    "in_band": coverage[mol],
                    "note": "model-derived signal fraction in this molecule's band windows",
                }
            )
    else:
        for mol in MOLECULES:
            molecules.append(
                {
                    "molecule": mol,
                    "contribution_fraction": 0.0,
                    "in_band": coverage[mol],
                    "note": "no retrieval best fit — contribution unavailable",
                }
            )

    return {
        "analysis_id": analysis_id,
        "planet_name": planet_name,
        "parameters": params,
        "molecules": molecules,
        "meta": {
            "scale_height_formula": "H = kB*T/(mu*mH*g)",
            "bands_version": bands_version,
            "cloud_constrained": cloud_constrained,
            "has_retrieval": has_retrieval,
        },
    }
