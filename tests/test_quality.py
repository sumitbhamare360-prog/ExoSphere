"""Phase 2 tests: data-quality assessment (quality/assess.py)."""

from __future__ import annotations

import json

import numpy as np

from conftest import make_feature_spectrum, make_synthetic_spectrum
from exosphere.quality.assess import (
    DEFAULT_QUALITY_CONFIG_PATH,
    QualityConfig,
    assess,
    load_quality_config,
    running_median,
)


def test_config_loads_bands_and_thresholds():
    cfg = QualityConfig.from_yaml()
    assert cfg.path == DEFAULT_QUALITY_CONFIG_PATH
    assert set(cfg.molecule_bands) == {"H2O", "CO2", "CO", "CH4", "SO2"}
    lo_nominal, hi_nominal = cfg.nominal_range
    assert (lo_nominal, hi_nominal) == (0.6, 5.3)
    for molecule, windows in cfg.molecule_bands.items():
        for lo, hi in windows:
            assert lo < hi, f"{molecule} window {lo}-{hi} not ascending"
            assert 0.5 <= lo and hi <= 5.4, f"{molecule} window outside PRISM range"
    for key in (
        "molecule_coverage_good",
        "molecule_snr_good",
        "molecule_snr_limited",
        "coverage_wavelength_coverage_good",
        "coverage_snr_good",
    ):
        assert key in cfg.thresholds, key


def test_running_median_is_nan_safe():
    values = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    out = running_median(values, 3)
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[4])
    assert np.all(np.isfinite(running_median(values[[0, 2, 3, 4]], 3)))


def test_featureless_spectrum_rates_low():
    rng = np.random.default_rng(0)
    n = 300
    wave = np.linspace(0.6, 5.3, n)
    depth = 0.015 + rng.normal(0, 2e-5, n)
    spectrum = make_synthetic_spectrum(wave, depth, np.full(n, 2e-5))
    report = assess(spectrum)
    # No spectral structure: no molecule window can support a claim.
    assert report.max_band_snr is not None and report.max_band_snr < 3.0
    assert report.suitability in {"POOR", "LIMITED"}
    assert all(rating in {"POOR", "LIMITED"} for rating in report.molecule_ratings.values())


def test_good_full_range_spectrum_rates_good_everywhere():
    wave, depth, sigma = make_feature_spectrum()
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    report = assess(spectrum)
    assert report.wavelength_coverage_fraction > 0.99
    assert report.suitability == "GOOD"
    assert all(rating == "GOOD" for rating in report.molecule_ratings.values())
    assert report.median_snr is not None and report.median_snr > 0.5
    for value in report.band_snr.values():
        assert value is not None and value >= 5.0


def test_missing_4p2_4p4_window_degrades_co2_co_so2_not_h2o():
    """Spectrum without the 4.2-4.4 um range: CO2/CO/SO2 <= LIMITED, H2O GOOD."""
    wave, depth, sigma = make_feature_spectrum(hi=4.1, molecules=("H2O", "CO2"))
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    report = assess(spectrum)
    assert report.molecule_ratings["H2O"] == "GOOD"
    for molecule in ("CO2", "CO", "SO2"):
        assert report.molecule_ratings[molecule] in {"POOR", "LIMITED"}, (
            f"{molecule} should be degraded, got {report.molecule_ratings[molecule]}"
        )
    # Coverage alone explains CO2 (2 of 3 windows) and CO (1 of 2).
    by_name = {m.molecule: m for m in report.molecules}
    assert 0.3 <= by_name["CO2"].coverage_fraction < 0.6
    assert by_name["CO"].coverage_fraction < 0.3


def test_report_is_json_serializable_round_trip():
    wave, depth, sigma = make_feature_spectrum()
    depth[10] = depth[10] + 0.004  # injected spike
    depth[20] = np.nan  # injected gap point
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    report = assess(spectrum)
    payload = report.to_json()
    parsed = json.loads(payload)
    assert parsed["suitability"] in {"GOOD", "LIMITED", "POOR"}
    assert parsed["molecule_ratings"]["H2O"] in {"GOOD", "LIMITED", "POOR"}
    restored = type(report).from_json(payload)
    assert restored.model_dump() == report.model_dump()


def test_outliers_gaps_and_bad_uncertainties_are_reported():
    n = 200
    wave = np.linspace(0.6, 5.3, n)
    depth = 0.015 + 0.0005 * np.sin((wave - 0.6) * 2.0)
    sigma = np.full(n, 2e-5)
    depth[50] = depth[50] + 0.005  # spike
    depth[60] = np.nan  # gap
    sigma[70] = np.nan  # bad uncertainty
    sigma[80] = 1e-13  # implausibly tiny
    flags = ["OK"] * n
    flags[90] = "SATURATED"
    spectrum = make_synthetic_spectrum(wave, depth, sigma, quality_flags=flags)
    before = spectrum.model_dump()
    report = assess(spectrum)

    assert spectrum.model_dump() == before, "assess() must not mutate its input"
    assert report.outlier_count >= 1
    assert report.nan_fraction > 0
    assert report.flagged_fraction > 0
    assert report.uncertainty.nan_count >= 1
    assert report.uncertainty.tiny_count >= 1
    assert report.bad_point_fraction > 0
    assert report.suitability != "GOOD"


def test_wavelength_gap_lowers_coverage_and_ratings():
    gap_lo, gap_hi = 2.0, 4.0
    wave, depth, sigma = make_feature_spectrum()
    keep = (wave < gap_lo) | (wave > gap_hi)
    spectrum = make_synthetic_spectrum(wave[keep], depth[keep], sigma[keep])
    report = assess(spectrum)
    # 0.6-2.0 and 4.0-5.3 of the nominal 0.6-5.3 range => 2.7/4.7 covered.
    assert 0.5 < report.wavelength_coverage_fraction < 0.7
    by_name = {m.molecule: m for m in report.molecules}
    # CH4 (1.6-1.8, 2.2-2.4, 3.2-3.5) loses most of its windows to the gap.
    assert by_name["CH4"].coverage_fraction < 0.6
    assert report.molecule_ratings["CH4"] in {"POOR", "LIMITED"}


def test_effective_resolving_power_matches_bin_widths():
    n = 100
    wave = np.linspace(1.0, 2.0, n)
    spacing = float(wave[1] - wave[0])
    spectrum = make_synthetic_spectrum(wave, np.full(n, 0.015), np.full(n, 1e-5))
    report = assess(spectrum)
    expected = float(np.median(wave / spacing))
    assert report.effective_resolving_power is not None
    assert abs(report.effective_resolving_power - expected) / expected < 0.01


def test_quality_config_default_cache_and_yaml_file_exists():
    cfg = load_quality_config()
    assert cfg.version == "molecule-bands-1.0.0"
    assert cfg.continuum_window_points >= 3 and cfg.continuum_window_points % 2 == 1
    assert cfg.outlier_n_sigma > 0
    raw = DEFAULT_QUALITY_CONFIG_PATH.read_text(encoding="utf-8")
    assert "SO2" in raw and "H2O" in raw


def test_assess_handles_emptyish_and_single_point_spectra():
    spectrum = make_synthetic_spectrum([1.0], [0.015], [1e-5], bin_width=0.1)
    report = assess(spectrum)
    assert report.n_points == 1
    assert report.suitability == "POOR"
    assert all(rating == "POOR" for rating in report.molecule_ratings.values())
    json.dumps(json.loads(report.to_json()))


def test_flagged_points_are_excluded_from_usable_coverage():
    n = 400
    wave, depth, sigma = make_feature_spectrum()
    flags = ["OK"] * n
    # Flag every point inside the first H2O window: usable coverage must drop.
    config = load_quality_config()
    lo, hi = config.molecule_bands["H2O"][0]
    for index, value in enumerate(wave):
        if lo <= value <= hi:
            flags[index] = "BAD"
    spectrum = make_synthetic_spectrum(wave, depth, sigma, quality_flags=flags)
    report = assess(spectrum)
    by_name = {m.molecule: m for m in report.molecules}
    assert by_name["H2O"].coverage_fraction < 1.0
    assert report.flagged_fraction > 0


def test_thresholds_are_embedded_in_report_for_provenance():
    spectrum = make_synthetic_spectrum(*make_feature_spectrum())
    report = assess(spectrum)
    assert report.config_version == "molecule-bands-1.0.0"
    assert "molecule_snr_good" in report.thresholds
    assert report.method["molecule_rating"]
    assert report.method["outliers"]
