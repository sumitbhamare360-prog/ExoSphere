"""Phase 2 tests: preprocessing (preprocess/clean.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_feature_spectrum, make_synthetic_spectrum
from exosphere.preprocess.clean import (
    PREPROCESS_VERSION,
    CleanOptions,
    PreprocessLog,
    clean,
)


def test_clean_removes_nan_flagged_and_spike_with_original_indices():
    n = 200
    wave = np.linspace(0.6, 5.3, n)
    depth = 0.015 + 0.0004 * np.sin((wave - 0.6) * 3.0)
    sigma = np.full(n, 2e-5)
    depth[25] = np.nan  # gap
    depth[100] = depth[100] + 0.006  # isolated spike
    flags = ["OK"] * n
    flags[150] = "SUSPECT"
    spectrum = make_synthetic_spectrum(wave, depth, sigma, quality_flags=flags)

    cleaned, log = clean(spectrum)

    reasons = {(item.index, item.reason) for item in log.removed}
    assert (25, "nan") in reasons
    assert (150, "flagged") in reasons
    # the spike plus its two contaminated neighbours are clipped
    spike_outliers = {i for i, reason in reasons if reason == "outlier"}
    assert 100 in spike_outliers
    assert len(spike_outliers) <= 3
    assert log.n_input == n
    assert log.n_output == n - len(log.removed)
    assert len(cleaned.wavelength) == log.n_output
    assert all(
        a < b
        for a, b in zip(cleaned.wavelength, cleaned.wavelength[1:], strict=False)
    ), "cleaned wavelengths must stay strictly ascending"


def test_clean_does_not_mutate_input():
    wave, depth, sigma = make_feature_spectrum(n_points=150)
    depth[10] = depth[10] + 0.005
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    before = spectrum.model_dump()
    clean(spectrum, CleanOptions(rebin="count", bin_count=50))
    assert spectrum.model_dump() == before


def test_clean_writes_preprocessing_version_into_provenance_only_on_output():
    wave, depth, sigma = make_feature_spectrum(n_points=120)
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    assert spectrum.provenance.preprocessing_version is None
    cleaned, log = clean(spectrum)
    assert cleaned.provenance.preprocessing_version == PREPROCESS_VERSION
    assert log.preprocessing_version == PREPROCESS_VERSION
    assert spectrum.provenance.preprocessing_version is None
    assert cleaned.provenance.analysis_id == spectrum.provenance.analysis_id
    assert cleaned.observation_id == spectrum.observation_id


def test_rebin_weighted_mean_and_error_propagation_analytic():
    """Verify new_depth = sum(w d)/sum(w) and new_sigma = 1/sqrt(sum w)."""
    wave = np.array([1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3])
    depth = np.array([0.010, 0.012, 0.014, 0.016, 0.020, 0.022, 0.024, 0.026])
    sigma = np.array([1e-5, 2e-5, 4e-5, 5e-5, 1e-5, 2e-5, 4e-5, 5e-5])
    spectrum = make_synthetic_spectrum(wave, depth, sigma)

    cleaned, log = clean(
        spectrum, CleanOptions(sigma_clip=False, rebin="count", bin_count=2)
    )
    assert log.rebin is not None
    groups = {g.output_index: g.input_indices for g in log.rebin.groups}
    assert len(groups) == 2

    for output_index, input_indices in groups.items():
        weights = 1.0 / np.square(sigma[input_indices])
        expected_depth = float(np.sum(weights * depth[input_indices]) / np.sum(weights))
        expected_sigma = float(1.0 / np.sqrt(np.sum(weights)))
        assert cleaned.transmission[output_index] == pytest.approx(
            expected_depth, rel=1e-12
        )
        assert cleaned.uncertainty[output_index] == pytest.approx(
            expected_sigma, rel=1e-12
        )
        # uncertainty of the mean is always smaller than the best single point
        assert cleaned.uncertainty[output_index] <= min(sigma[input_indices]) + 1e-15


def test_rebin_count_produces_correct_edges_and_centers():
    wave, depth, sigma = make_feature_spectrum(n_points=200)
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    cleaned, log = clean(
        spectrum, CleanOptions(sigma_clip=False, rebin="count", bin_count=40)
    )
    assert log.rebin is not None and log.rebin.n_output == 40
    assert len(cleaned.wavelength_bin_edges) == len(cleaned.wavelength) + 1
    edges = np.asarray(cleaned.wavelength_bin_edges)
    assert np.all(np.diff(edges) > 0)
    centers = np.asarray(cleaned.wavelength)
    assert np.all(np.diff(centers) > 0)
    # every output bin is the midpoint of its grid cell
    assert centers[0] > edges[0] and centers[-1] < edges[-1]


def test_rebin_resolution_drops_empty_bins_and_logs_them():
    wave, depth, sigma = make_feature_spectrum(n_points=300)
    keep = (wave < 2.0) | (wave > 3.0)
    spectrum = make_synthetic_spectrum(wave[keep], depth[keep], sigma[keep])
    cleaned, log = clean(
        spectrum,
        CleanOptions(sigma_clip=False, rebin="resolution", target_resolution_um=0.05),
    )
    assert log.rebin is not None
    assert log.rebin.mode == "resolution"
    assert len(log.rebin.empty_bin_centers) > 0
    assert log.rebin.n_output < len(log.rebin.grid_edges) - 1
    assert len(cleaned.wavelength) == log.rebin.n_output
    assert len(cleaned.wavelength_bin_edges) == len(cleaned.wavelength) + 1
    # no fabricated points: each output bin consumed at least one input point
    assert all(group.input_indices for group in log.rebin.groups)
    # total assigned inputs == points that entered the rebin
    assigned = sum(len(g.input_indices) for g in log.rebin.groups)
    assert assigned == log.rebin.n_input


def test_rebin_requires_its_parameters():
    wave, depth, sigma = make_feature_spectrum(n_points=50)
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    with pytest.raises(ValueError, match="target_resolution_um"):
        clean(spectrum, CleanOptions(sigma_clip=False, rebin="resolution"))
    with pytest.raises(ValueError, match="bin_count"):
        clean(spectrum, CleanOptions(sigma_clip=False, rebin="count"))


def test_bad_uncertainties_are_dropped():
    n = 150
    wave = np.linspace(0.6, 5.3, n)
    depth = 0.015 + 0.0003 * np.sin((wave - 0.6) * 4.0)
    sigma = np.full(n, 2e-5)
    sigma[40] = 1e-13  # implausibly tiny
    sigma[50] = 1.0  # huge: way beyond 10% of the median depth
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    cleaned, log = clean(spectrum)

    bad = [item.index for item in log.removed if item.reason == "bad_uncertainty"]
    assert set(bad) == {40, 50}
    assert len(cleaned.uncertainty) == n - 2
    assert all(u >= 1e-10 for u in cleaned.uncertainty)
    assert all(u <= 0.1 * 0.015 for u in cleaned.uncertainty)


def test_clean_raises_when_everything_is_removed():
    n = 30
    wave = np.linspace(0.6, 5.3, n)
    spectrum = make_synthetic_spectrum(
        wave, np.full(n, np.nan), np.full(n, 2e-5)
    )
    with pytest.raises(ValueError, match="removed every point"):
        clean(spectrum)


def test_preprocess_log_is_json_serializable():
    wave, depth, sigma = make_feature_spectrum(n_points=100)
    depth[5] = depth[5] + 0.005
    spectrum = make_synthetic_spectrum(wave, depth, sigma)
    _, log = clean(spectrum, CleanOptions(rebin="count", bin_count=30))
    payload = log.to_json()
    parsed = json.loads(payload)
    assert parsed["preprocessing_version"] == PREPROCESS_VERSION
    assert parsed["n_input"] == 100
    assert parsed["parameters"]["rebin"] == "count"
    # some grid cells can be empty after the outlier removal (they are dropped)
    assert 25 <= parsed["rebin"]["n_output"] <= 30
    assert isinstance(log, PreprocessLog)
    restored = PreprocessLog.model_validate_json(payload)
    assert restored.model_dump() == log.model_dump()


def test_clean_options_are_recorded_and_configurable():
    n = 80
    wave = np.linspace(0.6, 5.3, n)
    depth = 0.015 + 0.0004 * np.sin((wave - 0.6) * 2.0)  # smooth, no spikes
    spectrum = make_synthetic_spectrum(wave, depth, np.full(n, 2e-5))
    options = CleanOptions(clip_n_sigma=3.0, sigma_clip=True, drop_flagged=True)
    _, log = clean(spectrum, options)
    assert log.parameters["clip_n_sigma"] == 3.0
    assert log.parameters["drop_flagged"] is True
    assert log.parameters["sigma_clip"] is True
    # smooth spectrum: a 3-sigma clip must remove nothing
    assert log.n_output == 80
    assert log.removal_counts == {}
