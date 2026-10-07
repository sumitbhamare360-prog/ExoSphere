"""Unit tests for the Spectrum loaders (archive .tbl, CSV, x1d-style FITS)."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from conftest import PRISM_RATIO_ROWS, PRISM_SOURCE_ROWS, make_ipac_tbl
from exosphere.data.loaders import (
    bin_edges_from_centers,
    depth_to_fraction,
    load_archive_tbl,
    load_csv,
    load_x1d_fits,
    wavelength_to_um,
)


def test_load_archive_tbl_converts_source_units(ipac_tbl_path, provenance):
    spectrum = load_archive_tbl(ipac_tbl_path, provenance=provenance, observation_id="obs-1")

    source_wavelength = [float(row[0]) for row in PRISM_SOURCE_ROWS]
    source_depth = [float(row[2]) / 100.0 for row in PRISM_SOURCE_ROWS]
    source_uncertainty = [
        (abs(float(row[3])) + abs(float(row[4]))) / 2.0 / 100.0 for row in PRISM_SOURCE_ROWS
    ]

    np.testing.assert_allclose(spectrum.wavelength, source_wavelength, rtol=1e-12)
    np.testing.assert_allclose(spectrum.transmission, source_depth, rtol=1e-12)
    np.testing.assert_allclose(spectrum.uncertainty, source_uncertainty, rtol=1e-12)
    assert spectrum.target_id == "WASP-39 b"
    assert spectrum.instrument == "Near Infrared Spectrograph (NIRSpec)"
    assert spectrum.observation_id == "obs-1"
    assert spectrum.quality_flags == ["OK"] * 4
    assert len(spectrum.wavelength_bin_edges) == 5
    assert all(
        later > earlier
        for earlier, later in zip(
            spectrum.wavelength_bin_edges[:-1], spectrum.wavelength_bin_edges[1:], strict=False
        )
    )


def test_load_archive_tbl_falls_back_to_radius_ratio(tmp_path, provenance):
    path = tmp_path / "ratio_only.tbl"
    path.write_text(make_ipac_tbl(include_depth=False, include_ratio=True), encoding="utf-8")
    spectrum = load_archive_tbl(path, provenance=provenance)

    expected_depth = [
        float(PRISM_RATIO_ROWS[index % len(PRISM_RATIO_ROWS)][2]) ** 2
        for index in range(len(PRISM_SOURCE_ROWS))
    ]
    np.testing.assert_allclose(spectrum.transmission, expected_depth, rtol=1e-12)
    assert len(spectrum.wavelength) == 4


def test_load_archive_tbl_drops_rows_with_null_depth(tmp_path, provenance):
    path = tmp_path / "null_depth.tbl"
    path.write_text(make_ipac_tbl(null_depth_rows=(0,)), encoding="utf-8")
    spectrum = load_archive_tbl(path, provenance=provenance)
    assert len(spectrum.wavelength) == 3
    np.testing.assert_allclose(
        spectrum.wavelength, [float(row[0]) for row in PRISM_SOURCE_ROWS[1:]]
    )


def test_load_csv_converts_nm_and_ppm(tmp_path, provenance):
    path = tmp_path / "spectrum.csv"
    path.write_text(
        "wavelength,transmission,uncertainty\n"
        "1000,20899.7,25.71\n"
        "1100,21172.7,17.65\n"
        "1200,21419.1,12.76\n",
        encoding="utf-8",
    )
    spectrum = load_csv(
        path,
        provenance=provenance,
        observation_id="obs-csv",
        target_id="WASP-39 b",
        instrument="NIRSpec PRISM",
        wavelength_unit="nm",
        depth_unit="ppm",
    )
    np.testing.assert_allclose(spectrum.wavelength, [1.0, 1.1, 1.2], rtol=1e-12)
    np.testing.assert_allclose(spectrum.transmission, [0.0208997, 0.0211727, 0.0214191], rtol=1e-12)
    np.testing.assert_allclose(spectrum.uncertainty, [2.571e-5, 1.765e-5, 1.276e-5])
    assert len(spectrum.wavelength_bin_edges) == 4


def test_load_csv_missing_column_lists_available(tmp_path, provenance):
    path = tmp_path / "spectrum.csv"
    path.write_text("wavelength,depth\n1.0,0.01\n", encoding="utf-8")
    with pytest.raises(ValueError, match="available columns"):
        load_csv(
            path,
            provenance=provenance,
            observation_id="obs",
            target_id="t",
            instrument="i",
        )


def _write_fits(path, columns):
    table = Table(columns)
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table)]).writeto(path)


def test_load_x1d_fits_reads_units_and_values(tmp_path, provenance):
    path = tmp_path / "spec.fits"
    table = Table(
        [
            np.array([0.6, 1.0, 2.0], dtype=float) * 10000.0,  # Angstrom
            np.array([0.01, 0.011, 0.012]),
            np.array([1e-5, 2e-5, 3e-5]),
        ],
        names=("WAVELENGTH", "DEPTH", "DEPTH_ERR"),
    )
    table["WAVELENGTH"].unit = "Angstrom"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table)]).writeto(path)

    spectrum = load_x1d_fits(
        path,
        provenance=provenance,
        observation_id="obs-fits",
        target_id="WASP-39 b",
        instrument="NIRSpec PRISM",
    )
    np.testing.assert_allclose(spectrum.wavelength, [0.6, 1.0, 2.0], rtol=1e-12)
    np.testing.assert_allclose(spectrum.transmission, [0.01, 0.011, 0.012])
    assert len(spectrum.wavelength_bin_edges) == 4


def test_load_x1d_fits_rejects_flux_only_spectrum(tmp_path, provenance):
    path = tmp_path / "x1d_flux.fits"
    _write_fits(
        path,
        {
            "WAVELENGTH": np.array([0.6, 1.0]),
            "FLUX": np.array([1.0, 2.0]),
            "ERROR": np.array([0.1, 0.2]),
        },
    )
    with pytest.raises(ValueError, match="flux-only x1d spectrum"):
        load_x1d_fits(
            path,
            provenance=provenance,
            observation_id="obs",
            target_id="t",
            instrument="i",
        )


def test_load_x1d_fits_requires_wavelength_unit(tmp_path, provenance):
    path = tmp_path / "no_unit.fits"
    _write_fits(
        path,
        {
            "WAVELENGTH": np.array([0.6, 1.0]),
            "DEPTH": np.array([0.01, 0.011]),
            "DEPTH_ERR": np.array([1e-5, 2e-5]),
        },
    )
    with pytest.raises(ValueError, match="wavelength unit"):
        load_x1d_fits(
            path,
            provenance=provenance,
            observation_id="obs",
            target_id="t",
            instrument="i",
        )


def test_wavelength_and_depth_conversions():
    np.testing.assert_allclose(wavelength_to_um([1000.0, 5000.0], "nm"), [1.0, 5.0])
    np.testing.assert_allclose(wavelength_to_um([10000.0], "Angstrom"), [1.0], rtol=1e-12)
    np.testing.assert_allclose(depth_to_fraction([100.0], "percent"), [1.0])
    np.testing.assert_allclose(depth_to_fraction([1e6], "ppm"), [1.0])
    with pytest.raises(ValueError, match="unsupported wavelength unit"):
        wavelength_to_um([1.0], "blorp")
    with pytest.raises(ValueError, match="unsupported depth unit"):
        depth_to_fraction([1.0], "banana")


def test_bin_edges_from_centers_edge_cases():
    assert bin_edges_from_centers([1.0, 2.0]) == [0.5, 1.5, 2.5]
    assert bin_edges_from_centers([1.0], widths=[0.4]) == [0.8, 1.2]
    with pytest.raises(ValueError, match="single point without bin widths"):
        bin_edges_from_centers([1.0])
    with pytest.raises(ValueError, match="strictly ascending"):
        bin_edges_from_centers([2.0, 1.0])
