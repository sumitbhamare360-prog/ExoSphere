"""L1 data validation: source spectrum -> Spectrum -> back must round-trip.

L1 (AGENTS.md section 7): ingest a source spectrum and reproduce it, matching
the source values within float tolerance. Source values are the NASA Exoplanet
Archive WASP-39 b NIRSpec PRISM transmission-spectrum rows recorded in
tests/conftest.py (percent depths, micron wavelengths).
"""

import numpy as np

from conftest import PRISM_SOURCE_ROWS, make_ipac_tbl
from exosphere.core.provenance import Provenance
from exosphere.data.loaders import load_archive_tbl


def test_l1_source_to_spectrum_and_back(tmp_path):
    provenance = Provenance(analysis_id="EXO-000001", planet="WASP-39 b")
    path = tmp_path / "l1_source.tbl"
    path.write_text(make_ipac_tbl(), encoding="utf-8")

    spectrum = load_archive_tbl(path, provenance=provenance, observation_id="l1")

    # Spectrum -> source convention: fraction -> percent, um stays um.
    round_trip_wavelength = np.asarray(spectrum.wavelength, dtype=float)
    round_trip_depth_percent = np.asarray(spectrum.transmission, dtype=float) * 100.0
    round_trip_uncertainty_percent = (
        np.asarray(spectrum.uncertainty, dtype=float) * 100.0
    )

    source_wavelength = np.array([float(row[0]) for row in PRISM_SOURCE_ROWS])
    source_depth_percent = np.array([float(row[2]) for row in PRISM_SOURCE_ROWS])
    source_uncertainty_percent = np.array(
        [(abs(float(row[3])) + abs(float(row[4]))) / 2.0 for row in PRISM_SOURCE_ROWS]
    )

    np.testing.assert_allclose(round_trip_wavelength, source_wavelength, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        round_trip_depth_percent, source_depth_percent, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        round_trip_uncertainty_percent, source_uncertainty_percent, rtol=1e-12, atol=1e-12
    )


def test_l1_survives_npz_round_trip(tmp_path):
    provenance = Provenance(analysis_id="EXO-000001", planet="WASP-39 b")
    source_path = tmp_path / "l1_source.tbl"
    source_path.write_text(make_ipac_tbl(), encoding="utf-8")

    spectrum = load_archive_tbl(source_path, provenance=provenance, observation_id="l1")
    reloaded = type(spectrum).load(spectrum.save(tmp_path / "l1_spectrum"))

    assert reloaded == spectrum
    np.testing.assert_allclose(
        np.asarray(reloaded.transmission) * 100.0,
        [float(row[2]) for row in PRISM_SOURCE_ROWS],
        rtol=1e-12,
        atol=1e-12,
    )
    assert reloaded.provenance == spectrum.provenance
    assert reloaded.observation_id == "l1"
