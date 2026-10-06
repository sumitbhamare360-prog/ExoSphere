"""Tests for the Spectrum data contract (AGENTS.md section 4)."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import UNITS, Spectrum


def make_spectrum_kwargs(**overrides):
    kwargs = {
        "wavelength": [0.6, 1.0, 2.0, 5.0],
        "transmission": [0.01, 0.011, 0.012, 0.013],
        "uncertainty": [1e-5, 2e-5, 3e-5, 4e-5],
        "wavelength_bin_edges": [0.5, 0.8, 1.5, 3.0, 5.3],
        "quality_flags": ["OK", "OK", "OK", "OK"],
        "observation_id": "obs-001",
        "target_id": "WASP-39 b",
        "instrument": "NIRSpec PRISM",
        "provenance": Provenance(analysis_id="EXO-000001", planet="WASP-39 b"),
    }
    kwargs.update(overrides)
    return kwargs


def test_valid_spectrum_accepted():
    spectrum = Spectrum(**make_spectrum_kwargs())
    assert spectrum.wavelength == [0.6, 1.0, 2.0, 5.0]
    assert spectrum.observation_id == "obs-001"
    assert spectrum.provenance.analysis_id == "EXO-000001"
    assert UNITS["transmission"] == "fractional transit depth ((Rp/Rs)^2)"
    assert Spectrum.units() == UNITS
    json.dumps(spectrum.to_dict())


def test_array_like_inputs_are_accepted():
    kwargs = make_spectrum_kwargs()
    kwargs["wavelength"] = np.asarray(kwargs["wavelength"], dtype=np.float64)
    kwargs["transmission"] = tuple(kwargs["transmission"])
    spectrum = Spectrum(**kwargs)
    assert spectrum.wavelength == [0.6, 1.0, 2.0, 5.0]


def test_rejects_unequal_array_lengths():
    kwargs = make_spectrum_kwargs(transmission=[0.01, 0.011, 0.012])
    with pytest.raises(ValidationError, match="equal lengths"):
        Spectrum(**kwargs)


def test_rejects_mismatched_quality_flags_length():
    kwargs = make_spectrum_kwargs(quality_flags=["OK"])
    with pytest.raises(ValidationError, match="quality_flags must be empty"):
        Spectrum(**kwargs)


def test_rejects_nan_wavelength():
    kwargs = make_spectrum_kwargs(wavelength=[0.6, np.nan, 2.0, 5.0])
    with pytest.raises(ValidationError, match="must not contain NaN"):
        Spectrum(**kwargs)


@pytest.mark.parametrize("bad_uncertainty", [0.0, -1e-5])
def test_rejects_nonpositive_uncertainty(bad_uncertainty):
    kwargs = make_spectrum_kwargs(
        uncertainty=[1e-5, 2e-5, bad_uncertainty, 4e-5],
    )
    with pytest.raises(ValidationError, match="uncertainty must be > 0"):
        Spectrum(**kwargs)


def test_rejects_unsorted_wavelength():
    kwargs = make_spectrum_kwargs(wavelength=[1.0, 0.6, 2.0, 5.0])
    with pytest.raises(ValidationError, match="strictly ascending"):
        Spectrum(**kwargs)


def test_rejects_wrong_bin_edges_length():
    kwargs = make_spectrum_kwargs(wavelength_bin_edges=[0.5, 1.5, 3.0, 5.3])
    with pytest.raises(ValidationError, match="wavelength_bin_edges must have length"):
        Spectrum(**kwargs)


def test_dict_round_trip_is_exact():
    spectrum = Spectrum(**make_spectrum_kwargs())
    restored = Spectrum.from_dict(spectrum.to_dict())
    assert restored == spectrum
    assert restored.to_dict() == spectrum.to_dict()


def test_npz_save_load_round_trip_is_exact(tmp_path):
    spectrum = Spectrum(**make_spectrum_kwargs())
    path = spectrum.save(tmp_path / "spectrum")
    assert path.suffix == ".npz"
    restored = Spectrum.load(path)
    assert restored == spectrum
    assert restored.wavelength == spectrum.wavelength
    assert restored.transmission == spectrum.transmission
    assert restored.uncertainty == spectrum.uncertainty
    assert restored.wavelength_bin_edges == spectrum.wavelength_bin_edges
    assert restored.quality_flags == spectrum.quality_flags
    assert restored.provenance == spectrum.provenance


def test_json_save_load_round_trip_is_exact():
    spectrum = Spectrum(**make_spectrum_kwargs())
    restored = Spectrum.from_json(spectrum.to_json())
    assert restored == spectrum
