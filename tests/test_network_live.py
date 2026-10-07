"""Optional live-network tests (deselected by default).

Run with: pytest -m network
"""

import pytest

from exosphere.data.exoarchive import get_planet_params, list_literature_spectra


@pytest.mark.network
def test_live_get_planet_params():
    params = get_planet_params("WASP-39 b")
    assert params.planet_name == "WASP-39 b"
    assert params.planet_radius_earth is not None and params.planet_radius_earth > 0
    assert params.surface_gravity_m_s2 is not None and params.surface_gravity_m_s2 > 0


@pytest.mark.network
def test_live_wasp39b_prism_spectrum_is_listed():
    spectra = list_literature_spectra("WASP-39 b", spec_type="Transmission")
    assert spectra, "expected literature transmission spectra for WASP-39 b"
    prism = [s for s in spectra if "prism" in f"{s.instrument or ''} {s.note or ''}".lower()]
    assert prism, "expected at least one NIRSpec PRISM transmission spectrum"
