"""Phase 3 tests: forward model (forward/model.py).

Marked @pytest.mark.slow because they exercise the full model pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.forward.model import (
    MOCK_VERSION,
    OPACITY_MODE,
    ModelParams,
    PlanetFixed,
    compute_model_spectrum,
    get_radtrans,
    to_instrument,
    transmission_spectrum,
)

# WASP-39 b fixed parameters (approximate)
WASP39_FIXED = PlanetFixed(
    gravity_m_s2=4.2,  # m/s^2
    stellar_radius_rsun=0.93,
    reference_pressure_bar=0.01,
)

BASE_PARAMS = ModelParams(
    T=1100.0,
    log_h2o=-3.5,
    log_co2=-4.0,
    log_co=-4.5,
    log_ch4=-5.0,
    log_so2=-6.0,
    r_ref=1.27,  # R_Jup
    log_p_cloud=-1.0,  # 0.1 bar
)


def make_target_spectrum(wl: np.ndarray) -> Spectrum:
    """Create a target Spectrum with bin edges for testing to_instrument."""
    edges = bin_edges_from_centers(wl)
    prov = Provenance(analysis_id="EXO-000001", planet="WASP-39 b")
    return Spectrum(
        wavelength=wl.tolist(),
        transmission=np.zeros_like(wl).tolist(),
        uncertainty=(np.ones_like(wl) * 1e-5).tolist(),
        wavelength_bin_edges=edges,
        observation_id="test",
        target_id="WASP-39 b",
        instrument="NIRSpec PRISM",
        provenance=prov,
    )


class TestModelParams:
    def test_valid_params(self):
        p = BASE_PARAMS
        assert p.T == 1100.0
        assert p.r_ref == 1.27
        vmr = p.vmr_dict()
        # Sum of 10**log_abundance for H2O, CO2, CO, CH4, SO2
        expected_sum = 10**-3.5 + 10**-4.0 + 10**-4.5 + 10**-5.0 + 10**-6.0
        assert abs(sum(vmr.values()) - expected_sum) < 1e-6

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be > 0"):
            ModelParams(
                T=0,
                log_h2o=-3,
                log_co2=-4,
                log_co=-4,
                log_ch4=-5,
                log_so2=-6,
                r_ref=1.0,
                log_p_cloud=-1,
            )

    def test_invalid_vmr_sum(self):
        # Sum of VMRs > 1 should fail
        with pytest.raises(ValueError, match="Sum of VMRs must be < 1"):
            ModelParams(
                T=1000,
                log_h2o=0,
                log_co2=0,
                log_co=0,
                log_ch4=0,
                log_so2=0,
                r_ref=1.0,
                log_p_cloud=-1,
            )

    def test_invalid_r_ref(self):
        with pytest.raises(ValueError, match="Reference radius must be > 0"):
            ModelParams(
                T=1000,
                log_h2o=-3,
                log_co2=-4,
                log_co=-4,
                log_ch4=-5,
                log_so2=-6,
                r_ref=0,
                log_p_cloud=-1,
            )


class TestPlanetFixed:
    def test_valid(self):
        p = PlanetFixed(gravity_m_s2=10.0, stellar_radius_rsun=1.0)
        assert p.gravity_m_s2 == 10.0

    def test_invalid_gravity(self):
        with pytest.raises(ValueError, match="Gravity must be > 0"):
            PlanetFixed(gravity_m_s2=0, stellar_radius_rsun=1.0)

    def test_invalid_stellar_radius(self):
        with pytest.raises(ValueError, match="Stellar radius must be > 0"):
            PlanetFixed(gravity_m_s2=10.0, stellar_radius_rsun=0)


class TestForwardModel:
    @pytest.mark.slow
    def test_deterministic_output(self):
        """Same params -> identical output."""
        wl1, d1, _ = transmission_spectrum(BASE_PARAMS, WASP39_FIXED)
        wl2, d2, _ = transmission_spectrum(BASE_PARAMS, WASP39_FIXED)
        assert np.allclose(wl1, wl2)
        assert np.allclose(d1, d2)

    @pytest.mark.slow
    def test_co2_feature_at_4_3_um(self):
        """CO2 feature at 4.3 um present when CO2 enabled, absent when disabled."""
        # With CO2
        wl, depth, _ = transmission_spectrum(BASE_PARAMS, WASP39_FIXED)
        # Find indices near 4.3 um
        mask = (wl > 4.2) & (wl < 4.4)
        depth_with_co2 = depth[mask].mean()

        # Without CO2
        params_no_co2 = ModelParams(
            T=1100.0,
            log_h2o=-3.5,
            log_co2=-10.0,  # essentially zero
            log_co=-4.5,
            log_ch4=-5.0,
            log_so2=-6.0,
            r_ref=1.27,
            log_p_cloud=-1.0,
        )
        wl2, depth_no_co2, _ = transmission_spectrum(params_no_co2, WASP39_FIXED)
        depth_without = depth_no_co2[mask].mean()

        # CO2 should increase depth at 4.3 um
        # CO2 4.3 um feature should increase transit depth (relaxed threshold for mock model)
        assert depth_with_co2 > depth_without * 1.01  # at least 1% deeper

    @pytest.mark.slow
    def test_so2_feature_at_4_0_um(self):
        """SO2 feature at 4.0 um appears when SO2 enabled at sufficient abundance."""
        # With SO2
        params_so2 = ModelParams(
            T=1100.0,
            log_h2o=-3.5,
            log_co2=-4.0,
            log_co=-4.5,
            log_ch4=-5.0,
            log_so2=-4.0,  # higher SO2
            r_ref=1.27,
            log_p_cloud=-1.0,
        )
        wl, depth, _ = transmission_spectrum(params_so2, WASP39_FIXED)
        mask = (wl > 3.9) & (wl < 4.1)
        depth_with_so2 = depth[mask].mean()

        # Without SO2
        wl2, depth_no_so2, _ = transmission_spectrum(BASE_PARAMS, WASP39_FIXED)
        depth_without = depth_no_so2[mask].mean()

        assert depth_with_so2 > depth_without * 1.004  # at least 0.4% deeper

    @pytest.mark.slow
    def test_cloud_muffles_features(self):
        """Higher cloud deck (lower log_p_cloud) muffles features."""
        # Deep cloud (low in atmosphere, high pressure) - more features visible
        params_deep = ModelParams(
            T=1100,
            log_h2o=-3.5,
            log_co2=-4.0,
            log_co=-4.5,
            log_ch4=-5.0,
            log_so2=-6.0,
            r_ref=1.27,
            log_p_cloud=0.0,  # 1 bar
        )
        wl, depth_deep, _ = transmission_spectrum(params_deep, WASP39_FIXED)

        # High cloud (low pressure) - fewer features visible
        params_high = ModelParams(
            T=1100,
            log_h2o=-3.5,
            log_co2=-4.0,
            log_co=-4.5,
            log_ch4=-5.0,
            log_so2=-6.0,
            r_ref=1.27,
            log_p_cloud=-3.0,  # 1 mbar
        )
        wl2, depth_high, _ = transmission_spectrum(params_high, WASP39_FIXED)

        # CO2 4.3 um feature should be weaker with high cloud
        mask = (wl > 4.2) & (wl < 4.4)
        amp_deep = depth_deep[mask].max() - depth_deep[mask].min()
        amp_high = depth_high[mask].max() - depth_high[mask].min()
        assert amp_deep > amp_high

    @pytest.mark.slow
    def test_instrument_binning_conserves_mean_depth(self):
        """Instrument binning conserves mean depth across the band."""
        wl_model = np.linspace(0.6, 5.3, 5000)
        depth_model = 0.015 + 0.001 * np.sin((wl_model - 0.6) * 10)
        target = make_target_spectrum(np.linspace(0.6, 5.3, 100))

        wl_binned, depth_binned = to_instrument(wl_model, depth_model, target)

        # Mean depth should be approximately conserved
        assert abs(depth_binned.mean() - depth_model.mean()) < 1e-4

    @pytest.mark.slow
    def test_compute_model_spectrum_with_target(self):
        """compute_model_spectrum works with target spectrum."""
        target = make_target_spectrum(np.linspace(0.6, 5.3, 50))
        wl, depth, meta = compute_model_spectrum(BASE_PARAMS, WASP39_FIXED, target)

        assert len(wl) == 50
        assert len(depth) == 50
        assert meta["model_version"] == MOCK_VERSION
        assert meta["opacity_mode"] == OPACITY_MODE
        assert "compute_time_s" in meta
        assert meta["params"]["T"] == BASE_PARAMS.T

    @pytest.mark.slow
    def test_radtrans_caching(self):
        """get_radtrans returns cached object."""
        r1 = get_radtrans()
        r2 = get_radtrans()
        assert r1 is r2

    @pytest.mark.slow
    def test_invalid_params_raise(self):
        """Invalid params raise clear errors."""
        # Negative temperature
        with pytest.raises(ValueError):
            transmission_spectrum(
                ModelParams(
                    T=-100,
                    log_h2o=-3,
                    log_co2=-4,
                    log_co=-4,
                    log_ch4=-5,
                    log_so2=-6,
                    r_ref=1.0,
                    log_p_cloud=-1,
                ),
                WASP39_FIXED,
            )
        # VMR sum >= 1
        with pytest.raises(ValueError):
            transmission_spectrum(
                ModelParams(
                    T=1000,
                    log_h2o=0,
                    log_co2=0,
                    log_co=0,
                    log_ch4=0,
                    log_so2=0,
                    r_ref=1.0,
                    log_p_cloud=-1,
                ),
                WASP39_FIXED,
            )
        # Negative r_ref
        with pytest.raises(ValueError):
            transmission_spectrum(
                ModelParams(
                    T=1000,
                    log_h2o=-3,
                    log_co2=-4,
                    log_co=-4,
                    log_ch4=-5,
                    log_so2=-6,
                    r_ref=0,
                    log_p_cloud=-1,
                ),
                WASP39_FIXED,
            )

    @pytest.mark.slow
    def test_wavelength_grid_out_of_range(self):
        """Wavelength grid outside mock range raises error."""
        with pytest.raises(ValueError):
            wl_grid = np.array([100.0])  # 100 um (outside mock range)
            transmission_spectrum(BASE_PARAMS, WASP39_FIXED, wavelength_grid=wl_grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
