"""Unit tests for data/mast.py with the astroquery Observations client mocked."""

import pytest
from astropy.table import Table

from exosphere.data import mast
from exosphere.data.mast import (
    JWSTObservation,
    MASTError,
    ProductInfo,
    download_product,
    search_jwst_observations,
)

OBS_COLUMNS = {
    "obs_id": ["jw01366-o004_t001_nirspec_clear-prism-s1600a1-sub512"],
    "obsid": [233647483],
    "target_name": ["WASP-39"],
    "instrument_name": ["NIRSPEC/SLIT"],
    "proposal_id": ["1366"],
    "dataproduct_type": ["timeseries"],
    "calib_level": [3],
    "filters": ["CLEAR"],
    "t_min": [60000.5],
}

PRODUCT_COLUMNS = {
    "obs_id": ["jw01366-o004_t001_nirspec_clear-prism-s1600a1-sub512"] * 3,
    "obsID": [233647483] * 3,
    "productFilename": [
        "jw01366-o004_t001_nirspec_clear-prism-s1600a1-sub512_x1dints.fits",
        "jw01366-o004_t001_nirspec_clear-prism-s1600a1-sub512_uncal.fits",
        "jw01366-o004_t001_nirspec_clear-prism-s1600a1-sub512_i2d.fits",
    ],
    "productType": ["SCIENCE", "SCIENCE", "SCIENCE"],
    "productSubGroupDescription": ["X1DINTS", "UNCAL", "I2D"],
    "calib_level": [3, 1, 3],
    "size": [1024, 4096, 2048],
    "dataURI": [
        "mast:JWST/product/jw01366_x1dints.fits",
        "mast:JWST/product/jw01366_uncal.fits",
        "mast:JWST/product/jw01366_i2d.fits",
    ],
    "description": ["extracted 1D", "uncalibrated", "stage 3 2D"],
}

X1D_NAME = PRODUCT_COLUMNS["productFilename"][0]
UNCAL_NAME = PRODUCT_COLUMNS["productFilename"][1]


class FakeObservations:
    def __init__(self, *, obs_table=None, manifest_path=None):
        self.obs_table = obs_table if obs_table is not None else Table(OBS_COLUMNS)
        self.product_table = Table(PRODUCT_COLUMNS)
        self.manifest_path = manifest_path
        self.query_calls = []
        self.download_calls = []

    def query_criteria(self, **criteria):
        self.query_calls.append(criteria)
        if "obs_id" in criteria:
            if criteria["obs_id"] == OBS_COLUMNS["obs_id"][0]:
                return self.obs_table
            return Table({name: [] for name in OBS_COLUMNS})
        return self.obs_table

    def get_product_list(self, observations):
        return self.product_table

    def download_products(self, products, **kwargs):
        self.download_calls.append((products, kwargs))
        from astropy.table import Table as AstropyTable

        local = str(self.manifest_path) if self.manifest_path else "missing.fits"
        return AstropyTable(
            {
                "Local Path": [local],
                "Status": ["COMPLETE"],
                "Message": [""],
                "URL": ["https://mast.stsci.edu/api/v0.1/Download/file"],
            }
        )


def test_search_returns_metadata_with_products(monkeypatch):
    fake = FakeObservations()
    monkeypatch.setattr(mast, "Observations", fake)

    observations = search_jwst_observations("WASP-39*")
    assert len(observations) == 1
    observation = observations[0]
    assert isinstance(observation, JWSTObservation)
    assert observation.obs_id == OBS_COLUMNS["obs_id"][0]
    assert observation.obsid == 233647483
    assert observation.instrument_name == "NIRSPEC/SLIT"
    assert observation.proposal_id == "1366"
    assert observation.dataproduct_type == "timeseries"
    assert observation.t_min_mjd == 60000.5
    assert len(observation.products) == 3
    x1d = next(p for p in observation.products if p.product_subgroup == "X1DINTS")
    uncal = next(p for p in observation.products if p.product_subgroup == "UNCAL")
    assert x1d.is_calibrated_1d()
    assert not uncal.is_calibrated_1d()
    assert fake.query_calls[0] == {
        "obs_collection": "JWST",
        "target_name": "WASP-39*",
    }


def test_search_without_products(monkeypatch):
    fake = FakeObservations()
    monkeypatch.setattr(mast, "Observations", fake)

    observations = search_jwst_observations("WASP-39", include_products=False)
    assert observations[0].products == []


def test_download_product_calibrated_1d(tmp_path, monkeypatch):
    local = tmp_path / X1D_NAME
    local.write_bytes(b"fits-bytes")
    fake = FakeObservations(manifest_path=local)
    monkeypatch.setattr(mast, "Observations", fake)

    result = download_product(
        OBS_COLUMNS["obs_id"][0], X1D_NAME, cache_dir=tmp_path
    )
    assert result == local
    products, kwargs = fake.download_calls[0]
    assert len(products) == 1
    assert str(products[0]["productFilename"]) == X1D_NAME
    assert kwargs["download_dir"] == str(tmp_path)
    assert kwargs["flat"] is True
    assert kwargs["cache"] is True


def test_download_product_rejects_raw_product(monkeypatch):
    fake = FakeObservations()
    monkeypatch.setattr(mast, "Observations", fake)

    with pytest.raises(ValueError, match="calibrated 1D"):
        download_product(OBS_COLUMNS["obs_id"][0], UNCAL_NAME)
    assert fake.download_calls == []


def test_download_product_unknown_obs_id(monkeypatch):
    fake = FakeObservations()
    monkeypatch.setattr(mast, "Observations", fake)

    with pytest.raises(MASTError, match="no JWST observation"):
        download_product("jw99999-o001_t001_nirspec_prism", X1D_NAME)


def test_download_product_unknown_filename(monkeypatch):
    fake = FakeObservations()
    monkeypatch.setattr(mast, "Observations", fake)

    with pytest.raises(MASTError, match="not found among products"):
        download_product(OBS_COLUMNS["obs_id"][0], "does_not_exist.fits")


def test_download_product_failed_manifest(monkeypatch, tmp_path):
    fake = FakeObservations(manifest_path=tmp_path / "never_written.fits")
    monkeypatch.setattr(mast, "Observations", fake)

    with pytest.raises(MASTError, match="failed"):
        download_product(OBS_COLUMNS["obs_id"][0], X1D_NAME)


def test_product_info_model_and_validation():
    info = ProductInfo(product_filename="a_x1d.fits", product_type="SCIENCE",
                       product_subgroup="X1D")
    assert info.is_calibrated_1d()
    assert not ProductInfo(
        product_filename="a_uncal.fits", product_type="SCIENCE",
        product_subgroup="UNCAL"
    ).is_calibrated_1d()
