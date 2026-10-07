"""JWST observation metadata and calibrated-product downloads via MAST.

Scope (AGENTS.md section 1): metadata search plus calibrated 1D products only;
raw detector data is never downloaded and no raw calibration is performed.
Column names below come from ``Observations.get_metadata("observations")`` /
``get_metadata("products")`` in the installed astroquery version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Row, Table
from astroquery.mast import Observations
from pydantic import BaseModel, Field

from exosphere.core.config import load_config

CALIBRATED_1D_SUBGROUPS = ("X1D", "X1DINTS")


class MASTError(RuntimeError):
    """Raised when a MAST query or product download fails."""


class ProductInfo(BaseModel):
    """Metadata of one MAST data product (never the file itself)."""

    product_filename: str
    obs_id: str | None = None
    product_type: str | None = None
    product_subgroup: str | None = None
    calib_level: int | None = None
    size_bytes: int | None = None
    data_uri: str | None = None
    description: str | None = None

    def is_calibrated_1d(self) -> bool:
        """True for extracted 1D science products (X1D / X1DINTS)."""
        type_ok = (self.product_type or "").upper() == "SCIENCE"
        subgroup_ok = (self.product_subgroup or "").upper() in CALIBRATED_1D_SUBGROUPS
        return type_ok and subgroup_ok


class JWSTObservation(BaseModel):
    """JWST observation metadata (instrument, program, ids) + optional products."""

    obs_id: str
    obsid: int
    target_name: str | None = None
    instrument_name: str | None = None
    proposal_id: str | None = None
    dataproduct_type: str | None = None
    calib_level: int | None = None
    filters: str | None = None
    t_min_mjd: float | None = None
    products: list[ProductInfo] = Field(default_factory=list)


def _value(row, column: str):
    if column not in row.colnames:
        return None
    value = row[column]
    if value is None or np.ma.is_masked(value):
        return None
    return value


def _text(row, column: str) -> str | None:
    value = _value(row, column)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "--":
        return None
    return text


def _int(row, column: str) -> int | None:
    value = _value(row, column)
    return None if value is None else int(value)


def _float(row, column: str) -> float | None:
    value = _value(row, column)
    return None if value is None else float(value)


def _product_info(row) -> ProductInfo:
    size = _value(row, "size")
    return ProductInfo(
        product_filename=str(_text(row, "productFilename") or ""),
        obs_id=_text(row, "obs_id"),
        product_type=_text(row, "productType"),
        product_subgroup=_text(row, "productSubGroupDescription"),
        calib_level=_int(row, "calib_level"),
        size_bytes=int(size) if size is not None else None,
        data_uri=_text(row, "dataURI"),
        description=_text(row, "description"),
    )


def search_jwst_observations(
    target: str, *, include_products: bool = True
) -> list[JWSTObservation]:
    """Search JWST observations for ``target`` (metadata only).

    ``target`` is passed straight to MAST as a target_name criterion; wildcards
    (``*``/``%``) are allowed, e.g. ``"WASP-39*"``.
    """
    obs_table = Observations.query_criteria(obs_collection="JWST", target_name=target)
    observations: list[JWSTObservation] = []
    for row in obs_table:
        obs_id = _text(row, "obs_id")
        obsid = _int(row, "obsid")
        if obs_id is None or obsid is None:
            continue
        observations.append(
            JWSTObservation(
                obs_id=obs_id,
                obsid=obsid,
                target_name=_text(row, "target_name"),
                instrument_name=_text(row, "instrument_name"),
                proposal_id=_text(row, "proposal_id"),
                dataproduct_type=_text(row, "dataproduct_type"),
                calib_level=_int(row, "calib_level"),
                filters=_text(row, "filters"),
                t_min_mjd=_float(row, "t_min"),
            )
        )

    if include_products and observations:
        product_table = Observations.get_product_list(obs_table)
        by_group: dict[int, list] = {}
        for product_row in product_table:
            group = _int(product_row, "obsID")
            if group is not None:
                by_group.setdefault(group, []).append(product_row)
        for observation in observations:
            observation.products = [
                _product_info(product_row) for product_row in by_group.get(observation.obsid, [])
            ]
    return observations


def _default_download_dir() -> Path:
    return load_config().data_cache_dir / "mast"


def _resolve_product_table(obs_id: str, product) -> Table:
    if isinstance(product, Row):
        return product.table[product.index : product.index + 1]
    if isinstance(product, Table):
        return product
    if isinstance(product, ProductInfo):
        product_name = product.product_filename
    elif isinstance(product, dict):
        product_name = str(product.get("product_filename") or product.get("productFilename") or "")
    elif isinstance(product, str):
        product_name = product
    else:
        raise MASTError(f"unsupported product reference: {type(product).__name__}")
    if not product_name:
        raise MASTError(f"no product filename given for obs_id {obs_id!r}")

    obs_table = Observations.query_criteria(obs_collection="JWST", obs_id=obs_id)
    if not len(obs_table):
        raise MASTError(f"no JWST observation found with obs_id {obs_id!r}")
    product_table = Observations.get_product_list(obs_table)
    for index, row in enumerate(product_table):
        if str(row["productFilename"]) == product_name:
            return product_table[index : index + 1]
    raise MASTError(f"product {product_name!r} not found among products of obs_id {obs_id!r}")


def download_product(obs_id: str, product, *, cache_dir: Path | None = None) -> Path:
    """Download one chosen calibrated 1D product into ``data_cache/mast/``.

    ``product`` may be a product filename, a ProductInfo, or an astropy
    product row. Raw/uncalibrated products are rejected.
    """
    product_table = _resolve_product_table(obs_id, product)
    info = _product_info(product_table[0])
    if not info.is_calibrated_1d():
        raise ValueError(
            f"only calibrated 1D science products (subgroups "
            f"{list(CALIBRATED_1D_SUBGROUPS)}) are downloaded in V1; "
            f"got productType={info.product_type!r}, "
            f"productSubGroupDescription={info.product_subgroup!r}"
        )

    directory = Path(cache_dir) if cache_dir is not None else _default_download_dir()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = Observations.download_products(
        product_table,
        download_dir=str(directory),
        flat=True,
        cache=True,
        verbose=False,
    )
    if manifest is None or not len(manifest):
        raise MASTError(f"no download manifest returned for {info.product_filename!r}")
    local_path = Path(str(manifest[0]["Local Path"]))
    if not local_path.exists():
        status = manifest[0]["Status"]
        message = manifest[0]["Message"]
        raise MASTError(
            f"download of {info.product_filename!r} failed: status={status}, message={message}"
        )
    return local_path


__all__ = [
    "JWSTObservation",
    "MASTError",
    "ProductInfo",
    "download_product",
    "search_jwst_observations",
]
