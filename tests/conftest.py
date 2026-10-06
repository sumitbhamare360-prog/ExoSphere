"""Shared fixtures for Phase 1 data-acquisition tests."""

from __future__ import annotations

import pytest

from exosphere.core.provenance import Provenance

# Source values (microns / percent) taken from the NASA Exoplanet Archive
# download of the WASP-39 b NIRSpec PRISM transmission spectrum
# (spec_path 40/24/96/78/WASP_39_b_3.11466_5502_6.tbl, Carter et al. 2024).
PRISM_SOURCE_ROWS = [
    # central_wavelength_um, bandwidth_um, depth_percent, err1_percent, err2_percent, limit
    ("0.52132", "0.00850", "2.08997", "0.02571", "-0.02551", "0"),
    ("0.53004", "0.00893", "2.11727", "0.01765", "-0.01748", "0"),
    ("0.53919", "0.00938", "2.14191", "0.01276", "-0.01266", "0"),
    ("0.54883", "0.00989", "2.15292", "0.01013", "-0.01015", "0"),
]

PRISM_RATIO_ROWS = [
    # central_wavelength_um, bandwidth_um, Rp/R*, err1, err2
    ("0.52132", "0.00850", "0.14457", "0.00089", "-0.00088"),
    ("0.53004", "0.00893", "0.14551", "0.00061", "-0.00060"),
]


def make_ipac_tbl(
    rows=PRISM_SOURCE_ROWS,
    *,
    include_depth: bool = True,
    include_ratio: bool = False,
    null_depth_rows: tuple[int, ...] = (),
) -> str:
    """Build a small IPAC .tbl in the exact layout of the archive downloads."""
    lines = [
        "\\PL_NAME = WASP-39 b",
        "\\SPEC_TYPE = Transmission",
        "\\INSTRUMENT = Near Infrared Spectrograph (NIRSpec)",
        "\\FACILITY = NASA 6.5m James Webb Space Telescope (JWST) Satellite Mission",
        "\\NOTE = PRISM, Native resolution",
        "\\REFERENCE = Carter et al. 2024",
        "\\",
    ]
    names = ["CENTRALWAVELNG", "BANDWIDTH"]
    types = ["double", "double"]
    units = ["microns", "microns"]
    if include_depth:
        names += ["PL_TRANDEP", "PL_TRANDEPERR1", "PL_TRANDEPERR2", "PL_TRANDEPLIM"]
        types += ["double", "double", "double", "long"]
        units += ["%", "%", "%", ""]
    if include_ratio:
        names += ["PL_RATROR", "PL_RATRORERR1", "PL_RATRORERR2"]
        types += ["double", "double", "double"]
        units += ["", "", ""]
    widths = [len(name) for name in names]
    lines.append("|" + "|".join(names) + "|")
    lines.append("|" + "|".join(f"{t:>{w}s}" for t, w in zip(types, widths, strict=False)) + "|")
    lines.append("|" + "|".join(f"{u:>{w}s}" for u, w in zip(units, widths, strict=False)) + "|")
    lines.append("|" + "|".join("null".rjust(w) for w in widths) + "|")

    for index, row in enumerate(rows):
        cells = [row[0], row[1]]
        if include_depth:
            if index in null_depth_rows:
                cells += ["null", "null", "null", "0"]
            else:
                cells += [row[2], row[3], row[4], row[5]]
        if include_ratio:
            ratio_row = PRISM_RATIO_ROWS[index % len(PRISM_RATIO_ROWS)]
            cells += [ratio_row[2], ratio_row[3], ratio_row[4]]
        # data rows align with the header: single spaces where the header has '|'
        cells = [cell.rjust(w) for cell, w in zip(cells, widths, strict=False)]
        lines.append(" " + " ".join(cells))
    return "\n".join(lines) + "\n"


@pytest.fixture
def provenance() -> Provenance:
    return Provenance(analysis_id="EXO-000001", planet="WASP-39 b")


@pytest.fixture
def ipac_tbl_path(tmp_path):
    path = tmp_path / "WASP_39_b_prism.tbl"
    path.write_text(make_ipac_tbl(), encoding="utf-8")
    return path
