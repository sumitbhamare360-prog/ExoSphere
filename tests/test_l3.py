"""Phase 9b tests: L3 comparison helpers (pure functions, no DB)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_l3 import (  # noqa: E402
    LITERATURE,
    metallicity_and_c_o,
    molecule_verdicts,
    parameter_verdicts,
    render_markdown,
    stability_verdict,
)


def test_literature_references_present():
    for key in ["rustamkulov2023", "carter2024", "alderson2023", "feinstein2023",
                "powell2024", "moran2024"]:
        assert "citation" in LITERATURE[key]
    assert LITERATURE["rustamkulov2023"]["H2O_sigma"] == 33.0
    assert LITERATURE["rustamkulov2023"]["CH4_detected"] is False


def test_metallicity_and_c_o():
    out = metallicity_and_c_o({
        "log_h2o": -3.0, "log_co2": -3.5, "log_co": -5.0,
        "log_ch4": -6.0, "log_so2": -6.0,
    })
    # heavy sum ~1.3e-3 -> ~1.3x solar; C/O ~ (1e-5+3e-4+1e-6)/(1e-3+...) ~ 0.2
    assert out["metallicity_solar"] is not None
    assert 0.5 < out["metallicity_solar"] < 5.0
    assert out["C_O"] is not None
    assert 0.05 < out["C_O"] < 1.0
    missing = metallicity_and_c_o({})
    assert missing == {"metallicity_solar": None, "C_O": None}


def test_molecule_verdicts_expected_pattern():
    detections = {
        "H2O": {"lnB": 172.0},
        "CO2": {"lnB": 165.0},
        "CO": {"lnB": -8.0},
        "CH4": {"lnB": -8.0},
        "SO2": {"lnB": None},
    }
    quality = {m: "GOOD" for m in detections}
    verdicts = molecule_verdicts(detections, quality)
    assert verdicts["H2O"]["grade"] == "PASS"
    assert verdicts["CO2"]["grade"] == "PASS"
    assert verdicts["CH4"]["grade"] == "PASS"  # not constrained, as published
    assert verdicts["CO"]["grade"] == "PASS"  # not constrained is a PASS here
    assert verdicts["SO2"]["grade"] == "PARTIAL"  # not assessed


def test_molecule_verdicts_poor_overrides():
    detections = {"H2O": {"lnB": 200.0}}
    verdicts = molecule_verdicts(detections, {"H2O": "POOR"})
    assert verdicts["H2O"]["grade"] == "SKIP"


def test_molecule_verdicts_ch4_claim_fails():
    detections = {"CH4": {"lnB": 12.0}}
    verdicts = molecule_verdicts(detections, {"CH4": "GOOD"})
    assert verdicts["CH4"]["grade"] == "FAIL"


def test_parameter_verdicts():
    median = {"T": 1100.0, "log_ch4": -6.0, "log_h2o": -3.0, "log_co2": -3.5,
              "log_co": -5.0, "log_so2": -6.0}
    ci95 = {"T": [900.0, 1300.0], "log_ch4": [-8.0, -4.5]}
    out = parameter_verdicts(median, ci95)
    assert out["T"].startswith("PASS")
    assert out["CH4_UL"].startswith("PASS")
    assert out["metallicity"].startswith("PASS")
    bad = parameter_verdicts({"T": 300.0}, {"T": [290.0, 310.0], "log_ch4": [-3.0, -1.0]})
    assert bad["T"].startswith("FAIL")


def test_stability_verdict():
    medians = [
        {"T": 1100.0, "log_h2o": -3.0},
        {"T": 1120.0, "log_h2o": -3.1},
    ]
    ci68 = [
        {"T": [1000.0, 1200.0], "log_h2o": [-3.5, -2.5]},
        {"T": [1020.0, 1220.0], "log_h2o": [-3.6, -2.6]},
    ]
    assert stability_verdict(medians, ci68).startswith("PASS")
    far = [{"T": 1100.0}, {"T": 2000.0}]
    far_ci = [{"T": [1050.0, 1150.0]}, {"T": [1950.0, 2050.0]}]
    assert stability_verdict(far, far_ci).startswith("FAIL")
    assert stability_verdict(medians[:1], ci68[:1]).startswith("PARTIAL")


def test_render_markdown_has_all_sections():
    ctx = {
        "generated": "2026-10-10",
        "analyses": ["EXO-000002"],
        "roundtrip": "PASS: ...",
        "molecules": {
            "H2O": {"published": "33σ", "ours": "ln B = 172", "grade": "PASS: x"},
        },
        "parameters": {"T": "PASS: ..."},
        "fit": "reduced chi2 = 2.0",
        "ml": "no contradiction",
        "stability": "PASS: ...",
        "overall": "PASS (...)",
    }
    md = render_markdown(ctx)
    for heading in ["## (a)", "## (b)", "## (c)", "## (d)", "## (e)",
                    "## Seed stability", "## References", "## Overall"]:
        assert heading in md
    assert "Rustamkulov" in md
