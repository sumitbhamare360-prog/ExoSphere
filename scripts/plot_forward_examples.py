"""Generate example forward model spectra (Phase 3).

Produces a 2x2 figure with four examples:
1. H2O-only atmosphere
2. H2O + CO2
3. H2O + CO2 + SO2
4. Cloudy version of (3)

Saves to outputs/forward_examples.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exosphere.forward.model import (  # noqa: E402
    MOCK_VERSION,
    OPACITY_MODE,
    ModelParams,
    PlanetFixed,
    compute_model_spectrum,
)

OUTPUT_PNG = REPO_ROOT / "outputs" / "forward_examples.png"


# WASP-39 b fixed parameters
FIXED = PlanetFixed(
    gravity_m_s2=4.2,
    stellar_radius_rsun=0.93,
    reference_pressure_bar=0.01,
)


def make_spectrum(label: str, **param_kwargs) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a model spectrum with given parameters."""
    base = dict(
        T=1100.0,
        log_h2o=-3.5,
        log_co2=-4.0,
        log_co=-4.5,
        log_ch4=-5.0,
        log_so2=-6.0,
        r_ref=1.27,
        log_p_cloud=-1.0,
    )
    base.update(param_kwargs)
    params = ModelParams(**base)
    wl, depth, meta = compute_model_spectrum(params, FIXED)
    return wl, depth, label


def main() -> int:
    # Ensure outputs directory exists
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # Four example spectra
    examples = [
        make_spectrum(
            "H₂O only",
            log_co2=-10,
            log_co=-10,
            log_ch4=-10,
            log_so2=-10,
        ),
        make_spectrum(
            "H₂O + CO₂",
            log_co=-10,
            log_ch4=-10,
            log_so2=-10,
        ),
        make_spectrum(
            "H₂O + CO₂ + SO₂",
            log_co=-10,
            log_ch4=-10,
        ),
        make_spectrum(
            "Cloudy (log_p_cloud=0)",
            log_co=-10,
            log_ch4=-10,
            log_p_cloud=0.0,  # high cloud at 1 bar
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (wl, depth, label) in zip(axes, examples, strict=True):
        ax.plot(wl, depth * 1e6, "k-", linewidth=0.8)
        ax.set_xlim(0.6, 5.3)
        ax.set_ylim(0, None)
        ax.set_title(label, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Wavelength [μm]")
        ax.set_ylabel("Transit depth [ppm]")

        # Annotate key features
        ax.axvline(1.4, color="blue", alpha=0.3, linestyle="--", linewidth=0.8)
        ax.axvline(2.7, color="red", alpha=0.3, linestyle="--", linewidth=0.8)
        ax.axvline(4.3, color="green", alpha=0.3, linestyle="--", linewidth=0.8)
        ax.axvline(4.0, color="orange", alpha=0.3, linestyle="--", linewidth=0.8)

    fig.suptitle(
        f"ExoSphere Forward Model Examples (mock: {MOCK_VERSION}, {OPACITY_MODE})",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Dashed lines: H₂O (1.4 μm), CO₂ (2.7, 4.3 μm), SO₂ (4.0 μm)",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)

    print(f"Figure saved to {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
