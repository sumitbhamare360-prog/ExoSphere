"""Plot retrieval results: corner plot and best-fit spectrum with credible band.

Saves to outputs/retrieval_<analysis_id>.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from exosphere.core.spectrum import Spectrum  # noqa: E402
from exosphere.forward.model import PlanetFixed  # noqa: E402
from exosphere.retrieval.results import RetrievalResult  # noqa: E402


def plot_retrieval(
    result: RetrievalResult, fixed: PlanetFixed, output_path: Path, spectrum: Spectrum | None = None
) -> Path:
    """Create corner plot and best-fit spectrum with credible band.

    Args:
        result: RetrievalResult
        fixed: PlanetFixed parameters
        output_path: Path to save the figure
        spectrum: Optional observed Spectrum (for best-fit plot)

    Returns:
        Path to saved figure
    """

    samples, labels, weights = result.corner_data()

    # Figure: 2x2 grid (corner + spectrum)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], hspace=0.3, wspace=0.25)

    # Corner plot (upper left, spans both columns)
    ax_corner = fig.add_subplot(gs[0, :])
    corner_plot(ax_corner, samples, labels, weights)

    # Best-fit spectrum (lower left)
    ax_spec = fig.add_subplot(gs[1, 0])
    plot_best_fit_spectrum(ax_spec, result, fixed, spectrum)

    # Detection summary (lower right)
    ax_det = fig.add_subplot(gs[1, 1])
    plot_detection_summary(ax_det, result)

    fig.suptitle(
        f"Retrieval: {result.provenance.analysis_id if result.provenance else 'N/A'}"
        f" ({result.sampler}, seed={result.seed})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def corner_plot(ax, samples: np.ndarray, labels: list[str], weights: np.ndarray):
    """Simple corner plot (diagonal: 1D marginals; off-diagonal: 2D contours)."""

    ndim = samples.shape[1]
    # Subsample for speed
    n_samples = min(5000, len(samples))
    idx = np.random.choice(len(samples), size=n_samples, replace=False, p=weights)
    samples[idx]
    subweights = weights[idx]
    subweights = subweights / subweights.sum()

    for _i in range(ndim):
        for _j in range(ndim):
            # Create subplot in a grid
            pass

    # Simplified: just 1D marginals on diagonal
    for _ in range(ndim):
        break

    # For now, just plot 1D marginals on a single axis (simplified)
    for i in range(ndim):
        ax.hist(
            samples[:, i],
            bins=40,
            density=True,
            alpha=0.5,
            weights=weights,
            histtype="step",
            label=labels[i],
        )

    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Posterior marginals")


def plot_best_fit_spectrum(
    ax, result: RetrievalResult, fixed: PlanetFixed, spectrum: Spectrum | None = None
):
    """Plot best-fit model with 68% credible band."""
    wl, depth = result.best_fit_spectrum(fixed)
    median, lo, hi = result.credible_band_spectrum(fixed, n_draws=50)

    ax.fill_between(wl, lo * 1e6, hi * 1e6, alpha=0.3, color="blue", label="68% credible band")
    ax.plot(wl, depth * 1e6, "b-", linewidth=1.5, label="Best fit")

    if spectrum is not None:
        obs_wl = np.array(spectrum.wavelength)
        obs_depth = np.array(spectrum.transmission) * 1e6
        obs_err = np.array(spectrum.uncertainty) * 1e6
        ax.errorbar(
            obs_wl,
            obs_depth,
            yerr=obs_err,
            fmt=".",
            markersize=2,
            color="black",
            alpha=0.5,
            label="Data",
        )

    ax.set_xlim(0.6, 5.3)
    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Transit depth [ppm]")
    ax.set_title("Best-fit spectrum + 68% credible band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_detection_summary(ax, result: RetrievalResult):
    """Plot detection summary: Bayes factors and upper limits."""
    molecules = list(result.bayes_factors.keys())
    if not molecules:
        ax.text(
            0.5,
            0.5,
            "No detection info available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Molecule detection")
        return

    bfs = [result.bayes_factors[m] for m in molecules]
    [np.sqrt(2 * bf) if bf > 0 else 0 for bf in bfs]

    colors = ["green" if bf > 3 else "orange" if bf > 1 else "red" for bf in bfs]

    y_pos = np.arange(len(molecules))
    ax.barh(y_pos, bfs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(molecules)
    ax.set_xlabel("ln Bayes factor (ln B)")
    ax.set_title("Molecule detection (ln B)")
    ax.axvline(3, color="green", linestyle="--", alpha=0.5, label="Strong (ln B=3)")
    ax.axvline(1, color="orange", linestyle="--", alpha=0.5, label="Tentative (ln B=1)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="x")

    # Add sigma annotations
    for i, (_mol, bf) in enumerate(zip(molecules, bfs, strict=False)):
        if bf > 0:
            sigma = np.sqrt(2 * bf)
            ax.text(bf + 0.1, i, f"{sigma:.1f}σ", va="center", fontsize=8)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot retrieval results")
    parser.add_argument("--result", type=str, help="Path to RetrievalResult .npz or .json")
    parser.add_argument("--spectrum", type=str, help="Path to observed Spectrum .npz (optional)")
    parser.add_argument("--output", type=str, help="Output figure path")
    args = parser.parse_args()

    if not args.result:
        parser.error("--result required")

    result = (
        RetrievalResult.load_npz(args.result)
        if args.result.endswith(".npz")
        else RetrievalResult.from_json(Path(args.result).read_text())
    )

    fixed = PlanetFixed(gravity_m_s2=4.2, stellar_radius_rsun=0.93, reference_pressure_bar=0.01)

    spectrum = None
    if args.spectrum:
        spectrum = Spectrum.load(args.spectrum)

    output_path = (
        Path(args.output) if args.output else Path("outputs") / f"retrieval_{int(time.time())}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_retrieval(result, fixed, output_path, spectrum)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    import time

    main()
