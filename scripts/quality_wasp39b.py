"""Quality assessment + preprocessing report for the cached WASP-39 b PRISM spectrum.

Runs ``quality.assess`` on the cached benchmark spectrum, then ``preprocess.clean``
and assesses the cleaned result. Prints both reports and the preprocessing log,
and saves a diagnostic plot to ``outputs/quality_wasp39b.png``.

Usage:
    python scripts/quality_wasp39b.py [--file PATH] [--no-plot]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from exosphere.core.config import load_config  # noqa: E402
from exosphere.core.spectrum import Spectrum  # noqa: E402
from exosphere.preprocess.clean import clean  # noqa: E402
from exosphere.quality.assess import assess, load_quality_config, running_median  # noqa: E402

OUTPUT_PNG = REPO_ROOT / "outputs" / "quality_wasp39b.png"


def _pick_cached_spectrum(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"spectrum file not found: {path}")
        return path
    benchmark_dir = load_config().data_cache_dir / "benchmark"
    candidates = sorted(benchmark_dir.glob("*.npz"))
    if not candidates:
        raise SystemExit(
            f"no cached benchmark spectrum in {benchmark_dir}\nrun: python scripts/fetch_wasp39b.py"
        )
    return candidates[0]


def _plot(spectrum, cleaned, report_before, report_after, log, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    config = load_quality_config()
    wave = np.asarray(spectrum.wavelength)
    depth = np.asarray(spectrum.transmission)
    sigma = np.asarray(spectrum.uncertainty)
    clean_wave = np.asarray(cleaned.wavelength)
    clean_depth = np.asarray(cleaned.transmission)

    removed_indices = [item.index for item in log.removed]

    figure, (axis_depth, axis_snr) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1]
    )

    # Molecule band windows (measured vs. modelled labelling per AGENTS.md 4)
    band_colors = {
        "H2O": "#1f77b4",
        "CO2": "#d62728",
        "CO": "#2ca02c",
        "CH4": "#9467bd",
        "SO2": "#ff7f0e",
    }
    for molecule, windows in config.molecule_bands.items():
        for index, (lo, hi) in enumerate(windows):
            axis_depth.axvspan(
                lo,
                hi,
                alpha=0.12,
                color=band_colors.get(molecule, "gray"),
                label=f"{molecule} band" if index == 0 else None,
                zorder=0,
            )

    axis_depth.errorbar(
        wave,
        depth * 1e6,
        yerr=sigma * 1e6,
        fmt=".",
        markersize=3,
        color="0.55",
        elinewidth=0.7,
        label="input (measured)",
        zorder=1,
    )
    if removed_indices:
        axis_depth.scatter(
            wave[removed_indices],
            depth[removed_indices] * 1e6,
            marker="x",
            s=45,
            color="red",
            linewidths=1.4,
            label=f"removed ({len(removed_indices)})",
            zorder=3,
        )
    axis_depth.plot(
        clean_wave,
        clean_depth * 1e6,
        "-",
        color="#003366",
        linewidth=1.2,
        label="cleaned (modelled by preprocess)",
        zorder=2,
    )
    axis_depth.set_ylabel("Transit depth [ppm]")
    axis_depth.set_title(
        f"{spectrum.target_id} / {spectrum.instrument} - "
        f"quality: raw {report_before.suitability}, cleaned {report_after.suitability}"
    )
    axis_depth.legend(loc="upper right", fontsize=8, ncol=2)
    axis_depth.grid(alpha=0.25)

    # Point S/N against the local continuum (method shown in the report)
    clean_sigma = np.asarray(cleaned.uncertainty)
    continuum = running_median(clean_depth, config.continuum_window_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        point_snr = np.abs(clean_depth - continuum) / clean_sigma
    axis_snr.plot(clean_wave, point_snr, ".", markersize=3, color="#003366")
    axis_snr.axhline(
        config.thresholds["molecule_snr_good"],
        color="green",
        linestyle="--",
        linewidth=1,
        label="band S/N good (5)",
    )
    axis_snr.axhline(
        config.thresholds["molecule_snr_limited"],
        color="orange",
        linestyle="--",
        linewidth=1,
        label="band S/N limited (2)",
    )
    axis_snr.set_xlabel("Wavelength [um]")
    axis_snr.set_ylabel("Point S/N\n(|depth - continuum| / sigma)")
    axis_snr.set_ylim(bottom=0)
    axis_snr.legend(loc="upper right", fontsize=8)
    axis_snr.grid(alpha=0.25)

    ratings = ", ".join(f"{m.molecule} {m.rating}" for m in report_after.molecules)
    figure.text(
        0.5,
        0.005,
        f"per-molecule (cleaned): {ratings}",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="path to a cached .npz Spectrum")
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG output")
    args = parser.parse_args(argv)

    path = _pick_cached_spectrum(args.file)
    spectrum = Spectrum.load(path)
    print(f"input: {path}")
    print("=" * 78)

    report_before = assess(spectrum)
    print("RAW SPECTRUM")
    print(report_before.summary_text())
    print()

    cleaned, log = clean(spectrum)
    print("PREPROCESSING")
    print(f"  version: {log.preprocessing_version}")
    print(f"  removed: {log.n_input} -> {log.n_output} points, reasons: {log.removal_counts}")
    print(f"  parameters: {log.parameters}")
    print()

    report_after = assess(cleaned)
    print("CLEANED SPECTRUM")
    print(report_after.summary_text())
    print()

    if not args.no_plot:
        _plot(spectrum, cleaned, report_before, report_after, log, OUTPUT_PNG)
        print(f"plot written to {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
