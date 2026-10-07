"""Opacity setup helper for petitRADTRANS.

Checks PRT_INPUT_DATA_PATH, lists which species files are present/missing for
the V1 molecule set (H2O, CO2, CO, CH4, SO2 plus H2-H2/H2-He CIA and Rayleigh),
and downloads what pRT supports downloading.

On Windows + Python 3.13, real pRT cannot be installed (see DECISIONS.md item 20).
This script documents the expected opacity files and provides a mock mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from exosphere.core.config import load_config  # noqa: E402

# V1 molecule set + background
REQUIRED_SPECIES = [
    "H2O",
    "CO2",
    "CO",
    "CH4",
    "SO2",
    "H2-H2",
    "H2-He",
    "Rayleigh",
]

# Expected file patterns in pRT input data directory
# pRT expects files like: H2O_R_1000_0.3-30mu.h5, etc.
FILE_PATTERNS = {
    "H2O": "H2O_*.h5",
    "CO2": "CO2_*.h5",
    "CO": "CO_*.h5",
    "CH4": "CH4_*.h5",
    "SO2": "SO2_*.h5",
    "H2-H2": "H2-H2_*.h5",
    "H2-He": "H2-He_*.h5",
    "Rayleigh": "Rayleigh_*.h5",
}

# Low-resolution mode: R=1000, wavelength range 0.3-30 um (covers 0.6-5.3 um)
DEFAULT_RESOLUTION = 1000
DEFAULT_WAVE_RANGE = (0.3, 30.0)  # um


def check_opacity_dir(opacity_dir: Path) -> dict[str, list[Path]]:
    """Check which opacity files are present."""
    found = {}
    missing = {}
    for species, pattern in FILE_PATTERNS.items():
        files = list(opacity_dir.glob(pattern))
        if files:
            found[species] = files
        else:
            missing[species] = pattern
    return {"found": found, "missing": missing}


def print_status(opacity_dir: Path) -> int:
    """Print opacity status and return exit code (0 if all found, 1 if missing)."""
    print(f"Checking opacity directory: {opacity_dir}")
    print(
        f"Resolution: R={DEFAULT_RESOLUTION}, "
        f"range: {DEFAULT_WAVE_RANGE[0]}-{DEFAULT_WAVE_RANGE[1]} um"
    )
    print()

    result = check_opacity_dir(opacity_dir)

    if result["found"]:
        print("FOUND:")
        for species, files in result["found"].items():
            for f in files:
                print(f"  {species}: {f.name}")
    else:
        print("FOUND: (none)")

    print()
    if result["missing"]:
        print("MISSING:")
        for species, pattern in result["missing"].items():
            print(f"  {species}: {pattern}")
        print()
        print("Run with --download to attempt download (requires working pRT install).")
        return 1
    else:
        print("All required opacity files present.")
        return 0


def download_opacities(opacity_dir: Path, resolution: int = DEFAULT_RESOLUTION) -> int:
    """Attempt to download opacities using pRT's built-in downloader.

    Requires a working pRT installation. On Windows+Python3.13 this will fail;
    see DECISIONS.md item 20.
    """
    try:
        from petitRADTRANS import Radtrans
    except ImportError as e:
        print(f"ERROR: pRT not installed: {e}")
        print("Real pRT installation required for download. See DECISIONS.md item 20.")
        print("Use mock mode instead: python scripts/plot_forward_examples.py")
        return 1

    print(f"Downloading opacities to {opacity_dir} at R={resolution}...")
    # pRT's Radtrans constructor downloads missing files when path is set
    # This is a placeholder - actual download depends on pRT version
    try:
        _ = Radtrans(
            line_species=list(REQUIRED_SPECIES[:5]),  # H2O, CO2, CO, CH4, SO2
            rayleigh_species=["H2", "He"],
            continuum_opacities=["H2-H2", "H2-He"],
            wlen_bords_micron=DEFAULT_WAVE_RANGE,
            mode="c-k",
        )
        print("Download attempted (check pRT logs).")
        return 0
    except Exception as e:
        print(f"Download failed: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path", help="Opacity data directory (default: $PRT_INPUT_DATA_PATH or config)"
    )
    parser.add_argument(
        "--download", action="store_true", help="Attempt to download missing opacities via pRT"
    )
    parser.add_argument(
        "--resolution", type=int, default=DEFAULT_RESOLUTION, help="Spectral resolution R"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    opacity_dir = Path(args.path) if args.path else cfg.prt_input_data_path

    if opacity_dir is None:
        print("ERROR: PRT_INPUT_DATA_PATH not set. Set environment variable or use --path.")
        return 1

    opacity_dir = Path(opacity_dir).expanduser().resolve()
    opacity_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        return download_opacities(opacity_dir, args.resolution)
    else:
        return print_status(opacity_dir)


if __name__ == "__main__":
    raise SystemExit(main())
