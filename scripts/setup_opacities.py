"""Opacity setup for real petitRADTRANS (pRT 3.4, correlated-k R1000).

V1 file manifest (verified against the Keeper share; sizes in bytes as served).
Paths are relative to the pRT input-data root (PRT_INPUT_DATA_PATH), which
must contain the standard ``opacities/...`` layout (see pRT docs, "Setting up
the input_data directory").

Keeper downloads use plain HTTPS (no selenium): the share answers direct
``/files/?p=<path>&dl=1`` links. TLS needs the certifi CA bundle on this box
(system store lacks the HARICA academic CA).

Usage (run with the pRT venv so ``certifi``/``requests`` exist):
    .\\.venv311\\Scripts\\python.exe scripts/setup_opacities.py --status
    .\\.venv311\\Scripts\\python.exe scripts/setup_opacities.py --download
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

KEEPER_BASE = "https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0/files/?p="

# (relative path under input_data root, served size in bytes or None)
# Sizes verified 2026-10-12 via response headers.
V1_FILES: list[tuple[str, int | None]] = [
    (
        "opacities/lines/correlated_k/H2O/1H2-16O/"
        "1H2-16O__POKAZATEL.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
        388997984,
    ),
    (
        "opacities/lines/correlated_k/CO2/12C-16O2/"
        "12C-16O2__UCL-4000.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
        388998528,
    ),
    (
        "opacities/lines/correlated_k/CO/12C-16O/"
        "12C-16O__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5",
        128730792,
    ),
    (
        "opacities/lines/correlated_k/CH4/12C-1H4/"
        "12C-1H4__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5",
        158406312,
    ),
    (
        "opacities/lines/correlated_k/SO2/32S-16O2/"
        "32S-16O2__ExoAmes.R1000_0.3-50mu.ktable.petitRADTRANS.h5",
        388998528,
    ),
]

# CIA files (H2-H2, H2-He) are still needed: ~20 Keeper path variants probed
# without a hit (see DECISIONS.md). The backend runs without CIA files until
# their exact share paths are known; Rayleigh needs no files (analytic).
PENDING_CIA = ["H2-H2 (BoRi)", "H2-He (BoRi)"]


def input_data_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    from exosphere.core.config import load_config

    cfg = load_config()
    root = Path(
        __import__("os").environ.get(
            "PRT_INPUT_DATA_PATH", str(cfg.data_cache_dir / "prt_input_data")
        )
    )
    return root.expanduser().resolve()


def _session():
    import certifi
    import requests

    session = requests.Session()
    session.verify = certifi.where()
    return session


def download_file(session, rel_path: str, dest: Path, expected_size: int | None) -> str:
    """Download one Keeper file; skip if present with matching size. Returns status."""
    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = KEEPER_BASE + urllib.parse.quote("/" + rel_path, safe="") + "&dl=1"
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, timeout=600, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "octet-stream" not in content_type:
            raise RuntimeError(f"unexpected content-type {content_type!r} for {rel_path}")
        total = int(response.headers.get("content-length", "0") or 0)
        done = 0
        start = __import__("time").time()
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r  {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB "
                        f"({done / max(time.time() - start, 1e-6) / 1e6:.1f} MB/s)",
                        end="",
                    )
        print()
    # HDF5 magic check
    with open(tmp, "rb") as handle:
        magic = handle.read(8)
    if magic != b"\x89HDF\r\n\x1a\n":
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{rel_path}: not an HDF5 file (bad magic)")
    if expected_size and tmp.stat().st_size != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{rel_path}: size mismatch after download")
    tmp.replace(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()[:16]
    return f"downloaded sha256:{digest}"


def status(root: Path) -> int:
    print(f"pRT input-data root: {root}")
    print("V1 correlated-k files (R1000):")
    missing = 0
    total_bytes = 0
    for rel_path, size in V1_FILES:
        dest = root / rel_path
        if dest.exists():
            state = f"present ({dest.stat().st_size / 1e6:.0f} MB)"
            total_bytes += dest.stat().st_size
        else:
            state = f"MISSING (expected ~{size / 1e6:.0f} MB)"
            missing += 1
        print(f"  [{state}] {rel_path}")
    print(f"on-disk V1 total: {total_bytes / 1e6:.0f} MB")
    print(f"CIA still pending (backend runs without): {', '.join(PENDING_CIA)}")
    print("Rayleigh H2/He: analytic in pRT, no files needed.")
    return 1 if missing else 0


def download(root: Path) -> int:
    session = _session()
    failures = 0
    for rel_path, size in V1_FILES:
        dest = root / rel_path
        try:
            print(f"{dest.name}: {download_file(session, rel_path, dest, size)}")
        except Exception as error:
            print(f"{dest.name}: FAILED: {error}")
            failures += 1
    return 1 if failures else status(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", help="input-data root (default: $PRT_INPUT_DATA_PATH)")
    parser.add_argument("--download", action="store_true", help="fetch the V1 files")
    parser.add_argument("--status", action="store_true", help="report present/missing (default)")
    args = parser.parse_args(argv)
    root = input_data_root(args.path)
    root.mkdir(parents=True, exist_ok=True)
    if args.download:
        return download(root)
    return status(root)


if __name__ == "__main__":
    raise SystemExit(main())
