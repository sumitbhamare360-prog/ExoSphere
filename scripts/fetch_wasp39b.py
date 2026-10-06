"""Fetch the WASP-39 b NIRSpec PRISM transmission spectrum(s) from MAST/Exoplanet Archive.

The NASA Exoplanet Archive Atmospheric Spectroscopy table is queried for
WASP-39 b. Transmission spectra whose instrument/note mention PRISM are
downloaded into data_cache/, checksummed (SHA256), loaded into Spectrum, and
summarised in a JSON manifest.

If no PRISM spectrum exists, the spectra that *are* available are listed and
the script exits with status 1 (no silent substitution).

Usage:
    python scripts/fetch_wasp39b.py [--target "WASP-39 b"] [--refresh]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from exosphere.core.config import load_config  # noqa: E402
from exosphere.core.provenance import AnalysisIdGenerator, Provenance  # noqa: E402
from exosphere.data.exoarchive import (  # noqa: E402
    ExoplanetArchiveError,
    LiteratureSpectrum,
    cached_spectrum_path,
    get_spectrum,
    list_literature_spectra,
    sha256_of_file,
)

DEFAULT_TARGET = "WASP-39 b"


def _is_prism(spectrum: LiteratureSpectrum) -> bool:
    haystack = " ".join(
        part for part in (spectrum.instrument, spectrum.note, spectrum.facility) if part
    )
    return "prism" in haystack.lower()


def _report_available(spectra: list[LiteratureSpectrum]) -> None:
    print("Available spectra in the Exoplanet Archive for this target:")
    print(f"  {'type':16s} | {'instrument':45s} | {'range [um]':16s} | reference")
    for spectrum in sorted(spectra, key=lambda s: (s.spectrum_type, s.instrument or "")):
        if spectrum.min_wavelength_um is not None and spectrum.max_wavelength_um is not None:
            range_text = f"{spectrum.min_wavelength_um:.3f}-{spectrum.max_wavelength_um:.3f}"
        else:
            range_text = "n/a"
        print(
            f"  {spectrum.spectrum_type:16s} | {str(spectrum.instrument):45s} "
            f"| {range_text:16s} | {spectrum.reference} ({spectrum.bibcode})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached archive responses"
    )
    args = parser.parse_args(argv)

    out_dir = load_config().data_cache_dir / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = load_config().data_cache_dir / "exoarchive"

    spectra = list_literature_spectra(args.target, cache_dir=cache_dir, refresh=args.refresh)
    transmission = [s for s in spectra if s.spectrum_type.lower() == "transmission"]
    prism = [s for s in transmission if _is_prism(s)]

    if not prism:
        print(
            f"No NIRSpec PRISM transmission spectrum found in the NASA Exoplanet "
            f"Archive for {args.target!r}."
        )
        _report_available(spectra)
        return 1

    print(
        f"Found {len(prism)} PRISM transmission spectrum(s) for {args.target!r} "
        f"({len(spectra)} spectra total in the archive):"
    )

    generator = AnalysisIdGenerator()
    manifest: dict = {
        "target": args.target,
        "generated_utc": datetime.now(UTC).isoformat(),
        "source": "NASA Exoplanet Archive, Atmospheric Spectroscopy table",
        "spectra": [],
    }

    for spectrum in prism:
        provenance = Provenance(
            analysis_id=generator.next_id(),
            planet=args.target,
            observation_id=spectrum.spectrum_id,
            telescope="JWST",
            instrument=spectrum.instrument,
            source_archive="NASA Exoplanet Archive (Atmospheric Spectroscopy table)",
            input_data_version=spectrum.bibcode,
        )
        loaded = get_spectrum(spectrum, provenance=provenance, cache_dir=cache_dir,
                              refresh=args.refresh)
        raw_path = cached_spectrum_path(spectrum.spec_path, cache_dir)
        checksum = sha256_of_file(raw_path)
        sidecar = raw_path.with_suffix(raw_path.suffix + ".sha256")
        sidecar.write_text(f"{checksum}  {raw_path.name}\n", encoding="utf-8")

        npz_path = out_dir / f"{spectrum.spec_path.replace('/', '_')}.npz"
        loaded.save(npz_path)

        entry = {
            "spectrum_id": spectrum.spectrum_id,
            "reference": spectrum.reference,
            "bibcode": spectrum.bibcode,
            "instrument": spectrum.instrument,
            "note": spectrum.note,
            "num_datapoints_source": spectrum.num_datapoints,
            "num_datapoints_loaded": len(loaded.wavelength),
            "wavelength_min_um": min(loaded.wavelength),
            "wavelength_max_um": max(loaded.wavelength),
            "depth_min_fraction": min(loaded.transmission),
            "depth_max_fraction": max(loaded.transmission),
            "source_file": str(raw_path),
            "source_sha256": checksum,
            "spectrum_file": str(npz_path),
        }
        manifest["spectra"].append(entry)
        print(
            f"  OK {spectrum.spectrum_id}\n"
            f"     reference: {spectrum.reference} ({spectrum.bibcode})\n"
            f"     instrument: {spectrum.instrument} | note: {spectrum.note}\n"
            f"     points: {entry['num_datapoints_loaded']} | "
            f"range: {entry['wavelength_min_um']:.4f}-{entry['wavelength_max_um']:.4f} um\n"
            f"     sha256: {checksum}\n"
            f"     saved: {npz_path}"
        )

    manifest_path = out_dir / "wasp39b_prism_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExoplanetArchiveError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
