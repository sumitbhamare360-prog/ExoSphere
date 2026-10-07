"""Synthetic dataset generation for ML molecule classifier (Phase 5).

Generates labeled spectra using the forward model with the label definition:
molecule present if log VMR >= -6 AND noiseless feature amplitude in its
band windows exceeds 1x per-point noise level.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exosphere.core.provenance import Provenance
from exosphere.core.spectrum import Spectrum
from exosphere.data.loaders.binning import bin_edges_from_centers
from exosphere.forward.model import ModelParams, PlanetFixed, compute_model_spectrum
from exosphere.quality.assess import load_quality_config

# Default WASP-39 b fixed parameters
DEFAULT_FIXED = PlanetFixed(
    gravity_m_s2=4.2,
    stellar_radius_rsun=0.93,
    reference_pressure_bar=0.01,
)

# Molecule order (matches forward model)
MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]
N_MOLECULES = len(MOLECULES)

# Default dataset parameters
DEFAULT_N_SAMPLES = 20000
DEFAULT_N_SAMPLES_CI = 3000


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    n_samples: int = DEFAULT_N_SAMPLES
    target_grid: str = "wasp39b_prism"  # use cleaned WASP-39b PRISM grid
    noise_scale_min: float = 0.5
    noise_scale_max: float = 5.0
    vertical_offset_sigma: float = 1e-4
    uncertainty_misestimation_frac: float = 0.05
    absent_prob: float = 0.4  # probability molecule is set to log VMR = -12
    seed: int = 42
    dataset_version: str = "ml-dataset-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        """Stable hash of configuration for versioning."""
        content = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(content).hexdigest()[:16]


def load_target_grid() -> tuple[np.ndarray, np.ndarray]:
    """Load the cleaned WASP-39 b PRISM wavelength grid and bin edges."""
    from exosphere.core.config import load_config

    cfg = load_config()
    benchmark_dir = cfg.data_cache_dir / "benchmark"
    # Use the Carter et al. 2024 spectrum as the reference grid
    grid_file = benchmark_dir / "40_24_96_78_WASP_39_b_3.11466_5502_6.tbl.npz"
    if not grid_file.exists():
        raise FileNotFoundError(
            f"Reference grid not found at {grid_file}. Run fetch_wasp39b.py first."
        )
    spec = Spectrum.load(grid_file)
    return np.array(spec.wavelength), np.array(spec.wavelength_bin_edges)


def sample_atmosphere_params(rng: np.random.Generator) -> ModelParams:
    """Sample atmosphere parameters from the prior distribution."""
    # T uniform in [300, 2500]
    T = float(rng.uniform(300.0, 2500.0))

    # log VMRs: with prob absent_prob, set to -12 (absent), else uniform in [-12, -1]
    log_vmrs = []
    for _ in range(5):
        if rng.random() < 0.4:  # 40% chance absent
            log_vmrs.append(-12.0)
        else:
            log_vmrs.append(float(rng.uniform(-12.0, -1.0)))

    # r_ref: uniform in [0.7, 1.3] * catalog (1.27 R_Jup)
    r_ref = float(rng.uniform(0.7 * 1.27, 1.3 * 1.27))

    # log_p_cloud: uniform in [-6, 2]
    log_p_cloud = float(rng.uniform(-6.0, 2.0))

    return ModelParams(
        T=T,
        log_h2o=log_vmrs[0],
        log_co2=log_vmrs[1],
        log_co=log_vmrs[2],
        log_ch4=log_vmrs[3],
        log_so2=log_vmrs[4],
        r_ref=r_ref,
        log_p_cloud=log_p_cloud,
    )


def compute_feature_amplitude(
    wl: np.ndarray,
    depth_with: np.ndarray,
    depth_without: np.ndarray,
    band_windows: list[tuple[float, float]],
) -> float:
    """Compute max feature amplitude in band windows.

    Returns max depth deviation in the molecule's band windows.
    """
    deviation = np.abs(depth_with - depth_without)
    max_amp = 0.0
    for lo, hi in band_windows:
        mask = (wl >= lo) & (wl <= hi)
        if np.any(mask):
            max_amp = max(max_amp, float(np.max(deviation[mask])))
    return max_amp


def compute_label(
    params: ModelParams,
    fixed: PlanetFixed,
    band_windows: dict[str, list[tuple[float, float]]],
    noise_level: float,
) -> np.ndarray:
    """Compute binary labels for each molecule.

    Label = 1 if log VMR >= -6 AND feature amplitude > noise_level.
    Uses native model grid for feature amplitude computation.
    """
    labels = np.zeros(5, dtype=np.float32)

    # Compute full model spectrum on native grid
    wl_native, depth_with, _ = compute_model_spectrum(params, DEFAULT_FIXED)

    for i, mol in enumerate(MOLECULES):
        log_vmr = getattr(params, f"log_{mol.lower()}")
        if log_vmr < -6:
            labels[i] = 0.0
            continue

        # Compute feature amplitude by comparing with/without this molecule
        params_without = ModelParams(
            T=params.T,
            log_h2o=params.log_h2o if mol != "H2O" else -12.0,
            log_co2=params.log_co2 if mol != "CO2" else -12.0,
            log_co=params.log_co if mol != "CO" else -12.0,
            log_ch4=params.log_ch4 if mol != "CH4" else -12.0,
            log_so2=params.log_so2 if mol != "SO2" else -12.0,
            r_ref=params.r_ref,
            log_p_cloud=params.log_p_cloud,
        )

        # Compute noiseless spectra on native grid
        _, depth_without, _ = compute_model_spectrum(params_without, DEFAULT_FIXED)

        # Compute feature amplitude in this molecule's band windows (on native grid)
        amp = compute_feature_amplitude(wl_native, depth_with, depth_without, band_windows[mol])

        # Label present if amplitude > noise_level
        labels[i] = 1.0 if amp > noise_level else 0.0

    return labels


def apply_domain_randomization(
    depth: np.ndarray,
    uncertainty: np.ndarray,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply domain randomization: vertical offset and uncertainty mis-estimation."""
    # Random vertical offset
    offset = rng.normal(0.0, config.vertical_offset_sigma)
    depth = depth + offset

    # Uncertainty mis-estimation (5% by default)
    mis_factor = 1.0 + rng.uniform(
        -config.uncertainty_misestimation_frac, config.uncertainty_misestimation_frac
    )
    uncertainty = uncertainty * mis_factor

    return depth, uncertainty


def generate_dataset(
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Generate a synthetic dataset.

    Returns:
        X: (n_samples, 2, n_wavelength) - depth and uncertainty channels
        y: (n_samples, 5) - binary labels per molecule
        params_list: list of dicts with true parameters
        metadata: list of dicts with generation metadata per sample
    """
    # Load target grid
    wl_target, edges_target = load_target_grid()

    # Get WASP-39 b median uncertainty profile for noise scaling
    from exosphere.core.config import load_config

    cfg = load_config()
    bench_file = cfg.data_cache_dir / "benchmark" / "40_24_96_78_WASP_39_b_3.11466_5502_6.tbl.npz"
    ref_spec = Spectrum.load(bench_file)
    ref_uncertainty = np.array(ref_spec.uncertainty)

    # Interpolate reference uncertainty to target grid
    from scipy.interpolate import interp1d

    ref_wl = np.array(Spectrum.load(bench_file).wavelength)
    f_interp = interp1d(
        ref_wl, ref_uncertainty, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    base_uncertainty = f_interp(wl_target)
    median_uncertainty = float(np.median(base_uncertainty))

    X = np.zeros((config.n_samples, 2, len(wl_target)), dtype=np.float32)
    y = np.zeros((config.n_samples, 5), dtype=np.float32)
    params_list = []
    metadata = []

    for i in range(config.n_samples):
        # Sample atmosphere
        params = sample_atmosphere_params(np.random.default_rng(config.seed + i))

        # Compute noiseless spectrum on target grid
        wl_model, depth_model, _ = compute_model_spectrum(params, DEFAULT_FIXED)

        # Bin to target grid
        from exosphere.forward.model import to_instrument

        target_spectrum = Spectrum(
            wavelength=wl_target.tolist(),
            transmission=np.zeros_like(wl_target).tolist(),
            uncertainty=np.ones_like(wl_target).tolist(),
            wavelength_bin_edges=bin_edges_from_centers(wl_target),
            observation_id="gen",
            target_id="SYNTH",
            instrument="NIRSpec PRISM",
            provenance=Provenance(analysis_id="EXO-000001", planet="SYNTH"),
        )
        _, depth_binned = to_instrument(wl_model, depth_model, target_spectrum)

        # Sample noise scale and compute noise level
        rng_sample = np.random.default_rng(config.seed + i + 10000)
        noise_scale = float(rng_sample.uniform(0.5, 5.0))
        noise_level = median_uncertainty * noise_scale

        # Also bin uncertainty to target grid
        uncertainty_model = np.full_like(depth_model, noise_level)
        _, uncertainty_binned = to_instrument(wl_model, uncertainty_model, target_spectrum)

        # Apply domain randomization
        depth_noisy, uncertainty_binned = apply_domain_randomization(
            depth_binned,
            uncertainty_binned,
            np.random.default_rng(i),
            type(
                "Config",
                (),
                {"vertical_offset_sigma": 1e-4, "uncertainty_misestimation_frac": 0.05},
            )(),
        )

        # Compute labels
        labels = compute_label(
            ModelParams(
                T=params.T,
                log_h2o=params.log_h2o,
                log_co2=params.log_co2,
                log_co=params.log_co,
                log_ch4=params.log_ch4,
                log_so2=params.log_so2,
                r_ref=params.r_ref,
                log_p_cloud=params.log_p_cloud,
            ),
            DEFAULT_FIXED,
            {k: v for k, v in load_quality_config().molecule_bands.items()},
            noise_level=median_uncertainty * 1.0,  # use median noise as threshold
        )

        # Store
        X[i, 0] = depth_noisy.astype(np.float32)
        X[i, 1] = uncertainty_binned.astype(np.float32)
        y[i] = labels

        params_list.append(
            {
                "T": params.T,
                "log_h2o": params.log_h2o,
                "log_co2": params.log_co2,
                "log_co": params.log_co,
                "log_ch4": params.log_ch4,
                "log_so2": params.log_so2,
                "r_ref": params.r_ref,
                "log_p_cloud": params.log_p_cloud,
            }
        )

        metadata.append(
            {
                "seed": i,
                "noise_scale": noise_scale,
                "labels": labels.tolist(),
            }
        )

    return X, y, params_list, metadata


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    params_list: list[dict],
    metadata: list[dict],
    config: DatasetConfig,
    output_dir: Path,
) -> Path:
    """Save dataset to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.savez_compressed(
        output_dir / "dataset.npz",
        X=X,
        y=y,
    )

    # Save metadata
    metadata_dict = {
        "config": config.to_dict(),
        "config_hash": config.hash,
        "params": params_list,
        "metadata": metadata,
        "molecules": MOLECULES,
        "n_samples": len(metadata),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)

    return output_dir


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load dataset from disk."""
    data = np.load(dataset_dir / "dataset.npz")
    X = data["X"]
    y = data["y"]

    with open(dataset_dir / "metadata.json") as f:
        metadata = json.load(f)

    return X, y, metadata


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    import argparse

    parser = argparse.ArgumentParser(description="Generate ML dataset")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--output-dir", type=str, default="data_cache/ml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DatasetConfig(n_samples=args.n_samples, seed=args.seed)
    output_dir = Path(args.output_dir)

    print(f"Generating {args.n_samples} samples...")
    start = time.time()

    X, y, params, metadata = generate_dataset(config)

    output_dir = Path(args.output_dir)
    save_dataset(X, y, [], [], config, output_dir)

    print(f"Generated {len(y)} samples in {time.time() - start:.1f}s")
    print(f"Saved to {output_dir}")
