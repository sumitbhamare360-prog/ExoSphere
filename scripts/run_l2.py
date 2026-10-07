"""L2 Synthetic Validation: known atmosphere -> synthetic spectrum
+ noise -> retrieval -> recovered params.

Runs retrieval on synthetic spectra with known parameters and checks recovery.
Outputs results table to outputs/l2_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
from exosphere.core.provenance import Provenance  # noqa: E402
from exosphere.core.spectrum import Spectrum  # noqa: E402
from exosphere.data.loaders.binning import bin_edges_from_centers  # noqa: E402
from exosphere.forward.model import ModelParams, PlanetFixed, compute_model_spectrum  # noqa: E402
from exosphere.retrieval.detection import detection_summary  # noqa: E402
from exosphere.retrieval.samplers import SamplerConfig, run_dynesty  # noqa: E402

# Test cases: (name, ModelParams kwargs)
TEST_CASES = [
    ("H2O_CO2", dict(log_co=-10, log_ch4=-10, log_so2=-10)),
    ("H2O_CO2_CO_SO2", dict(log_ch4=-10)),
    ("cloudy_H2O", dict(log_co2=-10, log_co=-10, log_ch4=-10, log_so2=-10, log_p_cloud=0.0)),
    ("all_five", {}),
]

# Noise levels: (name, multiplier on median uncertainty)
NOISE_LEVELS = [
    ("WASP39_median", 1.0),
    ("WASP39_3x", 3.0),
]

SEEDS = [42, 123, 456]


def make_synthetic_spectrum(
    params: ModelParams,
    fixed: PlanetFixed,
    noise_multiplier: float,
    seed: int,
) -> Spectrum:
    """Generate synthetic spectrum with noise."""
    wl, depth, _ = compute_model_spectrum(params, fixed)
    rng = np.random.default_rng(seed)

    # Use median uncertainty from model
    # For synthetic test, use a fixed relative uncertainty
    median_depth = np.median(depth)
    # Use uncertainty proportional to depth (typical for photon noise)
    base_uncertainty = median_depth * 0.001  # ~0.1% relative precision
    uncertainty = np.full_like(depth, base_uncertainty) * noise_multiplier

    noisy_depth = depth + rng.normal(0, uncertainty)

    edges = bin_edges_from_centers(wl)
    prov = Provenance(analysis_id="EXO-000001", planet="SYNTH")
    return Spectrum(
        wavelength=wl.tolist(),
        transmission=noisy_depth.tolist(),
        uncertainty=uncertainty.tolist(),
        wavelength_bin_edges=edges,
        quality_flags=[],
        observation_id=f"L2_{seed}",
        target_id="SYNTH b",
        instrument="NIRSpec PRISM",
        provenance=prov,
    )


def run_single_test(
    name: str,
    params: ModelParams,
    fixed: PlanetFixed,
    noise_multiplier: float,
    seed: int,
    sampler_config: SamplerConfig,
) -> dict[str, Any]:
    """Run retrieval on a single synthetic spectrum and evaluate recovery."""
    start = time.time()

    # Generate synthetic data
    spectrum = make_synthetic_spectrum(params, fixed, noise_multiplier, seed)

    # Run retrieval
    result = run_dynesty(spectrum, fixed, sampler_config, seed)

    time.time() - start

    # Evaluate recovery
    true_params = np.array(
        [
            params.T,
            params.log_h2o,
            params.log_co2,
            params.log_co,
            params.log_ch4,
            params.log_so2,
            params.r_ref,
            params.log_p_cloud,
        ]
    )

    in_ci68 = []
    in_ci95 = []
    for i in range(8):
        lo68, hi68 = result.ci_68[i]
        lo95, hi95 = result.ci_95[i]
        in_ci68.append(lo68 <= true_params[i] <= hi68)
        in_ci95.append(lo95 <= true_params[i] <= hi95)

    # Detection results
    det_summary = detection_summary(result, None, None, sampler_config, seed)

    return {
        "name": name,
        "noise": noise_multiplier,
        "seed": seed,
        "true_params": true_params.tolist(),
        "recovered_median": result.median.tolist(),
        "in_ci68": in_ci68,
        "in_ci95": in_ci95,
        "logz": result.logz,
        "logz_err": result.logz_err,
        "runtime_s": time.time() - start,
        "detection": det_summary,
    }


def run_l2_validation(
    sampler_config: SamplerConfig,
    full: bool = False,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run full L2 validation suite."""
    fixed = PlanetFixed(gravity_m_s2=4.2, stellar_radius_rsun=0.93, reference_pressure_bar=0.01)

    base_params = dict(
        T=1100.0,
        log_h2o=-3.5,
        log_co2=-4.0,
        log_co=-4.5,
        log_ch4=-5.0,
        log_so2=-6.0,
        r_ref=1.27,
        log_p_cloud=-1.0,
    )

    test_cases = TEST_CASES
    noise_levels = NOISE_LEVELS
    test_seeds = seeds or [42]

    if not full:
        # Fast mode: subset
        test_cases = test_cases[:2]
        noise_levels = noise_levels[:1]
        test_seeds = test_seeds[:1]

    results = []
    for case_name, case_params in test_cases:
        params = ModelParams(**{**base_params, **case_params})
        for noise_name, noise_mult in noise_levels:
            for seed in test_seeds:
                print(f"Running: {case_name} | {noise_name} | seed={seed}")
                test_name = f"{case_name}_{noise_name}_seed{seed}"
                result = run_single_test(
                    test_name,
                    params,
                    fixed,
                    noise_multiplier=noise_mult,
                    seed=seed,
                    sampler_config=sampler_config,
                )
                result["case"] = case_name
                result["noise_level"] = noise_name
                results.append(result)

    return results


def summarize_results(results: list[dict]) -> str:
    """Generate markdown results table."""
    lines = [
        "# L2 Synthetic Validation Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total tests: {len(results)}",
        "",
        "| Test | Case | Noise | Seed | Params in 68% CI | Params in 95% CI | logZ | Runtime (s) |",
        "|------|------|-------|------|------------------|------------------|------|-------------|",
    ]

    for r in results:
        n_params = 8
        in68 = sum(r["in_ci68"])
        in95 = sum(r["in_ci95"])
        lines.append(
            f"| {r['name']} | {r.get('case', '')} | {r.get('noise_level', '')} | {r['seed']} | "
            f"{in68}/{n_params} | {in95}/{n_params} | {r['logz']:.2f} | {r['runtime_s']:.1f} |"
        )

    lines.append("")
    lines.append("## Per-parameter coverage")

    # Per-parameter summary
    param_names = [
        "T",
        "log_h2o",
        "log_co2",
        "log_co",
        "log_ch4",
        "log_so2",
        "r_ref",
        "log_p_cloud",
    ]
    for i, pname in enumerate(param_names):
        in68_count = sum(1 for r in results if r["in_ci68"][i])
        in95_count = sum(1 for r in results if r["in_ci95"][i])
        lines.append(
            f"- {pname}: 68% CI: {in68_count}/{len(results)}, 95% CI: {in95_count}/{len(results)}"
        )

    lines.append("")
    lines.append("## Detection summary")
    for r in results:
        det = r.get("detection", {})
        if det:
            detected = [m for m, v in det.items() if v.get("status") == "detected"]
            if detected:
                lines.append(f"- {r['name']}: detected {', '.join(detected)}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run L2 synthetic validation")
    parser.add_argument(
        "--full", action="store_true", help="Run full suite (all cases, noise levels, seeds)"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to test")
    parser.add_argument(
        "--n-live", type=int, default=200, help="Number of live points (lower = faster)"
    )
    parser.add_argument("--dlogz", type=float, default=0.05, help="Stopping criterion dlogz")
    parser.add_argument(
        "--output", type=str, default="outputs/l2_results.md", help="Output markdown file"
    )
    args = parser.parse_args()

    sampler_config = SamplerConfig(
        sampler="dynesty",
        n_live=args.n_live,
        dlogz=args.dlogz,
        max_iter=50000,
        seed=42,
        error_inflation=0.0,
    )

    print("Starting L2 synthetic validation...")
    start = time.time()
    results = run_l2_validation(sampler_config, full=args.full, seeds=args.seeds)
    total_time = time.time() - start

    print(f"Completed in {total_time:.1f} s")

    # Save results markdown
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summarize_results(results) + f"\n\nTotal runtime: {total_time:.1f} s\n")

    # Also save raw results as JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"Results saved to {output_path} and {json_path}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    main()
