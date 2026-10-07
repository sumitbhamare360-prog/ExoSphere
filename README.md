# ExoSphere

End-to-end exoplanet atmospheric characterization platform:
observed JWST transmission spectrum -> molecular signatures -> Bayesian atmospheric
retrieval -> posteriors/uncertainty -> dashboard + 3D scientific digital twin.

Pilot target: WASP-39 b (JWST / NIRSpec PRISM, 0.6-5.3 um). Code is planet-independent.
See `AGENTS.md` for scope, frozen V1 decisions, and working protocol.

## Setup (Phase 0)

Requires Python 3.11+ (3.11 is the target version).

```powershell
# from the repository root
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `PRT_INPUT_DATA_PATH` | petitRADTRANS opacity data directory (needed only from the forward-model phase onwards) | unset |

Note: petitRADTRANS, JAX/JAXNS, dynesty, and torch are intentionally **not** installed in
Phase 0; they are added in later phases per `DECISIONS.md`.

## Run checks

```powershell
pytest               # unit + L1 tests (live-network tests deselected)
pytest -m network    # optional live tests against NASA Exoplanet Archive / MAST
ruff check .
```

## Data acquisition (Phase 1)

```powershell
python scripts\fetch_wasp39b.py    # benchmark: WASP-39 b NIRSpec PRISM spectra
```

Downloads literature spectra from the NASA Exoplanet Archive into
`data_cache/` (with SHA256 checksums) and loads them into `Spectrum`. MAST
observation metadata / calibrated 1D products are available via
`exosphere.data.mast` (astroquery). All responses are cached under
`data_cache/` (gitignored).

## Quality assessment + preprocessing (Phase 2)

```powershell
python scripts\quality_wasp39b.py   # assess + clean the cached spectrum, write outputs\quality_wasp39b.png
```

- `config/molecule_bands.yaml` - molecule band windows + ALL quality thresholds.
- `exosphere.quality.assess(spectrum)` - JSON-serializable `QualityReport`:
  overall GOOD/LIMITED/POOR suitability + per-molecule rating (data capability
  in that molecule's windows, not molecule presence).
- `exosphere.preprocess.clean(spectrum, options)` - drops NaN/flagged/bad-sigma
  points, robust MAD spike clip, optional inverse-variance rebin; returns a new
  `Spectrum` + `PreprocessLog` (input never mutated).

## Quality assessment + preprocessing (Phase 2)

```powershell
python scripts\quality_wasp39b.py   # assess + clean the cached spectrum, write outputs\quality_wasp39b.png
```

- `config/molecule_bands.yaml` - molecule band windows + ALL quality thresholds.
- `exosphere.quality.assess(spectrum)` - JSON-serializable `QualityReport`:
  overall GOOD/LIMITED/POOR suitability + per-molecule rating (data capability
  in that molecule's windows, not molecule presence).
- `exosphere.preprocess.clean(spectrum, options)` - drops NaN/flagged/bad-sigma
  points, robust MAD spike clip, optional inverse-variance rebin; returns a new
  `Spectrum` + `PreprocessLog` (input never mutated).

## Forward model (Phase 3)

```powershell
python scripts\plot_forward_examples.py   # generates outputs\forward_examples.png
```

- `scripts/setup_opacities.py` — check `PRT_INPUT_DATA_PATH`, list required
  opacities for H2O/CO2/CO/CH4/SO2 + H2-H2/H2-He CIA + Rayleigh.
- `src/exosphere/forward/model.py` — mock forward model (`opacity_mode = mock-gaussian-R1000`):
  `ModelParams`, `PlanetFixed`, `transmission_spectrum()`, `to_instrument()`,
  `compute_model_spectrum()`. Real pRT unavailable on Windows/Python 3.13/numpy 2.x;
  fallback documented in `DECISIONS.md`.
- `scripts/plot_forward_examples.py` — 4-panel figure:
  H2O-only; H2O+CO2; H2O+CO2+SO2; cloudy version.

- `scripts/plot_forward_examples.py` — 4-panel figure:
  H2O-only; H2O+CO2; H2O+CO2+SO2; cloudy version.

## Bayesian retrieval + L2 validation (Phase 4)

```powershell
python scripts\run_l2.py --n-live 100 --dlogz 0.05   # L2 synthetic validation
python scripts\plot_retrieval.py --result outputs/retrieval_*.npz --spectrum data_cache/benchmark/40_24_96_78_WASP_39_b_3.11466_5502_6.tbl.npz --output outputs/retrieval_wasp39b.png
```

- `src/exosphere/retrieval/` — Bayesian nested sampling retrieval:
  `priors.py`, `likelihood.py`, `samplers.py` (dynesty + JAXNS stub), `results.py`, `detection.py`.
  dynesty (working) + JAXNS (stub; needs JAX-traceable forward model).
- `scripts/run_l2.py` — L2 synthetic validation: generates synthetic spectra, runs retrieval, evaluates recovery; outputs `outputs/l2_results.md` + `.json`.
- `scripts/plot_retrieval.py` — corner plot + best-fit spectrum with 68% credible band + detection summary → `outputs/retrieval_<id>.png`.
- `tests/validation/test_l2_synthetic.py` — L2 validation tests (`@pytest.mark.slow`).

## Repository layout

```
src/exosphere/   core/, data/, quality/, preprocess/, forward/, retrieval/, ml/, api/, report/
web/             React dashboard + 3D twin
tests/           unit + validation (L1/L2/L3)
data_cache/      gitignored local data cache
```
