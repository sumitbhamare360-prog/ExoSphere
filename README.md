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
  `priors.py`, `likelihood.py`, `samplers.py` (dynesty 2.1.5 pinned; JAXNS stub needs
  a JAX-traceable forward model), `results.py`, `detection.py`.
- `scripts/run_l2.py` — L2 synthetic validation: generates synthetic spectra, runs retrieval, evaluates recovery; outputs `outputs/l2_results.md` + `.json`.
- `scripts/plot_retrieval.py` — corner plot + best-fit spectrum with 68% credible band + detection summary → `outputs/retrieval_<id>.png`.

## Repository layout

```
src/exosphere/   core/, data/, quality/, preprocess/, forward/, retrieval/, ml/, api/, report/
web/             React dashboard + 3D twin
tests/           unit + validation (L1/L2/L3)
data_cache/      gitignored local data cache
```

## Quickstart (Phase 9b)

Prerequisites: Python 3.11+ with `.venv` (see Setup above), Node 20+ for `web/`.
Long retrieval runs need foreground execution with generous timeouts on this box;
background workers are unreliable here (see `DECISIONS.md`).

```powershell
# 1. Backend API (dev SQLite database is created on startup)
.\.venv\Scripts\python.exe -m uvicorn exosphere.api.app:app --host 0.0.0.0 --port 8000

# 2. Dashboard (new terminal, from web/)
cd web; npm install; npm run dev      # production build: npm run build

# 3. Run a WASP-39 b analysis (benchmark spectrum is cached; needs a Planet
#    + Observation row first — easiest via the Analyze page in the dashboard,
#    or see outputs/run_l3_pipeline.py for a scripted full run)

# 4. Generate the scientific report for a finished analysis
.\.venv\Scripts\python.exe -m exosphere.report.build EXO-000002 --format html
# -> data_cache/reports/EXO-000002_report.html (PDF unsupported here: no Pango libs)

# 5. Validation entry points
.\.venv\Scripts\python.exe -m pytest -q -m "not slow and not network"  # fast gates
.\.venv\Scripts\python.exe scripts\run_l2.py --help                     # L2 synthetic suite
.\.venv\Scripts\python.exe scripts\run_l3.py --help                     # L3 real-data comparison
.\.venv\Scripts\python.exe scripts\audit_scope.py                       # AGENTS.md sections 2-3 audit
```

Docker: `docker compose up --build` (SQLite persisted in a volume; mount real
pRT opacities at `./opacities`, read-only). Postgres via `--profile postgres`
with `DATABASE_URL` set (see `docker-compose.yml`). CI (`.github/workflows/ci.yml`):
ruff + fast pytest + web lint/test/build on push/PR; slow validation is manual.

## Architecture (text diagram)

```
NASA archives ──> data/ ──> quality/ ──> preprocess/ ──> Spectrum ─┬─> ml/ (candidate scores, advisory only)
   (TAP/MAST,        assess          clean (+log)                    │
    cached)                                                          ├─> retrieval/ (dynesty posterior + ln B) ─> api/ (SQLite) ─> web/ (dashboard + 3D twin)
                                                                     └─> report/ (HTML, wording-guarded) ──────────────^
```

Every number shown anywhere carries a posterior/credible interval or an explicit
"not run" marker; ML scores are always labelled "ML candidate score".

## Validation status (see PROGRESS.md for detail)

| Level | Status (2026-10-11) |
|---|---|
| L1 data round-trip | PASS (test + L3 recheck: 147 pts, max abs diff 0.0) |
| L2 synthetic | NOT RUN (runner repaired and ready; needs ~hours of compute) |
| L3 real WASP-39 b | PARTIAL/FAIL overall: H2O + CO2 supported (agree with literature), CH4 absent (agree), SO2 unconfirmed with PRISM-only data; T cool-mode too cool, reduced chi2 ~41, seed bimodality documented. Full table: `outputs/l3_wasp39b.md` |

## Known limitations

- Mock isothermal/constant-abundance forward model (no pRT on this box); no stellar
  contamination treatment; grey clouds only; Na/K/H2S out of V1 scope.
- Nested-sampling evidence below ln B ~ 30 carries sampler systematics; verdicts in
  the 3-15 range are reported as ambiguous, never detections.
- Posterior can be bimodal (cool/high-cloud vs hot/low-cloud); seeds 42/123 agree,
  seed 7 found the second mode at comparable evidence.
- ML classifier untrained here (`models/` empty); pipeline emits labelled placeholder scores.
- Long jobs must run in the foreground (background workers die); CPU is thermally throttled.

## Roadmap (from AGENTS.md)

- **V2**: native MAST JWST TSO ingestion, G395H/NIRISS/NIRCam modes, multi-observation
  fusion, forward-model emulator, stellar contamination, Bayesian model comparison.
- **V3**: multi-telescope, emission/phase curves, observation planning, AI assistant.
