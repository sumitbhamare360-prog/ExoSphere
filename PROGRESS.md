# PROGRESS.md — ExoSphere

## Phase 0 — project scaffold + core schema (done, commit `76723aa`)

- Repo scaffold: `pyproject.toml` (pinned deps, pytest + ruff config), `AGENTS.md`,
  `.gitignore` (`data_cache/`, `.venv/`), package layout per AGENTS.md §9.
- `src/exosphere/core/`:
  - `spectrum.py` — pydantic `Spectrum` (wavelength, transmission, uncertainty,
    wavelength_bin_edges, quality_flags, observation_id, target_id, instrument,
    provenance); units fixed: `um`, fractional transit depth `(Rp/Rs)^2`;
    validates lengths, strict wavelength ordering, positive uncertainties;
    JSON + `.npz` round-trip (`allow_pickle=False`).
  - `provenance.py` — `Provenance` record (`analysis_id` = `EXO-\d{6}`),
    `generate_analysis_id`, `AnalysisIdGenerator`.
  - `config.py` — `load_config()`, data-cache path, `PRT_INPUT_DATA_PATH` (optional).
  - `rng.py` — seeded RNG helpers (`make_seed`, `make_rng`, `SeededRNG`).
- Tests: 31 passing.

## Phase 1 — data acquisition + ingestion (done)

### What was built

- `src/exosphere/data/loaders/` (all consume the `Spectrum` contract, AGENTS.md §4):
  - `units.py` — `wavelength_to_um`, `depth_to_fraction` (fraction/ppm/% handled
    explicitly; aliases + scale factors).
  - `binning.py` — `bin_edges_from_centers` (N+1 edges from centers/bandwidth,
    see DECISIONS.md item 5).
  - `ipac.py` — archive `.tbl` (IPAC format) -> `Spectrum` via
    `astropy.io.ascii.read(format="ipac")`, incl. `%` -> fraction and
    radius-ratio fallback (DECISIONS 3-4).
  - `csv_loader.py`, `fits_loader.py` — CSV and x1d-style FITS (depth column
    required; flux-only x1d rejected, DECISIONS 7).
- `src/exosphere/data/exoarchive.py` — NASA Exoplanet Archive:
  TAP sync (CSV), `get_planet_params` -> `PlanetParams` (surface gravity derived
  from mass+radius), `list_literature_spectra` -> `LiteratureSpectrum`,
  spectrum file download (Firefly URL, cached + SHA256, DECISIONS 6).
- `src/exosphere/data/mast.py` — astroquery MAST: JWST observation search
  (fields from astroquery metadata), calibrated 1D (`SCIENCE` + `X1D/X1DINTS`)
  product download, manifest handling. Column names verified at runtime, not
  invented (DECISIONS 2).
- `scripts/fetch_wasp39b.py` — benchmark fetch: all PRISM transmission spectra,
  `.tbl` + `.npz` + SHA256 + JSON manifest; exits 1 if none found.

### Tests (run: `pytest` / `pytest -m network` / `ruff check .`)

- `tests/conftest.py` — realistic IPAC fixture generator (`make_ipac_tbl`,
  `PRISM_SOURCE_ROWS`, `PRISM_RATIO_ROWS`), fixtures `provenance`, `ipac_tbl_path`.
- `tests/test_loaders.py` — IPAC/CSV/FITS parsing, units, bin edges, ratio fallback.
- `tests/test_exoarchive.py` — TAP/planets/spectra parsing with mocked HTTP.
- `tests/test_mast.py` — observation/product selection with mocked astroquery.
- `tests/test_l1_data_validation.py` — **L1**: parse -> `Spectrum`, saved ->
  reloaded, round-trip match.
- `tests/test_network_live.py` — `@pytest.mark.network` live tests
  (archive + MAST), deselected by default.
- Current status: **59 passed, 2 network tests pass live; `ruff check .` clean.**

### WASP-39 b spectra found in the NASA Exoplanet Archive (31 total)

JWST transmission spectra (14):

| Instrument | range (um) | pts | Reference | Bibcode |
|---|---|---|---|---|
| NIRSpec PRISM (native, dilution corrected) | 0.5213-5.3441 | 147 | Carter et al. 2024 | 2024NatAs...8.1008C |
| NIRSpec PRISM | 0.5315-5.3441 | 207 | Rustamkulov et al. 2023 | 2023Natur.614..659R |
| NIRISS SOSS (Feinstein) | 0.631-2.7965 | 331 | Feinstein et al. 2023 | 2023Natur.614..670F |
| NIRISS (Carter, optical/NIR) | 0.633-0.8484 | 27 | Carter et al. 2024 | 2024NatAs...8.1008C |
| NIRISS (Carter) | 0.8771-2.8048 | 108 | Carter et al. 2024 | 2024NatAs...8.1008C |
| NIRCam | 2.4275-4.0175 | 107 | Ahrer et al. 2023 | 2023Natur.614..653A |
| NIRCam | 2.4328-3.9883 | 49 | Carter et al. 2024 | 2024NatAs...8.1008C |
| NIRSpec | 2.7388-3.6865 | 30 | Carter et al. 2024 | 2024NatAs...8.1008C |
| NIRSpec G395H | 2.7629-5.1691 | 344 | Alderson et al. 2023 | 2023Natur.614..664A |
| NIRSpec (ERS pipelines: Tiberius) | 3.0007-5.3405 | 243 | ERS Team 2022 | 2022arXiv220811692T |
| NIRSpec (ERS: Eureka!) | 3.0085-5.288 | 94 | ERS Team 2022 | 2022arXiv220811692T |
| NIRSpec (ERS: FIREFLy) | 3.0139-5.5476 | 95 | ERS Team 2022 | 2022arXiv220811692T |
| NIRSpec (ERS: tshirt) | 3.017-5.465 | 67 | ERS Team 2022 | 2022arXiv220811692T |
| NIRSpec | 3.8486-5.1279 | 29 | Carter et al. 2024 | 2024NatAs...8.1008C |

Other transmission (11): HST STIS 0.33-0.965 um (Fischer 2016, Sing 2016),
HST WFC3 (Wakeford 2018, Panek 2023), ground-based FORS2/OSIRIS/Mexman/Marconi,
IRAC 3.6/4.5 um (Sing 2016, Fischer 2016, Baxter 2021), MIRI-LRS 5.125-11.875 um
(Powell 2024, x3). Eclipse (3): IRAC (Kammer 2015, Deming 2023 x2).

**Benchmark result:** `python scripts/fetch_wasp39b.py` fetched both PRISM
transmission spectra (147 and 207 points, 0.52-5.34 um) -> SHA256 checksums
(`3359e371…65e3`, `135d6f22…63e7`), `.npz` copies and
`data_cache/benchmark/wasp39b_prism_manifest.json`.

### Files touched (Phase 1)

`pyproject.toml` (astroquery, requests, network marker) ·
`src/exosphere/data/exoarchive.py` · `src/exosphere/data/mast.py` ·
`src/exosphere/data/loaders/{__init__,units,binning,ipac,csv_loader,fits_loader}.py` ·
`scripts/fetch_wasp39b.py` · `tests/{conftest,test_loaders,test_exoarchive,test_mast,test_l1_data_validation,test_network_live}.py` ·
`DECISIONS.md` · `README.md`

### Open issues

- Asymmetric archive errors collapsed to a single sigma (DECISIONS 3).
- Archive spectra come with `BANDWIDTH`-derived bin edges only (DECISIONS 5).
- pRT, JAXNS, torch not installed yet (later phases).
- `surface_gravity_m_s2` derived, not an archive value (DECISIONS 8).

### Next step

~~**Phase 6** (see `PHASE_PROMPTS.md`)~~ -> done, see below.

## Phase 6 --- Backend API, database, provenance (done)

### What was built

- `src/exosphere/api/config.py` --- settings from environment variables (database URL, CORS, job limits, retrieval defaults).
- `src/exosphere/api/db.py` --- SQLAlchemy 2.0 models with async support:
  - `Planet`, `Observation`, `SpectrumFile` (catalog + data management)
  - `Analysis` (job tracking: status, stage, progress, config, provenance)
  - `Posterior`, `MLResult`, `QualityReport`, `DetectionResult`, `Report` (results)
  - SQLite default, PostgreSQL via `DATABASE_URL` env var; alembic migrations.
- `src/exosphere/api/schemas.py` --- Pydantic request/response models with explicit units and AGENTS.md §3 labels (e.g., `candidate_score` not `abundance`).
- `src/exosphere/pipeline.py` --- `Pipeline` class orchestrating: load planet params -> load spectrum -> quality -> preprocess -> ML -> retrieval (dynesty) -> detection -> finalize; progress callback; per-stage DB status updates; provenance written.
- `src/exosphere/api/app.py` --- FastAPI app with endpoints:
  - `GET /health` --- health check
  - `GET /planets/search?q=` --- Exoplanet Archive lookup (cached)
  - `GET /planets/{name}` --- params + available MAST observations + literature spectra
  - `POST /analyses` --- create analysis job, returns `analysis_id`, starts background job
  - `GET /analyses` / `GET /analyses/{id}` / `GET /analyses/{id}/status` --- history & status
  - `GET /analyses/{id}/spectrum` --- observed + cleaned spectrum
  - `GET /analyses/{id}/quality` --- quality report
  - `GET /analyses/{id}/ml` --- ML candidate scores (labelled "ML candidate score")
  - `GET /analyses/{id}/retrieval` --- logZ, best fit, medians, 68%/95% CI
  - `GET /analyses/{id}/posterior` --- samples for corner plot
  - `GET /analyses/{id}/detection` --- ln Bayes factors, upper limits
  - `GET /analyses/{id}/model` --- best-fit spectrum with 68% credible band
  - `GET /analyses/{id}/provenance` --- full provenance
  - `DELETE /analyses/{id}` --- delete analysis + cascade
- `scripts/run_server.sh` --- start script with DB init
- Background jobs: `ProcessPoolExecutor` (1-2 workers configurable), DB-backed status survives page refresh; running jobs marked `interrupted` on restart.
- Retrieval concurrency limited to 1-2 jobs (configurable via `max_concurrent_jobs`).
- Reproducibility: same planet + spectrum + options + seed -> same stored result. `validate_provenance()` checks completeness.

### Tests

- Unit tests: **115 passed**, ruff clean.
- API tests with TestClient: all endpoints, status transitions, failure handling, interrupted-job recovery.
- E2E test (`@pytest.mark.slow`): WASP-39 b with minimal retrieval settings.

### Real-data smoke test

- `POST /analyses` with WASP-39 b Carter 2024 PRISM spectrum -> analysis created, background job runs dynesty retrieval -> `GET /analyses/{id}/retrieval` returns logZ, posteriors.

### Files touched (Phase 6)

`src/exosphere/api/{config.py,db.py,schemas.py,app.py,__init__.py}`, `src/exosphere/pipeline.py`, `src/exosphere/api/db.py`, `scripts/run_server.sh`, `pyproject.toml` (fastapi, uvicorn, sqlalchemy, alembic, pydantic-settings, httpx, python-multipart, python-dotenv, aiofiles), `DECISIONS.md`, `PROGRESS.md`, `README.md`.

### Open issues

- dynesty slow at production n_live (500-1000); L2 full suite takes >30 min. CI uses low n_live.
- JAXNS incompatible with numpy forward model; needs JAX-traceable forward model (future work).
- Error-inflation parameter not yet tested with free-fit mode.
- Nested-model comparison multiplies runtime (~8x for 5 molecules); optional batched mode available.
- Real pRT still unavailable on Windows/Python 3.13 (mock model used).
- Background job persistence: in-memory `running_jobs` dict; production needs Redis.
- No authentication/authorization yet.

### Next step

**Phase 7** (see `PHASE_PROMPTS.md`): Report generation, web dashboard (React + TypeScript + Plotly + three.js).