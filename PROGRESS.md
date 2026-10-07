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

~~**Phase 2** (quality module)~~ -> done, see below.

## Phase 2 — quality assessment + preprocessing (done)

### What was built

- `config/molecule_bands.yaml` — band windows (um) for H2O/CO2/CO/CH4/SO2 over
  0.6-5.3 um + every quality threshold (snapshot embedded in each report;
  rationale in DECISIONS 11-12).
- `src/exosphere/quality/assess.py` — `assess(spectrum) -> QualityReport`
  (pydantic, JSON-serializable): median point S/N vs local continuum
  (running median, 21 pts), per-band S/N (90th percentile of point S/N),
  wavelength coverage (gap-safe point-spacing widths), flagged/NaN fractions,
  MAD-based outlier count, uncertainty sanity (non-positive/NaN/huge/tiny),
  effective R from bin widths, overall GOOD/LIMITED/POOR (= worst metric) and
  per-molecule GOOD/LIMITED/POOR from coverage + in-band S/N. Also shared
  stats: `running_median`, `mad_outlier_mask`, `is_flagged`.
- `src/exosphere/preprocess/clean.py` — `clean(spectrum, options) -> (Spectrum,
  PreprocessLog)`: drop NaN / flagged / bad-uncertainty points, robust MAD
  spike clip (5 sigma default, configurable), optional inverse-variance rebin
  (`count` or `resolution` grid; d = Σwd/Σw, σ = 1/√Σw; empty bins dropped +
  logged). Log records every removed point with ORIGINAL index + reason,
  parameters, and version `preprocess-1.0.0` (also written into the output
  spectrum's provenance). Input is never mutated.
- `scripts/quality_wasp39b.py` — assesses + cleans the cached benchmark
  spectrum, prints both reports + the preprocess log, writes
  `outputs/quality_wasp39b.png` (spectrum with band windows, removed points,
  point-S/N panel; gitignored).

### WASP-39 b quality report summary (benchmark)

Carter et al. 2024 PRISM spectrum (`40/24/96/78/...5502_6.tbl.npz`, 147 pts,
0.5213-5.3441 um), raw **and** cleaned (clean removed nothing: 147 -> 147,
reasons `{}`):

| Metric | Value |
|---|---|
| Overall suitability | **GOOD** |
| Wavelength coverage | 0.991 of 0.6-5.3 um |
| Median point S/N | 1.21 |
| Best band S/N | 10.12 (CH4 windows) |
| Effective resolving power | R ≈ 94.5 |
| Flagged / NaN / outliers | 0 / 0 / 0 |
| Uncertainty sanity | median σ 7.5e-5 frac depth; 0 non-positive, 0 huge, 0 tiny |

| Molecule | Rating | Band coverage | Band S/N |
|---|---|---|---|
| H2O | GOOD | 0.988 | 9.92 |
| CO2 | GOOD | 0.994 | 6.00 |
| CO | POOR | 0.996 | 1.63 |
| CH4 | GOOD* | 0.994 | 10.12 |
| SO2 | LIMITED | 0.996 | 2.54 |

Rustamkulov et al. 2023 PRISM (207 pts): overall **GOOD**, H2O GOOD (5.7),
CO2 LIMITED (3.3), CO POOR (1.4), CH4 GOOD (7.3), SO2 LIMITED (2.3).

\* ratings measure data capability in the molecule's windows, not molecule
presence (DECISIONS 14) — CH4 is *not* detected in WASP-39 b; only the
retrieval decides presence. CO POOR / SO2 LIMITED match the weak per-point
structure of those bands in PRISM native-resolution data.

### Tests (run: `pytest` / `pytest -m network` / `ruff check .`)

- `tests/test_quality.py` (12) — config loading, S/N/coverage/outlier/
  uncertainty metrics, JSON round-trip, **phase requirement**: spectrum without
  4.2-4.4 um -> CO2/CO/SO2 POOR-or-LIMITED while H2O stays GOOD; gap, flagged
  points, single point, featureless spectrum.
- `tests/test_preprocess.py` (11) — removal reasons + original indices,
  input-not-mutated, provenance version, analytic rebin check (Σwd/Σw and
  1/√Σw to 1e-12), rebin edge/empty-bin handling, bad uncertainties,
  all-removed ValueError, JSON log.
- `conftest.py` added `make_synthetic_spectrum` / `make_feature_spectrum`.
- Status: **83 passed + 2 network tests pass live; `ruff check .` clean.**

### Files touched (Phase 2)

`config/molecule_bands.yaml` · `src/exosphere/quality/assess.py` ·
`src/exosphere/preprocess/clean.py` · `scripts/quality_wasp39b.py` ·
`tests/{conftest,test_quality,test_preprocess}.py` · `pyproject.toml`
(pyyaml, matplotlib) · `.gitignore` (outputs/) · `DECISIONS.md` · `README.md`

### Open issues

- Band S/N is a capability metric; overlapping absorbers (H2O wings) lift the
  CH4-window score (DECISIONS 14).
- Coverage widths use min-neighbour spacing: ~1% undercount where native
  sampling coarsens (conservative, DECISIONS 16).
- Featureless but very precise spectra rate LIMITED/POOR on the S/N metric by
  design (no structure => no molecular claim possible).
- Asymmetric archive errors still collapsed to a single sigma (Phase 1).

### Next step

**Phase 3** (see `PHASE_PROMPTS.md`): forward model with petitRADTRANS
(needs `PRT_INPUT_DATA_PATH` opacity data; pRT not installed yet).
