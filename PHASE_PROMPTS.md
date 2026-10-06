# ExoSphere: Phase Prompts

## How to use
1. Create the repo folder. Put `AGENTS.md` in the root. Create an empty `PROGRESS.md` and `DECISIONS.md`.
2. For each phase, paste **one** prompt into a fresh agent session (Claude Code, Cursor, etc.).
3. Do not start the next phase until the current one's **Done when** checks pass.
4. If an agent drifts, reply: "Re-read AGENTS.md section 2 and 3, and stay in scope."

Every prompt starts with the same prefix, written once here to save tokens:

> **PREFIX:** Read `AGENTS.md` and `PROGRESS.md` first. Do only the phase below. Write tests. Run tests and lint. Update `PROGRESS.md`. Final reply max 10 lines.

---

## Phase 0: Scaffold and core schema
```
PREFIX. Phase 0: Scaffold.
Create the repo layout from AGENTS.md section 9, pyproject.toml (pinned deps, uv or pip), ruff/pytest config, .gitignore, README with setup steps.
Implement core/: Spectrum dataclass (section 4, with explicit units + validation of array lengths/NaNs), Provenance model (section 5), config loader (paths, data_cache, pRT opacity path via env var), seeded RNG helper.
Done when: pytest passes with tests for Spectrum validation and Provenance serialization.
```

## Phase 1: Data acquisition and ingestion
```
PREFIX. Phase 1: Data acquisition.
Implement data/exoarchive.py: fetch planet+host-star parameters via NASA Exoplanet Archive TAP, and list literature transmission spectra from the Atmospheric Spectroscopy table. Cache results.
Implement data/mast.py: query JWST observations for a target via astroquery.mast (metadata only, plus download of a chosen calibrated 1D product). No raw calibration.
Implement data/loaders/: converters from (a) Exoplanet Archive spectrum rows, (b) a local CSV/FITS x1d-style file, into Spectrum.
Fetch and store the WASP-39 b benchmark spectrum (NIRSpec PRISM) in data_cache with a checksum.
Tests mock network calls; one optional live test marked `network`.
Done when: L1 test passes: WASP-39 b loads into Spectrum and round-trips to the source values; units documented.
```

## Phase 2: Quality assessment and preprocessing
```
PREFIX. Phase 2: Quality + preprocessing.
quality/: compute S/N, coverage, missing fraction, outliers, resolution, uncertainty sanity. Per-molecule rating (H2O, CO2, CO, CH4, SO2) from coverage and S/N of each molecule's main bands in 0.6-5.3 um (put band definitions in a config file with sources noted). Output GOOD/LIMITED/POOR.
preprocess/: apply quality flags, sigma-clip outliers (log what was removed), optional rebin with correct uncertainty propagation, return new Spectrum with preprocessing version in provenance.
Done when: tests on synthetic spectra with injected outliers/gaps behave correctly; WASP-39 b gets a sensible report; no input mutated.
```

## Phase 3: Forward model (pRT)
```
PREFIX. Phase 3: Forward model.
First inspect the installed petitRADTRANS version and its docs; write the setup (opacity download/path) into README.
forward/: function params -> synthetic transmission spectrum, per AGENTS.md section 2 (isothermal T, constant mixing ratios of H2O/CO2/CO/CH4/SO2, H2/He background, grey cloud/cloud-top pressure, fixed gravity and stellar radius from catalog). Provide an instrument-resolution convolution/binning step to match a Spectrum's wavelength grid.
Done when: a test produces a spectrum showing CO2 feature near 4.3 um when CO2 is enabled; results are deterministic; a notebook/script plots a few example spectra; runtime per model call is recorded in PROGRESS.md.
```

## Phase 4: Bayesian retrieval and synthetic validation
```
PREFIX. Phase 4: Retrieval.
retrieval/: priors for the 8 free params (justify ranges in DECISIONS.md), Gaussian likelihood using Spectrum uncertainty, sampler interface with JAXNS primary and dynesty fallback (same API), posterior summary (median, 68%/95% credible intervals, corner plot data), per-molecule evidence summary (e.g. posterior mass above a detection-relevant abundance threshold; define in DECISIONS.md), save posterior samples + provenance.
Define L2 tolerances in DECISIONS.md, then write the L2 synthetic validation: several known atmospheres + noise levels -> retrieve -> check recovery. Report failures honestly (e.g. degeneracies like T vs. radius vs. cloud).
Done when: L2 suite runs (small settings for CI, full settings via flag) and results table is in PROGRESS.md.
```

## Phase 5: ML molecule classifier
```
PREFIX. Phase 5: ML.
Settle the "molecule present" label definition in DECISIONS.md.
ml/: synthetic dataset generator using forward/ (vary abundances, T, clouds, S/N, resolution; noise augmentation; seeded; stored with a version hash). 1D CNN multi-label (sigmoid) in PyTorch, train/val/test split by atmosphere configuration, metrics per molecule (AUC, precision/recall, calibration), inference function taking a Spectrum and returning candidate scores labelled as "ML candidate score".
Evaluate on the real WASP-39 b spectrum and report honestly, including failures and domain-gap caveats.
Done when: model + metrics saved with version; inference tested; PROGRESS.md notes domain-gap observations.
```

## Phase 6: Backend API, database, provenance
```
PREFIX. Phase 6: Backend.
api/: FastAPI + SQLAlchemy (SQLite dev, Postgres-ready). Tables: planets, observations, spectra, analyses, posteriors, ml_results, reports. Endpoints: search/fetch planet, load spectrum, run quality, run ML, start retrieval as background job with status polling, get results, list analysis history. Analysis IDs like EXO-000021. Every analysis stores full provenance (AGENTS.md section 5).
Done when: end-to-end API test runs WASP-39 b through quality -> ML -> retrieval (small settings) and the stored record is complete and reproducible from its seed.
```

## Phase 7: Dashboard
```
PREFIX. Phase 7: Dashboard (web/, React + TypeScript + Plotly).
Pages: target/observation picker, interactive spectrum viewer (data, error bars, best-fit model + credible band, molecule band overlays), data-quality panel, molecule panel (ML candidate score clearly separate from retrieval posterior and evidence), posterior/corner view, analysis history.
Implement the flagship link: click a wavelength/feature -> candidate molecule -> evidence -> posterior, using one shared state store (so the 3D phase can plug in).
Done when: the flow works against the real API on WASP-39 b; component tests pass; UI labels distinguish measured / ML / model-derived.
```

## Phase 8: 3D digital twin
```
PREFIX. Phase 8: 3D twin (web/, three.js or react-three-fiber).
Render from API data only: planet scaled by radius, translucent atmosphere shell from scale height (H = kT/(mu*m_H*g), formula shown in code comments), generic cloud/haze layer driven by cloud-top pressure, host star from catalog, orbit from catalog elements, day/night terminator, transit-position view.
Connect to the shared state: selecting a molecule highlights its modelled atmospheric contribution (not geography).
Add a persistent legend/disclaimer: modelled, not a photograph; no surface or spatial molecule claims (AGENTS.md section 3, rule 4).
Done when: renders for WASP-39 b and one other planet from the archive; selecting molecules updates the view; no forbidden features present.
```

## Phase 9: Report, L3 validation, hardening
```
PREFIX. Phase 9: Report + real validation.
report/: generate a scientific report (HTML/PDF) per analysis: target, data source, quality, preprocessing, ML scores (labelled), retrieval setup, posteriors/credible intervals, fit plots, limitations, full provenance.
Run L3: WASP-39 b NIRSpec PRISM vs published results. Compare detected features and parameter ranges, document agreements and discrepancies without forcing agreement.
Hardening: error handling, logging, caching, Dockerfile/compose, README quickstart, CI.
Done when: one command produces the WASP-39 b report; L1/L2/L3 results are summarized in PROGRESS.md.
```

---

## Handy mini-prompts

**Resume after a break / new agent:**
```
Read AGENTS.md and PROGRESS.md. Summarize in 5 lines where the project stands and what the next step is. Don't change code yet.
```

**Bug fix:**
```
Read AGENTS.md and PROGRESS.md. Bug: <describe + paste error>. Find the root cause, fix minimally, add a regression test, update PROGRESS.md. Max 10-line reply.
```

**Scientific review (use a fresh session):**
```
Read AGENTS.md. Review <module> for scientific correctness only: units, likelihood, priors, claims vs rules in section 3. List issues by severity. Don't edit code.
```

**Scope-creep check:**
```
Compare the current repo against AGENTS.md sections 2 and 3. List anything that violates the frozen scope or non-negotiable rules.
```
