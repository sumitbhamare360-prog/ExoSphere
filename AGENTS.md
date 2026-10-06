# AGENTS.md: ExoSphere project context

> Read this fully, then read `PROGRESS.md`. Do only the task you were given. Do not expand scope.

## 1. What this is
ExoSphere is an end-to-end exoplanet **atmospheric characterization** platform. It solves an inverse problem:
**observed JWST transmission spectrum -> molecular signatures -> physics-based Bayesian atmospheric retrieval -> posteriors/uncertainty -> dashboard + 3D "scientific digital twin".**

Central question: *Given the observed transmission spectrum, which atmospheric composition and parameters are most consistent with the data, and how strongly does the data support that?*

## 2. Frozen V1 decisions (do NOT change; propose changes in `DECISIONS.md` only)
| Item | Choice |
|---|---|
| Technique | Transit transmission spectroscopy only |
| Telescope / instrument | JWST / NIRSpec PRISM (~0.6-5.3 um) |
| Pilot/benchmark target | WASP-39 b (benchmark, not the only planet; code must be planet-independent) |
| Observation source | NASA/STScI MAST (astroquery.mast, exo.MAST) |
| Metadata + literature spectra | NASA Exoplanet Archive (TAP; Atmospheric Spectroscopy table) |
| V1 input | Calibrated/extracted 1D spectrum: wavelength, transit depth, uncertainty. NO raw detector calibration |
| Molecules | H2O, CO2, CO, CH4, SO2. Bulk background H2/He |
| Excluded in V1 | Na, K, H2S, O3, O2, NH3, hydrocarbons, emission, phase curves, direct imaging, reflected light, multi-telescope, life detection, surface maps |
| Forward model | petitRADTRANS (pRT) |
| Atmosphere | Isothermal T; constant-with-altitude mixing ratios; grey cloud / cloud-top pressure; mass/gravity and stellar radius fixed from catalog |
| Free params | T, log(H2O), log(CO2), log(CO), log(CH4), log(SO2), reference radius, cloud-top pressure |
| Retrieval | Bayesian nested sampling. Primary: pRT + JAXNS. Fallback: pRT + dynesty |
| ML (V1) | 1D CNN multi-label classifier (sigmoid) giving molecule *candidate scores*, trained on synthetic spectra from the forward model |
| ML authority | Supporting only. Physics retrieval is the authoritative result |
| 3D | Digital twin from measured/inferred params. No surfaces, continents, oceans, spatial molecule maps, or "photos" |

## 3. Non-negotiable rules
1. ML scores are NOT abundances and NOT detections. Always label them "ML candidate score".
2. Always report posterior + credible interval, never a bare point value.
3. Never show a molecule result when the data-quality module rates it UNSUITABLE; show LIMITED/GOOD status instead.
4. 3D/UI must label modelled vs. measured vs. assumed. No fabricated geography.
5. Every analysis has full provenance (see section 5). No result without it.
6. Scientific code must be deterministic given a seed. Seeds are stored in provenance.
7. An emulator or ML model may enter inference only after validation against the physics model (V2).
8. Validation is mandatory (see section 7). A UI that renders a result is not "done".
9. Do not invent data, API fields, or library functions. If unsure, check the installed package docs or source, or ask.

## 4. Internal data contract
```
Spectrum:
  wavelength[]            (um)
  transmission[]          (transit depth, fractional; state units explicitly)
  uncertainty[]           (same units as transmission)
  wavelength_bin_edges[]
  quality_flags[]
  observation_id, target_id, instrument
  provenance
```
All downstream modules consume only `Spectrum`. They never care whether it came from MAST, the Exoplanet Archive, or a benchmark file.

## 5. Provenance record (one per analysis)
`analysis_id` (e.g. EXO-000021), planet, observation_id, telescope, instrument, source archive, input data version/hash, preprocessing version, ML model version, retrieval model version + parameters (priors, sampler, n_live, seed), timestamp, result reference.

## 6. Data quality module output
Before retrieval, evaluate S/N, wavelength coverage, missing data, outliers, uncertainty, resolution. Return overall suitability (GOOD/LIMITED/POOR) plus a per-molecule rating (e.g. H2O GOOD, CH4 LIMITED), based on whether that molecule's main absorption bands are covered with adequate S/N.

## 7. Validation levels
- **L1 Data:** ingest a source spectrum and reproduce it (round-trip match).
- **L2 Synthetic:** known atmosphere -> synthetic spectrum + noise -> retrieval -> recovered params within predefined tolerances (define tolerances in `DECISIONS.md`).
- **L3 Real:** run on WASP-39 b NIRSpec PRISM; compare features and model against published NASA/literature results.

## 8. Proposed stack (defaults; changeable via DECISIONS.md)
Python 3.11; numpy/scipy/astropy/astroquery/pandas; petitRADTRANS, JAXNS (fallback dynesty); PyTorch for CNN; FastAPI; PostgreSQL (SQLite for dev) via SQLAlchemy; React + TypeScript + Plotly (spectrum) + three.js/react-three-fiber (3D); pytest; ruff.

## 9. Repo layout
```
exosphere/
  AGENTS.md  PROGRESS.md  DECISIONS.md
  src/exosphere/
    core/        spectrum schema, config, provenance
    data/        mast.py, exoarchive.py, loaders/ (benchmark files)
    quality/     suitability + per-molecule rating
    preprocess/  clean, mask, rebin, normalize
    forward/     pRT model wrapper, parameterization
    retrieval/   priors, likelihood, samplers, posterior summary
    ml/          synthetic dataset gen, CNN, train, infer
    api/         FastAPI app, DB models
    report/      scientific report generator
  web/           React dashboard + 3D twin
  tests/         unit + validation (L1/L2/L3)
  data_cache/    gitignored
```

## 10. Known risks to check, not assume
- pRT API changed between major versions. Inspect the installed version before writing code.
- pRT opacity files are large downloads and need an env var path. Document setup in README.
- JAXNS/JAX versions can conflict with other deps. Pin versions; keep dynesty fallback working.
- "Molecule present" label for ML training needs a definition (e.g. abundance above a threshold AND feature amplitude above noise). Settle it in DECISIONS.md in the ML phase.
- Units and transit-depth conventions (ppm vs fraction, (Rp/Rs)^2) must be explicit everywhere.

## 11. Working protocol for agents
1. Read this file and `PROGRESS.md`. Do not re-read the whole repo; open only the files you need.
2. Implement only the assigned phase. Write tests with it.
3. Run tests and lint. Fix failures before finishing.
4. Update `PROGRESS.md`: what was done, files touched, how to run it, open issues, next step.
5. Log any scientific/design deviation in `DECISIONS.md`.
6. Final reply: at most 10 lines. Summarize and list blockers.

## 12. Later versions (do not build now)
V2: native MAST JWST TSO ingestion, G395H/NIRISS/NIRCam, multi-observation fusion, forward-model emulator, stellar contamination, alternative model comparison with Bayesian evidence.
V3: multi-telescope, emission/phase curves, observation planning, AI assistant.
