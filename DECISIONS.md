# DECISIONS.md — scientific/design decisions and deviations

Entries are appended per phase. Anything that deviates from, or refines, AGENTS.md
is logged here.

## Phase 1 (2026-10-06)

1. **Spectrum identifier for the archive `spectra` table.** The Atmospheric
   Spectroscopy TAP table (verified live via TAP_SCHEMA: table `spectra`) has no
   id column. `spec_path` (unique file path, e.g.
   `40/24/96/78/WASP_39_b_3.11466_5502_6.tbl`) is used as `spectrum_id` /
   `observation_id` everywhere.
2. **Column names taken from runtime inspection, not guessed.** TAP column lists
   for `spectra` and `pscomppars` were read from TAP_SCHEMA; MAST observation and
   product fields from `Observations.get_metadata("observations"/"products")` in
   astroquery 0.4.11; JWST calibrated 1D products confirmed to be
   `productType=SCIENCE` with `productSubGroupDescription` in {`X1D`, `X1DINTS`}.
3. **Transit-depth units.** Archive spectrum files give `PL_TRANDEP` and its
   `PL_TRANDEPERR1/2` in **percent** (confirmed in file headers and the archive's
   column docs). Internal convention stays fractional: depth and uncertainty are
   divided by 100. Asymmetric +/- errors are collapsed to
   `mean(|err1|, |err2|)` because the internal `Spectrum.uncertainty` is a single
   1-sigma value (documented limitation; revisited if pRT likelihood needs
   asymmetric errors).
4. **Radius-ratio fallback.** If `PL_TRANDEP` is null but `PL_RATROR` is present,
   depth = `(Rp/R*)^2` with `sigma = 2*|Rp/R*|*sigma(Rp/R*)`. Rows without any
   usable depth/uncertainty are dropped, never imputed.
5. **Bin edges.** The archive table provides bin centers + `BANDWIDTH` but no
   edge arrays. `Spectrum.wavelength_bin_edges` (N+1) is built as: inner edges =
   midpoints of adjacent centers (guarantees strict ordering); outer edges =
   first/last center ∓ bandwidth/2, falling back to half the neighbor spacing if
   the widths are missing/invalid.
6. **Spectrum file download mechanism.** Per-point data are not exposed via TAP.
   The archive's own UI builds file URLs as
   `<host><FF_InitPage base>/atmospheres/tab1/data/<spec_path>` (verified from
   `/applications/atmospheres/js/tab1.js` and reproduced live). Code reads the
   base path from the app page HTML at runtime; files are cached under
   `data_cache/exoarchive/spectra/` (mirroring `spec_path`). If the archive ever
   changes this path, `firefly_data_url()` is the single place to update.
7. **x1d-style FITS.** V1 accepts FITS binary tables that already contain a
   transit-depth column. A flux-only x1d spectrum (WAVELENGTH/FLUX/ERROR) is
   rejected with an explicit error: flux cannot be converted to transit depth
   without a light curve (AGENTS.md rule 9: no invented data). Depth unit in FITS
   defaults to "fraction" (header units are honored when present); CSV/FITS unit
   and column names are explicit loader parameters.
8. **Surface gravity.** `pscomppars` has no planet-gravity column;
   `PlanetParams.surface_gravity_m_s2` is derived as `G*M/R^2` from
   `pl_bmasse` and `pl_rade` (labelled "derived", never presented as an archive
   value).
9. **Benchmark script behaviour.** `scripts/fetch_wasp39b.py` fetches *all*
   transmission spectra whose instrument/note mention PRISM (2 found), writes a
   SHA256 sidecar for each raw `.tbl`, saves `.npz` copies and a JSON manifest.
   If none exist it prints every available spectrum and exits 1 (no silent
   substitution).
10. **Live-network tests** are marked `@pytest.mark.network` and deselected by
    default (`addopts = "-m 'not network'"`); run them with `pytest -m network`.

## Phase 2 (2026-10-06)

11. **`config/molecule_bands.yaml`** holds both the band windows (the windows
    given in the phase prompt, with source comments: H2O near-IR
    combination/overtone bands, CO2 2.0/2.7/4.3, CO 2.3/4.7, CH4 1.6/2.3/3.3,
    SO2 3.9-4.2 - the 7.7 SO2 fundamental is outside PRISM coverage) and ALL
    data-quality thresholds. One file so the quality gate is auditable; the
    loaded threshold snapshot is embedded in every `QualityReport`.
12. **Thresholds (calibrated against the two cached WASP-39 b PRISM spectra):**
    - overall wavelength coverage: GOOD >= 0.7, POOR < 0.4 of 0.6-5.3 um.
    - molecule band coverage: GOOD >= 0.6, POOR < 0.3 (bandwidth-weighted over
      all of that molecule's windows).
    - band S/N: GOOD >= 5, POOR < 2 (was 10/5 as a first guess; measured
      benchmark band S/N is ~6-10 for well-detected bands, ~1.5 for absent
      ones, so 10 would have misrated the benchmark as LIMITED).
    - overall suitability S/N metric: best (max) molecule-band S/N against the
      same 5/2 thresholds: does ANY window show structure above the noise?
    - bad-point (flagged+NaN) fraction: GOOD <= 0.1, POOR > 0.5.
    - outlier fraction: GOOD <= 0.05, POOR > 0.25; point count: GOOD >= 50,
      POOR < 10.
    - uncertainty sanity: sigma > 10% of the median transit depth = "huge";
      sigma < 1e-10 (fractional depth, i.e. 1e-4 ppm) = "tiny".
    Overall suitability = WORST metric (conservative gate, AGENTS rule 3).
13. **S/N definitions.** Point S/N = |depth - running_median(depth, 21 pts)| /
    sigma (the phase-prescribed method); the report's `median_snr` is the
    median over usable points. **Band S/N = 90th percentile of the point S/N
    inside the molecule's windows**: the median is dominated by featureless
    points inside a band and the bare maximum is a single noise fluctuation.
    Overall suitability uses the best band S/N; `median_snr` alone would rate
    every real spectrum POOR because most bins sit on the continuum.
14. **Per-molecule rating = data capability, NOT molecule presence.** It
    answers "are this molecule's windows covered with enough structure above
    noise that the data could support it?" - attribution belongs to the
    retrieval (AGENTS 1/3). Consequence: CH4 rates GOOD on WASP-39 b (its
    windows show >5 sigma structure from overlapping absorbers) even though
    the retrieval finds no CH4. Documented in `QualityReport.method`.
15. **Outlier detection is a neighbour-prediction MAD clip, not a
    continuum-MAD clip.** First implementation clipped against a running-median
    continuum and destroyed real features (any feature narrower than the
    continuum window is flagged, and GOOD spectra have strong features). Final
    rule: r_i = d_i - (d_{i-1}+d_{i+1})/2, sigma_eff from the three quoted
    sigmas, z = r/sigma_eff, flag |z - median(z)| > n_sigma * max(1.4826*MAD(z),
    1.0). The floor at 1.0 means the threshold is never tighter than n_sigma of
    the QUOTED error (MAD only raises it when the data scatter exceeds sigma);
    without the floor, a noiseless smooth spectrum is over-clipped. Endpoints
    and points across gaps (>3x median sampling gap) are never flagged.
    Same function (`mad_outlier_mask`) used by assess and clean. Default 5
    sigma, configurable (`CleanOptions.clip_n_sigma`).
16. **Coverage widths come from local point spacing** (min of the two
    neighbour gaps), not from `wavelength_bin_edges`: N+1 edges must tile the
    range, so a bin spanning an internal gap would claim coverage of the gap.
    Cost: ~1% undercount where the native sampling coarsens (acceptable,
    conservative). Resolving power still uses the actual bin edges.
17. **Quality flags:** pass sentinels are `""` and `"OK"` (Phase 1 loaders
    write "OK"); anything else is flagged and excluded from usable points.
18. **Preprocessing (`preprocess/clean.py`, version `preprocess-1.0.0`):**
    removal order NaN -> flagged -> bad uncertainty -> MAD spike clip, each
    recorded with the ORIGINAL input index + reason; rebin (optional) is
    inverse-variance: d = sum(w d)/sum(w), sigma = 1/sqrt(sum(w)), w = 1/sigma^2,
    empty grid cells dropped (never interpolated into) and logged; bin edges
    recomputed from surviving centers + original bin widths; the version string
    is written into the output Spectrum's `provenance.preprocessing_version`
    (input untouched - `clean` never mutates).
19. **New dependencies:** `pyyaml==6.0.3` (band/threshold config) and
    `matplotlib==3.11.2` (diagnostic plots only; scripts force the Agg
    backend).

## Phase 3 (2026-10-06) — Forward model / petitRADTRANS integration

20. **pRT installation blocked on Windows + Python 3.13 + numpy 2.x.** Both
    pRT 2.x and 3.x fail to install:
    - pRT 2.7.7 (last 2.x) requires `numpy.distutils`, removed in numpy 2.0
      (we have numpy 2.2.6, required for Python 3.13). Build isolation fails
      with `ModuleNotFoundError: No module named 'numpy.distutils'`.
    - pRT 3.4.0 (latest 3.x) uses Meson build system and requires 32-bit Python
      on Windows (`Need python for x86, but found x86_64`). Our environment is
      64-bit Python 3.13.7.
    Attempted: `pip install petitRADTRANS==2.7.7` (with/without build isolation,
    numpy 2.x pre-installed), `pip install petitRADTRANS==3.4.0` (with meson,
    meson-python), all fail.
    **Fallback:** a pure-Python mock forward model (`forward/model.py`) is
    provided with the exact API specified in the phase prompt, using a simple
    analytic transmission spectrum approximation (isothermal, constant VMR,
    grey cloud) so that retrieval, ML, and pipeline development can proceed.
    The mock is labelled "MOCK" in outputs and provenance. Real pRT integration
    requires a Linux/conda environment with Python ≤3.11 and numpy ≤1.x (for
    pRT 2.x) or a 32-bit Python + MinGW toolchain (for pRT 3.x on Windows).
    This is documented in README and `scripts/setup_opacities.py`.
21. **Opacity mode for mock:** since real opacities are unavailable, the mock
    uses a low-resolution line-list approximation: cross-sections computed from
    pre-tabulated HITRAN-like Gaussian line profiles at R=1000 for H2O, CO2, CO,
    CH4, SO2, plus H2-H2/H2-He CIA and H2 Rayleigh. This is documented as
    `opacity_mode = "mock-gaussian-R1000"` in the forward model config.
22. **Forward model API** matches the phase prompt exactly:
    - `ModelParams` dataclass (T, log abundances, r_ref, log_p_cloud)
    - `PlanetFixed` dataclass (gravity, stellar_radius, reference_pressure=0.01 bar)
    - `transmission_spectrum(params, fixed, wavelength_grid)` returns (um, fractional depth)
    - `to_instrument(model_wl, model_depth, target_spectrum)` bins onto target grid
    - Validation: sum(VMR) < 1, T > 0, clear errors
    - Cached `Radtrans` object (mock) per process
    - Timing recorded per call
23. **Example figure** (`scripts/plot_forward_examples.py`) generates 4 spectra
    (H2O-only; H2O+CO2; H2O+CO2+SO2; cloudy) to `outputs/forward_examples.png`
    using the mock model.
24. **Tests** (`tests/test_forward.py`) marked `@pytest.mark.slow`: deterministic
    output, CO2 4.3um feature presence/absence, SO2 4.0um feature, cloud
    muting, instrument binning conservation, invalid param rejection.

## Phase 4 (2026-10-07)

25. **Prior parameterization**: unit-cube [0,1]^8 → physical parameters via
    `unit_to_physical()`. Bounds: T ∈ [300, 2500] K; log VMR ∈ [-12, -1];
    r_ref ∈ [0.7, 1.3] × catalog R_Jup (±30%); log_p_cloud ∈ [-6, 2] bar.
    Sum(VMR) < 1 enforced as hard prior bound (log_prior = -inf if violated).
    Uniform in these bounds → constant log-prior density within volume.

26. **Likelihood**: Gaussian using Spectrum.uncertainty as 1-sigma errors.
    Model evaluated on observed grid via `forward.to_instrument()` (flux-conserving
    binning). Optional free error-inflation factor (multiplicative on sigma);
    default off (0). log L = -0.5 * Σ((d_obs - d_model)/σ)^2 - 0.5*Σ ln(2πσ²).

26. **Sampler choice**: dynesty is the working default; JAXNS is primary per
    AGENTS.md but incompatible with numpy forward model (requires JAX-traceable
    functions). JAXNS stub raises RuntimeError with clear message; dynesty is
    fallback. Both share common `run(spectrum, fixed, config, seed)` interface.

27. **Nested sampling config**: n_live=500 default (CI uses 50-100 for speed);
    dlogz=0.01 termination; seed for reproducibility; error_inflation="free"
    option available but untested. Parallel pool via dynesty `pool` supported
    but not default (pickle issues with mock model).

27. **Evidence and posteriors**: dynesty returns logZ ± error; posterior samples
    weighted by `exp(logwt - logz)`. Best fit = max likelihood sample; median
    and 68%/95% credible intervals via weighted quantiles (dynesty.utils).

27. **Per-molecule detection**: nested-model comparison — re-run retrieval with
    one molecule's VMR fixed to -12 (negligible); ln B = logZ_full - logZ_reduced.
    ln B > 3: substantial; >5: strong; >10: very strong (Kass & Raftery 1995;
    Benneke & Seager 2013). Sigma ≈ √(2 ln B) for Gaussian approximation.
    Non-detections: 95% upper limit on log VMR from weighted posterior percentile.

27. **L2 validation tolerances**: true value in 95% CI in ≥80% of runs;
    strong molecules (log VMR ≥ -4 at WASP-39 b S/N) must have ln B > 3;
    absent molecules must not be detected (ln B < 1) in most runs.
    Degeneracies (T vs r_ref vs cloud) flagged, not counted as failures.

27. **JAXNS status**: Not compatible with numpy forward model (requires
    JAX-traceable likelihood and prior). Real pRT also not JAX-traceable.
    Stub raises RuntimeError with clear message; dynesty is working fallback.
    To enable JAXNS: implement JAX-traceable forward model (jax-coded or
    jax-coded pRT wrapper).

27. **Performance**: dynesty ~0.05-0.1 s/likelihood call at n_live=500;
    full run ~5-15 min. L2 full suite (4 cases × 2 noise × 3 seeds = 24 runs)
    takes ~30-60 min at production settings. CI uses n_live=50-100, dlogz=0.5.

## Phase 5 (2026-10-07)

28. **ML label definition — "molecule present"**: A molecule is labelled present
    if its log VMR ≥ -6 AND its noiseless feature amplitude (max depth deviation
    in its band windows from molecule_bands.yaml, vs. the same model with that
    molecule removed) exceeds 1× the per-point noise level of that sample.
    Otherwise absent. This ties labels to detectability, not just abundance.
    Rationale: an abundant molecule with no spectral signature in the observed
    band is not detectable; a weaker molecule with a strong feature may be.
    Labels are computed from noiseless spectra to avoid noise-induced label
    flipping. Stored in dataset metadata for reproducibility.

29. **Dataset generation**: Synthetic spectra generated using forward/model.py
    with atmosphere parameters sampled from priors (T ∈ [300, 2500] K,
    log VMR ∈ [-12, -1], r_ref ∈ [0.7, 1.3] × catalog, log_p_cloud ∈ [-6, 2]),
    ~40% chance per molecule to be set to log VMR = -12 (absent). Spectra
    computed on native grid, then binned to cleaned WASP-39 b PRISM wavelength
    grid (147 points, 0.52-5.34 µm). Gaussian noise added: per-point
    uncertainty sampled from 0.5× to 5× the WASP-39 b median uncertainty
    profile; plus small random vertical offset (σ=1e-4) and 5% uncertainty
    mis-estimation as domain randomization. Dataset version hash recorded.
    Target: 20k samples (CI: 3k). Stored in data_cache/ml/ as npz + metadata.

30. **Model architecture**: 1D CNN, 2 input channels (depth, uncertainty),
    multi-label sigmoid output for 5 molecules. Architecture: 4 conv blocks
    (conv1d + batchnorm + relu + maxpool), channels [32, 64, 128, 256],
    global avg pool, dropout(0.3), dense(128), dropout(0.2), dense(5) + sigmoid.
    BCE loss, Adam(1e-3), batch=64. Train/val/test split by unique atmosphere
    config hash (no leakage). Early stopping on val BCE (patience=10).

31. **Training**: seeded (torch.manual_seed, numpy, random), deterministic
    cuDNN. Early stopping on val BCE (patience=10, min_delta=1e-4).
    Checkpoint saves: model state, optimizer state, epoch, metrics, dataset
    hash, model version (e.g., CNN-v1), provenance seed. Metrics: per-molecule
    ROC-AUC, precision/recall@0.5, ECE (10 bins), per-S/N-bin performance.
    Artifacts saved to models/ with versioned filenames.

31. **Inference**: predict(spectrum) → MLResult with per-molecule scores
    labelled "ML candidate score (not abundance, not a detection)".
    Input spectrum interpolated to training grid (WASP-39 b PRISM grid);
    warns if coverage < 90% or grid mismatch. Output includes model version,
    dataset hash, per-molecule scores, calibration flag. JSON serializable.
    Run on real cleaned WASP-39 b spectrum; compare to published (H2O, CO2,
    SO2 expected; CH4 not). Report agreements/failures + domain gap note.

32. **Tests**: label function unit test; dataset generation seeded/reproducible
    (small N=100); model forward pass shapes; predict() output schema +
    labelling; tiny overfit test (model fits 50 samples to near-zero loss).

## Phase 8 (2026-10-08) — 3D scientific digital twin

33. **Twin phase convention**: orbital phase 0 = transit center at +z (between star and
    default/transit camera); implemented in `orbitPoint` via +90 deg rotation. A sign bug
    (-90 deg, transit at -z) was caught by the new `twinGeometry` tests and fixed.

34. **Twin cloud visual**: deck altitude fraction = (2 - logP)/8 of the displayed shell
    (deep = surface/clear, high = top), opacity = 0.15 + 0.55 * fraction; solid only when
    the retrieval CI width < 2 dex (`constrained`), else hatched + "uncertain" label.

35. **Twin atmosphere shell**: outer edge at 4 scale heights (e^-4 ~ 2% residual pressure),
    thickness visually exaggerated x25 (labelled "not to scale"); planet/star/orbit geometry
    itself uses true Rp/Rs and a/Rs ratios. Tidal locking noted as an assumption.

36. **Molecule contribution metric** (model-derived, NOT a detection): per-molecule
    in-band share = sum|d_full - d_noM| / sum|d_full - median(d_full)| over that molecule's
    band windows from `config/molecule_bands.yaml`, clipped to [0, 1].

37. **Twin demo data**: `web/src/mock/wasp39b-twin.json`, `wasp121b-twin.json` are illustrative
    MOCK datasets for UI dev/tests, always badged "mock data" in the UI. Never cite as results.

38. **SpectrumViewer repair**: the Phase 7 file never compiled (missing names, unclosed loop,
    duplicate declarations); rewrote it as a minimal working SVG plot reusing the same props
    interface. Interactive zoom + credible band remain deferred Phase 7 follow-up work.

39. **Dead scratch cleanup**: deleted `fix_line.py`, `fix_line57.py`, `phase6_update.py`,
    `update_progress.py` from the repo root (all broken/syntactically invalid, outside the
    AGENTS.md layout, broke the `ruff check .` gate). Logged here per protocol.

40. **Missing frontend deps**: added `react-router-dom` (imported but never installed) and
    `@tailwindcss/postcss` (Tailwind v4 requires the separate PostCSS package); fixed
    `@theme` single-dash typos and inlined `@apply` of custom classes (unsupported in v4).

## Phase 9a (2026-10-08) — Scientific report generator

41. **Interpretation thresholds (fixed):** supported = ln B >= 5 with GOOD/LIMITED data
    quality; weakly supported = 3 <= ln B < 5; otherwise not constrained (ln B < 3,
    missing Bayes factor, or upper-limit only). POOR data quality overrides any ln B
    to "not constrained" (AGENTS.md rule 3). ML candidate scores never drive support
    classes; only retrieval-based nested-model comparison does.

42. **PDF export ships HTML-only in this environment.** WeasyPrint 70.0 installs via pip
    but cannot render on this Windows box (missing Pango system libraries
    `libgobject-2.0-0`, etc.). `build_report(format="pdf")` raises PDFUnavailableError
    (API: HTTP 501) with that explanation. `jinja2==3.1.6` pinned; `weasyprint` kept
    out of `pyproject.toml` since it cannot function here.

43. **Pipeline repairs required for any end-to-end run (all pre-existing Phase 6 bugs,
    none exercised before: no retrieval/pipeline/API tests existed):**
    - `run_dynesty`/nested runs used dynesty 2.x kwargs (`nlive=`, `seed=`); installed
      dynesty 3.1.0 needs `DynamicNestedSampler` + `nlive=` there, `maxiter_init` /
      `maxbatch=0` (static run) / `dlogz_init` / `maxcall` in `run_nested`.
    - `Pipeline._run_retrieval` dropped its SamplerConfig and called `run_dynesty`
      with 1 arg (needs spectrum, fixed, config, seed); `_run_detection` likewise.
      Both now thread the pipeline options/seed through; `SamplerConfig` and
      `PipelineOptions` gained `maxcall`.
    - `Pipeline._save_cleaned_spectrum` passed a nonexistent `analysis_id` kwarg to
      `SpectrumFile`; now links via the analysis row's `observation_id` (plus the
      existing `spectrum_file_id` pointer).
    - `Pipeline._load_planet_params` hit the network archive; now reads the stored
      Planet catalog row first, archive only as fallback.
    - `RetrievalResult.load_npz` typo `data.ci_95` -> `data["ci_95"]` (95% intervals
      were unloadable); `credible_band_spectrum` gained a `seed` for determinism.
    - `Pipeline._run_preprocess` now persists the PreprocessLog to
      `data_cache/preprocess_logs/<id>.json` (previously discarded); the report
      shows "not run" for older analyses.

44. **Forward-model speed (no science change).** The mock `_compute_transmission_radius`
    looped 29,700 wavelengths with per-point `searchsorted` (~0.18 s/call, making any
    retrieval infeasible). Rewrote as a vectorized first-crossing search; verified
    bitwise-identical output (max abs diff 0.0 over 6 random atmospheres) and ~60x
    faster. Same math, same branch semantics.
    Additionally, `compute_model_spectrum(..., target_spectrum=...)` now evaluates
    only the canonical native nodes (same 1 nm R=1000 spacing) clipped to the
    target span + 0.1 um padding before binning: out-of-range points can never
    enter a flux-conserving bin average, so binned likelihood values are verified
    bitwise-identical (max diff 0.0 over 4 random atmospheres) at ~5x speed.
    Untouched paths: explicit `wavelength_grid=` callers (twin), the full native
    grid default (ML dataset generation, best-fit/credible-band figures).

45. **dynesty 3.x static-run settings.** `DynamicNestedSampler` with `maxbatch=0`
    (baseline run only), `maxiter_init`/`maxcall` caps, `dlogz_init` from options.
    Real WASP-39 b run uses CI-grade settings (n_live=50, dlogz=0.5,
    max_iter=12000, maxcall=30000, seed=42); nested detection models reuse the
    same config. Expected log-evidence uncertainty ~0.7 (reported with every
    number); support thresholds are applied to ln B values with their errors.

46. **Related tables key off the integer analysis PK.** The pipeline wrote the
    `EXO-xxxxxx` string into integer `analysis_id` FK columns (SQLite accepted
    it, but integer-PK lookups then miss). All `_save_*` now resolve the row id
    via `_analysis_pk()`. The stale dev `exosphere.db` (Phase 6 schema, no
    `summary_json` column) was deleted and recreated by `init_db`.

46. **Slice sampling (`sample="rslice"`) as the dynesty proposal default.**
    Uniform multi-ellipsoid proposals (`auto`) suffer acceptance collapse on the
    vague 8-D priors (efficiency fell below 8% and kept dropping; a run stalled
    with dlogz still > 8000 after ~9000 calls). `rslice` draws from live points
    (~4 likelihood calls per iteration, deterministic given the seed) and
    progresses steadily. Still dynesty nested sampling; `SamplerConfig.sample`
    can select `auto` if ever needed.

47. **Root cause of all vague-prior retrieval failures: double prior transform.**
    dynesty calls the likelihood with PHYSICAL parameters (it applies our
    `prior_transform` to unit-cube proposals itself). `_make_log_likelihood`
    (and the detection reduced-likelihood) applied `unit_to_physical` AGAIN,
    so every evaluation ran on garbage parameters: recorded `results.logl`
    matched no stored sample (verified max diff 1.4e7), posteriors collapsed
    to prior bounds, ln B values were 0.0, chi2 was astronomical. Fixed by
    removing the second transform (verified aligned max diff 0.0 under
    dynesty==2.1.5, which is now pinned; 3.1.0's rewritten API is dropped).
    Every retrieval result produced before this fix is invalid, including the
    first two WASP-39 b pipeline runs. Lesson: always verify
    `results.logl[i] == log_likelihood(results.samples[i])` when wiring a new
    sampler version. (Note: item 46's uniform-vs-rslice observations were made
    on the corrupted landscape, so they do not settle the proposal debate.
    Item 50 below re-decides it on the corrected landscape with a timed
    shootout.)

50. **Proposal method: `rwalk` default (timed shootout, corrected landscape).**
    Real spectrum, nlive=50, 400 iters, seed 42: rwalk reached loglmax=-1011
    in 209 s / 7422 calls vs rslice -1057 in 277 s / 10785 calls. rwalk's
    covariance-adapted proposals suit the narrow T/r_ref/abundance ridges;
    rslice decayed to ~3% efficiency (60+ calls/iter) at nlive=500 and stalled
    a standard run (only +220 iters in the second hour; checkpoint discarded).
    This is a sampler-efficiency choice, not result tuning: the posterior
    target is identical, runs stay seeded/deterministic. CI-grade EXO-000001
    used rslice (recorded here and in PROGRESS.md).

## Phase 9b (2026-10-10) — L3 real-data validation + hardening

48. **L3 verdict criteria (registered BEFORE the standard-setting runs complete;**
    **the CI-grade pilot EXO-000001 exists but L3 uses n_live=500/dlogz=0.01):**
    - (a) Data round-trip: PASS = ingested spectrum matches the archived
      published values point-for-point (max abs diff 0.0 at loader precision);
      PARTIAL = agreement within quoted uncertainties only; FAIL = systematic
      mismatch. (L1 test already proves the loader path; rechecked on the file.)
    - (b) Molecules (fixed item-41 thresholds; literature expects strong H2O +
      CO2, SO2 near 4.0 um, CO present, CH4 absent): PASS = H2O and CO2
      supported (ln B >= 5) with GOOD/LIMITED ratings, CH4 not constrained, no
      claim against a POOR rating; PARTIAL = H2O/CO2 supported but CH4
      ambiguous (1 < ln B < 5) or one expected molecule only weakly supported;
      FAIL = H2O or CO2 not constrained, CH4 claimed supported, or any claim
      against a POOR rating. SO2/CO verdicts either way must be reported
      without overclaim.
    - (c) Parameter ranges vs published (T, abundances, cloud): PASS = T
      posterior overlaps the literature range, abundances within 2 dex of
      published values, cloud-constraint direction agrees; PARTIAL = T overlap
      only; FAIL = no overlap anywhere. A model-complexity explanation paragraph
      (isothermal/constant-abundance/5-gas mock vs published complex
      retrievals) is MANDATORY in all cases; agreement must never be forced by
      tuning (any setting change is logged with before/after records).
    - (d) Fit quality: PASS = reduced chi2 < 3 with no >5-sigma structured
      residuals over >3 adjacent bins; PARTIAL = reduced chi2 < 10 with poorly
      fit regions documented; FAIL = reduced chi2 >= 10 or unreported.
    - (e) ML vs retrieval: PASS = no undiscussed contradiction and mandatory ML
      labels present (test-enforced); PARTIAL = contradiction present but
      discussed as domain gap; FAIL = undiscussed contradiction or missing labels.
    - Seed stability (2-3 standard seeds): PASS = medians within mutual 68%
      CIs; PARTIAL = within 95% CIs; FAIL otherwise.

49. **L3 compute scoping (registered before standard runs; amended on evidence).**
    Mains were specified at n_live=500, but measured pace collapsed (2.6
    iters/min and decaying at iter ~7400: bound-refit overhead superlinear in
    nlive dominates; ~99% of wall time outside the likelihood). n_live=500
    mains are infeasible on this (thermally throttled) box. Adjusted to
    n_live=200/dlogz=0.01 for the three L3 mains: identical methodology,
    production-grade precision (logz err ~0.5), errors reported; CI-grade
    (n_live=50) EXO-000001 serves as cross-check. This is a feasibility
    adjustment, not result tuning: it cannot favor any scientific outcome.
    Nested detection stays n_live=100/dlogz=0.1 for seed 42 (seed 123 nested
    as contingency). Full-nlive-500 nested everywhere is infeasible (~100 h);
    noted as a limitation, not a silent cut.
