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
