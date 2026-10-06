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
