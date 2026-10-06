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

## Repository layout

```
src/exosphere/   core/, data/, quality/, preprocess/, forward/, retrieval/, ml/, api/, report/
web/             React dashboard + 3D twin
tests/           unit + validation (L1/L2/L3)
data_cache/      gitignored local data cache
```
