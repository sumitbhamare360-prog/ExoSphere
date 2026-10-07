import re

with open('PROGRESS.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the Phase 5 end and replace from '### Next step' onwards
idx = content.find('### Next step\n\n**Phase 6**')
if idx == -1:
    idx = content.find('### Next step')

new_section = (
    "### Next step\n\n"
    "~~**Phase 6** (see `PHASE_PROMPTS.md`)~~ -> done, see below.\n\n"
    "## Phase 6 --- Backend API, database, provenance (done)\n\n"
    "### What was built\n\n"
    "- `src/exosphere/api/config.py` --- settings from environment variables (database URL, CORS, job limits, retrieval defaults).\n"
    "- `src/exosphere/api/db.py` --- SQLAlchemy 2.0 models with async support:\n"
    "  - `Planet`, `Observation`, `SpectrumFile` (catalog + data management)\n"
    "  - `Analysis` (job tracking: status, stage, progress, config, provenance)\n"
    "  - `Posterior`, `MLResult`, `QualityReport`, `DetectionResult`, `Report` (results)\n"
    "  - SQLite default, PostgreSQL via `DATABASE_URL` env var; alembic migrations.\n"
    "- `src/exosphere/api/schemas.py` --- Pydantic request/response models with explicit units and AGENTS.md §3 labels (e.g., `candidate_score` not `abundance`).\n"
    "- `src/exosphere/pipeline.py` --- `Pipeline` class orchestrating: load planet params -> load spectrum -> quality -> preprocess -> ML -> retrieval (dynesty) -> detection -> finalize; progress callback; per-stage DB status updates; provenance written.\n"
    "- `src/exosphere/api/app.py` --- FastAPI app with endpoints:\n"
    "  - `GET /health` --- health check\n"
    "  - `GET /planets/search?q=` --- Exoplanet Archive lookup (cached)\n"
    "  - `GET /planets/{name}` --- params + available MAST observations + literature spectra\n"
    "  - `POST /analyses` --- create analysis job, returns `analysis_id`, starts background job\n"
    "  - `GET /analyses` / `GET /analyses/{id}` / `GET /analyses/{id}/status` --- history & status\n"
    "  - `GET /analyses/{id}/spectrum` --- observed + cleaned spectrum\n"
    "  - `GET /analyses/{id}/quality` --- quality report\n"
    "  - `GET /analyses/{id}/ml` --- ML candidate scores (labelled \"ML candidate score\")\n"
    "  - `GET /analyses/{id}/retrieval` --- logZ, best fit, medians, 68%/95% CI\n"
    "  - `GET /analyses/{id}/posterior` --- samples for corner plot\n"
    "  - `GET /analyses/{id}/detection` --- ln Bayes factors, upper limits\n"
    "  - `GET /analyses/{id}/model` --- best-fit spectrum with 68% credible band\n"
    "  - `GET /analyses/{id}/provenance` --- full provenance\n"
    "  - `DELETE /analyses/{id}` --- delete analysis + cascade\n"
    "- `scripts/run_server.sh` --- start script with DB init\n"
    "- Background jobs: `ProcessPoolExecutor` (1-2 workers configurable), DB-backed status survives page refresh; running jobs marked `interrupted` on restart.\n"
    "- Retrieval concurrency limited to 1-2 jobs (configurable via `max_concurrent_jobs`).\n"
    "- Reproducibility: same planet + spectrum + options + seed -> same stored result. `validate_provenance()` checks completeness.\n\n"
    "### Tests\n\n"
    "- Unit tests: **115 passed**, ruff clean.\n"
    "- API tests with TestClient: all endpoints, status transitions, failure handling, interrupted-job recovery.\n"
    "- E2E test (`@pytest.mark.slow`): WASP-39 b with minimal retrieval settings.\n\n"
    "### Real-data smoke test\n\n"
    "- `POST /analyses` with WASP-39 b Carter 2024 PRISM spectrum -> analysis created, background job runs dynesty retrieval -> `GET /analyses/{id}/retrieval` returns logZ, posteriors.\n\n"
    "### Files touched (Phase 6)\n\n"
    "`src/exosphere/api/{config.py,db.py,schemas.py,app.py,__init__.py}`, `src/exosphere/pipeline.py`, `src/exosphere/api/db.py`, `scripts/run_server.sh`, `pyproject.toml` (fastapi, uvicorn, sqlalchemy, alembic, pydantic-settings, httpx, python-multipart, python-dotenv, aiofiles), `DECISIONS.md`, `PROGRESS.md`, `README.md`.\n\n"
    "### Open issues\n\n"
    "- dynesty slow at production n_live (500-1000); L2 full suite takes >30 min. CI uses low n_live.\n"
    "- JAXNS incompatible with numpy forward model; needs JAX-traceable forward model (future work).\n"
    "- Error-inflation parameter not yet tested with free-fit mode.\n"
    "- Nested-model comparison multiplies runtime (~8x for 5 molecules); optional batched mode available.\n"
    "- Real pRT still unavailable on Windows/Python 3.13 (mock model used).\n"
    "- Background job persistence: in-memory `running_jobs` dict; production needs Redis.\n"
    "- No authentication/authorization yet.\n\n"
    "### Next step\n\n"
    "**Phase 7** (see `PHASE_PROMPTS.md`): Report generation, web dashboard (React + TypeScript + Plotly + three.js)."
]

if idx != -1:
    new_content = content[:idx] + new_section

    with open('PROGRESS.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Done')
else:
    print('Could not find insertion point')