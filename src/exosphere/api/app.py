"""FastAPI application for ExoSphere."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from exosphere.api.config import settings
from exosphere.api.db import (
    Analysis,
    AnalysisStage,
    AnalysisStatus,
    Observation,
    Planet,
    QualityReport,
    async_session_maker,
    close_db,
    init_db,
)
from exosphere.api.schemas import (
    AnalysisCreate,
    AnalysisCreateResponse,
    AnalysisDetail,
    AnalysisListItem,
    AnalysisStatusResponse,
    BestFitSpectrumResponse,
    DetectionResultItem,
    DetectionResultsResponse,
    HealthResponse,
    MLScoresResponse,
    MoleculeBandsResponse,
    ObservationSummary,
    PaginatedAnalyses,
    PlanetDetail,
    PlanetSearchResult,
    PosteriorSamplesResponse,
    ProvenanceResponse,
    QualityReportResponse,
    ReportRequest,
    ReportResponse,
    RetrievalSummary,
    SpectrumResponse,
    TwinParametersResponse,
)
from exosphere.core.logging import get_logger
from exosphere.pipeline import PipelineOptions

# In-memory job tracking (in production, use Redis or similar)
running_jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()


app = FastAPI(
    title="ExoSphere API",
    description="Exoplanet atmospheric characterization platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for outputs
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

api_log = get_logger("exosphere.api")


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP errors as clean JSON (no stack traces in responses, ever)."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: traceback goes to the server log, clients get a message."""
    api_log.error("unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Dependency for DB session
async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session


async def _get_analysis(db: AsyncSession, analysis_id: str) -> Analysis:
    """Load an analysis with its relations, or raise 404.

    Related tables (posteriors, ML, quality, detections) key off the
    integer primary key, never the EXO-xxxxxx string.
    """
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(Analysis)
        .options(
            selectinload(Analysis.planet),
            selectinload(Analysis.observation),
            selectinload(Analysis.spectrum_file),
        )
        .where(Analysis.analysis_id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


async def _latest_row(db: AsyncSession, model: Any, pk: int) -> Any | None:
    """Newest row of a per-analysis table (rebuilds supersede older rows)."""
    result = await db.execute(
        select(model).where(model.analysis_id == pk).order_by(desc(model.id))
    )
    return result.scalars().first()


# Background job runner
async def run_analysis_job(
    analysis_id: str,
    planet_name: str,
    observation_ref: str,
    options: PipelineOptions,
):
    """Run analysis in background."""
    from exosphere.pipeline import PipelineOptions

    # Mark as running in job tracker
    running_jobs[analysis_id] = {
        "status": "running",
        "stage": "init",
        "progress": 0.0,
        "started_at": datetime.utcnow(),
    }

    try:
        PipelineOptions(
            n_live=options.n_live,
            dlogz=options.dlogz,
            max_iter=options.max_iter,
            seed=options.seed,
            run_quality=options.run_quality,
            run_preprocess=options.run_preprocess,
            run_ml=options.run_ml,
            run_retrieval=options.run_retrieval,
            run_detection=options.run_detection,
            run_quality_check=options.run_quality_check,
            error_inflation=options.error_inflation,
            error_inflation_free=options.error_inflation_free,
            fixed_params=options.fixed_params,
        )

        # Create pipeline and run
        from exosphere.pipeline import Pipeline
        from exosphere.pipeline import PipelineOptions as PipeOptions

        pipe_options = PipeOptions(
            n_live=options.n_live,
            dlogz=options.dlogz,
            max_iter=options.max_iter,
            maxcall=options.maxcall,
            seed=options.seed,
            run_quality=options.run_quality,
            run_preprocess=options.run_preprocess,
            run_ml=options.run_ml,
            run_retrieval=options.run_retrieval,
            run_detection=options.run_detection,
            run_quality_check=options.run_quality_check,
            error_inflation=options.error_inflation
            if options.error_inflation is not None
            else 0.0,
            error_inflation_free=options.error_inflation_free,
            fixed_params=options.fixed_params,
        )

        pipeline = Pipeline(analysis_id, pipe_options)
        await pipeline.run(
            planet_name=planet_name,
            observation_ref=observation_ref,
            fixed_params=options.fixed_params,
        )

        running_jobs[analysis_id]["status"] = "completed"
        running_jobs[analysis_id]["completed_at"] = datetime.utcnow()

    except Exception as e:
        running_jobs[analysis_id]["status"] = "failed"
        running_jobs[analysis_id]["error"] = str(e)


# --- Health ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        database="connected",
    )


# --- Planet endpoints ---
@app.get("/planets/search", response_model=list[PlanetSearchResult])
async def search_planets(
    q: str = Query(..., description="Planet name to search for"),
    db: AsyncSession = Depends(get_db),
):
    """Search planets by name."""
    result = await db.execute(select(Planet).where(Planet.name.ilike(f"%{q}%")).limit(20))
    planets = result.scalars().all()
    return [
        PlanetSearchResult(
            name=p.name,
            pl_rade=p.pl_rade,
            pl_bmasse=p.pl_bmasse,
            pl_orbper=p.pl_orbper,
            pl_eqt=p.pl_eqt,
            st_teff=p.st_teff,
            st_rad=p.st_rad,
        )
        for p in planets
    ]


@app.get("/planets/{name}", response_model=PlanetDetail)
async def get_planet(
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """Get planet details with available observations."""
    result = await db.execute(select(Planet).where(Planet.name.ilike(name)))
    planet = result.scalar_one_or_none()
    if not planet:
        raise HTTPException(status_code=404, detail="Planet not found")

    # Get observations
    obs_result = await db.execute(select(Observation).where(Observation.planet_id == planet.id))
    observations = obs_result.scalars().all()

    return PlanetDetail(
        name=planet.name,
        pl_rade=planet.pl_rade,
        pl_bmasse=planet.pl_bmasse,
        pl_bmassprov=planet.pl_bmassprov,
        pl_eqt=planet.pl_eqt,
        pl_insol=planet.pl_insol,
        pl_orbper=planet.pl_orbper,
        pl_orbsmax=planet.pl_orbsmax,
        pl_orbeccen=planet.pl_orbeccen,
        pl_orbincl=planet.pl_orbincl,
        pl_tranmid=planet.pl_tranmid,
        pl_ratror=planet.pl_ratror,
        pl_ratdor=planet.pl_ratdor,
        st_rad=planet.st_rad,
        st_teff=planet.st_teff,
        st_mass=planet.st_mass,
        st_logg=planet.st_logg,
        st_met=planet.st_met,
        st_spectype=planet.st_spectype,
        surface_gravity_m_s2=planet.surface_gravity_m_s2,
        observations=[
            ObservationSummary(
                observation_id=o.observation_id,
                instrument=o.instrument,
                facility=o.telescope,
                wavelength_min_um=o.wavelength_min_um,
                wavelength_max_um=o.wavelength_max_um,
                num_points=o.num_points,
                reference=o.source_archive,
                spectrum_file_path=o.spectrum_file_path,
            )
            for o in observations
        ],
    )


# --- Analysis endpoints ---
@app.post("/analyses", response_model=AnalysisCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    analysis: AnalysisCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create a new analysis job."""
    # Generate analysis ID
    result = await db.execute(select(func.max(Analysis.id)))
    max_id = result.scalar() or 0
    analysis_id = f"EXO-{max_id + 1:06d}"

    # Verify planet exists
    result = await db.execute(select(Planet).where(Planet.name.ilike(analysis.planet_name)))
    planet = result.scalar_one_or_none()
    if not planet:
        raise HTTPException(status_code=404, detail=f"Planet '{analysis.planet_name}' not found")

    # Determine observation
    if analysis.observation_id:
        obs_result = await db.execute(
            select(Observation).where(Observation.observation_id == analysis.observation_id)
        )
        observation = obs_result.scalar_one_or_none()
        if not observation:
            raise HTTPException(status_code=404, detail="Observation not found")
    else:
        # Use best available (first transmission spectrum)
        obs_result = await db.execute(
            select(Observation).where(Observation.planet_id == planet.id).limit(1)
        )
        observation = obs_result.scalar_one_or_none()
        if not observation:
            raise HTTPException(status_code=404, detail="No observations found for planet")

    # Create analysis record
    analysis_obj = Analysis(
        analysis_id=analysis_id,
        planet_id=planet.id,
        observation_id=observation.id,
        status=AnalysisStatus.PENDING,
        stage=AnalysisStage.INIT,
        seed=analysis.seed,
        config_json={
            "n_live": analysis.n_live,
            "dlogz": analysis.dlogz,
            "max_iter": analysis.max_iter,
            "maxcall": analysis.maxcall,
            "run_ml": analysis.run_ml,
            "run_detection": analysis.run_detection,
            "run_quality": analysis.run_quality,
            "run_preprocess": analysis.run_preprocess,
            "error_inflation": analysis.error_inflation,
            "error_inflation_free": analysis.error_inflation_free,
            "fixed_params": analysis.fixed_params,
        },
    )

    db.add(analysis_obj)
    await db.commit()
    await db.refresh(analysis_obj)

    # The pipeline loads spectra from file paths: prefer the observation's
    # stored spectrum file, falling back to the observation id string
    # (resolved against data_cache by the pipeline).
    observation_ref = observation.spectrum_file_path or analysis.observation_id or ""

    # Start background job
    background_tasks.add_task(
        run_analysis_job,
        analysis_obj.analysis_id,
        analysis.planet_name,
        observation_ref,
        PipelineOptions(
            n_live=analysis.n_live,
            dlogz=analysis.dlogz,
            max_iter=analysis.max_iter,
            maxcall=analysis.maxcall,
            seed=analysis.seed,
            run_ml=analysis.run_ml,
            run_detection=analysis.run_detection,
            run_quality=analysis.run_quality,
            run_preprocess=analysis.run_preprocess,
            run_quality_check=analysis.run_quality_check,
            error_inflation=analysis.error_inflation,
            error_inflation_free=analysis.error_inflation_free,
            fixed_params=analysis.fixed_params,
        ),
    )

    return AnalysisCreateResponse(
        analysis_id=analysis_obj.analysis_id,
        status=analysis_obj.status,
        message="Analysis started",
    )


@app.get("/analyses", response_model=PaginatedAnalyses)
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List analyses with pagination."""
    query = select(Analysis).order_by(desc(Analysis.created_at))

    if status:
        query = query.where(Analysis.status == status)

    # Count total
    total_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = total_result.scalar()

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    analyses = result.scalars().all()

    items = [
        AnalysisListItem(
            analysis_id=a.analysis_id,
            planet_name=a.planet.name if a.planet else "Unknown",
            observation_id=a.observation.observation_id if a.observation else "Unknown",
            status=a.status,
            stage=a.stage,
            progress=a.progress,
            created_at=a.created_at,
            completed_at=a.completed_at,
        )
        for a in analyses
    ]

    return PaginatedAnalyses(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@app.get("/analyses/{analysis_id}", response_model=AnalysisDetail)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full analysis details."""
    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisDetail(
        analysis_id=analysis.analysis_id,
        planet_name=analysis.planet.name if analysis.planet else "Unknown",
        observation_id=analysis.observation.observation_id if analysis.observation else "Unknown",
        status=analysis.status,
        stage=analysis.stage,
        progress=analysis.progress,
        current_step=analysis.current_step,
        error_message=analysis.error_message,
        config=analysis.config_json,
        seed=analysis.seed,
        n_live=analysis.config_json.get("n_live", 500),
        dlogz=analysis.config_json.get("dlogz", 0.01),
        max_iter=analysis.config_json.get("max_iter", 50000),
        created_at=analysis.created_at,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        updated_at=analysis.updated_at,
        provenance=analysis.provenance_json,
    )


@app.get("/analyses/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get analysis status."""
    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return AnalysisStatusResponse(
        analysis_id=analysis.analysis_id,
        status=analysis.status,
        stage=analysis.stage,
        progress=analysis.progress,
        current_step=analysis.current_step,
        error_message=analysis.error_message,
        created_at=analysis.created_at,
        started_at=analysis.started_at,
        completed_at=analysis.completed_at,
        updated_at=analysis.updated_at,
    )


@app.get("/analyses/{analysis_id}/spectrum", response_model=SpectrumResponse)
async def get_spectrum(
    analysis_id: str,
    cleaned: bool = Query(False, description="Return cleaned spectrum if available"),
    db: AsyncSession = Depends(get_db),
):
    """Get spectrum data for an analysis."""
    from pathlib import Path

    from exosphere.core.spectrum import Spectrum

    analysis = await _get_analysis(db, analysis_id)

    # Determine which spectrum file to load
    spec_path: str | None = None
    if cleaned and analysis.spectrum_file is not None:
        spec_path = analysis.spectrum_file.file_path
    elif analysis.observation is not None and analysis.observation.spectrum_file_path:
        spec_path = analysis.observation.spectrum_file_path
    if not spec_path or not Path(spec_path).exists():
        raise HTTPException(status_code=404, detail="Spectrum file not found")

    spectrum = Spectrum.load(spec_path)

    return SpectrumResponse(
        wavelength_um=spectrum.wavelength,
        transmission=spectrum.transmission,
        uncertainty=spectrum.uncertainty,
        wavelength_bin_edges_um=spectrum.wavelength_bin_edges,
        quality_flags=spectrum.quality_flags,
        observation_id=spectrum.observation_id,
        target_id=spectrum.target_id,
        instrument=spectrum.instrument,
        provenance=spectrum.provenance.model_dump(mode="json") if spectrum.provenance else None,
    )


@app.get("/analyses/{analysis_id}/quality", response_model=QualityReportResponse)
async def get_quality_report(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get quality assessment report."""
    analysis = await _get_analysis(db, analysis_id)
    result = await db.execute(select(QualityReport).where(QualityReport.analysis_id == analysis.id))
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Quality report not found")

    stored = dict(report.report_json or {})
    uncertainty = stored.get("uncertainty", {}) or {}
    molecules = stored.get("molecules", []) or []
    return QualityReportResponse(
        analysis_id=analysis_id,
        overall_suitability=stored.get("suitability", "POOR"),
        median_snr=stored.get("median_snr"),
        max_band_snr=stored.get("max_band_snr"),
        wavelength_coverage_fraction=stored.get("wavelength_coverage_fraction", 0.0),
        flagged_fraction=stored.get("flagged_fraction", 0.0),
        nan_fraction=stored.get("nan_fraction", 0.0),
        outlier_count=stored.get("outlier_count", 0),
        outlier_fraction=stored.get("outlier_fraction", 0.0),
        median_uncertainty=uncertainty.get("median"),
        uncertainty_non_positive_count=uncertainty.get("non_positive_count", 0),
        uncertainty_nan_count=uncertainty.get("nan_count", 0),
        uncertainty_huge_count=uncertainty.get("huge_count", 0),
        uncertainty_tiny_count=uncertainty.get("tiny_count", 0),
        effective_resolving_power=stored.get("effective_resolving_power"),
        molecule_ratings=stored.get("molecule_ratings", {}),
        molecule_details={entry.get("molecule", "?"): entry for entry in molecules},
        thresholds=stored.get("thresholds", {}),
        config_version=stored.get("config_version", ""),
        method=stored.get("method", {}),
    )


@app.get("/analyses/{analysis_id}/ml", response_model=MLScoresResponse)
async def get_ml_results(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get ML candidate scores."""
    from exosphere.api.db import MLResult as DBMLResult

    analysis = await _get_analysis(db, analysis_id)
    result = await db.execute(select(DBMLResult).where(DBMLResult.analysis_id == analysis.id))
    ml_result = result.scalars().first()
    if not ml_result:
        raise HTTPException(status_code=404, detail="ML results not found")

    details = dict(getattr(ml_result, "details_json", None) or {})
    wavelength_range = details.get("input_wavelength_range") or (0.0, 0.0)
    return MLScoresResponse(
        analysis_id=analysis_id,
        scores=ml_result.scores_json,
        model_version=ml_result.model_version,
        dataset_hash=ml_result.dataset_hash,
        model_config_hash=ml_result.model_config_hash,
        timestamp=ml_result.created_at.isoformat() + "Z",
        input_coverage_fraction=float(details.get("input_coverage_fraction", 0.0)),
        input_wavelength_range_um=(float(wavelength_range[0]), float(wavelength_range[1])),
        grid_match=bool(details.get("grid_match", False)),
        warnings=list(details.get("warnings", ["ML metadata not stored for this run"])),
    )


@app.get("/analyses/{analysis_id}/retrieval", response_model=RetrievalSummary)
async def get_retrieval(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get retrieval summary (median + credible intervals, never bare points)."""
    from pathlib import Path

    from exosphere.api.db import Posterior

    analysis = await _get_analysis(db, analysis_id)
    posterior = await _latest_row(db, Posterior, analysis.id)
    if not posterior:
        raise HTTPException(status_code=404, detail="Retrieval results not found")

    summary = dict(posterior.summary_json or {})
    ci_95: dict[str, list[float]] = {}
    runtime_s = 0.0
    npz_path = Path(posterior.file_path) if posterior.file_path else None
    if npz_path is not None and npz_path.exists():
        from exosphere.retrieval.results import RetrievalResult

        result = RetrievalResult.load_npz(npz_path)
        for i, name in enumerate(result.param_names):
            ci_95[name] = [float(result.ci_95[i, 0]), float(result.ci_95[i, 1])]
        runtime_s = float(result.runtime_s)

    config = dict(analysis.config_json or {})
    return RetrievalSummary(
        analysis_id=analysis_id,
        logz=posterior.logz or 0.0,
        logz_err=posterior.logz_err or 0.0,
        best_fit={k: float(v) for k, v in (summary.get("best_fit", {}) or {}).items()},
        median={k: float(v) for k, v in (summary.get("median", {}) or {}).items()},
        ci_68={
            k: [float(lo), float(hi)] for k, (lo, hi) in (summary.get("ci_68", {}) or {}).items()
        },
        ci_95=ci_95,
        param_names=list(summary.get("param_names", [])),
        n_samples=posterior.n_samples or 0,
        n_live=posterior.n_live or 0,
        dlogz=posterior.dlogz or 0.0,
        runtime_s=runtime_s,
        sampler="dynesty",
        seed=analysis.seed,
        error_inflation=float(config.get("error_inflation") or 0.0),
    )


@app.get("/analyses/{analysis_id}/posterior", response_model=PosteriorSamplesResponse)
async def get_posterior(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get posterior samples for corner plot."""
    from pathlib import Path

    from exosphere.api.db import Posterior

    analysis = await _get_analysis(db, analysis_id)
    posterior = await _latest_row(db, Posterior, analysis.id)
    if not posterior:
        raise HTTPException(status_code=404, detail="Posterior not found")

    npz_path = Path(posterior.file_path) if posterior.file_path else None
    if npz_path is None or not npz_path.exists():
        raise HTTPException(status_code=404, detail="Posterior samples file not found")

    from exosphere.retrieval.results import RetrievalResult

    result = RetrievalResult.load_npz(npz_path)
    return PosteriorSamplesResponse(
        analysis_id=analysis_id,
        samples=[[float(v) for v in row] for row in result.samples],
        weights=[float(w) for w in result.weights],
        param_names=list(result.param_names),
        logz=posterior.logz or 0.0,
        logz_err=posterior.logz_err or 0.0,
    )


@app.get("/analyses/{analysis_id}/detection", response_model=DetectionResultsResponse)
async def get_detection(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detection results."""
    from sqlalchemy import select

    from exosphere.api.db import DetectionResult as DBDetectionResult

    analysis = await _get_analysis(db, analysis_id)
    result = await db.execute(
        select(DBDetectionResult)
        .where(DBDetectionResult.analysis_id == analysis.id)
        .order_by(desc(DBDetectionResult.id))
    )
    detections = result.scalars().all()

    results = [
        DetectionResultItem(
            molecule=d.molecule,
            ln_bayes_factor=d.ln_bayes_factor,
            ln_bayes_factor_err=d.ln_bayes_factor_err,
            sigma_equivalent=d.sigma_equivalent,
            status=d.status,
            upper_limit_log_vmr=d.upper_limit_log_vmr,
            details=d.details_json,
        )
        for d in detections
    ]

    summary = {d.molecule: d.status for d in detections}

    return DetectionResultsResponse(
        analysis_id=analysis_id,
        results=results,
        summary=summary,
    )


@app.get("/analyses/{analysis_id}/model", response_model=BestFitSpectrumResponse)
async def get_model(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get best-fit model spectrum with credible band (binned to observed grid)."""
    from pathlib import Path

    import numpy as np

    from exosphere.api.db import Posterior
    from exosphere.core.spectrum import Spectrum
    from exosphere.forward.model import PlanetFixed, to_instrument
    from exosphere.retrieval.results import RetrievalResult

    analysis = await _get_analysis(db, analysis_id)
    posterior = await _latest_row(db, Posterior, analysis.id)
    if not posterior or not posterior.file_path or not Path(posterior.file_path).exists():
        raise HTTPException(status_code=404, detail="Retrieval results not found")

    spec_path: str | None = None
    if analysis.spectrum_file is not None:
        spec_path = analysis.spectrum_file.file_path
    elif analysis.observation is not None and analysis.observation.spectrum_file_path:
        spec_path = analysis.observation.spectrum_file_path
    if not spec_path or not Path(spec_path).exists():
        raise HTTPException(status_code=404, detail="Spectrum file not found")
    spectrum = Spectrum.load(spec_path)

    planet = analysis.planet
    if planet is None or planet.surface_gravity_m_s2 is None or planet.st_rad is None:
        raise HTTPException(status_code=404, detail="Catalog parameters missing for model")
    fixed = PlanetFixed(
        gravity_m_s2=float(planet.surface_gravity_m_s2),
        stellar_radius_rsun=float(planet.st_rad),
        reference_pressure_bar=0.01,
    )
    result = RetrievalResult.load_npz(posterior.file_path)
    native_wl, native_best = result.best_fit_spectrum(fixed)
    mw, mb = to_instrument(np.asarray(native_wl), np.asarray(native_best), spectrum)
    med, lo, hi = result.credible_band_spectrum(fixed, n_draws=64, seed=analysis.seed)
    _, med_b = to_instrument(np.asarray(native_wl), np.asarray(med), spectrum)
    _, lo_b = to_instrument(np.asarray(native_wl), np.asarray(lo), spectrum)
    _, hi_b = to_instrument(np.asarray(native_wl), np.asarray(hi), spectrum)
    return BestFitSpectrumResponse(
        analysis_id=analysis_id,
        wavelength_um=[float(v) for v in mw],
        best_fit_depth=[float(v) for v in mb],
        median_depth=[float(v) for v in med_b],
        ci_lo=[float(v) for v in lo_b],
        ci_hi=[float(v) for v in hi_b],
        observed_wavelength_um=[float(v) for v in spectrum.wavelength],
        observed_depth=[float(v) for v in spectrum.transmission],
        observed_uncertainty=[float(v) for v in spectrum.uncertainty],
    )


@app.get("/analyses/{analysis_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full provenance record."""
    analysis = await _get_analysis(db, analysis_id)
    observation = analysis.observation
    stored = dict(analysis.provenance_json or {})
    config = dict(analysis.config_json or {})

    return ProvenanceResponse(
        analysis_id=analysis_id,
        planet=analysis.planet.name if analysis.planet else None,
        observation_id=observation.observation_id if observation else None,
        telescope=observation.telescope if observation else None,
        instrument=observation.instrument if observation else None,
        source_archive=observation.source_archive if observation else None,
        input_data_version=stored.get("input_data_version"),
        input_data_hash=stored.get("input_data_hash"),
        preprocessing_version=stored.get("preprocessing_version"),
        ml_model_version=stored.get("ml_model_version"),
        retrieval_model_version=stored.get("retrieval_model_version"),
        retrieval_parameters=config,
        timestamp=analysis.created_at.isoformat() + "Z",
        result_reference=stored.get("result_reference"),
    )


@app.get("/config/molecule-bands", response_model=MoleculeBandsResponse)
async def get_molecule_bands():
    """Molecule absorption band windows + quality config version."""
    from exosphere.quality.assess import load_quality_config

    cfg = load_quality_config()
    return MoleculeBandsResponse(
        version=cfg.version,
        wavelength_range_um=[cfg.wavelength_range_um[0], cfg.wavelength_range_um[1]],
        molecules={mol: [[lo, hi] for lo, hi in wins] for mol, wins in cfg.molecule_bands.items()},
    )


@app.get("/analyses/{analysis_id}/twin", response_model=TwinParametersResponse)
async def get_twin_parameters(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Scientific digital twin parameters with provenance source tags.

    Planet/star/orbit values come from the catalog (measured), the atmosphere
    temperature, cloud-top pressure and radius from the retrieval posterior
    median when a retrieval summary was stored (inferred), quantities computed
    from those (derived), and documented defaults otherwise (assumed).
    """
    from pathlib import Path

    from sqlalchemy.orm import selectinload

    from exosphere.api.db import Posterior
    from exosphere.core.spectrum import Spectrum
    from exosphere.twin import build_twin_parameters

    result = await db.execute(
        select(Analysis)
        .options(
            selectinload(Analysis.planet),
            selectinload(Analysis.observation),
            selectinload(Analysis.spectrum_file),
        )
        .where(Analysis.analysis_id == analysis_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    planet = analysis.planet
    catalog = {
        "pl_rade": planet.pl_rade if planet else None,
        "pl_bmasse": planet.pl_bmasse if planet else None,
        "st_rad": planet.st_rad if planet else None,
        "st_teff": planet.st_teff if planet else None,
        "pl_orbsmax": planet.pl_orbsmax if planet else None,
        "pl_orbper": planet.pl_orbper if planet else None,
        "pl_orbincl": planet.pl_orbincl if planet else None,
        "pl_orbeccen": planet.pl_orbeccen if planet else None,
        "pl_eqt": planet.pl_eqt if planet else None,
        "pl_ratdor": planet.pl_ratdor if planet else None,
        "surface_gravity_m_s2": planet.surface_gravity_m_s2 if planet else None,
    }

    retrieval: dict | None = None
    post_result = await db.execute(
        select(Posterior).where(Posterior.analysis_id == analysis.id)
    )
    posterior = post_result.scalars().first()
    if posterior is not None and posterior.summary_json:
        summary = posterior.summary_json
        retrieval = {
            "median": summary.get("median", {}),
            "ci_68": summary.get("ci_68", {}),
            "best_fit": summary.get("best_fit", {}),
        }

    spectrum: dict | None = None
    spec_path: str | None = None
    if analysis.spectrum_file is not None:
        spec_path = analysis.spectrum_file.file_path
    elif analysis.observation is not None and analysis.observation.spectrum_file_path:
        spec_path = analysis.observation.spectrum_file_path
    if spec_path and Path(spec_path).exists():
        spec = Spectrum.load(spec_path)
        spectrum = {
            "wavelength": list(spec.wavelength),
            "transmission": list(spec.transmission),
        }

    twin = build_twin_parameters(
        analysis_id=analysis_id,
        planet_name=planet.name if planet else "unknown",
        catalog=catalog,
        retrieval=retrieval,
        spectrum=spectrum,
    )
    return TwinParametersResponse(**twin)


@app.post("/analyses/{analysis_id}/report", response_model=ReportResponse)
async def create_report(
    analysis_id: str,
    body: ReportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Build the scientific report from stored products (HTML; PDF if available)."""
    from exosphere.api.db import Report
    from exosphere.report.build import (
        REPORT_VERSION,
        PDFUnavailableError,
        ReportNotFoundError,
        ReportWordingError,
        build_report_async,
    )

    try:
        await build_report_async(analysis_id, format=body.format)
    except ReportNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PDFUnavailableError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    except ReportWordingError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:  # pragma: no cover - raced deletion
        raise HTTPException(status_code=404, detail="Analysis not found")
    rep_result = await db.execute(
        select(Report)
        .where(Report.analysis_id == analysis.id)
        .order_by(desc(Report.id))
    )
    report = rep_result.scalars().first()
    if report is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="Report was not stored")
    return ReportResponse(
        analysis_id=analysis_id,
        format=body.format,
        file_path=report.file_path,
        file_hash=report.file_hash,
        report_version=REPORT_VERSION,
        created_at=report.created_at,
    )


@app.get("/analyses/{analysis_id}/report")
async def download_report(
    analysis_id: str,
    format: str = Query(default="html", description="Report format to download"),
    db: AsyncSession = Depends(get_db),
):
    """Download a previously built scientific report file."""
    from fastapi.responses import FileResponse

    from exosphere.api.db import Report

    if format not in ("html", "pdf"):
        raise HTTPException(status_code=422, detail="format must be 'html' or 'pdf'")
    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    rep_result = await db.execute(
        select(Report)
        .where(Report.analysis_id == analysis.id)
        .order_by(desc(Report.id))
    )
    report = None
    for row in rep_result.scalars().all():
        if row.file_path.endswith(f".{format}"):
            report = row
            break
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {format} report built yet; POST /analyses/{analysis_id}/report first",
        )
    media_type = "text/html" if format == "html" else "application/pdf"
    return FileResponse(
        report.file_path,
        media_type=media_type,
        filename=f"{analysis_id}_report.{format}",
    )


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete an analysis and all associated data."""
    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    await db.delete(analysis)
    await db.commit()
    return {"message": "Analysis deleted"}


# Register routes (already defined above)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
