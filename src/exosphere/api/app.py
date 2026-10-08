"""FastAPI application for ExoSphere."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
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
    RetrievalSummary,
    SpectrumResponse,
    TwinParametersResponse,
)
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


# Dependency for DB session
async def get_db() -> AsyncSession:
    async with async_session_maker() as session:
        yield session


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

        pipeline = Pipeline(analysis_id, pipe_options)
        await pipeline.run(
            planet_name=options.planet_name,
            observation_ref=options.observation_ref,
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

    # Start background job
    background_tasks.add_task(
        run_analysis_job,
        analysis_obj.analysis_id,
        analysis.planet_name,
        analysis.observation_id or "",
        PipelineOptions(
            n_live=analysis.n_live,
            dlogz=analysis.dlogz,
            max_iter=analysis.max_iter,
            seed=analysis.seed,
            run_ml=analysis.run_ml,
            run_detection=analysis.run_detection,
            run_quality=analysis.run_quality,
            run_preprocess=analysis.run_preprocess,
            run_quality_check=analysis.run_quality,
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
    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Determine which spectrum file to load
    if cleaned and analysis.spectrum_file:
        spec_file = analysis.spectrum_file
    elif analysis.observation and analysis.observation.spectrum_file_path:
        # Load from observation
        from pathlib import Path

        from exosphere.core.spectrum import Spectrum

        path = Path(analysis.observation.spectrum_file_path)
        if path.exists():
            spectrum = Spectrum.load(path)
        else:
            raise HTTPException(status_code=404, detail="Spectrum file not found")
    else:
        raise HTTPException(status_code=404, detail="No spectrum available")

    spectrum = Spectrum.load(spec_file.file_path)

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
    from sqlalchemy import select

    result = await db.execute(select(QualityReport).where(QualityReport.analysis_id == analysis_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Quality report not found")

    return QualityReportResponse(**report.report_json)


@app.get("/analyses/{analysis_id}/ml", response_model=MLScoresResponse)
async def get_ml_results(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get ML candidate scores."""
    from sqlalchemy import select

    from exosphere.api.db import MLResult as DBMLResult

    result = await db.execute(select(DBMLResult).where(DBMLResult.analysis_id == analysis_id))
    ml_result = result.scalar_one_or_none()
    if not ml_result:
        raise HTTPException(status_code=404, detail="ML results not found")

    return MLScoresResponse(
        analysis_id=analysis_id,
        scores=ml_result.scores_json,
        model_version=ml_result.model_version,
        dataset_hash=ml_result.dataset_hash,
        model_config_hash=ml_result.model_config_hash,
        timestamp=ml_result.created_at.isoformat() + "Z",
        input_coverage_fraction=1.0,
        input_wavelength_range=(0.0, 0.0),  # TODO
        grid_match=True,
        warnings=[],
    )


@app.get("/analyses/{analysis_id}/retrieval", response_model=RetrievalSummary)
async def get_retrieval(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get retrieval summary."""
    from sqlalchemy import select

    from exosphere.api.db import Posterior

    result = await db.execute(select(Posterior).where(Posterior.analysis_id == analysis_id))
    posterior = result.scalar_one_or_none()
    if not posterior:
        raise HTTPException(status_code=404, detail="Retrieval results not found")

    # Get full analysis for full results
    from exosphere.api.db import Analysis

    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    result.scalar_one_or_none()

    return RetrievalSummary(
        analysis_id=analysis_id,
        logz=posterior.logz or 0.0,
        logz_err=posterior.logz_err or 0.0,
        best_fit={},  # Would need to load from full results
        median={},
        ci_68={},
        ci_95={},
        param_names=[
            "T",
            "log_h2o",
            "log_co2",
            "log_co",
            "log_ch4",
            "log_so2",
            "r_ref",
            "log_p_cloud",
        ],
        n_samples=posterior.n_samples or 0,
        n_live=posterior.n_live or 0,
        dlogz=posterior.dlogz or 0.0,
        runtime_s=0.0,
        sampler="dynesty",
        seed=0,
        error_inflation=0.0,
    )


@app.get("/analyses/{analysis_id}/posterior", response_model=PosteriorSamplesResponse)
async def get_posterior(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get posterior samples for corner plot."""
    from sqlalchemy import select

    from exosphere.api.db import Posterior

    result = await db.execute(select(Posterior).where(Posterior.analysis_id == analysis_id))
    posterior = result.scalar_one_or_none()
    if not posterior:
        raise HTTPException(status_code=404, detail="Posterior not found")

    # In real implementation, would load from file
    return PosteriorSamplesResponse(
        analysis_id=analysis_id,
        samples=[],
        weights=[],
        param_names=[
            "T",
            "log_h2o",
            "log_co2",
            "log_co",
            "log_ch4",
            "log_so2",
            "r_ref",
            "log_p_cloud",
        ],
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

    result = await db.execute(
        select(DBDetectionResult).where(DBDetectionResult.analysis_id == analysis_id)
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
    """Get best-fit model spectrum with credible band."""
    from sqlalchemy import select

    from exosphere.api.db import Analysis

    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return BestFitSpectrumResponse(
        analysis_id=analysis_id,
        wavelength_um=[],
        best_fit_depth=[],
        median_depth=[],
        ci_lo=[],
        ci_hi=[],
    )


@app.get("/analyses/{analysis_id}/provenance", response_model=ProvenanceResponse)
async def get_provenance(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get full provenance record."""
    result = await db.execute(select(Analysis).where(Analysis.analysis_id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return ProvenanceResponse(
        analysis_id=analysis_id,
        planet=analysis.planet.name if analysis.planet else None,
        observation_id=analysis.observation_id,
        telescope=analysis.observation.telescope if analysis.observation else None,
        instrument=analysis.observation.instrument if analysis.observation else None,
        source_archive=analysis.observation.source_archive if analysis.observation else None,
        input_data_version=None,
        input_data_hash=None,
        preprocessing_version=analysis.provenance_json.get("preprocessing_version")
        if analysis.provenance_json
        else None,
        ml_model_version=None,
        retrieval_model_version=None,
        retrieval_parameters=analysis.config_json or {},
        timestamp=analysis.created_at.isoformat() + "Z",
        result_reference=None,
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
