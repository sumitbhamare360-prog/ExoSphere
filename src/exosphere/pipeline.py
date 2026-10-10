"""Pipeline orchestration for ExoSphere analysis."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from exosphere.api.db import (
    Analysis,
    DetectionResult,
    Posterior,
    SpectrumFile,
    async_session_maker,
)
from exosphere.api.db import (
    MLResult as DBMLResult,
)
from exosphere.api.db import (
    QualityReport as DBQualityReport,
)
from exosphere.core.config import load_config
from exosphere.core.spectrum import Spectrum
from exosphere.data.exoarchive import get_planet_params
from exosphere.forward.model import PlanetFixed
from exosphere.ml.infer import MLClassifier
from exosphere.preprocess.clean import CleanOptions, clean
from exosphere.quality.assess import assess
from exosphere.retrieval.detection import detection_summary
from exosphere.retrieval.samplers import SamplerConfig, run_dynesty


@dataclass
class PipelineOptions:
    """Options controlling pipeline execution."""

    run_quality: bool = True
    run_preprocess: bool = True
    run_ml: bool = True
    run_retrieval: bool = True
    run_detection: bool = True
    run_quality_check: bool = True
    n_live: int = 500
    dlogz: float = 0.01
    max_iter: int = 50000
    maxcall: int | None = None
    seed: int = 42
    error_inflation: float = 0.0
    error_inflation_free: bool = False
    fixed_params: dict = field(default_factory=dict)


@dataclass
class PipelineState:
    """Current state of pipeline execution."""

    analysis_id: str
    stage: str = "init"
    progress: float = 0.0
    current_step: str = ""
    error_message: str | None = None
    started_at: float | None = None
    stage_times: dict = field(default_factory=dict)
    spectrum: Any | None = None
    quality_report: Any | None = None
    cleaned_spectrum: Any | None = None
    ml_result: Any | None = None
    retrieval_result: Any | None = None
    detection_results: dict | None = None
    fixed: Any | None = None
    sampler_config: Any | None = None


class ProgressCallback:
    """Callback for progress updates."""

    def __init__(self, update_func: Callable[[str, float, str], Any]):
        self.update_func = update_func

    def __call__(self, stage: str, progress: float, step: str):
        self.update_func(stage, progress, step)


class Pipeline:
    """Main analysis pipeline orchestrator."""

    def __init__(
        self,
        analysis_id: str,
        options: PipelineOptions,
        progress_callback: Callable[[str, float, str], Any] | None = None,
    ):
        self.analysis_id = analysis_id
        self.options = options
        self.progress_callback = None
        if progress_callback:
            self.progress_callback = ProgressCallback(progress_callback)
        self.state = PipelineState(analysis_id=analysis_id)
        self._start_time = time.time()

    def _update_progress(self, stage: str, progress: float, step: str, error: str | None = None):
        """Update pipeline progress."""
        self.state.stage = stage
        self.state.progress = progress
        self.state.current_step = step
        if error:
            self.state.error_message = error
        if self.progress_callback:
            self.progress_callback(stage, progress, step)

    def _record_stage_time(self, stage: str):
        """Record elapsed time for a stage."""
        now = time.time()
        if self.state.started_at:
            elapsed = now - self.state.started_at
        else:
            elapsed = 0
        self.state.stage_times[stage] = elapsed

    async def _update_db_status(
        self,
        status: str,
        stage: str,
        progress: float,
        step: str,
        error: str | None = None,
    ):
        """Update analysis status in database."""
        from datetime import datetime

        async with async_session_maker() as session:
            async with session.begin():
                result = await session.execute(
                    select(Analysis).where(Analysis.analysis_id == self.analysis_id)
                )
                analysis = result.scalar_one_or_none()
                if analysis:
                    analysis.status = status
                    analysis.stage = stage
                    analysis.progress = progress
                    analysis.current_step = step
                    if error:
                        analysis.error_message = error
                    if status == "running" and not analysis.started_at:
                        analysis.started_at = datetime.utcnow()
                    if status in ("completed", "failed", "interrupted"):
                        analysis.completed_at = datetime.utcnow()
                    await session.commit()

    async def run(
        self,
        planet_name: str,
        observation_ref: str,
        spectrum_path: str | None = None,
        fixed_params: dict | None = None,
    ) -> dict:
        """Run the full analysis pipeline."""
        self.state.started_at = time.time()
        await self._update_db_status("running", "init", 0.0, "Initializing analysis")

        try:
            # Stage 1: Load planet parameters
            self._update_progress("loading", 0.05, "Loading planet parameters")
            await self._update_db_status("running", "loading", 0.05, "Loading planet parameters")

            self.state.fixed = await self._load_planet_params(planet_name)

            # Stage 2: Load spectrum
            self._update_progress("loading", 0.1, "Loading spectrum")

            spectrum = await self._load_spectrum(observation_ref)
            self.state.spectrum = spectrum

            # Stage 3: Quality assessment
            if self.options.run_quality_check:
                self._update_progress("quality", 0.2, "Assessing data quality")
                quality_report = await self._run_quality(spectrum)
                self.state.quality_report = quality_report
                await self._save_quality_report(quality_report)

            # Stage 4: Preprocessing
            if self.options.run_preprocess:
                self._update_progress("preprocess", 0.3, "Preprocessing spectrum")
                cleaned = await self._run_preprocess(spectrum)
                self.state.cleaned_spectrum = cleaned
                await self._save_cleaned_spectrum(cleaned)
            else:
                self.state.cleaned_spectrum = self.state.spectrum

            # Stage 5: ML inference
            if self.options.run_ml:
                self._update_progress("ml", 0.4, "Running ML inference")
                ml_result = await self._run_ml_inference(self.state.cleaned_spectrum)
                self.state.ml_result = ml_result
                await self._save_ml_result(ml_result)

            # Stage 6: Retrieval
            if self.options.run_retrieval:
                self._update_progress("retrieval", 0.5, "Running Bayesian retrieval")
                retrieval_result = await self._run_retrieval(
                    self.state.cleaned_spectrum,
                    fixed_params=None,
                )
                self.state.retrieval_result = retrieval_result
                await self._save_retrieval_result(retrieval_result)

            # Stage 7: Detection
            if self.options.run_detection and self.state.retrieval_result:
                self._update_progress("detection", 0.85, "Running detection analysis")
                detection_results = await self._run_detection()
                self.state.detection_results = detection_results
                await self._save_detection_results(detection_results)

            # Stage 8: Finalize
            self._update_progress("finalize", 0.95, "Finalizing results")
            await self._finalize()

            self._update_progress("completed", 1.0, "Analysis completed")

            return self._build_result()

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self._update_progress("failed", 0.0, error_msg, str(e))
            await self._update_db_status("failed", "failed", 0.0, "Pipeline failed", str(e))
            raise

    def _update_progress(self, stage: str, progress: float, step: str, error: str | None = None):
        self._record_stage_time(stage)
        if self.progress_callback:
            self.progress_callback(stage, progress, step)

    def _record_stage_time(self, stage: str):
        now = time.time()
        if self.state.started_at:
            elapsed = now - self.state.started_at
        else:
            elapsed = 0
        self.state.stage_times[stage] = elapsed

    async def _load_planet_params(self, planet_name: str):
        """Load fixed planet parameters, preferring the stored catalog row.

        Falls back to the NASA Exoplanet Archive (network) only when the
        database row is missing or lacks gravity/stellar radius.
        """
        from sqlalchemy.orm import selectinload

        gravity: float | None = None
        stellar_radius: float | None = None
        async with async_session_maker() as session:
            result = await session.execute(
                select(Analysis)
                .options(selectinload(Analysis.planet))
                .where(Analysis.analysis_id == self.analysis_id)
            )
            analysis = result.scalar_one_or_none()
            planet = analysis.planet if analysis is not None else None
            if planet is not None:
                gravity = planet.surface_gravity_m_s2
                stellar_radius = planet.st_rad
        if gravity is None or stellar_radius is None:
            planet_params = get_planet_params(planet_name)
            gravity = planet_params.surface_gravity_m_s2
            stellar_radius = planet_params.stellar_radius_solar
        return PlanetFixed(
            gravity_m_s2=float(gravity),
            stellar_radius_rsun=float(stellar_radius),
            reference_pressure_bar=0.01,
        )

    async def _load_spectrum(self, observation_ref: str) -> Spectrum:
        """Load spectrum from file or archive."""
        # Check if it's a file path
        path = Path(observation_ref)
        if path.exists():
            return Spectrum.load(path)

        # Try to find in benchmark (observation ids use "/" separators while
        # cached files flatten them to "_")
        bench_dir = Path("data_cache/benchmark")
        for pattern in (f"*{observation_ref}*.npz", f"*{observation_ref.replace('/', '_')}*.npz"):
            matches = sorted(bench_dir.glob(pattern))
            if matches:
                return Spectrum.load(matches[0])

        # Check cache
        cache_path = Path("data_cache") / observation_ref
        if cache_path.exists():
            return Spectrum.load(cache_path)

        raise FileNotFoundError(f"Spectrum not found: {observation_ref}")

    async def _run_quality(self, spectrum: Spectrum) -> Any:
        """Run quality assessment."""
        return assess(spectrum)

    async def _run_preprocess(self, spectrum: Spectrum) -> Spectrum:
        """Run preprocessing and persist the PreprocessLog for the report."""
        options = CleanOptions(
            drop_nan=True,
            drop_flagged=True,
            drop_bad_uncertainty=True,
            sigma_clip=True,
            clip_n_sigma=5.0,
            rebin=None,
        )
        cleaned, log = clean(spectrum, options)
        config = load_config()
        log_path = config.data_cache_dir / "preprocess_logs" / f"{self.analysis_id}.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(log.to_json(), encoding="utf-8")
        return cleaned

    async def _run_ml_inference(self, spectrum: Any) -> Any:
        """Run ML inference."""
        # Find model checkpoint
        model_dir = Path("models")
        checkpoints = list(model_dir.glob("*.pt"))
        if not checkpoints:
            # Return mock result if no model
            from exosphere.ml.infer import MLResult

            return MLResult(
                scores={mol: 0.0 for mol in ["H2O", "CO2", "CO", "CH4", "SO2"]},
                model_version="CNN-v1",
                dataset_hash="unknown",
                model_config_hash="",
                timestamp=datetime.utcnow().isoformat() + "Z",
                input_coverage_fraction=1.0,
                input_wavelength_range=(
                    float(self.state.spectrum.wavelength[0]),
                    float(self.state.spectrum.wavelength[-1]),
                ),
                grid_match=True,
                warnings=["No trained model available; scores are all-zero placeholders"],
            )

        # Load classifier
        classifier = MLClassifier(str(checkpoints[-1]))
        return classifier.predict(self.state.cleaned_spectrum)

    async def _run_retrieval(
        self,
        spectrum: Spectrum,
        fixed_params: dict | None = None,
    ) -> Any:
        """Run Bayesian retrieval with the pipeline sampler settings."""
        config = SamplerConfig(
            n_live=self.options.n_live,
            dlogz=self.options.dlogz,
            max_iter=self.options.max_iter,
            maxcall=self.options.maxcall,
            seed=self.options.seed,
            error_inflation=self.options.error_inflation,
            checkpoint_path=str(
                load_config().data_cache_dir / "checkpoints" / f"{self.analysis_id}.pkl"
            ),
        )
        Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        self.state.sampler_config = config
        result = await asyncio.to_thread(
            run_dynesty,
            self.state.cleaned_spectrum,
            self.state.fixed,
            config,
            self.options.seed,
        )
        return result

    async def _run_detection(self) -> dict:
        """Run per-molecule detection with the same fixed/config/seed as retrieval."""
        if not self.state.retrieval_result:
            return {}

        checkpoint_dir = load_config().data_cache_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return detection_summary(
            self.state.retrieval_result,
            self.state.cleaned_spectrum,
            self.state.fixed,
            self.state.sampler_config,
            self.options.seed,
            checkpoint_dir=str(checkpoint_dir),
        )

    async def _analysis_pk(self) -> int:
        """Integer primary key of this analysis row (related tables key off it)."""
        async with async_session_maker() as session:
            result = await session.execute(
                select(Analysis).where(Analysis.analysis_id == self.analysis_id)
            )
            analysis = result.scalar_one()
            return analysis.id

    async def _save_quality_report(self, report):
        analysis_pk = await self._analysis_pk()
        async with async_session_maker() as session:
            async with session.begin():
                db_report = DBQualityReport(
                    analysis_id=analysis_pk,
                    report_json=report.model_dump(mode="json"),
                )
                session.add(db_report)

    async def _save_cleaned_spectrum(self, spectrum: Spectrum):
        """Save cleaned spectrum to file and DB."""
        config = load_config()
        output_path = config.data_cache_dir / "cleaned" / f"{self.analysis_id}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spectrum.save(output_path)

        # Save to DB
        from sqlalchemy import select

        async with async_session_maker() as session:
            async with session.begin():
                result = await session.execute(
                    select(Analysis).where(Analysis.analysis_id == self.analysis_id)
                )
                analysis = result.scalar_one_or_none()
                spectrum_file = SpectrumFile(
                    observation_id=analysis.observation_id if analysis else None,
                    file_path=str(output_path),
                    file_hash=hashlib.sha256(open(output_path, "rb").read()).hexdigest(),
                    wavelength_min_um=float(min(spectrum.wavelength)),
                    wavelength_max_um=float(max(spectrum.wavelength)),
                    num_points=len(spectrum.wavelength),
                )
                session.add(spectrum_file)
                await session.flush()

                # Update analysis with spectrum file ID
                if analysis:
                    analysis.spectrum_file_id = spectrum_file.id

    async def _save_ml_result(self, result: Any):
        analysis_pk = await self._analysis_pk()
        details: dict = {}
        if hasattr(result, "to_dict"):
            try:
                details = dict(result.to_dict())
            except Exception:
                details = {}
        async with async_session_maker() as session:
            async with session.begin():
                db_result = DBMLResult(
                    analysis_id=analysis_pk,
                    scores_json=result.scores,
                    model_version=result.model_version,
                    dataset_hash=result.dataset_hash,
                    model_config_hash=result.model_config_hash,
                    details_json=details,
                )
                session.add(db_result)

    async def _save_retrieval_result(self, result: Any):
        import hashlib

        from exosphere.core.config import load_config

        config = load_config()
        output_path = config.data_cache_dir / "posteriors" / f"{self.analysis_id}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_npz(output_path)
        file_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
        param_names = list(result.param_names)
        summary = {
            "median": {
                name: float(value)
                for name, value in zip(param_names, result.median, strict=True)
            },
            "ci_68": {
                name: [float(lo), float(hi)]
                for name, (lo, hi) in zip(param_names, result.ci_68, strict=True)
            },
            "best_fit": {
                name: float(value)
                for name, value in zip(param_names, result.best_fit, strict=True)
            },
            "param_names": param_names,
        }
        async with async_session_maker() as session:
            async with session.begin():
                posterior = Posterior(
                    analysis_id=await self._analysis_pk(),
                    file_path=str(output_path),
                    file_hash=file_hash,
                    logz=result.logz,
                    logz_err=result.logz_err,
                    n_samples=result.n_samples,
                    n_live=self.options.n_live,
                    dlogz=self.options.dlogz,
                    summary_json=summary,
                )
                session.add(posterior)

    async def _save_detection_results(self, results: dict):
        analysis_pk = await self._analysis_pk()
        async with async_session_maker() as session:
            async with session.begin():
                for mol, info in results.items():
                    det = DetectionResult(
                        analysis_id=analysis_pk,
                        molecule=mol,
                        ln_bayes_factor=info.get("ln_B"),
                        ln_bayes_factor_err=info.get("ln_B_err"),
                        sigma_equivalent=info.get("sigma"),
                        status=info.get("status", "not_detected"),
                        upper_limit_log_vmr=info.get("upper_limit"),
                        details_json=info,
                    )
                    session.add(det)

    async def _finalize(self):
        """Generate final outputs and reports."""
        # Generate corner plot
        # Generate best-fit spectrum
        # Generate report
        await self._update_db_status("completed", "completed", 1.0, "Analysis completed")

    def _build_result(self) -> dict:
        """Build final result dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "status": "completed",
            "stage": "completed",
            "progress": 1.0,
        }


async def run_analysis(
    planet_name: str,
    observation_ref: str,
    options: PipelineOptions,
    progress_callback: Callable[[str, float, str], Any] | None = None,
) -> dict:
    """Convenience function to run analysis."""
    analysis_id = f"EXO-{int(time.time() * 1000) % 1000000:06d}"
    pipeline = Pipeline(analysis_id, options, progress_callback)
    return await pipeline.run(planet_name, observation_ref)


# Convenience function for simple usage
async def quick_analysis(
    planet_name: str,
    observation_ref: str,
    n_live: int = 500,
    dlogz: float = 0.01,
    seed: int = 42,
) -> dict:
    """Run a quick analysis with default options."""
    options = PipelineOptions(
        n_live=n_live,
        dlogz=dlogz,
        seed=seed,
    )
    return await run_analysis(planet_name, observation_ref, options)
