"""Database models and session management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from exosphere.api.config import settings
from exosphere.core.provenance import Provenance


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class AnalysisStatus(str, PyEnum):
    """Status of an analysis job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class AnalysisStage(str, PyEnum):
    """Current stage of the analysis pipeline."""

    INIT = "init"
    LOADING = "loading"
    QUALITY = "quality"
    PREPROCESS = "preprocess"
    ML = "ml"
    RETRIEVAL = "retrieval"
    DETECTION = "detection"
    FINALIZE = "finalize"
    COMPLETED = "completed"


class Planet(Base):
    """Exoplanet catalog entry."""

    __tablename__ = "planets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    # Catalog parameters (from NASA Exoplanet Archive)
    pl_rade: Mapped[float | None] = mapped_column(nullable=True)  # Earth radii
    pl_bmasse: Mapped[float | None] = mapped_column(nullable=True)  # Earth masses
    pl_bmassprov: Mapped[str | None] = mapped_column(nullable=True)
    pl_eqt: Mapped[float | None] = mapped_column(nullable=True)  # K
    pl_insol: Mapped[float | None] = mapped_column(nullable=True)
    pl_orbper: Mapped[float | None] = mapped_column(nullable=True)  # days
    pl_orbsmax: Mapped[float | None] = mapped_column(nullable=True)  # AU
    pl_orbeccen: Mapped[float | None] = mapped_column(nullable=True)
    pl_orbincl: Mapped[float | None] = mapped_column(nullable=True)  # deg
    pl_tranmid: Mapped[float | None] = mapped_column(nullable=True)  # BJD
    pl_ratror: Mapped[float | None] = mapped_column(nullable=True)
    pl_ratdor: Mapped[float | None] = mapped_column(nullable=True)
    st_rad: Mapped[float | None] = mapped_column(nullable=True)  # Solar radii
    st_teff: Mapped[float | None] = mapped_column(nullable=True)  # K
    st_mass: Mapped[float | None] = mapped_column(nullable=True)  # Solar masses
    st_logg: Mapped[float | None] = mapped_column(nullable=True)  # cgs
    st_met: Mapped[float | None] = mapped_column(nullable=True)  # [Fe/H]
    st_spectype: Mapped[str | None] = mapped_column(nullable=True)
    # Derived
    surface_gravity_m_s2: Mapped[float | None] = mapped_column(nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    observations: Mapped[list[Observation]] = relationship(
        back_populates="planet", cascade="all, delete-orphan"
    )
    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="planet", cascade="all, delete-orphan"
    )


class Observation(Base):
    """Observation metadata (from MAST or literature)."""

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    planet_id: Mapped[int] = mapped_column(ForeignKey("planets.id", ondelete="CASCADE"), index=True)
    observation_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    target_id: Mapped[str] = mapped_column(String(100))
    instrument: Mapped[str] = mapped_column(String(100))
    telescope: Mapped[str] = mapped_column(String(100), default="JWST")
    source_archive: Mapped[str] = mapped_column(String(50), default="MAST")
    # Spectrum metadata
    wavelength_min_um: Mapped[float | None] = mapped_column(nullable=True)
    wavelength_max_um: Mapped[float | None] = mapped_column(nullable=True)
    num_points: Mapped[int | None] = mapped_column(nullable=True)
    proposal_id: Mapped[str | None] = mapped_column(nullable=True)
    filters: Mapped[str | None] = mapped_column(nullable=True)
    t_min: Mapped[float | None] = mapped_column(nullable=True)
    # Data file reference
    spectrum_file_path: Mapped[str | None] = mapped_column(nullable=True)
    spectrum_file_hash: Mapped[str | None] = mapped_column(nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    planet: Mapped[Planet] = relationship(back_populates="observations")
    spectra: Mapped[list[SpectrumFile]] = relationship(
        back_populates="observation", cascade="all, delete-orphan"
    )
    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="observation", cascade="all, delete-orphan"
    )


class SpectrumFile(Base):
    """Spectrum data file stored on disk."""

    __tablename__ = "spectrum_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observations.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    # Metadata
    wavelength_min_um: Mapped[float]
    wavelength_max_um: Mapped[float]
    num_points: Mapped[int]
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    observation: Mapped[Observation] = relationship(back_populates="spectra")


class Analysis(Base):
    """Analysis job record."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(
        String(20), unique=True, index=True
    )  # EXO-000001 format
    planet_id: Mapped[int] = mapped_column(ForeignKey("planets.id", ondelete="CASCADE"), index=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("observations.id", ondelete="CASCADE"), index=True
    )
    # Status tracking
    status: Mapped[AnalysisStatus] = mapped_column(
        Enum(AnalysisStatus), default=AnalysisStatus.PENDING, index=True
    )
    stage: Mapped[AnalysisStage] = mapped_column(Enum(AnalysisStage), default=AnalysisStage.INIT)
    progress: Mapped[float] = mapped_column(default=0.0)  # 0.0 to 1.0
    current_step: Mapped[str] = mapped_column(String(100), default="")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Configuration
    seed: Mapped[int] = mapped_column(default=42)
    config_json: Mapped[dict] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON,
        default={},
    )
    # Provenance (stored as JSON)
    provenance_json: Mapped[dict] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON,
        default={},
    )
    # Results references
    spectrum_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("spectrum_files.id", ondelete="SET NULL"), nullable=True
    )
    posterior_file_path: Mapped[str | None] = mapped_column(nullable=True)
    report_file_path: Mapped[str | None] = mapped_column(nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    planet: Mapped[Planet] = relationship(back_populates="analyses")
    observation: Mapped[Observation] = relationship(back_populates="analyses")
    spectrum_file: Mapped[SpectrumFile | None] = relationship()
    posteriors: Mapped[list[Posterior]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    ml_results: Mapped[list[MLResult]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    quality_reports: Mapped[list[QualityReport]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    detection_results: Mapped[list[DetectionResult]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )

    @property
    def provenance(self) -> Provenance | None:
        if self.provenance_json:
            return Provenance.model_validate(self.provenance_json)
        return None

    @provenance.setter
    def provenance(self, value: Provenance | None):
        if value is None:
            self.provenance_json = {}
        else:
            self.provenance_json = value.model_dump(mode="json")


class Posterior(Base):
    """Posterior samples file reference."""

    __tablename__ = "posteriors"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64))
    # Summary statistics
    logz: Mapped[float | None] = mapped_column(nullable=True)
    logz_err: Mapped[float | None] = mapped_column(nullable=True)
    n_samples: Mapped[int | None] = mapped_column(nullable=True)
    n_live: Mapped[int | None] = mapped_column(nullable=True)
    dlogz: Mapped[float | None] = mapped_column(nullable=True)
    # Retrieval summary: median / ci_68 / best_fit per parameter + param_names
    summary_json: Mapped[dict | None] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON,
        nullable=True,
    )
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped[Analysis] = relationship(back_populates="posteriors")


class MLResult(Base):
    """ML classification results."""

    __tablename__ = "ml_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    # Per-molecule scores (0-1)
    scores_json: Mapped[dict] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON
    )
    # Metadata
    model_version: Mapped[str] = mapped_column(String(50))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    model_config_hash: Mapped[str] = mapped_column(String(64))
    # Full MLResult.to_dict() (coverage, grid match, warnings)
    details_json: Mapped[dict] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON,
        default={},
    )
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped[Analysis] = relationship(back_populates="ml_results")


class QualityReport(Base):
    """Data quality assessment report."""

    __tablename__ = "quality_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    # Full report as JSON
    report_json: Mapped[dict] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON
    )
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped[Analysis] = relationship(back_populates="quality_reports")


class DetectionResult(Base):
    """Per-molecule detection results."""

    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    molecule: Mapped[str] = mapped_column(String(20), index=True)
    ln_bayes_factor: Mapped[float | None] = mapped_column(nullable=True)
    ln_bayes_factor_err: Mapped[float | None] = mapped_column(nullable=True)
    sigma_equivalent: Mapped[float | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(20))  # detected/tentative/not_detected
    upper_limit_log_vmr: Mapped[float | None] = mapped_column(nullable=True)
    # Full nested model comparison results
    details_json: Mapped[dict] = mapped_column(
        SQLiteJSON if "sqlite" in settings.database_url else JSON,
        default={},
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped[Analysis] = relationship(back_populates="detection_results")


class Report(Base):
    """Generated report files."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    report_type: Mapped[str] = mapped_column(String(50))  # corner, spectrum, summary
    file_path: Mapped[str] = mapped_column(String(500))
    file_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    analysis: Mapped[Analysis] = relationship(back_populates="reports")


# Database engine and session
engine = create_async_engine(
    settings.database_url.replace("sqlite://", "sqlite+aiosqlite://"),
    echo=False,
    pool_pre_ping=True,
)

async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncSession:
    """FastAPI dependency for database session."""
    async with async_session_maker() as session:
        yield session


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    await engine.dispose()
