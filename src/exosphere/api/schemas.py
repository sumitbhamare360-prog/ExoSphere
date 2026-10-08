"""API request/response schemas with explicit units and labels."""

from __future__ import annotations

from datetime import datetime
from enum import Enum as PyEnum
from typing import Literal

from pydantic import BaseModel, Field

# --- Enums ---


class AnalysisStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class AnalysisStage(str, PyEnum):
    INIT = "init"
    LOADING = "loading"
    QUALITY = "quality"
    PREPROCESS = "preprocess"
    ML = "ml"
    RETRIEVAL = "retrieval"
    DETECTION = "detection"
    FINALIZE = "finalize"
    COMPLETED = "completed"


class DetectionStatus(str, PyEnum):
    DETECTED = "detected"
    TENTATIVE = "tentative"
    NOT_DETECTED = "not_detected"


class MoleculeName(str, PyEnum):
    H2O = "H2O"
    CO2 = "CO2"
    CO = "CO"
    CH4 = "CH4"
    SO2 = "SO2"


# --- Request/Response Models ---


class PlanetSearchResult(BaseModel):
    """Result from planet search."""

    name: str
    pl_rade: float | None = None
    pl_bmasse: float | None = None
    pl_orbper: float | None = None
    pl_eqt: float | None = None
    st_teff: float | None = None
    st_rad: float | None = None


class PlanetDetail(BaseModel):
    """Detailed planet information."""

    name: str
    pl_rade: float | None = None
    pl_bmasse: float | None = None
    pl_bmassprov: str | None = None
    pl_eqt: float | None = None
    pl_insol: float | None = None
    pl_orbper: float | None = None
    pl_orbsmax: float | None = None
    pl_orbeccen: float | None = None
    pl_orbincl: float | None = None
    pl_tranmid: float | None = None
    pl_ratror: float | None = None
    pl_ratdor: float | None = None
    st_rad: float | None = None
    st_teff: float | None = None
    st_mass: float | None = None
    st_logg: float | None = None
    st_met: float | None = None
    st_spectype: str | None = None
    surface_gravity_m_s2: float | None = None
    # Available observations
    observations: list[ObservationSummary] = []


class ObservationSummary(BaseModel):
    """Summary of an observation."""

    observation_id: str
    instrument: str
    facility: str
    wavelength_min_um: float | None = None
    wavelength_max_um: float | None = None
    num_points: int | None = None
    reference: str | None = None
    spectrum_file_path: str | None = None


class ObservationDetail(BaseModel):
    """Full observation metadata."""

    observation_id: str
    target_id: str
    instrument: str
    facility: str
    proposal_id: str | None = None
    filters: str | None = None
    wavelength_min_um: float | None = None
    wavelength_max_um: float | None = None
    num_points: int | None = None
    reference: str | None = None
    spectrum_file_path: str | None = None
    spectrum_file_hash: str | None = None


class AnalysisCreate(BaseModel):
    """Request to create a new analysis."""

    planet_name: str = Field(..., description="Planet name (e.g., WASP-39 b)")
    observation_id: str | None = Field(
        None, description="Observation ID (if not provided, uses best available)"
    )
    # Retrieval options
    n_live: int = Field(default=500, ge=50, le=2000)
    dlogz: float = Field(default=0.01, gt=0.0, le=1.0)
    max_iter: int = 50000
    seed: int = 42
    # Options
    run_ml: bool = True
    run_detection: bool = True
    run_quality: bool = True
    run_preprocess: bool = True
    error_inflation: float | None = None
    # Fixed parameters (optional overrides)
    fixed_params: dict = Field(default_factory=dict)


class AnalysisCreateResponse(BaseModel):
    """Response when creating an analysis."""

    analysis_id: str
    status: str
    message: str


class AnalysisStatusResponse(BaseModel):
    """Analysis status response."""

    analysis_id: str
    status: Literal["pending", "running", "completed", "failed", "interrupted"]
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    current_step: str
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime


class AnalysisDetail(BaseModel):
    """Full analysis detail."""

    analysis_id: str
    planet_name: str
    observation_id: str
    status: str
    stage: str
    progress: float
    current_step: str
    error_message: str | None = None
    config: dict = {}
    seed: int
    n_live: int
    dlogz: float
    max_iter: int
    seed: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime
    provenance: dict | None = None


class SpectrumResponse(BaseModel):
    """Spectrum data response."""

    wavelength_um: list[float] = Field(..., description="Wavelength in micrometers")
    transmission: list[float] = Field(..., description="Transit depth (fractional, (Rp/Rs)^2)")
    uncertainty: list[float] = Field(
        ..., description="1-sigma uncertainty, same units as transmission"
    )
    wavelength_bin_edges_um: list[float] = Field(..., description="Bin edges in micrometers")
    quality_flags: list[str] = []
    observation_id: str
    target_id: str
    instrument: str
    provenance: dict | None = None


class QualityReportResponse(BaseModel):
    """Data quality report."""

    analysis_id: str
    overall_suitability: Literal["GOOD", "LIMITED", "POOR"]
    median_snr: float | None
    max_band_snr: float | None
    wavelength_coverage_fraction: float
    flagged_fraction: float
    nan_fraction: float
    outlier_count: int
    outlier_fraction: float
    median_uncertainty: float | None
    uncertainty_non_positive_count: int
    uncertainty_nan_count: int
    uncertainty_huge_count: int
    uncertainty_tiny_count: int
    effective_resolving_power: float | None
    overall_suitability: str
    molecule_ratings: dict[str, str]  # molecule -> GOOD/LIMITED/POOR
    molecule_details: dict[str, dict]
    thresholds: dict
    config_version: str
    method: dict


class MLScoresResponse(BaseModel):
    """ML molecule candidate scores."""

    analysis_id: str
    scores: dict[str, float] = Field(..., description="Per-molecule ML candidate scores (0-1)")
    model_version: str
    dataset_hash: str
    model_config_hash: str
    timestamp: str
    input_coverage_fraction: float
    input_wavelength_range_um: tuple[float, float]
    grid_match: bool
    warnings: list[str] = []

    # Property to ensure proper labeling
    @property
    def labelled_scores(self) -> dict[str, str]:
        return {
            k: f"{v:.4f} (ML candidate score, not abundance, not a detection)"
            for k, v in self.scores.items()
        }


class RetrievalSummary(BaseModel):
    """Retrieval summary."""

    analysis_id: str
    logz: float
    logz_err: float
    best_fit: dict[str, float]
    median: dict[str, float]
    ci_68: dict[str, list[float]]
    ci_95: dict[str, list[float]]
    param_names: list[str]
    n_samples: int
    n_live: int
    dlogz: float
    runtime_s: float
    sampler: str
    seed: int
    error_inflation: float


class PosteriorSamplesResponse(BaseModel):
    """Posterior samples for corner plot."""

    analysis_id: str
    samples: list[list[float]]
    weights: list[float]
    param_names: list[str]
    logz: float
    logz_err: float


class DetectionResultItem(BaseModel):
    """Per-molecule detection result."""

    molecule: str
    ln_bayes_factor: float | None
    ln_bayes_factor_err: float | None
    sigma_equivalent: float | None
    status: Literal["detected", "tentative", "not_detected"]
    upper_limit_log_vmr: float | None
    details: dict = {}


class DetectionResultsResponse(BaseModel):
    """All detection results for an analysis."""

    analysis_id: str
    results: list[DetectionResultItem]
    summary: dict[str, str]  # molecule -> status


class BestFitSpectrumResponse(BaseModel):
    """Best-fit model spectrum with credible band."""

    analysis_id: str
    wavelength_um: list[float]
    best_fit_depth: list[float]
    median_depth: list[float]
    ci_lo: list[float]
    ci_hi: list[float]
    observed_wavelength_um: list[float] | None = None
    observed_depth: list[float] | None = None
    observed_uncertainty: list[float] | None = None


class ProvenanceResponse(BaseModel):
    """Full provenance record."""

    analysis_id: str
    planet: str | None = None
    observation_id: str | None = None
    telescope: str | None = None
    instrument: str | None = None
    source_archive: str | None = None
    input_data_version: str | None = None
    input_data_hash: str | None = None
    preprocessing_version: str | None = None
    ml_model_version: str | None = None
    retrieval_model_version: str | None = None
    retrieval_parameters: dict
    timestamp: str
    result_reference: str | None = None


class AnalysisListItem(BaseModel):
    """Analysis list item for history."""

    analysis_id: str
    planet_name: str
    observation_id: str
    status: str
    stage: str
    progress: float
    created_at: datetime
    completed_at: datetime | None = None


class PaginatedAnalyses(BaseModel):
    items: list[AnalysisListItem]
    total: int
    page: int
    page_size: int


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class TwinParameter(BaseModel):
    """One twin parameter with its provenance source tag."""

    name: str = Field(..., description="Machine-readable parameter name")
    label: str = Field(..., description="Human-readable label")
    value: float | None = Field(..., description="Parameter value (None when unavailable)")
    unit: str = Field(..., description="Explicit unit string")
    source: Literal["measured", "inferred", "derived", "assumed"] = Field(
        ..., description="Provenance source tag"
    )
    ci_68: list[float] | None = Field(
        default=None, description="68% credible interval [lo, hi] (inferred params only)"
    )
    constrained: bool | None = Field(
        default=None,
        description="Whether an inferred parameter is constrained (cloud-top pressure)",
    )
    note: str | None = Field(default=None, description="How the value was obtained")


class TwinMoleculeContribution(BaseModel):
    """Model-derived share of one molecule's in-band spectral signal."""

    molecule: str
    contribution_fraction: float = Field(
        ..., ge=0.0, le=1.0, description="Model-derived signal fraction in band windows"
    )
    in_band: bool = Field(..., description="Whether any band window overlaps the spectrum")
    note: str = Field(..., description="How the value was obtained")


class TwinParametersResponse(BaseModel):
    """Scientific digital twin parameters for one analysis."""

    analysis_id: str
    planet_name: str
    parameters: list[TwinParameter]
    molecules: list[TwinMoleculeContribution]
    meta: dict = Field(default_factory=dict)


class MoleculeBandsResponse(BaseModel):
    """Molecule absorption band windows + quality config version."""

    version: str
    wavelength_range_um: list[float]
    molecules: dict[str, list[list[float]]]


class ReportRequest(BaseModel):
    """Request a scientific report build (HTML is always available)."""

    format: Literal["html", "pdf"] = Field(
        default="html",
        description="Report format; 'pdf' needs WeasyPrint system libraries",
    )


class ReportResponse(BaseModel):
    """Metadata for a built scientific report."""

    analysis_id: str
    format: str
    file_path: str
    file_hash: str
    report_version: str
    created_at: datetime | None = None


# --- Helper to create analysis_id ---
def generate_analysis_id() -> str:
    """Generate next analysis ID in EXO-XXXXXX format."""
    import asyncio

    from sqlalchemy import func, select

    from exosphere.api.db import Analysis, async_session_maker

    async def _get_next_id() -> str:
        async with async_session_maker() as session:
            result = await session.execute(select(func.max(Analysis.id)))
            max_id = result.scalar() or 0
            return f"EXO-{max_id + 1:06d}"

    return asyncio.run(_get_next_id())


# Update forward references
ObservationSummary.model_rebuild()
ObservationDetail.model_rebuild()
