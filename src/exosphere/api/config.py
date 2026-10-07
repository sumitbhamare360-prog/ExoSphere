"""API configuration settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="sqlite:///./exosphere.db",
        description="Database URL (SQLite for dev, Postgres for prod)",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins",
    )

    # Background jobs
    max_concurrent_jobs: int = Field(
        default=1,
        description="Max concurrent retrieval jobs (CPU heavy)",
    )
    job_poll_interval: float = 5.0

    # File storage
    data_cache_dir: Path = Path("data_cache")
    spectra_store_dir: Path = Path("data_cache/store")

    # Retrieval defaults
    default_n_live: int = 500
    default_dlogz: float = 0.01
    max_iter: int = 50000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
