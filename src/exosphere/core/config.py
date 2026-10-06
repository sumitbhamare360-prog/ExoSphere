"""Project paths and optional external paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PRT_INPUT_DATA_PATH_ENV_VAR = "PRT_INPUT_DATA_PATH"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"


@dataclass(frozen=True)
class Config:
    """Resolved project paths."""

    data_cache_dir: Path
    prt_input_data_path: Path | None


def load_config() -> Config:
    """Load paths: data cache with a sensible default, pRT opacities from env var."""
    raw_prt_path = os.environ.get(PRT_INPUT_DATA_PATH_ENV_VAR, "").strip()
    prt_input_data_path = Path(raw_prt_path).expanduser() if raw_prt_path else None
    return Config(
        data_cache_dir=DEFAULT_DATA_CACHE_DIR,
        prt_input_data_path=prt_input_data_path,
    )
