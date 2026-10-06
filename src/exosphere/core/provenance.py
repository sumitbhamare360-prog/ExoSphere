"""Provenance record per analysis (AGENTS.md section 5)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

ANALYSIS_ID_PATTERN = re.compile(r"^EXO-\d{6}$")
_SEQUENCE_MAX = 999_999


def generate_analysis_id(sequence: int) -> str:
    """Format a sequence number as an analysis id, e.g. 1 -> 'EXO-000001'."""
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        raise TypeError(f"sequence must be an int, got {type(sequence).__name__}")
    if not 0 <= sequence <= _SEQUENCE_MAX:
        raise ValueError(f"sequence must be in [0, {_SEQUENCE_MAX}], got {sequence}")
    return f"EXO-{sequence:06d}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class AnalysisIdGenerator:
    """Produces sequential analysis ids: EXO-000001, EXO-000002, ..."""

    def __init__(self, start: int = 1) -> None:
        generate_analysis_id(start)
        self._next = start

    def next_id(self) -> str:
        analysis_id = generate_analysis_id(self._next)
        self._next += 1
        return analysis_id


class Provenance(BaseModel):
    """Full provenance record; JSON serializable via model_dump/model_validate."""

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
    retrieval_parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=_utc_now)
    result_reference: str | None = None

    @field_validator("analysis_id")
    @classmethod
    def _check_analysis_id(cls, value: str) -> str:
        if not ANALYSIS_ID_PATTERN.match(value):
            raise ValueError(f"analysis_id must match EXO-000001 format, got {value!r}")
        return value
