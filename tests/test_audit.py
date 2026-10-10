"""Phase 9b tests: scope audit and provenance persistence."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Point the API at a throwaway database BEFORE importing exosphere.api modules.
_TMPDIR = tempfile.mkdtemp(prefix="exosphere-audit-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMPDIR) / 'audit_test.db'}"

# scripts/ holds the audit tool (namespace import via repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(scope="module", autouse=True)
def _isolated_database():
    """Wipe the shared test database once: all DB modules share one engine."""
    from conftest import reset_database

    reset_database()


def test_audit_scans_pass_on_repo():
    import scripts.audit_scope as audit

    assert audit.check_molecules() == []
    assert audit.check_ml_labels() == []
    assert audit.check_no_geography() == []
    assert audit.check_ui_labels() == []


def test_audit_provenance_missing_db_skipped(tmp_path):
    import scripts.audit_scope as audit

    missing = tmp_path / "nope.db"
    assert audit.check_provenance(str(missing)) == ["db missing: skipped"]


def test_audit_main_returns_zero_when_clean(monkeypatch, tmp_path):
    import scripts.audit_scope as audit

    monkeypatch.setattr(audit, "check_provenance", lambda db_path: [])
    assert audit.main(["--db", str(tmp_path / "nope.db")]) == 0


def test_pipeline_writes_provenance():
    """_finalize persists the AGENTS.md section 5 record."""
    from exosphere.api.db import Analysis, AnalysisStage, AnalysisStatus, async_session_maker
    from exosphere.pipeline import Pipeline, PipelineOptions

    async def _seed() -> int:
        from exosphere.api.db import init_db

        await init_db()
        async with async_session_maker() as session:
            async with session.begin():
                analysis = Analysis(
                    analysis_id="EXO-000099",
                    planet_id=1,
                    observation_id=1,
                    status=AnalysisStatus.RUNNING,
                    stage=AnalysisStage.FINALIZE,
                    seed=7,
                )
                # Detached planet/observation FKs: row references nothing, but
                # _write_provenance tolerates missing relations.
                session.add(analysis)
                await session.flush()
                return analysis.id

    async def _read() -> dict:
        async with async_session_maker() as session:
            from sqlalchemy import select

            analysis = (
                await session.execute(
                    select(Analysis).where(Analysis.analysis_id == "EXO-000099")
                )
            ).scalar_one()
            return dict(analysis.provenance_json or {})

    asyncio.run(_seed())
    pipe = Pipeline("EXO-000099", PipelineOptions(seed=7, run_detection=False))
    pipe.state.planet_name = "TEST-1 b"
    asyncio.run(pipe._write_provenance())
    stored = asyncio.run(_read())
    assert stored["analysis_id"] == "EXO-000099"
    assert stored["planet"] == "TEST-1 b"
    assert stored["retrieval_parameters"]["seed"] == 7
    assert stored["timestamp"]
