"""Phase 9b tests: structured logging and clean API errors."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path

# Point the API at a throwaway database BEFORE importing exosphere.api modules.
_TMPDIR = tempfile.mkdtemp(prefix="exosphere-logging-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMPDIR) / 'logging_test.db'}"

from exosphere.core.logging import configure_logging, get_logger  # noqa: E402
from exosphere.pipeline import Pipeline, PipelineOptions  # noqa: E402


def test_logger_binds_analysis_id(caplog):
    configure_logging()
    log = get_logger("exosphere.test", "EXO-000001")
    with caplog.at_level(logging.INFO, logger="exosphere.test"):
        log.info("hello")
    assert any(
        getattr(record, "analysis_id", None) == "EXO-000001" for record in caplog.records
    )


def test_pipeline_progress_logs_analysis_id(caplog):
    pipeline = Pipeline("EXO-000099", PipelineOptions(run_detection=False))
    with caplog.at_level(logging.INFO, logger="exosphere.pipeline"):
        pipeline._update_progress("quality", 0.2, "testing")
    assert any(
        getattr(record, "analysis_id", None) == "EXO-000099" for record in caplog.records
    )


def test_unhandled_errors_return_clean_json():
    import json

    from fastapi import Request

    from exosphere.api.app import _unhandled_exception_handler

    async def _call():
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
        return await _unhandled_exception_handler(request, RuntimeError("boom"))

    response = asyncio.run(_call())
    assert response.status_code == 500
    body = json.loads(response.body.decode())
    assert body == {"detail": "Internal server error"}
    assert "Traceback" not in json.dumps(body)


def test_http_errors_stay_clean_json():
    _seed()
    from fastapi.testclient import TestClient

    from exosphere.api.app import app

    with TestClient(app) as client:
        resp = client.get("/analyses/EXO-999999/quality")
        assert resp.status_code == 404
        assert "detail" in resp.json()
        assert "Traceback" not in resp.text


def _seed() -> None:
    from exosphere.api.db import async_session_maker, init_db

    async def _run():
        await init_db()
        async with async_session_maker():
            pass

    asyncio.run(_run())
