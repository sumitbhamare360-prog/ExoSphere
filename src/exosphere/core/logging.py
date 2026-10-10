"""Structured logging for ExoSphere (Phase 9b hardening).

Every record carries timestamp, level, logger name and message; pipeline and
retrieval logs additionally bind ``analysis_id`` so runs can be traced.
API responses never include stack traces (see ``api.app`` handlers); tracebacks
go to the server log only.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


class AnalysisAdapter(logging.LoggerAdapter):
    """Logger adapter that injects ``analysis_id`` into every record."""

    def __init__(self, logger: logging.Logger, analysis_id: str) -> None:
        super().__init__(logger, {"analysis_id": analysis_id})

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        merged = dict(kwargs.get("extra") or {})
        merged.update(self.extra)
        kwargs["extra"] = merged
        return msg, kwargs


class _ContextFilter(logging.Filter):
    """Guarantee the ``analysis_id`` field exists on every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "analysis_id"):
            record.analysis_id = "-"
        return True


_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root handler once (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    chosen = (level or os.environ.get("EXOSPHERE_LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] [analysis=%(analysis_id)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.addFilter(_ContextFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(getattr(logging, chosen, logging.INFO))
    # Third-party samplers are chatty; keep them at WARNING by default.
    for noisy in ("dynesty", "matplotlib", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str, analysis_id: str | None = None) -> logging.Logger | AnalysisAdapter:
    """Module logger, optionally bound to an analysis id."""
    configure_logging()
    logger = logging.getLogger(name)
    if analysis_id is not None:
        return AnalysisAdapter(logger, analysis_id)
    return logger
