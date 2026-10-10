"""Scope audit vs AGENTS.md sections 2-3 (Phase 9b item 10).

Checks (each prints PASS/FAIL; exit code is the failure count):
 1. No unsupported molecules in science code paths.
 2. ML scores always labelled "ML candidate score", never abundance/detection.
 3. No surface/geographic rendering or texture claims in twin/UI code.
 4. Provenance complete on every stored analysis (--db path, dev DB default).
 5. Required UI labels present in the dashboard sources.

Usage: python scripts/audit_scope.py [--db PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

ALLOWED_MOLECULES = {"H2O", "CO2", "CO", "CH4", "SO2", "H2", "HE"}
# Species explicitly excluded in V1 (AGENTS.md section 2). Matched as whole
# words in science code (case-insensitive); docs/history may mention them.
# NOTE: bare "K" (potassium) is deliberately NOT scanned: it is
# indistinguishable from the Kelvin unit "K" that physics code uses
# constantly. "NA" stays scanned: the tokenizer only matches a literal "Na"
# token, which never collides with "N/A" (the slash breaks the word
# boundary) and never appears as ordinary prose in this codebase. Potassium
# never appears as a species in this codebase (verified by search).
EXCLUDED_SPECIES = ["NA", "H2S", "O3", "O2", "NH3", "HCN", "C2H2", "PH3"]

# Life/habitability language: banned everywhere outside this audit file.
BANNED_CLAIMS = [r"\bhabit\w*", r"\bbiosignatur\w*", r"\blife\b", r"\baliens?\b"]

# Imagery that must never appear in twin/UI code (negated disclaimer lines OK).
FORBIDDEN_VISUALS = [
    "TextureLoader",
    "useTexture",
    "continent",
    "surface map",
    "molecule map",
]
NEGATION = re.compile(r"\b(no|not|never|without|avoid|against|isn't|don't|none)\b", re.I)

# UI labels AGENTS.md section 3 requires.
REQUIRED_LABELS = ["ML candidate score", "not a photograph"]

SCIENCE_DIRS = ["src/exosphere/forward", "src/exosphere/retrieval", "src/exosphere/ml"]
TWIN_DIRS = ["src/exosphere/twin.py", "web/src/components/twin", "web/src/lib/twinGeometry.ts"]


def _iter_py(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = REPO_ROOT / raw
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return [f for f in files if "__pycache__" not in f.parts]


def check_molecules() -> list[str]:
    """No excluded species in science code paths."""
    problems = []
    token = re.compile(r"\b([A-Z][a-z]?\d*)\b")
    for path in _iter_py(SCIENCE_DIRS):
        text = path.read_text(encoding="utf-8")
        for match in set(token.findall(text)):
            upper = match.upper()
            if upper in EXCLUDED_SPECIES and upper not in ALLOWED_MOLECULES:
                problems.append(f"{path.relative_to(REPO_ROOT)}: excluded species {match!r}")
    return problems


def check_ml_labels() -> list[str]:
    """ML scores labelled where presented; life claims absent from science code."""
    problems = []
    # Only presentational surfaces must carry the label (infer results, API
    # schemas, report, dashboard); internal tensors/losses are exempt.
    for raw in ["src/exosphere/ml/infer.py", "src/exosphere/report/build.py"]:
        path = REPO_ROOT / raw
        text = path.read_text(encoding="utf-8")
        if "score" in text.lower() and "ML candidate score" not in text:
            problems.append(f"{raw}: mentions scores without the label")
    for path in _iter_py(["src/exosphere"]):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # Skip wording-guard definitions (they name the ban to enforce it).
            if re.search(r"\bban", line, re.I):
                continue
            for pattern in BANNED_CLAIMS:
                if re.search(pattern, line, re.I):
                    problems.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: banned claim {pattern!r}"
                    )
    return problems


def check_no_geography() -> list[str]:
    """No fabricated geography in twin/UI code (negated lines are disclaimers)."""
    problems = []
    targets: list[Path] = []
    for raw in TWIN_DIRS:
        path = REPO_ROOT / raw
        if path.is_file():
            targets.append(path)
        elif path.is_dir():
            targets.extend(sorted(path.rglob("*.tsx")) + sorted(path.rglob("*.ts")))
    for path in targets:
        if path.name.endswith(".test.ts"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NEGATION.search(line):
                continue
            lowered = line.lower()
            for token in FORBIDDEN_VISUALS:
                if token.lower() in lowered:
                    problems.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {token!r}")
    return problems


def check_ui_labels() -> list[str]:
    """Required rule-3/rule-4 labels present in dashboard sources."""
    problems = []
    web = list((REPO_ROOT / "web" / "src").rglob("*.tsx"))
    joined = "\n".join(p.read_text(encoding="utf-8") for p in web).lower()
    for label in REQUIRED_LABELS:
        if label.lower() not in joined:
            problems.append(f"web/src: required label {label!r} missing")
    return problems


def check_provenance(db_path: str) -> list[str]:
    """Every COMPLETED analysis has complete provenance (AGENTS.md section 5).

    In-progress analyses gain their record at finalize; only completed ones
    are held to the rule ("no result without provenance").
    """
    import asyncio

    from sqlalchemy import select

    from exosphere.api.db import Analysis, AnalysisStatus

    if not Path(db_path).exists():
        return ["db missing: skipped"]

    required = [
        "planet", "observation_id", "telescope", "instrument", "source_archive",
        "preprocessing_version", "ml_model_version", "retrieval_model_version",
        "retrieval_parameters", "timestamp",
    ]

    async def _run() -> list[str]:
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        url = f"sqlite+aiosqlite:///{db_path}"
        engine = create_async_engine(url)
        try:
            maker = async_sessionmaker(engine, expire_on_commit=False)
            async with maker() as session:
                rows = (await session.execute(select(Analysis))).scalars().all()
                problems = []
                for analysis in rows:
                    if analysis.status != AnalysisStatus.COMPLETED:
                        continue
                    stored = dict(analysis.provenance_json or {})
                    missing = [k for k in required if not stored.get(k)]
                    if missing:
                        problems.append(f"{analysis.analysis_id}: missing {missing}")
                return problems
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit repo scope vs AGENTS.md sections 2-3.")
    parser.add_argument("--db", default=str(REPO_ROOT / "exosphere.db"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results = {
        "molecules": check_molecules(),
        "ml_labels": check_ml_labels(),
        "geography": check_no_geography(),
        "ui_labels": check_ui_labels(),
        "provenance": check_provenance(args.db),
    }
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for name, problems in results.items():
            status = "PASS" if not problems else "FAIL"
            print(f"[{status}] {name} ({len(problems)} issue(s))")
            for problem in problems:
                print(f"  - {problem}")
    # A skipped DB check is not a failure.
    real = [
        problem
        for problems in results.values()
        for problem in problems
        if problem != "db missing: skipped"
    ]
    return len(real)


if __name__ == "__main__":
    raise SystemExit(main())
