"""Guardrails for declared runtime imports and lazy database setup."""

from __future__ import annotations

import ast
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"

# Optional extras are deliberately absent from the fast CI environment, so
# importlib.metadata cannot map their top-level module names there.
MODULE_DISTRIBUTION_OVERRIDES = {
    "PIL": "pillow",
    "dotenv": "python-dotenv",
    "multipart": "python-multipart",
    "pydantic_settings": "pydantic-settings",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "yaml": "pyyaml",
}


def _import_roots() -> set[str]:
    """Third-party modules imported while a source module is imported.

    Function-local imports are intentionally excluded: JAXNS and PDF export
    are optional features that fail with their own actionable runtime errors
    when their optional integrations are absent.
    """
    roots: set[str] = set()
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _declared_distributions() -> set[str]:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        *config["project"]["dependencies"],
        *config["project"].get("optional-dependencies", {}).get("ml", []),
    ]
    return {
        requirement.split("==", maxsplit=1)[0].lower().replace("_", "-")
        for requirement in requirements
    }


def test_source_imports_are_declared_runtime_dependencies() -> None:
    """Every non-stdlib, non-local module imported under ``src`` is declared."""
    package_to_distribution = importlib.metadata.packages_distributions()
    declared = _declared_distributions()
    missing: dict[str, list[str]] = {}
    for module in sorted(_import_roots()):
        if module in sys.stdlib_module_names or module in {"__future__", "exosphere"}:
            continue
        distributions = package_to_distribution.get(module)
        if distributions is None and module in MODULE_DISTRIBUTION_OVERRIDES:
            distributions = [MODULE_DISTRIBUTION_OVERRIDES[module]]
        if distributions is None:
            missing[module] = ["no installed distribution mapping"]
            continue
        normalized = {dist.lower().replace("_", "-") for dist in distributions}
        if not normalized & declared:
            missing[module] = distributions
    assert not missing, f"undeclared runtime imports: {missing}"


def test_api_import_is_lazy_about_sqlite_driver_and_database_file(tmp_path: Path) -> None:
    """Importing the API must not import aiosqlite or create a local database."""
    command = [
        sys.executable,
        "-c",
        "import pathlib, sys; import exosphere.api; "
        "assert 'aiosqlite' not in sys.modules; "
        "assert not pathlib.Path('exosphere.db').exists()",
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
