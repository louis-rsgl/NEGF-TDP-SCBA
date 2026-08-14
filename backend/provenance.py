"""Machine-readable software, runtime, method, and citation provenance."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


SOFTWARE_NAME = "NEGF-TDPSCBA"
REPOSITORY_URL = "https://github.com/louis-rsgl/NEGF-TDPSCBA"


def citation_records() -> list[dict[str, Any]]:
    """Return citations without inventing an unpublished DOI or release version."""
    return [
        {
            "applies_to": "software implementation",
            "type": "software",
            "authors": ["Louis Rossignol", "Hong Guo"],
            "title": (
                "NEGF-TDPSCBA: Time-Dependent Quantum Transport with "
                "Frozen-SCBA Phonons"
            ),
            "repository": REPOSITORY_URL,
            "doi": None,
            "note": "Use a release DOI when one becomes available.",
        },
        {
            "applies_to": "frozen electron-phonon pulse method",
            "type": "manuscript",
            "authors": ["Louis Rossignol", "Hong Guo"],
            "title": (
                "Time-dependent quantum transport far from equilibrium: "
                "A nonlinear response theory with frozen SCBA phonons"
            ),
            "repository_file": (
                "E-phonon_Keldysh_frozen_SCBA_refined_proofs_v26.tex"
            ),
            "doi": None,
            "arxiv": None,
            "note": "Publication identifier requires author input.",
        },
        {
            "applies_to": "finite-bandwidth partition-free pulse theory",
            "type": "article",
            "authors": ["J. Maciejko", "J. Wang", "H. Guo"],
            "title": (
                "Time-dependent quantum transport far from equilibrium: "
                "An exact nonlinear response theory"
            ),
            "journal": "Physical Review B",
            "volume": "74",
            "article": "085324",
            "year": 2006,
            "doi": "10.1103/PhysRevB.74.085324",
        },
        {
            "applies_to": "Minimal Pole Method",
            "type": "article",
            "authors": ["L. Zhang", "Y. Yu", "E. Gull"],
            "title": (
                "Minimal pole representation and analytic continuation of "
                "matrix-valued correlation functions"
            ),
            "journal": "Physical Review B",
            "volume": "110",
            "article": "235131",
            "year": 2024,
            "doi": "10.1103/PhysRevB.110.235131",
        },
    ]


def _git_value(repository_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def git_state(repository_root: Path | None = None) -> dict[str, Any]:
    """Return the available revision state, or explicit nulls outside Git."""
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    commit = _git_value(root, "rev-parse", "HEAD")
    description = _git_value(root, "describe", "--tags", "--always", "--dirty")
    status = _git_value(root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "description": description,
        "dirty": None if commit is None else bool(status),
    }


def dependency_versions() -> dict[str, str | None]:
    """Report direct numerical dependencies without importing them."""
    distributions = (
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "kneed",
        "mini_pole",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def software_provenance(repository_root: Path | None = None) -> dict[str, Any]:
    """Return JSON-serializable software and execution-environment metadata."""
    return {
        "software": {
            "name": SOFTWARE_NAME,
            "repository": REPOSITORY_URL,
            "release_version": None,
            "release_doi": None,
            "git": git_state(repository_root),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "dependencies": dependency_versions(),
        },
    }


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def system_provenance(system, repository_root: Path | None = None) -> dict[str, Any]:
    """Return software provenance plus a System's public numerical settings."""
    settings = {
        field.name: _json_value(getattr(system, field.name))
        for field in fields(system)
        if not field.name.startswith("_")
    }
    result = software_provenance(repository_root)
    result["calculation"] = {
        "pulse_protocol": system.pulse_protocol,
        "stationary_reference": (
            "biased" if system.reference_is_biased else "unbiased"
        ),
        "stationary_approximation": system.scba_mode,
        "retarded_reduction": "MiniPole with validated causal AAA fallback",
        "units": {
            "backend_energy": "Gamma",
            "backend_time": "hbar/Gamma",
            "backend_current": "e Gamma/hbar",
        },
        "settings": settings,
    }
    result["citations"] = citation_records()
    return result
