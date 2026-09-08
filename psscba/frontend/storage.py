from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml

from psscba.frontend.configuration import RunConfig, load_config
from psscba.backend.model.units import current_from_uA, current_to_uA, time_from_ps, time_to_ps
from psscba.types import OuterIteration, PSSCBAResult, StationaryKernel
from psscba.backend.poles.minimal import PoleBasis


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_value(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_text(path: Path, payload: str) -> None:
    atomic_bytes(path, payload.encode("utf-8"))


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def unit_metadata(config: RunConfig) -> dict[str, Any]:
    """Describe the natural calculation units and the on-disk legacy units.

    The frozen runner kept all numerical work in ``hbar/Gamma`` and
    ``e Gamma/hbar`` but serialized its observable traces as ps and µA.  New
    runs retain that convention at the storage boundary.
    """
    gamma = float(config.model.gamma_eV)
    time_factor = float(time_to_ps(1.0, gamma))
    current_factor = float(current_to_uA(1.0, gamma))
    return {
        "convention": "original_runner_v1",
        "gamma_eV": gamma,
        "internal": {
            "energy": "Gamma",
            "time": "hbar/Gamma",
            "current": "e Gamma/hbar",
        },
        "storage": {
            "energy": "Gamma",
            "time": "ps",
            "current": "uA",
        },
        "conversion": {
            "time_ps_per_hbar_over_gamma": time_factor,
            "current_uA_per_e_gamma_over_hbar": current_factor,
        },
        "arrays": {
            "observable_time": "ps",
            "current_L": "uA",
            "current_R": "uA",
            "current_corrected_L": "uA (diagnostic only)",
            "current_corrected_R": "uA (diagnostic only)",
            "continuity_residual_raw": "e Gamma/hbar",
            "projection_time": "hbar/Gamma",
            "projection_lag": "hbar/Gamma",
            "amplitude_A2": "dimensionless |A|^2",
            "amplitude_A2_L": "dimensionless |A_L|^2",
            "amplitude_A2_R": "dimensionless |A_R|^2",
            "amplitude_C2": "dimensionless |C|^2",
            "amplitude_C2_minus_omega": "dimensionless |C(t,-omega_0,E')|^2",
            "amplitude_C2_plus_omega": "dimensionless |C(t,+omega_0,E')|^2",
        },
    }


def write_units_metadata(run_dir: Path, config: RunConfig) -> None:
    atomic_json(Path(run_dir) / "units.json", unit_metadata(config))


def read_units_metadata(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "units.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_saved_observables(
    run_dir: str | Path,
    config: RunConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load serialized time/current arrays in the solver's natural units.

    Runs written before the storage-unit manifest are treated as natural-unit
    files, which preserves read/plot compatibility with the previous active
    package.  New runs are identified by ``units.json`` and use the original
    µA/ps storage convention.
    """
    run_dir = Path(run_dir)
    data = run_dir / "data"
    time = np.load(data / "observable_time.npy", allow_pickle=False)
    currents = {
        lead: np.load(data / f"current_{lead}.npy", allow_pickle=False)
        for lead in ("L", "R")
        if (data / f"current_{lead}.npy").exists()
    }
    metadata = read_units_metadata(run_dir)
    if metadata is None:
        return np.asarray(time, dtype=float), {
            lead: np.asarray(values) for lead, values in currents.items()
        }
    storage = metadata.get("storage", {})
    if storage.get("time") == "ps":
        time = time_from_ps(time, config.model.gamma_eV)
    if storage.get("current") == "uA":
        currents = {
            lead: current_from_uA(values, config.model.gamma_eV)
            for lead, values in currents.items()
        }
    return np.asarray(time, dtype=float), currents


def load_saved_current_views(
    run_dir: str | Path,
    config: RunConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load raw and optional conservation-corrected current views.

    The raw currents remain the archive-compatible primary products.  When a
    noninteracting conserving correction was produced, the corrected files
    are loaded alongside them so plotting can show both views explicitly.
    Values are returned in the solver's natural units, matching
    :func:`load_saved_observables`.
    """
    run_dir = Path(run_dir)
    time, currents = load_saved_observables(run_dir, config)
    data = run_dir / "data"
    corrected = {
        lead: np.load(data / f"current_corrected_{lead}.npy", allow_pickle=False)
        for lead in ("L", "R")
        if (data / f"current_corrected_{lead}.npy").exists()
    }
    metadata = read_units_metadata(run_dir)
    if metadata is not None and metadata.get("storage", {}).get("current") == "uA":
        corrected = {
            lead: current_from_uA(values, config.model.gamma_eV)
            for lead, values in corrected.items()
        }
    return time, currents, corrected


def create_run_directory(
    config: RunConfig,
    *,
    label: str | None = None,
    root: str | Path | None = None,
    exact_name: bool = False,
) -> Path:
    root = Path(config.output_root) if root is None else Path(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = label or f"{config.protocol.name}_W{config.model.bandwidth:g}_g{config.model.coupling_meV:g}"
    run_dir = root / (name if exact_name else f"{stamp}_{name}")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_yaml(run_dir / "resolved.yaml", config.to_dict())
    write_units_metadata(run_dir, config)
    from psscba.frontend.provenance import software_provenance

    atomic_json(run_dir / "provenance.json", software_provenance())
    atomic_json(run_dir / "status.json", {"status": "created"})
    return run_dir


def save_kernel(directory: Path, kernel: StationaryKernel) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_npy(directory / "energy.npy", kernel.energy)
    atomic_npy(directory / "sigma_retarded.npy", kernel.sigma_retarded)
    atomic_npy(directory / "sigma_lesser.npy", kernel.sigma_lesser)
    atomic_npy(directory / "sigma_dynamic_retarded.npy", kernel.sigma_dynamic_retarded)
    atomic_npy(directory / "spectral_width.npy", kernel.spectral_width)
    if kernel.projected_green_retarded is not None:
        atomic_npy(directory / "projected_green_retarded.npy", kernel.projected_green_retarded)
    if kernel.projected_green_lesser is not None:
        atomic_npy(directory / "projected_green_lesser.npy", kernel.projected_green_lesser)
    atomic_json(
        directory / "kernel.json",
        {
            "sigma_hartree": kernel.sigma_hartree,
            "phonon_occupation": kernel.phonon_occupation,
            "method": kernel.method,
            "protocol": kernel.protocol,
            "iteration": kernel.iteration,
            "metadata": dict(kernel.metadata),
        },
    )


def save_pole_basis(directory: Path, basis: PoleBasis | None) -> None:
    if basis is None:
        for path in (
            directory / "pole_basis_zeta.npy",
            directory / "pole_basis_weights.npy",
            directory / "pole_basis.json",
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        return
    atomic_npy(directory / "pole_basis_zeta.npy", basis.zeta)
    if basis.weights is not None:
        atomic_npy(directory / "pole_basis_weights.npy", basis.weights)
    else:
        try:
            (directory / "pole_basis_weights.npy").unlink()
        except FileNotFoundError:
            pass
    atomic_json(directory / "pole_basis.json", {
        "energy_digest": basis.energy_digest,
        "system_signature": basis.system_signature,
        "fit_tolerance": basis.fit_tolerance,
        "fit_terms": basis.fit_terms,
        "source_digest": basis.source_digest,
        "weights_present": basis.weights is not None,
        "origin_iteration": basis.origin_iteration,
        "method": basis.method,
        "fit_converged": basis.fit_converged,
    })


def load_pole_basis(directory: Path) -> PoleBasis | None:
    zeta_path = directory / "pole_basis_zeta.npy"
    metadata_path = directory / "pole_basis.json"
    if not zeta_path.exists() or not metadata_path.exists():
        return None
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    weights_path = directory / "pole_basis_weights.npy"
    weights = (
        np.load(weights_path, allow_pickle=False)
        if weights_path.exists()
        else None
    )
    metadata.pop("weights_present", None)
    return PoleBasis(
        np.load(zeta_path, allow_pickle=False),
        weights=weights,
        **metadata,
    )


def load_kernel(directory: Path) -> StationaryKernel:
    with open(directory / "kernel.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    projected_r = directory / "projected_green_retarded.npy"
    projected_l = directory / "projected_green_lesser.npy"
    return StationaryKernel(
        energy=np.load(directory / "energy.npy", allow_pickle=False),
        sigma_retarded=np.load(directory / "sigma_retarded.npy", allow_pickle=False),
        sigma_lesser=np.load(directory / "sigma_lesser.npy", allow_pickle=False),
        sigma_hartree=metadata["sigma_hartree"],
        sigma_dynamic_retarded=np.load(
            directory / "sigma_dynamic_retarded.npy", allow_pickle=False
        ),
        spectral_width=np.load(directory / "spectral_width.npy", allow_pickle=False),
        phonon_occupation=metadata["phonon_occupation"],
        projected_green_retarded=(
            np.load(projected_r, allow_pickle=False) if projected_r.exists() else None
        ),
        projected_green_lesser=(
            np.load(projected_l, allow_pickle=False) if projected_l.exists() else None
        ),
        method=metadata["method"],
        protocol=metadata["protocol"],
        iteration=metadata["iteration"],
        metadata=metadata.get("metadata", {}),
    )


def checkpoint_writer(run_dir: Path):
    def write(kernel: StationaryKernel, history: tuple[OuterIteration, ...], basis=None) -> None:
        save_kernel(run_dir / "checkpoint", kernel)
        save_pole_basis(run_dir / "checkpoint", basis)
        atomic_json(run_dir / "outer_history.json", [asdict(item) for item in history])
        atomic_json(
            run_dir / "status.json",
            {"status": "running", "iteration": kernel.iteration},
        )
    return write


def load_checkpoint(
    run_dir: str | Path,
) -> tuple[RunConfig, StationaryKernel, tuple[OuterIteration, ...]]:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "resolved.yaml")
    with open(run_dir / "outer_history.json", encoding="utf-8") as handle:
        history = tuple(OuterIteration(**item) for item in json.load(handle))
    return config, load_kernel(run_dir / "checkpoint"), history


def save_result(run_dir: Path, result: PSSCBAResult) -> None:
    config = load_config(run_dir / "resolved.yaml", allow_read_only_legacy=True)
    write_units_metadata(run_dir, config)
    save_kernel(run_dir / "final_kernel", result.kernel)
    if result.initial_kernel is not None:
        save_kernel(run_dir / "initial_kernel", result.initial_kernel)
    data = run_dir / "data"
    projection = result.trajectory.projected
    arrays = {
        "projection_time": result.trajectory.time,
        # The original runner serialized observable time in ps and currents
        # in µA.  All other two-time/reconstruction arrays remain in the
        # backend's natural units and are annotated by units.json.
        "observable_time": time_to_ps(
            result.trajectory.observable_time, config.model.gamma_eV
        ),
        "reconstruction_energy": result.trajectory.energy,
        "projection_lag": projection.lag,
        "projected_green_retarded_time": projection.green_retarded,
        "projected_green_lesser_time": projection.green_lesser,
        "projected_green_retarded_energy": projection.green_retarded_energy,
        "projected_green_lesser_energy": projection.green_lesser_energy,
        "q_green_retarded_time": projection.q_green_retarded_time,
        "q_green_lesser_time": projection.q_green_lesser_time,
        "occupation": result.trajectory.occupation,
        "continuity_residual_raw": result.trajectory.raw_continuity_residual,
        "continuity_residual": result.trajectory.continuity_residual,
        "collision_source": result.trajectory.collision_source,
        "collision_source_finite": result.trajectory.collision_source_finite,
        "collision_source_prehistory": result.trajectory.collision_source_prehistory,
    }
    # A/C are internal retarded amplitudes.  Save their real squared
    # magnitudes on the same (time, energy) grid for reproducible surface
    # diagnostics; no complex or dense two-time arrays are needed.
    for lead, values in result.trajectory.amplitude_a_squared.items():
        arrays[f"amplitude_A2_{lead}"] = values
    arrays["amplitude_C2"] = result.trajectory.amplitude_c_squared
    for frequency, values in result.trajectory.amplitude_c_squared_by_frequency.items():
        branch = "minus" if float(frequency) < 0.0 else "plus"
        arrays[f"amplitude_C2_{branch}_omega"] = values
    for lead, values in result.trajectory.currents.items():
        arrays[f"current_{lead}"] = current_to_uA(values, config.model.gamma_eV)
    # Conservation-corrected currents are diagnostic views only.  Keep the
    # canonical current_{L,R}.npy arrays raw so archive comparisons and
    # publication figures use the physical pole/Filon result.
    for lead, values in result.trajectory.corrected_currents.items():
        arrays[f"current_corrected_{lead}"] = current_to_uA(
            values, config.model.gamma_eV
        )
    for name, values in arrays.items():
        if values is not None:
            atomic_npy(data / f"{name}.npy", np.asarray(values))
    atomic_json(run_dir / "outer_history.json", [asdict(item) for item in result.history])
    atomic_json(
        run_dir / "diagnostics.json",
        {**dict(result.diagnostics), **dict(result.trajectory.diagnostics)},
    )
    atomic_json(
        run_dir / "status.json",
        {"status": "complete" if result.converged else "failed"},
    )
