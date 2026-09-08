from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from psscba.frontend.campaign import case_config, run_case
from psscba.frontend.configuration import (
    CONFIG_FORMAT,
    ProtocolConfig,
    RunConfig,
    require_shared_production_grid,
)
from psscba.frontend.storage import atomic_json, atomic_yaml, create_run_directory
from psscba.frontend.provenance import software_provenance
from psscba.backend.solver import solve_fixed_kernel
from psscba.backend.stationary.initializer import initial_zero_pulse_kernel
from psscba.backend.observables import (
    direct_oracle_observables,
    finite_window_observables,
)
from psscba.backend.protocols.analytic import time_derivative


def _resolved_continuity_scaled_error(
    case, trajectory, *, corrected: bool = False
) -> float:
    """Return a continuity error away from switching stencils.

    The canonical currents stored on a trajectory are always the raw
    pole/Filon values.  Non-interacting runs may additionally carry a
    conservation-corrected view for diagnostics.  Keeping the selector here
    explicit prevents the validation code from accidentally comparing an
    occupation produced on one quadrature with a different current view.
    """
    time = np.asarray(trajectory.observable_time)
    if corrected:
        if not trajectory.corrected_currents or trajectory.continuity_residual is None:
            raise ValueError("Corrected continuity was requested but is unavailable.")
        currents = trajectory.corrected_currents
        residual = np.asarray(trajectory.continuity_residual)
    else:
        currents = trajectory.currents
        if trajectory.raw_continuity_residual is None:
            raise ValueError("Raw continuity is unavailable on this trajectory.")
        residual = np.asarray(trajectory.raw_continuity_residual)
    resolved = np.abs(time) > 2.1 * case.projection.time_step
    if case.protocol.name == "square":
        resolved &= (
            np.abs(time - float(case.protocol.duration))
            > 2.1 * case.projection.time_step
        )
    if not np.any(resolved):
        resolved[:] = True
    derivative = time_derivative(np.asarray(trajectory.occupation), time)
    scale = max(
        1e-12,
        *(float(np.max(np.abs(values))) for values in currents.values()),
        float(np.max(np.abs(derivative))),
    )
    return float(np.max(np.abs(residual[resolved])) / scale)


def run_oracle_preflight(config: RunConfig, *, progress=None) -> dict[str, float]:
    """Compare production transforms with the literal A/Psi oracle.

    The short window keeps the quadratic oracle affordable while retaining
    the configured time and energy spacings.  This preflight intentionally
    runs before the campaign ladder and fails closed.
    """
    energy_bound = min(
        20.0,
        abs(config.numerics.energy_min),
        config.numerics.energy_max,
    )
    time_step = config.projection.time_step
    time_min = max(config.projection.time_min, -8.0)
    time_max = min(config.projection.time_max, 0.2)
    report: dict[str, float] = {}
    # The literal oracle remains compact because its dense A/Psi arrays scale
    # quadratically with the energy grid.  Conservation, however, must be
    # tested with the configured production reconstruction trajectory and
    # configured production current quadrature used by real cases.  Build
    # those trajectories separately below; never combine a compact-grid
    # occupation with a wide-grid current.
    protocols = ("upward", "downward", "square")

    def emit(protocol: str, stage: str, **details) -> None:
        if progress is not None:
            progress(protocol, stage, details)

    for protocol_index, protocol in enumerate(protocols, start=1):
        protocol_bound = min(6.0, energy_bound) if protocol == "square" else energy_bound
        duration = None
        if protocol == "square":
            duration = 0.5 * time_max + 0.37 * time_step
        compact_points = (
            int(round(2.0 * protocol_bound / config.numerics.energy_step))
            + 1
        )
        emit(
            protocol,
            "prepare",
            protocol_index=protocol_index,
            protocol_total=len(protocols),
            compact_energy_points=compact_points,
            production_energy_points=len(config.numerics.energy_grid()),
            time_points=int(round((time_max - time_min) / time_step)) + 1,
            energy_bound=protocol_bound,
        )
        preflight = replace(
            case_config(config, protocol, 20.0, 0.0),
            protocol=ProtocolConfig(name=protocol, duration=duration),
            projection=replace(
                config.projection,
                time_min=time_min,
                time_max=time_max,
                dense_validation=True,
            ),
            numerics=replace(
                config.numerics,
                energy_min=-protocol_bound,
                energy_max=protocol_bound,
                current_energy_min=-protocol_bound,
                current_energy_max=protocol_bound,
                current_energy_points=compact_points,
                current_time_points=int(round(time_max / time_step)) + 1,
                enforce_hard_gates=False,
            ),
        )
        stage_start = perf_counter()
        emit(protocol, "stationary_initialization_start")
        kernel = initial_zero_pulse_kernel(preflight)
        emit(
            protocol,
            "stationary_initialization_done",
            duration_seconds=perf_counter() - stage_start,
        )

        def trajectory_progress(phase, details):
            # The fixed-kernel solver already reports meaningful batch fields.
            # Mirror those at the preflight level so a dense call cannot look
            # idle in the campaign log.
            emit(protocol, f"trajectory_{phase}", **dict(details))

        stage_start = perf_counter()
        emit(protocol, "fixed_kernel_start")
        trajectory = solve_fixed_kernel(
            preflight,
            kernel,
            dense=True,
            observables=False,
            progress=trajectory_progress,
        )
        emit(
            protocol,
            "fixed_kernel_done",
            duration_seconds=perf_counter() - stage_start,
        )
        stage_start = perf_counter()
        emit(protocol, "fast_current_start")
        fast = finite_window_observables(
            preflight,
            kernel,
            trajectory,
            prefer_time_domain=False,
        )
        emit(protocol, "fast_current_done", duration_seconds=perf_counter() - stage_start)
        stage_start = perf_counter()
        emit(protocol, "dense_oracle_start")
        direct = direct_oracle_observables(
            preflight,
            kernel,
            trajectory,
        )
        emit(protocol, "dense_oracle_done", duration_seconds=perf_counter() - stage_start)
        for lead in ("L", "R"):
            direct_current = np.asarray(direct.currents[lead])
            scale = max(float(np.max(np.abs(direct_current))), 1e-12)
            report[f"{protocol}_current_{lead}_scaled_error"] = float(
                np.max(
                    np.abs(np.asarray(fast.currents[lead]) - direct_current)
                )
                / scale
            )
        report[f"{protocol}_occupation_error"] = float(
            np.max(
                np.abs(
                    np.asarray(fast.occupation) - np.asarray(direct.occupation)
                )
            )
        )
        report[f"{protocol}_retarded_green_error"] = float(
            np.max(
                np.abs(
                    fast.projected.green_retarded_energy
                    - direct.projected.green_retarded_energy
                )
            )
        )
        report[f"{protocol}_lesser_green_error"] = float(
            np.max(
                np.abs(
                    fast.projected.green_lesser_energy
                    - direct.projected.green_lesser_energy
                )
            )
        )
        emit(
            protocol,
            "oracle_comparison_done",
            current_L_scaled_error=report[f"{protocol}_current_L_scaled_error"],
            current_R_scaled_error=report[f"{protocol}_current_R_scaled_error"],
            occupation_error=report[f"{protocol}_occupation_error"],
            retarded_green_error=report[f"{protocol}_retarded_green_error"],
            lesser_green_error=report[f"{protocol}_lesser_green_error"],
        )

        # Fast/direct agreement above validates two implementations of the
        # same compact discretization.  Conservation is a separate test on a
        # production-grid trajectory.  This avoids mixing the compact oracle
        # occupation with a wide production current quadrature.
        configured_points = config.numerics.current_energy_points
        configured_min = config.numerics.current_energy_min
        configured_max = config.numerics.current_energy_max
        if configured_min is None or configured_max is None:
            configured_min = config.numerics.energy_min
            configured_max = config.numerics.energy_max
        if configured_points is None:
            configured_points = len(config.numerics.energy_grid())
        conservation_case = replace(
            case_config(config, protocol, 20.0, 0.0),
            protocol=ProtocolConfig(name=protocol, duration=duration),
            projection=replace(
                config.projection,
                time_min=time_min,
                time_max=time_max,
                dense_validation=False,
            ),
            numerics=replace(
                config.numerics,
                current_energy_min=float(configured_min),
                current_energy_max=float(configured_max),
                current_energy_points=int(configured_points),
                enforce_hard_gates=False,
            ),
        )
        stage_start = perf_counter()
        emit(
            protocol,
            "production_conservation_start",
            energy_points=int(configured_points),
            energy_min=float(configured_min),
            energy_max=float(configured_max),
        )
        production_kernel = initial_zero_pulse_kernel(conservation_case)
        production_trajectory = solve_fixed_kernel(
            conservation_case,
            production_kernel,
            dense=False,
            observables=False,
            progress=trajectory_progress,
        )
        conserving_grid = finite_window_observables(
            conservation_case,
            production_kernel,
            production_trajectory,
            prefer_time_domain=False,
            conserve_noninteracting=True,
        )
        report[f"{protocol}_raw_continuity_scaled_error"] = (
            _resolved_continuity_scaled_error(
                conservation_case, conserving_grid, corrected=False
            )
        )
        report[f"{protocol}_corrected_continuity_scaled_error"] = (
            _resolved_continuity_scaled_error(
                conservation_case, conserving_grid, corrected=True
            )
        )
        emit(
            protocol,
            "production_conservation_done",
            duration_seconds=perf_counter() - stage_start,
            raw_continuity_scaled_error=report[f"{protocol}_raw_continuity_scaled_error"],
            corrected_continuity_scaled_error=report[f"{protocol}_corrected_continuity_scaled_error"],
        )
    return report


@dataclass(frozen=True)
class ValidationCase:
    """One resolved validation job plus its reporting/gating metadata."""

    name: str
    config: RunConfig
    preparation_fixed: bool = False
    tier: str = "approximation"
    gating: bool = False
    family: str = "thermal"

    # Preserve the historical tuple-unpacking API used by notebooks/tests.
    def __iter__(self):
        yield self.name
        yield self.config
        yield self.preparation_fixed


def _validation_case_fields(item):
    """Normalize new metadata-bearing cases and legacy three-tuples."""
    if isinstance(item, ValidationCase):
        return item
    name, config, preparation_fixed = item
    return ValidationCase(
        name=name,
        config=config,
        preparation_fixed=preparation_fixed,
        tier="approximation",
        gating=False,
        family="legacy",
    )


def _resolved_occupation(config: RunConfig) -> float:
    if config.model.phonon_occupation is not None:
        return float(config.model.phonon_occupation)
    return float(1.0 / np.expm1(config.model.beta_ph * config.model.phonon_energy))


def _validation_plan_digest(planned: list[ValidationCase]) -> str:
    payload = [
        {
            "name": item.name,
            "config": item.config.to_dict(),
            "preparation_fixed": item.preparation_fixed,
            "tier": item.tier,
            "gating": item.gating,
            "family": item.family,
        }
        for item in planned
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _write_validation_products(session_dir: Path, manifest: dict) -> None:
    """Collect completed diagnostics and render family-separated heatmaps."""
    records: list[dict[str, object]] = []
    for job in manifest.get("jobs", []):
        run_directory = Path(job.get("run_directory", ""))
        diagnostics_path = run_directory / "diagnostics.json"
        if not diagnostics_path.exists():
            continue
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({
            **{key: job.get(key) for key in (
                "index", "case_id", "name", "tier", "gating", "family",
                "validation_family", "occupation_mode",
                "protocol", "bandwidth", "phonon_energy", "phonon_occupation",
                "resolved_phonon_occupation", "coupling_meV", "coupling_over_gamma",
                "coupling_over_omega", "method", "preparation_fixed",
            )},
            **diagnostics,
        })
    atomic_json(session_dir / "validation_summary.json", records)
    if not records:
        return
    from psscba.backend.plotting.service import write_aggregate_heatmaps

    for family in sorted({str(item.get("family")) for item in records}):
        selected = [item for item in records if str(item.get("family")) == family]
        write_aggregate_heatmaps(
            selected,
            session_dir / "validation_qa" / family,
        )


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    scale = max(1.0e-12, float(np.max(np.abs(first))), float(np.max(np.abs(second))))
    return float(np.max(np.abs(first - second)) / scale)


def _compare_validation_runs(first: Path, second: Path) -> dict[str, float]:
    """Compare two resolved cases after interpolating onto common output nodes."""
    result: dict[str, float] = {}
    first_data, second_data = first / "data", second / "data"
    try:
        time_a = np.load(first_data / "observable_time.npy", allow_pickle=False)
        time_b = np.load(second_data / "observable_time.npy", allow_pickle=False)
        lo, hi = max(time_a[0], time_b[0]), min(time_a[-1], time_b[-1])
        mask = (time_a >= lo) & (time_a <= hi)
        if np.count_nonzero(mask) < 2:
            return result
        common = time_a[mask]
        for name in ("current_L", "current_R", "occupation", "collision_source"):
            path_a, path_b = first_data / f"{name}.npy", second_data / f"{name}.npy"
            if not path_a.exists() or not path_b.exists():
                continue
            values_a = np.asarray(np.load(path_a, allow_pickle=False), dtype=float)
            values_b = np.asarray(np.load(path_b, allow_pickle=False), dtype=float)
            interp_b = np.interp(common, time_b, values_b)
            result[f"{name}_relative_difference"] = _relative_difference(values_a[mask], interp_b)
        energy_a = np.load(first_data / "reconstruction_energy.npy", allow_pickle=False)
        energy_b = np.load(second_data / "reconstruction_energy.npy", allow_pickle=False)
        for name in ("sigma_retarded", "sigma_lesser"):
            path_a = first / "final_kernel" / f"{name}.npy"
            path_b = second / "final_kernel" / f"{name}.npy"
            if not path_a.exists() or not path_b.exists():
                continue
            values_a = np.load(path_a, allow_pickle=False)
            values_b = np.load(path_b, allow_pickle=False)
            interp_b = np.interp(energy_a, energy_b, values_b.real) + 1j * np.interp(
                energy_a, energy_b, values_b.imag
            )
            result[f"{name}_relative_difference"] = _relative_difference(
                np.abs(values_a), np.abs(interp_b)
            )
    except (OSError, ValueError, IndexError):
        return result
    return result


def _resolution_convergence(manifest: dict) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compare the two finest members of each numerical-resolution family."""
    jobs = {
        str(item.get("name")): item
        for item in manifest.get("jobs", [])
        if item.get("status") in ("accepted", "complete")
    }
    # The quality key is larger for the more resolved member of each axis.
    # This keeps the comparison independent of the order in which YAML lists
    # values and handles the two-dimensional streamed-window axis explicitly.
    families = {
        "time_step": (r"^numerical_dt_(.+)$", lambda g: 1.0 / float(g[0])),
        "reconstruction_window": (r"^numerical_reconstruction_window_(.+)$", lambda g: float(g[0])),
        "reconstruction_step": (r"^numerical_reconstruction_step_(.+)$", lambda g: 1.0 / float(g[0])),
        "current_window": (r"^numerical_current_window_(.+)$", lambda g: float(g[0])),
        "current_step": (r"^numerical_current_step_(.+)$", lambda g: 1.0 / float(g[0])),
        "prehistory_tmin": (r"^numerical_prehistory_tmin_(.+)$", lambda g: abs(float(g[0]))),
        "prehistory_order": (r"^numerical_prehistory_order_(.+)$", lambda g: float(g[0])),
        "prehistory_scale": (r"^numerical_prehistory_scale_(.+)$", lambda g: float(g[0])),
        "stationary_eta": (r"^numerical_stationary_eta_(.+)$", lambda g: 1.0 / float(g[0])),
        "mpm": (r"^numerical_mpm_(.+)$", lambda g: 1.0 / float(g[0])),
        "streamed_window": (
            r"^numerical_streamed_window_(.+)_(.+)$",
            lambda g: float(g[1]) - float(g[0]),
        ),
    }
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for family, (pattern, quality_key) in families.items():
        matches = []
        for name, item in jobs.items():
            match = re.match(pattern, name)
            if match:
                try:
                    value = float(quality_key(match.groups()))
                except (TypeError, ValueError, IndexError):
                    continue
                matches.append((value, name, item))
        if len(matches) < 2:
            continue
        matches.sort(key=lambda item: item[0])
        selected = sorted(matches, key=lambda item: item[0], reverse=True)[:2]
        first, second = selected[0], selected[1]
        first_dir, second_dir = Path(first[2]["run_directory"]), Path(second[2]["run_directory"])
        metrics = _compare_validation_runs(first_dir, second_dir)
        row = {"family": family, "first": first[1], "second": second[1], **metrics}
        rows.append(row)
        limits = {
            "current_L_relative_difference": 1.0e-3,
            "current_R_relative_difference": 1.0e-3,
            "collision_source_relative_difference": 1.0e-3,
            "occupation_relative_difference": 1.0e-4,
            "sigma_retarded_relative_difference": 1.0e-3,
            "sigma_lesser_relative_difference": 1.0e-3,
        }
        exceeded = {
            key: value for key, value in metrics.items()
            if key in limits and value >= limits[key]
        }
        if exceeded:
            failures.append({"family": family, "first": first[1], "second": second[1], "metrics": exceeded})
    return rows, failures


def _mode_case(
    base: RunConfig,
    protocol: str,
    bandwidth: float,
    coupling: float,
    frequency: float,
    *,
    hot: bool = False,
    dense: bool = False,
    time_min: float | None = None,
    time_max: float | None = None,
) -> RunConfig:
    """Resolve one case without embedding a physical frequency in Python."""
    case = case_config(base, protocol, bandwidth, coupling, frequency)
    model = replace(case.model, phonon_occupation=None, beta_ph=base.model.beta_ph)
    if hot:
        occupation = float(base.validation.hot_occupation)
        model = replace(
            model,
            phonon_occupation=occupation,
            beta_ph=float(np.log1p(1.0 / occupation) / frequency),
        )
    projection = replace(case.projection, dense_validation=dense)
    if time_min is not None:
        projection = replace(projection, time_min=float(time_min))
    if time_max is not None:
        intervals = int(np.ceil((float(time_max) - projection.time_min) / projection.time_step))
        projection = replace(
            projection,
            time_max=projection.time_min + intervals * projection.time_step,
        )
    resolved = replace(case, model=model, projection=projection)
    resolved.validate()
    return resolved


def _square_case(base: RunConfig, duration: float, *, hot: bool) -> RunConfig:
    case = _mode_case(
        base, "square", base.model.bandwidth, base.model.coupling_meV,
        base.model.phonon_energy, hot=hot,
    )
    projection = case.projection
    required_max = max(float(projection.time_max), float(duration))
    intervals = int(np.ceil((required_max - projection.time_min) / projection.time_step))
    return replace(
        case,
        protocol=ProtocolConfig(name="square", duration=float(duration)),
        projection=replace(
            projection,
            time_max=projection.time_min + intervals * projection.time_step,
        ),
    )


def validation_cases(base: RunConfig) -> list[ValidationCase]:
    """Build a deduplicated, tier-labelled validation ladder from YAML axes."""
    v = base.validation
    cases: list[ValidationCase] = []
    seen: set[str] = set()
    base_frequency = float(base.model.phonon_energy)
    base_bandwidth = float(base.model.bandwidth)
    base_coupling = float(base.model.coupling_meV)
    if base_coupling not in v.couplings_meV:
        base_coupling = min(v.trusted_couplings_meV, key=lambda value: abs(value - base_coupling))

    def add(
        name: str,
        config: RunConfig,
        *,
        tier: str,
        gating: bool,
        family: str,
        preparation_fixed: bool = False,
    ) -> None:
        config.validate()
        signature = json.dumps(
            {"config": config.to_dict(), "preparation_fixed": preparation_fixed},
            sort_keys=True,
            default=str,
        )
        if signature in seen:
            return
        seen.add(signature)
        cases.append(ValidationCase(name, config, preparation_fixed, tier, gating, family))

    # Foundation: exact electronic limits for every configured protocol and W.
    for protocol in v.protocols:
        for bandwidth in v.bandwidths:
            add(
                f"foundation_g0_{protocol}_W{bandwidth:g}",
                _mode_case(base, protocol, bandwidth, 0.0, base_frequency),
                tier="foundation", gating=True, family="g0",
            )

    # Thermal baseline: dense only where collision/projection diagnostics need it.
    thermal_reference = _mode_case(
        base, "upward", base_bandwidth, base_coupling, base_frequency,
        dense=True, time_max=v.dense_time_max,
    )
    thermal_reference = replace(
        thermal_reference,
        numerics=replace(
            thermal_reference.numerics,
            energy_min=-float(v.compact_energy_bound),
            energy_max=float(v.compact_energy_bound),
        ),
    )
    thermal_reference.validate()
    add("numerical_reference_thermal_dense", thermal_reference,
        tier="numerical", gating=True, family="thermal")
    add(
        "numerical_reference_thermal_streamed",
        _mode_case(base, "upward", base_bandwidth, base_coupling, base_frequency),
        tier="numerical", gating=True, family="thermal",
    )
    zero = replace(
        thermal_reference,
        protocol=ProtocolConfig(name="zero"),
        model=replace(thermal_reference.model, device_shift=0.0, left_shift=0.0, right_shift=0.0),
    )
    add("numerical_zero_pulse_dense", zero,
        tier="numerical", gating=True, family="thermal")

    # Time-resolution cases use +/-40 so all requested time steps obey Nyquist.
    for dt in v.time_steps:
        dt_time_min = thermal_reference.projection.time_min
        dt_intervals = int(np.ceil((v.dense_time_max - dt_time_min) / float(dt)))
        dt_case = replace(
            thermal_reference,
            projection=replace(
                thermal_reference.projection,
                time_step=float(dt),
                time_max=dt_time_min + dt_intervals * float(dt),
            ),
            numerics=replace(
                thermal_reference.numerics,
                energy_min=-40.0,
                energy_max=40.0,
            ),
        )
        add(f"numerical_dt_{dt:g}", dt_case,
            tier="numerical", gating=True, family="thermal")

    for bound in v.reconstruction_energy_bounds:
        bound = float(bound)
        safe_dt = min(
            float(thermal_reference.projection.time_step),
            0.9 * np.pi / bound,
        )
        intervals = int(np.ceil((v.dense_time_max - thermal_reference.projection.time_min) / safe_dt))
        case = replace(
            thermal_reference,
            projection=replace(
                thermal_reference.projection,
                dense_validation=False,
                time_step=safe_dt,
                time_max=thermal_reference.projection.time_min + intervals * safe_dt,
            ),
            numerics=replace(
                thermal_reference.numerics,
                energy_min=-bound, energy_max=bound,
            ),
        )
        add(f"numerical_reconstruction_window_{bound:g}", case,
            tier="numerical", gating=True, family="thermal")
    for step in v.reconstruction_energy_steps:
        case = replace(
            thermal_reference,
            projection=replace(thermal_reference.projection, dense_validation=False),
            numerics=replace(thermal_reference.numerics, energy_step=float(step)),
        )
        add(f"numerical_reconstruction_step_{step:g}", case,
            tier="numerical", gating=True, family="thermal")
    for bound in v.current_energy_bounds:
        points = int(round(2.0 * float(bound) / thermal_reference.numerics.energy_step)) + 1
        case = replace(
            thermal_reference,
            projection=replace(thermal_reference.projection, dense_validation=False),
            numerics=replace(
                thermal_reference.numerics,
                current_energy_min=-float(bound), current_energy_max=float(bound),
                current_energy_points=points,
            ),
        )
        add(f"numerical_current_window_{bound:g}", case,
            tier="numerical", gating=True, family="thermal")
    for step in v.current_energy_steps:
        bound = max(v.current_energy_bounds)
        points = int(round(2.0 * float(bound) / float(step))) + 1
        case = replace(
            thermal_reference,
            projection=replace(thermal_reference.projection, dense_validation=False),
            numerics=replace(
                thermal_reference.numerics,
                current_energy_min=-float(bound), current_energy_max=float(bound),
                current_energy_points=points,
            ),
        )
        add(f"numerical_current_step_{step:g}", case,
            tier="numerical", gating=True, family="thermal")

    for time_min in v.prehistory_time_mins:
        case = replace(
            thermal_reference,
            numerics=replace(
                thermal_reference.numerics,
                energy_min=-v.compact_energy_bound,
                energy_max=v.compact_energy_bound,
                energy_step=0.05,
            ),
            projection=replace(thermal_reference.projection, time_min=float(time_min)),
        )
        add(f"numerical_prehistory_tmin_{time_min:g}", case,
            tier="numerical", gating=True, family="collision")
    for order in v.prehistory_quadrature_orders:
        case = replace(
            thermal_reference,
            numerics=replace(
                thermal_reference.numerics,
                collision_prehistory_quadrature_order=int(order),
                energy_min=-v.compact_energy_bound,
                energy_max=v.compact_energy_bound,
                energy_step=0.05,
            ),
        )
        add(f"numerical_prehistory_order_{order}", case,
            tier="numerical", gating=True, family="collision")
    for scale in v.prehistory_scales:
        case = replace(
            thermal_reference,
            numerics=replace(
                thermal_reference.numerics,
                collision_prehistory_scale=float(scale),
                energy_min=-v.compact_energy_bound,
                energy_max=v.compact_energy_bound,
                energy_step=0.05,
            ),
        )
        add(f"numerical_prehistory_scale_{scale:g}", case,
            tier="numerical", gating=True, family="collision")
    for eta in v.stationary_etas:
        case = replace(
            thermal_reference,
            projection=replace(thermal_reference.projection, dense_validation=False),
            numerics=replace(thermal_reference.numerics, eta=float(eta)),
        )
        add(f"numerical_stationary_eta_{eta:.0e}", case,
            tier="numerical", gating=True, family="thermal")
    for tolerance in v.mpm_tolerances:
        case = replace(
            thermal_reference,
            projection=replace(thermal_reference.projection, dense_validation=False),
            numerics=replace(
                thermal_reference.numerics,
                mpm_search_tolerance=float(tolerance),
                mpm_final_tolerance=float(tolerance),
            ),
        )
        add(f"numerical_mpm_{tolerance:.0e}", case,
            tier="numerical", gating=True, family="thermal")
    for time_min, time_max in zip(v.streamed_window_mins, v.streamed_window_maxs):
        case = _mode_case(
            base, "upward", base_bandwidth, base_coupling, base_frequency,
            time_min=time_min, time_max=time_max,
        )
        add(f"numerical_streamed_window_{time_min:g}_{time_max:g}", case,
            tier="numerical", gating=True, family="projection")

    def thermal(protocol: str, bandwidth: float, coupling: float, frequency: float) -> RunConfig:
        return _mode_case(base, protocol, bandwidth, coupling, frequency)

    # Pairwise thermal slices through the baseline point.
    if v.approximation_pairwise:
        for frequency in v.frequencies:
            for coupling in v.couplings_meV:
                add(f"thermal_frequency_coupling_f{frequency:g}_g{coupling:g}",
                    thermal("upward", base_bandwidth, coupling, frequency),
                    tier="approximation", gating=False, family="thermal")
            for bandwidth in v.bandwidths:
                add(f"thermal_frequency_bandwidth_f{frequency:g}_W{bandwidth:g}",
                    thermal("upward", bandwidth, base_coupling, frequency),
                    tier="approximation", gating=False, family="thermal")
            for protocol in v.protocols:
                add(f"thermal_frequency_protocol_f{frequency:g}_{protocol}",
                    thermal(protocol, base_bandwidth, base_coupling, frequency),
                    tier="approximation", gating=False, family="thermal")
        for coupling in v.couplings_meV:
            for bandwidth in v.bandwidths:
                add(f"thermal_coupling_bandwidth_g{coupling:g}_W{bandwidth:g}",
                    thermal("upward", bandwidth, coupling, base_frequency),
                    tier="approximation", gating=False, family="thermal")
            for protocol in v.protocols:
                add(f"thermal_coupling_protocol_g{coupling:g}_{protocol}",
                    thermal(protocol, base_bandwidth, coupling, base_frequency),
                    tier="approximation", gating=False, family="thermal")
        for bandwidth in v.bandwidths:
            for protocol in v.protocols:
                add(f"thermal_bandwidth_protocol_W{bandwidth:g}_{protocol}",
                    thermal(protocol, bandwidth, base_coupling, base_frequency),
                    tier="approximation", gating=False, family="thermal")

    # Fixed-N0 stress slices isolate frequency from the Bose thermal factor.
    if v.hot_stress_pairwise:
        for frequency in v.frequencies:
            for coupling in v.couplings_meV:
                add(f"hot_frequency_coupling_f{frequency:g}_g{coupling:g}",
                    _mode_case(base, "upward", base_bandwidth, coupling, frequency, hot=True),
                    tier="stress", gating=False, family="hot_N1")
            for protocol in v.protocols:
                add(f"hot_frequency_protocol_f{frequency:g}_{protocol}",
                    _mode_case(base, protocol, base_bandwidth, base_coupling, frequency, hot=True),
                    tier="stress", gating=False, family="hot_N1")
    for duration in v.square_durations:
        add(f"thermal_square_duration_{duration:g}", _square_case(base, duration, hot=False),
            tier="approximation", gating=False, family="thermal_square")
        add(f"hot_square_duration_{duration:g}", _square_case(base, duration, hot=True),
            tier="stress", gating=False, family="hot_N1_square")

    if v.compare_preparation_fixed:
        for frequency in v.frequencies:
            for protocol in v.protocols:
                add(
                    f"preparation_fixed_frequency_protocol_f{frequency:g}_{protocol}",
                    thermal(protocol, base_bandwidth, base_coupling, frequency),
                    tier="approximation", gating=False, family="preparation_fixed",
                    preparation_fixed=True,
                )
        for coupling in v.couplings_meV:
            for protocol in v.protocols:
                add(
                    f"preparation_fixed_coupling_protocol_g{coupling:g}_{protocol}",
                    thermal(protocol, base_bandwidth, coupling, base_frequency),
                    tier="approximation", gating=False, family="preparation_fixed",
                    preparation_fixed=True,
                )
    return cases


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_validation_run_status(session_path: str | Path, run_dir: str | Path, status: str) -> None:
    path = Path(session_path)
    if path.is_dir():
        path = path / "campaign.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target_path = Path(run_dir).expanduser().resolve()
    session_root = path.parent

    def resolve_record_path(value: str) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        candidates = [
            Path.cwd() / candidate,
            session_root / candidate,
            path.parent / candidate,
        ]
        for option in candidates:
            if option.exists():
                return option.resolve()
        return candidates[0].resolve()

    match = next(
        (
            item
            for item in manifest["jobs"]
            if item.get("run_directory")
            and resolve_record_path(item["run_directory"]) == target_path
        ),
        None,
    )
    if match is None:
        raise ValueError(f"Run {str(target_path)!r} is not registered in {path}.")
    match["status"] = status
    if status in ("accepted", "failed", "interrupted"):
        match["finished_at"] = _utc()
    manifest["status"] = "running" if status in ("running", "accepted") else status
    manifest["updated_at"] = _utc()
    atomic_json(path, manifest)


def run_validation_program(config: RunConfig, *, session_path: str | Path | None = None) -> list[Path]:
    """Run or resume the ordered validation ladder in a branded session tree."""
    from psscba.frontend.dashboard import CampaignDashboard

    require_shared_production_grid(config)
    planned = [_validation_case_fields(item) for item in validation_cases(config)]
    plan_digest = _validation_plan_digest(planned)
    root = Path(config.output_root)
    if session_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_dir = root / f"validation_{stamp}"
        manifest_path = session_dir / "campaign.json"
    else:
        requested = Path(session_path)
        manifest_path = requested if requested.suffix == ".json" else requested / "campaign.json"
        session_dir = manifest_path.parent

    if not manifest_path.exists():
        job_root = session_dir / "jobs"
        job_root.mkdir(parents=True, exist_ok=False)
        provenance = software_provenance()
        manifest_jobs = []
        for index, item in enumerate(planned):
            case = item.config
            manifest_jobs.append(
                {
                    "index": index,
                    "case_id": f"{index:03d}_{item.name}",
                    "name": item.name,
                    "tier": item.tier,
                    "gating": item.gating,
                    "family": item.family,
                    "validation_family": item.family,
                    "preparation_fixed": item.preparation_fixed,
                    "method": "preparation_fixed" if item.preparation_fixed else "psscba",
                    "occupation_mode": (
                        "explicit" if case.model.phonon_occupation is not None else "thermal"
                    ),
                    "status": "planned",
                    "run_directory": str((job_root / f"{index:03d}_{item.name}").resolve()),
                    "protocol": case.protocol.name,
                    "bandwidth": case.model.bandwidth,
                    "phonon_energy": case.model.phonon_energy,
                    "phonon_occupation": case.model.phonon_occupation,
                    "resolved_phonon_occupation": _resolved_occupation(case),
                    "coupling_meV": case.model.coupling_meV,
                    "coupling_over_gamma": case.model.coupling,
                    "coupling_over_omega": case.model.coupling / case.model.phonon_energy,
                    "created_at": None,
                    "started_at": None,
                    "finished_at": None,
                }
            )
        manifest = {
            "config_format": CONFIG_FORMAT,
            "kind": "validation",
            "software": provenance["software"],
            "status": "running",
            "created_at": _utc(),
            "updated_at": _utc(),
            "max_workers": config.campaign.max_workers,
            "blas_threads": config.campaign.blas_threads,
            "validation_plan_digest": plan_digest,
            "config": config.to_dict(),
            "jobs": manifest_jobs,
        }
        atomic_json(manifest_path, manifest)
        atomic_yaml(session_dir / "resolved.yaml", config.to_dict())
        dashboard = CampaignDashboard(session_dir, manifest)
        dashboard.start()
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_format") != CONFIG_FORMAT:
            raise ValueError("This validation session uses a retired configuration format.")
        if manifest.get("kind") != "validation":
            raise ValueError("The requested session is not a validation session.")
        if len(manifest.get("jobs", ())) != len(planned):
            raise ValueError("Validation session plan does not match this configuration.")
        if manifest.get("validation_plan_digest") != plan_digest:
            raise ValueError("Validation session plan does not match this configuration.")
        manifest["status"] = "running"
        manifest["updated_at"] = _utc()
        atomic_json(manifest_path, manifest)
        dashboard = CampaignDashboard(session_dir, manifest)
        dashboard.refresh(reason="validation resumed")

    dashboard.log_message(
        f"\n[{_utc()}] fast/direct oracle and production-grid conservation preflight started"
    )

    def preflight_progress(protocol: str, stage: str, details: dict) -> None:
        """Mirror preflight progress into both campaign views.

        The dense oracle has no per-job tracker, so without this callback the
        launcher appears frozen between the preflight start and its final
        report.  Keep the message scalar and flushed; detailed numerical data
        remains in ``preflight.json``.
        """
        index = details.get("protocol_index", "?")
        total = details.get("protocol_total", len(("upward", "downward", "square")))
        suffix: list[str] = []
        for key in (
            "compact_energy_points",
            "production_energy_points",
            "time_points",
            "batch_completed",
            "batch_total",
            "duration_seconds",
            "current_L_scaled_error",
            "current_R_scaled_error",
            "raw_continuity_scaled_error",
            "corrected_continuity_scaled_error",
        ):
            if key in details:
                value = details[key]
                if isinstance(value, float):
                    suffix.append(f"{key}={value:.3e}")
                else:
                    suffix.append(f"{key}={value}")
        line = f"PREFLIGHT [{index}/{total}] {protocol.upper():8s} {stage}"
        if suffix:
            line += "  " + " ".join(suffix)
        print(line, flush=True)
        dashboard.log_message(f"[{_utc()}] {line}")
        # Refreshing the HTML here makes the current preflight stage visible
        # even though no validation job has started yet.
        dashboard.manifest["preflight"] = {
            "protocol": protocol,
            "stage": stage,
            "details": details,
            "updated_at": _utc(),
        }
        dashboard.refresh(reason=f"preflight {protocol} {stage}")

    preflight = run_oracle_preflight(config, progress=preflight_progress)
    preflight_path = session_dir / "preflight.json"
    atomic_json(preflight_path, preflight)
    raw_warnings = {
        name: value
        for name, value in preflight.items()
        if name.endswith("_raw_continuity_scaled_error") and value >= 1e-4
    }
    if raw_warnings:
        dashboard.log_message(
            "[{}] raw g=0 continuity WARNING (diagnostic only): {}".format(
                _utc(),
                ", ".join(f"{name}={value:.3e}" for name, value in raw_warnings.items()),
            )
        )
    preflight_failures = {
        name: value
        for name, value in preflight.items()
        if not name.endswith("_raw_continuity_scaled_error")
        and value
        >= (1e-4 if name.endswith("_corrected_continuity_scaled_error") else 1e-5)
    }
    gating_failures: list[dict[str, object]] = []
    diagnostic_failures: list[dict[str, object]] = []
    if preflight_failures:
        dashboard.manifest["preflight_failures"] = preflight_failures
        dashboard.log_message(
            f"[{_utc()}] oracle/conservation preflight FAILED; continuing; "
            f"see {preflight_path.name}\n"
        )
    outputs: list[Path] = []
    # Dense references and the small foundation/collision references remain
    # serial.  Streamed numerical-resolution cases and the larger
    # approximation/stress maps are independent and can use the campaign
    # worker pool.  Legacy three-tuples stay serial for compatibility with
    # notebook callers and lightweight tests.
    parallel_indices = {
        index for index, item in enumerate(planned)
        if config.campaign.max_workers > 1
        and item.family != "legacy"
        and item.family not in {"g0", "collision"}
        and not item.config.projection.dense_validation
    }
    if parallel_indices:
        dashboard.log_message(
            f"[{_utc()}] launching {len(parallel_indices)} streamed diagnostic "
            f"cases with {config.campaign.max_workers} workers"
        )
        futures = {}
        with ProcessPoolExecutor(max_workers=config.campaign.max_workers) as executor:
            for index in sorted(parallel_indices):
                item = planned[index]
                record = dashboard.jobs[index]
                if record.get("status") in ("accepted", "complete"):
                    if record.get("run_directory"):
                        outputs.append(Path(record["run_directory"]))
                    continue
                case_id = str(record["case_id"])
                run_dir = Path(record["run_directory"])
                if not run_dir.exists():
                    run_dir = create_run_directory(
                        item.config, label=case_id, root=session_dir / "jobs", exact_name=True
                    )
                dashboard.update_status(case_id, "queued", created_at=_utc())
                record["started_at"] = _utc()
                future = executor.submit(
                    run_case,
                    item.config,
                    preparation_fixed=item.preparation_fixed,
                    run_dir=run_dir,
                    blas_threads=config.campaign.blas_threads,
                )
                futures[future] = (index, item, case_id, run_dir)
            pending = set(futures)
            while pending:
                completed, pending = wait(
                    pending,
                    timeout=min(
                        config.tracking.heartbeat_seconds,
                        config.tracking.live_refresh_seconds,
                    ),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    index, item, case_id, run_dir = futures[future]
                    try:
                        output = future.result()
                    except Exception as exc:
                        failure = {
                            "index": index,
                            "case_id": case_id,
                            "run_directory": str(run_dir.resolve()),
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "tier": item.tier,
                            "gating": item.gating,
                            "family": item.family,
                        }
                        dashboard.update_status(
                            case_id, "failed", finished_at=_utc(),
                            failure={"type": type(exc).__name__, "message": str(exc)},
                        )
                        (gating_failures if item.gating else diagnostic_failures).append(failure)
                        dashboard.manifest.setdefault("failures", []).append(failure)
                        dashboard.log_message(
                            f"[{_utc()}] validation case FAILED; continuing: {case_id} "
                            f"({type(exc).__name__}: {exc})"
                        )
                    else:
                        outputs.append(output)
                        dashboard.update_status(case_id, "accepted", finished_at=_utc())
                    dashboard.refresh(reason=f"streamed worker completed {case_id}")
    for index, item in enumerate(planned):
        if index in parallel_indices:
            continue
        name, case, preparation = item.name, item.config, item.preparation_fixed
        record = dashboard.jobs[index]
        if record.get("name") != name:
            raise ValueError(f"Validation session case {index} does not match {name!r}.")
        if record.get("status") in ("accepted", "complete"):
            if record.get("run_directory"):
                outputs.append(Path(record["run_directory"]))
            continue
        case_id = str(record["case_id"])
        dashboard.update_status(case_id, "starting", started_at=_utc())
        record["started_at"] = _utc()
        dashboard.manifest["current_case"] = index
        dashboard.refresh(reason=f"starting validation case {index + 1}/{len(planned)}")

        run_dir = Path(record["run_directory"])
        if not run_dir.exists():
            run_dir = create_run_directory(
                case,
                label=case_id,
                root=session_dir / "jobs",
                exact_name=True,
            )
        dashboard.update_status(case_id, "running", created_at=_utc())

        try:
            output = run_case(case, preparation_fixed=preparation, run_dir=run_dir)
        except (KeyboardInterrupt, SystemExit):
            dashboard.update_status(case_id, "interrupted", finished_at=_utc())
            dashboard.manifest["status"] = "interrupted"
            atomic_json(manifest_path, dashboard.manifest)
            dashboard.refresh(reason="validation interrupted")
            raise
        except Exception as exc:
            failure = {
                "index": index,
                "case_id": case_id,
                "run_directory": str(run_dir.resolve()),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            dashboard.update_status(
                case_id,
                "failed",
                finished_at=_utc(),
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            # A validation failure is case-local: keep the hard gate intact,
            # but continue through the planned grid/time-window ladder so the
            # campaign accumulates evidence instead of stopping at the first
            # failing reference case.
            failure["tier"] = item.tier
            failure["gating"] = item.gating
            failure["family"] = item.family
            (gating_failures if item.gating else diagnostic_failures).append(failure)
            dashboard.manifest.setdefault("failures", []).append(failure)
            dashboard.manifest["status"] = "running"
            atomic_json(manifest_path, dashboard.manifest)
            dashboard.log_message(
                f"[{_utc()}] validation case FAILED; continuing: {case_id} "
                f"({type(exc).__name__}: {exc})"
            )
            dashboard.refresh(reason=f"validation failed; continuing after {case_id}")
            continue
        outputs.append(output)
        dashboard.update_status(case_id, "accepted", finished_at=_utc())
        dashboard.refresh(reason=f"accepted validation case {index + 1}/{len(planned)}")
    _write_validation_products(session_dir, dashboard.manifest)
    convergence_rows, convergence_failures = _resolution_convergence(dashboard.manifest)
    atomic_json(session_dir / "validation_convergence.json", convergence_rows)
    if convergence_failures:
        gating_failures.extend(
            {
                "case_id": f"convergence_{item['family']}",
                "type": "ResolutionConvergenceError",
                "message": json.dumps(item["metrics"], sort_keys=True),
                "tier": "numerical",
                "gating": True,
                "family": "convergence",
            }
            for item in convergence_failures
        )
    if preflight_failures or gating_failures:
        dashboard.manifest["status"] = "failed_gates"
        dashboard.manifest["finished_at"] = _utc()
        dashboard.manifest["failure_summary"] = {
            "preflight": preflight_failures,
            "gating_count": len(gating_failures),
            "gating_case_ids": [str(item["case_id"]) for item in gating_failures],
            "diagnostic_count": len(diagnostic_failures),
            "diagnostic_case_ids": [str(item["case_id"]) for item in diagnostic_failures],
        }
        dashboard.manifest.pop("current_case", None)
        atomic_json(manifest_path, dashboard.manifest)
        dashboard.log_message(
            f"[{_utc()}] validation completed with "
            f"{len(gating_failures) + len(diagnostic_failures)} failed case(s); "
            "all planned cases were attempted"
        )
        dashboard.finish()
        raise RuntimeError(
            f"Validation completed with {len(gating_failures) + len(diagnostic_failures)} "
            f"failed case(s): {len(gating_failures)} gating and "
            f"{len(diagnostic_failures)} diagnostic; "
            f"inspect {manifest_path} and each failed job's failure.json."
        )
    if diagnostic_failures:
        dashboard.manifest["status"] = "complete_with_diagnostic_failures"
        dashboard.manifest["finished_at"] = _utc()
        dashboard.manifest["failure_summary"] = {
            "gating_count": 0,
            "diagnostic_count": len(diagnostic_failures),
            "diagnostic_case_ids": [str(item["case_id"]) for item in diagnostic_failures],
        }
        dashboard.manifest.pop("current_case", None)
        atomic_json(manifest_path, dashboard.manifest)
        dashboard.log_message(
            f"[{_utc()}] validation completed with {len(diagnostic_failures)} "
            "diagnostic failure(s); production gates passed"
        )
        dashboard.finish()
        return outputs
    dashboard.manifest["status"] = "complete"
    dashboard.manifest["finished_at"] = _utc()
    dashboard.manifest.pop("current_case", None)
    atomic_json(manifest_path, dashboard.manifest)
    dashboard.finish()
    return outputs
