from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping
from time import perf_counter
import inspect
from typing import Callable

import numpy as np

from psscba.frontend.configuration import CaseConfig, RunConfig
from psscba.backend.observables import ObservableEvaluator
from psscba.backend.poles.minimal import PoleBasis, PoleCache, build_pole_cache
from psscba.backend.protocols.square import SquareKernelCacheStore
from psscba.backend.reconstruction.engine import solve_fixed_kernel_trajectory
from psscba.backend.stationary.initializer import (
    StationaryInitializer,
    make_system,
    mix_kernels,
)
from psscba.backend.self_energy import (
    SCBAFunctional,
    kernel_residual,
    kernel_residual_details,
    scba_candidate,
    weighted_rms,
)
from psscba.types import (
    FixedKernelTrajectory,
    OuterIteration,
    PSSCBAResult,
    StationaryKernel,
)


def _readonly(values, dtype=float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CaseWorkspace:
    """Case-wide immutable grids and quadrature data reused by every step."""

    config: CaseConfig
    energy: np.ndarray
    time: np.ndarray
    energy_weights: np.ndarray
    time_weights: np.ndarray
    system: object
    lead_linewidths: Mapping[str, np.ndarray]
    estimated_peak_bytes: int
    square_cache_store: SquareKernelCacheStore

    @classmethod
    def build(cls, config: CaseConfig | RunConfig) -> "CaseWorkspace":
        config = config.case if isinstance(config, RunConfig) else config
        config.validate()
        energy = config.numerics.energy_grid()
        time = config.projection.time_grid()
        energy_weights = np.full(len(energy), float(energy[1] - energy[0]))
        energy_weights[[0, -1]] *= 0.5
        time_weights = np.full(len(time), float(time[1] - time[0]))
        time_weights[[0, -1]] *= 0.5
        system = make_system(config)
        lead_linewidths = {
            lead: _readonly(
                system.Gamma0(lead) * system.W**2 / (energy**2 + system.W**2)
            )
            for lead in system.lead_names
        }
        estimated_peak_bytes = int(
            16 * len(time) * len(time)
            + 16 * len(time) * len(energy) * (len(system.lead_names) + 2)
        )
        return cls(
            config=config,
            energy=_readonly(energy),
            time=_readonly(time),
            energy_weights=_readonly(energy_weights),
            time_weights=_readonly(time_weights),
            system=system,
            lead_linewidths=MappingProxyType(lead_linewidths),
            estimated_peak_bytes=estimated_peak_bytes,
            square_cache_store=SquareKernelCacheStore(),
        )


class FixedKernelSolver:
    """One complete pulse solve for one immutable installed kernel."""

    def __init__(
        self,
        workspace: CaseWorkspace,
        kernel: StationaryKernel,
        previous_basis: PoleBasis | PoleCache | None = None,
    ) -> None:
        self.workspace = workspace
        self.kernel = kernel
        self.pole_cache = None
        self.previous_basis = previous_basis
        self.pole_fit_duration_seconds: float | None = None

    def solve_trajectory(self, *, dense=None, progress=None) -> FixedKernelTrajectory:
        if self.pole_cache is None:
            _progress(progress, "pole_basis_reuse", attempt=self.previous_basis is not None)
            _progress(progress, "mpm_fit", reused=self.previous_basis is not None)
            self.workspace.system.mpm_tol = self.workspace.config.numerics.mpm_search_tolerance
            fit_start = perf_counter()
            self.pole_cache = build_pole_cache(
                self.workspace.system,
                self.kernel,
                previous_basis=self.previous_basis,
                progress=progress,
            )
            self.pole_fit_duration_seconds = perf_counter() - fit_start
        return solve_fixed_kernel_trajectory(
            self.workspace.config,
            self.kernel,
            dense=dense,
            progress=progress,
            system=self.workspace.system,
            pole_cache=self.pole_cache,
            time_grid=self.workspace.time,
            energy_grid=self.workspace.energy,
            square_cache_store=self.workspace.square_cache_store,
            emit_setup_progress=False,
        )


@dataclass(frozen=True)
class IterationOutcome:
    trajectory: FixedKernelTrajectory
    candidate: StationaryKernel
    residual_retarded: float
    residual_lesser: float
    residual_retarded_numerator: float
    residual_retarded_denominator: float
    residual_lesser_numerator: float
    residual_lesser_denominator: float
    pole_cache: PoleCache
    pole_fit_duration_seconds: float | None = None
    square_cache_hits: int | None = None


@dataclass(frozen=True)
class SCBAIteration:
    """Kernel-local state for exactly one projected SCBA outer iteration."""

    workspace: CaseWorkspace
    installed: StationaryKernel
    index: int
    previous_basis: PoleBasis | PoleCache | None = None

    def solve(self, *, progress=None) -> IterationOutcome:
        solver = FixedKernelSolver(
            self.workspace, self.installed, previous_basis=self.previous_basis
        )
        trajectory = solver.solve_trajectory(progress=progress)
        _progress(progress, "scba_candidate", outer_step=self.index)
        functional = SCBAFunctional(self.workspace.config)
        candidate = functional.candidate(trajectory, self.installed)
        residual = functional.residual(self.installed, candidate)
        return IterationOutcome(
            trajectory,
            candidate,
            residual.retarded,
            residual.lesser,
            residual.retarded_numerator,
            residual.retarded_denominator,
            residual.lesser_numerator,
            residual.lesser_denominator,
            solver.pole_cache,
            solver.pole_fit_duration_seconds,
            getattr(self.workspace.square_cache_store, "size", None),
        )


class SingleCaseSolver:
    """Public backend solver unit for one fully resolved parameter case."""

    def __init__(self, config: CaseConfig | RunConfig) -> None:
        self.workspace = CaseWorkspace.build(config)

    def solve(self, **kwargs) -> PSSCBAResult:
        return solve_psscba(self.workspace.config, workspace=self.workspace, **kwargs)


CheckpointCallback = Callable[[StationaryKernel, tuple[OuterIteration, ...]], None]
ProgressCallback = Callable[[str, dict], None]


def _progress(callback, phase: str, **details) -> None:
    if callback is not None:
        callback(phase, details)


def _write_checkpoint(callback, kernel, history, basis) -> None:
    if callback is None:
        return
    try:
        accepts_basis = len(inspect.signature(callback).parameters) >= 3
    except (TypeError, ValueError):
        accepts_basis = False
    if accepts_basis:
        callback(kernel, history, basis)
    else:
        callback(kernel, history)


def _green_change(current, previous, grid, floor):
    if previous is None:
        return None, None
    return (
        weighted_rms(current.projected.green_retarded_energy - previous.projected.green_retarded_energy, grid)
        / max(weighted_rms(current.projected.green_retarded_energy, grid), floor),
        weighted_rms(current.projected.green_lesser_energy - previous.projected.green_lesser_energy, grid)
        / max(weighted_rms(current.projected.green_lesser_energy, grid), floor),
    )


def _same_tolerance(left: float | None, right: float) -> bool:
    if left is None:
        return False
    return abs(float(left) - float(right)) <= 1e-18 * max(
        1.0, abs(float(right))
    )


def _completed_stage(
    history: list[OuterIteration] | tuple[OuterIteration, ...],
    *,
    mpm_tolerance: float,
    outer_tolerance: float,
) -> bool:
    """Whether the checkpoint already records an accepted stage endpoint."""
    if not history:
        return False
    latest = history[-1]
    return bool(
        _same_tolerance(latest.mpm_tolerance, mpm_tolerance)
        and max(latest.residual_retarded, latest.residual_lesser) < outer_tolerance
    )


def solve_fixed_kernel(
    config: CaseConfig | RunConfig,
    kernel: StationaryKernel,
    *,
    dense: bool | None = None,
    observables: bool = True,
    progress: ProgressCallback | None = None,
) -> FixedKernelTrajectory:
    config = config.case if isinstance(config, RunConfig) else config
    # Production pole/Filon currents consume the factorized trajectory and
    # do not require dense two-time Green functions.  Dense storage is only
    # requested explicitly for projection/filtering or validation diagnostics.
    trajectory = solve_fixed_kernel_trajectory(
        config,
        kernel,
        dense=dense,
        progress=progress,
        amplitude_frequency=(
            float(config.model.phonon_energy)
            if config.plotting.enabled and observables else None
        ),
    )
    if observables:
        _progress(progress, "currents")
        return ObservableEvaluator(config, kernel).evaluate(
            trajectory,
            progress=progress,
        )
    return trajectory


def solve_preparation_fixed(config: CaseConfig | RunConfig, *, progress: ProgressCallback | None = None) -> PSSCBAResult:
    config = config.case if isinstance(config, RunConfig) else config
    kernel = StationaryInitializer(config).preparation_fixed()
    trajectory = solve_fixed_kernel_trajectory(
        config,
        kernel,
        dense=config.projection.dense_validation,
        progress=progress,
        amplitude_frequency=(float(config.model.phonon_energy) if config.plotting.enabled else None),
    )
    evaluator = ObservableEvaluator(config, kernel)
    if config.projection.dense_validation:
        _progress(progress, "discarded_sector_diagnostics")
        trajectory = evaluator.with_discarded_sectors(trajectory)
    else:
        trajectory = replace(
            trajectory,
            diagnostics={
                **trajectory.diagnostics,
                "dense_validation_performed": False,
                "discarded_sector_diagnostics": "skipped_production",
            },
        )
    _progress(progress, "currents")
    trajectory = evaluator.evaluate(trajectory, progress=progress)
    return PSSCBAResult(
        converged=True,
        kernel=kernel,
        trajectory=trajectory,
        history=(),
        initial_kernel=kernel,
        diagnostics={"method": "preparation_fixed"},
    )


def _outer_stage(
    config: CaseConfig,
    workspace: CaseWorkspace,
    installed: StationaryKernel,
    history: list[OuterIteration],
    *,
    checkpoint: CheckpointCallback | None,
    progress: ProgressCallback | None,
    step_completed: Callable[[OuterIteration, tuple[OuterIteration, ...]], None] | None,
    cumulative_start: float,
    max_iterations: int | None = None,
    previous_basis: PoleBasis | PoleCache | None = None,
) -> tuple[StationaryKernel, FixedKernelTrajectory, bool, PoleCache | None]:
    previous: FixedKernelTrajectory | None = None
    limit = config.numerics.outer_max_iter if max_iterations is None else max_iterations
    if limit < 1:
        raise RuntimeError("The total PS-SCBA outer iteration budget is exhausted.")
    cumulative_elapsed = cumulative_start
    latest_pole_cache: PoleCache | None = None
    for offset in range(limit):
        step_start = perf_counter()
        outer_step = len(history)
        _progress(progress, "initialization", outer_step=outer_step)
        iteration = SCBAIteration(workspace, installed, outer_step, previous_basis)
        outcome = iteration.solve(progress=progress)
        latest_pole_cache = outcome.pole_cache
        previous_basis = latest_pole_cache
        trajectory = outcome.trajectory
        candidate = outcome.candidate
        residual_r = outcome.residual_retarded
        residual_l = outcome.residual_lesser
        numerator_r = outcome.residual_retarded_numerator
        denominator_r = outcome.residual_retarded_denominator
        numerator_l = outcome.residual_lesser_numerator
        denominator_l = outcome.residual_lesser_denominator
        change_r, change_l = _green_change(
            trajectory, previous, installed.energy, config.numerics.green_floor
        )
        from psscba.frontend.tracking import resource_sample

        resources = resource_sample()
        # ``ru_maxrss`` is process-local.  On resume a fresh process reports a
        # lower peak than the original process, which makes the convergence
        # history look as if memory suddenly dropped.  Preserve the run-wide
        # high-water mark while retaining the current RSS sample.
        prior_peak = max(
            (int(item.peak_rss_bytes) for item in history if item.peak_rss_bytes is not None),
            default=0,
        )
        if resources.get("peak_rss_bytes") is not None:
            resources["peak_rss_bytes"] = max(
                int(resources["peak_rss_bytes"]), prior_peak
            )
        step_duration = perf_counter() - step_start
        cumulative_elapsed += step_duration
        record = OuterIteration(
            iteration=len(history),
            residual_retarded=residual_r,
            residual_lesser=residual_l,
            projected_green_retarded_change=(None if change_r is None else float(change_r)),
            projected_green_lesser_change=(None if change_l is None else float(change_l)),
            projected_occupation=trajectory.projected.occupation,
            mpm_tolerance=config.numerics.mpm_search_tolerance,
            step_duration_seconds=step_duration,
            cumulative_duration_seconds=cumulative_elapsed,
            rss_bytes=resources["rss_bytes"],
            peak_rss_bytes=resources["peak_rss_bytes"],
            sigma_hartree_installed=installed.sigma_hartree,
            sigma_hartree_candidate=candidate.sigma_hartree,
            residual_retarded_numerator=numerator_r,
            residual_retarded_denominator=denominator_r,
            residual_lesser_numerator=numerator_l,
            residual_lesser_denominator=denominator_l,
            pole_fit_duration_seconds=outcome.pole_fit_duration_seconds,
            pole_basis_reused=bool(getattr(outcome.pole_cache, "pole_basis_reused", False)),
            square_cache_hits=outcome.square_cache_hits,
        )
        history.append(record)
        accepted = max(residual_r, residual_l) < config.numerics.outer_tolerance
        if not accepted:
            _progress(progress, "mixing_checkpoint", outer_step=outer_step)
            installed = mix_kernels(
                installed,
                candidate,
                config.numerics.outer_mixing,
                iteration=len(history),
            )
            if checkpoint is not None and (
                (offset + 1) % config.numerics.checkpoint_every == 0
            ):
                _write_checkpoint(
                    checkpoint, installed, tuple(history), latest_pole_cache.basis
                )
        elif checkpoint is not None:
            _write_checkpoint(
                checkpoint, installed, tuple(history), latest_pole_cache.basis
            )
        # Include mixing/checkpoint work in the reported SCF step timing.
        completed_duration = perf_counter() - step_start
        # ``history`` already contains the complete pre-existing history on a
        # resume.  Adding ``cumulative_start`` again double-counts every old
        # step and creates artificial jumps in cumulative wall time.  Rebuild
        # the scalar from the per-step durations instead.
        completed_cumulative = sum(
            item.step_duration_seconds or 0.0 for item in history[:-1]
        ) + completed_duration
        record = replace(
            record,
            step_duration_seconds=completed_duration,
            cumulative_duration_seconds=completed_cumulative,
        )
        history[-1] = record
        _progress(
            progress,
            "scba_candidate",
            outer_step=outer_step,
            residual_retarded=residual_r,
            residual_lesser=residual_l,
            loss=max(residual_r, residual_l),
            projected_occupation=trajectory.projected.occupation,
            sigma_hartree_installed=installed.sigma_hartree,
            sigma_hartree_candidate=candidate.sigma_hartree,
        )
        if step_completed is not None:
            step_completed(record, tuple(history))
        if accepted:
            return installed, trajectory, True, latest_pole_cache
        previous = trajectory
    return installed, trajectory, False, latest_pole_cache


def solve_psscba(
    config: CaseConfig | RunConfig,
    *,
    workspace: CaseWorkspace | None = None,
    initial_kernel: StationaryKernel | None = None,
    initial_basis: PoleBasis | None = None,
    initial_history: tuple[OuterIteration, ...] = (),
    checkpoint: CheckpointCallback | None = None,
    progress: ProgressCallback | None = None,
    step_completed: Callable[[OuterIteration, tuple[OuterIteration, ...]], None] | None = None,
) -> PSSCBAResult:
    """Solve the installed-and-re-evaluated PS-SCBA outer fixed point."""
    config = config.case if isinstance(config, RunConfig) else config
    config.validate()
    workspace = CaseWorkspace.build(config) if workspace is None else workspace
    if workspace.config != config:
        raise ValueError("CaseWorkspace was built for a different resolved case.")
    _progress(progress, "initialization")
    installed = (
        StationaryInitializer(config).zero_pulse()
        if initial_kernel is None
        else initial_kernel
    )
    starting_kernel = installed
    history: list[OuterIteration] = list(initial_history)
    cumulative_start = sum(item.step_duration_seconds or 0.0 for item in history)
    final_stage_required = bool(
        config.model.coupling_meV != 0.0
        and config.numerics.mpm_final_tolerance
        != config.numerics.mpm_search_tolerance
    )
    # The last completed history row identifies the stage of a resumable
    # checkpoint.  A strict-stage checkpoint must continue strictly; a
    # search-stage checkpoint that already met the outer tolerance must not
    # perform one redundant loose iteration before strict finalization.
    active_tolerance = config.numerics.mpm_search_tolerance
    if (
        initial_history
        and final_stage_required
        and _same_tolerance(
            initial_history[-1].mpm_tolerance,
            config.numerics.mpm_final_tolerance,
        )
    ):
        active_tolerance = config.numerics.mpm_final_tolerance
    active_config = replace(
        config,
        numerics=replace(
            config.numerics,
            mpm_search_tolerance=active_tolerance,
        ),
    )
    active_workspace = (
        workspace if active_config == config else CaseWorkspace.build(active_config)
    )
    trajectory: FixedKernelTrajectory | None = None
    pole_cache: PoleCache | None = None
    pole_seed: PoleBasis | PoleCache | None = initial_basis
    if _completed_stage(
        history,
        mpm_tolerance=active_tolerance,
        outer_tolerance=config.numerics.outer_tolerance,
    ):
        converged = True
        _progress(
            progress,
            "initialization",
            message=(
                "resume checkpoint already satisfies this MPM stage; "
                "skipping redundant outer iteration"
            ),
        )
    else:
        installed, trajectory, converged, pole_cache = _outer_stage(
            active_config,
            active_workspace,
            installed,
            history,
            checkpoint=checkpoint,
            progress=progress,
            step_completed=step_completed,
            cumulative_start=cumulative_start,
            max_iterations=config.numerics.outer_max_iter - len(history),
            previous_basis=pole_seed,
        )
        if pole_cache is not None:
            pole_seed = pole_cache
    if not converged:
        if trajectory is None:
            raise RuntimeError("PS-SCBA stopped without a fixed-kernel trajectory.")
        return PSSCBAResult(
            converged=False,
            kernel=installed,
            trajectory=trajectory,
            history=tuple(history),
            initial_kernel=starting_kernel,
            diagnostics={"failure": "outer_search_nonconvergence"},
        )

    production_config = active_config
    production_workspace = active_workspace
    if (
        final_stage_required
        and not _same_tolerance(
            active_tolerance, config.numerics.mpm_final_tolerance
        )
    ):
        final_config = replace(
            config,
            numerics=replace(
                config.numerics,
                mpm_search_tolerance=config.numerics.mpm_final_tolerance,
            ),
        )
        final_workspace = CaseWorkspace.build(final_config)
        installed, trajectory, converged, pole_cache = _outer_stage(
            final_config,
            final_workspace,
            installed,
            history,
            checkpoint=checkpoint,
            progress=progress,
            step_completed=step_completed,
            cumulative_start=sum(item.step_duration_seconds or 0.0 for item in history),
            max_iterations=config.numerics.outer_max_iter - len(history),
            previous_basis=pole_seed,
        )
        if pole_cache is not None:
            pole_seed = pole_cache
        production_config = final_config
        production_workspace = final_workspace
    diagnostics = {
        "method": "psscba",
        "scba_mode": (
            "noninteracting_exact" if config.model.coupling_meV == 0.0
            else "self_consistent"
        ),
        "outer_iterations": len(history),
        "final_retarded_residual": history[-1].residual_retarded,
        "final_lesser_residual": history[-1].residual_lesser,
        "projected_occupation": history[-1].projected_occupation,
    }
    if converged:
        # The accepted trajectory is recomputed with the final MPM tolerance,
        # then the installed kernel is tested once more against that trajectory.
        # A resumable checkpoint may already satisfy the outer tolerance and
        # therefore skip ``_outer_stage``.  Build the final accepted cache
        # explicitly in that path so the rerun uses exactly the same fitted
        # residues, and so the cache can be persisted below.
        if pole_cache is None:
            pole_cache = build_pole_cache(
                production_workspace.system,
                installed,
                previous_basis=pole_seed,
                progress=progress,
            )
        dense_validation = bool(production_config.projection.dense_validation)
        trajectory = solve_fixed_kernel_trajectory(
            production_config, installed, dense=dense_validation, progress=progress,
            pole_cache=pole_cache,
            square_cache_store=production_workspace.square_cache_store,
            previous_basis=(None if pole_cache is not None else pole_seed),
            amplitude_frequency=(
                float(production_config.model.phonon_energy)
                if production_config.plotting.enabled else None
            ),
        )
        final_candidate = scba_candidate(production_config, trajectory, installed)
        final_r, final_l = kernel_residual(
            installed, final_candidate, production_config.numerics.sigma_floor
        )
        if max(final_r, final_l) >= config.numerics.outer_tolerance:
            converged = False
        diagnostics["final_retarded_residual"] = final_r
        diagnostics["final_lesser_residual"] = final_l
        installed = replace(
            installed,
            projected_green_retarded=trajectory.projected.green_retarded_energy,
            projected_green_lesser=trajectory.projected.green_lesser_energy,
        )
        evaluator = ObservableEvaluator(production_config, installed)
        if dense_validation:
            _progress(progress, "discarded_sector_diagnostics")
            trajectory = evaluator.with_discarded_sectors(trajectory)
        else:
            trajectory = replace(
                trajectory,
                diagnostics={
                    **trajectory.diagnostics,
                    "dense_validation_performed": False,
                    "discarded_sector_diagnostics": "skipped_production",
                },
            )
        _progress(progress, "currents")
        trajectory = evaluator.evaluate(
            trajectory,
            progress=progress,
            previous_basis=pole_cache if pole_cache is not None else pole_seed,
        )
        # Keep the latest accepted basis in the resumable checkpoint even when
        # this invocation only performed the final accepted-kernel rerun.
        _write_checkpoint(checkpoint, installed, tuple(history), pole_cache.basis)
    return PSSCBAResult(
        converged=converged,
        kernel=installed,
        trajectory=trajectory,
        history=tuple(history),
        initial_kernel=starting_kernel,
        diagnostics=diagnostics,
    )
