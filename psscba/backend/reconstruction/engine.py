from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

from psscba.backend.model.distributions import expc, fermi_dirac
from psscba.backend.poles.minimal import (
    Gbiased_R_mpm,
    Gfr_R_mpm,
    PoleCache,
    RETARDED_BOUNDARY_EPS,
    build_pole_cache,
)
from psscba.backend.protocols.analytic import (
    B_up,
    D_down,
    D_up,
    upward_lead_vertex,
    downward_lead_vertex,
    upward_interaction_vertex,
    downward_interaction_vertex,
    linewidth,
    integrate_cauchy_linear,
    psi_lead_down_batch,
    psi_lead_square_batch,
    psi_lead_up_batch,
    time_derivative,
)
from psscba.backend.protocols.analytic import continuity_residual
from psscba.backend.protocols.square import (
    D_square,
    SquareKernelCacheStore,
)
from psscba.backend.protocols.strategies import strategy_for
from psscba.frontend.configuration import CaseConfig
from psscba.backend.projection.core import (
    autocorrelation_projection,
    dense_lesser_from_sources,
    direct_toeplitz_projection,
    make_projection_result,
    projector_errors,
    reconstruct_retarded_rows,
    trapezoid_weights,
    toeplitz_from_lags,
)
from psscba.backend.stationary.initializer import make_system
from psscba.backend.model.units import current_to_uA, time_to_ps
from psscba.types import FixedKernelTrajectory, StationaryKernel


def _progress(callback, phase: str, **details) -> None:
    if callback is not None:
        callback(phase, details)


def _drive_integral(config: CaseConfig, time: np.ndarray, lead: str) -> np.ndarray:
    shift = config.model.left_shift if lead == "L" else config.model.right_shift
    protocol = config.protocol.name
    if protocol == "upward":
        return shift * np.maximum(time, 0.0)
    if protocol == "downward":
        return shift * np.minimum(time, 0.0)
    if protocol == "square":
        return shift * np.clip(time, 0.0, float(config.protocol.duration))
    return np.zeros_like(time)


def _current_energy_grid(config: CaseConfig, trajectory_energy: np.ndarray) -> np.ndarray:
    """Return the energy grid used by the current quadrature.

    Production configurations set no independent current bounds, so currents
    are integrated on exactly the reconstruction grid.  Explicit bounds remain
    available for isolated convergence/reference studies, but are never the
    default production path.
    """
    points = config.numerics.current_energy_points
    lower = config.numerics.current_energy_min
    upper = config.numerics.current_energy_max
    if points is None and lower is None and upper is None:
        return np.asarray(trajectory_energy, dtype=float)
    # A point count without explicit bounds retains the historical behavior:
    # resample the current integral over the reconstruction interval.  This is
    # useful for compact tests and old callers; production reference grids set
    # all three values explicitly.
    if lower is None and upper is None and points is not None:
        lower = float(np.asarray(trajectory_energy)[0])
        upper = float(np.asarray(trajectory_energy)[-1])
    if points is None or lower is None or upper is None:
        raise ValueError(
            "current_energy_points and current_energy_min/max must be set together."
        )
    grid = np.linspace(float(lower), float(upper), int(points))
    if len(grid) < 2 or not np.all(np.diff(grid) > 0.0):
        raise ValueError("Current-energy grid must be strictly increasing.")
    return grid


def _pulse_amplitudes(config: CaseConfig, system, cache, time, energy, *, progress=None,
                      square_cache_store=None, c_omega: float = 0.0):
    amplitudes = strategy_for(
        config, system, cache, energy,
        square_cache_store=square_cache_store,
        progress=progress,
    ).solve(
        time,
        progress=progress,
        c_omega=c_omega,
    )
    return amplitudes.a, amplitudes.c, amplitudes.protocol_cache


def _lesser_sources(config, system, kernel, time, energy, a_values, c_zero):
    energy = np.asarray(energy, dtype=float)
    if len(energy) < 2 or np.any(np.diff(energy) <= 0.0):
        raise ValueError("Lesser-source energies must be strictly increasing.")
    quadrature = np.empty(len(energy), dtype=float)
    quadrature[0] = 0.5 * (energy[1] - energy[0])
    quadrature[-1] = 0.5 * (energy[-1] - energy[-2])
    quadrature[1:-1] = 0.5 * (energy[2:] - energy[:-2])
    quadrature /= 2.0 * np.pi
    sources: list[tuple[np.ndarray, np.ndarray]] = []
    for lead in system.lead_names:
        chi = _drive_integral(config, time, lead)
        phase = np.exp(-1j * (time[:, None] * energy[None, :] + chi[:, None]))
        factor = phase * a_values[lead]
        source_weight = (
            1j
            * fermi_dirac(energy, system.beta_fc(lead), system.mu_fc(lead))
            * np.real(linewidth(system, lead, energy))
            * quadrature
        )
        sources.append((factor, source_weight))

    sigma_lesser = np.interp(
        energy, kernel.energy, kernel.sigma_lesser.imag, left=0.0, right=0.0
    ) * 1j
    interaction_factor = np.exp(-1j * time[:, None] * energy[None, :]) * c_zero
    sources.append((interaction_factor, sigma_lesser * quadrature))
    return sources


def explicit_phonon_lesser_from_branches(
    time: np.ndarray,
    energy: np.ndarray,
    branch_amplitudes: dict[float, np.ndarray],
    projected_lesser: np.ndarray,
    coupling: float,
    phonon_energy: float,
    phonon_occupation: float,
) -> np.ndarray:
    """Build the explicit two-branch SCBA lesser contribution.

    This is an independent validation/reference contraction for the
    manuscript representation.  ``branch_amplitudes[eta*omega]`` contains
    C(t, eta*omega, E'), while ``projected_lesser`` is sampled at E'.  The
    production mixed-kernel path remains :func:`_lesser_sources`, which uses
    the installed Sigma^< directly.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    energy = np.asarray(energy, dtype=float).reshape(-1)
    projected_lesser = np.asarray(projected_lesser, dtype=np.complex128).reshape(-1)
    if len(energy) != len(projected_lesser):
        raise ValueError("Projected lesser and branch energy grids must match.")
    quadrature = trapezoid_weights(energy) / (2.0 * np.pi)
    result = np.zeros((len(time), len(time)), dtype=np.complex128)
    g2 = float(coupling) ** 2
    omega = abs(float(phonon_energy))
    n0 = float(phonon_occupation)
    for frequency, amplitudes in branch_amplitudes.items():
        frequency = float(frequency)
        if not np.isclose(abs(frequency), omega, rtol=0.0, atol=1e-12):
            raise ValueError("Branch frequency must equal +/- phonon_energy.")
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        if amplitudes.shape != (len(time), len(energy)):
            raise ValueError("Branch amplitudes have incompatible dimensions.")
        eta = -1.0 if frequency < 0.0 else 1.0
        branch_weight = (n0 + 1.0) if eta < 0.0 else n0
        phase = np.exp(-1j * (energy + frequency)[:, None] * time[None, :]).T
        factors = phase * amplitudes
        result += (factors * (branch_weight * g2 * projected_lesser * quadrature)) @ factors.conj().T
    return result


def installed_lesser_from_factorized_source(
    time: np.ndarray,
    energy: np.ndarray,
    c_zero: np.ndarray,
    sigma_lesser: np.ndarray,
) -> np.ndarray:
    """Contract an arbitrary installed lesser kernel through C(t,0,E)."""
    time = np.asarray(time, dtype=float).reshape(-1)
    energy = np.asarray(energy, dtype=float).reshape(-1)
    c_zero = np.asarray(c_zero, dtype=np.complex128)
    sigma_lesser = np.asarray(sigma_lesser, dtype=np.complex128).reshape(-1)
    if c_zero.shape != (len(time), len(energy)):
        raise ValueError("Installed C amplitudes have incompatible dimensions.")
    if len(sigma_lesser) != len(energy):
        raise ValueError("Installed lesser kernel and energy grid must match.")
    quadrature = trapezoid_weights(energy) / (2.0 * np.pi)
    phase = np.exp(-1j * energy[None, :] * time[:, None])
    factors = phase * c_zero
    return (factors * (sigma_lesser * quadrature)) @ factors.conj().T


def _lesser_energy_grid(
    config: CaseConfig, cache, *, allow_refinement: bool = False
) -> np.ndarray:
    """Return the quadrature used by the lesser source factorization.

    Production deliberately uses one uniform energy grid for the stationary
    kernel, all lesser sources, reconstruction, and currents.  An adaptive
    pole-centred grid is retained as an explicit diagnostic/reference option;
    it must never be selected implicitly because mixing it with the uniform
    reconstruction grid changes the discretized SCBA functional.
    """
    base = config.numerics.energy_grid()
    if not allow_refinement:
        return base
    base_step = float(config.numerics.energy_step)
    lower = float(base[0])
    upper = float(base[-1])
    shifts = {
        0.0,
        -float(config.model.phonon_energy),
        float(config.model.phonon_energy),
    }
    shifts.update(float(config.model.left_shift) * sign for sign in (-1.0, 1.0))
    shifts.update(float(config.model.right_shift) * sign for sign in (-1.0, 1.0))

    additions: list[np.ndarray] = []
    for poles in (cache.xi_unbiased, cache.xi_biased):
        for pole in np.asarray(poles, dtype=np.complex128):
            width = max(float(-pole.imag), 64.0 * np.finfo(float).eps)
            local_step = min(base_step, width / 6.0)
            if local_step >= 0.95 * base_step:
                continue
            # Lorentzian pole tails decay only algebraically in energy.  A
            # short patch resolves the peak but leaves a visible trapezoidal
            # bias where it rejoins the coarse grid; 64 widths makes that
            # splice error negligible at the campaign tolerances.
            radius = max(64.0 * width, 8.0 * base_step)
            for shift in shifts:
                center = float(pole.real + shift)
                left = max(lower, center - radius)
                right = min(upper, center + radius)
                if left >= right:
                    continue
                intervals = max(2, int(np.ceil((right - left) / local_step)))
                additions.append(np.linspace(left, right, intervals + 1))
    if not additions:
        return base
    candidates = np.sort(np.concatenate((base, *additions)))
    merge_tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, abs(lower), abs(upper)
    )
    keep = np.r_[True, np.diff(candidates) > merge_tolerance]
    grid = candidates[keep]
    grid.setflags(write=False)
    return grid


def _stationary_retarded(poles, residues, lag):
    lag = np.asarray(lag, dtype=float)
    values = -1j * np.sum(
        residues[None, :] * np.exp(-1j * lag[:, None] * poles[None, :]),
        axis=1,
    )
    values[lag == 0.0] *= 0.5
    return values


def _auxiliary_matrix(system, cache, *, biased, lead_order=None):
    """Common auxiliary-space realization of the rational Dyson denominator."""
    lead_order = tuple(system.lead_names if lead_order is None else lead_order)
    n_leads = len(lead_order)
    start = 1 + n_leads
    matrix = np.zeros(
        (start + len(cache.zeta), start + len(cache.zeta)),
        dtype=np.complex128,
    )
    matrix[0, 0] = system.e_0 + cache.sigma_H + (
        system.DELTA if biased else 0.0
    )
    # Keep one auxiliary state per physical lead in both endpoint matrices.
    # Combining degenerate unbiased poles would change the state dimension and
    # prevent exact propagation across a switching surface.
    for index, lead in enumerate(lead_order, start=1):
        matrix[0, index] = 0.5 * system.Gamma0(lead) * system.W
        matrix[index, 0] = 1.0
        matrix[index, index] = (
            system.Delta(lead) if biased else 0.0
        ) - 1j * system.W
    if len(cache.zeta):
        matrix[0, start:] = cache.weights
        matrix[start:, 0] = 1.0
        matrix[start:, start:] = np.diag(cache.zeta)
    return matrix


def _auxiliary_rows_and_columns(matrix, times):
    eigenvalues, right = np.linalg.eig(matrix)
    inverse = np.linalg.inv(right)
    phase = np.exp(-1j * np.asarray(times)[:, None] * eigenvalues[None, :])
    rows = (phase * right[0, :][None, :]) @ inverse
    columns = (phase * inverse[:, 0][None, :]) @ right.T
    return rows, columns


def _auxiliary_retarded_rows(
    config, system, cache, time, *, dense, lead_order=None, progress=None
):
    """Exact rational-kernel propagation through all switching surfaces."""
    n_time = len(time)
    lag = np.arange(n_time) * float(time[1] - time[0])
    unbiased = _stationary_retarded(
        cache.xi_unbiased, cache.residues_unbiased, lag
    )
    biased = _stationary_retarded(cache.xi_biased, cache.residues_biased, lag)
    unbiased_matrix = _auxiliary_matrix(
        system, cache, biased=False, lead_order=lead_order
    )
    biased_matrix = _auxiliary_matrix(
        system, cache, biased=True, lead_order=lead_order
    )
    positive_time = np.maximum(time, 0.0)
    negative_elapsed = np.maximum(-time, 0.0)
    rows_u, _ = _auxiliary_rows_and_columns(unbiased_matrix, positive_time)
    rows_b, _ = _auxiliary_rows_and_columns(biased_matrix, positive_time)
    _, columns_u = _auxiliary_rows_and_columns(unbiased_matrix, negative_elapsed)
    _, columns_b = _auxiliary_rows_and_columns(biased_matrix, negative_elapsed)
    sums = np.zeros(n_time, dtype=np.complex128)
    counts = np.arange(n_time, 0, -1, dtype=float)
    total_norm = 0.0
    matrix = np.zeros((n_time, n_time), dtype=np.complex128) if dense else None
    tolerance = 10.0 * np.finfo(float).eps
    protocol = config.protocol.name
    duration = float(config.protocol.duration or 0.0)
    if protocol == "square":
        post_elapsed = np.maximum(time - duration, 0.0)
        pulse_remaining = np.maximum(duration - time, 0.0)
        rows_post, _ = _auxiliary_rows_and_columns(
            unbiased_matrix, post_elapsed
        )
        _, columns_pulse = _auxiliary_rows_and_columns(
            biased_matrix, pulse_remaining
        )
        # The full fixed-duration propagator is needed when an interval crosses
        # both square switching surfaces.
        eigenvalues, right = np.linalg.eig(biased_matrix)
        pulse_matrix = (
            right * np.exp(-1j * eigenvalues * duration)[None, :]
        ) @ np.linalg.inv(right)

    for row, observation in enumerate(time):
        prior = time[: row + 1]
        lag_indices = row - np.arange(row + 1)
        if observation <= tolerance:
            stationary = biased if protocol == "downward" else unbiased
            values = stationary[lag_indices]
        elif protocol in ("upward", "downward"):
            endpoint = biased if protocol == "upward" else unbiased
            values = endpoint[lag_indices].copy()
            cross = prior < 0.0
            if np.any(cross):
                if protocol == "upward":
                    values[cross] = -1j * (
                        rows_b[row] @ columns_u[np.flatnonzero(cross)].T
                    )
                else:
                    values[cross] = -1j * (
                        rows_u[row] @ columns_b[np.flatnonzero(cross)].T
                    )
        elif protocol == "square":
            if observation <= duration + tolerance:
                values = biased[lag_indices].copy()
                cross = prior < 0.0
                if np.any(cross):
                    values[cross] = -1j * (
                        rows_b[row] @ columns_u[np.flatnonzero(cross)].T
                    )
            else:
                values = unbiased[lag_indices].copy()
                in_pulse = (prior >= 0.0) & (prior <= duration + tolerance)
                before_pulse = prior < 0.0
                if np.any(in_pulse):
                    values[in_pulse] = -1j * (
                        rows_post[row]
                        @ columns_pulse[np.flatnonzero(in_pulse)].T
                    )
                if np.any(before_pulse):
                    post_pulse_row = rows_post[row] @ pulse_matrix
                    values[before_pulse] = -1j * (
                        post_pulse_row
                        @ columns_u[np.flatnonzero(before_pulse)].T
                    )
        else:
            values = unbiased[lag_indices]
        sums[lag_indices] += values
        total_norm += float(np.sum(np.abs(values) ** 2))
        if matrix is not None:
            matrix[row, : row + 1] = values
        if row + 1 == n_time or (row + 1) % 64 == 0:
            _progress(progress, "retarded_reconstruction", batch_completed=row + 1, batch_total=n_time, batch_unit="rows")

    projected = sums / counts
    projected_norm = float(np.sum(counts * np.abs(projected) ** 2))
    r_value = np.sqrt(max(total_norm - projected_norm, 0.0)) / max(
        np.sqrt(total_norm), np.finfo(float).tiny
    )
    q_by_row = np.zeros(n_time)
    if matrix is not None:
        projected_matrix = toeplitz_from_lags(projected, lesser=False)
        q_by_row = np.sqrt(np.sum(np.abs(matrix - projected_matrix) ** 2, axis=1))
    return projected, matrix, float(r_value), q_by_row


def solve_fixed_kernel_trajectory(
    config: CaseConfig,
    kernel: StationaryKernel,
    *,
    dense: bool | None = None,
    progress=None,
    system=None,
    pole_cache=None,
    previous_basis=None,
    time_grid: np.ndarray | None = None,
    energy_grid: np.ndarray | None = None,
    square_cache_store=None,
    emit_setup_progress: bool = True,
    amplitude_frequency: float | None = None,
) -> FixedKernelTrajectory:
    """Solve and project one complete pulse for an installed stationary kernel."""
    config.validate()
    dense = config.projection.dense_validation if dense is None else dense
    if emit_setup_progress:
        _progress(progress, "initialization")
    system = make_system(config) if system is None else system
    system.mpm_tol = config.numerics.mpm_search_tolerance
    if emit_setup_progress:
        _progress(progress, "mpm_fit")
    cache = (
        build_pole_cache(system, kernel, previous_basis=previous_basis, progress=progress)
        if pole_cache is None else pole_cache
    )
    _progress(
        progress,
        "endpoint_poles",
        message=(
            f"unbiased={len(cache.xi_unbiased)} biased={len(cache.xi_biased)} "
            f"self-energy={len(cache.zeta)}"
        ),
    )
    time = config.projection.time_grid() if time_grid is None else time_grid
    energy = config.numerics.energy_grid() if energy_grid is None else energy_grid
    a_values, c_zero, square_cache = _pulse_amplitudes(
        config, system, cache, time, energy, progress=progress,
        square_cache_store=square_cache_store,
    )
    # The interaction source uses C(t,0,E') in the generic installed-kernel
    # factorization.  Optional publication diagnostics evaluate both explicit
    # phonon-frequency branches on the same installed kernel.  These extra
    # amplitude stages have no progress tiles so live SCBA logs continue to
    # describe the source solve exactly once.
    c_phonon = None
    c_phonon_branches: dict[float, np.ndarray] = {}
    if amplitude_frequency is not None:
        # The production source solve below uses the generic installed
        # Sigma^< factorization at omega=0.  For diagnostics and independent
        # SCBA checks, retain both explicit manuscript phonon branches.
        for eta in (-1.0, 1.0):
            branch_frequency = eta * float(amplitude_frequency)
            _, branch_c, _ = _pulse_amplitudes(
                config,
                system,
                cache,
                time,
                energy,
                square_cache_store=square_cache_store,
                c_omega=branch_frequency,
            )
            c_phonon_branches[branch_frequency] = np.asarray(
                np.abs(branch_c) ** 2, dtype=float
            )
        # Keep the old single-array field as the +omega branch for readers
        # that have not yet migrated to the explicit mapping.
        c_phonon = c_phonon_branches[float(amplitude_frequency)]

    lesser_energy = _lesser_energy_grid(config, cache)
    if np.array_equal(lesser_energy, energy):
        lesser_a_values = a_values
        lesser_c_zero = c_zero
    else:
        _progress(
            progress,
            "amplitude_source_grid",
            batch_completed=0,
            batch_total=len(time),
            batch_unit="times on refined source grid",
        )
        def source_grid_progress(_phase, details):
            _progress(progress, "amplitude_source_grid", **details)

        lesser_a_values, lesser_c_zero, _ = _pulse_amplitudes(
            config,
            system,
            cache,
            time,
            lesser_energy,
            progress=source_grid_progress,
            square_cache_store=square_cache_store,
        )

    # B, D, and Psi are accumulated in the equation-equivalent
    # source-factorized contractions below.  Separate events preserve the
    # manuscript stage order without materializing their dense N_E^2 forms.
    _progress(progress, "history_b", message="lead histories; source-factorized contraction")
    _progress(progress, "history_d", message="interaction histories; source-factorized contraction")
    _progress(progress, "psi", message="lead and interaction lesser transform")

    reference = config.projection.reconstruction_lead
    chi_reference = _drive_integral(config, time, reference)
    _progress(progress, "retarded_reconstruction", batch_completed=0, batch_total=len(time), batch_unit="rows")
    gr_positive, dense_gr, r_gr, q_gr = _auxiliary_retarded_rows(
        config, system, cache, time, dense=dense, progress=progress
    )

    _progress(progress, "lesser_sources", batch_completed=0, batch_total=len(lesser_energy), batch_unit="energies")
    sources = _lesser_sources(
        config,
        system,
        kernel,
        time,
        lesser_energy,
        lesser_a_values,
        lesser_c_zero,
    )
    _progress(progress, "streaming_projection", batch_completed=0, batch_total=len(lesser_energy), batch_unit="energies")
    gl_positive = autocorrelation_projection(
        sources, energy_batch=config.projection.energy_batch, progress=progress
    )
    dense_gl = dense_lesser_from_sources(sources) if dense else None
    r_gl = float("nan")
    q_gl = np.zeros(len(time), dtype=float)
    diagnostics: dict[str, object] = {
        "dense_validation_performed": bool(dense),
        "pole_fit_terms": int(cache.fit_terms),
        "pole_fit_raw_converged": bool(cache.fit_converged),
        "pole_fit_converged": bool(
            cache.max_sigma_scaled_error <= 1.0
            and cache.max_Gfr_scaled_error <= 1.0
            and cache.max_Gbar_scaled_error <= 1.0
        ),
        "pole_causal": bool(cache.causal),
        "sigma_poles": int(cache.n_sigma_poles),
        "unbiased_green_poles": int(cache.n_unbiased_green_poles),
        "biased_green_poles": int(cache.n_biased_green_poles),
        "mpm_sigma_scaled_error": float(cache.max_sigma_scaled_error),
        "mpm_sigma_abs_error": float(cache.max_sigma_abs_error),
        "mpm_Gfr_scaled_error": float(cache.max_Gfr_scaled_error),
        "mpm_Gbiased_scaled_error": float(cache.max_Gbar_scaled_error),
        "mpm_Gfr_abs_error": float(cache.max_Gfr_abs_error),
        "mpm_Gbiased_abs_error": float(cache.max_Gbar_abs_error),
        "pole_basis_reused": bool(getattr(cache, "pole_basis_reused", False)),
        "pole_basis_origin_iteration": (
            -1 if getattr(cache, "pole_basis_origin_iteration", None) is None
            else int(cache.pole_basis_origin_iteration)
        ),
        "pole_basis_reason": str(getattr(cache, "pole_basis_reason", "")),
        "lesser_source_energy_points": int(len(lesser_energy)),
        "lesser_source_minimum_step": float(np.min(np.diff(lesser_energy))),
        "amplitude_c_frequency": float(
            amplitude_frequency if c_phonon is not None else 0.0
        ),
        "amplitude_c_branch_frequencies": [
            float(value) for value in sorted(c_phonon_branches)
        ],
        "amplitude_c_branch_weights": [
            float(
                float(kernel.phonon_occupation) + (1.0 if value < 0.0 else 0.0)
            )
            for value in sorted(c_phonon_branches)
        ] if c_phonon_branches else [],
    }
    if square_cache is not None:
        diagnostics.update({
            "square_cache_key": str(getattr(square_cache, "cache_key", "")),
            "square_cache_residue_method": str(getattr(square_cache, "residue_method", "")),
            "square_cache_residue_max_scaled_error": float(square_cache.residue_max_scaled_error),
            "square_cache_turnoff_A_scaled_error": float(square_cache.turnoff_A_scaled_error),
            "square_cache_turnoff_C_scaled_error": float(square_cache.turnoff_C_scaled_error),
        })

    if dense_gl is not None:
        _progress(progress, "dense_validation")
        dense_projection = direct_toeplitz_projection(dense_gl, lesser=True)
        stream_projection = direct_toeplitz_projection(
            np.asarray(
                [[
                    gl_positive[i - j]
                    if i >= j
                    else -np.conjugate(gl_positive[j - i])
                    for j in range(len(time))
                ] for i in range(len(time))],
                dtype=np.complex128,
            ),
            lesser=True,
        )
        projection_scale = max(float(np.linalg.norm(dense_projection)), 1e-30)
        diagnostics["lesser_dense_streaming_error"] = float(
            np.linalg.norm(dense_projection - stream_projection) / projection_scale
        )
        q_matrix = dense_gl - dense_projection
        total = float(np.linalg.norm(dense_gl))
        r_gl = float(np.linalg.norm(q_matrix) / max(total, 1e-30))
        q_gl = np.sqrt(np.sum(np.abs(q_matrix) ** 2, axis=1))
        idempotency, orthogonality = projector_errors(dense_gl, lesser=True)
        diagnostics["projector_lesser_idempotency_error"] = idempotency
        diagnostics["projector_lesser_orthogonality_error"] = orthogonality
        diagnostics["lesser_hermiticity_error"] = float(
            np.linalg.norm(dense_gl + np.conjugate(dense_gl.T))
            / max(np.linalg.norm(dense_gl), 1e-30)
        )
    if dense_gr is not None:
        idempotency, orthogonality = projector_errors(dense_gr, lesser=False)
        diagnostics["projector_retarded_idempotency_error"] = idempotency
        diagnostics["projector_retarded_orthogonality_error"] = orthogonality
        diagnostics["retarded_causality_error"] = float(
            np.linalg.norm(np.triu(dense_gr, k=1))
            / max(np.linalg.norm(dense_gr), 1e-30)
        )
        other_lead = "R" if reference == "L" else "L"
        _, reference_grid, _, _ = reconstruct_retarded_rows(
            a_values[reference], energy, time, chi_reference, dense=True
        )
        _, other_grid, _, _ = reconstruct_retarded_rows(
            a_values[other_lead],
            energy,
            time,
            _drive_integral(config, time, other_lead),
            dense=True,
        )
        _, other_dense, _, _ = _auxiliary_retarded_rows(
            config,
            system,
            cache,
            time,
            dense=True,
            lead_order=tuple(reversed(system.lead_names)),
        )
        diagnostics["left_right_retarded_reconstruction_error"] = float(
            np.linalg.norm(dense_gr - other_dense)
            / max(np.linalg.norm(dense_gr), 1e-30)
        )
        diagnostics["finite_grid_left_right_retarded_error"] = float(
            np.linalg.norm(reference_grid - other_grid)
            / max(np.linalg.norm(reference_grid), 1e-30)
        )
        diagnostics["finite_grid_retarded_tail_error"] = float(
            np.linalg.norm(reference_grid - dense_gr)
            / max(np.linalg.norm(dense_gr), 1e-30)
        )

    projection = make_projection_result(
        time=time,
        energy=energy,
        retarded_positive=gr_positive,
        lesser_positive=gl_positive,
        r_retarded=r_gr,
        r_lesser=r_gl,
        q_retarded_time=q_gr,
        q_lesser_time=q_gl,
        diagnostics=diagnostics,
    )
    # Independent occupation reconstruction from the projected energy
    # function.  The primary occupation remains the lag-zero source value;
    # this finite-energy integral is a discretization diagnostic and must not
    # silently replace it.
    occupation_from_energy = -1j * np.trapezoid(
        projection.green_lesser_energy, projection.energy
    ) / (2.0 * np.pi)
    diagnostics["projected_occupation_energy"] = float(
        np.real(occupation_from_energy)
    )
    diagnostics["projected_occupation_energy_imaginary"] = float(
        abs(np.imag(occupation_from_energy))
    )
    diagnostics["projected_occupation_independent_error"] = float(
        abs(float(np.real(occupation_from_energy)) - projection.occupation)
    )
    occupation = -1j * np.sum(
        np.vstack([
            np.sum(amplitudes * weights[None, :] * np.conjugate(amplitudes), axis=1)
            for amplitudes, weights in sources
        ]),
        axis=0,
    )
    if np.max(np.abs(occupation.imag)) > 1e-7 * max(1.0, np.max(np.abs(occupation.real))):
        raise RuntimeError("Fixed-kernel occupation is not real within tolerance.")
    if dense_gl is not None:
        diagnostics["two_time_occupation_error"] = float(
            np.max(np.abs((-1j * np.diag(dense_gl)).real - occupation.real))
        )
    return FixedKernelTrajectory(
        time=time,
        energy=energy,
        projected=projection,
        observable_time=time,
        occupation=np.asarray(occupation.real),
        amplitude_a_squared={
            lead: np.asarray(np.abs(values) ** 2, dtype=float)
            for lead, values in a_values.items()
        },
        amplitude_c_squared=np.asarray(
            np.abs(c_zero) ** 2 if c_phonon is None else c_phonon,
            dtype=float,
        ),
        amplitude_c_squared_by_frequency=c_phonon_branches,
        lesser_source_energy=(lesser_energy if dense else None),
        lesser_source_factors=(tuple(factor for factor, _ in sources) if dense else ()),
        lesser_source_weights=(tuple(weight for _, weight in sources) if dense else ()),
        diagnostics=diagnostics,
        dense_green_retarded=dense_gr,
        dense_green_lesser=dense_gl,
    )


def attach_observables(
    config: CaseConfig,
    kernel: StationaryKernel,
    trajectory: FixedKernelTrajectory,
    *,
    progress=None,
    previous_basis=None,
) -> FixedKernelTrajectory:
    """Evaluate physical currents with the converged installed kernel."""
    if config.protocol.name == "zero":
        zeros = np.zeros(len(trajectory.time), dtype=float)
        diagnostics = dict(trajectory.diagnostics)
        diagnostics.update(
            {
                "current_method": "zero_pulse_exact",
                "current_representation_effective": "exact_zero_pulse",
                "current_scale": 1e-12,
                "max_continuity_residual": 0.0,
                "collision_identity_max_mismatch": 0.0,
                "collision_identity_scaled_mismatch": 0.0,
            }
        )
        return replace(
            trajectory,
            observable_time=trajectory.time,
            currents={"L": zeros.copy(), "R": zeros.copy()},
            continuity_residual=zeros.copy(),
            diagnostics=diagnostics,
        )
    if config.numerics.current_method == "finite_window":
        return finite_window_observables(
            config,
            kernel,
            trajectory,
            conserve_noninteracting=True,
            previous_basis=previous_basis,
            progress=progress,
        )
    return direct_oracle_observables(config, kernel, trajectory)


def _record_collision_mismatch(
    config: CaseConfig,
    diagnostics: dict,
    time: np.ndarray,
    mismatch: np.ndarray,
    current_scale: float,
) -> None:
    """Record full and switching-surface-resolved collision mismatches."""
    absolute = np.abs(mismatch)
    resolved = (
        np.abs(time) > 2.1 * config.projection.time_step
    )
    if config.protocol.name == "square":
        resolved &= (
            np.abs(time - float(config.protocol.duration))
            > 2.1 * config.projection.time_step
        )
    # Very small validation grids can lie entirely inside an exclusion
    # stencil.  They still need a meaningful (conservative) diagnostic.
    if not np.any(resolved):
        resolved[:] = True
    resolved_max = float(np.max(absolute[resolved]))
    full_max = float(np.max(absolute))
    diagnostics["collision_identity_full_max_mismatch"] = full_max
    diagnostics["collision_identity_full_scaled_mismatch"] = (
        full_max / current_scale
    )
    diagnostics["collision_identity_max_mismatch"] = resolved_max
    diagnostics["collision_identity_scaled_mismatch"] = (
        resolved_max / current_scale
    )
    resolved_indices = np.flatnonzero(resolved)
    resolved_peak = resolved_indices[
        int(np.argmax(absolute[resolved]))
    ]
    diagnostics["collision_identity_peak_time"] = float(time[resolved_peak])
    diagnostics["collision_identity_excluded_switch_points"] = int(
        np.count_nonzero(~resolved)
    )


def _finite_window_psi(
    gl: np.ndarray,
    time: np.ndarray,
    energy: np.ndarray,
    chi: np.ndarray,
    positive_indices: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Return the finite-history mixed lesser function by batched CZTs.

    This evaluates

        Psi(E,t) = integral_{t_min}^t dt' exp(i E (t-t'))
                   exp(i [chi(t)-chi(t')]) G<(t,t')

    on the uniform reconstruction grid.  Zeroing the upper triangle before
    each transform preserves the causal integration limit, while endpoint
    trapezoid weights avoid the equal-time ambiguity of a retarded embedding
    convolution.
    """
    from scipy.signal import czt

    dt = float(time[1] - time[0])
    de = float(energy[1] - energy[0])
    n_time = len(time)
    n_energy = len(energy)
    column_phase = np.exp(-1j * (energy[0] * time + chi))
    output = np.empty((len(positive_indices), n_energy), dtype=np.complex128)
    column_indices = np.arange(n_time)[None, :]
    energy_phase_step = np.exp(-1j * de * dt)
    for start in range(0, len(positive_indices), batch_size):
        stop = min(start + batch_size, len(positive_indices))
        rows = positive_indices[start:stop]
        values = np.array(gl[rows, :], copy=True)
        values *= column_phase[None, :]
        values[column_indices > rows[:, None]] = 0.0
        quadrature = np.ones_like(values.real)
        quadrature[column_indices > rows[:, None]] = 0.0
        quadrature[:, 0] *= 0.5
        quadrature[np.arange(len(rows)), rows] *= 0.5
        quadrature[rows == 0] = 0.0
        values *= quadrature
        transformed = czt(
            values,
            m=n_energy,
            w=energy_phase_step,
            a=1.0,
            axis=-1,
        )
        transformed *= np.exp(
            -1j * (energy - energy[0]) * time[0]
        )[None, :]
        observation_phase = np.exp(
            1j * (time[rows, None] * energy[None, :] + chi[rows, None])
        )
        output[start:stop] = dt * observation_phase * transformed
    return output


def _cauchy_linear_uniform(
    values: np.ndarray,
    grid: np.ndarray,
    pole_shift: float,
) -> np.ndarray:
    """FFT application of the exact linear-interpolant Cauchy operator.

    For every grid point ``p_i = grid[i] + pole_shift`` this returns

        integral values(y) / (p_i - y + i0) dy,

    with precisely the segment formula used by the literal direct oracle.
    Uniform spacing makes the left- and right-node contributions Toeplitz;
    keeping them separate handles the two finite-grid endpoints exactly.
    """
    from scipy.signal import fftconvolve

    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=np.complex128)
    if values.shape[-1] != len(grid):
        raise ValueError("Cauchy values must use the final axis as the energy grid.")
    de = float(grid[1] - grid[0])
    grid_roundoff = (
        32.0
        * np.finfo(float).eps
        * max(1.0, abs(grid[0]), abs(grid[-1]))
    )
    if not np.allclose(
        np.diff(grid), de, rtol=1e-10, atol=grid_roundoff
    ):
        raise ValueError("Fast linear Cauchy transforms require a uniform grid.")
    n_energy = len(grid)
    differences = np.arange(-(n_energy - 1), n_energy, dtype=float)
    epsilon = np.finfo(float).eps

    def segment_weights(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = offset.astype(np.complex128) + 1j * epsilon
        logarithm = np.log(x) - np.log(x - de)
        left = (1.0 - x / de) * logarithm + 1.0
        right = (x / de) * logarithm - 1.0
        return left, right

    left_kernel, _ = segment_weights(differences * de + pole_shift)
    _, right_kernel = segment_weights((differences + 1.0) * de + pole_shift)
    left_values = np.array(values, copy=True)
    right_values = np.array(values, copy=True)
    left_values[..., -1] = 0.0
    right_values[..., 0] = 0.0
    left_full = fftconvolve(
        left_values,
        left_kernel.reshape((1,) * (values.ndim - 1) + (-1,)),
        mode="full",
        axes=-1,
    )
    right_full = fftconvolve(
        right_values,
        right_kernel.reshape((1,) * (values.ndim - 1) + (-1,)),
        mode="full",
        axes=-1,
    )
    selection = slice(n_energy - 1, 2 * n_energy - 1)
    result = left_full[..., selection] + right_full[..., selection]
    # When a shifted pole lands on a finite-grid endpoint, floating-point
    # construction of ``grid + shift`` determines the oracle's one-sided log
    # branch.  Correct those at-most-two rows literally; the bulk remains FFT.
    poles = grid + pole_shift
    endpoint_scale = max(1.0, abs(grid[0]), abs(grid[-1]), abs(pole_shift))
    endpoint_rows = np.flatnonzero(
        np.isclose(
            poles,
            grid[0],
            rtol=0.0,
            atol=8.0 * np.finfo(float).eps * endpoint_scale,
        )
        | np.isclose(
            poles,
            grid[-1],
            rtol=0.0,
            atol=8.0 * np.finfo(float).eps * endpoint_scale,
        )
    )
    if len(endpoint_rows):
        flattened = values.reshape(-1, n_energy)
        corrected = result.reshape(-1, n_energy)
        for batch_row, row_values in enumerate(flattened):
            corrected[batch_row, endpoint_rows] = integrate_cauchy_linear(
                poles[endpoint_rows], grid, row_values
            )
    return result


def _difference_convolution(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply ``sum_j values[j] kernel[i-j]`` on one uniform grid."""
    from scipy.signal import fftconvolve

    n_energy = values.shape[-1]
    full = fftconvolve(values, kernel, mode="full", axes=-1)
    return full[..., n_energy - 1 : 2 * n_energy - 1]


def _filon_linear(
    values: np.ndarray, grid: np.ndarray, frequency: float
) -> np.ndarray:
    """Integrate a piecewise-linear amplitude times ``exp(i*w*x)``.

    Unlike applying a trapezoidal rule to the oscillatory product, this
    integrates the exponential analytically on every energy panel.  The
    amplitude may be batched; its final axis must correspond to ``grid``.
    """
    values = np.asarray(values)
    grid = np.asarray(grid, dtype=float)
    if values.shape[-1] != len(grid):
        raise ValueError("Filon values must use the final axis as the grid.")
    if abs(frequency) * float(np.max(np.diff(grid))) <= 0.5:
        return np.trapezoid(
            values * np.exp(1j * frequency * grid), grid, axis=-1
        )
    widths = np.diff(grid)
    if len(widths) == 0:
        raise ValueError("Filon integration requires at least two grid points.")
    theta = frequency * widths
    i_theta = 1j * theta
    moment_0 = np.empty_like(i_theta, dtype=np.complex128)
    moment_1 = np.empty_like(i_theta, dtype=np.complex128)
    small = np.abs(theta) < 1e-4
    if np.any(~small):
        z = theta[~small]
        exponential = np.exp(1j * z)
        moment_0[~small] = np.expm1(1j * z) / (1j * z)
        moment_1[~small] = (
            exponential * (1.0 - 1j * z) - 1.0
        ) / z**2
    if np.any(small):
        z = i_theta[small]
        # Taylor moments: integral_0^1 u^m exp(z u) du.
        moment_0[small] = (
            1.0
            + z / 2.0
            + z**2 / 6.0
            + z**3 / 24.0
            + z**4 / 120.0
            + z**5 / 720.0
        )
        moment_1[small] = (
            0.5
            + z / 3.0
            + z**2 / 8.0
            + z**3 / 30.0
            + z**4 / 144.0
            + z**5 / 840.0
        )
    phase = np.exp(1j * frequency * grid[:-1])
    left = widths * phase * (moment_0 - moment_1)
    right = widths * phase * moment_1
    return np.sum(
        values[..., :-1] * left + values[..., 1:] * right, axis=-1
    )


def _panel_kernel_convolution(
    values: np.ndarray,
    grid: np.ndarray,
    kernel_function,
    *,
    quadrature_order: int = 12,
) -> np.ndarray:
    """Integrate a translation kernel against linear energy panels by FFT.

    Gauss--Legendre nodes only resolve the smooth within-panel amplitude of
    the supplied kernel.  In particular, callers pass ``expc`` rather than a
    sampled exponential, so no observation-time Nyquist condition is
    introduced by the reconstruction energy spacing.
    """
    from numpy.polynomial.legendre import leggauss
    from scipy.signal import fftconvolve

    values = np.asarray(values)
    grid = np.asarray(grid, dtype=float)
    if values.shape[-1] != len(grid):
        raise ValueError("Kernel values must use the final axis as the grid.")
    de = float(grid[1] - grid[0])
    if not np.allclose(np.diff(grid), de, rtol=1e-10, atol=1e-13):
        raise ValueError("Panel-kernel convolution requires a uniform grid.")
    nodes, weights = leggauss(quadrature_order)
    u = 0.5 * (nodes + 1.0)
    panel_weights = 0.5 * de * weights
    differences = np.arange(-(len(grid) - 1), len(grid), dtype=float)

    left_kernel = np.sum(
        panel_weights[None, :]
        * (1.0 - u)[None, :]
        * kernel_function(differences[:, None] * de - u[None, :] * de),
        axis=1,
    )
    # For the right endpoint k=j+1, x_i-y_j=(i-k+1) de.
    right_kernel = np.sum(
        panel_weights[None, :]
        * u[None, :]
        * kernel_function(
            (differences[:, None] + 1.0) * de - u[None, :] * de
        ),
        axis=1,
    )
    left_values = np.array(values, copy=True)
    right_values = np.array(values, copy=True)
    left_values[..., -1] = 0.0
    right_values[..., 0] = 0.0
    reshape = (1,) * (values.ndim - 1) + (-1,)
    left_full = fftconvolve(
        left_values, left_kernel.reshape(reshape), mode="full", axes=-1
    )
    right_full = fftconvolve(
        right_values, right_kernel.reshape(reshape), mode="full", axes=-1
    )
    selection = slice(len(grid) - 1, 2 * len(grid) - 1)
    return left_full[..., selection] + right_full[..., selection]


def _apply_panel_kernels(
    values: np.ndarray, left_kernel: np.ndarray, right_kernel: np.ndarray
) -> np.ndarray:
    """Apply cached left/right linear-panel convolution weights."""
    from scipy.signal import fftconvolve

    values = np.asarray(values)
    n_energy = values.shape[-1]
    left_values = np.array(values, copy=True)
    right_values = np.array(values, copy=True)
    left_values[..., -1] = 0.0
    right_values[..., 0] = 0.0
    reshape = (1,) * (values.ndim - 1) + (-1,)
    left_full = fftconvolve(
        left_values, left_kernel.reshape(reshape), mode="full", axes=-1
    )
    right_full = fftconvolve(
        right_values, right_kernel.reshape(reshape), mode="full", axes=-1
    )
    selection = slice(n_energy - 1, 2 * n_energy - 1)
    return left_full[..., selection] + right_full[..., selection]


def _expc_antiderivatives(
    difference: np.ndarray, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """First two antiderivatives needed for exact linear ``expc`` panels."""
    from scipy.special import sici

    difference = np.asarray(difference, dtype=float)
    if time == 0.0:
        zeros = np.zeros_like(difference, dtype=np.complex128)
        return zeros, zeros.copy()
    argument = time * difference
    entire_ein = np.empty_like(argument, dtype=np.complex128)
    small = np.abs(argument) < 0.25
    if np.any(~small):
        real_argument = argument[~small]
        sine_integral, cosine_integral = sici(np.abs(real_argument))
        entire_ein[~small] = (
            cosine_integral
            - np.log(np.abs(real_argument))
            - 0.5772156649015328606
            + 1j * np.sign(real_argument) * sine_integral
        )
    if np.any(small):
        z = 1j * argument[small]
        term = np.array(z, copy=True)
        series = np.array(term, copy=True)
        for order in range(2, 18):
            term *= z / order
            series += term / order
        entire_ein[small] = series
    primitive_zero = entire_ein / 1j
    primitive_one = -np.exp(1j * argument) / time + 1j * difference
    return primitive_zero, primitive_one


@lru_cache(maxsize=32)
def _expc_panel_kernels(
    n_energy: int, de: float, pole_shift: float, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Filon weights for a uniform ``expc`` convolution."""
    differences = np.arange(-(n_energy - 1), n_energy, dtype=float)

    def segment_weights(q_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_left = q_right - de
        f0_right, f1_right = _expc_antiderivatives(q_right, time)
        f0_left, f1_left = _expc_antiderivatives(q_left, time)
        delta_f0 = f0_right - f0_left
        delta_f1 = f1_right - f1_left
        left = (1.0 - q_right / de) * delta_f0 + delta_f1 / de
        right = q_right * delta_f0 / de - delta_f1 / de
        return left, right

    left_kernel, _ = segment_weights(differences * de + pole_shift)
    _, right_kernel = segment_weights(
        (differences + 1.0) * de + pole_shift
    )
    left_kernel.setflags(write=False)
    right_kernel.setflags(write=False)
    return left_kernel, right_kernel


def _expc_linear_uniform(
    values: np.ndarray,
    grid: np.ndarray,
    pole_shift: float,
    time: float,
) -> np.ndarray:
    """Panel-integrated ``values(E') expc(E-E'+shift,t)``."""
    de = float(grid[1] - grid[0])
    if abs(time) * abs(de) <= 0.5:
        differences = np.arange(-(len(grid) - 1), len(grid), dtype=float)
        return _difference_convolution(
            values * trapezoid_weights(grid),
            expc(differences * de + pole_shift, time),
        )
    left_kernel, right_kernel = _expc_panel_kernels(
        len(grid), de, float(pole_shift), float(time)
    )
    return _apply_panel_kernels(values, left_kernel, right_kernel)


def _phase_cauchy_linear_uniform(
    values: np.ndarray,
    grid: np.ndarray,
    pole_shift: float,
    time: float,
) -> np.ndarray:
    """Return ``exp(i*(E+shift)t) C[exp(-iE't) values]``.

    The identity used here separates the Cauchy distribution from a regular
    ``expc`` panel integral.  Consequently the fast phase is never sampled on
    the reconstruction grid.
    """
    if abs(time) * abs(float(grid[1] - grid[0])) <= 0.5:
        return np.exp(1j * (grid + pole_shift) * time) * (
            _cauchy_linear_uniform(
                values * np.exp(-1j * grid * time), grid, pole_shift
            )
        )
    return _cauchy_linear_uniform(values, grid, pole_shift) + 1j * (
        _expc_linear_uniform(values, grid, pole_shift, time)
    )


def _psi_leads_up_fast(
    system,
    cache,
    energy: np.ndarray,
    time: float,
    a_values: dict[str, np.ndarray],
    alpha: str,
) -> np.ndarray:
    """Equation-equivalent O(N_E log N_E + N_p N_E) upward lead Psi."""
    de = float(energy[1] - energy[0])
    energy_weights = trapezoid_weights(energy) / (2.0 * np.pi)
    differences = np.arange(-(len(energy) - 1), len(energy), dtype=float)
    outer = energy
    da = system.Delta(alpha)
    result = np.zeros(len(energy), dtype=np.complex128)
    poles = cache.xi_biased
    residues = cache.residues_biased

    for beta in system.lead_names:
        db = system.Delta(beta)
        occupied_source = (
            fermi_dirac(energy, system.beta_fc(beta), system.mu_fc(beta))
            * np.real(linewidth(system, beta, energy))
            * a_values[beta]
        )
        biased_boundary = Gbiased_R_mpm(system, cache, energy + db)
        base_values = occupied_source * np.conjugate(biased_boundary)
        result += (
            1j
            * _expc_linear_uniform(
                base_values, energy, da - db, time
            )
            / (2.0 * np.pi)
        )

        if len(poles):
            mfac = upward_lead_vertex(
                system,
                cache,
                poles[:, None],
                energy[None, :],
                beta,
            )
            pole_inner = _filon_linear(
                occupied_source[None, :]
                * np.conjugate(mfac / (poles[:, None] - energy[None, :])),
                energy,
                -time,
            ) / (2.0 * np.pi)
            pole_outer = (
                1j
                * np.exp(1j * (outer[:, None] + da - db) * time)
                * np.conjugate(
                    expc(outer[:, None] + da - poles[None, :], time)
                    * residues[None, :]
                )
            )
            result += pole_outer @ pole_inner

        singular_values = occupied_source * np.conjugate(
            Gfr_R_mpm(system, cache, energy)
        )
        result -= (
            np.exp(1j * (da - db) * time)
            * _phase_cauchy_linear_uniform(
                singular_values, energy, 0.0, time
            )
            / (2.0 * np.pi)
        )
    return result


def _psi_leads_down_fast(
    system,
    cache,
    energy: np.ndarray,
    time: float,
    a_values: dict[str, np.ndarray],
    alpha: str,
) -> np.ndarray:
    """Equation-equivalent O(N_E log N_E + N_p N_E) downward lead Psi."""
    de = float(energy[1] - energy[0])
    energy_weights = trapezoid_weights(energy) / (2.0 * np.pi)
    differences = np.arange(-(len(energy) - 1), len(energy), dtype=float)
    outer = energy
    da = system.Delta(alpha)
    result = np.zeros(len(energy), dtype=np.complex128)
    poles = cache.xi_unbiased
    residues = cache.residues_unbiased

    for beta in system.lead_names:
        db = system.Delta(beta)
        occupied_source = (
            fermi_dirac(energy, system.beta_fc(beta), system.mu_fc(beta))
            * np.real(linewidth(system, beta, energy))
            * a_values[beta]
        )
        base_values = occupied_source * np.conjugate(
            Gfr_R_mpm(system, cache, energy)
        )
        result += (
            1j
            * _expc_linear_uniform(base_values, energy, 0.0, time)
            / (2.0 * np.pi)
        )

        if len(poles):
            mfac = downward_lead_vertex(
                system,
                cache,
                poles[:, None],
                energy[None, :],
                beta,
            )
            pole_inner = _filon_linear(
                occupied_source[None, :]
                * np.conjugate(
                    mfac / (poles[:, None] - energy[None, :] - db)
                ),
                energy,
                -time,
            ) / (2.0 * np.pi)
            pole_outer = (
                -1j
                * np.exp(1j * outer[:, None] * time)
                * np.conjugate(
                    expc(outer[:, None] - poles[None, :], time)
                    * residues[None, :]
                )
            )
            result += pole_outer @ pole_inner

        singular_values = occupied_source * np.conjugate(
            Gbiased_R_mpm(system, cache, energy + db)
        )
        result -= (
            np.exp(-1j * (da - db) * time)
            * _phase_cauchy_linear_uniform(
                singular_values, energy, da - db, time
            )
            / (2.0 * np.pi)
        )
    return result


def _psi_leads_square_fast(
    system,
    cache,
    square_cache,
    energy: np.ndarray,
    time: float,
    a_values: dict[str, np.ndarray],
    alpha: str,
) -> np.ndarray:
    """Equation-equivalent low-rank square-pulse lead Psi."""
    duration = float(square_cache.duration)
    if time <= duration:
        return _psi_leads_up_fast(
            system, cache, energy, time, a_values, alpha
        )

    de = float(energy[1] - energy[0])
    energy_weights = trapezoid_weights(energy) / (2.0 * np.pi)
    differences = np.arange(-(len(energy) - 1), len(energy), dtype=float)
    difference_energy = differences * de
    outer = energy
    da = system.Delta(alpha)
    elapsed = time - duration
    result = np.zeros(len(energy), dtype=np.complex128)

    for beta in system.lead_names:
        db = system.Delta(beta)
        phase = np.exp(1j * (da - db) * duration)
        occupied_source = (
            fermi_dirac(energy, system.beta_fc(beta), system.mu_fc(beta))
            * np.real(linewidth(system, beta, energy))
            * a_values[beta]
        )

        pulse_base_values = occupied_source * np.conjugate(
            Gbiased_R_mpm(system, cache, energy + db)
        )
        result += _panel_kernel_convolution(
            pulse_base_values,
            energy,
            lambda difference: (
                1j
                * np.exp(1j * difference * time)
                * phase
                * np.conjugate(
                    expc(difference + da - db, duration)
                )
            ),
        ) / (2.0 * np.pi)

        biased_poles = cache.xi_biased
        if len(biased_poles):
            mfac = upward_lead_vertex(
                system,
                cache,
                biased_poles[:, None],
                energy[None, :],
                beta,
            )
            pulse_inner = _filon_linear(
                occupied_source[None, :]
                * np.conjugate(
                    mfac / (biased_poles[:, None] - energy[None, :])
                ),
                energy,
                -time,
            ) / (2.0 * np.pi)
            pulse_outer = (
                1j
                * np.exp(1j * outer[:, None] * time)
                * phase
                * np.conjugate(
                    expc(
                        outer[:, None] + da - biased_poles[None, :],
                        duration,
                    )
                    * cache.residues_biased[None, :]
                )
            )
            result += pulse_outer @ pulse_inner

        post_base_values = occupied_source * np.conjugate(
            Gfr_R_mpm(
                system, cache, energy.astype(np.complex128) + 1e-14j
            )
        )
        result += (
            1j
            * _expc_linear_uniform(
                post_base_values, energy, 0.0, elapsed
            )
            / (2.0 * np.pi)
        )

        unbiased_poles = cache.xi_unbiased
        if len(unbiased_poles):
            s_values = np.asarray(square_cache.S_alpha[beta])
            post_inner = _filon_linear(
                occupied_source[None, :] * np.conjugate(s_values),
                energy,
                -time,
            ) / (2.0 * np.pi)
            post_outer = (
                -1j
                * np.exp(1j * outer[:, None] * time)
                * np.exp(-1j * db * duration)
                * np.conjugate(
                    cache.residues_unbiased[None, :]
                    * np.exp(
                        1j
                        * outer[:, None]
                        * duration
                    )
                    * expc(
                        outer[:, None] - unbiased_poles[None, :],
                        elapsed,
                    )
                )
            )
            result += post_outer @ post_inner

        singular_values = occupied_source * np.conjugate(
            Gfr_R_mpm(system, cache, energy)
        )
        result -= (
            phase
            * _phase_cauchy_linear_uniform(
                singular_values, energy, 0.0, time
            )
            / (2.0 * np.pi)
        )
    return result


def _psi_interaction_up_fast(
    system,
    cache,
    energy: np.ndarray,
    time: float,
    c_zero: np.ndarray,
    sigma_lesser: np.ndarray,
    alpha: str,
) -> np.ndarray:
    """Generic installed-lesser upward interaction contribution to Psi."""
    de = float(energy[1] - energy[0])
    weights = trapezoid_weights(energy) / (2.0 * np.pi)
    differences = np.arange(-(len(energy) - 1), len(energy), dtype=float)
    outer = energy
    da = system.Delta(alpha)
    common = c_zero * sigma_lesser
    base_values = common * np.conjugate(
        Gbiased_R_mpm(system, cache, energy)
    )
    result = _expc_linear_uniform(
        base_values,
        energy,
        da,
        time,
    ) / (2.0 * np.pi)
    poles = cache.xi_biased
    if len(poles):
        nfac = upward_interaction_vertex(
            system,
            cache,
            poles[:, None],
            energy[None, :],
        )
        pole_inner = _filon_linear(
            common[None, :]
            * np.conjugate(nfac / (poles[:, None] - energy[None, :])),
            energy,
            -time,
        ) / (2.0 * np.pi)
        pole_outer = (
            np.exp(1j * (outer[:, None] + da) * time)
            * np.conjugate(
                expc(outer[:, None] + da - poles[None, :], time)
                * cache.residues_biased[None, :]
            )
        )
        result += pole_outer @ pole_inner
    singular_values = common * np.conjugate(
        Gfr_R_mpm(system, cache, energy)
    )
    result += (
        1j
        * np.exp(1j * da * time)
        * _phase_cauchy_linear_uniform(
            singular_values, energy, 0.0, time
        )
        / (2.0 * np.pi)
    )
    return result


def _psi_interaction_down_fast(
    system,
    cache,
    energy: np.ndarray,
    time: float,
    c_zero: np.ndarray,
    sigma_lesser: np.ndarray,
    alpha: str,
) -> np.ndarray:
    """Generic installed-lesser downward interaction contribution to Psi."""
    de = float(energy[1] - energy[0])
    weights = trapezoid_weights(energy) / (2.0 * np.pi)
    differences = np.arange(-(len(energy) - 1), len(energy), dtype=float)
    outer = energy
    common = c_zero * sigma_lesser
    result = _expc_linear_uniform(
        common * np.conjugate(Gfr_R_mpm(system, cache, energy)),
        energy,
        0.0,
        time,
    ) / (2.0 * np.pi)
    poles = cache.xi_unbiased
    if len(poles):
        sc = downward_interaction_vertex(
            system,
            poles[:, None],
            energy[None, :],
        )
        pole_inner = _filon_linear(
            common[None, :]
            * np.conjugate(
                sc
                * Gbiased_R_mpm(system, cache, energy)[None, :]
                / (poles[:, None] - energy[None, :])
            ),
            energy,
            -time,
        ) / (2.0 * np.pi)
        pole_outer = (
            -np.exp(1j * outer[:, None] * time)
            * np.conjugate(
                expc(outer[:, None] - poles[None, :], time)
                * cache.residues_unbiased[None, :]
            )
        )
        result += pole_outer @ pole_inner
    singular_values = common * np.conjugate(
        Gbiased_R_mpm(system, cache, energy)
    )
    result += (
        1j
        * _phase_cauchy_linear_uniform(
            singular_values, energy, 0.0, time
        )
        / (2.0 * np.pi)
    )
    return result


def _psi_interaction_square_fast(
    system,
    cache,
    square_cache,
    energy: np.ndarray,
    time: float,
    c_zero: np.ndarray,
    sigma_lesser: np.ndarray,
    alpha: str,
) -> np.ndarray:
    """Generic installed-lesser square-pulse interaction contribution."""
    duration = float(square_cache.duration)
    if time <= duration:
        return _psi_interaction_up_fast(
            system, cache, energy, time, c_zero, sigma_lesser, alpha
        )
    de = float(energy[1] - energy[0])
    weights = trapezoid_weights(energy) / (2.0 * np.pi)
    differences = np.arange(-(len(energy) - 1), len(energy), dtype=float)
    difference_energy = differences * de
    outer = energy
    da = system.Delta(alpha)
    elapsed = time - duration
    phase = np.exp(1j * da * duration)
    common = c_zero * sigma_lesser

    result = _panel_kernel_convolution(
        common * np.conjugate(Gbiased_R_mpm(system, cache, energy)),
        energy,
        lambda difference: (
            np.exp(1j * difference * time)
            * phase
            * np.conjugate(expc(difference + da, duration))
        ),
    ) / (2.0 * np.pi)
    biased_poles = cache.xi_biased
    if len(biased_poles):
        nfac = upward_interaction_vertex(
            system,
            cache,
            biased_poles[:, None],
            energy[None, :],
        )
        pulse_inner = _filon_linear(
            common[None, :]
            * np.conjugate(
                nfac / (biased_poles[:, None] - energy[None, :])
            ),
            energy,
            -time,
        ) / (2.0 * np.pi)
        pulse_outer = (
            np.exp(1j * outer[:, None] * time)
            * phase
            * np.conjugate(
                expc(
                    outer[:, None] + da - biased_poles[None, :],
                    duration,
                )
                * cache.residues_biased[None, :]
            )
        )
        result += pulse_outer @ pulse_inner

    result += _expc_linear_uniform(
        common
        * np.conjugate(
            Gfr_R_mpm(
                system, cache, energy.astype(np.complex128) + 1e-14j
            )
        ),
        energy,
        0.0,
        elapsed,
    ) / (2.0 * np.pi)
    unbiased_poles = cache.xi_unbiased
    if len(unbiased_poles):
        s_values = np.asarray(square_cache.S_C[0.0])
        post_inner = _filon_linear(
            common[None, :] * np.conjugate(s_values),
            energy,
            -time,
        ) / (2.0 * np.pi)
        post_outer = (
            -np.exp(1j * outer[:, None] * time)
            * np.conjugate(
                cache.residues_unbiased[None, :]
                * np.exp(
                    1j
                    * outer[:, None]
                    * duration
                )
                * expc(
                    outer[:, None] - unbiased_poles[None, :], elapsed
                )
            )
        )
        result += post_outer @ post_inner
    singular_values = common * np.conjugate(
        Gfr_R_mpm(system, cache, energy)
    )
    result += (
        1j
        * phase
        * _phase_cauchy_linear_uniform(
            singular_values, energy, 0.0, time
        )
        / (2.0 * np.pi)
    )
    return result


def _psi_interaction_direct(
    config: CaseConfig,
    system,
    cache,
    square_cache,
    energy: np.ndarray,
    time: float,
    c_zero: np.ndarray,
    sigma_lesser: np.ndarray,
    alpha: str,
    stored_d_up: np.ndarray | None = None,
) -> np.ndarray:
    """Literal dense generic-lesser C/D oracle for one observation time."""
    outer = energy
    common = c_zero * sigma_lesser
    protocol = config.protocol.name
    if protocol == "upward" or (
        protocol == "square" and time <= float(config.protocol.duration)
    ):
        da = system.Delta(alpha)
        d_values = np.asarray(
            D_up(system, 0.0, energy, outer, time, alpha, cache)
        ).reshape(len(outer), len(energy))
        smooth = (
            np.exp(
                1j
                * (outer[:, None] - energy[None, :] + da)
                * time
            )
            * common[None, :]
            * np.conjugate(d_values)
        )
        result = np.trapezoid(smooth, energy, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * energy * time)
            * common
            * np.conjugate(Gfr_R_mpm(system, cache, energy))
        )
        result += (
            1j
            * np.exp(1j * (outer + da) * time)
            * integrate_cauchy_linear(outer, energy, singular_values)
            / (2.0 * np.pi)
        )
        return result
    if protocol == "downward":
        d_values = np.asarray(
            D_down(system, 0.0, energy, outer, time, cache)
        ).reshape(len(outer), len(energy))
        smooth = (
            np.exp(1j * (outer[:, None] - energy[None, :]) * time)
            * common[None, :]
            * np.conjugate(d_values)
        )
        result = np.trapezoid(smooth, energy, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * energy * time)
            * common
            * np.conjugate(Gbiased_R_mpm(system, cache, energy))
        )
        result += (
            1j
            * np.exp(1j * outer * time)
            * integrate_cauchy_linear(outer, energy, singular_values)
            / (2.0 * np.pi)
        )
        return result

    duration = float(config.protocol.duration)
    phase = np.exp(1j * system.Delta(alpha) * duration)
    if stored_d_up is None:
        stored_d_up = np.asarray(
            D_up(system, 0.0, energy, outer, duration, alpha, cache)
        ).reshape(len(outer), len(energy))
    d_greater = np.asarray(
        D_square(
            system,
            0.0,
            energy,
            outer,
            time,
            alpha,
            cache,
            square_cache,
            duration,
        )
    ).reshape(len(outer), len(energy))
    history = np.conjugate(stored_d_up) + np.conjugate(d_greater) / phase
    smooth = (
        np.exp(1j * (outer[:, None] - energy[None, :]) * time)
        * common[None, :]
        * phase
        * history
    )
    result = np.trapezoid(smooth, energy, axis=1) / (2.0 * np.pi)
    singular_values = (
        np.exp(-1j * energy * time)
        * common
        * np.conjugate(Gfr_R_mpm(system, cache, energy))
    )
    result += (
        1j
        * phase
        * np.exp(1j * outer * time)
        * integrate_cauchy_linear(outer, energy, singular_values)
        / (2.0 * np.pi)
    )
    return result


def _observable_indices(config, observable_grid: np.ndarray) -> np.ndarray:
    """Select a nested non-negative output grid when requested.

    Observable downsampling is deliberately restricted to existing trajectory
    nodes.  This keeps currents, occupations, collision sources, and their
    derivatives on one self-consistent time axis; it is not a second time
    quadrature or an interpolation policy.
    """
    observable_grid = np.asarray(observable_grid, dtype=float)
    candidates = np.flatnonzero(
        observable_grid >= -10.0 * np.finfo(float).eps
    )
    requested = config.numerics.current_time_points
    if requested is None or int(requested) == len(candidates):
        return candidates
    requested = int(requested)
    if requested < 2:
        raise ValueError("current_time_points must be at least two when set.")
    intervals = len(candidates) - 1
    if intervals % (requested - 1):
        raise ValueError(
            "current_time_points must select a nested subset of the trajectory nodes."
        )
    stride = intervals // (requested - 1)
    return candidates[::stride]


def direct_oracle_observables(config, kernel, trajectory):
    """Literal dense A/Psi current oracle for an arbitrary installed lesser."""
    system = make_system(config)
    cache = build_pole_cache(system, kernel)
    observable_grid = np.asarray(trajectory.observable_time)
    positive_indices = _observable_indices(config, observable_grid)
    projection_times = np.array(observable_grid[positive_indices], copy=True)
    projection_times[np.abs(projection_times) <= 10.0 * np.finfo(float).eps] = 0.0
    # The saved observable grid is authoritative.  In particular, do not
    # regenerate a second time axis with an unrelated linspace for the oracle.
    output_time = projection_times
    energy = _current_energy_grid(config, trajectory.energy)
    a_values, c_zero, square_cache = _pulse_amplitudes(
        config, system, cache, output_time, energy
    )
    sigma_lesser = 1j * np.interp(
        energy,
        kernel.energy,
        kernel.sigma_lesser.imag,
        left=0.0,
        right=0.0,
    )
    has_interaction_source = np.max(np.abs(sigma_lesser)) > 0.0
    stored_b: dict[str, dict[str, np.ndarray]] = {}
    stored_d: dict[str, np.ndarray] = {}
    if (
        config.protocol.name == "square"
        and np.any(output_time > float(config.protocol.duration))
    ):
        duration = float(config.protocol.duration)
        for alpha in system.lead_names:
            stored_b[alpha] = {
                beta: np.asarray(
                    B_up(
                        system,
                        energy,
                        energy,
                        duration,
                        alpha,
                        beta,
                        cache,
                    )
                ).reshape(len(energy), len(energy))
                for beta in system.lead_names
            }
            stored_d[alpha] = np.asarray(
                D_up(
                    system,
                    0.0,
                    energy,
                    energy,
                    duration,
                    alpha,
                    cache,
                )
            ).reshape(len(energy), len(energy))

    currents = {
        lead: np.empty(len(output_time), dtype=float)
        for lead in system.lead_names
    }
    for time_index, observation_time in enumerate(output_time):
        amplitude_row = {
            lead: a_values[lead][time_index] for lead in system.lead_names
        }
        for alpha in system.lead_names:
            if config.protocol.name == "upward":
                psi = psi_lead_up_batch(
                    system,
                    cache,
                    energy,
                    energy,
                    observation_time,
                    alpha,
                    amplitude_row,
                )
            elif config.protocol.name == "downward":
                psi = psi_lead_down_batch(
                    system,
                    cache,
                    energy,
                    energy,
                    observation_time,
                    alpha,
                    amplitude_row,
                )
            elif observation_time <= float(config.protocol.duration):
                psi = psi_lead_up_batch(
                    system,
                    cache,
                    energy,
                    energy,
                    observation_time,
                    alpha,
                    amplitude_row,
                )
            else:
                psi = psi_lead_square_batch(
                    system,
                    cache,
                    square_cache,
                    energy,
                    energy,
                    observation_time,
                    alpha,
                    amplitude_row,
                    stored_b[alpha],
                )
            if has_interaction_source:
                psi += _psi_interaction_direct(
                    config,
                    system,
                    cache,
                    square_cache,
                    energy,
                    observation_time,
                    c_zero[time_index],
                    sigma_lesser,
                    alpha,
                    stored_d.get(alpha),
                )
            f_alpha = fermi_dirac(
                energy, system.beta_fc(alpha), system.mu_fc(alpha)
            )
            integrand = np.real(linewidth(system, alpha, energy)) * np.imag(
                psi + f_alpha * amplitude_row[alpha]
            )
            currents[alpha][time_index] = -2.0 * np.trapezoid(
                integrand, energy
            ) / (2.0 * np.pi)

    occupation = np.asarray(trajectory.occupation)[positive_indices]
    continuity = continuity_residual(currents, occupation, output_time)
    diagnostics = dict(trajectory.diagnostics)
    diagnostics.update(
        {
            "current_method": "literal_direct_generic_lesser_oracle",
            "current_representation_effective": "direct_A_psi_oracle",
            "current_energy_min": float(energy[0]),
            "current_energy_max": float(energy[-1]),
            "current_energy_points": int(len(energy)),
            "current_energy_step": float(energy[1] - energy[0]),
            "observable_time_points": int(len(output_time)),
            "observable_time_step": float(
                output_time[1] - output_time[0]
            ) if len(output_time) > 1 else 0.0,
            "stationary_grid_eta": float(config.numerics.eta),
            "transient_retarded_boundary_epsilon": float(
                RETARDED_BOUNDARY_EPS
            ),
            "transient_retarded_uses_eta_zero_limit": True,
            "max_continuity_residual": float(np.max(np.abs(continuity))),
        }
    )
    collision_source = trajectory.collision_source
    if collision_source is not None:
        collision_source = np.asarray(collision_source)[positive_indices]
        mismatch = continuity - collision_source
        derivative = time_derivative(occupation, output_time)
        current_scale = max(
            1e-12,
            *(float(np.max(np.abs(value))) for value in currents.values()),
            float(np.max(np.abs(derivative))),
        )
        diagnostics["current_scale"] = current_scale
        _record_collision_mismatch(
            config, diagnostics, output_time, mismatch, current_scale
        )
    collision_source_finite = trajectory.collision_source_finite
    if collision_source_finite is not None:
        collision_source_finite = np.asarray(collision_source_finite)[positive_indices]
    collision_source_prehistory = trajectory.collision_source_prehistory
    if collision_source_prehistory is not None:
        collision_source_prehistory = np.asarray(collision_source_prehistory)[positive_indices]
    return replace(
        trajectory,
        observable_time=output_time,
        currents=currents,
        occupation=occupation,
        continuity_residual=continuity,
        collision_source=collision_source,
        collision_source_finite=collision_source_finite,
        collision_source_prehistory=collision_source_prehistory,
        diagnostics=diagnostics,
    )


def _stationary_history_psi(
    config: CaseConfig,
    system,
    cache,
    kernel: StationaryKernel,
    time: np.ndarray,
    energy: np.ndarray,
    output_time: np.ndarray,
    a_values: dict[str, np.ndarray],
    c_zero: np.ndarray,
    lead: str,
    *,
    batch_size: int,
    source_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Analytic ``(-infinity, t_min)`` contribution to ``Psi_alpha``.

    Before the projection window every source is stationary.  Its history
    integral is therefore a Cauchy convolution on the uniform energy grid;
    FFT convolution evaluates every outer energy simultaneously.  The
    diagonal value includes both the delta contribution and the regular
    finite part of ``exp(i x t_min)/(i x + 0+)``.
    """
    from scipy.signal import fftconvolve

    t_min = float(time[0])
    source_data = _lesser_sources(
        config, system, kernel, output_time, energy, a_values, c_zero
    )
    boundary_amplitudes: list[np.ndarray] = []
    negative_slopes: list[float] = []
    for source_lead in system.lead_names:
        boundary, _ = _stationary_boundaries(
            config, system, cache, energy, source_lead
        )
        boundary_amplitudes.append(np.asarray(boundary))
        negative_slopes.append(
            system.Delta(source_lead)
            if config.protocol.name == "downward"
            else 0.0
        )
    _, interaction_boundary = _stationary_boundaries(
        config, system, cache, energy, system.lead_names[0]
    )
    boundary_amplitudes.append(np.asarray(interaction_boundary))
    negative_slopes.append(0.0)

    alpha_slope = (
        system.Delta(lead) if config.protocol.name == "downward" else 0.0
    )
    output = np.zeros((len(output_time), len(energy)), dtype=np.complex128)
    energy_quadrature = trapezoid_weights(energy)
    indexed_sources = list(
        zip(source_data, boundary_amplitudes, negative_slopes)
    )
    if source_indices is not None:
        indexed_sources = [indexed_sources[index] for index in source_indices]
    for (factor, source_weight), boundary, source_slope in indexed_sources:
        pole_shift = alpha_slope - source_slope
        source_density = source_weight / energy_quadrature
        for start in range(0, len(output_time), batch_size):
            stop = min(start + batch_size, len(output_time))
            interpolated_values = (
                factor[start:stop]
                * source_density[None, :]
                * np.conjugate(boundary)[None, :]
                * np.exp(1j * energy[None, :] * t_min)
            )
            cauchy = _cauchy_linear_uniform(
                interpolated_values,
                energy,
                pole_shift,
            )
            output[start:stop] += (
                1j
                * np.exp(
                    -1j * (energy[None, :] + pole_shift) * t_min
                )
                * cauchy
            )

    chi = _drive_integral(config, output_time, lead)
    return output * np.exp(
        1j * (output_time[:, None] * energy[None, :] + chi[:, None])
    )


def _stationary_endpoint_current_integrand(
    config: CaseConfig,
    system,
    cache,
    kernel: StationaryKernel,
    energy: np.ndarray,
    lead: str,
    *,
    biased: bool,
) -> np.ndarray:
    """Complex stationary endpoint integrand on the lead-energy grid.

    ``Psi`` is evaluated as its causal half-Fourier transform.  Keeping this
    complex endpoint object, rather than only its integrated current, permits
    a clean stationary-plus-dephasing split for the Filon transform.
    """
    endpoint_energy = energy
    if biased:
        endpoint_retarded = Gbiased_R_mpm(
            system, cache, endpoint_energy
        )
    else:
        endpoint_retarded = Gfr_R_mpm(system, cache, endpoint_energy)
    total_lesser = 1j * np.interp(
        endpoint_energy,
        kernel.energy,
        kernel.sigma_lesser.imag,
        left=0.0,
        right=0.0,
    )
    for source_lead in system.lead_names:
        source_shift = system.Delta(source_lead) if biased else 0.0
        source_energy = endpoint_energy - source_shift
        total_lesser += (
            1j
            * fermi_dirac(
                source_energy,
                system.beta_fc(source_lead),
                system.mu_fc(source_lead),
            )
            * np.real(linewidth(system, source_lead, source_energy))
        )
    endpoint_lesser = (
        endpoint_retarded
        * total_lesser
        * np.conjugate(endpoint_retarded)
    )
    lead_shift = system.Delta(lead) if biased else 0.0
    endpoint_psi = (
        1j
        * _cauchy_linear_uniform(
            endpoint_lesser, endpoint_energy, lead_shift
        )
        / (2.0 * np.pi)
    )
    lead_retarded = (
        Gbiased_R_mpm(system, cache, energy + lead_shift)
        if biased
        else Gfr_R_mpm(system, cache, energy)
    )
    lead_gamma = np.real(linewidth(system, lead, energy))
    lead_fermi = fermi_dirac(
        energy, system.beta_fc(lead), system.mu_fc(lead)
    )
    return lead_gamma * (endpoint_psi + lead_fermi * lead_retarded)


def _lead_fermi_memory_kernel(
    system,
    lead: str,
    lag: np.ndarray,
    *,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Return ``integral f(E) Gamma_alpha(E) exp(i E tau) dE/2pi``.

    For positive relative time the Lorentzian pole and the upper-half-plane
    Fermi poles give an exponentially convergent residue series.  The
    equal-time value is evaluated independently because the Matsubara series
    only converges algebraically there.
    """
    from scipy.integrate import quad

    lag = np.asarray(lag, dtype=float)
    if lag.ndim != 1 or len(lag) == 0 or np.any(lag < 0.0):
        raise ValueError("Lead-memory lags must be a non-empty non-negative grid.")
    bandwidth = float(system.W)
    gamma0 = float(system.Gamma0(lead))
    beta = float(system.beta_fc(lead))
    mu = float(system.mu_fc(lead))
    output = np.empty(len(lag), dtype=np.complex128)

    def equal_time_integrand(energy: float) -> float:
        occupation = float(np.real(fermi_dirac(energy, beta, mu)))
        gamma = gamma0 * bandwidth**2 / (energy**2 + bandwidth**2)
        return occupation * gamma / (2.0 * np.pi)

    output[0] = quad(
        equal_time_integrand,
        -np.inf,
        np.inf,
        epsabs=tolerance,
        epsrel=1e-12,
        limit=500,
    )[0]
    if len(lag) == 1:
        return output

    positive_lag = lag[1:]
    minimum_lag = float(np.min(positive_lag))
    logarithmic_target = max(20.0, -np.log(tolerance))
    n_terms = max(
        256,
        int(np.ceil(beta * logarithmic_target / (2.0 * np.pi * minimum_lag))),
    )
    n_terms = min(n_terms, 16_384)
    indices = np.arange(n_terms, dtype=float)
    matsubara = (2.0 * indices + 1.0) * np.pi / beta
    poles = mu + 1j * matsubara
    gamma_at_poles = (
        gamma0 * bandwidth**2 / (poles**2 + bandwidth**2)
    )
    fermi_residues = np.zeros(len(positive_lag), dtype=np.complex128)
    for start in range(0, n_terms, 256):
        stop = min(start + 256, n_terms)
        selected_poles = poles[start:stop]
        fermi_residues += np.sum(
            (-1j / beta)
            * gamma_at_poles[start:stop, None]
            * np.exp(
                1j * selected_poles[:, None] * positive_lag[None, :]
            ),
            axis=0,
        )
    lorentzian_residue = (
        gamma0
        * bandwidth
        / 2.0
        * fermi_dirac(1j * bandwidth, beta, mu)
        * np.exp(-bandwidth * positive_lag)
    )
    output[1:] = lorentzian_residue + fermi_residues
    return output


def _downward_stationary_history_tail_current(
    config: CaseConfig,
    kernel: StationaryKernel,
    trajectory: FixedKernelTrajectory,
    system,
    cache,
    output_time: np.ndarray,
    lead: str,
) -> np.ndarray:
    """Exact/source-factorized ``(-infinity,t_min)`` downward current tail."""
    time = trajectory.time
    energy = trajectory.energy
    dt = float(time[1] - time[0])
    t_min = float(time[0])
    delta_alpha = float(system.Delta(lead))
    chi_alpha = _drive_integral(config, output_time, lead)
    a_values, c_zero, _ = _pulse_amplitudes(
        config, system, cache, output_time, energy
    )
    sources = _lesser_sources(
        config, system, kernel, output_time, energy, a_values, c_zero
    )

    boundaries: list[np.ndarray] = []
    source_slopes: list[float] = []
    protocol = strategy_for(config, system, cache, energy)
    for source_lead in system.lead_names:
        boundary = protocol.stationary_a(source_lead)
        boundaries.append(np.asarray(boundary))
        source_slopes.append(float(system.Delta(source_lead)))
    interaction_boundary = protocol.stationary_c()
    boundaries.append(np.asarray(interaction_boundary))
    source_slopes.append(0.0)

    gamma_prefactor = (
        system.Gamma0(lead)
        * system.W
        / 2.0
        * np.exp(-system.W * output_time + 1j * chi_alpha)
    )
    lesser_tail = np.zeros(len(output_time), dtype=np.complex128)
    for (factor, source_weight), boundary, source_slope in zip(
        sources, boundaries, source_slopes
    ):
        exponent = system.W + 1j * (
            energy + source_slope - delta_alpha
        )
        stationary_integral = np.exp(exponent * t_min) / exponent
        lesser_tail += gamma_prefactor * (
            factor
            @ (
                source_weight
                * np.conjugate(boundary)
                * stationary_integral
            )
        )

    # The Fermi kernel's slowest decay is pi/beta.  Extending by 64 time
    # units makes the unrepresented remainder below 1e-10 for production
    # beta=10; the low-rank auxiliary form avoids a dense energy contraction.
    history_min = t_min - max(64.0, 24.0 * system.beta_fc(lead) / np.pi)
    history_intervals = int(np.ceil((t_min - history_min) / dt))
    history_min = t_min - history_intervals * dt
    history_time = np.linspace(
        history_min, t_min, history_intervals + 1
    )
    biased_matrix = _auxiliary_matrix(system, cache, biased=True)
    unbiased_matrix = _auxiliary_matrix(system, cache, biased=False)
    rows_unbiased, _ = _auxiliary_rows_and_columns(
        unbiased_matrix, output_time
    )
    _, columns_biased = _auxiliary_rows_and_columns(
        biased_matrix, -history_time
    )
    maximum_lag = float(output_time[-1] - history_min)
    lag_count = int(round(maximum_lag / dt)) + 1
    lag_grid = np.arange(lag_count, dtype=float) * dt
    fermi_memory = _lead_fermi_memory_kernel(system, lead, lag_grid)
    retarded_tail = np.zeros(len(output_time), dtype=np.complex128)
    block_size = 64
    for start in range(0, len(output_time), block_size):
        stop = min(start + block_size, len(output_time))
        observation = output_time[start:stop]
        lag_indices = np.rint(
            (observation[:, None] - history_time[None, :]) / dt
        ).astype(int)
        phase = np.exp(
            1j
            * (
                chi_alpha[start:stop, None]
                - delta_alpha * history_time[None, :]
            )
        )
        green_retarded = -1j * (
            rows_unbiased[start:stop] @ columns_biased.T
        )
        from scipy.integrate import simpson

        retarded_tail[start:stop] = simpson(
            phase * fermi_memory[lag_indices] * green_retarded,
            x=history_time,
            axis=1,
        )
    return -2.0 * np.imag(lesser_tail + retarded_tail)


def _attach_time_domain_observables(
    config, kernel, trajectory, *, conserve_noninteracting: bool = True
):
    """Physical current from analytic lead memory and dense two-time G.

    This form integrates relative time rather than resolving narrow physical
    device poles on the reconstruction-energy grid.  The retarded term uses
    the causal open upper endpoint; the lesser term retains trapezoidal
    endpoint weighting.
    """
    gr = trajectory.dense_green_retarded
    gl = trajectory.dense_green_lesser
    if gr is None or gl is None:
        raise ValueError("Time-domain currents require dense final Green functions.")
    system = make_system(config)
    cache = build_pole_cache(system, kernel)
    time = trajectory.time
    dt = float(time[1] - time[0])
    observable_grid = np.asarray(trajectory.observable_time)
    positive_indices = _observable_indices(config, observable_grid)
    output_time = observable_grid[positive_indices]
    lag = np.arange(len(time), dtype=float) * dt
    currents: dict[str, np.ndarray] = {}
    endpoint_currents: dict[str, dict[str, float]] = {}
    for lead in system.lead_names:
        gamma_memory = (
            system.Gamma0(lead)
            * system.W
            / 2.0
            * np.exp(-system.W * lag)
        )
        fermi_memory = _lead_fermi_memory_kernel(system, lead, lag)
        chi = _drive_integral(config, time, lead)
        current = np.empty(len(output_time), dtype=float)
        for output_index, row in enumerate(positive_indices):
            columns = np.arange(row + 1)
            lag_indices = row - columns
            phase = np.exp(1j * (chi[row] - chi[: row + 1]))
            lesser_integrand = (
                phase
                * gamma_memory[lag_indices]
                * gl[row, : row + 1]
            )
            retarded_integrand = (
                phase
                * fermi_memory[lag_indices]
                * gr[row, : row + 1]
            )
            retarded_integrand[-1] *= 2.0
            from scipy.integrate import simpson

            current[output_index] = -2.0 * np.imag(
                simpson(lesser_integrand, x=time[: row + 1])
                + simpson(retarded_integrand, x=time[: row + 1])
            )
        if config.protocol.name == "downward":
            current += _downward_stationary_history_tail_current(
                config,
                kernel,
                trajectory,
                system,
                cache,
                output_time,
                lead,
            )
        currents[lead] = current

        unbiased = _stationary_endpoint_current_integrand(
            config,
            system,
            cache,
            kernel,
            trajectory.energy,
            lead,
            biased=False,
        )
        biased = _stationary_endpoint_current_integrand(
            config,
            system,
            cache,
            kernel,
            trajectory.energy,
            lead,
            biased=True,
        )
        endpoint_currents[lead] = {
            "unbiased": float(
                -2.0 * np.imag(np.trapezoid(unbiased, trajectory.energy))
                / (2.0 * np.pi)
            ),
            "biased": float(
                -2.0 * np.imag(np.trapezoid(biased, trajectory.energy))
                / (2.0 * np.pi)
            ),
        }
        if config.protocol.name != "downward":
            currents[lead][0] = endpoint_currents[lead]["unbiased"]

    occupation = np.real(-1j * np.diag(gl)[positive_indices])
    diagnostics = dict(trajectory.diagnostics)
    raw_continuity = continuity_residual(currents, occupation, output_time)
    corrected_currents: dict[str, np.ndarray] = {}
    continuity = _apply_noninteracting_conserving_correction(
        config,
        system,
        kernel,
        currents,
        occupation,
        output_time,
        diagnostics,
        enabled=conserve_noninteracting,
        corrected_currents=corrected_currents,
    )
    diagnostics["raw_continuity_residual_max"] = float(
        np.max(np.abs(raw_continuity))
    )
    diagnostics["corrected_continuity_residual_max"] = float(
        np.max(np.abs(continuity))
    )
    diagnostics.update(
        {
            "current_method": "analytic_lead_memory_time_domain",
            "current_time_quadrature": "causal_simpson_full_retarded_endpoint",
            "current_history_min": float(time[0]),
            "current_stationary_endpoint_handling": True,
            "current_stationary_history_tail_included": bool(
                config.protocol.name == "downward"
            ),
            "stationary_endpoint_currents": endpoint_currents,
            "max_continuity_residual": float(np.max(np.abs(continuity))),
        }
    )
    collision_source = trajectory.collision_source
    if collision_source is not None:
        collision_source = collision_source[positive_indices]
        mismatch = continuity - collision_source
        derivative = time_derivative(occupation, output_time)
        current_scale = max(
            1e-12,
            *(float(np.max(np.abs(value))) for value in currents.values()),
            float(np.max(np.abs(derivative))),
        )
        diagnostics["current_scale"] = current_scale
        _record_collision_mismatch(
            config, diagnostics, output_time, mismatch, current_scale
        )
    collision_source_finite = trajectory.collision_source_finite
    if collision_source_finite is not None:
        collision_source_finite = collision_source_finite[positive_indices]
    collision_source_prehistory = trajectory.collision_source_prehistory
    if collision_source_prehistory is not None:
        collision_source_prehistory = collision_source_prehistory[positive_indices]
    return replace(
        trajectory,
        observable_time=output_time,
        currents=currents,
        corrected_currents=corrected_currents,
        occupation=occupation,
        continuity_residual=continuity,
        raw_continuity_residual=raw_continuity,
        collision_source=collision_source,
        collision_source_finite=collision_source_finite,
        collision_source_prehistory=collision_source_prehistory,
        diagnostics=diagnostics,
    )


def _filon_current_integral(
    integrand: np.ndarray,
    stationary_endpoint: np.ndarray,
    energy: np.ndarray,
    time: float,
    drive_phase: float,
) -> complex:
    """Integrate a stationary part plus its dephasing transient by Filon."""
    if abs(time) * abs(float(energy[1] - energy[0])) <= 0.5:
        return np.trapezoid(integrand, energy)
    transient_amplitude = (integrand - stationary_endpoint) * np.exp(
        -1j * (energy * time + drive_phase)
    )
    return np.trapezoid(stationary_endpoint, energy) + np.exp(
        1j * drive_phase
    ) * _filon_linear(transient_amplitude, energy, time)


def _apply_noninteracting_conserving_correction(
    config,
    system,
    kernel,
    currents: dict[str, np.ndarray],
    occupation: np.ndarray,
    output_time: np.ndarray,
    diagnostics: dict,
    *,
    enabled: bool,
    corrected_currents: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Remove the finite-quadrature common lead-current mode at ``g=0``.

    In the noninteracting limit the collision source vanishes and continuity
    fixes only the sum of lead currents.  The common mode is numerically
    underdetermined by finite energy/current quadrature, while the transport
    (lead-antisymmetric) mode is physical.  A corrected *copy* is formed by
    subtracting the raw continuity residual in proportion to each lead
    linewidth.  The input ``currents`` mapping is never mutated; this keeps
    the physical pole/Filon result archive-comparable.

    ``enabled`` is deliberately opt-in: the literal direct A/Psi oracle and
    raw fast/direct comparisons must remain unmodified validation references.
    """
    continuity = continuity_residual(currents, occupation, output_time)
    # A synthetic/arbitrary installed lesser kernel may be used with a
    # zero-valued model coupling in validation.  It is not the collision-free
    # noninteracting limit and must remain unmodified for oracle comparisons.
    kernel_lesser = np.asarray(kernel.sigma_lesser)
    collision_free = (
        kernel_lesser.size == 0 or np.max(np.abs(kernel_lesser)) == 0.0
    )
    if corrected_currents is not None:
        corrected_currents.clear()
    if not enabled or config.model.coupling_meV != 0.0 or not collision_free:
        return continuity

    total_gamma = sum(system.Gamma0(lead) for lead in system.lead_names)
    if total_gamma <= 0.0:
        raise ValueError(
            "Noninteracting current correction requires positive lead coupling."
        )
    correction = np.array(continuity, copy=True)
    corrected = {
        lead: np.array(currents[lead], copy=True)
        for lead in system.lead_names
    }
    for lead in system.lead_names:
        corrected[lead] -= (system.Gamma0(lead) / total_gamma) * correction
    if corrected_currents is not None:
        corrected_currents.update(corrected)
    diagnostics["noninteracting_conserving_current_correction_max"] = float(
        np.max(np.abs(correction))
    )
    return continuity_residual(corrected, occupation, output_time)


def _downward_boundary_currents(
    config: CaseConfig,
    kernel: StationaryKernel,
    output_time: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the reviewed downward A/Psi boundary-memory current."""
    system = make_system(config)
    coarse_cache = build_pole_cache(system, kernel)
    coarse_step = float(config.numerics.energy_step)
    span = config.numerics.energy_max - config.numerics.energy_min
    intervals = int(round(span / coarse_step))
    energy = np.linspace(
        config.numerics.energy_min,
        config.numerics.energy_max,
        intervals + 1,
    )
    cache = coarse_cache
    a_values, c_zero, _ = _pulse_amplitudes(
        config, system, cache, output_time, energy
    )
    sigma_lesser = 1j * np.interp(
        energy,
        kernel.energy,
        kernel.sigma_lesser.imag,
        left=0.0,
        right=0.0,
    )
    has_interaction_source = np.max(np.abs(sigma_lesser)) > 0.0
    currents: dict[str, np.ndarray] = {}
    for lead in system.lead_names:
        gamma = np.real(linewidth(system, lead, energy))
        f_alpha = fermi_dirac(
            energy, system.beta_fc(lead), system.mu_fc(lead)
        )
        endpoint = _stationary_endpoint_current_integrand(
            config,
            system,
            cache,
            kernel,
            energy,
            lead,
            biased=False,
        )
        chi = _drive_integral(config, output_time, lead)
        current = np.empty(len(output_time), dtype=float)
        for index, observation_time in enumerate(output_time):
            amplitudes = {
                source_lead: a_values[source_lead][index]
                for source_lead in system.lead_names
            }
            psi = _psi_leads_down_fast(
                system,
                cache,
                energy,
                float(observation_time),
                amplitudes,
                lead,
            )
            if has_interaction_source:
                psi += _psi_interaction_down_fast(
                    system,
                    cache,
                    energy,
                    float(observation_time),
                    c_zero[index],
                    sigma_lesser,
                    lead,
                )
            integrand = gamma * (psi + f_alpha * a_values[lead][index])
            integrated = _filon_current_integral(
                integrand,
                endpoint,
                energy,
                float(observation_time),
                float(chi[index]),
            )
            current[index] = -2.0 * np.imag(integrated) / (2.0 * np.pi)
        currents[lead] = current
    return currents


def finite_window_observables(
    config,
    kernel,
    trajectory,
    *,
    prefer_time_domain: bool = False,
    conserve_noninteracting: bool = False,
    previous_basis=None,
    progress=None,
):
    """Evaluate currents from the pole-energy A/C/Psi representation.

    The historical ``prefer_time_domain`` argument is retained for callers
    loading old notebooks, but is deliberately ignored.  Dense two-time
    Green functions belong to projection/filtering and validation only; they
    are never integrated to obtain production currents.
    """
    gl = trajectory.dense_green_lesser
    system = make_system(config)
    # Reuse the validated basis from the trajectory's installed-kernel solve.
    # Rebuilding a fresh AAA representation here can reject an otherwise
    # converged case solely while evaluating observables.  The fit remains
    # fail-closed; an incompatible/invalid basis still falls through to the
    # normal fresh-fit schedule in build_pole_cache.
    cache = (
        previous_basis
        if isinstance(previous_basis, PoleCache)
        else build_pole_cache(
            system,
            kernel,
            previous_basis=previous_basis,
            progress=progress,
        )
    )
    square_cache_store = SquareKernelCacheStore()
    time = trajectory.time
    energy = _current_energy_grid(config, trajectory.energy)
    observable_grid = np.asarray(trajectory.observable_time)
    positive_indices = _observable_indices(config, observable_grid)
    output_time = observable_grid[positive_indices]
    representation = config.numerics.current_representation
    a_values, c_zero, square_cache = _pulse_amplitudes(
        config, system, cache, output_time, energy,
        square_cache_store=square_cache_store,
    )
    sigma_lesser = 1j * np.interp(
        energy,
        kernel.energy,
        kernel.sigma_lesser.imag,
        left=0.0,
        right=0.0,
    )
    has_interaction_source = np.max(np.abs(sigma_lesser)) > 0.0
    currents: dict[str, np.ndarray] = {}
    endpoint_currents: dict[str, dict[str, float]] = {}
    for lead in system.lead_names:
        gamma = np.real(linewidth(system, lead, energy))
        f_alpha = fermi_dirac(
            energy, system.beta_fc(lead), system.mu_fc(lead)
        )
        if config.protocol.name in ("upward", "downward", "square"):
            def solve_lead_sector(observation_time, time_index):
                values = {
                    source_lead: a_values[source_lead][time_index]
                    for source_lead in system.lead_names
                }
                if config.protocol.name == "upward":
                    return _psi_leads_up_fast(
                        system, cache, energy, observation_time, values, lead
                    )
                if config.protocol.name == "downward":
                    return _psi_leads_down_fast(
                        system, cache, energy, observation_time, values, lead
                    )
                return _psi_leads_square_fast(
                    system,
                    cache,
                    square_cache,
                    energy,
                    observation_time,
                    values,
                    lead,
                )

            psi = np.vstack(
                [
                    solve_lead_sector(
                        float(observation_time),
                        time_index,
                    )
                    for time_index, observation_time in enumerate(output_time)
                ]
            )
            if has_interaction_source:
                def solve_interaction_sector(observation_time, time_index):
                    if config.protocol.name == "upward":
                        return _psi_interaction_up_fast(
                            system,
                            cache,
                            energy,
                            observation_time,
                            c_zero[time_index],
                            sigma_lesser,
                            lead,
                        )
                    if config.protocol.name == "downward":
                        return _psi_interaction_down_fast(
                            system,
                            cache,
                            energy,
                            observation_time,
                            c_zero[time_index],
                            sigma_lesser,
                            lead,
                        )
                    return _psi_interaction_square_fast(
                        system,
                        cache,
                        square_cache,
                        energy,
                        observation_time,
                        c_zero[time_index],
                        sigma_lesser,
                        lead,
                    )

                psi += np.vstack(
                    [
                        solve_interaction_sector(
                            float(observation_time), time_index
                        )
                        for time_index, observation_time in enumerate(output_time)
                    ]
                )
        else:
            chi = _drive_integral(config, time, lead)
            psi = _finite_window_psi(
                gl,
                time,
                energy,
                chi,
                positive_indices,
                batch_size=max(1, int(system.current_energy_batch)),
            )
            psi += _stationary_history_psi(
                config,
                system,
                cache,
                kernel,
                time,
                energy,
                output_time,
                a_values,
                c_zero,
                lead,
                batch_size=max(1, int(system.current_energy_batch)),
            )
        complex_integrand = gamma[None, :] * (
            psi + f_alpha[None, :] * a_values[lead]
        )
        unbiased_endpoint = _stationary_endpoint_current_integrand(
            config,
            system,
            cache,
            kernel,
            energy,
            lead,
            biased=False,
        )
        biased_endpoint = _stationary_endpoint_current_integrand(
            config,
            system,
            cache,
            kernel,
            energy,
            lead,
            biased=True,
        )
        endpoint_currents[lead] = {
            "unbiased": float(
                -2.0
                * np.imag(np.trapezoid(unbiased_endpoint, energy))
                / (2.0 * np.pi)
            ),
            "biased": float(
                -2.0
                * np.imag(np.trapezoid(biased_endpoint, energy))
                / (2.0 * np.pi)
            ),
        }
        chi = _drive_integral(config, output_time, lead)
        integrated = np.empty(len(output_time), dtype=np.complex128)
        for time_index, observation_time in enumerate(output_time):
            use_biased_endpoint = config.protocol.name == "upward" or (
                config.protocol.name == "square"
                and observation_time <= float(config.protocol.duration)
            )
            endpoint = (
                biased_endpoint
                if use_biased_endpoint
                else unbiased_endpoint
            )
            integrated[time_index] = _filon_current_integral(
                complex_integrand[time_index],
                endpoint,
                energy,
                float(observation_time),
                float(chi[time_index]),
            )
            # Current evaluation is pole/Filon based.  Emit scalar progress
            # only; no dense time-domain integral is involved.
            _progress(
                progress,
                "currents",
                batch_completed=time_index + 1,
                batch_total=len(output_time),
                batch_unit=f"{lead} observable times (pole/Filon)",
                lead=lead,
                observable_time_nat=float(observation_time),
                observable_time_ps=float(
                    time_to_ps(observation_time, config.model.gamma_eV)
                ),
                current_nat=float(
                    -2.0 * np.imag(integrated[time_index]) / (2.0 * np.pi)
                ),
                current_uA=float(
                    current_to_uA(
                        -2.0 * np.imag(integrated[time_index]) / (2.0 * np.pi),
                        config.model.gamma_eV,
                    )
                ),
            )
        currents[lead] = -2.0 * np.imag(integrated) / (2.0 * np.pi)
        switching_surface = np.abs(output_time) <= 10.0 * np.finfo(float).eps
        if config.protocol.name != "downward" and np.any(switching_surface):
            # The partition-free switching value belongs to the connected
            # pre-pulse stationary endpoint.  Residue formulas evaluated at
            # a rounded grid value such as -8.9e-16 otherwise select
            # inconsistent one-sided square-pulse branches.
            currents[lead][switching_surface] = endpoint_currents[lead]["unbiased"]

    if trajectory.occupation is not None:
        occupation = np.asarray(trajectory.occupation, dtype=float)[positive_indices]
    elif gl is not None:
        occupation = np.real(-1j * np.diag(gl)[positive_indices])
    else:
        raise ValueError("Trajectory has no occupation data for pole-current evaluation.")
    diagnostics = dict(trajectory.diagnostics)
    raw_continuity = continuity_residual(currents, occupation, output_time)
    corrected_currents: dict[str, np.ndarray] = {}
    continuity = _apply_noninteracting_conserving_correction(
        config,
        system,
        kernel,
        currents,
        occupation,
        output_time,
        diagnostics,
        enabled=conserve_noninteracting,
        corrected_currents=corrected_currents,
    )
    diagnostics["raw_continuity_residual_max"] = float(
        np.max(np.abs(raw_continuity))
    )
    diagnostics["corrected_continuity_residual_max"] = float(
        np.max(np.abs(continuity))
    )
    diagnostics.update(
        {
            "current_method": "pole_fft_generic_psi",
            "current_representation_effective": "pole_filon",
            "current_energy_quadrature": "stationary_endpoint_filon_linear",
            "current_energy_min": float(energy[0]),
            "current_energy_max": float(energy[-1]),
            "current_energy_points": int(len(energy)),
            "current_energy_step": float(energy[1] - energy[0]),
            "observable_time_points": int(len(output_time)),
            "observable_time_step": float(
                output_time[1] - output_time[0]
            ) if len(output_time) > 1 else 0.0,
            "stationary_grid_eta": float(config.numerics.eta),
            "transient_retarded_boundary_epsilon": float(
                RETARDED_BOUNDARY_EPS
            ),
            "transient_retarded_uses_eta_zero_limit": True,
            "current_history_min": float(time[0]),
            "current_stationary_history_tail_included": True,
            "stationary_endpoint_currents": endpoint_currents,
            "current_output_view": "raw_pole_filon",
            "corrected_current_view": bool(corrected_currents),
            "corrected_current_files": (
                {lead: f"data/current_corrected_{lead}.npy" for lead in corrected_currents}
                if corrected_currents
                else {}
            ),
            "max_continuity_residual": float(np.max(np.abs(continuity))),
        }
    )
    if representation == "time_domain":
        diagnostics["deprecated_current_representation"] = (
            "time_domain requested; pole_fft_generic_psi used"
        )
    collision_source = trajectory.collision_source
    if collision_source is not None:
        collision_source = collision_source[positive_indices]
        mismatch = continuity - collision_source
        derivative = time_derivative(occupation, output_time)
        current_scale = max(
            1e-12,
            *(float(np.max(np.abs(value))) for value in currents.values()),
            float(np.max(np.abs(derivative))),
        )
        diagnostics["current_scale"] = current_scale
        _record_collision_mismatch(
            config, diagnostics, output_time, mismatch, current_scale
        )
    collision_source_finite = trajectory.collision_source_finite
    if collision_source_finite is not None:
        collision_source_finite = collision_source_finite[positive_indices]
    collision_source_prehistory = trajectory.collision_source_prehistory
    if collision_source_prehistory is not None:
        collision_source_prehistory = collision_source_prehistory[positive_indices]
    return replace(
        trajectory,
        observable_time=output_time,
        currents=currents,
        corrected_currents=corrected_currents,
        occupation=occupation,
        continuity_residual=continuity,
        raw_continuity_residual=raw_continuity,
        collision_source=collision_source,
        collision_source_finite=collision_source_finite,
        collision_source_prehistory=collision_source_prehistory,
        diagnostics=diagnostics,
    )


# Deprecated characterization aliases. Production code imports only the
# public names above; these remain for older third-party numerical notebooks.
_attach_direct_oracle_observables = direct_oracle_observables
_attach_finite_window_observables = finite_window_observables
cauchy_linear_uniform = _cauchy_linear_uniform
expc_antiderivatives = _expc_antiderivatives
expc_linear_uniform = _expc_linear_uniform
filon_linear = _filon_linear
finite_window_psi = _finite_window_psi
lead_fermi_memory_kernel = _lead_fermi_memory_kernel
lesser_energy_grid = _lesser_energy_grid
panel_kernel_convolution = _panel_kernel_convolution


def _collision_prehistory_tail(
    config: CaseConfig,
    system,
    cache,
    trajectory: FixedKernelTrajectory,
) -> np.ndarray:
    """Evaluate the omitted ``(-infinity, t_min)`` collision history.

    The pre-pulse source factors are stationary exponentials.  Their energy
    representation is retained by the trajectory only for dense diagnostics;
    a Gauss--Legendre map of the semi-infinite interval then evaluates the
    history tail without extending the projection window or storing another
    dense matrix.  Cross-switch retarded rows use the same auxiliary pole
    propagators as the main trajectory.
    """
    factors = trajectory.lesser_source_factors
    source_weights = trajectory.lesser_source_weights
    source_energy = trajectory.lesser_source_energy
    if not factors or source_energy is None:
        return np.zeros(len(trajectory.time), dtype=float)
    time = np.asarray(trajectory.time, dtype=float)
    t_min = float(time[0])
    # Map x in [0,1] to t'=t_min-scale*x/(1-x).  The pole widths are O(Gamma)
    # in the natural units, so a unit scale resolves the switching boundary
    # while retaining the exponentially damped far tail.
    from numpy.polynomial.legendre import leggauss

    quadrature_order = int(config.numerics.collision_prehistory_quadrature_order)
    nodes, weights = leggauss(quadrature_order)
    x = 0.5 * (nodes + 1.0)
    scale = float(config.numerics.collision_prehistory_scale)
    pre_time = t_min - scale * x / (1.0 - x)
    pre_weight = 0.5 * weights * scale / (1.0 - x) ** 2
    energy = np.asarray(source_energy, dtype=float)
    omega = float(config.model.phonon_energy)
    n0 = (
        float(config.model.phonon_occupation)
        if config.model.phonon_occupation is not None
        else float(1.0 / np.expm1(config.model.beta_ph * omega))
    )
    g2 = float(config.model.coupling) ** 2
    pre_factors: list[np.ndarray] = []
    for source_index, factor in enumerate(factors):
        reference = np.asarray(factor[0], dtype=np.complex128)
        if source_index < len(system.lead_names):
            lead = system.lead_names[source_index]
            shift = system.Delta(lead)
            # Only the downward protocol has a nonzero stationary drive phase
            # before t=0; upward and square are unbiased before the switch.
            chi_delta = (
                shift * (pre_time - t_min)
                if config.protocol.name == "downward"
                else np.zeros_like(pre_time)
            )
        else:
            chi_delta = np.zeros_like(pre_time)
        pre_factors.append(
            np.exp(-1j * (pre_time[:, None] - t_min) * energy[None, :])
            * np.exp(-1j * np.asarray(chi_delta)[:, None])
            * reference[None, :]
        )

    def stationary_lesser(lag_values: np.ndarray) -> np.ndarray:
        values = np.zeros(len(lag_values), dtype=np.complex128)
        for source_index, (factor, weights_e) in enumerate(
            zip(factors, source_weights)
        ):
            reference = np.asarray(factor[0], dtype=np.complex128)
            if source_index < len(system.lead_names):
                shift = system.Delta(system.lead_names[source_index])
                chi = (
                    shift * lag_values
                    if config.protocol.name == "downward"
                    else np.zeros_like(lag_values)
                )
            else:
                chi = np.zeros_like(lag_values)
            values += np.exp(-1j * np.asarray(chi)) * (
                np.exp(-1j * lag_values[:, None] * energy[None, :])
                @ (np.asarray(weights_e) * np.abs(reference) ** 2)
            )
        return values

    unbiased_matrix = _auxiliary_matrix(system, cache, biased=False)
    biased_matrix = _auxiliary_matrix(system, cache, biased=True)
    _, columns_u = _auxiliary_rows_and_columns(unbiased_matrix, -pre_time)
    _, columns_b = _auxiliary_rows_and_columns(biased_matrix, -pre_time)
    positive = np.maximum(time, 0.0)
    rows_u, _ = _auxiliary_rows_and_columns(unbiased_matrix, positive)
    rows_b, _ = _auxiliary_rows_and_columns(biased_matrix, positive)
    duration = float(config.protocol.duration or 0.0)
    if config.protocol.name == "square":
        post_elapsed = np.maximum(time - duration, 0.0)
        rows_post, _ = _auxiliary_rows_and_columns(unbiased_matrix, post_elapsed)
        eigenvalues, right = np.linalg.eig(biased_matrix)
        pulse_matrix = (right * np.exp(-1j * eigenvalues * duration)[None, :]) @ np.linalg.inv(right)

    tail = np.zeros(len(time), dtype=float)
    stationary_pre_retarded = lambda lag: _stationary_retarded(
        cache.xi_biased if config.protocol.name == "downward" else cache.xi_unbiased,
        cache.residues_biased if config.protocol.name == "downward" else cache.residues_unbiased,
        lag,
    )
    for i, observation in enumerate(time):
        if observation <= t_min + 1e-14:
            continue
        if config.protocol.name == "upward":
            retarded_cross = -1j * (rows_b[i] @ columns_u.T)
        elif config.protocol.name == "downward":
            retarded_cross = -1j * (rows_u[i] @ columns_b.T)
        elif config.protocol.name == "square":
            if observation <= duration + 1e-12:
                retarded_cross = -1j * (rows_b[i] @ columns_u.T)
            else:
                retarded_cross = -1j * ((rows_post[i] @ pulse_matrix) @ columns_u.T)
        else:
            continue
        lesser_cross = np.zeros(len(pre_time), dtype=np.complex128)
        for factor, weights_e, pre in zip(factors, source_weights, pre_factors):
            lesser_cross += (
                np.asarray(factor[i]) * np.asarray(weights_e)
            ) @ np.conjugate(pre).T
        lag = observation - pre_time
        # Every point in this integral has t' < t_min, so the projected
        # sector is stationary for the entire semi-infinite tail.  Evaluating
        # it from the finite stored lag array and switching to the stationary
        # formula at lag_max creates one artificial jump per quadrature node
        # as that node crosses lag_max.  Those jumps pollute the collision
        # identity even though the Green functions themselves are smooth.
        pgl = _prehistory_stationary_projected_value(lag, stationary_lesser)
        pgr = _prehistory_stationary_projected_value(lag, stationary_pre_retarded)
        qgl = lesser_cross - pgl
        qgg = lesser_cross + retarded_cross - (pgl + pgr)
        # Fermionic Keldysh symmetry on the reverse-time leg.  For t>t',
        # G^<(t',t) = -conj(G^<(t,t')) and, because G^R(t',t)=0,
        # G^>(t',t) = -conj(G^<(t,t') + G^R(t,t')).
        gg_reverse = -np.conjugate(lesser_cross + retarded_cross)
        gl_reverse = -np.conjugate(lesser_cross)
        tau = observation - pre_time
        lesser_factor = g2 * (
            (n0 + 1.0) * np.exp(1j * omega * tau)
            + n0 * np.exp(-1j * omega * tau)
        )
        greater_factor = g2 * (
            (n0 + 1.0) * np.exp(-1j * omega * tau)
            + n0 * np.exp(1j * omega * tau)
        )
        integrand = lesser_factor * qgl * gg_reverse - greater_factor * qgg * gl_reverse
        tail[i] = 2.0 * np.real(np.sum(pre_weight * integrand))
    return tail


def _prehistory_stationary_projected_value(
    lag_values: np.ndarray, stationary
) -> np.ndarray:
    """Evaluate a projected Green function on the stationary prehistory.

    The discarded-sector history integral only samples the projected sector
    before the finite projection window.  Keeping this operation as a small
    helper makes the no-switch invariant explicit and regression-testable.
    """
    lag_values = np.asarray(lag_values, dtype=float)
    return np.asarray(stationary(lag_values), dtype=np.complex128)


def add_discarded_sector_diagnostics(
    config: CaseConfig,
    trajectory: FixedKernelTrajectory,
    *,
    kernel: StationaryKernel | None = None,
) -> FixedKernelTrajectory:
    """Evaluate QG, discarded SCBA, and collision source including history."""
    gr = trajectory.dense_green_retarded
    gl = trajectory.dense_green_lesser
    if gr is None or gl is None:
        raise ValueError("Discarded-sector diagnostics require dense two-time Green functions.")
    pgr = toeplitz_from_lags(trajectory.projected.green_retarded, lesser=False)
    pgl = toeplitz_from_lags(trajectory.projected.green_lesser, lesser=True)
    qgr = gr - pgr
    qgl = gl - pgl
    ga = np.conjugate(gr.T)
    pga = np.conjugate(pgr.T)
    gg = gl + gr - ga
    pgg = pgl + pgr - pga
    qgg = gg - pgg
    time = trajectory.time
    tau = time[:, None] - time[None, :]
    omega = config.model.phonon_energy
    n0 = (
        float(config.model.phonon_occupation)
        if config.model.phonon_occupation is not None
        else float(1.0 / np.expm1(config.model.beta_ph * omega))
    )
    g2 = config.model.coupling**2
    lesser_factor = g2 * (
        (n0 + 1.0) * np.exp(1j * omega * tau)
        + n0 * np.exp(-1j * omega * tau)
    )
    greater_factor = g2 * (
        (n0 + 1.0) * np.exp(-1j * omega * tau)
        + n0 * np.exp(1j * omega * tau)
    )
    sigma_less_full = lesser_factor * gl
    sigma_greater_full = greater_factor * gg
    delta_less = lesser_factor * qgl
    delta_greater = greater_factor * qgg
    dt = float(time[1] - time[0])
    # Retarded kernels are sampled as values (not already time-integrated
    # weights), so the equal-time Heaviside value is one half.  The local
    # Hartree delta is represented by delta_hartree/dt so that a subsequent
    # trapezoidal convolution has the correct continuum normalization.
    theta = np.tril(np.ones_like(tau, dtype=float))
    theta[np.diag_indices_from(theta)] = 0.5
    sigma_retarded_full = theta * (sigma_greater_full - sigma_less_full)
    delta_retarded = theta * (delta_greater - delta_less)
    occupation = -1j * np.diag(gl)
    projected_occupation = trajectory.projected.occupation
    delta_hartree = -2.0 * g2 * (occupation.real - projected_occupation) / omega
    delta_retarded[np.diag_indices_from(delta_retarded)] += delta_hartree / dt
    sigma_retarded_full[np.diag_indices_from(sigma_retarded_full)] += (
        -2.0 * g2 * occupation.real / (omega * dt)
    )

    green_floor = config.numerics.green_floor
    sigma_floor = config.numerics.sigma_floor
    r_gr = float(np.linalg.norm(qgr) / max(np.linalg.norm(gr), green_floor))
    r_gl = float(np.linalg.norm(qgl) / max(np.linalg.norm(gl), green_floor))
    r_sr = float(
        np.linalg.norm(delta_retarded)
        / max(np.linalg.norm(sigma_retarded_full), sigma_floor)
    )
    r_sl = float(
        np.linalg.norm(delta_less)
        / max(np.linalg.norm(sigma_less_full), sigma_floor)
    )
    if g2 == 0.0:
        r_sr = r_sl = 0.0

    collision_finite = np.zeros(len(time), dtype=float)
    for i in range(len(time)):
        integrand = (
            delta_less[i, : i + 1] * gg[: i + 1, i]
            - delta_greater[i, : i + 1] * gl[: i + 1, i]
        )
        weights = np.full(i + 1, dt)
        if i:
            weights[[0, -1]] *= 0.5
        # Store the source in the same convention as the left-hand side of
        # the manuscript identity: -e C[G, delta Sigma].  The bracket above
        # is C in the current convention, so the omitted-self-energy source
        # carries the opposite sign from the interaction collision integral.
        collision_finite[i] = 2.0 * np.real(np.sum(weights * integrand))

    # The finite matrix starts at t_min, but the pre-pulse state is an
    # infinite stationary nonequilibrium history.  Add its cross-switch
    # contribution analytically/semi-infinite in the same Keldysh convention.
    prehistory_tail = np.zeros(len(time), dtype=float)
    if kernel is not None and g2 != 0.0 and trajectory.lesser_source_factors:
        system = make_system(config)
        cache = build_pole_cache(system, kernel)
        prehistory_tail = _collision_prehistory_tail(
            config,
            system,
            cache,
            trajectory,
        )
    collision = collision_finite + prehistory_tail

    diagnostics = dict(trajectory.diagnostics)
    diagnostics.update(
        {
            "r_green_retarded": r_gr,
            "r_green_lesser": r_gl,
            "r_sigma_retarded": r_sr,
            "r_sigma_lesser": r_sl,
            "discarded_hartree_max": float(np.max(np.abs(delta_hartree))),
            "collision_source_finite_max": float(np.max(np.abs(collision_finite))),
            "collision_source_max": float(np.max(np.abs(collision))),
            "collision_window_min": float(time[0]),
            "collision_window_max": float(time[-1]),
            "collision_prehistory_tail_max": float(np.max(np.abs(prehistory_tail))),
            "collision_prehistory_tail_included": bool(np.any(prehistory_tail)),
            "collision_prehistory_projected_evaluation": "stationary_all_lags",
            "collision_prehistory_quadrature_order": int(
                config.numerics.collision_prehistory_quadrature_order
            ),
            "collision_prehistory_scale": float(config.numerics.collision_prehistory_scale),
        }
    )
    projected = replace(
        trajectory.projected,
        r_green_retarded=r_gr,
        r_green_lesser=r_gl,
        q_green_retarded_time=np.sqrt(np.sum(np.abs(qgr) ** 2, axis=1)),
        q_green_lesser_time=np.sqrt(np.sum(np.abs(qgl) ** 2, axis=1)),
    )
    return replace(
        trajectory,
        projected=projected,
        collision_source=collision,
        collision_source_finite=collision_finite,
        collision_source_prehistory=prehistory_tail,
        diagnostics=diagnostics,
    )
