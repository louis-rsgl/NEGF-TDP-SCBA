"""Finite-duration square-pulse kernels from the v26 residue construction.

The stationary electron--phonon kernel is frozen at the unbiased boundary.
The pulse window uses the biased Green poles and the propagation after the
second switch uses the unbiased Green poles.  No stationary object is rebuilt
at the turnoff time.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from scipy.integrate import quad_vec

from backend.distribution import expc
from backend.minimal_poles import Gbiased_R_mpm, Gfr_R_mpm, PoleCache
from backend.system_classes import Lead, System


@dataclass(frozen=True)
class SquareKernelCache:
    duration: float
    energies: np.ndarray
    S_alpha: Mapping[Lead, np.ndarray]
    S_C: Mapping[float, np.ndarray]
    residue_max_abs_error: float
    residue_max_scaled_error: float
    residue_candidate_count: int
    residue_cluster_count: int
    residue_cancelled_count: int
    turnoff_A_abs_error: float
    turnoff_A_scaled_error: float
    turnoff_C_abs_error: float
    turnoff_C_scaled_error: float


@dataclass
class _ResidueStats:
    max_abs_error: float = 0.0
    max_scaled_error: float = 0.0
    candidates: int = 0
    clusters: int = 0
    cancelled: int = 0


def _readonly(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128).copy()
    result.setflags(write=False)
    return result


def _analytic_arguments(values) -> np.ndarray:
    """Evaluate rational residue formulas on the retarded boundary.

    ``Gfr_R_mpm`` uses ``System.ETA`` for real-grid reconstruction tests.  The
    arrowhead poles, however, represent the exact rational eta->0 function.
    Square-pulse residue amplitudes must use that same analytic representation
    or the turnoff identity mixes two different regulators.
    """
    result = np.asarray(values, dtype=np.complex128)
    return np.where(np.abs(result.imag) <= 1e-15, result + 1e-14j, result)


def _i0_scale(sys: System) -> float:
    return 1e-11 * max(1.0, abs(sys.W), abs(sys.DELTA))


def _cluster_candidates(candidates, tolerance: float) -> list[np.ndarray]:
    pending = [complex(value) for value in candidates if np.isfinite(value)]
    clusters: list[list[complex]] = []
    while pending:
        cluster = [pending.pop()]
        changed = True
        while changed:
            changed = False
            for value in pending[:]:
                if min(abs(value - member) for member in cluster) <= tolerance:
                    cluster.append(value)
                    pending.remove(value)
                    changed = True
        clusters.append(cluster)
    return [np.asarray(cluster, dtype=np.complex128) for cluster in clusters]


def _contour_residue(
    function: Callable[[np.ndarray], np.ndarray],
    center: complex,
    radius: float,
    n_theta: int,
) -> complex:
    angles = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    unit = np.exp(1j * angles)
    points = center + radius * unit
    values = np.asarray(function(points), dtype=np.complex128)
    return complex(np.mean(values * radius * unit))


def lower_half_plane_residue(
    function: Callable[[np.ndarray], np.ndarray],
    candidates,
    *,
    merge_tolerance: float = 1e-7,
    n_theta: int = 64,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 1e-6,
    stats: _ResidueStats | None = None,
) -> complex:
    """Return ``-sum Res F`` using the complete assembled integrand.

    Candidate poles are deliberately not pre-cancelled.  Integrating a small
    circle around a removable candidate returns zero and therefore tests the
    cancellation in the same expression used by the calculation.
    """
    values = np.asarray(candidates, dtype=np.complex128).reshape(-1)
    values = values[np.isfinite(values) & (values.imag < 0.0)]
    if values.size == 0:
        return 0.0j
    scale = max(1.0, float(np.max(np.abs(values))))
    clusters = _cluster_candidates(values, merge_tolerance * scale)
    centers = np.asarray([np.mean(group) for group in clusters])
    total = 0.0j
    for index, group in enumerate(clusters):
        center = complex(centers[index])
        spread = float(np.max(np.abs(group - center))) if len(group) else 0.0
        if len(clusters) > 1:
            separation = float(
                np.min(np.abs(center - np.delete(centers, index)))
            )
        else:
            separation = np.inf
        base = 2e-5 * max(1.0, abs(center))
        radius = max(base, 4.0 * spread)
        if np.isfinite(separation):
            radius = min(radius, 0.2 * separation)
        if radius <= 1.05 * spread:
            raise RuntimeError("Square residue candidates could not be isolated.")
        high = _contour_residue(function, center, radius, n_theta)
        low = _contour_residue(function, center, radius, max(16, n_theta // 2))
        shrunk = _contour_residue(function, center, 0.7 * radius, n_theta)
        error = max(abs(high - low), abs(high - shrunk))
        scaled = error / (abs_tolerance + rel_tolerance * abs(high))
        if stats is not None:
            stats.candidates += len(group)
            stats.clusters += 1
            stats.max_abs_error = max(stats.max_abs_error, float(error))
            stats.max_scaled_error = max(stats.max_scaled_error, float(scaled))
            if abs(high) <= abs_tolerance:
                stats.cancelled += 1
        if scaled > 1.0:
            raise RuntimeError(
                "Square internal residue failed contour convergence: "
                f"center={center!r}, abs/scaled={error:.3e}/{scaled:.3e}."
            )
        total -= high
    return total


def chi_square(z, energy, delta: float, duration: float, sign: int, i0: float):
    z = np.asarray(z, dtype=np.complex128)
    a = z - energy + sign * delta
    b = z - energy
    return delta * np.expm1(1j * a * duration) / (
        (a - 1j * i0) * (b - 1j * i0)
    )


def _Q_alpha(sys, cache, z, zp, energy, duration, alpha):
    da = sys.Delta(alpha)
    i0 = _i0_scale(sys)
    finite_difference = 0.0j
    for mu in sys.lead_names:
        finite_difference += (
            sys.Delta(mu) * sys.Gamma0(mu) * sys.W
            / (2.0 * (energy + 1j * sys.W) * (zp + da - sys.Delta(mu) + 1j * sys.W))
        )
    bracket = (
        da / (zp - energy + 1j * i0)
        + (sys.DELTA + finite_difference) * Gfr_R_mpm(sys, cache, energy)
    )
    return (
        np.exp(1j * (z - zp - da) * duration)
        * Gbiased_R_mpm(sys, cache, zp + da)
        * bracket
        / ((zp - energy + da + 1j * i0) * (z - zp - da - 1j * i0))
    )


def _Q_C(sys, cache, z, zp, energy, duration):
    i0 = _i0_scale(sys)
    finite_difference = 0.0j
    for mu in sys.lead_names:
        finite_difference += (
            sys.Delta(mu) * sys.Gamma0(mu) * sys.W
            / (2.0 * (energy + 1j * sys.W) * (zp - sys.Delta(mu) + 1j * sys.W))
        )
    return (
        np.exp(1j * (z - zp) * duration)
        * Gbiased_R_mpm(sys, cache, zp)
        * (sys.DELTA + finite_difference)
        * Gfr_R_mpm(sys, cache, energy)
        / ((zp - energy + 1j * i0) * (z - zp - 1j * i0))
    )


def _residue_kwargs(sys, stats):
    return dict(
        merge_tolerance=sys.pole_merge_tol,
        n_theta=sys.square_residue_n_theta,
        abs_tolerance=sys.square_residue_abs_tol,
        rel_tolerance=sys.square_residue_rel_tol,
        stats=stats,
    )


def I_alpha_square(sys, cache, z, energy, duration, alpha, stats=None):
    i0 = _i0_scale(sys)
    da = sys.Delta(alpha)

    def integrand(zp):
        zp = np.asarray(zp, dtype=np.complex128)
        first = np.zeros_like(zp)
        coefficient = np.full_like(zp, complex(sys.DELTA))
        for beta in sys.lead_names:
            db = sys.Delta(beta)
            first += (
                db
                * chi_square(z, zp, db, duration, -1, i0)
                / ((zp - energy + db + 1j * i0) * (zp - energy + 1j * i0))
                * sys.Gamma0(beta) * sys.W / (2.0 * (zp + 1j * sys.W))
                * Gfr_R_mpm(sys, cache, energy)
            )
            coefficient += (
                db * sys.Gamma0(beta) * sys.W
                / (2.0 * (z + 1j * sys.W) * (zp + da - db + 1j * sys.W))
            )
        return first - coefficient * _Q_alpha(
            sys, cache, z, zp, energy, duration, alpha
        )

    candidates = []
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        candidates.extend((
            energy - db - 1j * i0,
            energy - 1j * i0,
            z - db - 1j * i0,
            z - 1j * i0,
            -1j * sys.W,
            -da + db - 1j * sys.W,
        ))
    candidates.extend(cache.xi_biased - da)
    candidates.extend((energy - da - 1j * i0, z - da - 1j * i0))
    return lower_half_plane_residue(integrand, candidates, **_residue_kwargs(sys, stats))


def I_C_square(sys, cache, z, energy, duration, stats=None):
    i0 = _i0_scale(sys)

    def integrand(zp):
        zp = np.asarray(zp, dtype=np.complex128)
        first = np.zeros_like(zp)
        coefficient = np.full_like(zp, complex(sys.DELTA))
        for beta in sys.lead_names:
            db = sys.Delta(beta)
            first += (
                db
                * chi_square(z, zp, db, duration, -1, i0)
                / ((zp - energy + db + 1j * i0) * (zp - energy + 1j * i0))
                * sys.Gamma0(beta) * sys.W / (2.0 * (zp + 1j * sys.W))
                * Gfr_R_mpm(sys, cache, energy)
            )
            coefficient += (
                db * sys.Gamma0(beta) * sys.W
                / (2.0 * (z + 1j * sys.W) * (zp - db + 1j * sys.W))
            )
        return first - coefficient * _Q_C(sys, cache, z, zp, energy, duration)

    candidates = []
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        candidates.extend((
            energy - db - 1j * i0,
            energy - 1j * i0,
            z - db - 1j * i0,
            z - 1j * i0,
            -1j * sys.W,
            db - 1j * sys.W,
        ))
    candidates.extend(cache.xi_biased)
    return lower_half_plane_residue(integrand, candidates, **_residue_kwargs(sys, stats))


def S_alpha_square(sys, cache, z, energy, duration, alpha, stats=None):
    if duration == 0.0:
        return 0.0j
    i0 = _i0_scale(sys)
    value = chi_square(z, energy, sys.Delta(alpha), duration, -1, i0)
    bracket = sys.DELTA * np.expm1(1j * (z - energy) * duration) / (
        z - energy - 1j * i0
    )
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        bracket += sys.Gamma0(beta) * sys.W / 2.0 * (
            chi_square(z, energy, db, duration, -1, i0) / (energy + 1j * sys.W)
            - np.exp(-1j * db * duration)
            * chi_square(z, energy, db, duration, +1, i0)
            / (z + 1j * sys.W)
        )
    return (
        value
        + bracket * Gfr_R_mpm(sys, cache, energy)
        + I_alpha_square(sys, cache, z, energy, duration, alpha, stats)
    )


def S_C_square(sys, cache, z, energy, duration, stats=None):
    if duration == 0.0:
        return 0.0j
    i0 = _i0_scale(sys)
    bracket = sys.DELTA * np.expm1(1j * (z - energy) * duration) / (
        z - energy - 1j * i0
    )
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        bracket += sys.Gamma0(beta) * sys.W / 2.0 * (
            chi_square(z, energy, db, duration, -1, i0) / (energy + 1j * sys.W)
            - np.exp(-1j * db * duration)
            * chi_square(z, energy, db, duration, +1, i0)
            / (z + 1j * sys.W)
        )
    return (
        bracket * Gfr_R_mpm(sys, cache, energy)
        + I_C_square(sys, cache, z, energy, duration, stats)
    )


def _evaluate_S_alpha(sys, cache, energies, duration, alpha, stats=None):
    energies = _analytic_arguments(energies)
    return np.asarray([
        [S_alpha_square(sys, cache, pole, energy, duration, alpha, stats)
         for energy in energies]
        for pole in cache.xi_unbiased
    ], dtype=np.complex128)


def _evaluate_S_C(sys, cache, shifted_energies, duration, stats=None):
    shifted_energies = _analytic_arguments(shifted_energies)
    return np.asarray([
        [S_C_square(sys, cache, pole, energy, duration, stats)
         for energy in shifted_energies]
        for pole in cache.xi_unbiased
    ], dtype=np.complex128)


def _post_A(sys, cache, energies, time, duration, alpha, S_values):
    energies = _analytic_arguments(energies).reshape(-1)
    poles = cache.xi_unbiased[:, None]
    residues = cache.residues_unbiased[:, None]
    return Gfr_R_mpm(sys, cache, energies) - np.exp(
        1j * sys.Delta(alpha) * duration
    ) * np.sum(
        residues * np.exp(-1j * (poles - energies[None, :]) * time) * S_values,
        axis=0,
    )


def _post_C(sys, cache, shifted, time, S_values):
    shifted = _analytic_arguments(shifted).reshape(-1)
    poles = cache.xi_unbiased[:, None]
    residues = cache.residues_unbiased[:, None]
    return Gfr_R_mpm(sys, cache, shifted) - np.sum(
        residues * np.exp(-1j * (poles - shifted[None, :]) * time) * S_values,
        axis=0,
    )


def build_square_kernel_cache(sys, energies, duration=None, cache=None):
    from backend.observables import _A_up_residue, _C_up_residue

    sys.validate_pulse_protocol()
    if sys.pulse_protocol != "square":
        raise RuntimeError("Square kernels require a System with pulse_protocol='square'.")
    duration = sys.pulse_duration if duration is None else float(duration)
    if duration is None or not np.isfinite(duration) or duration < 0.0:
        raise ValueError("A finite nonnegative square-pulse duration is required.")
    energies = np.asarray(energies, dtype=float).reshape(-1)
    cache = sys.prepare_poles() if cache is None else cache
    existing = sys._square_kernel_cache
    if (
        isinstance(existing, SquareKernelCache)
        and existing.duration == duration
        and np.array_equal(existing.energies, energies)
    ):
        return existing
    stats = _ResidueStats()
    S_alpha = {
        lead: _readonly(_evaluate_S_alpha(sys, cache, energies, duration, lead, stats))
        for lead in sys.lead_names
    }
    S_C = {}
    for omega in sorted({-float(sys.w_q), float(sys.w_q)}):
        S_C[omega] = _readonly(
            _evaluate_S_C(sys, cache, energies + omega, duration, stats)
        )

    analytic = energies.astype(np.complex128) + 1e-14j
    max_a_abs = max_a_scaled = 0.0
    for lead in sys.lead_names:
        raw = _post_A(sys, cache, analytic, duration, duration, lead, S_alpha[lead])
        expected = np.asarray(
            _A_up_residue(sys, analytic, duration, lead, cache, enforce_boundary=False)
        ).reshape(-1)
        difference = np.abs(raw - expected)
        band = sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(expected)
        max_a_abs = max(max_a_abs, float(np.max(difference)))
        max_a_scaled = max(max_a_scaled, float(np.max(difference / band)))
    max_c_abs = max_c_scaled = 0.0
    for omega, values in S_C.items():
        raw = _post_C(sys, cache, analytic + omega, duration, values)
        expected = np.asarray(
            _C_up_residue(sys, omega, analytic, duration, cache, enforce_boundary=False)
        ).reshape(-1)
        difference = np.abs(raw - expected)
        band = sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(expected)
        max_c_abs = max(max_c_abs, float(np.max(difference)))
        max_c_scaled = max(max_c_scaled, float(np.max(difference / band)))
    if max_a_scaled > 1.0 or max_c_scaled > 1.0:
        raise RuntimeError(
            "Square turnoff residue validation failed: "
            f"A abs/scaled={max_a_abs:.3e}/{max_a_scaled:.3e}, "
            f"C abs/scaled={max_c_abs:.3e}/{max_c_scaled:.3e}."
        )
    result = SquareKernelCache(
        duration=float(duration),
        energies=_readonly(energies),
        S_alpha=MappingProxyType(S_alpha),
        S_C=MappingProxyType(S_C),
        residue_max_abs_error=stats.max_abs_error,
        residue_max_scaled_error=stats.max_scaled_error,
        residue_candidate_count=stats.candidates,
        residue_cluster_count=stats.clusters,
        residue_cancelled_count=stats.cancelled,
        turnoff_A_abs_error=max_a_abs,
        turnoff_A_scaled_error=max_a_scaled,
        turnoff_C_abs_error=max_c_abs,
        turnoff_C_scaled_error=max_c_scaled,
    )
    sys._square_kernel_cache = result
    return result


def A_square(sys, energy, t, alpha, cache=None, square_cache=None, duration=None):
    from backend.observables import A_up

    cache = sys.prepare_poles() if cache is None else cache
    duration = sys.pulse_duration if duration is None else float(duration)
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    output = np.empty((len(energies), len(times)), dtype=np.complex128)
    before = times <= duration
    if np.any(before):
        output[:, before] = np.asarray(
            A_up(sys, energies, times[before], alpha, cache)
        ).reshape(len(energies), np.count_nonzero(before))
    if np.any(~before):
        if square_cache is not None and np.array_equal(energies.real, square_cache.energies):
            S_values = square_cache.S_alpha[alpha]
        else:
            S_values = _evaluate_S_alpha(sys, cache, energies, duration, alpha)
        for index in np.flatnonzero(~before):
            output[:, index] = _post_A(
                sys, cache, energies, times[index], duration, alpha, S_values
            )
    return output.reshape(-1)[0] if np.asarray(energy).ndim == 0 and np.asarray(t).ndim == 0 else np.squeeze(output)


def C_square(sys, omega, energy_prime, t, cache=None, square_cache=None, duration=None):
    from backend.observables import C_up

    cache = sys.prepare_poles() if cache is None else cache
    duration = sys.pulse_duration if duration is None else float(duration)
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    output = np.empty((len(primes), len(times)), dtype=np.complex128)
    before = times <= duration
    if np.any(before):
        output[:, before] = np.asarray(
            C_up(sys, omega, primes, times[before], cache)
        ).reshape(len(primes), np.count_nonzero(before))
    if np.any(~before):
        key = float(omega)
        if square_cache is not None and key in square_cache.S_C and np.array_equal(primes.real, square_cache.energies):
            S_values = square_cache.S_C[key]
        else:
            S_values = _evaluate_S_C(sys, cache, primes + omega, duration)
        for index in np.flatnonzero(~before):
            output[:, index] = _post_C(
                sys, cache, primes + omega, times[index], S_values
            )
    return output.reshape(-1)[0] if np.asarray(energy_prime).ndim == 0 and np.asarray(t).ndim == 0 else np.squeeze(output)


def B_square(sys, energy, energy_prime, t, beta, cache=None, square_cache=None, duration=None):
    """Post-turnoff ``B_beta^>`` history (zero at and before turnoff)."""
    cache = sys.prepare_poles() if cache is None else cache
    duration = sys.pulse_duration if duration is None else float(duration)
    outer = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    inner = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    time = float(t)
    if time <= duration:
        result = np.zeros((len(outer), len(inner)), dtype=np.complex128)
    else:
        if square_cache is not None and np.array_equal(inner.real, square_cache.energies):
            S_values = square_cache.S_alpha[beta]
        else:
            S_values = _evaluate_S_alpha(sys, cache, inner, duration, beta)
        x = cache.xi_unbiased[None, None, :]
        r = cache.residues_unbiased[None, None, :]
        e = outer[:, None, None]
        ep = inner[None, :, None]
        pole_sum = np.sum(
            r * np.exp(1j * (e - x) * duration)
            * expc(e - x, time - duration)
            * S_values.T[None, :, :], axis=2
        )
        result = (
            np.exp(1j * (outer[:, None] - inner[None, :]) * duration)
            * expc(outer[:, None] - inner[None, :], time - duration)
            * Gfr_R_mpm(sys, cache, _analytic_arguments(inner))[None, :]
            - np.exp(1j * sys.Delta(beta) * duration) * pole_sum
        )
    return result.reshape(-1)[0] if np.asarray(energy).ndim == 0 and np.asarray(energy_prime).ndim == 0 else np.squeeze(result)


def D_square(sys, omega, energy_prime, energy, t, alpha=None, cache=None, square_cache=None, duration=None):
    """Post-turnoff ``D_alpha^>`` history (independent of alpha)."""
    cache = sys.prepare_poles() if cache is None else cache
    duration = sys.pulse_duration if duration is None else float(duration)
    outer = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    inner = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    shifted = inner + omega
    time = float(t)
    if time <= duration:
        result = np.zeros((len(outer), len(inner)), dtype=np.complex128)
    else:
        key = float(omega)
        if square_cache is not None and key in square_cache.S_C and np.array_equal(inner.real, square_cache.energies):
            S_values = square_cache.S_C[key]
        else:
            S_values = _evaluate_S_C(sys, cache, shifted, duration)
        x = cache.xi_unbiased[None, None, :]
        r = cache.residues_unbiased[None, None, :]
        e = outer[:, None, None]
        E = shifted[None, :, None]
        pole_sum = np.sum(
            r * np.exp(1j * (e - x) * duration)
            * expc(e - x, time - duration)
            * S_values.T[None, :, :], axis=2
        )
        result = (
            np.exp(1j * (outer[:, None] - shifted[None, :]) * duration)
            * expc(outer[:, None] - shifted[None, :], time - duration)
            * Gfr_R_mpm(sys, cache, _analytic_arguments(shifted))[None, :]
            - pole_sum
        )
    return result.reshape(-1)[0] if np.asarray(energy).ndim == 0 and np.asarray(energy_prime).ndim == 0 else np.squeeze(result)


def A_square_direct(sys, energy, t, alpha, cache=None, duration=None, bounds=None):
    cache = sys.prepare_poles() if cache is None else cache
    duration = sys.pulse_duration if duration is None else float(duration)
    lower, upper = bounds or (sys.e_min, sys.e_max)
    integral, _ = quad_vec(
        lambda z: np.exp(-1j * (z - energy) * t)
        * Gfr_R_mpm(sys, cache, z)
        * S_alpha_square(sys, cache, z, energy, duration, alpha)
        / (2j * np.pi),
        lower, upper, epsabs=1e-8, epsrel=1e-7, limit=1000,
    )
    return Gfr_R_mpm(sys, cache, energy) + np.exp(
        1j * sys.Delta(alpha) * duration
    ) * integral


def C_square_direct(sys, omega, energy_prime, t, cache=None, duration=None, bounds=None):
    cache = sys.prepare_poles() if cache is None else cache
    duration = sys.pulse_duration if duration is None else float(duration)
    shifted = energy_prime + omega
    lower, upper = bounds or (sys.e_min, sys.e_max)
    integral, _ = quad_vec(
        lambda z: np.exp(-1j * (z - shifted) * t)
        * Gfr_R_mpm(sys, cache, z)
        * S_C_square(sys, cache, z, shifted, duration)
        / (2j * np.pi),
        lower, upper, epsabs=1e-8, epsrel=1e-7, limit=1000,
    )
    return Gfr_R_mpm(sys, cache, shifted) + integral
