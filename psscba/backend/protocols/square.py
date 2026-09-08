"""Finite-duration square-pulse kernels from the v26 residue construction.

The stationary electron--phonon kernel is installed at the unbiased boundary.
The pulse window uses the biased Green poles and the propagation after the
second switch uses the unbiased Green poles.  No stationary object is rebuilt
at the turnoff time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from scipy.integrate import quad_vec

from psscba.backend.model.distributions import expc
from psscba.backend.poles.minimal import (
    Gbiased_R_mpm,
    Gfr_R_mpm,
    PoleCache,
    sigma_mpm,
)
from psscba.backend.model.system import Lead, System


@dataclass(frozen=True)
class SquareKernelCache:
    """Square-pulse pole amplitudes in an overflow-safe representation.

    The rows of ``S_alpha`` and ``S_C`` correspond to the unbiased poles
    ``z`` and store ``exp(-1j*z*duration) * S(z, energy)``. Keeping this
    propagation factor with the amplitude prevents the individually huge
    ``S`` values of deep wide-band poles from overflowing for long pulses.
    """

    duration: float
    energies: np.ndarray
    S_alpha: Mapping[Lead, np.ndarray]
    S_C: Mapping[float, np.ndarray]
    residue_max_abs_error: float
    residue_max_scaled_error: float
    residue_candidate_count: int
    residue_cluster_count: int
    residue_cancelled_count: int
    residue_negligible_count: int
    turnoff_A_abs_error: float
    turnoff_A_scaled_error: float
    turnoff_C_abs_error: float
    turnoff_C_scaled_error: float
    residue_method: str = "scalar_contour"
    cache_key: str = ""


class SquareKernelCacheStore:
    """Per-case store for immutable square caches on multiple energy grids."""

    def __init__(self) -> None:
        self._entries: dict[tuple, SquareKernelCache] = {}
        self._kernel_fingerprint: str | None = None

    @staticmethod
    def _digest(values) -> str:
        array = np.ascontiguousarray(np.asarray(values))
        return hashlib.sha256(array.view(np.uint8)).hexdigest()

    def key(self, sys, cache: PoleCache, energies, duration: float) -> tuple:
        parts = [cache.zeta, cache.weights, cache.xi_unbiased, cache.xi_biased,
                 np.asarray([cache.sigma_H], dtype=np.complex128)]
        kernel_fingerprint = self._digest(np.concatenate([
            np.asarray(part, dtype=np.complex128).view(np.float64)
            for part in parts
        ]))
        if self._kernel_fingerprint != kernel_fingerprint:
            self._entries.clear()
            self._kernel_fingerprint = kernel_fingerprint
        energy_digest = self._digest(np.asarray(energies, dtype=np.float64))
        return (
            kernel_fingerprint,
            energy_digest,
            float(duration),
            int(getattr(sys, "square_residue_n_theta", 64)),
            float(getattr(sys, "square_residue_abs_tol", 1e-8)),
            float(getattr(sys, "square_residue_rel_tol", 1e-6)),
            float(getattr(sys, "square_residue_cancellation_rel_tol", 1e-12)),
            (-float(sys.w_q), 0.0, float(sys.w_q)),
        )

    def get(self, key: tuple) -> SquareKernelCache | None:
        return self._entries.get(key)

    def put(self, key: tuple, value: SquareKernelCache) -> SquareKernelCache:
        self._entries[key] = value
        return value

    def get_or_build(self, sys, cache: PoleCache, energies, duration: float,
                     builder, **kwargs) -> SquareKernelCache:
        key = self.key(sys, cache, energies, duration)
        existing = self.get(key)
        if existing is not None:
            return existing
        value = builder(sys, energies, duration=duration, cache=cache,
                        store=self, **kwargs)
        return value

    @property
    def size(self) -> int:
        return len(self._entries)


@dataclass
class _ResidueStats:
    max_abs_error: float = 0.0
    max_scaled_error: float = 0.0
    candidates: int = 0
    clusters: int = 0
    cancelled: int = 0
    negligible: int = 0


def _readonly(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128).copy()
    result.setflags(write=False)
    return result


def _analytic_arguments(values) -> np.ndarray:
    """Evaluate rational residue formulas on the retarded boundary.

    The arrowhead poles and all transient propagators represent the exact
    rational eta->0+ function.  Keep square-pulse residue amplitudes on that
    same analytic boundary so the turnoff identity never mixes regulators.
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


def _contour_residue_and_scale(
    function: Callable[[np.ndarray], np.ndarray],
    center: complex,
    radius: float,
    n_theta: int,
) -> tuple[complex, float]:
    angles = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
    unit = np.exp(1j * angles)
    points = center + radius * unit
    terms = np.asarray(function(points), dtype=np.complex128) * radius * unit
    return complex(np.mean(terms)), float(np.max(np.abs(terms)))


def lower_half_plane_residue(
    function: Callable[[np.ndarray], np.ndarray],
    candidates,
    *,
    merge_tolerance: float = 1e-7,
    n_theta: int = 64,
    abs_tolerance: float = 1e-8,
    rel_tolerance: float = 1e-6,
    cancellation_rel_tolerance: float = 1e-12,
    contribution_weight: float = 1.0,
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
        high, high_scale = _contour_residue_and_scale(
            function, center, radius, n_theta
        )
        low, low_scale = _contour_residue_and_scale(
            function, center, radius, max(16, n_theta // 2)
        )
        shrunk, shrunk_scale = _contour_residue_and_scale(
            function, center, 0.7 * radius, n_theta
        )
        error = max(abs(high - low), abs(high - shrunk))
        weighted_error = contribution_weight * error
        weighted_value = contribution_weight * abs(high)
        scaled = weighted_error / (
            abs_tolerance + rel_tolerance * weighted_value
        )
        contour_scale = max(high_scale, low_scale, shrunk_scale)
        weighted_magnitude = contribution_weight * max(
            abs(high), abs(low), abs(shrunk), error
        )
        removable = weighted_magnitude <= (
            cancellation_rel_tolerance * contribution_weight * contour_scale
        )
        negligible = not removable and weighted_magnitude <= abs_tolerance
        if stats is not None:
            stats.candidates += len(group)
            stats.clusters += 1
            stats.max_abs_error = max(stats.max_abs_error, float(weighted_error))
            stats.max_scaled_error = max(stats.max_scaled_error, float(scaled))
            if removable:
                stats.cancelled += 1
            elif negligible:
                stats.negligible += 1
        if removable or negligible:
            continue
        if scaled > 1.0:
            raise RuntimeError(
                "Square internal residue failed contour convergence: "
                f"center={center!r}, abs/scaled={error:.3e}/{scaled:.3e}."
            )
        total -= high
    return total


def lower_half_plane_residue_batched(
    functions,
    candidates,
    *,
    block_size: int = 64,
    **kwargs,
) -> np.ndarray:
    """Evaluate a family of contour residues with bounded batching.

    The scalar ``lower_half_plane_residue`` remains the reference evaluator;
    this adapter provides one stable production seam for vectorized contour
    kernels and guarantees deterministic ordering for non-divisible blocks.
    """
    functions = list(functions)
    candidates = list(candidates)
    if len(functions) != len(candidates):
        raise ValueError("functions and candidates must have equal length")
    output = np.empty(len(functions), dtype=np.complex128)
    for start in range(0, len(functions), max(1, int(block_size))):
        stop = min(start + max(1, int(block_size)), len(functions))
        output[start:stop] = [
            lower_half_plane_residue(functions[i], candidates[i], **kwargs)
            for i in range(start, stop)
        ]
    return output


def chi_square(z, energy, delta: float, duration: float, sign: int, i0: float):
    z = np.asarray(z, dtype=np.complex128)
    a = z - energy + sign * delta
    b = z - energy
    return delta * np.expm1(1j * a * duration) / (
        (a - 1j * i0) * (b - 1j * i0)
    )


def _scaled_outer_chi_square(
    z, energy, delta: float, duration: float, sign: int, i0: float
):
    """Return ``exp(-1j*z*duration) * chi_square(z, ...)`` stably."""
    z = np.asarray(z, dtype=np.complex128)
    a = z - energy + sign * delta
    b = z - energy
    numerator = (
        np.exp(-1j * (energy - sign * delta) * duration)
        - np.exp(-1j * z * duration)
    )
    return delta * numerator / ((a - 1j * i0) * (b - 1j * i0))


def _biased_green_times_lead_rational(
    sys: System,
    cache: PoleCache,
    y,
    constant,
    coefficients: dict[float, complex | np.ndarray],
):
    """Evaluate ``Gbiased(y) * (constant + sum c_d/(y-d+iW))``.

    Written this way the expression loses many digits close to a Lorentzian
    embedding pole: the Dyson inverse diverges while the Green function tends
    to zero.  Multiplying numerator and denominator by the product of distinct
    lead denominators makes the v26 pole--zero cancellation explicit without
    removing the candidate from the residue test.
    """
    y = np.asarray(y, dtype=np.complex128)
    grouped_numerators: dict[float, float] = {}
    for lead in sys.lead_names:
        shift = float(sys.Delta(lead))
        grouped_numerators[shift] = grouped_numerators.get(shift, 0.0) + (
            0.5 * sys.Gamma0(lead) * sys.W
        )
    shifts = sorted(grouped_numerators)
    denominators = [y - shift + 1j * sys.W for shift in shifts]
    product = np.ones_like(y)
    for denominator in denominators:
        product *= denominator

    numerator = np.asarray(constant, dtype=np.complex128) * product
    dyson = (
        y - sys.e_0 - sys.DELTA - cache.sigma_H
        - sigma_mpm(y, cache.zeta, cache.weights)
    ) * product
    for index, shift in enumerate(shifts):
        product_without = np.ones_like(y)
        for other_index, denominator in enumerate(denominators):
            if other_index != index:
                product_without *= denominator
        dyson -= grouped_numerators[shift] * product_without
        if shift in coefficients:
            numerator += np.asarray(
                coefficients[shift], dtype=np.complex128
            ) * product_without
    return numerator / dyson


def _Q_alpha(sys, cache, z, zp, energy, duration, alpha):
    da = sys.Delta(alpha)
    i0 = _i0_scale(sys)
    coefficients: dict[float, complex | np.ndarray] = {}
    gfr = Gfr_R_mpm(sys, cache, energy)
    for mu in sys.lead_names:
        shift = float(sys.Delta(mu))
        coefficients[shift] = coefficients.get(shift, 0.0j) + (
            sys.Delta(mu) * sys.Gamma0(mu) * sys.W * gfr
            / (2.0 * (energy + 1j * sys.W))
        )
    constant = (
        da / (zp - energy + 1j * i0)
        + sys.DELTA * gfr
    )
    dressed_bracket = _biased_green_times_lead_rational(
        sys, cache, zp + da, constant, coefficients
    )
    return (
        np.exp(1j * (z - zp - da) * duration)
        * dressed_bracket
        / ((zp - energy + da + 1j * i0) * (z - zp - da - 1j * i0))
    )


def _Q_C(sys, cache, z, zp, energy, duration):
    i0 = _i0_scale(sys)
    coefficients: dict[float, complex | np.ndarray] = {}
    gfr = Gfr_R_mpm(sys, cache, energy)
    for mu in sys.lead_names:
        shift = float(sys.Delta(mu))
        coefficients[shift] = coefficients.get(shift, 0.0j) + (
            sys.Delta(mu) * sys.Gamma0(mu) * sys.W * gfr
            / (2.0 * (energy + 1j * sys.W))
        )
    dressed_bracket = _biased_green_times_lead_rational(
        sys, cache, zp, sys.DELTA * gfr, coefficients
    )
    return (
        np.exp(1j * (z - zp) * duration)
        * dressed_bracket
        / ((zp - energy + 1j * i0) * (z - zp - 1j * i0))
    )


def _scaled_outer_Q_alpha(sys, cache, z, zp, energy, duration, alpha):
    """Return ``exp(-1j*z*duration) * _Q_alpha(...)`` stably."""
    da = sys.Delta(alpha)
    i0 = _i0_scale(sys)
    coefficients: dict[float, complex | np.ndarray] = {}
    gfr = Gfr_R_mpm(sys, cache, energy)
    for mu in sys.lead_names:
        shift = float(sys.Delta(mu))
        coefficients[shift] = coefficients.get(shift, 0.0j) + (
            sys.Delta(mu) * sys.Gamma0(mu) * sys.W * gfr
            / (2.0 * (energy + 1j * sys.W))
        )
    constant = da / (zp - energy + 1j * i0) + sys.DELTA * gfr
    dressed_bracket = _biased_green_times_lead_rational(
        sys, cache, zp + da, constant, coefficients
    )
    return (
        np.exp(-1j * (zp + da) * duration)
        * dressed_bracket
        / ((zp - energy + da + 1j * i0) * (z - zp - da - 1j * i0))
    )


def _scaled_outer_Q_C(sys, cache, z, zp, energy, duration):
    """Return ``exp(-1j*z*duration) * _Q_C(...)`` stably."""
    i0 = _i0_scale(sys)
    coefficients: dict[float, complex | np.ndarray] = {}
    gfr = Gfr_R_mpm(sys, cache, energy)
    for mu in sys.lead_names:
        shift = float(sys.Delta(mu))
        coefficients[shift] = coefficients.get(shift, 0.0j) + (
            sys.Delta(mu) * sys.Gamma0(mu) * sys.W * gfr
            / (2.0 * (energy + 1j * sys.W))
        )
    dressed_bracket = _biased_green_times_lead_rational(
        sys, cache, zp, sys.DELTA * gfr, coefficients
    )
    return (
        np.exp(-1j * zp * duration)
        * dressed_bracket
        / ((zp - energy + 1j * i0) * (z - zp - 1j * i0))
    )


def _outer_contribution_weight(
    sys, cache, z, duration: float, *, scaled: bool = False
) -> float:
    z = complex(z)
    if len(cache.xi_unbiased) == 0:
        return 1.0
    index = int(np.argmin(np.abs(cache.xi_unbiased - z)))
    distance = abs(cache.xi_unbiased[index] - z)
    tolerance = sys.pole_merge_tol * max(1.0, abs(z))
    if distance > tolerance:
        # Direct real-axis diagnostics evaluate S(z) away from outer poles.
        return 1.0
    propagation = 1.0 if scaled else np.exp(z.imag * duration)
    return float(abs(cache.residues_unbiased[index]) * propagation)


def _residue_kwargs(sys, cache, z, duration, stats, *, scaled=False):
    return dict(
        merge_tolerance=sys.pole_merge_tol,
        n_theta=sys.square_residue_n_theta,
        abs_tolerance=sys.square_residue_abs_tol,
        rel_tolerance=sys.square_residue_rel_tol,
        cancellation_rel_tolerance=sys.square_residue_cancellation_rel_tol,
        contribution_weight=_outer_contribution_weight(
            sys, cache, z, duration, scaled=scaled
        ),
        stats=stats,
    )


def I_alpha_square(sys, cache, z, energy, duration, alpha, stats=None,
                   residue_method="batched_contour"):
    i0 = _i0_scale(sys)
    da = sys.Delta(alpha)

    def integrand(zp):
        zp = np.asarray(zp, dtype=np.complex128)
        first = np.zeros_like(zp)
        coefficient = np.full_like(zp, complex(sys.DELTA))
        for beta in sys.lead_names:
            db = sys.Delta(beta)
            if db == 0.0:
                continue
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
    return _dispatch_residue(
        integrand, candidates,
        _residue_kwargs(sys, cache, z, duration, stats), residue_method,
    )


def I_C_square(sys, cache, z, energy, duration, stats=None,
               residue_method="batched_contour"):
    i0 = _i0_scale(sys)

    def integrand(zp):
        zp = np.asarray(zp, dtype=np.complex128)
        first = np.zeros_like(zp)
        coefficient = np.full_like(zp, complex(sys.DELTA))
        for beta in sys.lead_names:
            db = sys.Delta(beta)
            if db == 0.0:
                continue
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
    return _dispatch_residue(
        integrand, candidates,
        _residue_kwargs(sys, cache, z, duration, stats), residue_method,
    )


def _scaled_outer_I_alpha_square(
    sys, cache, z, energy, duration, alpha, stats=None,
    residue_method="batched_contour"
):
    """Scaled form of ``I_alpha_square`` for an unbiased outer pole."""
    i0 = _i0_scale(sys)
    da = sys.Delta(alpha)

    def integrand(zp):
        zp = np.asarray(zp, dtype=np.complex128)
        first = np.zeros_like(zp)
        coefficient = np.full_like(zp, complex(sys.DELTA))
        for beta in sys.lead_names:
            db = sys.Delta(beta)
            if db == 0.0:
                continue
            first += (
                db
                * _scaled_outer_chi_square(
                    z, zp, db, duration, -1, i0
                )
                / ((zp - energy + db + 1j * i0)
                   * (zp - energy + 1j * i0))
                * sys.Gamma0(beta) * sys.W / (2.0 * (zp + 1j * sys.W))
                * Gfr_R_mpm(sys, cache, energy)
            )
            coefficient += (
                db * sys.Gamma0(beta) * sys.W
                / (2.0 * (z + 1j * sys.W)
                   * (zp + da - db + 1j * sys.W))
            )
        return first - coefficient * _scaled_outer_Q_alpha(
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
    return _dispatch_residue(
        integrand,
        candidates,
        _residue_kwargs(sys, cache, z, duration, stats, scaled=True),
        residue_method,
    )


def _scaled_outer_I_C_square(sys, cache, z, energy, duration, stats=None,
                             residue_method="batched_contour"):
    """Scaled form of ``I_C_square`` for an unbiased outer pole."""
    i0 = _i0_scale(sys)

    def integrand(zp):
        zp = np.asarray(zp, dtype=np.complex128)
        first = np.zeros_like(zp)
        coefficient = np.full_like(zp, complex(sys.DELTA))
        for beta in sys.lead_names:
            db = sys.Delta(beta)
            if db == 0.0:
                continue
            first += (
                db
                * _scaled_outer_chi_square(
                    z, zp, db, duration, -1, i0
                )
                / ((zp - energy + db + 1j * i0)
                   * (zp - energy + 1j * i0))
                * sys.Gamma0(beta) * sys.W / (2.0 * (zp + 1j * sys.W))
                * Gfr_R_mpm(sys, cache, energy)
            )
            coefficient += (
                db * sys.Gamma0(beta) * sys.W
                / (2.0 * (z + 1j * sys.W)
                   * (zp - db + 1j * sys.W))
            )
        return first - coefficient * _scaled_outer_Q_C(
            sys, cache, z, zp, energy, duration
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
            db - 1j * sys.W,
        ))
    candidates.extend(cache.xi_biased)
    return _dispatch_residue(
        integrand,
        candidates,
        _residue_kwargs(sys, cache, z, duration, stats, scaled=True),
        residue_method,
    )


def S_alpha_square(sys, cache, z, energy, duration, alpha, stats=None,
                   residue_method="batched_contour"):
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
        + I_alpha_square(sys, cache, z, energy, duration, alpha, stats, residue_method)
    )


def S_C_square(sys, cache, z, energy, duration, stats=None,
               residue_method="batched_contour"):
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
        + I_C_square(sys, cache, z, energy, duration, stats, residue_method)
    )


def _scaled_outer_S_alpha_square(
    sys, cache, z, energy, duration, alpha, stats=None,
    residue_method="batched_contour"
):
    """Return ``exp(-1j*z*duration) * S_alpha_square`` stably."""
    if duration == 0.0:
        return 0.0j
    i0 = _i0_scale(sys)
    outer_phase = np.exp(-1j * z * duration)
    value = _scaled_outer_chi_square(
        z, energy, sys.Delta(alpha), duration, -1, i0
    )
    bracket = sys.DELTA * (
        np.exp(-1j * energy * duration) - outer_phase
    ) / (z - energy - 1j * i0)
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        bracket += sys.Gamma0(beta) * sys.W / 2.0 * (
            _scaled_outer_chi_square(
                z, energy, db, duration, -1, i0
            ) / (energy + 1j * sys.W)
            - np.exp(-1j * db * duration)
            * _scaled_outer_chi_square(
                z, energy, db, duration, +1, i0
            ) / (z + 1j * sys.W)
        )
    return (
        value
        + bracket * Gfr_R_mpm(sys, cache, energy)
        + _scaled_outer_I_alpha_square(
            sys, cache, z, energy, duration, alpha, stats, residue_method
        )
    )


def _scaled_outer_S_C_square(
    sys, cache, z, energy, duration, stats=None,
    residue_method="batched_contour"
):
    """Return ``exp(-1j*z*duration) * S_C_square`` stably."""
    if duration == 0.0:
        return 0.0j
    i0 = _i0_scale(sys)
    outer_phase = np.exp(-1j * z * duration)
    bracket = sys.DELTA * (
        np.exp(-1j * energy * duration) - outer_phase
    ) / (z - energy - 1j * i0)
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        bracket += sys.Gamma0(beta) * sys.W / 2.0 * (
            _scaled_outer_chi_square(
                z, energy, db, duration, -1, i0
            ) / (energy + 1j * sys.W)
            - np.exp(-1j * db * duration)
            * _scaled_outer_chi_square(
                z, energy, db, duration, +1, i0
            ) / (z + 1j * sys.W)
        )
    return (
        bracket * Gfr_R_mpm(sys, cache, energy)
        + _scaled_outer_I_C_square(
            sys, cache, z, energy, duration, stats, residue_method
        )
    )


def _evaluate_S_alpha(sys, cache, energies, duration, alpha, stats=None,
                      residue_method="batched_contour"):
    energies = _analytic_arguments(energies)
    return _evaluate_square_residues_batched(
        lambda pole, block: [
            _scaled_outer_S_alpha_square(sys, cache, pole, energy, duration, alpha, stats, residue_method)
            for energy in block
        ], cache.xi_unbiased, energies,
    )


def _evaluate_S_C(sys, cache, shifted_energies, duration, stats=None,
                  residue_method="batched_contour"):
    shifted_energies = _analytic_arguments(shifted_energies)
    return _evaluate_square_residues_batched(
        lambda pole, block: [
            _scaled_outer_S_C_square(sys, cache, pole, energy, duration, stats, residue_method)
            for energy in block
        ], cache.xi_unbiased, shifted_energies,
    )


def _evaluate_square_residues_batched(evaluator, poles, energies, block_size=64):
    """Evaluate square residue families in bounded energy blocks.

    ``evaluator`` receives one pole and an energy block.  The block boundary
    is intentionally explicit so callers can later replace the scalar
    contour body with vectorized contour sampling without changing cache
    ownership or output shape.
    """
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)
    energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
    result = np.empty((len(poles), len(energies)), dtype=np.complex128)
    size = max(1, int(block_size))
    for start in range(0, len(energies), size):
        stop = min(start + size, len(energies))
        block = energies[start:stop]
        for row, pole in enumerate(poles):
            values = np.asarray(evaluator(pole, block), dtype=np.complex128)
            if values.shape != (stop - start,):
                values = values.reshape(stop - start)
            result[row, start:stop] = values
    return result


def analytic_square_residue(function, pole, *, step=1e-7):
    """Stable simple-pole residue estimate for diagnostic promotion tests.

    The assembled square integrands contain cancellations and clustered
    poles; production therefore continues to use the validated contour
    evaluator until all analytic cases pass.  This helper handles isolated
    simple poles and returns a finite zero for removable candidates.
    """
    pole = complex(pole)
    offsets = np.asarray([step, -step, 0.5 * step, -0.5 * step], dtype=float)
    values = np.asarray([
        (offset + 0.0j) * function(pole + offset)
        for offset in offsets
    ], dtype=np.complex128)
    estimate = 0.5 * (values[0] + values[1])
    if abs(estimate) <= 1e-10 * max(1.0, float(np.max(np.abs(values)))):
        return 0.0j
    return complex(estimate)


def analytic_lower_half_plane_residue(function, candidates, **kwargs) -> complex:
    """Evaluate isolated simple residues analytically, with contour fallback.

    Clustered candidates and any unstable local limit are delegated to the
    validated contour oracle.  This makes the analytic route fail safe while
    allowing simple rational terms to avoid contour quadrature.
    """
    values = np.asarray(candidates, dtype=np.complex128).reshape(-1)
    values = values[np.isfinite(values) & (values.imag < 0.0)]
    if not values.size:
        return 0.0j
    scale = max(1.0, float(np.max(np.abs(values))))
    clusters = _cluster_candidates(values, kwargs.get("merge_tolerance", 1e-7) * scale)
    if any(len(group) != 1 for group in clusters):
        return lower_half_plane_residue(function, values, **kwargs)
    total = 0.0j
    step = kwargs.pop("analytic_step", 1e-7)
    weight = float(kwargs.get("contribution_weight", 1.0))
    for group in clusters:
        pole = complex(group[0])
        try:
            value = analytic_square_residue(function, pole, step=step)
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            return lower_half_plane_residue(function, values, **kwargs)
        if not np.isfinite(value):
            return lower_half_plane_residue(function, values, **kwargs)
        total -= weight * value
    return total


def _dispatch_residue(function, candidates, kwargs, method: str):
    if method == "analytic":
        return analytic_lower_half_plane_residue(function, candidates, **kwargs)
    return lower_half_plane_residue(function, candidates, **kwargs)


def _post_A(sys, cache, energies, time, duration, alpha, S_values):
    energies = _analytic_arguments(energies).reshape(-1)
    poles = cache.xi_unbiased[:, None]
    residues = cache.residues_unbiased[:, None]
    return Gfr_R_mpm(sys, cache, energies) - np.exp(
        1j * sys.Delta(alpha) * duration
    ) * np.sum(
        residues
        * np.exp(-1j * (poles - energies[None, :]) * (time - duration))
        * np.exp(1j * energies[None, :] * duration)
        * S_values,
        axis=0,
    )


def _post_C(sys, cache, shifted, time, duration, S_values):
    shifted = _analytic_arguments(shifted).reshape(-1)
    poles = cache.xi_unbiased[:, None]
    residues = cache.residues_unbiased[:, None]
    return Gfr_R_mpm(sys, cache, shifted) - np.sum(
        residues
        * np.exp(-1j * (poles - shifted[None, :]) * (time - duration))
        * np.exp(1j * shifted[None, :] * duration)
        * S_values,
        axis=0,
    )


def build_square_kernel_cache(sys, energies, duration=None, cache=None,
                              store: SquareKernelCacheStore | None = None,
                              progress=None, residue_method: str | None = None):
    from psscba.backend.protocols.analytic import upward_a_residue, upward_c_residue

    sys.validate_pulse_protocol()
    if sys.pulse_protocol != "square":
        raise RuntimeError("Square kernels require a System with pulse_protocol='square'.")
    duration = sys.pulse_duration if duration is None else float(duration)
    residue_method = (
        getattr(sys, "square_residue_method", "batched_contour")
        if residue_method is None else residue_method
    )
    if residue_method not in ("batched_contour", "analytic"):
        raise ValueError("square residue method must be batched_contour or analytic")
    if duration is None or not np.isfinite(duration) or duration < 0.0:
        raise ValueError("A finite nonnegative square-pulse duration is required.")
    energies = np.asarray(energies, dtype=float).reshape(-1)
    cache = sys.prepare_poles() if cache is None else cache
    if store is None:
        store = getattr(sys, "_square_cache_store", None)
        if store is None:
            store = SquareKernelCacheStore()
            sys._square_cache_store = store
    key = store.key(sys, cache, energies, duration)
    existing = store.get(key)
    if existing is not None:
        if progress is not None:
            progress("square_cache_build", {
                "cache_hit": True, "grid_points": len(energies),
                "residue_method": existing.residue_method,
                "duration_seconds": 0.0,
            })
        return existing
    if progress is not None:
        progress("square_cache_build", {
            "cache_hit": False, "grid_points": len(energies),
            "residue_method": residue_method,
        })
    build_start = perf_counter()
    stats = _ResidueStats()
    actual_residue_method = residue_method
    S_alpha = {
        lead: _readonly(_evaluate_S_alpha(sys, cache, energies, duration, lead, stats, residue_method))
        for lead in sys.lead_names
    }
    S_C = {}
    # omega=0 is the generic installed Sigma^< source used by PS-SCBA;
    # +/-omega_q retain the decomposed preparation-fixed oracle sectors.
    for omega in sorted({-float(sys.w_q), 0.0, float(sys.w_q)}):
        S_C[omega] = _readonly(
            _evaluate_S_C(sys, cache, energies + omega, duration, stats, residue_method)
        )

    # Analytic promotion is guarded by an independent contour spot-check.
    # Any disagreement deterministically falls back to the batched contour
    # result before the cache can be installed.
    if residue_method == "analytic" and len(energies):
        sample = np.unique(np.asarray([0, len(energies) - 1], dtype=int))
        reference_stats = _ResidueStats()
        analytic_bad = False
        for lead in sys.lead_names:
            reference = _evaluate_S_alpha(
                sys, cache, energies[sample], duration, lead,
                reference_stats, "batched_contour"
            )
            if not np.allclose(reference, np.asarray(S_alpha[lead])[:, sample],
                               rtol=1e-7, atol=1e-9):
                analytic_bad = True
        for omega in sorted(S_C):
            reference = _evaluate_S_C(
                sys, cache, (energies + omega)[sample], duration,
                reference_stats, "batched_contour"
            )
            if not np.allclose(reference, np.asarray(S_C[omega])[:, sample],
                               rtol=1e-7, atol=1e-9):
                analytic_bad = True
        if analytic_bad:
            actual_residue_method = "batched_contour_fallback"
            stats = _ResidueStats()
            S_alpha = {
                lead: _readonly(_evaluate_S_alpha(
                    sys, cache, energies, duration, lead, stats, "batched_contour"
                )) for lead in sys.lead_names
            }
            S_C = {
                omega: _readonly(_evaluate_S_C(
                    sys, cache, energies + omega, duration, stats, "batched_contour"
                )) for omega in sorted({-float(sys.w_q), 0.0, float(sys.w_q)})
            }

    analytic = energies.astype(np.complex128) + 1e-14j
    max_a_abs = max_a_scaled = 0.0
    for lead in sys.lead_names:
        raw = _post_A(sys, cache, analytic, duration, duration, lead, S_alpha[lead])
        expected = np.asarray(
            upward_a_residue(sys, analytic, duration, lead, cache, enforce_boundary=False)
        ).reshape(-1)
        difference = np.abs(raw - expected)
        band = sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(expected)
        max_a_abs = max(max_a_abs, float(np.max(difference)))
        max_a_scaled = max(max_a_scaled, float(np.max(difference / band)))
    max_c_abs = max_c_scaled = 0.0
    for omega, values in S_C.items():
        raw = _post_C(
            sys, cache, analytic + omega, duration, duration, values
        )
        expected = np.asarray(
            upward_c_residue(sys, omega, analytic, duration, cache, enforce_boundary=False)
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
    if progress is not None:
        progress("square_residue_validation", {
            "validation": "PASS",
            "residue_method": actual_residue_method,
            "residue_max_scaled_error": stats.max_scaled_error,
        })
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
        residue_negligible_count=stats.negligible,
        turnoff_A_abs_error=max_a_abs,
        turnoff_A_scaled_error=max_a_scaled,
        turnoff_C_abs_error=max_c_abs,
        turnoff_C_scaled_error=max_c_scaled,
        residue_method=actual_residue_method,
        cache_key=repr(key),
    )
    store.put(key, result)
    sys._square_kernel_cache = result
    if progress is not None:
        progress("square_cache_build", {
            "cache_hit": False, "grid_points": len(energies),
            "residue_method": result.residue_method,
            "duration_seconds": perf_counter() - build_start,
            "residue_validation": "PASS",
        })
    return result


def A_square(sys, energy, t, alpha, cache=None, square_cache=None, duration=None):
    from psscba.backend.protocols.analytic import A_up

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
    from psscba.backend.protocols.analytic import C_up

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
                sys, cache, primes + omega, times[index], duration, S_values
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
            r * np.exp(1j * e * duration)
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
            r * np.exp(1j * e * duration)
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
