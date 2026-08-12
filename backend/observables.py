from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad_vec

from backend.distribution import expc, fermi_dirac
from backend.minimal_poles import (
    Gbar_R_mpm,
    Gbiased_R_mpm,
    Gfr_R_mpm,
    PoleCache,
)
from backend.system_classes import Lead, System
from backend.square_pulse import (
    A_square,
    A_square_direct,
    B_square,
    C_square,
    C_square_direct,
    D_square,
    build_square_kernel_cache,
)


@dataclass(frozen=True)
class TransientResult:
    t: np.ndarray
    currents: dict[Lead, np.ndarray]
    occupation: np.ndarray
    continuity_residual: np.ndarray
    diagnostics: dict[str, float | int | bool | str]


def linewidth(sys: System, lead: Lead, energy):
    energy = np.asarray(energy, dtype=np.complex128)
    return sys.Gamma0(lead) * sys.W**2 / (energy**2 + sys.W**2)


def integrate_cauchy_linear(poles, grid, values):
    """Integrate a linear interpolant as values(y)/(pole-y+i0) dy."""
    poles = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=np.complex128)
    if values.shape != grid.shape:
        raise ValueError("Cauchy integrand values must match the integration grid.")
    a = grid[:-1]
    b = grid[1:]
    slope = np.diff(values) / np.diff(grid)
    q = poles[:, None] + 1j * np.finfo(float).eps
    constant = values[:-1][None, :] + slope[None, :] * (q - a[None, :])
    segment = (
        -slope[None, :] * (b - a)[None, :]
        - constant * (np.log(q - b[None, :]) - np.log(q - a[None, :]))
    )
    return np.sum(segment, axis=1)


def _restore_shape(values, *inputs):
    scalar = all(np.asarray(item).ndim == 0 for item in inputs)
    return values.reshape(-1)[0] if scalar else np.squeeze(values)


def _frozen_kernel_less(frozen, energies):
    """Interpolate the exact lesser function used to freeze Sigma_ep^<."""
    energies = np.asarray(energies, dtype=float)
    if len(energies) == len(frozen.w) and np.allclose(energies, frozen.w):
        return frozen.G_kernel_less
    real = np.interp(
        energies, frozen.w, frozen.G_kernel_less.real, left=0.0, right=0.0
    )
    imag = np.interp(
        energies, frozen.w, frozen.G_kernel_less.imag, left=0.0, right=0.0
    )
    return real + 1j * imag


def _M_alpha(sys: System, cache: PoleCache, z, energy, alpha: Lead):
    z = np.asarray(z, dtype=np.complex128)
    energy = np.asarray(energy, dtype=np.complex128)
    delta_alpha = sys.Delta(alpha)
    lead_fd = np.zeros(np.broadcast_shapes(z.shape, energy.shape), dtype=np.complex128)
    for beta in sys.lead_names:
        lead_fd += (
            sys.Delta(beta) * sys.Gamma0(beta) * sys.W
            / (
                2.0
                * (z + 1j * sys.W)
                * (energy + delta_alpha - sys.Delta(beta) + 1j * sys.W)
            )
        )
    return (
        delta_alpha / (z - energy)
        + (sys.DELTA + lead_fd) * Gbar_R_mpm(sys, cache, energy + delta_alpha)
    )


def _S_C(sys: System, z, shifted_energy):
    z = np.asarray(z, dtype=np.complex128)
    shifted_energy = np.asarray(shifted_energy, dtype=np.complex128)
    out = np.full(
        np.broadcast_shapes(z.shape, shifted_energy.shape),
        complex(sys.DELTA),
        dtype=np.complex128,
    )
    for beta in sys.lead_names:
        out += (
            sys.Delta(beta) * sys.Gamma0(beta) * sys.W
            / (
                2.0
                * (z + 1j * sys.W)
                * (shifted_energy - sys.Delta(beta) + 1j * sys.W)
            )
        )
    return out


def A_down(sys: System, energy, t, alpha: Lead, cache: PoleCache | None = None):
    cache = sys.prepare_poles() if cache is None else cache
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    x = cache.xi[:, None, None]
    r = cache.residues[:, None, None]
    e = energies[None, :, None]
    tt = times[None, None, :]
    mfac = _M_alpha(sys, cache, x, e, alpha)
    transient = np.sum(
        r * np.exp(-1j * (x - e) * tt)
        * mfac / (x - e - sys.Delta(alpha)),
        axis=0,
    )
    out = Gfr_R_mpm(sys, cache, energies)[:, None] - transient
    zero_time = times == 0.0
    if np.any(zero_time):
        out[:, zero_time] = Gbar_R_mpm(
            sys, cache, energies + sys.Delta(alpha)
        )[:, None]
    return _restore_shape(out, energy, t)


def A_down_direct(
    sys: System,
    energy: float,
    t: float,
    alpha: Lead,
    cache: PoleCache | None = None,
    bounds: tuple[float, float] | None = None,
):
    """Validation quadrature for the defining real-axis A integral."""
    cache = sys.prepare_poles() if cache is None else cache
    lower, upper = bounds or (sys.e_min, sys.e_max)
    delta_alpha = sys.Delta(alpha)
    gbar_boundary = Gbar_R_mpm(sys, cache, energy + delta_alpha)

    def integrand(omega):
        lead_fd = 0.0j
        for beta in sys.lead_names:
            lead_fd += (
                sys.Delta(beta) * sys.Gamma0(beta) * sys.W
                / (
                    2.0 * (omega + 1j * sys.W)
                    * (energy + delta_alpha - sys.Delta(beta) + 1j * sys.W)
                )
            )
        mfac = (
            delta_alpha / (omega - energy - 1j * sys.ETA)
            + (sys.DELTA + lead_fd) * gbar_boundary
        )
        return (
            np.exp(-1j * (omega - energy) * t)
            * Gfr_R_mpm(sys, cache, omega)
            * mfac
            / (omega - energy - delta_alpha - 1j * sys.ETA)
            / (2j * np.pi)
        )

    integral, _ = quad_vec(
        integrand,
        lower,
        upper,
        epsabs=max(cache.tolerance, 1e-9),
        epsrel=1e-7,
        limit=1000,
    )
    return Gfr_R_mpm(sys, cache, energy) + integral


def B_down(
    sys: System,
    energy,
    energy_prime,
    t,
    beta: Lead,
    cache: PoleCache | None = None,
):
    cache = sys.prepare_poles() if cache is None else cache
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    x = cache.xi[None, None, :, None]
    r = cache.residues[None, None, :, None]
    e = energies[:, None, None, None]
    ep = primes[None, :, None, None]
    tt = times[None, None, None, :]
    mfac = _M_alpha(sys, cache, cache.xi[:, None], primes[None, :], beta)
    pole_term = np.sum(
        expc(e - x, tt) * r
        * mfac.T[None, :, :, None]
        / (x - ep - sys.Delta(beta)),
        axis=2,
    )
    out = (
        expc(
            energies[:, None, None] - primes[None, :, None],
            times[None, None, :],
        )
        * Gfr_R_mpm(sys, cache, primes)[None, :, None]
        - pole_term
    )
    return _restore_shape(out, energy, energy_prime, t)


def C_down(
    sys: System,
    omega: float,
    energy_prime,
    t,
    cache: PoleCache | None = None,
):
    cache = sys.prepare_poles() if cache is None else cache
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    shifted = primes + omega
    x = cache.xi[:, None, None]
    r = cache.residues[:, None, None]
    ep = shifted[None, :, None]
    tt = times[None, None, :]
    sc = _S_C(sys, x, ep)
    transient = np.sum(
        r * np.exp(-1j * (x - ep) * tt) * sc
        * Gbar_R_mpm(sys, cache, shifted)[None, :, None]
        / (x - ep),
        axis=0,
    )
    out = Gfr_R_mpm(sys, cache, shifted)[:, None] - transient
    zero_time = times == 0.0
    if np.any(zero_time):
        out[:, zero_time] = Gbar_R_mpm(sys, cache, shifted)[:, None]
    return _restore_shape(out, energy_prime, t)


def D_down(
    sys: System,
    omega: float,
    energy_prime,
    energy,
    t,
    cache: PoleCache | None = None,
):
    cache = sys.prepare_poles() if cache is None else cache
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    shifted = primes + omega
    x = cache.xi[None, None, :, None]
    r = cache.residues[None, None, :, None]
    e = energies[:, None, None, None]
    ep = shifted[None, :, None, None]
    tt = times[None, None, None, :]
    sc = _S_C(sys, cache.xi[:, None], shifted[None, :])
    pole_term = np.sum(
        expc(e - x, tt) * r
        * sc.T[None, :, :, None]
        * Gbar_R_mpm(sys, cache, shifted)[None, :, None, None]
        / (x - ep),
        axis=2,
    )
    out = (
        expc(
            energies[:, None, None] - shifted[None, :, None],
            times[None, None, :],
        )
        * Gfr_R_mpm(sys, cache, shifted)[None, :, None]
        - pole_term
    )
    return _restore_shape(out, energy_prime, energy, t)


def _M_up(sys: System, cache: PoleCache, z, energy, alpha: Lead):
    z = np.asarray(z, dtype=np.complex128)
    energy = np.asarray(energy, dtype=np.complex128)
    finite_difference = np.zeros(
        np.broadcast_shapes(z.shape, energy.shape), dtype=np.complex128
    )
    for beta in sys.lead_names:
        finite_difference += (
            sys.Delta(beta) * sys.Gamma0(beta) * sys.W
            / (
                2.0
                * (energy + 1j * sys.W)
                * (z - sys.Delta(beta) + 1j * sys.W)
            )
        )
    # With the code's biased embedding convention Sigma_beta(z-Delta_beta),
    # the Lorentzian difference entering the Dyson boundary has this effective
    # plus sign.  The raw A(0)=Gfr boundary gate guards it against regression.
    return (
        sys.Delta(alpha) / (z - energy - sys.Delta(alpha))
        + (sys.DELTA + finite_difference) * Gfr_R_mpm(sys, cache, energy)
    )


def _N_up(sys: System, cache: PoleCache, z, shifted_energy):
    z = np.asarray(z, dtype=np.complex128)
    shifted_energy = np.asarray(shifted_energy, dtype=np.complex128)
    finite_difference = np.zeros(
        np.broadcast_shapes(z.shape, shifted_energy.shape), dtype=np.complex128
    )
    for beta in sys.lead_names:
        finite_difference += (
            sys.Delta(beta) * sys.Gamma0(beta) * sys.W
            / (
                2.0
                * (shifted_energy + 1j * sys.W)
                * (z - sys.Delta(beta) + 1j * sys.W)
            )
        )
    # This is the same Dyson-consistent Lorentzian finite difference as _M_up.
    return (sys.DELTA + finite_difference) * Gfr_R_mpm(
        sys, cache, shifted_energy
    )


def _A_up_residue(
    sys: System,
    energy,
    t,
    alpha: Lead,
    cache: PoleCache,
    *,
    enforce_boundary: bool,
):
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    x = cache.xi_biased[:, None, None]
    r = cache.residues_biased[:, None, None]
    e = energies[None, :, None]
    tt = times[None, None, :]
    mfac = _M_up(sys, cache, x, e, alpha)
    transient = np.sum(
        r
        * np.exp(-1j * (x - e - sys.Delta(alpha)) * tt)
        * mfac
        / (x - e),
        axis=0,
    )
    out = Gbiased_R_mpm(
        sys, cache, energies + sys.Delta(alpha)
    )[:, None] + transient
    if enforce_boundary:
        zero_time = times == 0.0
        if np.any(zero_time):
            out[:, zero_time] = Gfr_R_mpm(sys, cache, energies)[:, None]
    return _restore_shape(out, energy, t)


def A_up(sys: System, energy, t, alpha: Lead, cache: PoleCache | None = None):
    """Upward-step retarded transform from the biased Green-pole residues."""
    cache = sys.prepare_poles() if cache is None else cache
    return _A_up_residue(
        sys, energy, t, alpha, cache, enforce_boundary=True
    )


def A_up_direct(
    sys: System,
    energy: float,
    t: float,
    alpha: Lead,
    cache: PoleCache | None = None,
    bounds: tuple[float, float] | None = None,
):
    """Validation quadrature for the defining upward real-axis A integral."""
    cache = sys.prepare_poles() if cache is None else cache
    lower, upper = bounds or (sys.e_min, sys.e_max)
    da = sys.Delta(alpha)
    gfr_boundary = Gfr_R_mpm(sys, cache, energy)

    def integrand(omega):
        finite_difference = 0.0j
        for beta in sys.lead_names:
            finite_difference += (
                sys.Delta(beta) * sys.Gamma0(beta) * sys.W
                / (
                    2.0
                    * (energy + 1j * sys.W)
                    * (omega + da - sys.Delta(beta) + 1j * sys.W)
                )
            )
        bracket = (
            da / (omega - energy - 1j * sys.ETA)
            + (sys.DELTA + finite_difference) * gfr_boundary
        )
        return (
            np.exp(-1j * (omega - energy) * t)
            * Gbiased_R_mpm(sys, cache, omega + da)
            * bracket
            / (omega - energy + da - 1j * sys.ETA)
            / (2j * np.pi)
        )

    integral, _ = quad_vec(
        integrand,
        lower,
        upper,
        epsabs=max(cache.tolerance, 1e-9),
        epsrel=1e-7,
        limit=1000,
    )
    return Gbiased_R_mpm(sys, cache, energy + da) - integral


def B_up(
    sys: System,
    energy,
    energy_prime,
    t,
    alpha: Lead,
    beta: Lead,
    cache: PoleCache | None = None,
):
    cache = sys.prepare_poles() if cache is None else cache
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    x = cache.xi_biased[None, None, :, None]
    r = cache.residues_biased[None, None, :, None]
    e = energies[:, None, None, None]
    ep = primes[None, :, None, None]
    tt = times[None, None, None, :]
    mfac = _M_up(
        sys, cache, cache.xi_biased[:, None], primes[None, :], beta
    )
    pole_term = np.sum(
        expc(e + sys.Delta(alpha) - x, tt)
        * r
        * mfac.T[None, :, :, None]
        / (x - ep),
        axis=2,
    )
    out = (
        expc(
            energies[:, None, None]
            - primes[None, :, None]
            + sys.Delta(alpha)
            - sys.Delta(beta),
            times[None, None, :],
        )
        * Gbiased_R_mpm(
            sys, cache, primes + sys.Delta(beta)
        )[None, :, None]
        + pole_term
    )
    return _restore_shape(out, energy, energy_prime, t)


def _C_up_residue(
    sys: System,
    omega: float,
    energy_prime,
    t,
    cache: PoleCache,
    *,
    enforce_boundary: bool,
):
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    shifted = primes + omega
    x = cache.xi_biased[:, None, None]
    r = cache.residues_biased[:, None, None]
    ep = shifted[None, :, None]
    tt = times[None, None, :]
    nfac = _N_up(sys, cache, x, ep)
    transient = np.sum(
        r * np.exp(-1j * (x - ep) * tt) * nfac / (x - ep), axis=0
    )
    out = Gbiased_R_mpm(sys, cache, shifted)[:, None] + transient
    if enforce_boundary:
        zero_time = times == 0.0
        if np.any(zero_time):
            out[:, zero_time] = Gfr_R_mpm(sys, cache, shifted)[:, None]
    return _restore_shape(out, energy_prime, t)


def C_up(
    sys: System,
    omega: float,
    energy_prime,
    t,
    cache: PoleCache | None = None,
):
    cache = sys.prepare_poles() if cache is None else cache
    return _C_up_residue(
        sys, omega, energy_prime, t, cache, enforce_boundary=True
    )


def C_up_direct(
    sys: System,
    omega: float,
    energy_prime: float,
    t: float,
    cache: PoleCache | None = None,
    bounds: tuple[float, float] | None = None,
):
    """Validation quadrature for the defining upward real-axis C integral."""
    cache = sys.prepare_poles() if cache is None else cache
    lower, upper = bounds or (sys.e_min, sys.e_max)
    shifted = omega + energy_prime
    gfr_boundary = Gfr_R_mpm(sys, cache, shifted)

    def integrand(nu):
        finite_difference = 0.0j
        for beta in sys.lead_names:
            finite_difference += (
                sys.Delta(beta) * sys.Gamma0(beta) * sys.W
                / (
                    2.0
                    * (shifted + 1j * sys.W)
                    * (nu - sys.Delta(beta) + 1j * sys.W)
                )
            )
        return (
            np.exp(-1j * (nu - shifted) * t)
            * Gbiased_R_mpm(sys, cache, nu)
            * (sys.DELTA + finite_difference)
            * gfr_boundary
            / (nu - shifted - 1j * sys.ETA)
            / (2j * np.pi)
        )

    integral, _ = quad_vec(
        integrand,
        lower,
        upper,
        epsabs=max(cache.tolerance, 1e-9),
        epsrel=1e-7,
        limit=1000,
    )
    return Gbiased_R_mpm(sys, cache, shifted) - integral


def D_up(
    sys: System,
    omega: float,
    energy_prime,
    energy,
    t,
    alpha: Lead,
    cache: PoleCache | None = None,
):
    cache = sys.prepare_poles() if cache is None else cache
    primes = np.atleast_1d(np.asarray(energy_prime, dtype=np.complex128))
    energies = np.atleast_1d(np.asarray(energy, dtype=np.complex128))
    times = np.atleast_1d(np.asarray(t, dtype=float))
    shifted = primes + omega
    x = cache.xi_biased[None, None, :, None]
    r = cache.residues_biased[None, None, :, None]
    e = energies[:, None, None, None]
    ep = shifted[None, :, None, None]
    tt = times[None, None, None, :]
    nfac = _N_up(
        sys, cache, cache.xi_biased[:, None], shifted[None, :]
    )
    pole_term = np.sum(
        expc(e + sys.Delta(alpha) - x, tt)
        * r
        * nfac.T[None, :, :, None]
        / (x - ep),
        axis=2,
    )
    out = (
        expc(
            energies[:, None, None]
            - shifted[None, :, None]
            + sys.Delta(alpha),
            times[None, None, :],
        )
        * Gbiased_R_mpm(sys, cache, shifted)[None, :, None]
        + pole_term
    )
    return _restore_shape(out, energy_prime, energy, t)


# Backwards-compatible names.
def A(sys: System, e, t, alpha: Lead):
    return A_down(sys, e, t, alpha)


def B(sys: System, e, e_prime, t, alpha: Lead):
    return B_down(sys, e, e_prime, t, alpha)


def C(sys: System, t, omega, e_prime):
    return C_down(sys, omega, e_prime, t)


def D(sys: System, omega, e_prime, e, t):
    return D_down(sys, omega, e_prime, e, t)


def _psi_lead_down_batch(sys, cache, outer, inner, time, alpha, A_values):
    result = np.zeros(len(outer), dtype=np.complex128)
    da = sys.Delta(alpha)
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        a_beta = A_values[beta]
        b_beta = B_down(sys, outer, inner, time, beta, cache)
        if b_beta.ndim == 1:
            b_beta = b_beta.reshape(len(outer), len(inner))
        smooth_integrand = (
            1j * np.exp(1j * (outer[:, None] - inner[None, :]) * time)
            * fermi_dirac(inner, sys.beta_fc(beta), sys.mu_fc(beta))[None, :]
            * a_beta[None, :]
            * linewidth(sys, beta, inner)[None, :]
            * np.conjugate(b_beta)
        )
        result += np.trapezoid(smooth_integrand, inner, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * inner * time)
            * fermi_dirac(inner, sys.beta_fc(beta), sys.mu_fc(beta))
            * a_beta * linewidth(sys, beta, inner)
            * np.conjugate(Gbar_R_mpm(sys, cache, inner + db))
        )
        result -= (
            np.exp(1j * outer * time)
            * integrate_cauchy_linear(outer + da - db, inner, singular_values)
            / (2.0 * np.pi)
        )
    return result


def _psi_ep_down_batch(sys, cache, frozen, outer, inner, time, alpha, C_values):
    if sys.g_q == 0.0:
        return np.zeros(len(outer), dtype=np.complex128)
    da = sys.Delta(alpha)
    g_less = _frozen_kernel_less(frozen, inner)
    result = np.zeros(len(outer), dtype=np.complex128)
    for omega, occupation_factor in (
        (-sys.w_q, frozen.N0 + 1.0),
        (+sys.w_q, frozen.N0),
    ):
        if occupation_factor == 0.0:
            continue
        c_value = C_values[omega]
        d_value = D_down(sys, omega, inner, outer, time, cache)
        if d_value.ndim == 1:
            d_value = d_value.reshape(len(outer), len(inner))
        common_inner = (
            c_value * sys.g_q**2 * g_less
        )
        smooth_integrand = (
            occupation_factor
            * np.exp(1j * (outer[:, None] - inner[None, :] - omega) * time)
            * common_inner[None, :]
            * np.conjugate(d_value)
        )
        result += np.trapezoid(smooth_integrand, inner, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * inner * time) * common_inner
            * np.conjugate(Gbar_R_mpm(sys, cache, inner + omega))
        )
        result += (
            1j * occupation_factor * np.exp(1j * (outer - omega) * time)
            * integrate_cauchy_linear(outer - omega + da, inner, singular_values)
            / (2.0 * np.pi)
        )
    return result


def _psi_lead_up_batch(sys, cache, outer, inner, time, alpha, A_values):
    result = np.zeros(len(outer), dtype=np.complex128)
    da = sys.Delta(alpha)
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        a_beta = A_values[beta]
        b_beta = B_up(sys, outer, inner, time, alpha, beta, cache)
        if b_beta.ndim == 1:
            b_beta = b_beta.reshape(len(outer), len(inner))
        smooth_integrand = (
            1j
            * np.exp(
                1j
                * (outer[:, None] - inner[None, :] + da - db)
                * time
            )
            * fermi_dirac(inner, sys.beta_fc(beta), sys.mu_fc(beta))[None, :]
            * a_beta[None, :]
            * linewidth(sys, beta, inner)[None, :]
            * np.conjugate(b_beta)
        )
        result += np.trapezoid(smooth_integrand, inner, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * inner * time)
            * fermi_dirac(inner, sys.beta_fc(beta), sys.mu_fc(beta))
            * a_beta
            * linewidth(sys, beta, inner)
            * np.conjugate(Gfr_R_mpm(sys, cache, inner))
        )
        result -= (
            np.exp(1j * (outer + da - db) * time)
            * integrate_cauchy_linear(outer, inner, singular_values)
            / (2.0 * np.pi)
        )
    return result


def _psi_ep_up_batch(sys, cache, frozen, outer, inner, time, alpha, C_values):
    if sys.g_q == 0.0:
        return np.zeros(len(outer), dtype=np.complex128)
    da = sys.Delta(alpha)
    g_less = _frozen_kernel_less(frozen, inner)
    result = np.zeros(len(outer), dtype=np.complex128)
    for omega, occupation_factor in (
        (-sys.w_q, frozen.N0 + 1.0),
        (+sys.w_q, frozen.N0),
    ):
        if occupation_factor == 0.0:
            continue
        c_value = C_values[omega]
        d_value = D_up(sys, omega, inner, outer, time, alpha, cache)
        if d_value.ndim == 1:
            d_value = d_value.reshape(len(outer), len(inner))
        common_inner = c_value * sys.g_q**2 * g_less
        smooth_integrand = (
            occupation_factor
            * np.exp(
                1j
                * (outer[:, None] - inner[None, :] - omega + da)
                * time
            )
            * common_inner[None, :]
            * np.conjugate(d_value)
        )
        result += np.trapezoid(smooth_integrand, inner, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * inner * time)
            * common_inner
            * np.conjugate(Gfr_R_mpm(sys, cache, inner + omega))
        )
        result += (
            1j
            * occupation_factor
            * np.exp(1j * (outer - omega + da) * time)
            * integrate_cauchy_linear(outer - omega, inner, singular_values)
            / (2.0 * np.pi)
        )
    return result


def _psi_lead_square_batch(
    sys, cache, square_cache, outer, inner, time, alpha, A_values,
    stored_B,
):
    result = np.zeros(len(outer), dtype=np.complex128)
    duration = square_cache.duration
    da = sys.Delta(alpha)
    for beta in sys.lead_names:
        db = sys.Delta(beta)
        phase = np.exp(1j * (da - db) * duration)
        a_beta = A_values[beta]
        b_up = stored_B[beta]
        b_greater = B_square(
            sys, outer, inner, time, beta, cache, square_cache, duration
        )
        if np.asarray(b_greater).ndim == 1:
            b_greater = np.asarray(b_greater).reshape(len(outer), len(inner))
        history = np.conjugate(b_up) + np.conjugate(b_greater) / phase
        common = (
            fermi_dirac(inner, sys.beta_fc(beta), sys.mu_fc(beta))
            * a_beta
            * linewidth(sys, beta, inner)
        )
        smooth = (
            1j * np.exp(1j * (outer[:, None] - inner[None, :]) * time)
            * common[None, :] * phase * history
        )
        result += np.trapezoid(smooth, inner, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * inner * time) * common
            * np.conjugate(Gfr_R_mpm(sys, cache, inner))
        )
        result -= (
            phase * np.exp(1j * outer * time)
            * integrate_cauchy_linear(outer, inner, singular_values)
            / (2.0 * np.pi)
        )
    return result


def _psi_ep_square_batch(
    sys, cache, frozen, square_cache, outer, inner, time, alpha,
    C_values, stored_D,
):
    if sys.g_q == 0.0:
        return np.zeros(len(outer), dtype=np.complex128)
    duration = square_cache.duration
    da = sys.Delta(alpha)
    phase = np.exp(1j * da * duration)
    g_less = _frozen_kernel_less(frozen, inner)
    result = np.zeros(len(outer), dtype=np.complex128)
    for omega, occupation_factor in (
        (-sys.w_q, frozen.N0 + 1.0),
        (+sys.w_q, frozen.N0),
    ):
        if occupation_factor == 0.0:
            continue
        c_value = C_values[omega]
        d_up = stored_D[omega]
        d_greater = D_square(
            sys, omega, inner, outer, time, alpha, cache, square_cache, duration
        )
        if np.asarray(d_greater).ndim == 1:
            d_greater = np.asarray(d_greater).reshape(len(outer), len(inner))
        history = np.conjugate(d_up) + np.conjugate(d_greater) / phase
        common = c_value * sys.g_q**2 * g_less
        smooth = (
            occupation_factor
            * np.exp(1j * (outer[:, None] - inner[None, :] - omega) * time)
            * common[None, :] * phase * history
        )
        result += np.trapezoid(smooth, inner, axis=1) / (2.0 * np.pi)
        singular_values = (
            np.exp(-1j * inner * time) * common
            * np.conjugate(Gfr_R_mpm(sys, cache, inner + omega))
        )
        result += (
            1j * occupation_factor * phase
            * np.exp(1j * (outer - omega) * time)
            * integrate_cauchy_linear(outer - omega, inner, singular_values)
            / (2.0 * np.pi)
        )
    return result


def _occupation(sys, frozen, inner, A_values, C_values):
    g_lead = np.zeros(len(inner), dtype=np.complex128)
    for beta in sys.lead_names:
        g_lead += (
            1j * fermi_dirac(inner, sys.beta_fc(beta), sys.mu_fc(beta))
            * linewidth(sys, beta, inner)
            * np.abs(A_values[beta]) ** 2
        )
    g_less_equal = np.trapezoid(g_lead, inner) / (2.0 * np.pi)
    if sys.g_q != 0.0:
        middle = _frozen_kernel_less(frozen, inner)
        g_ep_integrand = sys.g_q**2 * middle * (
            (frozen.N0 + 1.0) * np.abs(C_values[-sys.w_q]) ** 2
            + frozen.N0 * np.abs(C_values[+sys.w_q]) ** 2
        )
        g_less_equal += np.trapezoid(g_ep_integrand, inner) / (2.0 * np.pi)
    occupation = -1j * g_less_equal
    noise = abs(occupation.imag)
    if noise > 1e-7 * max(1.0, abs(occupation.real)):
        raise RuntimeError(f"Transient occupation is not real (noise={noise:.3e}).")
    return float(occupation.real)


def continuity_residual(currents, occupation, times):
    """Charge residual for J_alpha=-d<N_alpha>/dt and e=1."""
    current_sum = np.sum(np.vstack(list(currents.values())), axis=0)
    derivative = np.gradient(
        occupation, times, edge_order=2 if len(times) > 2 else 1
    )
    return current_sum - derivative


def _require_protocol(sys: System, frozen, protocol: str) -> None:
    sys.validate_pulse_protocol()
    if sys.pulse_protocol != protocol or frozen.pulse_protocol != protocol:
        raise RuntimeError(
            "Pulse protocol does not match the frozen stationary state: "
            f"requested={protocol!r}, system={sys.pulse_protocol!r}, "
            f"frozen={frozen.pulse_protocol!r}. Rebuild the System and frozen cache."
        )


def _current_grid(sys, frozen, omega_int_n_x, omega_int_n_omega):
    if (
        omega_int_n_x is not None
        and omega_int_n_omega is not None
        and omega_int_n_x != omega_int_n_omega
    ):
        raise ValueError(
            "The A/B/C/D grid implementation requires omega_int_n_x and "
            "omega_int_n_omega to match."
        )
    n_energy = omega_int_n_omega or omega_int_n_x or len(frozen.w)
    same_grid = (
        n_energy == len(frozen.w)
        and np.isclose(sys.e_min, frozen.w[0])
        and np.isclose(sys.e_max, frozen.w[-1])
    )
    if same_grid:
        return np.asarray(frozen.w)
    return np.linspace(sys.e_min, sys.e_max, n_energy)


def _transient_diagnostics(
    sys, frozen, cache, energies, continuity, extra: dict | None = None
):
    diagnostics: dict[str, float | int | bool | str] = {
        "pulse_protocol": sys.pulse_protocol,
        "stationary_reference": "biased" if frozen.reference_is_biased else "unbiased",
        "N0": frozen.N0,
        "scba_iterations": frozen.result.n_iter,
        "scba_converged": frozen.result.converged,
        "scba_GR_abs_residual": frozen.result.res_GR_abs,
        "scba_GR_rel_residual": frozen.result.res_GR_rel,
        "scba_Gless_abs_residual": frozen.result.res_Gless_abs,
        "scba_Gless_rel_residual": frozen.result.res_Gless_rel,
        "scba_sigma_abs_residual": frozen.result.res_Sigma_abs,
        "stationary_energy_min": float(frozen.w[0]),
        "stationary_energy_max": float(frozen.w[-1]),
        "stationary_energy_points": int(len(frozen.w)),
        "sigma_poles": cache.n_sigma_poles,
        "green_poles": cache.n_green_poles,
        "unbiased_green_poles": cache.n_unbiased_green_poles,
        "biased_green_poles": cache.n_biased_green_poles,
        "pole_causal": cache.causal,
        "mpm_tolerance": cache.tolerance,
        "pole_fit_method": cache.fit_method,
        "mpm_sigma_abs_error": cache.max_sigma_abs_error,
        "mpm_sigma_rel_error": cache.max_sigma_rel_error,
        "mpm_sigma_scaled_error": cache.max_sigma_scaled_error,
        "mpm_Gfr_abs_error": cache.max_Gfr_abs_error,
        "mpm_Gbiased_abs_error": cache.max_Gbar_abs_error,
        "mpm_Gfr_scaled_error": cache.max_Gfr_scaled_error,
        "mpm_Gbiased_scaled_error": cache.max_Gbar_scaled_error,
        # Compatibility diagnostic keys.
        "mpm_Gbar_abs_error": cache.max_Gbar_abs_error,
        "mpm_Gbar_scaled_error": cache.max_Gbar_scaled_error,
        "current_energy_min": float(energies[0]),
        "current_energy_max": float(energies[-1]),
        "current_energy_points": int(len(energies)),
        "max_continuity_residual": float(np.max(np.abs(continuity))),
    }
    if extra:
        diagnostics.update(extra)
    return diagnostics


def _upward_boundary_gate(sys: System, cache: PoleCache, energies: np.ndarray):
    # The arrowhead poles represent the analytic eta->0 rational functions.
    # A tiny positive imaginary probe bypasses retarded_argument's finite grid
    # broadening so the raw residue identity is tested like for like.
    analytic_energies = np.asarray(energies, dtype=np.complex128) + 1e-14j
    max_a_abs = 0.0
    max_a_scaled = 0.0
    expected_a = Gfr_R_mpm(sys, cache, analytic_energies)
    band_a = sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(expected_a)
    for alpha in sys.lead_names:
        raw = np.asarray(
            _A_up_residue(
                sys, analytic_energies, 0.0, alpha, cache, enforce_boundary=False
            )
        ).reshape(-1)
        difference = np.abs(raw - expected_a)
        max_a_abs = max(max_a_abs, float(np.max(difference)))
        max_a_scaled = max(
            max_a_scaled, float(np.max(difference / band_a))
        )

    max_c_abs = 0.0
    max_c_scaled = 0.0
    for omega in sorted({-float(sys.w_q), float(sys.w_q)}):
        expected_c = Gfr_R_mpm(sys, cache, analytic_energies + omega)
        raw = np.asarray(
            _C_up_residue(
                sys, omega, analytic_energies, 0.0, cache, enforce_boundary=False
            )
        ).reshape(-1)
        difference = np.abs(raw - expected_c)
        band_c = sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(expected_c)
        max_c_abs = max(max_c_abs, float(np.max(difference)))
        max_c_scaled = max(
            max_c_scaled, float(np.max(difference / band_c))
        )
    if max_a_scaled > 1.0 or max_c_scaled > 1.0:
        raise RuntimeError(
            "Upward residue boundary validation failed: "
            f"A abs/scaled={max_a_abs:.3e}/{max_a_scaled:.3e}, "
            f"C abs/scaled={max_c_abs:.3e}/{max_c_scaled:.3e}."
        )
    return {
        "upward_A0_abs_error": max_a_abs,
        "upward_A0_scaled_error": max_a_scaled,
        "upward_C0_abs_error": max_c_abs,
        "upward_C0_scaled_error": max_c_scaled,
    }


def current_all_down(
    sys: System,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
) -> TransientResult:
    if n_t < 2:
        raise ValueError("n_t must be at least 2 for continuity diagnostics.")
    frozen = sys.frozen_scba()
    _require_protocol(sys, frozen, "downward")
    cache = sys.prepare_poles()
    energies = _current_grid(sys, frozen, omega_int_n_x, omega_int_n_omega)
    n_energy = len(energies)
    times = np.linspace(0.0, t_max, n_t)
    currents = {lead: np.empty(n_t, dtype=float) for lead in sys.lead_names}
    occupation = np.empty(n_t, dtype=float)
    batch_size = max(1, int(sys.current_energy_batch))

    for time_index, time in enumerate(times):
        A_values = {
            lead: np.asarray(A_down(sys, energies, time, lead, cache)).reshape(-1)
            for lead in sys.lead_names
        }
        C_values = {
            -sys.w_q: np.asarray(C_down(sys, -sys.w_q, energies, time, cache)).reshape(-1),
            +sys.w_q: np.asarray(C_down(sys, +sys.w_q, energies, time, cache)).reshape(-1),
        }
        occupation[time_index] = _occupation(
            sys, frozen, energies, A_values, C_values
        )

        for alpha in sys.lead_names:
            current_integrand = np.empty(n_energy, dtype=float)
            for start in range(0, n_energy, batch_size):
                stop = min(start + batch_size, n_energy)
                outer = energies[start:stop]
                psi = _psi_lead_down_batch(
                    sys, cache, outer, energies, time, alpha, A_values
                )
                psi += _psi_ep_down_batch(
                    sys, cache, frozen, outer, energies, time, alpha, C_values
                )
                f_alpha = fermi_dirac(
                    outer, sys.beta_fc(alpha), sys.mu_fc(alpha)
                )
                current_integrand[start:stop] = np.real(
                    linewidth(sys, alpha, outer)
                ) * np.imag(psi + f_alpha * A_values[alpha][start:stop])
            currents[alpha][time_index] = -2.0 * np.trapezoid(
                current_integrand, energies
            ) / (2.0 * np.pi)

        if sys.verbose:
            sys.reporter().info(
                f"Transient current {time_index + 1}/{n_t} | t={time:.6e}"
            )

    continuity = continuity_residual(currents, occupation, times)
    diagnostics = _transient_diagnostics(sys, frozen, cache, energies, continuity)
    return TransientResult(
        t=np.asarray(times),
        currents=currents,
        occupation=occupation,
        continuity_residual=continuity,
        diagnostics=diagnostics,
    )


def current_all_up(
    sys: System,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
) -> TransientResult:
    if n_t < 2:
        raise ValueError("n_t must be at least 2 for continuity diagnostics.")
    frozen = sys.frozen_scba()
    _require_protocol(sys, frozen, "upward")
    cache = sys.prepare_poles()
    energies = _current_grid(sys, frozen, omega_int_n_x, omega_int_n_omega)
    n_energy = len(energies)
    boundary_diagnostics = _upward_boundary_gate(sys, cache, energies)
    times = np.linspace(0.0, t_max, n_t)
    currents = {lead: np.empty(n_t, dtype=float) for lead in sys.lead_names}
    occupation = np.empty(n_t, dtype=float)
    batch_size = max(1, int(sys.current_energy_batch))

    for time_index, time in enumerate(times):
        A_values = {
            lead: np.asarray(A_up(sys, energies, time, lead, cache)).reshape(-1)
            for lead in sys.lead_names
        }
        C_values = {
            -sys.w_q: np.asarray(
                C_up(sys, -sys.w_q, energies, time, cache)
            ).reshape(-1),
            +sys.w_q: np.asarray(
                C_up(sys, +sys.w_q, energies, time, cache)
            ).reshape(-1),
        }
        occupation[time_index] = _occupation(
            sys, frozen, energies, A_values, C_values
        )

        for alpha in sys.lead_names:
            current_integrand = np.empty(n_energy, dtype=float)
            for start in range(0, n_energy, batch_size):
                stop = min(start + batch_size, n_energy)
                outer = energies[start:stop]
                psi = _psi_lead_up_batch(
                    sys, cache, outer, energies, time, alpha, A_values
                )
                psi += _psi_ep_up_batch(
                    sys, cache, frozen, outer, energies, time, alpha, C_values
                )
                f_alpha = fermi_dirac(
                    outer, sys.beta_fc(alpha), sys.mu_fc(alpha)
                )
                current_integrand[start:stop] = np.real(
                    linewidth(sys, alpha, outer)
                ) * np.imag(psi + f_alpha * A_values[alpha][start:stop])
            currents[alpha][time_index] = -2.0 * np.trapezoid(
                current_integrand, energies
            ) / (2.0 * np.pi)

        if sys.verbose:
            sys.reporter().info(
                f"Upward transient current {time_index + 1}/{n_t} | t={time:.6e}"
            )

    continuity = continuity_residual(currents, occupation, times)
    diagnostics = _transient_diagnostics(
        sys, frozen, cache, energies, continuity, boundary_diagnostics
    )
    return TransientResult(
        t=np.asarray(times),
        currents=currents,
        occupation=occupation,
        continuity_residual=continuity,
        diagnostics=diagnostics,
    )


def square_time_grid(t_max: float, n_t: int, duration: float) -> np.ndarray:
    """Return an ``n_t`` grid containing the second switching time exactly."""
    if n_t < 2:
        raise ValueError("n_t must be at least 2 for continuity diagnostics.")
    if not np.isfinite(t_max) or t_max < 0.0:
        raise ValueError("t_max must be finite and nonnegative.")
    if not np.isfinite(duration) or duration < 0.0 or duration > t_max:
        raise ValueError("Square pulse duration must satisfy 0 <= duration <= t_max.")
    if duration == 0.0 or duration == t_max:
        return np.linspace(0.0, t_max, n_t)
    intervals = n_t - 1
    before = int(np.clip(round(intervals * duration / t_max), 1, intervals - 1))
    after = intervals - before
    return np.concatenate((
        np.linspace(0.0, duration, before + 1),
        np.linspace(duration, t_max, after + 1)[1:],
    ))


def current_all_square(
    sys: System,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
    pulse_duration: float | None = None,
) -> TransientResult:
    frozen = sys.frozen_scba()
    _require_protocol(sys, frozen, "square")
    duration = sys.pulse_duration if pulse_duration is None else float(pulse_duration)
    if duration is None:
        raise ValueError("pulse_duration is required for a square pulse.")
    if sys.pulse_duration is not None and not np.isclose(duration, sys.pulse_duration):
        raise RuntimeError(
            "Requested square duration differs from the System cache duration; "
            "construct a matching System."
        )
    cache = sys.prepare_poles()
    energies = _current_grid(sys, frozen, omega_int_n_x, omega_int_n_omega)
    n_energy = len(energies)
    times = square_time_grid(t_max, n_t, duration)
    square_cache = build_square_kernel_cache(sys, energies, duration, cache)
    currents = {lead: np.empty(n_t, dtype=float) for lead in sys.lead_names}
    occupation = np.empty(n_t, dtype=float)
    batch_size = max(1, int(sys.current_energy_batch))
    post_count = int(np.count_nonzero(times > duration))

    # These finite-window histories depend on outer energy batches, but not on
    # the post-turnoff observation time.  Cache them once for the entire run.
    stored_B: dict[tuple[str, str, int], np.ndarray] = {}
    stored_D: dict[tuple[str, float, int], np.ndarray] = {}
    if post_count:
        for start in range(0, n_energy, batch_size):
            stop = min(start + batch_size, n_energy)
            outer = energies[start:stop]
            for alpha in sys.lead_names:
                for beta in sys.lead_names:
                    value = B_up(
                        sys, outer, energies, duration, alpha, beta, cache
                    )
                    stored_B[(alpha, beta, start)] = np.asarray(value).reshape(
                        len(outer), n_energy
                    )
                for omega in sorted({-float(sys.w_q), float(sys.w_q)}):
                    value = D_up(
                        sys, omega, energies, outer, duration, alpha, cache
                    )
                    stored_D[(alpha, omega, start)] = np.asarray(value).reshape(
                        len(outer), n_energy
                    )

    for time_index, time in enumerate(times):
        before_turnoff = time <= duration
        if before_turnoff:
            A_values = {
                lead: np.asarray(A_up(sys, energies, time, lead, cache)).reshape(-1)
                for lead in sys.lead_names
            }
            C_values = {
                omega: np.asarray(C_up(sys, omega, energies, time, cache)).reshape(-1)
                for omega in sorted({-float(sys.w_q), float(sys.w_q)})
            }
        else:
            A_values = {
                lead: np.asarray(
                    A_square(sys, energies, time, lead, cache, square_cache, duration)
                ).reshape(-1)
                for lead in sys.lead_names
            }
            C_values = {
                omega: np.asarray(
                    C_square(sys, omega, energies, time, cache, square_cache, duration)
                ).reshape(-1)
                for omega in sorted({-float(sys.w_q), float(sys.w_q)})
            }
        occupation[time_index] = _occupation(
            sys, frozen, energies, A_values, C_values
        )

        for alpha in sys.lead_names:
            current_integrand = np.empty(n_energy, dtype=float)
            for start in range(0, n_energy, batch_size):
                stop = min(start + batch_size, n_energy)
                outer = energies[start:stop]
                if before_turnoff:
                    psi = _psi_lead_up_batch(
                        sys, cache, outer, energies, time, alpha, A_values
                    )
                    psi += _psi_ep_up_batch(
                        sys, cache, frozen, outer, energies, time, alpha, C_values
                    )
                else:
                    lead_histories = {
                        beta: stored_B[(alpha, beta, start)]
                        for beta in sys.lead_names
                    }
                    phonon_histories = {
                        omega: stored_D[(alpha, omega, start)]
                        for omega in sorted({-float(sys.w_q), float(sys.w_q)})
                    }
                    psi = _psi_lead_square_batch(
                        sys, cache, square_cache, outer, energies, time,
                        alpha, A_values, lead_histories,
                    )
                    psi += _psi_ep_square_batch(
                        sys, cache, frozen, square_cache, outer, energies, time,
                        alpha, C_values, phonon_histories,
                    )
                f_alpha = fermi_dirac(
                    outer, sys.beta_fc(alpha), sys.mu_fc(alpha)
                )
                current_integrand[start:stop] = np.real(
                    linewidth(sys, alpha, outer)
                ) * np.imag(psi + f_alpha * A_values[alpha][start:stop])
            currents[alpha][time_index] = -2.0 * np.trapezoid(
                current_integrand, energies
            ) / (2.0 * np.pi)
        if sys.verbose:
            sys.reporter().info(
                f"Square transient current {time_index + 1}/{n_t} | "
                f"t={time:.6e} | {'pulse' if before_turnoff else 'post-turnoff'}"
            )

    continuity = continuity_residual(currents, occupation, times)
    switch_index = int(np.flatnonzero(np.isclose(times, duration))[0])
    extra = {
        "pulse_duration": float(duration),
        "square_pre_turnoff_points": int(np.count_nonzero(times <= duration)),
        "square_post_turnoff_points": post_count,
        "square_turnoff_index": switch_index,
        "square_residue_n_theta": int(sys.square_residue_n_theta),
        "square_residue_abs_tolerance": sys.square_residue_abs_tol,
        "square_residue_rel_tolerance": sys.square_residue_rel_tol,
        "square_residue_abs_error": square_cache.residue_max_abs_error,
        "square_residue_scaled_error": square_cache.residue_max_scaled_error,
        "square_residue_candidates": square_cache.residue_candidate_count,
        "square_residue_clusters": square_cache.residue_cluster_count,
        "square_residue_cancelled": square_cache.residue_cancelled_count,
        "square_turnoff_A_abs_error": square_cache.turnoff_A_abs_error,
        "square_turnoff_A_scaled_error": square_cache.turnoff_A_scaled_error,
        "square_turnoff_C_abs_error": square_cache.turnoff_C_abs_error,
        "square_turnoff_C_scaled_error": square_cache.turnoff_C_scaled_error,
    }
    diagnostics = _transient_diagnostics(
        sys, frozen, cache, energies, continuity, extra
    )
    return TransientResult(
        t=np.asarray(times), currents=currents, occupation=occupation,
        continuity_residual=continuity, diagnostics=diagnostics,
    )


def current_all(
    sys: System,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
    protocol: str | None = None,
    pulse_duration: float | None = None,
) -> TransientResult:
    selected = sys.pulse_protocol if protocol is None else protocol
    if selected == "downward":
        return current_all_down(
            sys, t_max, n_t, omega_int_n_x, omega_int_n_omega
        )
    if selected == "upward":
        return current_all_up(
            sys, t_max, n_t, omega_int_n_x, omega_int_n_omega
        )
    if selected == "square":
        return current_all_square(
            sys, t_max, n_t, omega_int_n_x, omega_int_n_omega,
            pulse_duration=pulse_duration,
        )
    raise ValueError(f"Unknown pulse protocol {selected!r}.")


def current_alpha(
    sys: System,
    alpha: Lead,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
    protocol: str | None = None,
    pulse_duration: float | None = None,
):
    if alpha not in sys.leads:
        raise KeyError(f"Unknown lead {alpha!r}; expected one of {sys.lead_names!r}.")
    result = current_all(
        sys,
        t_max=t_max,
        n_t=n_t,
        omega_int_n_x=omega_int_n_x,
        omega_int_n_omega=omega_int_n_omega,
        protocol=protocol,
        pulse_duration=pulse_duration,
    )
    return result.t, result.currents[alpha]


def current_alpha_down(sys: System, alpha: Lead, *args, **kwargs):
    return current_alpha(sys, alpha, *args, protocol="downward", **kwargs)


def current_alpha_up(sys: System, alpha: Lead, *args, **kwargs):
    return current_alpha(sys, alpha, *args, protocol="upward", **kwargs)


def current_alpha_square(sys: System, alpha: Lead, *args, **kwargs):
    return current_alpha(sys, alpha, *args, protocol="square", **kwargs)
