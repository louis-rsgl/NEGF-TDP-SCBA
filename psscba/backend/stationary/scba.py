from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

from psscba.backend.model.distributions import bose_einstein, fermi_dirac


@dataclass(frozen=True)
class InstalledKernelData:
    """Immutable stationary interaction kernel installed in the pulse solver."""

    w: np.ndarray
    G_reference_R: np.ndarray
    G_reference_less: np.ndarray
    G_kernel_less: np.ndarray
    Sigma_ep_R: np.ndarray
    Sigma_ep_less: np.ndarray
    Sigma_H: float
    Sigma_ep_dyn_R: np.ndarray
    Gamma_ep: np.ndarray
    N0: float
    pulse_protocol: str
    reference_is_biased: bool
    result: "SolverResult"

@dataclass
class SolverResult:
    converged: bool
    n_iter: int
    res_GR_abs: float
    res_GR_rel: float
    res_Gless_abs: float
    res_Gless_rel: float
    res_Sigma_abs: float = np.inf
    history_res_GR_abs: list[float] = field(default_factory=list)
    history_res_GR_rel: list[float] = field(default_factory=list)
    history_res_Gless_abs: list[float] = field(default_factory=list)
    history_res_Gless_rel: list[float] = field(default_factory=list)
    history_res_Sigma_abs: list[float] = field(default_factory=list)


class SCBAConvergenceError(RuntimeError):
    def __init__(self, result: SolverResult):
        self.result = result
        super().__init__(
            "Stationary SCBA did not converge after "
            f"{result.n_iter} iterations (GR={result.res_GR_abs:.3e}, "
            f"G<={result.res_Gless_abs:.3e})."
        )


def _readonly(values, dtype=np.complex128) -> np.ndarray:
    out = np.array(values, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


class Solver:
    def __init__(self, sys, w_min: float, w_max: float, n_w: int):
        if n_w < 3:
            raise ValueError("SCBA requires at least three frequency points.")
        self.sys = sys
        self.w = np.linspace(w_min, w_max, n_w, dtype=float)
        self.dw = float(self.w[1] - self.w[0])
        self.max_iter = sys.scba_max_iter
        self.tol_abs = sys.scba_tol_abs
        self.tol_rel = sys.scba_tol_rel
        self.mixing = sys.scba_mixing
        self.min_iter = sys.scba_min_iter
        self.verbose = sys.verbose

        self.GR_values: Optional[np.ndarray] = None
        self.Gless_values: Optional[np.ndarray] = None
        self.Sigma_ep_R_values: Optional[np.ndarray] = None
        self.Sigma_ep_less_values: Optional[np.ndarray] = None
        self.Gamma_ep_values: Optional[np.ndarray] = None
        self.Sigma_H: float = 0.0
        self.result: Optional[SolverResult] = None
        self.installed_kernel: Optional[InstalledKernelData] = None

        self.history_res_GR_abs: list[float] = []
        self.history_res_GR_rel: list[float] = []
        self.history_res_Gless_abs: list[float] = []
        self.history_res_Gless_rel: list[float] = []
        self.history_res_Sigma_abs: list[float] = []

    @property
    def N0(self) -> float:
        if self.sys.N0 is not None:
            value = float(self.sys.N0)
        elif self.sys.g_q == 0.0 and self.sys.w_q == 0.0:
            value = 0.0
        else:
            value = float(np.real(bose_einstein(
                self.sys.w_q, self.sys.beta_ph, self.sys.mu_ph
            )))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"N0 must be finite and non-negative; got {value}.")
        return value

    def linewidth(self, lead, omega):
        shifted = np.asarray(omega) - self.sys.reference_lead_shift(lead)
        return self.sys.Gamma0(lead) * self.sys.W**2 / (
            shifted**2 + self.sys.W**2
        )

    def sigma_lead_R(self, omega):
        omega = np.asarray(omega, dtype=np.complex128)
        out = np.zeros_like(omega)
        for lead in self.sys.lead_names:
            out += 0.5 * self.sys.Gamma0(lead) * self.sys.W / (
                omega - self.sys.reference_lead_shift(lead) + 1j * self.sys.W
            )
        return out

    def sigma_lead_less(self, omega):
        omega = np.asarray(omega, dtype=float)
        out = np.zeros_like(omega, dtype=np.complex128)
        for lead in self.sys.lead_names:
            shifted = omega - self.sys.reference_lead_shift(lead)
            out += 1j * fermi_dirac(
                shifted, self.sys.beta_fc(lead), self.sys.mu_fc(lead)
            ) * self.linewidth(lead, omega)
        return out

    def initialize(self) -> None:
        lead_R = self.sigma_lead_R(self.w)
        GR0 = 1.0 / (
            self.w - self.sys.e_0 - self.sys.reference_device_shift
            - lead_R + 1j * self.sys.ETA
        )
        lead_less = self.sigma_lead_less(self.w)
        Gless0 = GR0 * lead_less * np.conjugate(GR0)
        self.GR_values = np.asarray(GR0, dtype=np.complex128)
        self.Gless_values = np.asarray(Gless0, dtype=np.complex128)
        self.Sigma_ep_R_values = np.zeros_like(self.GR_values)
        self.Sigma_ep_less_values = np.zeros_like(self.GR_values)
        self.Gamma_ep_values = np.zeros_like(self.w)
        self.Sigma_H = 0.0
        self.result = None
        self.installed_kernel = None
        self.history_res_GR_abs.clear()
        self.history_res_GR_rel.clear()
        self.history_res_Gless_abs.clear()
        self.history_res_Gless_rel.clear()
        self.history_res_Sigma_abs.clear()

    def _require_initialized(self) -> None:
        if self.GR_values is None or self.Gless_values is None:
            raise RuntimeError("Solver not initialized. Call initialize() or solve().")

    def _interp_real_grid(self, values: np.ndarray, x, *, retarded: bool):
        x_arr = np.asarray(x, dtype=np.complex128)
        if np.any(np.abs(x_arr.imag) > 1e-14):
            raise ValueError(
                "Grid SCBA functions are defined only on the real axis; use the "
                "installed-kernel MPM propagator for complex frequencies."
            )
        xr = x_arr.real
        real = np.interp(xr, self.w, values.real, left=np.nan, right=np.nan)
        imag = np.interp(xr, self.w, values.imag, left=np.nan, right=np.nan)
        out = real + 1j * imag
        outside = (xr < self.w[0]) | (xr > self.w[-1])
        if np.any(outside):
            tail = 1.0 / (x_arr + 1j * self.sys.ETA) if retarded else 0.0j
            out = np.where(outside, tail, out)
        return out.item() if out.ndim == 0 else out

    def GR(self, e):
        self._require_initialized()
        return self._interp_real_grid(self.GR_values, e, retarded=True)

    def Gless(self, e):
        self._require_initialized()
        return self._interp_real_grid(self.Gless_values, e, retarded=False)

    def GA(self, e):
        return np.conjugate(self.GR(e))

    @staticmethod
    def _hilbert_pv(gamma: np.ndarray, pad_factor: int = 4) -> np.ndarray:
        """Return the uniform-grid trapezoidal Cauchy principal value.

        The spacing cancels analytically between ``dw'`` and ``w-w'``.  Using
        a linear (not circular) FFT convolution makes this discretization
        consistent with the finite-grid spectral integral used for MiniPole's
        auxiliary samples. ``pad_factor`` is retained for API compatibility.
        """
        n = len(gamma)
        if n < 2:
            return np.zeros(n, dtype=float)
        weighted = np.asarray(gamma, dtype=float).copy()
        weighted[[0, -1]] *= 0.5
        offsets = np.arange(-(n - 1), n, dtype=float)
        kernel = np.zeros_like(offsets)
        nonzero = offsets != 0.0
        kernel[nonzero] = 1.0 / (2.0 * np.pi * offsets[nonzero])
        principal_value = fftconvolve(weighted, kernel, mode="same")
        # Restore the finite contribution of the two half-cells adjacent to
        # the omitted singular node.  For a local linear interpolant it is
        # exactly -f'(w_n) dw/(2 pi).
        local_correction = -np.gradient(weighted) / (2.0 * np.pi)
        return principal_value + local_correction

    def compute_ep_self_energies(self):
        self._require_initialized()
        assert self.GR_values is not None and self.Gless_values is not None
        n0 = self.N0
        g2 = self.sys.g_q**2
        wm = self.w - self.sys.w_q
        wp = self.w + self.sys.w_q

        gl_m = self.Gless(wm)
        gl_p = self.Gless(wp)
        sigma_less = g2 * ((n0 + 1.0) * gl_p + n0 * gl_m)

        greater = self.Gless_values + self.GR_values - np.conjugate(self.GR_values)
        gg_m = self._interp_real_grid(greater, wm, retarded=False)
        gg_p = self._interp_real_grid(greater, wp, retarded=False)
        sigma_greater = g2 * ((n0 + 1.0) * gg_m + n0 * gg_p)

        gamma_complex = 1j * (sigma_greater - sigma_less)
        imag_noise = float(np.max(np.abs(np.imag(gamma_complex))))
        scale = max(1.0, float(np.max(np.abs(gamma_complex))))
        if imag_noise > 1e-8 * scale:
            raise RuntimeError(
                f"Electron-phonon spectral density is not real (noise={imag_noise:.3e})."
            )
        gamma_ep = np.real(gamma_complex)
        sigma_dyn_R = self._hilbert_pv(gamma_ep) - 0.5j * gamma_ep

        occupation_complex = -1j * np.trapezoid(self.Gless_values, self.w) / (2.0 * np.pi)
        if abs(occupation_complex.imag) > 1e-8 * max(1.0, abs(occupation_complex.real)):
            raise RuntimeError("Stationary dot occupation is not real within tolerance.")
        occupation = float(occupation_complex.real)
        if self.sys.g_q == 0.0:
            sigma_H = 0.0
        elif self.sys.w_q == 0.0:
            raise ValueError("w_q must be nonzero when g_q is nonzero.")
        else:
            sigma_H = -2.0 * g2 * occupation / self.sys.w_q
        sigma_R = sigma_dyn_R + sigma_H
        return sigma_R, sigma_less, sigma_H, sigma_dyn_R, gamma_ep

    def update_trials(self):
        sigma_R, sigma_less, sigma_H, sigma_dyn_R, gamma_ep = (
            self.compute_ep_self_energies()
        )
        GR_trial = 1.0 / (
            self.w - self.sys.e_0 - self.sys.reference_device_shift
            - self.sigma_lead_R(self.w) - sigma_R + 1j * self.sys.ETA
        )
        total_less = self.sigma_lead_less(self.w) + sigma_less
        Gless_trial = GR_trial * total_less * np.conjugate(GR_trial)
        return GR_trial, Gless_trial, sigma_R, sigma_less, sigma_H, sigma_dyn_R, gamma_ep

    @staticmethod
    def linear_mix(old: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * old + alpha * new

    def integrated_l2_norm(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.sum(np.abs(x) ** 2) * self.dw))

    def absolute_residual(self, current: np.ndarray, trial: np.ndarray) -> float:
        return self.integrated_l2_norm(current - trial)

    def relative_residual(self, current: np.ndarray, trial: np.ndarray) -> float:
        return self.absolute_residual(current, trial) / max(
            self.integrated_l2_norm(current), float(self.sys.ETA)
        )

    def _validate_spectral_state(self) -> None:
        assert self.GR_values is not None
        spectral = -2.0 * np.imag(self.GR_values)
        spectral_weight = float(np.trapezoid(spectral, self.w) / (2.0 * np.pi))
        spectral_peak = max(float(np.max(np.abs(spectral))), np.finfo(float).tiny)
        edge_fraction = float(
            max(abs(spectral[0]), abs(spectral[-1])) / spectral_peak
        )
        if abs(spectral_weight - 1.0) > self.sys.scba_spectral_weight_tol:
            raise RuntimeError(
                "Stationary spectral sum-rule failure: "
                f"integral={spectral_weight:.6e}, expected 1 within "
                f"{self.sys.scba_spectral_weight_tol:.3e}."
            )
        if edge_fraction > self.sys.scba_edge_fraction_tol:
            raise RuntimeError(
                "Stationary frequency-window boundary failure: spectral edge/peak="
                f"{edge_fraction:.6e} exceeds {self.sys.scba_edge_fraction_tol:.3e}."
            )

    def solve_weak_born(self) -> SolverResult:
        """Strict O(g^2) installed kernel built from the no-phonon reference."""
        if self.GR_values is None or self.Gless_values is None:
            self.initialize()
        assert self.GR_values is not None and self.Gless_values is not None

        # compute_ep_self_energies reads the current arrays, which at this point
        # are precisely the protocol's no-phonon stationary reference.
        kernel_less = np.array(self.Gless_values, copy=True)
        sigma_R, sigma_less, sigma_H, sigma_dyn_R, gamma_ep = (
            self.compute_ep_self_energies()
        )
        GR_dressed = 1.0 / (
            self.w - self.sys.e_0 - self.sys.reference_device_shift
            - self.sigma_lead_R(self.w) - sigma_R + 1j * self.sys.ETA
        )
        total_less = self.sigma_lead_less(self.w) + sigma_less
        Gless_dressed = GR_dressed * total_less * np.conjugate(GR_dressed)
        self.GR_values = np.asarray(GR_dressed, dtype=np.complex128)
        self.Gless_values = 1j * np.imag(Gless_dressed)
        self.result = SolverResult(
            converged=True,
            n_iter=1,
            res_GR_abs=0.0,
            res_GR_rel=0.0,
            res_Gless_abs=0.0,
            res_Gless_rel=0.0,
            res_Sigma_abs=0.0,
        )
        self._validate_spectral_state()
        self.Sigma_ep_R_values = sigma_R
        self.Sigma_ep_less_values = sigma_less
        self.Sigma_H = float(sigma_H)
        self.Gamma_ep_values = gamma_ep
        self.installed_kernel = InstalledKernelData(
            w=_readonly(self.w, float),
            G_reference_R=_readonly(self.GR_values),
            G_reference_less=_readonly(self.Gless_values),
            G_kernel_less=_readonly(kernel_less),
            Sigma_ep_R=_readonly(sigma_R),
            Sigma_ep_less=_readonly(sigma_less),
            Sigma_H=float(sigma_H),
            Sigma_ep_dyn_R=_readonly(sigma_dyn_R),
            Gamma_ep=_readonly(gamma_ep, float),
            N0=self.N0,
            pulse_protocol=self.sys.pulse_protocol,
            reference_is_biased=self.sys.reference_is_biased,
            result=self.result,
        )
        return self.result

    def solve(self) -> SolverResult:
        if self.sys.scba_mode == "weak_born":
            return self.solve_weak_born()
        if self.sys.scba_mode != "self_consistent":
            raise ValueError(
                "scba_mode must be 'self_consistent' or 'weak_born'; got "
                f"{self.sys.scba_mode!r}."
            )
        if self.GR_values is None or self.Gless_values is None:
            self.initialize()
        assert self.GR_values is not None and self.Gless_values is not None

        rep = self.sys.reporter()
        if self.verbose:
            rep.section(
                "Stationary "
                f"{'biased' if self.sys.reference_is_biased else 'unbiased'} "
                "SCBA solve"
            )
            rep.info(
                f"grid={len(self.w)} [{self.w[0]}, {self.w[-1]}] | "
                f"N0={self.N0:.6e} | mixing={self.mixing}"
            )

        previous_sigma = np.zeros_like(self.w, dtype=np.complex128)
        diis_x: list[np.ndarray] = []
        diis_trial: list[np.ndarray] = []
        diis_residual: list[np.ndarray] = []
        converged = False
        residuals = (np.inf, np.inf, np.inf, np.inf, np.inf)
        for it in range(1, self.max_iter + 1):
            trial = self.update_trials()
            GR_trial, Gless_trial, sigma_R = trial[0], trial[1], trial[2]
            residuals = (
                self.absolute_residual(self.GR_values, GR_trial),
                self.relative_residual(self.GR_values, GR_trial),
                self.absolute_residual(self.Gless_values, Gless_trial),
                self.relative_residual(self.Gless_values, Gless_trial),
                self.absolute_residual(previous_sigma, sigma_R),
            )
            self.history_res_GR_abs.append(residuals[0])
            self.history_res_GR_rel.append(residuals[1])
            self.history_res_Gless_abs.append(residuals[2])
            self.history_res_Gless_rel.append(residuals[3])
            self.history_res_Sigma_abs.append(residuals[4])

            current_vector = np.concatenate((self.GR_values, self.Gless_values))
            trial_vector = np.concatenate((GR_trial, Gless_trial))
            residual_vector = trial_vector - current_vector
            diis_x.append(current_vector.copy())
            diis_trial.append(trial_vector.copy())
            diis_residual.append(residual_vector.copy())
            if len(diis_x) > self.sys.scba_diis_size:
                diis_x.pop(0)
                diis_trial.pop(0)
                diis_residual.pop(0)

            next_vector = current_vector + self.mixing * residual_vector
            if it >= self.sys.scba_diis_start and len(diis_residual) >= 2:
                count = len(diis_residual)
                pulay = np.empty((count + 1, count + 1), dtype=float)
                pulay[-1, :count] = 1.0
                pulay[:count, -1] = 1.0
                pulay[-1, -1] = 0.0
                residual_scale = max(
                    float(np.real(np.vdot(value, value)))
                    for value in diis_residual
                )
                for row in range(count):
                    for column in range(count):
                        pulay[row, column] = float(np.real(np.vdot(
                            diis_residual[row], diis_residual[column]
                        ))) / max(residual_scale, np.finfo(float).tiny)
                pulay[:count, :count] += (
                    self.sys.scba_diis_regularization * np.eye(count)
                )
                rhs = np.zeros(count + 1)
                rhs[-1] = 1.0
                try:
                    coefficients = np.linalg.solve(pulay, rhs)[:count]
                    mixed_current = sum(
                        coefficient * value
                        for coefficient, value in zip(coefficients, diis_x)
                    )
                    mixed_trial = sum(
                        coefficient * value
                        for coefficient, value in zip(coefficients, diis_trial)
                    )
                    damping = self.sys.scba_diis_damping
                    candidate = (
                        (1.0 - damping) * mixed_current
                        + damping * mixed_trial
                    )
                    if np.all(np.isfinite(candidate)):
                        next_vector = candidate
                except np.linalg.LinAlgError:
                    pass

            split = len(self.w)
            self.GR_values = next_vector[:split]
            # For a scalar level G<(w) is anti-Hermitian and therefore purely
            # imaginary on the real axis.  Pulay combinations can accumulate a
            # small unphysical real component; project it back onto the exact
            # Keldysh symmetry before evaluating the next SCBA kernel.
            self.Gless_values = 1j * np.imag(next_vector[split:])
            previous_sigma = np.asarray(sigma_R)
            if self.verbose and (it == 1 or it % 25 == 0):
                rep.info(
                    f"[SCBA] iter={it} GR={residuals[0]:.3e}/{residuals[1]:.3e} "
                    f"G<={residuals[2]:.3e}/{residuals[3]:.3e}"
                )

            gr_ok = residuals[0] < self.tol_abs and residuals[1] < self.tol_rel
            gl_ok = residuals[2] < self.tol_abs and residuals[3] < self.tol_rel
            if it >= self.min_iter and gr_ok and gl_ok:
                converged = True
                break

        self.result = SolverResult(
            converged=converged,
            n_iter=it,
            res_GR_abs=residuals[0],
            res_GR_rel=residuals[1],
            res_Gless_abs=residuals[2],
            res_Gless_rel=residuals[3],
            res_Sigma_abs=residuals[4],
            history_res_GR_abs=self.history_res_GR_abs.copy(),
            history_res_GR_rel=self.history_res_GR_rel.copy(),
            history_res_Gless_abs=self.history_res_Gless_abs.copy(),
            history_res_Gless_rel=self.history_res_Gless_rel.copy(),
            history_res_Sigma_abs=self.history_res_Sigma_abs.copy(),
        )
        if not converged:
            raise SCBAConvergenceError(self.result)

        # Re-evaluate the installed kernel from the final mixed state.
        kernel_less = np.array(self.Gless_values, copy=True)
        sigma_R, sigma_less, sigma_H, sigma_dyn_R, gamma_ep = (
            self.compute_ep_self_energies()
        )
        self._validate_spectral_state()
        self.Sigma_ep_R_values = sigma_R
        self.Sigma_ep_less_values = sigma_less
        self.Sigma_H = float(sigma_H)
        self.Gamma_ep_values = gamma_ep
        self.installed_kernel = InstalledKernelData(
            w=_readonly(self.w, float),
            G_reference_R=_readonly(self.GR_values),
            G_reference_less=_readonly(self.Gless_values),
            G_kernel_less=_readonly(kernel_less),
            Sigma_ep_R=_readonly(sigma_R),
            Sigma_ep_less=_readonly(sigma_less),
            Sigma_H=float(sigma_H),
            Sigma_ep_dyn_R=_readonly(sigma_dyn_R),
            Gamma_ep=_readonly(gamma_ep, float),
            N0=self.N0,
            pulse_protocol=self.sys.pulse_protocol,
            reference_is_biased=self.sys.reference_is_biased,
            result=self.result,
        )
        if self.verbose:
            rep.info(f"SCBA converged in {it} iterations")
        return self.result
