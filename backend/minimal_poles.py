from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from backend.SCBA import FrozenSCBA


MINIPOLE_COMMIT = "15e4a541d652d2584fd4680413d11b5571f78d9b"


@dataclass(frozen=True)
class PoleCache:
    zeta: np.ndarray
    weights: np.ndarray
    xi_unbiased: np.ndarray
    residues_unbiased: np.ndarray
    xi_biased: np.ndarray
    residues_biased: np.ndarray
    pulse_protocol: str
    sigma_H: float
    tolerance: float
    max_sigma_abs_error: float
    max_sigma_rel_error: float
    max_sigma_scaled_error: float
    max_Gfr_abs_error: float
    max_Gbar_abs_error: float
    max_Gfr_scaled_error: float
    max_Gbar_scaled_error: float
    causal: bool
    fit_method: str = "minipole"
    fit_terms: int = 0
    fit_converged: bool = True

    @property
    def n_sigma_poles(self) -> int:
        return len(self.zeta)

    @property
    def n_green_poles(self) -> int:
        return len(self.xi)

    @property
    def xi(self) -> np.ndarray:
        """Green poles active for the selected switching protocol."""
        return (
            self.xi_biased
            if self.pulse_protocol == "upward"
            else self.xi_unbiased
        )

    @property
    def residues(self) -> np.ndarray:
        """Green residues active for the selected switching protocol."""
        return (
            self.residues_unbiased
            if self.pulse_protocol != "upward"
            else self.residues_biased
        )

    @property
    def n_unbiased_green_poles(self) -> int:
        return len(self.xi_unbiased)

    @property
    def n_biased_green_poles(self) -> int:
        return len(self.xi_biased)


def _readonly(values) -> np.ndarray:
    out = np.array(values, dtype=np.complex128, copy=True).reshape(-1)
    out.setflags(write=False)
    return out


def retarded_argument(z, eta: float):
    arr = np.asarray(z, dtype=np.complex128)
    return np.where(np.abs(arr.imag) <= 1e-15, arr + 1j * eta, arr)


def sigma_mpm(z, zeta: np.ndarray, weights: np.ndarray):
    z_arr = np.asarray(z, dtype=np.complex128)
    if len(zeta) == 0:
        return np.zeros_like(z_arr, dtype=np.complex128)
    return np.sum(
        weights / (z_arr[..., np.newaxis] - zeta),
        axis=-1,
    )


def sigma_lead_unbiased_R(sys, z):
    return 0.5 * sum(sys.Gamma0(a) for a in sys.lead_names) * sys.W / (
        np.asarray(z, dtype=np.complex128) + 1j * sys.W
    )


def sigma_lead_biased_R(sys, z):
    z = np.asarray(z, dtype=np.complex128)
    out = np.zeros_like(z)
    for lead in sys.lead_names:
        out += 0.5 * sys.Gamma0(lead) * sys.W / (
            z - sys.Delta(lead) + 1j * sys.W
        )
    return out


def Gfr_R_mpm(sys, cache: PoleCache, z):
    zr = retarded_argument(z, sys.ETA)
    return 1.0 / (
        zr - sys.e_0 - cache.sigma_H - sigma_lead_unbiased_R(sys, zr)
        - sigma_mpm(zr, cache.zeta, cache.weights)
    )


def Gbiased_R_mpm(sys, cache: PoleCache, z):
    zr = retarded_argument(z, sys.ETA)
    return 1.0 / (
        zr - sys.e_0 - sys.DELTA - cache.sigma_H
        - sigma_lead_biased_R(sys, zr)
        - sigma_mpm(zr, cache.zeta, cache.weights)
    )


def Gbar_R_mpm(sys, cache: PoleCache, z):
    """Compatibility alias for the biased frozen-kernel propagator."""
    return Gbiased_R_mpm(sys, cache, z)


def auxiliary_self_energy_samples(
    frozen: FrozenSCBA, n_iw: int, beta_fit: float
) -> tuple[np.ndarray, np.ndarray]:
    if n_iw < 4:
        raise ValueError("mpm_n_iw must be at least 4.")
    if beta_fit <= 0.0:
        raise ValueError("mpm_beta_fit must be positive.")
    nu = (2 * np.arange(n_iw, dtype=float) + 1.0) * np.pi / beta_fit
    denominator = 1j * nu[:, None] - frozen.w[None, :]
    samples = np.trapezoid(
        frozen.Gamma_ep[None, :] / denominator,
        frozen.w,
        axis=1,
    ) / (2.0 * np.pi)
    return nu, samples


def _lead_pole_groups(sys, biased: bool) -> list[tuple[float, float]]:
    """Return (shift, numerator) pairs for distinct Lorentzian lead poles."""
    grouped: dict[float, float] = {}
    for lead in sys.lead_names:
        shift = float(sys.Delta(lead)) if biased else 0.0
        grouped[shift] = grouped.get(shift, 0.0) + 0.5 * sys.Gamma0(lead) * sys.W
    return sorted(
        ((shift, numerator) for shift, numerator in grouped.items() if numerator != 0.0),
        key=lambda item: item[0],
    )


def green_poles_from_sigma_mpm(sys, sigma_H, zeta, weights, biased: bool = False):
    m = len(zeta)
    lead_groups = _lead_pole_groups(sys, biased)
    n_leads = len(lead_groups)
    matrix = np.zeros((m + n_leads + 1, m + n_leads + 1), dtype=np.complex128)
    matrix[0, 0] = sys.e_0 + sigma_H + (sys.DELTA if biased else 0.0)
    for index, (shift, numerator) in enumerate(lead_groups, start=1):
        matrix[0, index] = numerator
        matrix[index, 0] = 1.0
        matrix[index, index] = shift - 1j * sys.W
    if m:
        start = n_leads + 1
        matrix[0, start:] = weights
        matrix[start:, 0] = 1.0
        matrix[start:, start:] = np.diag(zeta)
    return np.linalg.eigvals(matrix)


def green_residues(sys, xi, zeta, weights, biased: bool = False):
    out = np.empty_like(xi, dtype=np.complex128)
    for index, pole in enumerate(xi):
        derivative = 1.0 + sum(
            numerator / (pole - shift + 1j * sys.W) ** 2
            for shift, numerator in _lead_pole_groups(sys, biased)
        )
        if len(zeta):
            derivative += np.sum(weights / (pole - zeta) ** 2)
        out[index] = 1.0 / derivative
    return out


def _filter_green_poles(sys, xi, residues):
    scale = max(1.0, float(np.sum(np.abs(residues))))
    keep = np.abs(residues) > sys.pole_residue_tol * scale
    xi = np.asarray(xi)[keep]
    residues = np.asarray(residues)[keep]
    if len(xi) == 0:
        raise RuntimeError("All Green-function poles were filtered as negligible.")
    if np.any(xi.imag > sys.pole_causality_tol):
        bad = xi[xi.imag > sys.pole_causality_tol]
        raise RuntimeError(f"Causality failure: upper-half-plane Green poles {bad!r}")
    if len(xi) > 1:
        distances = np.abs(xi[:, None] - xi[None, :])
        distances[np.diag_indices_from(distances)] = np.inf
        scaled_tol = sys.pole_merge_tol * max(1.0, float(np.max(np.abs(xi))))
        if float(np.min(distances)) < scaled_tol:
            raise RuntimeError("Nearly degenerate Green poles make simple residues unstable.")
    order = np.argsort(xi.real)
    return xi[order], residues[order]


def _errors(target, approximation, floor: float):
    difference = np.abs(approximation - target)
    return (
        float(np.max(difference)),
        float(np.max(difference / np.maximum(np.abs(target), floor))),
    )


def _scaled_error(target, approximation, abs_tol: float, rel_tol: float):
    """Maximum error in units of the pointwise atol + rtol acceptance band."""
    denominator = abs_tol + rel_tol * np.abs(target)
    return float(np.max(np.abs(approximation - target) / denominator))


def _fit_with_minipole(frozen: FrozenSCBA, sys, tolerance: float):
    try:
        from mini_pole import MiniPole
    except ImportError as exc:
        raise RuntimeError(
            "MiniPole is required for g_q != 0. Install the pinned dependency "
            f"from requirements.txt. Import failed with: {exc}"
        ) from exc

    nu, samples = auxiliary_self_energy_samples(
        frozen, sys.mpm_n_iw, sys.mpm_beta_fit
    )
    fit = MiniPole(
        samples,
        nu,
        err=tolerance,
        symmetry=False,
        plane="z",
        compute_const=False,
    )
    zeta = np.asarray(fit.pole_location, dtype=np.complex128).reshape(-1)
    weights = np.asarray(fit.pole_weight, dtype=np.complex128).reshape(-1)
    if len(zeta) != len(weights) or len(zeta) == 0:
        raise RuntimeError("MiniPole returned an empty or inconsistent scalar fit.")
    # Analytic continuation can create tiny pole-zero artefacts above the real
    # axis.  Remove one only when its entire real-axis contribution fits inside
    # the configured reconstruction error band; physical upper-plane poles are
    # still fatal.
    upper = zeta.imag > sys.pole_causality_tol
    if np.any(upper):
        target = frozen.Sigma_ep_dyn_R
        denominator = (
            sys.mpm_fit_abs_tol
            + sys.mpm_fit_rel_tol * np.abs(target)
        )
        negligible = np.zeros(len(zeta), dtype=bool)
        w_eval = frozen.w + 1j * sys.ETA
        for index in np.flatnonzero(upper):
            contribution = weights[index] / (w_eval - zeta[index])
            negligible[index] = np.max(np.abs(contribution) / denominator) <= 1.0
        keep = ~negligible
        zeta = zeta[keep]
        weights = weights[keep]
    if np.any(zeta.imag > sys.pole_causality_tol):
        raise RuntimeError("MiniPole returned upper-half-plane self-energy poles.")
    return zeta, weights


def _aaa_term_schedule(initial: int, maximum: int, growth: float) -> tuple[int, ...]:
    """Return a finite geometric work schedule ending exactly at ``maximum``."""
    if initial < 2:
        raise ValueError("mpm_aaa_initial_terms must be at least 2.")
    if maximum < initial:
        raise ValueError(
            "mpm_aaa_max_terms must be greater than or equal to "
            "mpm_aaa_initial_terms."
        )
    if not np.isfinite(growth) or growth <= 1.0:
        raise ValueError("mpm_aaa_growth_factor must be finite and greater than 1.")

    values = [int(initial)]
    while values[-1] < maximum:
        following = max(values[-1] + 1, int(np.ceil(values[-1] * growth)))
        values.append(min(following, int(maximum)))
    return tuple(values)


def _fit_with_causal_aaa(
    frozen: FrozenSCBA,
    sys,
    *,
    max_terms: int,
):
    """Discover real-axis poles with AAA, then causally refit their residues.

    AAA is used only for candidate locations. Upper-half-plane candidates and
    its finite-interval constant are discarded; residues are recomputed from
    scratch on lower-half-plane poles against the frozen retarded self-energy.
    The resulting expansion is therefore analytic in the upper half-plane and
    is subjected to the same Sigma/G/Green-pole validation as MiniPole.
    """
    try:
        from scipy.interpolate import AAA
    except ImportError as exc:
        raise RuntimeError("SciPy AAA fallback is unavailable.") from exc

    effective_max_terms = min(int(max_terms), len(frozen.w) - 1)
    if effective_max_terms < 2:
        raise RuntimeError("The stationary grid is too small for causal AAA.")
    # SciPy warns when it reaches the work limit.  That does not by itself
    # invalidate our causal refit: the full-grid Sigma/G validation below is
    # the scientific acceptance criterion.  Record the state in PoleCache and
    # avoid emitting one warning for every adaptive attempt.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"AAA failed to converge within .* iterations\.",
            category=RuntimeWarning,
        )
        fit = AAA(
            frozen.w,
            frozen.Sigma_ep_dyn_R,
            rtol=sys.mpm_aaa_rtol,
            max_terms=effective_max_terms,
            clean_up=True,
        )
    aaa_errors = np.asarray(fit.errors, dtype=float)
    aaa_terms = int(len(aaa_errors))
    target_scale = float(np.max(np.abs(frozen.Sigma_ep_dyn_R)))
    aaa_converged = bool(
        aaa_terms > 0
        and aaa_errors[-1] <= sys.mpm_aaa_rtol * target_scale
    )
    candidates = np.asarray(fit.poles(), dtype=np.complex128).reshape(-1)
    zeta = candidates[candidates.imag < -sys.pole_causality_tol]
    if len(zeta) == 0:
        raise RuntimeError("AAA returned no lower-half-plane pole candidates.")
    design = 1.0 / (
        frozen.w[:, np.newaxis] + 1j * sys.ETA - zeta[np.newaxis, :]
    )
    w_eval = frozen.w + 1j * sys.ETA
    direct_Gfr = 1.0 / (
        w_eval - sys.e_0 - frozen.Sigma_H
        - sigma_lead_unbiased_R(sys, w_eval) - frozen.Sigma_ep_dyn_R
    )
    direct_Gbar = 1.0 / (
        w_eval - sys.e_0 - sys.DELTA - frozen.Sigma_H
        - sigma_lead_biased_R(sys, w_eval) - frozen.Sigma_ep_dyn_R
    )
    sigma_band = (
        sys.mpm_fit_abs_tol
        + sys.mpm_fit_rel_tol * np.abs(frozen.Sigma_ep_dyn_R)
    )
    gfr_band = (
        sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(direct_Gfr)
    )
    gbar_band = (
        sys.mpm_fit_abs_tol + sys.mpm_green_rel_tol * np.abs(direct_Gbar)
    )
    sensitivity = np.maximum.reduce((
        1.0 / sigma_band,
        np.abs(direct_Gfr) ** 2 / gfr_band,
        np.abs(direct_Gbar) ** 2 / gbar_band,
    ))
    sensitivity = np.minimum(sensitivity, np.quantile(sensitivity, 0.999))
    weights = np.linalg.lstsq(
        design * sensitivity[:, np.newaxis],
        frozen.Sigma_ep_dyn_R * sensitivity,
        rcond=1e-12,
    )[0]
    scale = max(1.0, float(np.max(np.abs(weights))))
    keep = np.abs(weights) > sys.pole_residue_tol * scale
    zeta = zeta[keep]
    weights = weights[keep]
    if len(zeta) == 0:
        raise RuntimeError("All causal AAA residues were negligible.")
    return zeta, weights, aaa_terms, aaa_converged


def _build_candidate(
    sys,
    frozen: FrozenSCBA,
    tolerance: float,
    fit_method: str = "minipole",
    *,
    aaa_max_terms: int | None = None,
) -> PoleCache:
    fit_terms = 0
    fit_converged = True
    if sys.g_q == 0.0 or np.max(np.abs(frozen.Sigma_ep_dyn_R)) == 0.0:
        zeta = np.empty(0, dtype=np.complex128)
        weights = np.empty(0, dtype=np.complex128)
    else:
        if fit_method == "minipole":
            zeta, weights = _fit_with_minipole(frozen, sys, tolerance)
        elif fit_method == "causal_aaa":
            if aaa_max_terms is None:
                raise ValueError("aaa_max_terms is required for causal_aaa.")
            zeta, weights, fit_terms, fit_converged = _fit_with_causal_aaa(
                frozen,
                sys,
                max_terms=aaa_max_terms,
            )
        else:
            raise ValueError(f"Unknown self-energy pole fit method {fit_method!r}.")

    xi_unbiased = green_poles_from_sigma_mpm(
        sys, frozen.Sigma_H, zeta, weights, biased=False
    )
    residues_unbiased = green_residues(
        sys, xi_unbiased, zeta, weights, biased=False
    )
    xi_unbiased, residues_unbiased = _filter_green_poles(
        sys, xi_unbiased, residues_unbiased
    )
    xi_biased = green_poles_from_sigma_mpm(
        sys, frozen.Sigma_H, zeta, weights, biased=True
    )
    residues_biased = green_residues(
        sys, xi_biased, zeta, weights, biased=True
    )
    xi_biased, residues_biased = _filter_green_poles(
        sys, xi_biased, residues_biased
    )
    if len(zeta) == 0:
        expected_unbiased = 1 + len(_lead_pole_groups(sys, biased=False))
        expected_biased = 1 + len(_lead_pole_groups(sys, biased=True))
        if len(xi_unbiased) != expected_unbiased or len(xi_biased) != expected_biased:
            raise RuntimeError(
                "Phonon-free Green pole count mismatch: "
                f"unbiased={len(xi_unbiased)}/{expected_unbiased}, "
                f"biased={len(xi_biased)}/{expected_biased}."
            )

    w_eval = frozen.w + 1j * sys.ETA
    sigma_fit = sigma_mpm(w_eval, zeta, weights)
    sigma_abs, sigma_rel = _errors(
        frozen.Sigma_ep_dyn_R, sigma_fit, sys.mpm_reconstruction_floor
    )
    sigma_scaled = _scaled_error(
        frozen.Sigma_ep_dyn_R,
        sigma_fit,
        sys.mpm_fit_abs_tol,
        sys.mpm_fit_rel_tol,
    )

    cache_stub = PoleCache(
        zeta=_readonly(zeta),
        weights=_readonly(weights),
        xi_unbiased=_readonly(xi_unbiased),
        residues_unbiased=_readonly(residues_unbiased),
        xi_biased=_readonly(xi_biased),
        residues_biased=_readonly(residues_biased),
        pulse_protocol=sys.pulse_protocol,
        sigma_H=frozen.Sigma_H,
        tolerance=tolerance,
        max_sigma_abs_error=sigma_abs,
        max_sigma_rel_error=sigma_rel,
        max_sigma_scaled_error=sigma_scaled,
        max_Gfr_abs_error=np.inf,
        max_Gbar_abs_error=np.inf,
        max_Gfr_scaled_error=np.inf,
        max_Gbar_scaled_error=np.inf,
        causal=True,
        fit_method=fit_method,
        fit_terms=fit_terms,
        fit_converged=fit_converged,
    )
    direct_Gfr = 1.0 / (
        w_eval - sys.e_0 - frozen.Sigma_H
        - sigma_lead_unbiased_R(sys, w_eval) - frozen.Sigma_ep_dyn_R
    )
    direct_Gbar = 1.0 / (
        w_eval - sys.e_0 - sys.DELTA - frozen.Sigma_H
        - sigma_lead_biased_R(sys, w_eval) - frozen.Sigma_ep_dyn_R
    )
    gfr_error = float(np.max(np.abs(Gfr_R_mpm(sys, cache_stub, frozen.w) - direct_Gfr)))
    gbar_error = float(np.max(np.abs(Gbar_R_mpm(sys, cache_stub, frozen.w) - direct_Gbar)))
    gfr_scaled = _scaled_error(
        direct_Gfr,
        Gfr_R_mpm(sys, cache_stub, frozen.w),
        sys.mpm_fit_abs_tol,
        sys.mpm_green_rel_tol,
    )
    gbar_scaled = _scaled_error(
        direct_Gbar,
        Gbar_R_mpm(sys, cache_stub, frozen.w),
        sys.mpm_fit_abs_tol,
        sys.mpm_green_rel_tol,
    )
    return PoleCache(
        zeta=cache_stub.zeta,
        weights=cache_stub.weights,
        xi_unbiased=cache_stub.xi_unbiased,
        residues_unbiased=cache_stub.residues_unbiased,
        xi_biased=cache_stub.xi_biased,
        residues_biased=cache_stub.residues_biased,
        pulse_protocol=cache_stub.pulse_protocol,
        sigma_H=cache_stub.sigma_H,
        tolerance=tolerance,
        max_sigma_abs_error=sigma_abs,
        max_sigma_rel_error=sigma_rel,
        max_sigma_scaled_error=sigma_scaled,
        max_Gfr_abs_error=gfr_error,
        max_Gbar_abs_error=gbar_error,
        max_Gfr_scaled_error=gfr_scaled,
        max_Gbar_scaled_error=gbar_scaled,
        causal=True,
        fit_method=fit_method,
        fit_terms=fit_terms,
        fit_converged=fit_converged,
    )


def _candidate_is_valid(cache: PoleCache) -> bool:
    return bool(
        cache.max_sigma_scaled_error <= 1.0
        and cache.max_Gfr_scaled_error <= 1.0
        and cache.max_Gbar_scaled_error <= 1.0
    )


def _candidate_error_summary(cache: PoleCache) -> str:
    return (
        f"sigma abs/rel={cache.max_sigma_abs_error:.3e}/"
        f"{cache.max_sigma_rel_error:.3e}, "
        f"scaled={cache.max_sigma_scaled_error:.3e}, G errors="
        f"{cache.max_Gfr_abs_error:.3e}/{cache.max_Gbar_abs_error:.3e}, "
        f"G scaled={cache.max_Gfr_scaled_error:.3e}/"
        f"{cache.max_Gbar_scaled_error:.3e}"
    )


def build_pole_cache(sys, frozen: FrozenSCBA) -> PoleCache:
    attempts = [float(sys.mpm_tol)]
    if sys.g_q != 0.0 and sys.mpm_tol > 1e-10:
        attempts.append(1e-10)
    failures: list[str] = []
    for tolerance in attempts:
        try:
            cache = _build_candidate(sys, frozen, tolerance)
            if _candidate_is_valid(cache):
                return cache
            failures.append(
                f"tol={tolerance:.1e}: {_candidate_error_summary(cache)}"
            )
        except Exception as exc:
            failures.append(f"tol={tolerance:.1e}: {exc}")
    if sys.g_q != 0.0:
        available_terms = len(frozen.w) - 1
        configured_schedule = _aaa_term_schedule(
            int(sys.mpm_aaa_initial_terms),
            int(sys.mpm_aaa_max_terms),
            float(sys.mpm_aaa_growth_factor),
        )
        # Very small test grids may support fewer terms than the configured
        # production schedule. Clamp only after validating the configuration,
        # and remove duplicate clamped attempts while preserving order.
        effective_schedule = tuple(dict.fromkeys(
            min(max_terms, available_terms)
            for max_terms in configured_schedule
        ))
        for max_terms in effective_schedule:
            try:
                cache = _build_candidate(
                    sys,
                    frozen,
                    float(sys.mpm_aaa_rtol),
                    fit_method="causal_aaa",
                    aaa_max_terms=max_terms,
                )
                if _candidate_is_valid(cache):
                    return cache
                failures.append(
                    f"causal_aaa(max_terms={max_terms}, "
                    f"terms={cache.fit_terms}, "
                    f"raw_converged={cache.fit_converged}): "
                    f"{_candidate_error_summary(cache)}"
                )
            except Exception as exc:
                failures.append(f"causal_aaa(max_terms={max_terms}): {exc}")
    raise RuntimeError("MPM validation failed; " + "; ".join(failures))
