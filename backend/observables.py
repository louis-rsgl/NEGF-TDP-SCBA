from __future__ import annotations

import numpy as np
import scipy as sp
from scipy.integrate import quad_vec

from backend.distribution import fermi_dirac, bose_einstein, expc
from backend.green_function import GR_eq, get_eq_poles_residues, R_gamma
from backend.system_classes import System, Lead


DEBUG_CURRENT = True
DEBUG_A_B = False
DEBUG_A_B_MAX_POLES = 3


def _dbg(msg: str) -> None:
    if DEBUG_CURRENT:
        print(msg, flush=True)


def _fmt_scalar(z) -> str:
    zc = complex(z)
    return (
        f"{zc.real:.6e}"
        + (" + " if zc.imag >= 0 else " - ")
        + f"{abs(zc.imag):.6e}j"
    )


def _summarize_array(name: str, arr) -> None:
    arr = np.asarray(arr)
    _dbg(
        f"{name}: "
        f"shape={arr.shape} | "
        f"dtype={arr.dtype} | "
        f"min|.|={np.min(np.abs(arr)):.6e} | "
        f"max|.|={np.max(np.abs(arr)):.6e}"
    )
    flat = arr.ravel()
    if flat.size > 0:
        _dbg(f"{name}[0]      = {_fmt_scalar(flat[0])}")
        _dbg(f"{name}[mid]    = {_fmt_scalar(flat[flat.size // 2])}")
        _dbg(f"{name}[-1]     = {_fmt_scalar(flat[-1])}")


def _omega_int_x_bounds(sys: System, margin: float = 2.0) -> tuple[float, float]:
    equi_poles = get_eq_poles_residues(sys)

    delta_diffs = [
        sys.Delta(alpha) - sys.Delta(mu)
        for alpha in sys.lead_names
        for mu in sys.lead_names
    ]
    delta_min = min(delta_diffs) if delta_diffs else 0.0
    delta_max = max(delta_diffs) if delta_diffs else 0.0

    pole_reals = [complex(w_r).real for _, w_r in equi_poles]
    pole_min = min(pole_reals) if pole_reals else sys.e_min
    pole_max = max(pole_reals) if pole_reals else sys.e_max

    x_min = min(sys.e_min + delta_min, pole_min) - margin
    x_max = max(sys.e_max + delta_max, pole_max) + margin

    return float(x_min), float(x_max)


def A(sys: System, e, t, alpha: Lead):
    Delta_alpha = sys.Delta(alpha)
    nph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

    term1 = GR_eq(sys, e)
    equi_poles = get_eq_poles_residues(sys)

    t = np.asarray(t, dtype=float)
    term2 = np.zeros_like(t, dtype=np.complex128)

    GR_noneq_e_alpha = sys.GR_noneq(e + Delta_alpha)

    if DEBUG_A_B:
        _dbg("=" * 90)
        _dbg(f"A() called | alpha={alpha} | e={_fmt_scalar(e)} | Delta_alpha={Delta_alpha:.6e}")
        _dbg(f"A(): nph={nph:.6e} | number of poles={len(equi_poles)}")
        _dbg(f"A(): GR_eq(sys,e)={_fmt_scalar(term1)}")
        _dbg(f"A(): GR_noneq(e + Delta_alpha)={_fmt_scalar(GR_noneq_e_alpha)}")

    for pole_idx, (R_r, w_r) in enumerate(equi_poles):
        omega_int_wr = sys.omega_int(w_r)

        GR_noneq_w_r = sys.GR_noneq(w_r)
        GR_noneq_w_rmw_q = sys.GR_noneq(w_r - sys.w_q)
        GR_noneq_w_rpw_q = sys.GR_noneq(w_r + sys.w_q)

        term2_1 = Delta_alpha / (w_r - e)
        term2_2 = sys.DELTA * GR_noneq_e_alpha

        term2_3 = (
            (2j * sys.w_q * sys.g_q**2 / np.pi)
            * GR_noneq_e_alpha
            * omega_int_wr
        )

        term2_4 = (
            (2j * sys.w_q * sys.g_q**2 * sys.DELTA / np.pi)
            * GR_noneq_e_alpha
            * GR_noneq_w_r
            * omega_int_wr
        )

        term2_5 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
        )

        term2_6 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_7 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
        )

        term2_8 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_9 = 0.0 + 0.0j
        for mu in sys.lead_names:
            Delta_mu = sys.Delta(mu)
            Gamma_mu = sys.Gamma0(mu)

            x_em = e + Delta_alpha - Delta_mu
            omega_int_em = sys.omega_int(x_em)

            GR_noneq_em = sys.GR_noneq(x_em)
            GR_noneq_em_mwq = sys.GR_noneq(x_em - sys.w_q)
            GR_noneq_em_pwq = sys.GR_noneq(x_em + sys.w_q)

            term2_9_1 = 0.5 * (
                (Gamma_mu * sys.W) / (w_r + 1j * sys.W)
                - (Gamma_mu * sys.W) / (x_em + 1j * sys.W)
            ) / (w_r - e + Delta_mu - Delta_alpha + 1j * sys.ETA)

            term2_9_2_1 = 2j * sys.w_q * omega_int_wr
            term2_9_2_2 = -2.0 * (nph + 1.0) * GR_noneq_w_rmw_q
            term2_9_2_3 = -2.0 * nph * GR_noneq_w_rpw_q

            term2_9_2 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_w_r * sys.g_q**2)
                / ((w_r + 1j * sys.W) * (w_r - e + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_2_1 + term2_9_2_2 + term2_9_2_3)

            term2_9_3_1 = -2j * sys.w_q * omega_int_em
            term2_9_3_2 = -2.0 * (nph + 1.0) * GR_noneq_em_mwq
            term2_9_3_3 = -2.0 * nph * GR_noneq_em_pwq

            term2_9_3 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_em * sys.g_q**2)
                / ((x_em + 1j * sys.W) * (w_r - e + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_3_1 + term2_9_3_2 + term2_9_3_3)

            term2_9 -= Delta_mu * (term2_9_1 + term2_9_2 + term2_9_3) * GR_noneq_e_alpha

        pref = -np.exp(-1j * (w_r - e) * t) * R_r / (w_r - e - Delta_alpha)
        pole_contrib = pref * (
            term2_1
            + term2_2
            + term2_3
            + term2_4
            + term2_5
            + term2_6
            + term2_7
            + term2_8
            + term2_9
        )

        term2 += pole_contrib

        if DEBUG_A_B and pole_idx < DEBUG_A_B_MAX_POLES:
            _dbg(
                f"A(): pole[{pole_idx}] | "
                f"R_r={_fmt_scalar(R_r)} | w_r={_fmt_scalar(w_r)} | "
                f"max|pref|={np.max(np.abs(pref)):.6e} | "
                f"|term2_1|={abs(term2_1):.6e} | "
                f"|term2_2|={abs(term2_2):.6e} | "
                f"|term2_3|={abs(term2_3):.6e} | "
                f"|term2_4|={abs(term2_4):.6e} | "
                f"|term2_5|={abs(term2_5):.6e} | "
                f"|term2_6|={abs(term2_6):.6e} | "
                f"|term2_7|={abs(term2_7):.6e} | "
                f"|term2_8|={abs(term2_8):.6e} | "
                f"|term2_9|={abs(term2_9):.6e} | "
                f"max|pole_contrib|={np.max(np.abs(pole_contrib)):.6e}"
            )

    out = term1 + term2

    if DEBUG_A_B:
        _dbg(f"A(): |term1|={abs(term1):.6e} | max|term2|={np.max(np.abs(term2)):.6e}")
        _dbg(f"A(): max|output|={np.max(np.abs(out)):.6e}")

    return out


def B(sys: System, e, e_prime, t, alpha: Lead):
    Delta_alpha = sys.Delta(alpha)
    nph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)

    t = np.asarray(t, dtype=float)
    term1 = expc(e - e_prime, t) * GR_eq(sys, e_prime)

    equi_poles = get_eq_poles_residues(sys)
    term2 = np.zeros_like(t, dtype=np.complex128)

    GR_noneq_e_prime_alpha = sys.GR_noneq(e_prime + Delta_alpha)

    if DEBUG_A_B:
        _dbg("=" * 90)
        _dbg(
            f"B() called | alpha={alpha} | e={_fmt_scalar(e)} | "
            f"e_prime={_fmt_scalar(e_prime)} | Delta_alpha={Delta_alpha:.6e}"
        )
        _dbg(f"B(): nph={nph:.6e} | number of poles={len(equi_poles)}")
        _dbg(f"B(): GR_noneq(e_prime + Delta_alpha)={_fmt_scalar(GR_noneq_e_prime_alpha)}")

    for pole_idx, (R_r, w_r) in enumerate(equi_poles):
        omega_int_wr = sys.omega_int(w_r)

        GR_noneq_w_r = sys.GR_noneq(w_r)
        GR_noneq_w_rmw_q = sys.GR_noneq(w_r - sys.w_q)
        GR_noneq_w_rpw_q = sys.GR_noneq(w_r + sys.w_q)

        term2_1 = Delta_alpha / (w_r - e_prime)
        term2_2 = sys.DELTA * GR_noneq_e_prime_alpha

        term2_3 = (
            (2j * sys.w_q * sys.g_q**2 / np.pi)
            * GR_noneq_e_prime_alpha
            * omega_int_wr
        )

        term2_4 = (
            (2j * sys.w_q * sys.g_q**2 * sys.DELTA / np.pi)
            * GR_noneq_e_prime_alpha
            * GR_noneq_w_r
            * omega_int_wr
        )

        term2_5 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
        )

        term2_6 = (
            2.0 * (nph + 1.0)
            * GR_noneq_w_rmw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_7 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
        )

        term2_8 = (
            2.0 * nph
            * GR_noneq_w_rpw_q
            * sys.g_q**2
            * GR_noneq_e_prime_alpha
            * sys.DELTA
            * GR_noneq_w_r
        )

        term2_9 = 0.0 + 0.0j
        for mu in sys.lead_names:
            Delta_mu = sys.Delta(mu)
            Gamma_mu = sys.Gamma0(mu)

            x_em = e_prime + Delta_alpha - Delta_mu
            omega_int_em = sys.omega_int(x_em)

            GR_noneq_em = sys.GR_noneq(x_em)
            GR_noneq_em_mwq = sys.GR_noneq(x_em - sys.w_q)
            GR_noneq_em_pwq = sys.GR_noneq(x_em + sys.w_q)

            term2_9_1 = 0.5 * (
                (Gamma_mu * sys.W) / (w_r + 1j * sys.W)
                - (Gamma_mu * sys.W) / (x_em + 1j * sys.W)
            ) / (w_r - e_prime + Delta_mu - Delta_alpha + 1j * sys.ETA)

            term2_9_2_1 = 2j * sys.w_q * omega_int_wr
            term2_9_2_2 = -2.0 * (nph + 1.0) * GR_noneq_w_rmw_q
            term2_9_2_3 = -2.0 * nph * GR_noneq_w_rpw_q

            term2_9_2 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_w_r * sys.g_q**2)
                / ((w_r + 1j * sys.W) * (w_r - e_prime + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_2_1 + term2_9_2_2 + term2_9_2_3)

            term2_9_3_1 = -2j * sys.w_q * omega_int_em
            term2_9_3_2 = -2.0 * (nph + 1.0) * GR_noneq_em_mwq
            term2_9_3_3 = -2.0 * nph * GR_noneq_em_pwq

            term2_9_3 = 0.5 * (
                (Gamma_mu * sys.W * GR_noneq_em * sys.g_q**2)
                / ((x_em + 1j * sys.W) * (w_r - e_prime + Delta_mu - Delta_alpha + 1j * sys.ETA))
            ) * (term2_9_3_1 + term2_9_3_2 + term2_9_3_3)

            term2_9 -= Delta_mu * (term2_9_1 + term2_9_2 + term2_9_3) * GR_noneq_e_prime_alpha

        pref = -expc(e - w_r, t) * R_r / (w_r - e_prime - Delta_alpha)
        pole_contrib = pref * (
            term2_1
            + term2_2
            + term2_3
            + term2_4
            + term2_5
            + term2_6
            + term2_7
            + term2_8
            + term2_9
        )

        term2 += pole_contrib

        if DEBUG_A_B and pole_idx < DEBUG_A_B_MAX_POLES:
            _dbg(
                f"B(): pole[{pole_idx}] | "
                f"R_r={_fmt_scalar(R_r)} | w_r={_fmt_scalar(w_r)} | "
                f"max|pref|={np.max(np.abs(pref)):.6e} | "
                f"|term2_1|={abs(term2_1):.6e} | "
                f"|term2_2|={abs(term2_2):.6e} | "
                f"|term2_3|={abs(term2_3):.6e} | "
                f"|term2_4|={abs(term2_4):.6e} | "
                f"|term2_5|={abs(term2_5):.6e} | "
                f"|term2_6|={abs(term2_6):.6e} | "
                f"|term2_7|={abs(term2_7):.6e} | "
                f"|term2_8|={abs(term2_8):.6e} | "
                f"|term2_9|={abs(term2_9):.6e} | "
                f"max|pole_contrib|={np.max(np.abs(pole_contrib)):.6e}"
            )

    out = term1 + term2

    if DEBUG_A_B:
        _dbg(f"B(): max|term1|={np.max(np.abs(term1)):.6e} | max|term2|={np.max(np.abs(term2)):.6e}")
        _dbg(f"B(): max|output|={np.max(np.abs(out)):.6e}")

    return out


def current_alpha(
    sys: System,
    alpha: Lead,
    t_max: float,
    n_t: int,
    omega_int_n_x: int | None = None,
    omega_int_n_omega: int | None = None,
):
    rep = sys.reporter()

    if sys.verbose:
        rep.section(f"Current evaluation for lead {alpha}")
        rep.info(f"t_max = {t_max:.6f} | n_t = {n_t}")

    t = np.linspace(0.0, t_max, n_t)
    Delta_alpha = sys.Delta(alpha)
    Gamma_alpha = sys.Gamma0(alpha)
    R_alpha = R_gamma(sys, alpha, "+")

    _dbg("=" * 90)
    _dbg(f"current_alpha() start | alpha={alpha}")
    _dbg(f"t_max={t_max:.6e} | n_t={n_t}")
    _dbg(f"Delta_alpha={Delta_alpha:.6e}")
    _dbg(f"Gamma_alpha={Gamma_alpha:.6e}")
    _dbg(f"R_alpha={_fmt_scalar(R_alpha)}")
    _dbg(f"sys.W={sys.W:.6e}")
    _dbg(f"sys.g_q={sys.g_q:.6e}")
    _dbg(f"sys.w_q={sys.w_q:.6e}")
    _dbg(f"sys.DELTA={sys.DELTA:.6e}")
    _dbg(f"sys.ETA={sys.ETA:.6e}")
    _dbg(f"e-range=[{sys.e_min:.6e}, {sys.e_max:.6e}]")
    _dbg(f"lead_names={list(sys.lead_names)}")

    result = sys.solve_noneq()

    if sys.verbose:
        rep.info(
            "nonequilibrium solution ready | "
            f"converged={result.converged} | "
            f"iterations={result.n_iter} | "
            f"GR_res_abs={result.res_GR_abs:.3e} | "
            f"GR_res_rel={result.res_GR_rel:.3e} | "
            f"G<_res_abs={result.res_Gless_abs:.3e} | "
            f"G<_res_rel={result.res_Gless_rel:.3e}"
        )

    _dbg(
        "solve_noneq() done | "
        f"converged={result.converged} | "
        f"iterations={result.n_iter} | "
        f"GR_abs={result.res_GR_abs:.6e} | "
        f"GR_rel={result.res_GR_rel:.6e} | "
        f"G<_abs={result.res_Gless_abs:.6e} | "
        f"G<_rel={result.res_Gless_rel:.6e}"
    )

    x_min, x_max = _omega_int_x_bounds(sys)
    _dbg(f"omega-int x bounds: x_min={x_min:.6e} | x_max={x_max:.6e}")

    sys.build_omega_int_table(
        x_min=x_min,
        x_max=x_max,
        n_x=omega_int_n_x,
        n_omega=omega_int_n_omega,
    )

    nph = bose_einstein(sys.w_q, sys.beta_ph, sys.mu_ph)
    _dbg(f"nph={nph:.6e}")

    # quick samples of key nonequilibrium objects
    sample_es = [
        sys.e_min,
        0.5 * (sys.e_min + sys.e_max),
        sys.e_max,
        Delta_alpha,
        Delta_alpha - sys.w_q,
        Delta_alpha + sys.w_q,
    ]
    _dbg("-" * 90)
    _dbg("Sampling key Green functions")
    for e0 in sample_es:
        try:
            gr = sys.GR_noneq(e0)
            gl = sys.Gless_noneq(e0)
            _dbg(
                f"e={e0:.6e} | "
                f"GR_noneq={_fmt_scalar(gr)} | "
                f"Gless_noneq={_fmt_scalar(gl)} | "
                f"|GR|={abs(gr):.6e} | "
                f"|G<|={abs(gl):.6e}"
            )
        except Exception as exc:
            _dbg(f"sampling failure at e={e0:.6e}: {exc}")

    def integrand1(e):
        return (
            fermi_dirac(e, sys.beta_fc(alpha), sys.mu_fc(alpha))
            * A(sys, e, t, alpha)
            * (e**2 + sys.W**2) ** (-1)
        )

    # sample integrand1
    _dbg("-" * 90)
    _dbg("Sampling integrand1")
    for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
        try:
            y = integrand1(e0)
            _summarize_array(f"integrand1(e={e0:.6e})", y)
        except Exception as exc:
            _dbg(f"integrand1 sampling failure at e={e0:.6e}: {exc}")

    if sys.verbose:
        with rep.timed("Evaluating term1"):
            val1, err1 = quad_vec(integrand1, sys.e_min, sys.e_max)
    else:
        val1, err1 = quad_vec(integrand1, sys.e_min, sys.e_max)

    pref1 = -2 * (Gamma_alpha * sys.W**2 / (2 * np.pi))
    term1 = pref1 * np.imag(val1)

    _dbg("-" * 90)
    _dbg("term1 diagnostics")
    _dbg(f"pref1={pref1:.6e}")
    _dbg(f"quad_vec err1={err1}")
    _summarize_array("val1", val1)
    _summarize_array("imag(val1)", np.imag(val1))
    _summarize_array("term1", term1)

    term2 = np.zeros_like(t, dtype=np.float64)
    term3 = np.zeros_like(t, dtype=np.float64)

    if sys.verbose:
        rep.info("Summing over lead index beta for term2 and term3")

    for beta in sys.lead_names:
        Gamma_beta = sys.Gamma0(beta)
        Delta_beta = sys.Delta(beta)
        beta_beta = sys.beta_fc(beta)
        mu_beta = sys.mu_fc(beta)

        if sys.verbose:
            rep.info(
                f"  beta = {beta} | "
                f"Gamma_beta = {Gamma_beta:.6f} | "
                f"Delta_beta = {Delta_beta:.6f}"
            )

        _dbg("-" * 90)
        _dbg(
            f"beta loop start | beta={beta} | "
            f"Gamma_beta={Gamma_beta:.6e} | "
            f"Delta_beta={Delta_beta:.6e} | "
            f"beta_beta={beta_beta:.6e} | "
            f"mu_beta={mu_beta:.6e}"
        )

        def integrand2(e):
            return (
                np.exp(-1j * e * t)
                * fermi_dirac(e, beta_beta, mu_beta)
                * A(sys, e, t, beta)
                * Gamma_beta
                * (e**2 + sys.W**2) ** (-1)
                * np.conjugate(sys.GR_noneq(e + Delta_beta))
                * 1j
                * R_alpha
                * np.exp(-sys.W * t)
                * (1j * sys.W - e + Delta_alpha - Delta_beta) ** (-1)
            )

        _dbg(f"term2 denominator check for beta={beta}")
        for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
            denom = (1j * sys.W - e0 + Delta_alpha - Delta_beta)
            _dbg(
                f"  e={e0:.6e} | denom={_fmt_scalar(denom)} | |denom|={abs(denom):.6e}"
            )

        _dbg(f"Sampling integrand2 for beta={beta}")
        for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
            try:
                y = integrand2(e0)
                _summarize_array(f"integrand2(beta={beta}, e={e0:.6e})", y)
            except Exception as exc:
                _dbg(f"integrand2 sampling failure at beta={beta}, e={e0:.6e}: {exc}")

        if sys.verbose:
            with rep.timed(f"Evaluating term2 contribution for beta={beta}"):
                val2, err2 = quad_vec(integrand2, sys.e_min, sys.e_max)
        else:
            val2, err2 = quad_vec(integrand2, sys.e_min, sys.e_max)

        pref2 = 2 * (sys.W**2 / (2 * np.pi))
        term2_beta = pref2 * np.imag(val2)
        term2 += term2_beta

        _dbg(f"term2 diagnostics for beta={beta}")
        _dbg(f"pref2={pref2:.6e}")
        _dbg(f"quad_vec err2={err2}")
        _summarize_array(f"val2(beta={beta})", val2)
        _summarize_array(f"imag(val2)(beta={beta})", np.imag(val2))
        _summarize_array(f"term2_beta(beta={beta})", term2_beta)
        _summarize_array("term2 cumulative", term2)

        def integrand3(e):
            return (
                np.exp(-1j * e * t)
                * fermi_dirac(e, beta_beta, mu_beta)
                * A(sys, e, t, beta)
                * Gamma_beta
                * (e**2 + sys.W**2) ** (-1)
                * R_alpha
                * np.exp(-sys.W * t)
                * np.conjugate(B(sys, 1j * sys.W, e, t, beta))
            )

        _dbg(f"Sampling integrand3 for beta={beta}")
        for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
            try:
                y = integrand3(e0)
                _summarize_array(f"integrand3(beta={beta}, e={e0:.6e})", y)
            except Exception as exc:
                _dbg(f"integrand3 sampling failure at beta={beta}, e={e0:.6e}: {exc}")

        if sys.verbose:
            with rep.timed(f"Evaluating term3 contribution for beta={beta}"):
                val3, err3 = quad_vec(integrand3, sys.e_min, sys.e_max)
        else:
            val3, err3 = quad_vec(integrand3, sys.e_min, sys.e_max)

        pref3 = 2 * (sys.W**2 / (2 * np.pi))
        term3_beta = pref3 * np.imag(val3)
        term3 += term3_beta

        _dbg(f"term3 diagnostics for beta={beta}")
        _dbg(f"pref3={pref3:.6e}")
        _dbg(f"quad_vec err3={err3}")
        _summarize_array(f"val3(beta={beta})", val3)
        _summarize_array(f"imag(val3)(beta={beta})", np.imag(val3))
        _summarize_array(f"term3_beta(beta={beta})", term3_beta)
        _summarize_array("term3 cumulative", term3)

    def integrand4(e):
        return (
            sys.GR_noneq(e)
            * np.conjugate(sys.GR_noneq(e))
            * (
                sys.Gless_noneq(e - sys.w_q) * nph
                + sys.Gless_noneq(e + sys.w_q) * (nph + 1.0)
            )
            * (
                (1.0 - np.exp(-(sys.W + 1j * e) * t)) / (sys.W + 1j * e)
                - (1j * (e - Delta_alpha)) / ((e - Delta_alpha) ** 2 + sys.W**2)
            )
        )

    _dbg("-" * 90)
    _dbg("Sampling integrand4")
    for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
        try:
            y = integrand4(e0)
            _summarize_array(f"integrand4(e={e0:.6e})", y)
        except Exception as exc:
            _dbg(f"integrand4 sampling failure at e={e0:.6e}: {exc}")

    if sys.verbose:
        with rep.timed("Evaluating term4"):
            val4, err4 = quad_vec(integrand4, sys.e_min, sys.e_max)
    else:
        val4, err4 = quad_vec(integrand4, sys.e_min, sys.e_max)

    pref4 = -2 * sys.g_q**2 * (sys.W * Gamma_alpha / (2 * np.pi))
    term4 = pref4 * np.imag(val4)

    _dbg("term4 diagnostics")
    _dbg(f"pref4={pref4:.6e}")
    _dbg(f"quad_vec err4={err4}")
    _summarize_array("val4", val4)
    _summarize_array("imag(val4)", np.imag(val4))
    _summarize_array("term4", term4)

    def integrand5(e):
        return (
            sys.GR_noneq(e + Delta_alpha)
            * np.conjugate(sys.GR_noneq(e + Delta_alpha))
            * sys.Gless_noneq(e + Delta_alpha - sys.w_q)
            * nph
        )

    _dbg("-" * 90)
    _dbg("Sampling integrand5")
    for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
        try:
            y = integrand5(e0)
            _dbg(
                f"integrand5(e={e0:.6e}) = {_fmt_scalar(y)} | |.|={abs(y):.6e}"
            )
        except Exception as exc:
            _dbg(f"integrand5 sampling failure at e={e0:.6e}: {exc}")

    if sys.verbose:
        with rep.timed("Evaluating term5"):
            val5, err5 = quad_vec(integrand5, sys.e_min, sys.e_max)
    else:
        val5, err5 = quad_vec(integrand5, sys.e_min, sys.e_max)

    pref5 = -2 * sys.g_q**2 * (sys.W**2 * Gamma_alpha / (2 * np.pi))
    term5 = pref5 * np.imag(val5)

    _dbg("term5 diagnostics")
    _dbg(f"pref5={pref5:.6e}")
    _dbg(f"quad_vec err5={err5}")
    _dbg(f"val5={_fmt_scalar(val5)} | imag(val5)={np.imag(val5):.6e}")
    _dbg(f"term5={term5:.6e}")

    def integrand6(e):
        return (
            sys.GR_noneq(e + Delta_alpha)
            * np.conjugate(sys.GR_noneq(e + Delta_alpha))
            * sys.Gless_noneq(e + Delta_alpha + sys.w_q)
            * (nph + 1.0)
        )

    _dbg("-" * 90)
    _dbg("Sampling integrand6")
    for e0 in [sys.e_min, -sys.W, 0.0, sys.W, sys.e_max]:
        try:
            y = integrand6(e0)
            _dbg(
                f"integrand6(e={e0:.6e}) = {_fmt_scalar(y)} | |.|={abs(y):.6e}"
            )
        except Exception as exc:
            _dbg(f"integrand6 sampling failure at e={e0:.6e}: {exc}")

    if sys.verbose:
        with rep.timed("Evaluating term6"):
            val6, err6 = quad_vec(integrand6, sys.e_min, sys.e_max)
    else:
        val6, err6 = quad_vec(integrand6, sys.e_min, sys.e_max)

    pref6 = -2 * sys.g_q**2 * (sys.W**2 * Gamma_alpha / (2 * np.pi))
    term6 = pref6 * np.imag(val6)

    _dbg("term6 diagnostics")
    _dbg(f"pref6={pref6:.6e}")
    _dbg(f"quad_vec err6={err6}")
    _dbg(f"val6={_fmt_scalar(val6)} | imag(val6)={np.imag(val6):.6e}")
    _dbg(f"term6={term6:.6e}")

    I_alpha = term1 + term2 + term3 + term4 + term5 + term6

    _dbg("=" * 90)
    _dbg("Final current decomposition")
    _summarize_array("term1", term1)
    _summarize_array("term2", term2)
    _summarize_array("term3", term3)
    _summarize_array("term4", term4)
    _dbg(f"term5 scalar={term5:.6e}")
    _dbg(f"term6 scalar={term6:.6e}")
    _summarize_array("I_alpha", I_alpha)

    _dbg(
        "Pointwise cancellation check | "
        f"max|term1+term2|={np.max(np.abs(term1 + term2)):.6e} | "
        f"max|term1+term2+term3|={np.max(np.abs(term1 + term2 + term3)):.6e} | "
        f"max|full|={np.max(np.abs(I_alpha)):.6e}"
    )

    if sys.verbose:
        rep.info(
            f"Current evaluation finished for lead {alpha} | "
            f"max|I| = {np.max(np.abs(I_alpha)):.6e}"
        )

    return t, I_alpha