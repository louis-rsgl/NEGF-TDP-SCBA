import numpy as np
import pytest

from psscba.backend.model.system import LeadParams, System
from psscba.backend.stationary.scba import SCBAConvergenceError, Solver


def make_system(**changes):
    values = dict(
        ETA=1e-4,
        DELTA=0.4,
        leads={
            "L": LeadParams(Gamma0=0.4, Delta=0.7, beta=2.0, mu=0.0),
            "R": LeadParams(Gamma0=0.6, Delta=0.0, beta=2.0, mu=0.0),
        },
        W=3.0,
        g_q=0.0,
        w_q=0.3,
        e_0=0.1,
        beta_ph=10.0,
        e_min=-12.0,
        e_max=12.0,
        omega_min=-12.0,
        omega_max=12.0,
        n_w_scba=241,
        scba_max_iter=10,
        scba_min_iter=2,
        scba_mixing=1.0,
        scba_tol_abs=1e-10,
        scba_tol_rel=1e-10,
        verbose=False,
    )
    values.update(changes)
    return System(**values)


def test_shifted_lorentzian_and_lead_lesser_use_same_gauge():
    sys = make_system()
    solver = Solver(sys, -4.0, 4.0, 101)
    at_center = solver.linewidth("L", sys.Delta("L"))
    assert at_center == pytest.approx(sys.Gamma0("L"))
    lesser = solver.sigma_lead_less(np.array([sys.Delta("L")]))[0]
    expected_l = 1j * 0.5 * sys.Gamma0("L")
    shifted_r = sys.Delta("L") - sys.Delta("R")
    gamma_r = sys.Gamma0("R") * sys.W**2 / (shifted_r**2 + sys.W**2)
    expected_r = 1j * (1.0 / (np.exp(2.0 * shifted_r) + 1.0)) * gamma_r
    assert lesser == pytest.approx(expected_l + expected_r)


def test_hilbert_transform_sign_on_lorentzian():
    w = np.linspace(-50.0, 50.0, 4097)
    width = 2.0
    gamma0 = 0.7
    gamma = gamma0 * width**2 / (w**2 + width**2)
    exact = gamma0 * width * w / (2.0 * (w**2 + width**2))
    calculated = Solver._hilbert_pv(gamma)
    assert np.max(np.abs(calculated[np.abs(w) < 10] - exact[np.abs(w) < 10])) < 2e-4


def test_g_zero_stationary_solution_and_installed_arrays():
    sys = make_system()
    result = sys.solve_stationary()
    frozen = sys.stationary_kernel()
    lead = sys._stationary_solver.sigma_lead_R(frozen.w)
    expected = 1.0 / (
        frozen.w - sys.e_0 - sys.DELTA - lead + 1j * sys.ETA
    )
    assert result.converged
    assert np.max(np.abs(frozen.G_reference_R - expected)) < 1e-12
    assert frozen.N0 == pytest.approx(1.0 / np.expm1(sys.beta_ph * sys.w_q))
    assert not frozen.G_reference_R.flags.writeable
    assert not hasattr(sys, "solve_noneq")
    assert not hasattr(sys, "frozen_scba")


def test_upward_stationary_reference_is_unbiased():
    sys = make_system(pulse_protocol="upward", scba_mode="weak_born")
    sys.solve_stationary()
    frozen = sys.stationary_kernel()
    gamma_total = sum(sys.Gamma0(lead) for lead in sys.lead_names)
    expected = 1.0 / (
        frozen.w
        - sys.e_0
        - 0.5 * gamma_total * sys.W / (frozen.w + 1j * sys.W)
        + 1j * sys.ETA
    )
    assert frozen.pulse_protocol == "upward"
    assert not frozen.reference_is_biased
    assert np.max(np.abs(frozen.G_reference_R - expected)) < 1e-12


def test_upward_unbiased_lesser_fdt_and_nonthermal_N0():
    sys = make_system(
        pulse_protocol="upward",
        scba_mode="weak_born",
        ETA=1e-8,
        N0=0.37,
    )
    sys.solve_stationary()
    frozen = sys.stationary_kernel()
    fermi = 1.0 / (np.exp(np.clip(sys.beta_fc("L") * frozen.w, -700, 700)) + 1.0)
    expected = fermi * (np.conjugate(frozen.G_reference_R) - frozen.G_reference_R)
    assert frozen.N0 == pytest.approx(0.37)
    assert np.max(np.abs(frozen.G_reference_less - expected)) < 2e-7


def test_interpolation_uses_controlled_tails_and_rejects_complex_grid_queries():
    sys = make_system()
    sys.solve_stationary()
    solver = sys._stationary_solver
    assert solver.Gless(sys.e_max + 10.0) == 0.0j
    assert solver.GR(sys.e_max + 10.0) == pytest.approx(
        1.0 / (sys.e_max + 10.0 + 1j * sys.ETA)
    )
    with pytest.raises(ValueError, match="real axis"):
        solver.GR(1.0 + 0.2j)


def test_nonconvergence_raises():
    sys = make_system(g_q=0.1, scba_max_iter=1, scba_min_iter=2)
    with pytest.raises(SCBAConvergenceError):
        sys.solve_stationary()
