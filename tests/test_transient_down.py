import numpy as np
import pytest

from backend.distribution import expc
from backend.minimal_poles import Gbar_R_mpm, Gfr_R_mpm
from backend.observables import (
    A_down,
    A_down_direct,
    B_down,
    C_down,
    D_down,
    current_all,
    continuity_residual,
)
from tests.test_frozen_scba import make_system


def test_expc_series_and_exact_zero():
    assert expc(0.0, 2.5) == 2.5
    z = 1e-14 + 2e-14j
    assert abs(expc(z, 3.0) - 3.0) < 1e-12


def test_downward_boundaries_for_phonon_free_case():
    sys = make_system()
    sys.solve_noneq()
    cache = sys.prepare_poles()
    energies = np.linspace(-2.0, 2.0, 17)
    a0 = A_down(sys, energies, 0.0, "L", cache)
    expected_a = Gbar_R_mpm(sys, cache, energies + sys.Delta("L"))
    assert np.max(np.abs(a0 - expected_a)) < 2e-4
    assert np.max(np.abs(B_down(sys, energies, energies, 0.0, "L", cache))) == 0.0
    a_long = A_down(sys, energies, 50.0, "L", cache)
    assert np.max(np.abs(a_long - Gfr_R_mpm(sys, cache, energies))) < 1e-10
    for omega in (-sys.w_q, sys.w_q):
        c0 = C_down(sys, omega, energies, 0.0, cache)
        expected_c = Gbar_R_mpm(sys, cache, energies + omega)
        assert np.max(np.abs(c0 - expected_c)) < 2e-4
        assert np.max(np.abs(D_down(sys, omega, energies, energies, 0.0, cache))) == 0.0


def test_small_full_current_is_finite():
    sys = make_system(
        n_w_scba=41,
        current_energy_batch=7,
        scba_tol_abs=1e-8,
        scba_tol_rel=1e-8,
    )
    result = current_all(sys, t_max=0.02, n_t=3, omega_int_n_omega=31)
    assert set(result.currents) == {"L", "R"}
    assert all(np.all(np.isfinite(value)) for value in result.currents.values())
    assert np.all(np.isfinite(result.occupation))
    assert np.all(np.isfinite(result.continuity_residual))


def test_current_grid_honors_transport_window_when_counts_match():
    sys = make_system(e_min=-3.0, e_max=3.0, n_w_scba=41)
    result = current_all(sys, t_max=0.01, n_t=2, omega_int_n_omega=41)
    assert np.all(np.isfinite(result.currents["L"]))
    assert result.diagnostics["current_energy_min"] == -3.0
    assert result.diagnostics["current_energy_max"] == 3.0


def test_A_residue_matches_defining_real_axis_quadrature():
    sys = make_system(ETA=2e-4, e_min=-100.0, e_max=100.0)
    sys.solve_noneq()
    cache = sys.prepare_poles()
    residue = A_down(sys, -0.3, 0.2, "L", cache)
    direct = A_down_direct(sys, -0.3, 0.2, "L", cache)
    assert abs(residue - direct) < 2e-3


def test_nonzero_coupling_uses_minipole_and_frozen_lesser_current():
    pytest.importorskip("mini_pole")
    sys = make_system(
        g_q=0.02,
        n_w_scba=121,
        scba_max_iter=500,
        scba_min_iter=5,
        scba_mixing=0.25,
        scba_tol_abs=2e-6,
        scba_tol_rel=2e-5,
        mpm_n_iw=128,
        mpm_tol=1e-6,
        mpm_fit_abs_tol=1e-3,
        mpm_fit_rel_tol=0.5,
        current_energy_batch=7,
    )
    result = current_all(sys, t_max=0.01, n_t=2, omega_int_n_omega=21)
    assert result.diagnostics["sigma_poles"] > 0
    assert result.diagnostics["green_poles"] > 2
    assert np.all(np.isfinite(result.continuity_residual))


def test_continuity_uses_tex_current_sign_convention():
    times = np.linspace(0.0, 1.0, 5)
    occupation = times**2
    currents = {
        "L": times,
        "R": times,
    }
    assert np.max(np.abs(continuity_residual(currents, occupation, times))) < 1e-14
