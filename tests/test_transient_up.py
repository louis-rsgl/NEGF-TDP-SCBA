import numpy as np
import pytest
from scipy.integrate import quad_vec

from backend.minimal_poles import Gbiased_R_mpm, Gfr_R_mpm
from backend.observables import (
    A_up,
    A_up_direct,
    B_up,
    C_up,
    C_up_direct,
    D_up,
    _A_up_residue,
    _C_up_residue,
    current_all,
)
from tests.test_frozen_scba import make_system


def make_upward_system(**changes):
    return make_system(
        pulse_protocol="upward",
        scba_mode="weak_born",
        **changes,
    )


def test_upward_residue_boundaries_and_asymptotes():
    sys = make_upward_system()
    sys.solve_stationary()
    cache = sys.prepare_poles()
    energies = np.linspace(-2.0, 2.0, 17)
    expected = Gfr_R_mpm(sys, cache, energies)
    assert np.max(np.abs(A_up(sys, energies, 0.0, "L", cache) - expected)) == 0.0
    raw_a = _A_up_residue(
        sys, energies, 0.0, "L", cache, enforce_boundary=False
    )
    assert np.max(np.abs(raw_a - expected)) < 5e-4
    for omega in (-sys.w_q, sys.w_q):
        expected_c = Gfr_R_mpm(sys, cache, energies + omega)
        assert np.max(np.abs(C_up(sys, omega, energies, 0.0, cache) - expected_c)) == 0.0
        raw_c = _C_up_residue(
            sys, omega, energies, 0.0, cache, enforce_boundary=False
        )
        assert np.max(np.abs(raw_c - expected_c)) < 5e-4
        assert np.max(np.abs(D_up(sys, omega, energies, energies, 0.0, "L", cache))) == 0.0
    assert np.max(np.abs(B_up(sys, energies, energies, 0.0, "L", "R", cache))) == 0.0

    late = 100.0
    assert np.max(
        np.abs(
            A_up(sys, energies, late, "L", cache)
            - Gbiased_R_mpm(sys, cache, energies + sys.Delta("L"))
        )
    ) < 1e-10
    assert np.max(
        np.abs(
            C_up(sys, sys.w_q, energies, late, cache)
            - Gbiased_R_mpm(sys, cache, energies + sys.w_q)
        )
    ) < 1e-10


def test_upward_residues_match_direct_real_axis_quadrature():
    sys = make_upward_system(ETA=2e-4, e_min=-100.0, e_max=100.0)
    sys.solve_stationary()
    cache = sys.prepare_poles()
    assert abs(
        A_up(sys, -0.3, 0.2, "L", cache)
        - A_up_direct(sys, -0.3, 0.2, "L", cache)
    ) < 2e-3
    assert abs(
        C_up(sys, sys.w_q, -0.3, 0.2, cache)
        - C_up_direct(sys, sys.w_q, -0.3, 0.2, cache)
    ) < 2e-3


def test_upward_B_and_D_match_time_integrals():
    sys = make_upward_system()
    sys.solve_stationary()
    cache = sys.prepare_poles()
    energy = -0.2
    prime = 0.35
    time = 0.4
    alpha = "L"
    beta = "R"
    expected_b, _ = quad_vec(
        lambda tau: np.exp(
            1j
            * (
                energy
                - prime
                + sys.Delta(alpha)
                - sys.Delta(beta)
            )
            * tau
        )
        * A_up(sys, prime, tau, beta, cache),
        0.0,
        time,
    )
    assert abs(
        B_up(sys, energy, prime, time, alpha, beta, cache) - expected_b
    ) < 1e-10

    omega = sys.w_q
    expected_d, _ = quad_vec(
        lambda tau: np.exp(
            1j * (energy - omega - prime + sys.Delta(alpha)) * tau
        )
        * C_up(sys, omega, prime, tau, cache),
        0.0,
        time,
    )
    assert abs(
        D_up(sys, omega, prime, energy, time, alpha, cache) - expected_d
    ) < 1e-10


def test_upward_current_dispatch_boundaries_and_mismatch_rejection():
    sys = make_upward_system(n_w_scba=61, current_energy_batch=7)
    result = current_all(sys, t_max=0.01, n_t=2, omega_int_n_omega=61)
    frozen = sys.frozen_scba()
    stationary_occupation = float(
        np.real(-1j * np.trapezoid(frozen.G_reference_less, frozen.w) / (2.0 * np.pi))
    )
    assert result.diagnostics["pulse_protocol"] == "upward"
    assert result.diagnostics["stationary_reference"] == "unbiased"
    assert result.diagnostics["unbiased_green_poles"] == 2
    assert result.diagnostics["biased_green_poles"] == 3
    assert result.diagnostics["upward_A0_scaled_error"] <= 1.0
    assert result.diagnostics["upward_C0_scaled_error"] <= 1.0
    assert result.occupation[0] == pytest.approx(stationary_occupation, abs=5e-5)
    assert all(np.all(np.isfinite(values)) for values in result.currents.values())
    assert np.all(np.isfinite(result.occupation))
    assert np.all(np.isfinite(result.continuity_residual))
    with pytest.raises(RuntimeError, match="does not match"):
        current_all(
            sys,
            t_max=0.01,
            n_t=2,
            omega_int_n_omega=21,
            protocol="downward",
        )
