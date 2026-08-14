from types import SimpleNamespace

import numpy as np
import pytest

import backend.minimal_poles as minimal_poles
from backend.minimal_poles import (
    Gbar_R_mpm,
    _aaa_term_schedule,
    _filter_green_poles,
    green_poles_from_sigma_mpm,
    green_residues,
    sigma_mpm,
)
from tests.test_frozen_scba import make_system


def test_sigma_reconstruction_and_arrowhead_roots():
    sys = make_system()
    zeta = np.array([-0.4 - 0.3j, 0.7 - 0.5j])
    weights = np.array([0.08 + 0.01j, 0.03 - 0.02j])
    xi = green_poles_from_sigma_mpm(sys, -0.02, zeta, weights)
    gamma_total = sum(sys.Gamma0(a) for a in sys.lead_names)
    denominator = (
        xi - sys.e_0 + 0.02
        - 0.5 * gamma_total * sys.W / (xi + 1j * sys.W)
        - sigma_mpm(xi, zeta, weights)
    )
    residues = green_residues(sys, xi, zeta, weights)
    assert np.max(np.abs(denominator)) < 1e-10
    assert np.all(np.isfinite(residues))


def test_g_zero_cache_has_two_maciejko_poles():
    sys = make_system()
    sys.solve_noneq()
    cache = sys.prepare_poles()
    assert cache.n_sigma_poles == 0
    assert cache.n_green_poles == 2
    assert cache.n_unbiased_green_poles == 2
    assert cache.n_biased_green_poles == 3
    assert np.all(cache.xi.imag < 0.0)
    assert cache.max_sigma_abs_error == 0.0
    assert np.max(np.abs(Gbar_R_mpm(sys, cache, sys.frozen_scba().w) - sys.frozen_scba().Gbar_R)) < 2e-3


def test_upward_cache_activates_three_biased_poles_and_residues():
    sys = make_system(pulse_protocol="upward", scba_mode="weak_born")
    sys.solve_stationary()
    cache = sys.prepare_poles()
    assert cache.n_sigma_poles == 0
    assert cache.n_unbiased_green_poles == 2
    assert cache.n_biased_green_poles == 3
    assert np.array_equal(cache.xi, cache.xi_biased)
    denominator = (
        cache.xi_biased
        - sys.e_0
        - sys.DELTA
        - sum(
            0.5 * sys.Gamma0(lead) * sys.W
            / (cache.xi_biased - sys.Delta(lead) + 1j * sys.W)
            for lead in sys.lead_names
        )
    )
    assert np.max(np.abs(denominator)) < 1e-10
    expected_residues = 1.0 / (
        1.0
        + sum(
            0.5 * sys.Gamma0(lead) * sys.W
            / (cache.xi_biased - sys.Delta(lead) + 1j * sys.W) ** 2
            for lead in sys.lead_names
        )
    )
    assert np.max(np.abs(cache.residues_biased - expected_residues)) < 1e-12


def test_green_pole_causality_and_degeneracy_failures():
    sys = make_system()
    with pytest.raises(RuntimeError, match="upper-half-plane"):
        _filter_green_poles(
            sys,
            np.array([0.1 + 0.2j]),
            np.array([1.0 + 0.0j]),
        )
    with pytest.raises(RuntimeError, match="Nearly degenerate"):
        _filter_green_poles(
            sys,
            np.array([0.1 - 0.2j, 0.1 + 1e-10 - 0.2j]),
            np.array([0.5 + 0.0j, 0.5 + 0.0j]),
        )


def test_adaptive_aaa_term_schedule_is_general_and_bounded():
    assert _aaa_term_schedule(80, 240, 1.5) == (80, 120, 180, 240)
    assert _aaa_term_schedule(120, 120, 2.0) == (120,)
    with pytest.raises(ValueError, match="initial_terms"):
        _aaa_term_schedule(1, 120, 1.5)
    with pytest.raises(ValueError, match="max_terms"):
        _aaa_term_schedule(121, 120, 1.5)
    with pytest.raises(ValueError, match="growth_factor"):
        _aaa_term_schedule(80, 120, 1.0)


def test_pole_builder_grows_aaa_until_full_grid_validation_passes(monkeypatch):
    calls = []

    def candidate(_sys, _frozen, _tolerance, fit_method="minipole", **kwargs):
        terms = kwargs.get("aaa_max_terms", 0)
        calls.append((fit_method, terms))
        scaled = 0.5 if fit_method == "causal_aaa" and terms >= 120 else 2.0
        return SimpleNamespace(
            max_sigma_abs_error=scaled,
            max_sigma_rel_error=scaled,
            max_sigma_scaled_error=scaled,
            max_Gfr_abs_error=scaled,
            max_Gbar_abs_error=scaled,
            max_Gfr_scaled_error=scaled,
            max_Gbar_scaled_error=scaled,
            fit_terms=terms,
            fit_converged=terms >= 120,
        )

    monkeypatch.setattr(minimal_poles, "_build_candidate", candidate)
    sys = SimpleNamespace(
        g_q=1.0,
        mpm_tol=1e-8,
        mpm_aaa_rtol=1e-6,
        mpm_aaa_initial_terms=80,
        mpm_aaa_max_terms=240,
        mpm_aaa_growth_factor=1.5,
    )
    frozen = SimpleNamespace(w=np.linspace(-1.0, 1.0, 1001))

    cache = minimal_poles.build_pole_cache(sys, frozen)

    assert cache.max_sigma_scaled_error == 0.5
    assert calls == [
        ("minipole", 0),
        ("minipole", 0),
        ("causal_aaa", 80),
        ("causal_aaa", 120),
    ]


def test_upward_nonzero_coupling_reconstructs_both_propagators():
    pytest.importorskip("mini_pole")
    sys = make_system(
        pulse_protocol="upward",
        scba_mode="weak_born",
        g_q=0.02,
        n_w_scba=121,
        mpm_n_iw=128,
        mpm_tol=1e-6,
        mpm_fit_abs_tol=1e-3,
        mpm_fit_rel_tol=0.5,
        mpm_green_rel_tol=0.1,
    )
    sys.solve_stationary()
    cache = sys.prepare_poles()
    assert cache.n_sigma_poles > 0
    assert cache.n_biased_green_poles == cache.n_unbiased_green_poles + 1
    assert cache.max_sigma_scaled_error <= 1.0
    assert cache.max_Gfr_scaled_error <= 1.0
    assert cache.max_Gbar_scaled_error <= 1.0
