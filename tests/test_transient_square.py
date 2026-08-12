import numpy as np
import pytest

from backend.minimal_poles import Gfr_R_mpm
from backend.observables import (
    A_up,
    B_square,
    C_up,
    D_square,
    current_all,
    square_time_grid,
)
from backend.square_pulse import (
    A_square,
    C_square,
    _biased_green_times_lead_rational,
    build_square_kernel_cache,
    lower_half_plane_residue,
)
from tests.test_frozen_scba import make_system


def make_square_system(**changes):
    values = dict(
        pulse_protocol="square", pulse_duration=0.4, scba_mode="weak_born"
    )
    values.update(changes)
    return make_system(**values)


def test_square_stationary_reference_and_two_pole_hierarchies():
    square = make_square_system()
    upward = make_system(pulse_protocol="upward", scba_mode="weak_born")
    square.solve_stationary()
    upward.solve_stationary()
    assert not square.frozen_scba().reference_is_biased
    assert np.allclose(
        square.frozen_scba().G_reference_R,
        upward.frozen_scba().G_reference_R,
    )
    cache = square.prepare_poles()
    assert cache.n_unbiased_green_poles == 2
    assert cache.n_biased_green_poles == 3
    assert np.array_equal(cache.xi, cache.xi_unbiased)


def test_square_time_grid_contains_turnoff_and_validates_duration():
    grid = square_time_grid(1.0, 8, 0.37)
    assert len(grid) == 8
    assert np.count_nonzero(grid == 0.37) == 1
    assert np.all(np.diff(grid) > 0.0)
    with pytest.raises(ValueError, match="duration"):
        square_time_grid(1.0, 8, 1.1)


def test_complete_integrand_residue_keeps_and_cancels_candidates():
    pole = 0.2 - 0.4j
    residue = lower_half_plane_residue(
        lambda z: 2.5 / (z - pole), [pole], n_theta=64
    )
    assert residue == pytest.approx(-2.5, abs=1e-10)
    removable = lower_half_plane_residue(
        lambda z: (z - pole) / (z - pole), [pole], n_theta=64
    )
    assert abs(removable) < 1e-10


def test_lead_pole_zero_is_evaluated_as_one_rational_product():
    sys = make_square_system()
    sys.solve_stationary()
    cache = sys.prepare_poles()
    # At the unshifted Lorentzian embedding pole Gbiased tends to zero.  The
    # combined expression must remain exactly finite rather than evaluating a
    # divergent Dyson term and a Green zero separately.
    value = _biased_green_times_lead_rational(
        sys, cache, -1j * sys.W, 1.0, {}
    )
    assert value == 0.0j


def test_negligible_outer_pole_uses_propagated_residue_error():
    pole = -1.0j
    value = lower_half_plane_residue(
        lambda z: 1e14 + 2.5 / (z - pole),
        [pole],
        n_theta=64,
        contribution_weight=1e-20,
    )
    assert np.isfinite(value)


def test_square_matches_upward_and_is_continuous_at_turnoff():
    sys = make_square_system()
    sys.solve_stationary()
    cache = sys.prepare_poles()
    energies = np.linspace(-1.0, 1.0, 7)
    kernels = build_square_kernel_cache(sys, energies, cache=cache)
    for time in (0.0, 0.2, sys.pulse_duration):
        assert np.array_equal(
            A_square(sys, energies, time, "L", cache, kernels),
            A_up(sys, energies, time, "L", cache),
        )
        assert np.array_equal(
            C_square(sys, sys.w_q, energies, time, cache, kernels),
            C_up(sys, sys.w_q, energies, time, cache),
        )
    assert kernels.turnoff_A_scaled_error <= 1.0
    assert kernels.turnoff_C_scaled_error <= 1.0
    assert np.max(np.abs(B_square(
        sys, energies, energies, sys.pulse_duration, "L", cache, kernels
    ))) == 0.0
    assert np.max(np.abs(D_square(
        sys, sys.w_q, energies, energies, sys.pulse_duration, "L", cache, kernels
    ))) == 0.0


def test_zero_duration_has_no_retarded_transient():
    sys = make_square_system(pulse_duration=0.0)
    sys.solve_stationary()
    cache = sys.prepare_poles()
    energies = np.linspace(-1.0, 1.0, 5)
    kernels = build_square_kernel_cache(sys, energies, cache=cache)
    expected = Gfr_R_mpm(sys, cache, energies + 1e-14j)
    assert np.allclose(A_square(sys, energies, 0.3, "L", cache, kernels), expected)
    assert np.allclose(
        C_square(sys, sys.w_q, energies, 0.3, cache, kernels),
        Gfr_R_mpm(sys, cache, energies + sys.w_q + 1e-14j),
    )


def test_small_square_current_dispatch_and_diagnostics():
    sys = make_square_system(
        pulse_duration=0.005,
        n_w_scba=31,
        current_energy_batch=7,
        e_min=-4.0,
        e_max=4.0,
        omega_min=-8.0,
        omega_max=8.0,
    )
    result = current_all(sys, t_max=0.01, n_t=3, omega_int_n_omega=21)
    assert result.t[1] == sys.pulse_duration
    assert result.diagnostics["pulse_protocol"] == "square"
    assert result.diagnostics["pulse_duration"] == sys.pulse_duration
    assert result.diagnostics["square_turnoff_A_scaled_error"] <= 1.0
    assert result.diagnostics["square_turnoff_C_scaled_error"] <= 1.0
    assert all(np.all(np.isfinite(values)) for values in result.currents.values())
    assert np.all(np.isfinite(result.occupation))
    with pytest.raises(RuntimeError, match="does not match"):
        current_all(sys, 0.01, 3, omega_int_n_omega=11, protocol="upward")
