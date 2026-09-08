from dataclasses import replace
from numpy.polynomial.legendre import leggauss

import numpy as np
import pytest

from psscba.backend.poles.minimal import Gbiased_R_mpm, Gfr_R_mpm, build_pole_cache
from psscba.backend.protocols.analytic import C_down, C_up
from psscba.backend.protocols.square import C_square
from psscba.backend.reconstruction import (
    explicit_phonon_lesser_from_branches,
    installed_lesser_from_factorized_source,
)
from psscba.backend.reconstruction.engine import _prehistory_stationary_projected_value
from psscba.backend.protocols.strategies import strategy_for
from psscba.backend.self_energy.functional import kernel_residual_details
from psscba.backend.stationary.initializer import initial_zero_pulse_kernel, make_system
from psscba.frontend.configuration import ProtocolConfig
from psscba.types import StationaryKernel
from tests.test_psscba_solver import small_config


def test_nonzero_frequency_c_uses_shifted_stationary_boundary():
    omega = 0.37
    for protocol in ("upward", "downward", "square"):
        config = small_config().case
        duration = 0.1 if protocol == "square" else None
        config = replace(
            config,
            protocol=ProtocolConfig(name=protocol, duration=duration),
        )
        system = make_system(config)
        kernel = initial_zero_pulse_kernel(config)
        cache = build_pole_cache(system, kernel)
        energy = config.numerics.energy_grid()[:9]
        strategy = strategy_for(config, system, cache, energy)
        amplitudes = strategy.solve(np.asarray([-0.2, 0.0, 0.1]), c_omega=omega)
        if protocol == "downward":
            expected = Gbiased_R_mpm(system, cache, energy + omega)
        else:
            expected = Gfr_R_mpm(system, cache, energy + omega)
        np.testing.assert_allclose(amplitudes.c[0], expected, rtol=0.0, atol=1e-13)


def test_both_phonon_c_branches_match_protocol_formulas_through_turnoff():
    config = small_config().case
    config = replace(
        config,
        protocol=ProtocolConfig(name="square", duration=0.1),
    )
    system = make_system(config)
    kernel = initial_zero_pulse_kernel(config)
    cache = build_pole_cache(system, kernel)
    energy = config.numerics.energy_grid()[:9]
    times = np.asarray([-0.2, 0.0, 0.05, 0.1, 0.2])
    strategy = strategy_for(config, system, cache, energy)
    square_cache = strategy.protocol_cache
    for frequency in (-float(config.model.phonon_energy), float(config.model.phonon_energy)):
        observed = strategy.solve(times, c_omega=frequency).c
        expected = np.stack([
            (
                Gfr_R_mpm(system, cache, energy + frequency)
                if time < 0.0
                else np.asarray(
                    C_square(system, frequency, energy, float(time), cache, square_cache, 0.1)
                ).reshape(-1)
            )
            for time in times
        ])
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_both_phonon_c_branches_match_step_protocols():
    for protocol in ("upward", "downward"):
        config = replace(small_config(0.1).case, protocol=ProtocolConfig(name=protocol))
        system = make_system(config)
        kernel = initial_zero_pulse_kernel(config)
        cache = build_pole_cache(system, kernel)
        energy = config.numerics.energy_grid()[:9]
        times = np.asarray([-0.2, 0.0, 0.05, 0.15])
        strategy = strategy_for(config, system, cache, energy)
        for frequency in (-0.37, 0.37):
            observed = strategy.solve(times, c_omega=frequency).c
            if protocol == "upward":
                active = lambda t: C_up(system, frequency, energy, float(t), cache)
                stationary = Gfr_R_mpm(system, cache, energy + frequency)
            else:
                active = lambda t: C_down(system, frequency, energy, float(t), cache)
                stationary = Gbiased_R_mpm(system, cache, energy + frequency)
            expected = np.stack([
                stationary if time < 0.0 else np.asarray(active(time)).reshape(-1)
                for time in times
            ])
            # At t=0 the protocol formula applies its exact switching
            # boundary, while negative rows use the stationary history.
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_collision_prehistory_projected_sector_has_no_quadrature_node_jumps():
    """The stationary projected tail must remain continuous at node crossings.

    The mapped semi-infinite quadrature nodes cross the finite lag endpoint at
    distinct observation times.  A finite-lag interpolation followed by a
    stationary fallback therefore creates artificial jumps; the prehistory
    path must use the stationary representation on both sides of every one.
    """
    nodes, _ = leggauss(64)
    x = 0.5 * (nodes + 1.0)
    t_min, lag_max = -2.0, 5.0
    pre_time = t_min - x / (1.0 - x)
    crossing_times = lag_max + t_min - x / (1.0 - x)
    crossing_times = crossing_times[(crossing_times >= 0.0) & (crossing_times <= 3.0)]
    assert crossing_times.size > 0

    stationary = lambda lag: np.exp(-0.2 * np.asarray(lag)) + 1j * np.asarray(lag) ** 2
    for crossing in crossing_times:
        # Locate the corresponding node from the predicted crossing time.
        node_index = int(np.argmin(np.abs((lag_max + t_min - x / (1.0 - x)) - crossing)))
        before = crossing - 1e-8 - pre_time[node_index]
        after = crossing + 1e-8 - pre_time[node_index]
        values = _prehistory_stationary_projected_value(
            np.asarray([before, after]), stationary
        )
        np.testing.assert_allclose(values, stationary(np.asarray([before, after])))
        assert abs(values[1] - values[0]) < 1e-5


def test_kernel_residual_is_normalized_by_installed_kernel():
    energy = np.linspace(-1.0, 1.0, 5)
    zeros = np.zeros(5, dtype=np.complex128)

    def make(sigma_r, sigma_l):
        return StationaryKernel(
            energy=energy,
            sigma_retarded=np.asarray(sigma_r, dtype=np.complex128),
            sigma_lesser=np.asarray(sigma_l, dtype=np.complex128),
            sigma_hartree=0.0,
            sigma_dynamic_retarded=np.asarray(sigma_r, dtype=np.complex128),
            spectral_width=np.zeros(5),
            phonon_occupation=0.0,
            method="test",
        )

    installed = make(np.ones(5), np.ones(5))
    candidate = make(1.1 * np.ones(5), 1.2 * np.ones(5))
    values = kernel_residual_details(installed, candidate, 1e-12)
    assert values[0] == pytest.approx(0.1)
    assert values[1] == pytest.approx(0.2)
    assert values[3] == pytest.approx(1.0)


def test_explicit_phonon_lesser_branch_contraction_matches_definition():
    time = np.asarray([-0.1, 0.0, 0.2])
    energy = np.asarray([-0.5, 0.0, 0.5])
    omega = 0.3
    n0 = 0.2
    coupling = 0.4
    projected = 0.1j * np.asarray([1.0, 2.0, 3.0])
    branches = {
        -omega: np.asarray([[1.0 + 0.1j, 0.8, 0.4j], [1.1, 0.7j, 0.3], [0.9, 0.6, 0.2j]]),
        omega: np.asarray([[0.5, 0.3j, 0.2], [0.4, 0.2, 0.1j], [0.3j, 0.1, 0.05]]),
    }
    observed = explicit_phonon_lesser_from_branches(
        time, energy, branches, projected, coupling, omega, n0
    )
    quadrature = np.asarray([0.25, 0.5, 0.25]) / (2.0 * np.pi)
    expected = np.zeros((len(time), len(time)), dtype=complex)
    for frequency, values in branches.items():
        weight = n0 + 1.0 if frequency < 0 else n0
        for i, ti in enumerate(time):
            for j, tj in enumerate(time):
                expected[i, j] += np.sum(
                    weight
                    * coupling**2
                    * projected
                    * quadrature
                    * values[i]
                    * np.conjugate(values[j])
                    * np.exp(-1j * (energy + frequency) * (ti - tj))
                )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)


def test_generic_installed_and_explicit_branches_agree_for_constant_kernel():
    # A constant projected lesser function removes only finite-support edge
    # effects, exposing the exact change-of-variable identity between the two
    # representations even when omega/dE is non-integer.
    # At zero relative time the finite-domain change of variables is exact;
    # a separate energy-window study covers the oscillatory tail error.
    time = np.asarray([0.0, 0.0, 0.0])
    energy = np.linspace(-4.0, 4.0, 81)
    omega = 0.37
    n0 = 0.2
    coupling = 0.4
    projected = 0.3j * np.ones(len(energy))
    c_zero = np.ones((len(time), len(energy)), dtype=complex)
    branches = {-omega: c_zero, omega: c_zero}
    sigma_lesser = coupling**2 * ((n0 + 1.0) + n0) * projected
    generic = installed_lesser_from_factorized_source(
        time, energy, c_zero, sigma_lesser
    )
    explicit = explicit_phonon_lesser_from_branches(
        time, energy, branches, projected, coupling, omega, n0
    )
    np.testing.assert_allclose(generic, explicit, rtol=0.0, atol=2e-12)
