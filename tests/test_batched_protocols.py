from dataclasses import replace

import numpy as np

from psscba.backend.poles.minimal import build_pole_cache
from psscba.backend.protocols.analytic import A_down, A_up, C_down, C_up
from psscba.backend.protocols.square import A_square, C_square
from psscba.backend.protocols.strategies import strategy_for
from psscba.backend.stationary.initializer import initial_zero_pulse_kernel, make_system
from psscba.frontend.configuration import ProjectionConfig, ProtocolConfig
from tests.test_psscba_solver import small_config


def test_blocked_upward_and_downward_pole_sums_match_scalar_calls():
    times = np.asarray([-0.2, 0.0, 0.05, 0.1, 0.2])
    for name, a_scalar, c_scalar in (("upward", A_up, C_up), ("downward", A_down, C_down)):
        config = small_config().case
        config = replace(
            config,
            protocol=ProtocolConfig(name=name),
            projection=replace(config.projection, energy_batch=7, amplitude_time_batch=2),
        )
        system = make_system(config)
        kernel = initial_zero_pulse_kernel(config)
        cache = build_pole_cache(system, kernel)
        energy = config.numerics.energy_grid()[:13]
        amplitudes = strategy_for(config, system, cache, energy).solve(times)
        expected_a = np.stack([
            np.asarray(a_scalar(system, energy, float(t), "L", cache)).reshape(-1)
            for t in times[1:]
        ])
        expected_c = np.stack([
            np.asarray(c_scalar(system, 0.0, energy, float(t), cache)).reshape(-1)
            for t in times[1:]
        ])
        np.testing.assert_allclose(amplitudes.a["L"][1:], expected_a, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(amplitudes.c[1:], expected_c, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(amplitudes.a["L"][0], amplitudes.a["L"][1], rtol=0.0, atol=1e-12)


def test_blocked_square_pole_sums_match_scalar_calls_at_turnoff():
    config = small_config().case
    config = replace(
        config,
        protocol=ProtocolConfig(name="square", duration=0.1),
        projection=replace(config.projection, energy_batch=7, amplitude_time_batch=2),
    )
    system = make_system(config)
    kernel = initial_zero_pulse_kernel(config)
    cache = build_pole_cache(system, kernel)
    energy = config.numerics.energy_grid()[:13]
    strategy = strategy_for(config, system, cache, energy)
    times = np.asarray([-0.2, 0.0, 0.05, 0.1, 0.2])
    amplitudes = strategy.solve(times)
    square_cache = strategy.protocol_cache
    expected_a = np.stack([
        np.asarray(A_square(system, energy, float(t), "L", cache, square_cache, 0.1)).reshape(-1)
        for t in times[1:]
    ])
    expected_c = np.stack([
        np.asarray(C_square(system, 0.0, energy, float(t), cache, square_cache, 0.1)).reshape(-1)
        for t in times[1:]
    ])
    np.testing.assert_allclose(amplitudes.a["L"][1:], expected_a, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(amplitudes.c[1:], expected_c, rtol=0.0, atol=1e-12)
