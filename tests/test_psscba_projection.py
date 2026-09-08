import numpy as np
import pytest

from psscba.backend.protocols.analytic import integrate_cauchy_linear
from psscba.backend.reconstruction.engine import cauchy_linear_uniform, finite_window_psi
from psscba.backend.projection.core import (
    autocorrelation_projection,
    dense_lesser_from_sources,
    direct_toeplitz_projection,
    projector_errors,
    reconstruct_retarded_rows,
)


def test_stationary_projector_idempotency_orthogonality_and_causality():
    time = np.linspace(-1.0, 1.0, 33)
    difference = time[:, None] - time[None, :]
    retarded = np.where(difference >= 0.0, -1j * np.exp(-difference), 0.0)
    projected = direct_toeplitz_projection(retarded, lesser=False)
    assert np.linalg.norm(projected - retarded) / np.linalg.norm(retarded) < 1e-14
    assert max(projector_errors(retarded, lesser=False)) < 1e-14
    assert np.max(np.abs(np.triu(projected, k=1))) == 0.0


def test_lesser_streaming_matches_dense_diagonal_average():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(31, 17)) + 1j * rng.normal(size=(31, 17))
    weights = 1j * rng.random(17)
    sources = [(source, weights)]
    dense = dense_lesser_from_sources(sources)
    projected = direct_toeplitz_projection(dense, lesser=True)
    streaming = autocorrelation_projection(sources, energy_batch=5)
    assert np.max(np.abs(np.diag(projected, k=0) - streaming[0])) < 1e-12
    for lag in range(31):
        assert np.max(np.abs(np.diag(projected, k=-lag) - streaming[lag])) < 1e-12
    assert np.linalg.norm(dense + np.conjugate(dense.T)) < 1e-12


def test_czt_retarded_reconstruction_matches_direct_quadrature():
    energy = np.linspace(-8.0, 8.0, 321)
    time = np.linspace(-0.2, 0.2, 5)
    pole = 0.3 - 0.7j
    amplitudes = np.tile(1.0 / (energy - pole), (len(time), 1))
    positive, dense, _, _ = reconstruct_retarded_rows(
        amplitudes, energy, time, np.zeros(len(time)), dense=True
    )
    weights = np.full(len(energy), energy[1] - energy[0])
    weights[[0, -1]] *= 0.5
    expected = np.array([
        np.sum(weights * np.exp(-1j * energy * lag) / (energy - pole)) / (2 * np.pi)
        for lag in np.arange(len(time)) * (time[1] - time[0])
    ])
    assert np.max(np.abs(positive - expected)) < 1e-12


def test_mixed_lesser_czt_matches_direct_time_quadrature():
    time = np.linspace(-0.2, 0.2, 9)
    energy = np.linspace(-2.0, 2.0, 9)
    difference = time[:, None] - time[None, :]
    lesser = 1j * np.exp(-0.3j * difference)
    positive = np.flatnonzero(time >= 0.0)
    transformed = finite_window_psi(
        lesser,
        time,
        energy,
        np.zeros_like(time),
        positive,
        batch_size=3,
    )
    for output_index, row in enumerate(positive):
        weights = np.full(row + 1, time[1] - time[0])
        weights[[0, -1]] *= 0.5
        for energy_index, value in enumerate(energy):
            expected = np.sum(
                weights
                * np.exp(1j * value * (time[row] - time[: row + 1]))
                * lesser[row, : row + 1]
            )
            assert transformed[output_index, energy_index] == pytest.approx(
                expected, abs=1e-12
            )


@pytest.mark.parametrize("pole_shift", [0.0, 0.35, -0.2])
def test_fft_linear_cauchy_matches_literal_segment_integral(pole_shift):
    grid = np.linspace(-3.0, 3.0, 121)
    values = np.vstack(
        [
            np.exp(-0.4 * grid**2) * (1.0 + 0.2j * grid),
            np.sin(grid) + 0.3j * np.cos(2.0 * grid),
        ]
    )
    fast = cauchy_linear_uniform(values, grid, pole_shift)
    for row in range(len(values)):
        direct = integrate_cauchy_linear(
            grid + pole_shift,
            grid,
            values[row],
        )
        assert np.max(np.abs(fast[row] - direct)) < 2e-12
