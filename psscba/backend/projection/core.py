from __future__ import annotations

import numpy as np
from scipy.signal import czt

from psscba.types import ProjectionResult


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 2 or not np.allclose(np.diff(grid), grid[1] - grid[0]):
        raise ValueError("A uniform grid with at least two points is required.")
    weights = np.full(len(grid), float(grid[1] - grid[0]))
    weights[[0, -1]] *= 0.5
    return weights


def toeplitz_from_lags(positive: np.ndarray, *, lesser: bool) -> np.ndarray:
    positive = np.asarray(positive, dtype=np.complex128)
    n = len(positive)
    out = np.empty((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            lag = i - j
            if lag >= 0:
                out[i, j] = positive[lag]
            elif lesser:
                out[i, j] = -np.conjugate(positive[-lag])
            else:
                out[i, j] = 0.0j
    return out


def direct_toeplitz_projection(matrix: np.ndarray, *, lesser: bool) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Projection input must be a square two-time matrix.")
    n = len(matrix)
    positive = np.asarray(
        [np.mean(np.diag(matrix, k=-lag)) for lag in range(n)],
        dtype=np.complex128,
    )
    return toeplitz_from_lags(positive, lesser=lesser)


def projector_errors(matrix: np.ndarray, *, lesser: bool) -> tuple[float, float]:
    projected = direct_toeplitz_projection(matrix, lesser=lesser)
    projected_twice = direct_toeplitz_projection(projected, lesser=lesser)
    complement = matrix - projected
    projected_complement = direct_toeplitz_projection(complement, lesser=lesser)
    scale = max(float(np.linalg.norm(projected)), np.finfo(float).tiny)
    return (
        float(np.linalg.norm(projected_twice - projected) / scale),
        float(np.linalg.norm(projected_complement) / scale),
    )


def reconstruct_retarded_rows(
    amplitudes: np.ndarray,
    energy: np.ndarray,
    time: np.ndarray,
    phase_integral: np.ndarray,
    *,
    dense: bool,
) -> tuple[np.ndarray, np.ndarray | None, float, np.ndarray]:
    """Inverse-transform A rows and accumulate the uniform Toeplitz projection."""
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)
    energy = np.asarray(energy, dtype=float)
    time = np.asarray(time, dtype=float)
    n_t, n_e = amplitudes.shape
    if n_e != len(energy) or n_t != len(time):
        raise ValueError("A amplitudes must have shape (n_time, n_energy).")
    dt = float(time[1] - time[0])
    de = float(energy[1] - energy[0])
    weights = trapezoid_weights(energy) / de
    transform_step = np.exp(-1j * de * dt)
    lag = np.arange(n_t, dtype=float) * dt
    energy_origin = np.exp(-1j * energy[0] * lag)
    sums = np.zeros(n_t, dtype=np.complex128)
    counts = np.arange(n_t, 0, -1, dtype=float)
    total_norm = 0.0
    q_by_row = np.zeros(n_t, dtype=float)
    dense_matrix = np.zeros((n_t, n_t), dtype=np.complex128) if dense else None

    for row in range(n_t):
        transformed = czt(
            amplitudes[row] * weights,
            m=n_t,
            w=transform_step,
            a=1.0,
        )
        transformed *= de * energy_origin / (2.0 * np.pi)
        valid = row + 1
        arguments = row - np.arange(valid)
        phase = np.exp(-1j * (phase_integral[row] - phase_integral[arguments]))
        values = phase * transformed[:valid]
        sums[:valid] += values
        total_norm += float(np.sum(np.abs(values) ** 2))
        if dense_matrix is not None:
            dense_matrix[row, : row + 1] = values[::-1]

    projected = sums / counts
    projected_norm = float(np.sum(counts * np.abs(projected) ** 2))
    q_norm = np.sqrt(max(total_norm - projected_norm, 0.0))
    r_value = q_norm / max(np.sqrt(total_norm), np.finfo(float).tiny)
    if dense_matrix is not None:
        projected_matrix = toeplitz_from_lags(projected, lesser=False)
        q_by_row = np.sqrt(np.sum(np.abs(dense_matrix - projected_matrix) ** 2, axis=1))
    return projected, dense_matrix, float(r_value), q_by_row


def autocorrelation_projection(
    sources: list[tuple[np.ndarray, np.ndarray]],
    *,
    energy_batch: int = 64,
    progress=None,
) -> np.ndarray:
    """Project a sum of energy-factorized lesser Green functions by FFT."""
    if not sources:
        raise ValueError("At least one lesser source is required.")
    n_t = sources[0][0].shape[0]
    n_fft = 1 << (2 * n_t - 1).bit_length()
    sums = np.zeros(n_t, dtype=np.complex128)
    total_batches = sum((len(weights) + energy_batch - 1) // energy_batch for _, weights in sources)
    completed_batches = 0
    for amplitudes, weights in sources:
        amplitudes = np.asarray(amplitudes, dtype=np.complex128)
        weights = np.asarray(weights, dtype=np.complex128)
        if amplitudes.shape[0] != n_t or amplitudes.shape[1] != len(weights):
            raise ValueError("Lesser source weights must match its energy columns.")
        for start in range(0, len(weights), energy_batch):
            stop = min(start + energy_batch, len(weights))
            transformed = np.fft.fft(amplitudes[:, start:stop], n=n_fft, axis=0)
            correlation = np.fft.ifft(
                transformed * np.conjugate(transformed), axis=0
            )[:n_t]
            sums += correlation @ weights[start:stop]
            completed_batches += 1
            if progress is not None:
                progress("streaming_projection", {
                    "batch_completed": completed_batches,
                    "batch_total": total_batches,
                    "batch_unit": "energy batches",
                })
    return sums / np.arange(n_t, 0, -1, dtype=float)


def dense_lesser_from_sources(
    sources: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    n_t = sources[0][0].shape[0]
    out = np.zeros((n_t, n_t), dtype=np.complex128)
    for amplitudes, weights in sources:
        out += (amplitudes * weights[None, :]) @ np.conjugate(amplitudes).T
    return out


def relative_time_to_energy(
    retarded_positive: np.ndarray,
    lesser_positive: np.ndarray,
    lag: np.ndarray,
    energy: np.ndarray,
    *,
    energy_batch: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    lag = np.asarray(lag, dtype=float)
    energy = np.asarray(energy, dtype=float)
    dt_weights = trapezoid_weights(lag)
    retarded_energy = np.empty(len(energy), dtype=np.complex128)
    lesser_energy = np.empty(len(energy), dtype=np.complex128)
    full_lag = np.concatenate((-lag[:0:-1], lag))
    full_lesser = np.concatenate(
        (-np.conjugate(lesser_positive[:0:-1]), lesser_positive)
    )
    full_weights = trapezoid_weights(full_lag)
    for start in range(0, len(energy), energy_batch):
        stop = min(start + energy_batch, len(energy))
        e = energy[start:stop, None]
        retarded_energy[start:stop] = np.exp(1j * e * lag[None, :]) @ (
            retarded_positive * dt_weights
        )
        lesser_energy[start:stop] = np.exp(1j * e * full_lag[None, :]) @ (
            full_lesser * full_weights
        )
    return retarded_energy, lesser_energy


def make_projection_result(
    *,
    time: np.ndarray,
    energy: np.ndarray,
    retarded_positive: np.ndarray,
    lesser_positive: np.ndarray,
    r_retarded: float,
    r_lesser: float,
    q_retarded_time: np.ndarray,
    q_lesser_time: np.ndarray,
    diagnostics: dict,
) -> ProjectionResult:
    lag = np.arange(len(time), dtype=float) * float(time[1] - time[0])
    gr_e, gl_e = relative_time_to_energy(
        retarded_positive, lesser_positive, lag, energy
    )
    greater_e = gl_e + gr_e - np.conjugate(gr_e)
    occupation = float(np.real(-1j * lesser_positive[0]))
    return ProjectionResult(
        lag=lag,
        green_retarded=retarded_positive,
        green_lesser=lesser_positive,
        green_greater=greater_e,
        energy=energy,
        green_retarded_energy=gr_e,
        green_lesser_energy=gl_e,
        occupation=occupation,
        r_green_retarded=r_retarded,
        r_green_lesser=r_lesser,
        q_green_retarded_time=q_retarded_time,
        q_green_lesser_time=q_lesser_time,
        diagnostics=diagnostics,
    )
