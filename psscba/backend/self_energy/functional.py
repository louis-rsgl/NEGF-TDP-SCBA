"""Projected stationary Hartree-Fock SCBA functional.

This module owns the map from a projected Green function to a stationary
interaction kernel. It knows nothing about protocols, tracking, storage, or
campaign execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psscba.backend.stationary.scba import Solver
from psscba.frontend.configuration import CaseConfig, RunConfig
from psscba.types import FixedKernelTrajectory, StationaryKernel


def _interpolate(values: np.ndarray, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (
        np.interp(points, grid, values.real, left=0.0, right=0.0)
        + 1j * np.interp(points, grid, values.imag, left=0.0, right=0.0)
    )


def weighted_rms(values: np.ndarray, grid: np.ndarray) -> float:
    weights = np.full(len(grid), float(grid[1] - grid[0]))
    weights[[0, -1]] *= 0.5
    return float(np.sqrt(np.sum(weights * np.abs(values) ** 2) / np.sum(weights)))


@dataclass(frozen=True)
class KernelResidual:
    retarded: float
    lesser: float
    retarded_numerator: float
    retarded_denominator: float
    lesser_numerator: float
    lesser_denominator: float


class SCBAFunctional:
    """Construct and compare one protocol-projected SCBA kernel candidate."""

    def __init__(self, config: CaseConfig | RunConfig) -> None:
        self.config = config

    def candidate(
        self,
        trajectory: FixedKernelTrajectory,
        installed: StationaryKernel,
    ) -> StationaryKernel:
        config = self.config
        energy = trajectory.projected.energy
        gr = trajectory.projected.green_retarded_energy
        gl = 1j * np.imag(trajectory.projected.green_lesser_energy)
        greater = gl + gr - np.conjugate(gr)
        g2 = config.model.coupling**2
        omega = config.model.phonon_energy
        n0 = installed.phonon_occupation
        if g2 == 0.0:
            zero = np.zeros_like(gr)
            return StationaryKernel(
                energy=energy,
                sigma_retarded=zero,
                sigma_lesser=zero,
                sigma_hartree=0.0,
                sigma_dynamic_retarded=zero,
                spectral_width=np.zeros(len(energy)),
                phonon_occupation=n0,
                projected_green_retarded=gr,
                projected_green_lesser=gl,
                method="psscba_candidate",
                protocol=config.protocol.name,
                iteration=installed.iteration + 1,
            )

        gl_minus = _interpolate(gl, energy, energy - omega)
        gl_plus = _interpolate(gl, energy, energy + omega)
        gg_minus = _interpolate(greater, energy, energy - omega)
        gg_plus = _interpolate(greater, energy, energy + omega)
        sigma_lesser = g2 * ((n0 + 1.0) * gl_plus + n0 * gl_minus)
        sigma_greater = g2 * ((n0 + 1.0) * gg_minus + n0 * gg_plus)
        gamma_complex = 1j * (sigma_greater - sigma_lesser)
        imaginary_error = float(np.max(np.abs(gamma_complex.imag)))
        if imaginary_error > 1e-8 * max(1.0, float(np.max(np.abs(gamma_complex)))):
            raise RuntimeError(
                "Projected interaction spectral width is not real "
                f"(error={imaginary_error:.3e})."
            )
        gamma = np.real(gamma_complex)
        sigma_dynamic = Solver._hilbert_pv(gamma) - 0.5j * gamma
        occupation = float(trajectory.projected.occupation)
        sigma_hartree = -2.0 * g2 * occupation / omega
        return StationaryKernel(
            energy=energy,
            sigma_retarded=sigma_hartree + sigma_dynamic,
            sigma_lesser=1j * np.imag(sigma_lesser),
            sigma_hartree=sigma_hartree,
            sigma_dynamic_retarded=sigma_dynamic,
            spectral_width=gamma,
            phonon_occupation=n0,
            projected_green_retarded=gr,
            projected_green_lesser=gl,
            method="psscba_candidate",
            protocol=config.protocol.name,
            iteration=installed.iteration + 1,
            metadata={"projected_occupation": occupation},
        )

    def residual(
        self,
        installed: StationaryKernel,
        candidate: StationaryKernel,
    ) -> KernelResidual:
        values = kernel_residual_details(
            installed,
            candidate,
            self.config.numerics.sigma_floor,
        )
        return KernelResidual(*values)


def kernel_residual_details(installed, candidate, floor):
    numerator_r = weighted_rms(installed.sigma_retarded - candidate.sigma_retarded, installed.energy)
    numerator_l = weighted_rms(installed.sigma_lesser - candidate.sigma_lesser, installed.energy)
    # Normalize by the installed iterate, as defined for the fixed-point
    # residual.  Using the candidate norm makes the stopping criterion depend
    # on the proposed update and is especially ill-conditioned near g=0.
    denominator_r = max(weighted_rms(installed.sigma_retarded, installed.energy), floor)
    denominator_l = max(weighted_rms(installed.sigma_lesser, installed.energy), floor)
    residual_r = numerator_r / denominator_r
    residual_l = numerator_l / denominator_l
    if np.max(np.abs(candidate.sigma_retarded)) == 0.0:
        residual_r = 0.0 if np.max(np.abs(installed.sigma_retarded)) == 0.0 else residual_r
    if np.max(np.abs(candidate.sigma_lesser)) == 0.0:
        residual_l = 0.0 if np.max(np.abs(installed.sigma_lesser)) == 0.0 else residual_l
    return float(residual_r), float(residual_l), numerator_r, denominator_r, numerator_l, denominator_l


def kernel_residual(installed, candidate, floor):
    values = kernel_residual_details(installed, candidate, floor)
    return values[0], values[1]


def scba_candidate(config, trajectory, installed):
    return SCBAFunctional(config).candidate(trajectory, installed)
