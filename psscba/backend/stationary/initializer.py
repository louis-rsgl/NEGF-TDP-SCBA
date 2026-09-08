from __future__ import annotations

from dataclasses import dataclass

from dataclasses import replace

import numpy as np

from psscba.backend.model.system import LeadParams, System
from psscba.backend.stationary.scba import Solver
from psscba.frontend.configuration import CaseConfig
from psscba.types import StationaryKernel


@dataclass(frozen=True)
class StationaryInitializer:
    """Build stationary initial or preparation-fixed kernels for one case."""

    config: CaseConfig

    def zero_pulse(self) -> StationaryKernel:
        return initial_zero_pulse_kernel(self.config)

    def preparation_fixed(self) -> StationaryKernel:
        return preparation_fixed_kernel(self.config)


def make_system(config: CaseConfig, *, zero_voltage: bool = False) -> System:
    """Construct the scalar backend model in dimensionless Gamma units."""
    model = config.model
    numerics = config.numerics
    protocol_name = config.protocol.name
    backend_protocol = "upward" if protocol_name == "zero" else protocol_name
    device_shift = 0.0 if zero_voltage or protocol_name == "zero" else model.device_shift
    left_shift = 0.0 if zero_voltage or protocol_name == "zero" else model.left_shift
    right_shift = 0.0 if zero_voltage or protocol_name == "zero" else model.right_shift
    grid = numerics.energy_grid()
    return System(
        pulse_protocol=backend_protocol,
        pulse_duration=(
            config.protocol.duration if backend_protocol == "square" else None
        ),
        ETA=numerics.eta,
        DELTA=device_shift,
        leads={
            "L": LeadParams(model.gamma_left, left_shift, model.beta, 0.0),
            "R": LeadParams(model.gamma_right, right_shift, model.beta, 0.0),
        },
        W=model.bandwidth,
        g_q=model.coupling,
        w_q=model.phonon_energy,
        e_0=model.epsilon_0,
        beta_ph=model.beta_ph,
        mu_ph=0.0,
        N0=model.phonon_occupation,
        beta_fd=model.beta,
        mu_fd=0.0,
        e_min=float(grid[0]),
        e_max=float(grid[-1]),
        omega_min=float(grid[0]),
        omega_max=float(grid[-1]),
        scba_max_iter=numerics.stationary_max_iter,
        scba_mode="self_consistent",
        scba_tol_abs=numerics.stationary_tolerance_abs,
        scba_tol_rel=numerics.stationary_tolerance_rel,
        scba_mixing=numerics.stationary_mixing,
        scba_min_iter=5,
        n_w_scba=len(grid),
        mpm_tol=numerics.mpm_search_tolerance,
        mpm_n_iw=numerics.mpm_n_iw,
        mpm_beta_fit=numerics.mpm_beta_fit,
        mpm_aaa_rtol=numerics.mpm_search_tolerance,
        mpm_aaa_initial_terms=numerics.mpm_aaa_initial_terms,
        mpm_aaa_max_terms=numerics.mpm_aaa_max_terms,
        mpm_aaa_growth_factor=numerics.mpm_aaa_growth_factor,
        square_residue_method=numerics.square_residue_method,
        verbose=numerics.verbose,
    )


def kernel_from_solver(solver: Solver, *, method: str, protocol: str) -> StationaryKernel:
    if solver.installed_kernel is None or solver.result is None:
        raise RuntimeError("Stationary solver did not produce an interaction kernel.")
    installed = solver.installed_kernel
    return StationaryKernel(
        energy=installed.w,
        sigma_retarded=installed.Sigma_ep_R,
        sigma_lesser=installed.Sigma_ep_less,
        sigma_hartree=installed.Sigma_H,
        sigma_dynamic_retarded=installed.Sigma_ep_dyn_R,
        spectral_width=installed.Gamma_ep,
        phonon_occupation=installed.N0,
        projected_green_retarded=installed.G_reference_R,
        projected_green_lesser=installed.G_reference_less,
        method=method,
        protocol=protocol,
        metadata={
            "stationary_iterations": solver.result.n_iter,
            "stationary_retarded_residual": solver.result.res_GR_abs,
            "stationary_lesser_residual": solver.result.res_Gless_abs,
        },
    )


def initial_zero_pulse_kernel(config: CaseConfig) -> StationaryKernel:
    """Return the zero-voltage stationary SCBA initial guess for PS-SCBA."""
    system = make_system(config, zero_voltage=True)
    grid = config.numerics.energy_grid()
    solver = Solver(system, float(grid[0]), float(grid[-1]), len(grid))
    solver.solve()
    return kernel_from_solver(solver, method="stationary_scba_initial", protocol="zero")


def preparation_fixed_kernel(config: CaseConfig) -> StationaryKernel:
    """Historical protocol-preparation stationary kernel for comparisons."""
    system = make_system(config)
    grid = config.numerics.energy_grid()
    solver = Solver(system, float(grid[0]), float(grid[-1]), len(grid))
    solver.solve()
    return kernel_from_solver(
        solver, method="preparation_fixed", protocol=config.protocol.name
    )


def with_mpm_tolerance(system: System, tolerance: float) -> System:
    system.mpm_tol = float(tolerance)
    system._pole_cache = None
    return system


def mix_kernels(
    installed: StationaryKernel,
    candidate: StationaryKernel,
    alpha: float,
    iteration: int,
) -> StationaryKernel:
    if not np.array_equal(installed.energy, candidate.energy):
        raise ValueError("Cannot mix stationary kernels on different grids.")
    sigma_r = (1.0 - alpha) * installed.sigma_retarded + alpha * candidate.sigma_retarded
    sigma_l = (1.0 - alpha) * installed.sigma_lesser + alpha * candidate.sigma_lesser
    sigma_h = (1.0 - alpha) * installed.sigma_hartree + alpha * candidate.sigma_hartree
    sigma_dyn = sigma_r - sigma_h
    width = -2.0 * np.imag(sigma_dyn)
    return StationaryKernel(
        energy=installed.energy,
        sigma_retarded=sigma_r,
        sigma_lesser=1j * np.imag(sigma_l),
        sigma_hartree=float(np.real(sigma_h)),
        sigma_dynamic_retarded=sigma_dyn,
        spectral_width=np.real(width),
        phonon_occupation=installed.phonon_occupation,
        projected_green_retarded=candidate.projected_green_retarded,
        projected_green_lesser=candidate.projected_green_lesser,
        method="psscba",
        protocol=candidate.protocol,
        iteration=iteration,
        metadata=candidate.metadata,
    )
