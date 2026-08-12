from __future__ import annotations

import numpy as np
import scipy.constants as const


HBAR_EV_S = const.hbar / const.e  # [eV*s]


def _positive_gamma(gamma_eV: float) -> float:
    gamma = float(gamma_eV)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"Gamma must be finite and positive in eV; got {gamma_eV}.")
    return gamma


def energy_ev_to_gamma(energy_eV, gamma_eV: float):
    """Convert a physical energy in eV to the backend's E/Gamma units."""
    return np.asarray(energy_eV) / _positive_gamma(gamma_eV)


def energy_mev_to_gamma(energy_meV, gamma_eV: float):
    """Convert a physical energy in meV to the backend's E/Gamma units."""
    return energy_ev_to_gamma(1e-3 * np.asarray(energy_meV), gamma_eV)


def energy_gamma_to_ev(energy_gamma, gamma_eV: float):
    """Convert a backend energy E/Gamma to eV."""
    return np.asarray(energy_gamma) * _positive_gamma(gamma_eV)


def energy_gamma_to_mev(energy_gamma, gamma_eV: float):
    """Convert a backend energy E/Gamma to meV."""
    return 1e3 * energy_gamma_to_ev(energy_gamma, gamma_eV)


def gamma_to_time_unit_s(gamma_eV: float) -> float:
    """
    Natural time unit corresponding to dimensionless t=1:
        t_phys = t * (ħ / Γ)
    with Γ given in eV.
    """
    return HBAR_EV_S / _positive_gamma(gamma_eV)


def gamma_to_current_unit_A(gamma_eV: float) -> float:
    """
    Natural current unit corresponding to dimensionless I=1:
        I_phys = I * (e Γ / ħ)
    with Γ given in eV.
    """
    return const.e * _positive_gamma(gamma_eV) / HBAR_EV_S


def time_to_si_seconds(t_dimless, gamma_eV: float):
    return np.asarray(t_dimless) * gamma_to_time_unit_s(gamma_eV)


def time_to_ps(t_dimless, gamma_eV: float):
    return 1e12 * time_to_si_seconds(t_dimless, gamma_eV)


def current_to_si_ampere(I_dimless, gamma_eV: float):
    return np.asarray(I_dimless) * gamma_to_current_unit_A(gamma_eV)


def current_to_uA(I_dimless, gamma_eV: float):
    return 1e6 * current_to_si_ampere(I_dimless, gamma_eV)
