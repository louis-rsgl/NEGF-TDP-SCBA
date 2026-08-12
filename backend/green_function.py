"""Analytic helpers for the phonon-free Lorentzian reference problem."""

from __future__ import annotations

import numpy as np

from backend.system_classes import Lead, System


def GR_eq(sys: System, w):
    """Return the unbiased, phonon-free retarded reference Green function."""
    z = np.asarray(w, dtype=np.complex128)
    gamma_total = sum(sys.Gamma0(lead) for lead in sys.lead_names)
    return 1.0 / (
        z - sys.e_0 - 0.5 * gamma_total * sys.W / (z + 1j * sys.W)
    )


def eq_poles_residues(sys: System):
    """Return the two Maciejko poles as ``(residue, pole)`` pairs."""
    gamma_total = sum(sys.Gamma0(lead) for lead in sys.lead_names)
    matrix = np.array(
        [
            [sys.e_0, 0.5 * gamma_total * sys.W],
            [1.0, -1j * sys.W],
        ],
        dtype=np.complex128,
    )
    poles = np.linalg.eigvals(matrix)
    residues = 1.0 / (
        1.0 + 0.5 * gamma_total * sys.W / (poles + 1j * sys.W) ** 2
    )
    order = np.argsort(poles.real)
    return [(residues[index], poles[index]) for index in order]


def get_eq_poles_residues(sys: System):
    if sys._cached_eq_poles is None:
        sys._cached_eq_poles = eq_poles_residues(sys)
    return sys._cached_eq_poles


def R_gamma(sys: System, alpha: Lead, sign: str) -> complex:
    if sign == "+":
        return -0.5j * sys.Gamma0(alpha) * sys.W
    if sign == "-":
        return 0.5j * sys.Gamma0(alpha) * sys.W
    raise ValueError("sign must be '+' or '-'")
