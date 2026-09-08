"""Protocol-specific fixed-kernel amplitude strategies.

The strategy boundary keeps pulse switching algebra out of the outer SCBA
driver.  Every instance is local to one installed kernel and therefore cannot
accidentally reuse physical poles after a kernel update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from psscba.backend.poles.minimal import Gbiased_R_mpm, Gfr_R_mpm
from psscba.backend.protocols.analytic import (
    A_down,
    A_up,
    C_down,
    C_up,
    downward_interaction_vertex,
    downward_lead_vertex,
    upward_c_residue,
    upward_interaction_vertex,
    upward_lead_vertex,
)
from psscba.backend.protocols.square import (
    A_square,
    C_square,
    build_square_kernel_cache,
)


Progress = Callable[[str, dict], None] | None


def _notify(progress: Progress, phase: str, **details: Any) -> None:
    if progress is not None:
        progress(phase, details)


@dataclass(frozen=True)
class ProtocolAmplitudes:
    a: dict[str, np.ndarray]
    c: np.ndarray
    protocol_cache: object | None = None


@dataclass(frozen=True)
class _PoleContraction:
    """Matrix form of one residue sum over an energy block.

    The analytic expressions all contain sums of the form
    ``sum_p K[p, m] exp(-1j * (xi[p] - E[m]) * t)``.  Factoring the phase
    gives ``K.T @ exp(-1j*xi*t)`` followed by an energy phase.  This keeps the
    temporary at ``(N_energy, N_time)`` and lets BLAS perform the contraction.
    """

    poles: np.ndarray
    coefficients: np.ndarray  # (N_pole, N_energy)
    energies: np.ndarray
    base: np.ndarray
    sign: int = 1
    pole_shift: complex = 0.0j
    boundary_zero: np.ndarray | None = None
    frequency: float = 0.0

    def evaluate(self, times: np.ndarray, pole_phase: np.ndarray | None = None) -> np.ndarray:
        times = np.asarray(times, dtype=float).reshape(-1)
        if pole_phase is None:
            pole_phase = np.exp(
                -1j * (self.poles + self.pole_shift)[:, None] * times[None, :]
            )
        values = self.sign * (self.coefficients.T @ pole_phase)
        values *= np.exp(1j * self.energies[:, None] * times[None, :])
        values += self.base[:, None]
        if self.boundary_zero is not None:
            zero = np.isclose(times, 0.0, atol=1e-14)
            if np.any(zero):
                values[:, zero] = self.boundary_zero[:, None]
        return values


def _up_a_plan(system, cache, energies, lead):
    energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
    poles = np.asarray(cache.xi_biased, dtype=np.complex128)
    residues = np.asarray(cache.residues_biased, dtype=np.complex128)
    delta = system.Delta(lead)
    x = poles[:, None]
    e = energies[None, :]
    vertex = upward_lead_vertex(system, cache, x, e, lead)
    return _PoleContraction(
        poles=poles,
        coefficients=residues[:, None] * vertex / (x - e),
        energies=energies,
        base=Gbiased_R_mpm(system, cache, energies + delta),
        pole_shift=-delta,
        boundary_zero=Gfr_R_mpm(system, cache, energies),
    )


def _up_c_plan(system, cache, omega, energies):
    energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
    shifted = energies + omega
    poles = np.asarray(cache.xi_biased, dtype=np.complex128)
    residues = np.asarray(cache.residues_biased, dtype=np.complex128)
    x = poles[:, None]
    ep = shifted[None, :]
    vertex = upward_interaction_vertex(system, cache, x, ep)
    return _PoleContraction(
        poles=poles,
        coefficients=residues[:, None] * vertex / (x - ep),
        energies=shifted,
        base=Gbiased_R_mpm(system, cache, shifted),
        boundary_zero=Gfr_R_mpm(system, cache, shifted),
        frequency=float(omega),
    )


def _down_a_plan(system, cache, energies, lead):
    energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
    poles = np.asarray(cache.xi, dtype=np.complex128)
    residues = np.asarray(cache.residues, dtype=np.complex128)
    delta = system.Delta(lead)
    x = poles[:, None]
    e = energies[None, :]
    vertex = downward_lead_vertex(system, cache, x, e, lead)
    return _PoleContraction(
        poles=poles,
        coefficients=residues[:, None] * vertex / (x - e - delta),
        energies=energies,
        base=Gfr_R_mpm(system, cache, energies),
        sign=-1,
        boundary_zero=Gbiased_R_mpm(system, cache, energies + delta),
    )


def _down_c_plan(system, cache, omega, energies):
    energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
    shifted = energies + omega
    poles = np.asarray(cache.xi, dtype=np.complex128)
    residues = np.asarray(cache.residues, dtype=np.complex128)
    x = poles[:, None]
    ep = shifted[None, :]
    vertex = downward_interaction_vertex(system, x, ep)
    return _PoleContraction(
        poles=poles,
        coefficients=residues[:, None] * vertex * Gbiased_R_mpm(system, cache, shifted)[None, :] / (x - ep),
        energies=shifted,
        base=Gfr_R_mpm(system, cache, shifted),
        sign=-1,
        boundary_zero=Gbiased_R_mpm(system, cache, shifted),
        frequency=float(omega),
    )


class ProtocolStrategy:
    name = "base"

    def __init__(self, config, system, pole_cache, energy: np.ndarray,
                 square_cache_store=None, progress=None) -> None:
        self.config = config
        self.system = system
        self.pole_cache = pole_cache
        self.energy = energy
        self.protocol_cache = None
        self.square_cache_store = square_cache_store

    def stationary_a(self, lead: str) -> np.ndarray:
        return Gfr_R_mpm(self.system, self.pole_cache, self.energy)

    def stationary_c(self, omega: float = 0.0) -> np.ndarray:
        return Gfr_R_mpm(
            self.system, self.pole_cache, self.energy + float(omega)
        )

    def evaluate_a(self, lead: str, observation_time: float) -> np.ndarray:
        raise NotImplementedError

    def evaluate_c(self, observation_time: float) -> np.ndarray:
        raise NotImplementedError

    def evaluate_a_block(self, lead: str, energies: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Evaluate one energy/time tile (fallback for specialized protocols)."""
        values = np.asarray(self.evaluate_a(lead, times), dtype=np.complex128)
        return values.reshape(len(energies), len(times))

    def evaluate_c_block(self, energies: np.ndarray, times: np.ndarray) -> np.ndarray:
        values = np.asarray(self.evaluate_c(times), dtype=np.complex128)
        return values.reshape(len(energies), len(times))

    def solve(self, time: np.ndarray, *, progress: Progress = None, c_omega: float = 0.0) -> ProtocolAmplitudes:
        dynamic = np.flatnonzero(time >= -1e-13)
        stationary = np.flatnonzero(time < -1e-13)
        a = {
            lead: np.empty((len(time), len(self.energy)), dtype=np.complex128)
            for lead in self.system.lead_names
        }
        c = np.empty((len(time), len(self.energy)), dtype=np.complex128)

        for lead in self.system.lead_names:
            if len(stationary):
                a[lead][stationary] = self.stationary_a(lead)[None, :]
        if len(stationary):
            # C(t, omega, E') has the stationary boundary at E' + omega.
            # The unshifted boundary is valid only for the installed source
            # factorization (omega=0).
            c[stationary] = self.stationary_c(c_omega)[None, :]

        energy_batch = max(1, int(getattr(self.config.projection, "energy_batch", 64)))
        time_batch = max(1, int(getattr(self.config.projection, "amplitude_time_batch", 16)))
        e_slices = [(start, min(start + energy_batch, len(self.energy)))
                    for start in range(0, len(self.energy), energy_batch)]
        t_slices = [(start, min(start + time_batch, len(dynamic)))
                    for start in range(0, len(dynamic), time_batch)]
        total_tiles = max(1, len(e_slices) * len(t_slices))

        # Construct coefficient plans once per energy block.  Time tiles only
        # apply the cached pole phases and BLAS contraction.
        a_plans = {
            (lead, e_start): self._build_a_plan(lead, self.energy[e_start:e_stop])
            for e_start, e_stop in e_slices
            for lead in self.system.lead_names
        }
        c_plans = {
            e_start: self._build_c_plan(self.energy[e_start:e_stop], c_omega)
            for e_start, e_stop in e_slices
        }

        _notify(progress, "amplitude_a", batch_completed=0, batch_total=total_tiles, batch_unit="energy_time_tiles")
        completed_tiles = 0
        # Energy blocks are the outer loop: all time blocks reuse the
        # pole/energy coefficient matrix constructed for that block.
        for e_start, e_stop in e_slices:
            energies = self.energy[e_start:e_stop]
            for t_start, t_stop in t_slices:
                indices = dynamic[t_start:t_stop]
                times = time[indices]
                for lead in self.system.lead_names:
                    values = self.evaluate_a_block(lead, energies, times, plan=a_plans[(lead, e_start)])
                    a[lead][np.ix_(indices, np.arange(e_start, e_stop))] = values.T
                completed_tiles += 1
                _notify(progress, "amplitude_a", batch_completed=completed_tiles,
                        batch_total=total_tiles, batch_unit="energy_time_tiles")

        _notify(progress, "amplitude_c", batch_completed=0, batch_total=total_tiles, batch_unit="energy_time_tiles")
        completed_tiles = 0
        for e_start, e_stop in e_slices:
            energies = self.energy[e_start:e_stop]
            for t_start, t_stop in t_slices:
                indices = dynamic[t_start:t_stop]
                times = time[indices]
                values = self.evaluate_c_block(energies, times, plan=c_plans[e_start])
                c[np.ix_(indices, np.arange(e_start, e_stop))] = values.T
                completed_tiles += 1
                _notify(progress, "amplitude_c", batch_completed=completed_tiles,
                        batch_total=total_tiles, batch_unit="energy_time_tiles")
        return ProtocolAmplitudes(a, c, self.protocol_cache)

    def _build_a_plan(self, lead, energies):
        return None

    def _build_c_plan(self, energies, omega=0.0):
        return None


class UpwardProtocol(ProtocolStrategy):
    name = "upward"

    def evaluate_a(self, lead: str, observation_time: float) -> np.ndarray:
        return A_up(self.system, self.energy, observation_time, lead, self.pole_cache)

    def evaluate_c(self, observation_time: float, omega: float = 0.0) -> np.ndarray:
        return C_up(self.system, omega, self.energy, observation_time, self.pole_cache)

    def _build_a_plan(self, lead, energies):
        return _up_a_plan(self.system, self.pole_cache, energies, lead)

    def _build_c_plan(self, energies, omega=0.0):
        return _up_c_plan(self.system, self.pole_cache, omega, energies)

    def evaluate_a_block(self, lead, energies, times, plan=None):
        return (plan or self._build_a_plan(lead, energies)).evaluate(times)

    def evaluate_c_block(self, energies, times, plan=None):
        return (plan or self._build_c_plan(energies)).evaluate(times)


class DownwardProtocol(ProtocolStrategy):
    name = "downward"

    def stationary_a(self, lead: str) -> np.ndarray:
        return Gbiased_R_mpm(
            self.system,
            self.pole_cache,
            self.energy + self.system.Delta(lead),
        )

    def stationary_c(self, omega: float = 0.0) -> np.ndarray:
        return Gbiased_R_mpm(
            self.system, self.pole_cache, self.energy + float(omega)
        )

    def evaluate_a(self, lead: str, observation_time: float) -> np.ndarray:
        return A_down(self.system, self.energy, observation_time, lead, self.pole_cache)

    def evaluate_c(self, observation_time: float, omega: float = 0.0) -> np.ndarray:
        return C_down(self.system, omega, self.energy, observation_time, self.pole_cache)

    def _build_a_plan(self, lead, energies):
        return _down_a_plan(self.system, self.pole_cache, energies, lead)

    def _build_c_plan(self, energies, omega=0.0):
        return _down_c_plan(self.system, self.pole_cache, omega, energies)

    def evaluate_a_block(self, lead, energies, times, plan=None):
        return (plan or self._build_a_plan(lead, energies)).evaluate(times)

    def evaluate_c_block(self, energies, times, plan=None):
        return (plan or self._build_c_plan(energies)).evaluate(times)


class SquareProtocol(ProtocolStrategy):
    name = "square"

    def __init__(self, config, system, pole_cache, energy: np.ndarray,
                 square_cache_store=None, progress=None) -> None:
        super().__init__(config, system, pole_cache, energy, square_cache_store, progress)
        self.protocol_cache = build_square_kernel_cache(
            system,
            energy,
            duration=float(config.protocol.duration),
            cache=pole_cache,
            store=square_cache_store,
            progress=progress,
        )

    def evaluate_a(self, lead: str, observation_time: float) -> np.ndarray:
        return A_square(
            self.system,
            self.energy,
            observation_time,
            lead,
            self.pole_cache,
            self.protocol_cache,
        )

    def evaluate_c(self, observation_time: float, omega: float = 0.0) -> np.ndarray:
        return C_square(
            self.system,
            omega,
            self.energy,
            observation_time,
            self.pole_cache,
            self.protocol_cache,
        )

    def _build_c_plan(self, energies, omega=0.0):
        return _up_c_plan(self.system, self.pole_cache, omega, energies)

    def evaluate_a_block(self, lead, energies, times, plan=None):
        energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
        times = np.asarray(times, dtype=float).reshape(-1)
        duration = float(self.config.protocol.duration)
        values = np.empty((len(energies), len(times)), dtype=np.complex128)
        before = times <= duration
        if np.any(before):
            values[:, before] = (plan or _up_a_plan(
                self.system, self.pole_cache, energies, lead
            )).evaluate(times[before])
        if np.any(~before):
            energy_start = int(np.searchsorted(self.energy, energies.real[0]))
            analytic_energies = np.where(
                np.abs(energies.imag) <= 1e-15, energies + 1e-14j, energies
            )
            square_values = np.asarray(
                self.protocol_cache.S_alpha[lead][:, :], dtype=np.complex128
            )[:, energy_start : energy_start + len(energies)]
            poles = np.asarray(self.pole_cache.xi_unbiased, dtype=np.complex128)
            coeff = (
                -np.exp(1j * self.system.Delta(lead) * duration)
                * np.asarray(self.pole_cache.residues_unbiased, dtype=np.complex128)[:, None]
                * square_values
                * np.exp(1j * poles[:, None] * duration)
            )
            post = _PoleContraction(
                poles=poles,
                coefficients=coeff,
                energies=analytic_energies,
                base=Gfr_R_mpm(self.system, self.pole_cache, analytic_energies),
            )
            values[:, ~before] = post.evaluate(times[~before])
        return values

    def evaluate_c_block(self, energies, times, plan=None):
        energies = np.asarray(energies, dtype=np.complex128).reshape(-1)
        times = np.asarray(times, dtype=float).reshape(-1)
        duration = float(self.config.protocol.duration)
        values = np.empty((len(energies), len(times)), dtype=np.complex128)
        before = times <= duration
        if np.any(before):
            active_plan = plan or self._build_c_plan(energies)
            values[:, before] = active_plan.evaluate(times[before])
        if np.any(~before):
            energy_start = int(np.searchsorted(self.energy, energies.real[0]))
            analytic_energies = np.where(
                np.abs(energies.imag) <= 1e-15, energies + 1e-14j, energies
            )
            omega = float(getattr(plan, "frequency", 0.0)) if plan is not None else 0.0
            shifted = analytic_energies + omega
            square_values = np.asarray(
                self.protocol_cache.S_C[omega][:, :], dtype=np.complex128
            )[:, energy_start : energy_start + len(energies)]
            poles = np.asarray(self.pole_cache.xi_unbiased, dtype=np.complex128)
            coeff = (
                -np.asarray(self.pole_cache.residues_unbiased, dtype=np.complex128)[:, None]
                * square_values
                * np.exp(1j * poles[:, None] * duration)
            )
            post = _PoleContraction(
                poles=poles,
                coefficients=coeff,
                energies=shifted,
                base=Gfr_R_mpm(self.system, self.pole_cache, shifted),
            )
            values[:, ~before] = post.evaluate(times[~before])
        return values


def strategy_for(config, system, pole_cache, energy: np.ndarray,
                 square_cache_store=None, progress=None) -> ProtocolStrategy:
    strategies = {
        "upward": UpwardProtocol,
        "zero": UpwardProtocol,
        "downward": DownwardProtocol,
        "square": SquareProtocol,
    }
    try:
        strategy = strategies[config.protocol.name]
    except KeyError as exc:
        raise ValueError(f"Unsupported protocol {config.protocol.name!r}.") from exc
    return strategy(config, system, pole_cache, energy, square_cache_store, progress)
