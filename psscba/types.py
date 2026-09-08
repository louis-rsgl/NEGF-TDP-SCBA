from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def readonly(values, dtype=None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def freeze_array(values, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StationaryKernel:
    energy: np.ndarray
    sigma_retarded: np.ndarray
    sigma_lesser: np.ndarray
    sigma_hartree: float
    sigma_dynamic_retarded: np.ndarray
    spectral_width: np.ndarray
    phonon_occupation: float
    projected_green_retarded: np.ndarray | None = None
    projected_green_lesser: np.ndarray | None = None
    method: str = "psscba"
    protocol: str = "zero"
    iteration: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.energy)
        arrays = (
            self.sigma_retarded,
            self.sigma_lesser,
            self.sigma_dynamic_retarded,
            self.spectral_width,
        )
        if any(len(value) != n for value in arrays):
            raise ValueError("Stationary-kernel arrays must share one energy grid.")
        if self.projected_green_retarded is not None and len(self.projected_green_retarded) != n:
            raise ValueError("Projected retarded Green function has the wrong size.")
        if self.projected_green_lesser is not None and len(self.projected_green_lesser) != n:
            raise ValueError("Projected lesser Green function has the wrong size.")
        object.__setattr__(self, "energy", readonly(self.energy, float))
        object.__setattr__(self, "sigma_retarded", readonly(self.sigma_retarded, complex))
        object.__setattr__(self, "sigma_lesser", readonly(self.sigma_lesser, complex))
        object.__setattr__(
            self, "sigma_dynamic_retarded", readonly(self.sigma_dynamic_retarded, complex)
        )
        object.__setattr__(self, "spectral_width", readonly(self.spectral_width, float))
        if self.projected_green_retarded is not None:
            object.__setattr__(
                self, "projected_green_retarded", readonly(self.projected_green_retarded, complex)
            )
        if self.projected_green_lesser is not None:
            object.__setattr__(
                self, "projected_green_lesser", readonly(self.projected_green_lesser, complex)
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    # Read-only compatibility names used by the reviewed MPM implementation.
    @property
    def w(self) -> np.ndarray:
        return self.energy

    @property
    def Sigma_ep_R(self) -> np.ndarray:
        return self.sigma_retarded

    @property
    def Sigma_ep_less(self) -> np.ndarray:
        return self.sigma_lesser

    @property
    def Sigma_H(self) -> float:
        return self.sigma_hartree

    @property
    def Sigma_ep_dyn_R(self) -> np.ndarray:
        return self.sigma_dynamic_retarded

    @property
    def Gamma_ep(self) -> np.ndarray:
        return self.spectral_width

    @property
    def N0(self) -> float:
        return self.phonon_occupation


@dataclass(frozen=True)
class ProjectionResult:
    lag: np.ndarray
    green_retarded: np.ndarray
    green_lesser: np.ndarray
    green_greater: np.ndarray
    energy: np.ndarray
    green_retarded_energy: np.ndarray
    green_lesser_energy: np.ndarray
    occupation: float
    r_green_retarded: float
    r_green_lesser: float
    q_green_retarded_time: np.ndarray
    q_green_lesser_time: np.ndarray
    diagnostics: Mapping[str, float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "lag",
            "green_retarded",
            "green_lesser",
            "green_greater",
            "energy",
            "green_retarded_energy",
            "green_lesser_energy",
            "q_green_retarded_time",
            "q_green_lesser_time",
        ):
            object.__setattr__(self, name, freeze_array(getattr(self, name)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class FixedKernelTrajectory:
    time: np.ndarray
    energy: np.ndarray
    projected: ProjectionResult
    observable_time: np.ndarray | None = None
    currents: Mapping[str, np.ndarray] = field(default_factory=dict)
    # Optional conservation-corrected view of the raw pole/Filon currents.
    # ``currents`` remains the physical, archive-comparable result; this view
    # is only for continuity/collision diagnostics.
    corrected_currents: Mapping[str, np.ndarray] = field(default_factory=dict)
    occupation: np.ndarray | None = None
    continuity_residual: np.ndarray | None = None
    raw_continuity_residual: np.ndarray | None = None
    collision_source: np.ndarray | None = None
    # Collision-source decomposition retained for convergence diagnostics.
    collision_source_finite: np.ndarray | None = None
    collision_source_prehistory: np.ndarray | None = None
    # Compact amplitude products used by production diagnostics.  These are
    # indexed as (time, energy) and retain only the non-negative products,
    # rather than the complex amplitudes or an N_t x N_t Green-function.
    amplitude_a_squared: Mapping[str, np.ndarray] = field(default_factory=dict)
    amplitude_c_squared: np.ndarray | None = None
    # Explicit phonon-branch C diagnostics keyed by eta=-1/+1.  The generic
    # installed-kernel source solve still uses C(t,0,E) Sigma^<(E); these
    # arrays expose the two manuscript branches without ambiguity.
    amplitude_c_squared_by_frequency: Mapping[float, np.ndarray] = field(default_factory=dict)
    # Optional factorized lesser sources retained only for dense collision
    # diagnostics.  They provide the stationary pre-pulse boundary needed by
    # the analytic semi-infinite history tail without retaining more dense
    # two-time data.
    lesser_source_energy: np.ndarray | None = None
    lesser_source_factors: tuple[np.ndarray, ...] = ()
    lesser_source_weights: tuple[np.ndarray, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    dense_green_retarded: np.ndarray | None = None
    dense_green_lesser: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", freeze_array(self.time, float))
        object.__setattr__(self, "energy", freeze_array(self.energy, float))
        observable_time = self.time if self.observable_time is None else self.observable_time
        object.__setattr__(self, "observable_time", freeze_array(observable_time, float))
        frozen_currents = {
            str(name): freeze_array(values, float)
            for name, values in self.currents.items()
        }
        object.__setattr__(self, "currents", MappingProxyType(frozen_currents))
        frozen_corrected_currents = {
            str(name): freeze_array(values, float)
            for name, values in self.corrected_currents.items()
        }
        object.__setattr__(
            self,
            "corrected_currents",
            MappingProxyType(frozen_corrected_currents),
        )
        frozen_a_squared = {
            str(name): freeze_array(values, float)
            for name, values in self.amplitude_a_squared.items()
        }
        for name, values in frozen_a_squared.items():
            if values.ndim != 2 or values.shape != (len(self.time), len(self.energy)):
                raise ValueError(
                    f"A^2 amplitude product for {name!r} must have shape "
                    f"({len(self.time)}, {len(self.energy)})."
                )
        object.__setattr__(self, "amplitude_a_squared", MappingProxyType(frozen_a_squared))
        for name in (
            "occupation",
            "continuity_residual",
            "raw_continuity_residual",
            "collision_source",
            "collision_source_finite",
            "collision_source_prehistory",
            "dense_green_retarded",
            "dense_green_lesser",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, freeze_array(value))
        if self.amplitude_c_squared is not None:
            c_squared = freeze_array(self.amplitude_c_squared, float)
            if c_squared.ndim != 2 or c_squared.shape != (len(self.time), len(self.energy)):
                raise ValueError(
                    f"C^2 amplitude product must have shape "
                    f"({len(self.time)}, {len(self.energy)})."
                )
            object.__setattr__(self, "amplitude_c_squared", c_squared)
        frozen_c_branches = {
            float(frequency): freeze_array(values, float)
            for frequency, values in self.amplitude_c_squared_by_frequency.items()
        }
        for frequency, values in frozen_c_branches.items():
            if values.ndim != 2 or values.shape != (len(self.time), len(self.energy)):
                raise ValueError(
                    f"C^2 branch for frequency {frequency:g} must have shape "
                    f"({len(self.time)}, {len(self.energy)})."
                )
        object.__setattr__(
            self,
            "amplitude_c_squared_by_frequency",
            MappingProxyType(frozen_c_branches),
        )
        if self.lesser_source_factors or self.lesser_source_weights:
            if len(self.lesser_source_factors) != len(self.lesser_source_weights):
                raise ValueError("Lesser source factors and weights must have equal length.")
            if self.lesser_source_energy is None:
                raise ValueError("Lesser source energy is required with source factors.")
            source_energy = freeze_array(self.lesser_source_energy, float)
            if any(
                np.asarray(factor).shape != (len(self.time), len(source_energy))
                or np.asarray(weight).shape != (len(source_energy),)
                for factor, weight in zip(self.lesser_source_factors, self.lesser_source_weights)
            ):
                raise ValueError("Lesser source factors have incompatible dimensions.")
            object.__setattr__(self, "lesser_source_energy", source_energy)
            object.__setattr__(
                self, "lesser_source_factors",
                tuple(freeze_array(factor, complex) for factor in self.lesser_source_factors),
            )
            object.__setattr__(
                self, "lesser_source_weights",
                tuple(freeze_array(weight, complex) for weight in self.lesser_source_weights),
            )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class OuterIteration:
    iteration: int
    residual_retarded: float
    residual_lesser: float
    projected_green_retarded_change: float | None
    projected_green_lesser_change: float | None
    projected_occupation: float
    mpm_tolerance: float
    step_duration_seconds: float | None = None
    cumulative_duration_seconds: float | None = None
    rss_bytes: int | None = None
    peak_rss_bytes: int | None = None
    sigma_hartree_installed: float | None = None
    sigma_hartree_candidate: float | None = None
    residual_retarded_numerator: float | None = None
    residual_retarded_denominator: float | None = None
    residual_lesser_numerator: float | None = None
    residual_lesser_denominator: float | None = None
    pole_fit_duration_seconds: float | None = None
    square_cache_duration_seconds: float | None = None
    amplitude_a_duration_seconds: float | None = None
    amplitude_c_duration_seconds: float | None = None
    pole_basis_reused: bool | None = None
    square_cache_hits: int | None = None


@dataclass(frozen=True)
class PSSCBAResult:
    converged: bool
    kernel: StationaryKernel
    trajectory: FixedKernelTrajectory
    history: tuple[OuterIteration, ...]
    initial_kernel: StationaryKernel | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    run_directory: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
