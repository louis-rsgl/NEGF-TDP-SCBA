"""Observable evaluation for a completed fixed-kernel trajectory."""

from __future__ import annotations

from dataclasses import dataclass

from psscba.backend.reconstruction.engine import (
    add_discarded_sector_diagnostics,
    attach_observables,
    direct_oracle_observables,
    finite_window_observables,
)
from psscba.frontend.configuration import CaseConfig, RunConfig
from psscba.types import FixedKernelTrajectory, StationaryKernel


@dataclass(frozen=True)
class ObservableEvaluator:
    config: CaseConfig | RunConfig
    kernel: StationaryKernel

    def evaluate(
        self,
        trajectory: FixedKernelTrajectory,
        *,
        progress=None,
        previous_basis=None,
    ) -> FixedKernelTrajectory:
        return attach_observables(
            self.config,
            self.kernel,
            trajectory,
            progress=progress,
            previous_basis=previous_basis,
        )

    def with_discarded_sectors(
        self,
        trajectory: FixedKernelTrajectory,
    ) -> FixedKernelTrajectory:
        return add_discarded_sector_diagnostics(self.config, trajectory, kernel=self.kernel)

    def direct_oracle(
        self,
        trajectory: FixedKernelTrajectory,
    ) -> FixedKernelTrajectory:
        return direct_oracle_observables(self.config, self.kernel, trajectory)

    def production_psi(
        self,
        trajectory: FixedKernelTrajectory,
        *,
        prefer_time_domain: bool = False,
        progress=None,
    ) -> FixedKernelTrajectory:
        return finite_window_observables(
            self.config,
            self.kernel,
            trajectory,
            prefer_time_domain=prefer_time_domain,
            progress=progress,
        )
