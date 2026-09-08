"""Numerical engine for one NEGF TDP PS-SCBA parameter case.

The backend owns scientific computation only. Campaign scheduling, durable
run management, and user-interface concerns live in :mod:`psscba.frontend`.
"""

from psscba.backend.solver import (
    CaseWorkspace,
    FixedKernelSolver,
    SCBAIteration,
    SingleCaseSolver,
)
from psscba.backend.observables import ObservableEvaluator
from psscba.backend.plotting import RunPlotter
from psscba.backend.projection import StationaryProjector
from psscba.backend.protocols import DownwardProtocol, SquareProtocol, UpwardProtocol
from psscba.backend.protocols import SquareKernelCacheStore
from psscba.backend.poles import PoleBasis
from psscba.backend.self_energy import SCBAFunctional
from psscba.backend.stationary import StationaryInitializer

__all__ = [
    "CaseWorkspace",
    "FixedKernelSolver",
    "DownwardProtocol",
    "ObservableEvaluator",
    "RunPlotter",
    "SCBAFunctional",
    "SCBAIteration",
    "SingleCaseSolver",
    "SquareProtocol",
    "StationaryInitializer",
    "StationaryProjector",
    "UpwardProtocol",
    "PoleBasis",
    "SquareKernelCacheStore",
]
