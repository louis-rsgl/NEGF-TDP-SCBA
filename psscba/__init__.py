"""Projected-stationary SCBA transport solver.

The public API is intentionally centered on protocol-conditioned stationary
kernels.  The historical preparation-fixed calculation is exposed as an
explicit comparison method rather than as the default physical model.
"""

from psscba.frontend.configuration import (
    CampaignConfig,
    CaseConfig,
    ModelConfig,
    NumericsConfig,
    ProjectionConfig,
    PlottingConfig,
    ProtocolConfig,
    RunConfig,
    SweepConfig,
    TrackingConfig,
    load_config,
)
from psscba.backend.solver import (
    CaseWorkspace,
    FixedKernelSolver,
    SCBAIteration,
    SingleCaseSolver,
    solve_fixed_kernel,
    solve_preparation_fixed,
    solve_psscba,
)
from psscba.types import (
    FixedKernelTrajectory,
    ProjectionResult,
    PSSCBAResult,
    StationaryKernel,
)

__all__ = [
    "CampaignConfig",
    "CaseConfig",
    "CaseWorkspace",
    "FixedKernelSolver",
    "FixedKernelTrajectory",
    "ModelConfig",
    "NumericsConfig",
    "ProjectionConfig",
    "PlottingConfig",
    "ProjectionResult",
    "ProtocolConfig",
    "PSSCBAResult",
    "RunConfig",
    "SCBAIteration",
    "SingleCaseSolver",
    "SweepConfig",
    "TrackingConfig",
    "StationaryKernel",
    "load_config",
    "solve_fixed_kernel",
    "solve_preparation_fixed",
    "solve_psscba",
]
