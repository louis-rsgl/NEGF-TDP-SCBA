"""Typed finite-window stationary projector service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from psscba.backend.projection.core import (
    autocorrelation_projection,
    direct_toeplitz_projection,
    make_projection_result,
    projector_errors,
)
from psscba.frontend.configuration import CaseConfig, RunConfig
from psscba.types import ProjectionResult


@dataclass(frozen=True)
class StationaryProjector:
    """Apply the configured uniform finite-window Toeplitz projector."""

    config: CaseConfig | RunConfig

    def project_lesser_sources(self, sources, *, progress=None) -> np.ndarray:
        return autocorrelation_projection(
            sources,
            energy_batch=self.config.projection.energy_batch,
            progress=progress,
        )

    def project_dense(self, matrix: np.ndarray, *, lesser: bool) -> np.ndarray:
        return direct_toeplitz_projection(matrix, lesser=lesser)

    def errors(self, matrix: np.ndarray, *, lesser: bool) -> tuple[float, float]:
        return projector_errors(matrix, lesser=lesser)

    def result(self, **kwargs) -> ProjectionResult:
        return make_projection_result(**kwargs)
