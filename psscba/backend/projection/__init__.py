"""Finite-window stationary projection services."""

from psscba.backend.projection.core import (
    autocorrelation_projection,
    dense_lesser_from_sources,
    direct_toeplitz_projection,
    make_projection_result,
    projector_errors,
    reconstruct_retarded_rows,
    relative_time_to_energy,
    toeplitz_from_lags,
    trapezoid_weights,
)
from psscba.backend.projection.projector import StationaryProjector

__all__ = [
    "StationaryProjector",
    "autocorrelation_projection",
    "dense_lesser_from_sources",
    "direct_toeplitz_projection",
    "make_projection_result",
    "projector_errors",
    "reconstruct_retarded_rows",
    "relative_time_to_energy",
    "toeplitz_from_lags",
    "trapezoid_weights",
]
