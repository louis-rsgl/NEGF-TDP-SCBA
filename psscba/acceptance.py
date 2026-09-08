from __future__ import annotations

from psscba.frontend.configuration import RunConfig
from psscba.types import PSSCBAResult


class NumericalAcceptanceError(RuntimeError):
    pass


def enforce_acceptance(config: RunConfig, result: PSSCBAResult) -> None:
    if not result.converged:
        raise NumericalAcceptanceError("Outer PS-SCBA fixed point did not converge.")
    diagnostics = result.trajectory.diagnostics
    failures: list[str] = []
    final_r = result.diagnostics.get("final_retarded_residual")
    final_l = result.diagnostics.get("final_lesser_residual")
    if final_r is None or final_l is None or max(final_r, final_l) >= config.numerics.outer_tolerance:
        failures.append("installed outer residual")
    thresholds = {}
    if config.projection.dense_validation:
        thresholds.update({
            "lesser_dense_streaming_error": 1e-10,
            "projector_lesser_idempotency_error": 1e-12,
            "projector_lesser_orthogonality_error": 1e-12,
            "projector_retarded_idempotency_error": 1e-12,
            "projector_retarded_orthogonality_error": 1e-12,
            "retarded_causality_error": 1e-12,
            "lesser_hermiticity_error": 1e-10,
            "left_right_retarded_reconstruction_error": 1e-5,
            "two_time_occupation_error": 1e-5,
        })
    for name, threshold in thresholds.items():
        if name not in diagnostics or diagnostics[name] >= threshold:
            failures.append(f"{name}<{threshold:g}")
    mpm_thresholds = {
        "mpm_sigma_scaled_error": 1.0,
        "mpm_Gfr_scaled_error": 1.0,
        "mpm_Gbiased_scaled_error": 1.0,
    }
    if not diagnostics.get("pole_fit_converged", False):
        failures.append("pole_fit_converged")
    if not diagnostics.get("pole_causal", False):
        failures.append("pole_causal")
    for name, threshold in mpm_thresholds.items():
        if name not in diagnostics or diagnostics[name] > threshold:
            failures.append(f"{name}<={threshold:g}")
    if config.projection.dense_validation:
        mismatch = diagnostics.get("collision_identity_scaled_mismatch")
        if mismatch is None or mismatch >= 1e-4:
            failures.append("collision_identity_scaled_mismatch<1e-4")
    if failures:
        raise NumericalAcceptanceError(
            "Hard numerical acceptance gates failed: " + ", ".join(failures)
        )
