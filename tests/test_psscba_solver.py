from dataclasses import replace

import numpy as np

from psscba import (
    ModelConfig,
    NumericsConfig,
    ProjectionConfig,
    ProtocolConfig,
    RunConfig,
    solve_psscba,
    solve_fixed_kernel,
)
from psscba.frontend.storage import checkpoint_writer, create_run_directory, load_checkpoint
from psscba.backend.observables import (
    direct_oracle_observables,
    finite_window_observables,
)
from psscba.backend.reconstruction.engine import (
    expc_linear_uniform,
    filon_linear,
    lead_fermi_memory_kernel,
    lesser_energy_grid,
    panel_kernel_convolution,
)
from psscba.backend.stationary.initializer import make_system
from psscba.backend.stationary.initializer import initial_zero_pulse_kernel
from psscba.types import OuterIteration
from psscba.backend.model.distributions import expc
from psscba.backend.poles.minimal import build_pole_cache
from psscba.backend.reconstruction.engine import _current_energy_grid
from psscba.backend.solver import _completed_stage


def small_config(coupling_meV=0.0, dense=False):
    return RunConfig(
        model=ModelConfig(
            bandwidth=3.0,
            coupling_meV=coupling_meV,
            device_shift=0.4,
            left_shift=0.7,
            beta=2.0,
            beta_ph=10.0,
            phonon_energy=0.3,
        ),
        protocol=ProtocolConfig(name="upward"),
        projection=ProjectionConfig(
            time_min=-0.2,
            time_max=0.2,
            time_step=0.1,
            dense_validation=dense,
        ),
        numerics=NumericsConfig(
            energy_min=-6.0,
            energy_max=6.0,
            energy_step=0.2,
            eta=1e-4,
            stationary_max_iter=20,
            outer_max_iter=3,
            outer_tolerance=1e-6,
            mpm_search_tolerance=1e-6,
            mpm_final_tolerance=1e-6,
            mpm_n_iw=64,
            current_energy_points=61,
        ),
    )


def test_noninteracting_projected_fixed_point_is_exact_and_finite():
    result = solve_psscba(small_config())
    assert result.converged
    assert len(result.history) == 1
    assert result.history[0].residual_retarded == 0.0
    assert result.history[0].residual_lesser == 0.0
    assert np.max(np.abs(result.kernel.sigma_retarded)) == 0.0
    # Production runs stay streamed; dense discarded-sector/collision data is
    # reserved for explicitly requested validation configurations.
    assert result.trajectory.collision_source is None
    assert result.trajectory.diagnostics["dense_validation_performed"] is False
    assert all(np.all(np.isfinite(values)) for values in result.trajectory.currents.values())
    assert len(result.trajectory.observable_time) == len(result.trajectory.currents["L"])
    assert result.trajectory.diagnostics["current_energy_quadrature"] == (
        "stationary_endpoint_filon_linear"
    )
    diagnostics = result.trajectory.diagnostics
    assert diagnostics["noninteracting_conserving_current_correction_max"] >= 0.0
    assert np.max(np.abs(result.trajectory.continuity_residual)) < 1e-12
    assert diagnostics["discarded_sector_diagnostics"] == "skipped_production"
    assert set(result.trajectory.diagnostics["stationary_endpoint_currents"]) == {
        "L",
        "R",
    }


def test_progress_callbacks_do_not_change_numerical_results():
    config = small_config()
    events = []
    tracked = solve_psscba(config, progress=lambda phase, details: events.append((phase, details)))
    plain = solve_psscba(config)
    assert events
    assert np.array_equal(tracked.kernel.sigma_retarded, plain.kernel.sigma_retarded)
    assert np.array_equal(tracked.trajectory.currents["L"], plain.trajectory.currents["L"])
    assert tracked.history[0].residual_retarded == plain.history[0].residual_retarded


def test_filon_linear_integrates_linear_oscillatory_panels():
    grid = np.linspace(-2.0, 3.0, 41)
    values = (1.2 - 0.7j) + (0.3 + 0.2j) * grid
    frequency = 37.0
    result = filon_linear(values, grid, frequency)
    dense = np.linspace(grid[0], grid[-1], 1_000_001)
    reference = np.trapezoid(
        ((1.2 - 0.7j) + (0.3 + 0.2j) * dense)
        * np.exp(1j * frequency * dense),
        dense,
    )
    assert abs(result - reference) < 1e-9


def test_closed_form_expc_panels_match_high_order_panel_oracle():
    grid = np.linspace(-5.0, 5.0, 201)
    values = np.exp(-grid**2) * (1.0 + 0.2j * grid)
    time = 96.0
    shift = 0.37
    result = expc_linear_uniform(values, grid, shift, time)
    reference = panel_kernel_convolution(
        values,
        grid,
        lambda difference: expc(difference + shift, time),
        quadrature_order=24,
    )
    assert np.max(np.abs(result - reference)) < 1e-12


def test_lorentzian_fermi_memory_kernel_has_exact_equal_time_value():
    config = small_config()
    system = make_system(config)
    lag = np.arange(5, dtype=float) * config.projection.time_step
    for lead in system.lead_names:
        memory = lead_fermi_memory_kernel(system, lead, lag)
        expected = system.Gamma0(lead) * system.W / 4.0
        assert abs(memory[0] - expected) < 1e-12
        assert np.all(np.isfinite(memory))


def test_lesser_quadrature_locally_resolves_narrow_physical_poles():
    base = small_config()
    config = replace(
        base,
        model=replace(
            base.model,
            bandwidth=1.0,
            device_shift=5.0,
            left_shift=10.0,
            right_shift=0.0,
            beta=10.0,
        ),
        protocol=ProtocolConfig(name="downward"),
        numerics=replace(
            base.numerics,
            energy_min=-20.0,
            energy_max=20.0,
            energy_step=0.1,
        ),
    )
    kernel = initial_zero_pulse_kernel(config)
    system = make_system(config)
    cache = build_pole_cache(system, kernel)
    # Production uses the declared reconstruction grid verbatim.  The
    # adaptive pole-centred variant is available only as an explicit
    # diagnostic/reference calculation.
    assert np.array_equal(lesser_energy_grid(config, cache), config.numerics.energy_grid())
    source_grid = lesser_energy_grid(config, cache, allow_refinement=True)
    assert len(source_grid) > len(config.numerics.energy_grid())
    assert np.min(np.diff(source_grid)) < 0.01


def test_downward_noninteracting_time_current_is_conserving():
    base = small_config(dense=True)
    config = replace(
        base,
        protocol=ProtocolConfig(name="downward"),
    )
    result = solve_psscba(config)
    diagnostics = result.trajectory.diagnostics
    assert diagnostics["current_method"] == "pole_fft_generic_psi"
    assert diagnostics["current_representation_effective"] == "pole_filon"
    assert "noninteracting_conserving_current_correction_max" in diagnostics
    assert np.max(np.abs(result.trajectory.continuity_residual)) < 1e-12


def test_noninteracting_correction_preserves_symmetric_transport_mode():
    config = small_config(dense=True)
    kernel = initial_zero_pulse_kernel(config)
    trajectory = solve_fixed_kernel(
        config, kernel, dense=True, observables=False
    )
    raw = finite_window_observables(
        config,
        kernel,
        trajectory,
        prefer_time_domain=False,
        conserve_noninteracting=False,
    )
    corrected = finite_window_observables(
        config,
        kernel,
        trajectory,
        prefer_time_domain=False,
        conserve_noninteracting=True,
    )
    raw_transport = raw.currents["L"] - raw.currents["R"]
    corrected_transport = corrected.currents["L"] - corrected.currents["R"]
    assert np.max(np.abs(raw_transport - corrected_transport)) < 1e-14
    # The conservation correction is a diagnostic view; canonical currents
    # must remain the raw pole/Filon values used for archive comparisons.
    assert np.array_equal(raw.currents["L"], corrected.currents["L"])
    assert np.array_equal(raw.currents["R"], corrected.currents["R"])
    assert set(corrected.corrected_currents) == {"L", "R"}
    assert np.max(
        np.abs(corrected.corrected_currents["L"] - corrected.currents["L"])
    ) > 0.0
    assert np.max(np.abs(corrected.continuity_residual)) < 1e-12


def test_interacting_kernel_does_not_receive_noninteracting_correction():
    config = small_config(coupling_meV=1.0, dense=True)
    kernel = initial_zero_pulse_kernel(config)
    trajectory = solve_fixed_kernel(config, kernel, dense=True)
    assert "noninteracting_conserving_current_correction_max" not in (
        trajectory.diagnostics
    )


def test_downward_fast_psi_kernel_matches_literal_oracle():
    base = small_config(dense=True)
    config = replace(base, protocol=ProtocolConfig(name="downward"))
    kernel = initial_zero_pulse_kernel(config)
    trajectory = solve_fixed_kernel(
        config, kernel, dense=True, observables=False
    )
    fast = finite_window_observables(
        config, kernel, trajectory, prefer_time_domain=False
    )
    direct = direct_oracle_observables(
        config, kernel, trajectory
    )
    for lead in ("L", "R"):
        assert np.max(
            np.abs(fast.currents[lead] - direct.currents[lead])
        ) < 1e-11


def test_pole_current_does_not_require_dense_two_time_storage():
    config = small_config()
    kernel = initial_zero_pulse_kernel(config)
    trajectory = solve_fixed_kernel(config, kernel, dense=False)
    assert trajectory.dense_green_retarded is None
    assert trajectory.dense_green_lesser is None
    assert trajectory.diagnostics["current_representation_effective"] == "pole_filon"


def test_downward_w100_g0_pole_current_regression_against_oracle():
    base = small_config()
    config = replace(
        base,
        model=replace(base.model, bandwidth=100.0),
        protocol=ProtocolConfig(name="downward"),
        projection=replace(base.projection, time_min=-8.0, time_max=0.2, time_step=0.1),
        numerics=replace(
            base.numerics,
            energy_min=-20.0,
            energy_max=20.0,
            energy_step=0.05,
            current_energy_points=801,
        ),
    )
    kernel = initial_zero_pulse_kernel(config)
    trajectory = solve_fixed_kernel(config, kernel, dense=False, observables=False)
    fast = finite_window_observables(config, kernel, trajectory)
    direct = direct_oracle_observables(config, kernel, trajectory)
    for lead in ("L", "R"):
        assert np.max(np.abs(fast.currents[lead] - direct.currents[lead])) < 1e-5


def test_current_quadrature_can_be_independent_of_reconstruction_grid():
    config = small_config()
    config = replace(
        config,
        numerics=replace(
            config.numerics,
            current_energy_min=-20.0,
            current_energy_max=20.0,
            current_energy_points=1001,
        ),
    )
    grid = _current_energy_grid(config.case, np.linspace(-100.0, 100.0, 9))
    assert len(grid) == 1001
    assert np.isclose(grid[0], -20.0)
    assert np.isclose(grid[-1], 20.0)


def test_dense_solver_records_projection_gates():
    result = solve_psscba(small_config(dense=True))
    diagnostics = result.trajectory.diagnostics
    assert diagnostics["lesser_dense_streaming_error"] < 1e-12
    assert diagnostics["projector_retarded_idempotency_error"] < 1e-12
    assert diagnostics["projector_lesser_orthogonality_error"] < 1e-12
    assert diagnostics["retarded_causality_error"] < 1e-12
    assert diagnostics["lesser_hermiticity_error"] < 1e-10


def test_generic_installed_lesser_kernel_needs_no_prior_projected_green():
    config = small_config(dense=True)
    kernel = initial_zero_pulse_kernel(config)
    arbitrary = replace(
        kernel,
        sigma_lesser=1j * 1e-3 * np.exp(-kernel.energy**2),
        projected_green_lesser=None,
    )
    trajectory = solve_fixed_kernel(
        config, arbitrary, dense=True, observables=False
    )
    assert trajectory.diagnostics["lesser_dense_streaming_error"] < 1e-12
    assert np.max(np.abs(trajectory.projected.green_lesser)) > 0.0


def test_generic_lesser_fast_current_matches_literal_direct_oracle():
    config = small_config(dense=True)
    kernel = initial_zero_pulse_kernel(config)
    arbitrary = replace(
        kernel,
        sigma_lesser=1j * 1e-3 * np.exp(-kernel.energy**2),
        projected_green_lesser=None,
    )
    fast = solve_fixed_kernel(
        replace(
            config,
            numerics=replace(config.numerics, current_method="finite_window"),
        ),
        arbitrary,
        dense=True,
    )
    direct = solve_fixed_kernel(
        replace(
            config,
            numerics=replace(config.numerics, current_method="direct_oracle"),
        ),
        arbitrary,
        dense=True,
    )
    for lead in ("L", "R"):
        assert np.max(np.abs(fast.currents[lead] - direct.currents[lead])) < 1e-12
    assert np.max(np.abs(fast.occupation - direct.occupation)) < 1e-12


def test_auxiliary_retarded_reconstruction_covers_every_protocol():
    for name, duration in (
        ("upward", None),
        ("downward", None),
        ("square", 0.1),
        ("zero", None),
    ):
        base = small_config(dense=True)
        config = replace(base, protocol=ProtocolConfig(name=name, duration=duration))
        result = solve_psscba(config)
        assert result.trajectory.diagnostics[
            "left_right_retarded_reconstruction_error"
        ] < 1e-12


def test_checkpoint_roundtrip_preserves_kernel_history_and_config(tmp_path):
    config = replace(small_config(), output_root=str(tmp_path))
    kernel = initial_zero_pulse_kernel(config)
    history = (
        OuterIteration(0, 0.2, 0.3, float("inf"), float("inf"), 0.5, 1e-6),
    )
    run_dir = create_run_directory(config, label="checkpoint")
    checkpoint_writer(run_dir)(kernel, history)
    loaded_config, loaded_kernel, loaded_history = load_checkpoint(run_dir)
    assert loaded_config == config
    assert np.array_equal(loaded_kernel.sigma_retarded, kernel.sigma_retarded)
    assert loaded_history == history


def test_resume_recognizes_only_the_completed_matching_mpm_stage():
    loose = OuterIteration(7, 2e-6, 9e-6, None, None, 0.5, 1e-6)
    strict = OuterIteration(8, 3e-6, 8e-6, None, None, 0.5, 1e-8)
    assert _completed_stage(
        (loose,), mpm_tolerance=1e-6, outer_tolerance=1e-5
    )
    assert not _completed_stage(
        (loose,), mpm_tolerance=1e-8, outer_tolerance=1e-5
    )
    assert _completed_stage(
        (loose, strict), mpm_tolerance=1e-8, outer_tolerance=1e-5
    )


def test_resume_skips_redundant_loose_step_before_strict_stage():
    base = small_config(coupling_meV=0.1, dense=False)
    config = replace(
        base,
        numerics=replace(
            base.numerics,
            outer_tolerance=1e3,
            outer_max_iter=4,
            mpm_search_tolerance=1e-6,
            mpm_final_tolerance=1e-8,
            mpm_aaa_initial_terms=8,
            mpm_aaa_max_terms=24,
            current_energy_points=31,
        ),
    )
    kernel = initial_zero_pulse_kernel(config)
    history = (
        OuterIteration(0, 1e-4, 2e-4, None, None, 0.5, 1e-6),
    )
    events = []
    result = solve_psscba(
        config,
        initial_kernel=kernel,
        initial_history=history,
        progress=lambda phase, details: events.append((phase, details)),
    )
    assert result.converged
    assert [item.mpm_tolerance for item in result.history] == [1e-6, 1e-8]
    assert any(
        "skipping redundant outer iteration" in str(details.get("message", ""))
        for _, details in events
    )
