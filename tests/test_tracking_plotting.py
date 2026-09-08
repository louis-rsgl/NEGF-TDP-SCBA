from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from psscba.frontend.configuration import RunConfig, load_config
from psscba.frontend.storage import load_checkpoint
from psscba.backend.plotting.qa import (
    _surface_subsample,
    plot_outer_convergence,
    publication_current,
    write_amplitude_surfaces,
    write_qa_plots,
)
from psscba.frontend.tracking import RunTracker, read_tracking_history, stale_heartbeat
from psscba.types import FixedKernelTrajectory, OuterIteration, ProjectionResult, PSSCBAResult, StationaryKernel


def _record(iteration=0):
    return OuterIteration(
        iteration, 1e-2, 2e-2, 3e-2, 4e-2, 0.51, 1e-6,
        step_duration_seconds=12.0, cumulative_duration_seconds=12.0,
        rss_bytes=1024, peak_rss_bytes=2048,
        sigma_hartree_installed=-0.1, sigma_hartree_candidate=-0.2,
        residual_retarded_numerator=1.0, residual_retarded_denominator=100.0,
        residual_lesser_numerator=2.0, residual_lesser_denominator=100.0,
    )


def test_retired_numbered_format_is_rejected_normally(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("schema_version: '1.0'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retired numbered configuration format"):
        load_config(path)


def test_legacy_run_is_readable_for_plotting_but_rejected_for_resume(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    raw = {"schema_version": "2.0", "output_root": str(tmp_path)}
    (run / "resolved.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    readable = load_config(run / "resolved.yaml", allow_read_only_legacy=True)
    assert readable.config_format == "negf-tdp-ps-scba"
    with pytest.raises(ValueError, match="read/plot only"):
        load_config(run / "resolved.yaml")
    with pytest.raises(ValueError, match="read/plot only"):
        load_checkpoint(run)


def test_tracker_writes_progress_jsonl_csv_and_live_svg(tmp_path):
    config = replace(load_config("examples/example.yaml"), output_root=str(tmp_path))
    run = tmp_path / "run"
    run.mkdir()
    tracker = RunTracker(run, config)
    tracker.start()
    tracker.update("amplitude_a", {"outer_step": 2, "batch_completed": 3, "batch_total": 10, "batch_unit": "times"})
    tracker.completed_step(_record(), (_record(),))
    tracker.close("interrupted")
    payload = json.loads((run / "progress.json").read_text())
    assert payload["state"] == "interrupted"
    assert payload["residuals_are_from_last_completed_step"]
    assert len(read_tracking_history(run / "tracking_history.jsonl")) >= 3
    assert (run / "convergence.csv").exists()
    assert (run / "events.jsonl").exists()
    assert (run / "plots/live/outer_convergence.svg").exists()
    log = (run / "solver.log").read_text(encoding="utf-8")
    assert "loss=max(R_sigma^R,R_sigma^<)" in log
    assert "mixing alpha=" in log
    assert "max_steps=" in log
    assert "Retarded lead amplitude A" in log
    assert "SCF COMPLETE" in log
    assert payload["last_completed_loss"] == pytest.approx(2e-2)
    assert payload["loss_tolerance"] == config.numerics.outer_tolerance
    assert payload["mixing_rate"] == config.numerics.outer_mixing
    assert payload["outer_max_iterations"] == config.numerics.outer_max_iter


def test_tracker_preserves_peak_rss_across_resume(tmp_path):
    config = replace(load_config("examples/example.yaml"), output_root=str(tmp_path))
    run = tmp_path / "run"
    run.mkdir()
    (run / "progress.json").write_text(
        json.dumps({"elapsed_seconds": 12.0, "peak_rss_bytes": 10_000}),
        encoding="utf-8",
    )
    tracker = RunTracker(run, config)
    tracker.start()
    payload = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    tracker.close("interrupted")
    assert payload["peak_rss_bytes"] >= 10_000


def test_stale_heartbeat_and_torn_jsonl(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({
        "state": "running", "timestamp": "2020-01-01T00:00:00+00:00", "heartbeat_seconds": 60,
    }), encoding="utf-8")
    assert stale_heartbeat(progress, now=2_000_000_000)
    history = tmp_path / "tracking.jsonl"
    history.write_text('{"ok": 1}\n{"torn":', encoding="utf-8")
    assert read_tracking_history(history) == [{"ok": 1}]


def test_convergence_svg_uses_mathtext_without_external_latex(tmp_path):
    config = load_config("examples/example.yaml")
    path = tmp_path / "convergence.svg"
    plot_outer_convergence(config, (_record(),), path)
    text = path.read_text(encoding="utf-8")
    assert "outer SCBA step" in text
    assert config.plotting.use_external_latex is False


def test_svg_generation_is_deterministic(tmp_path):
    config = load_config("examples/example.yaml")
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    plot_outer_convergence(config, (_record(),), first)
    plot_outer_convergence(config, (_record(),), second)
    assert first.read_bytes() == second.read_bytes()


def test_physical_current_plot_converts_time_and_current_units(tmp_path):
    config = load_config("examples/example.yaml")
    grid = np.asarray([-1.0, 0.0, 1.0])
    zeros = np.zeros(3, dtype=complex)
    projection = ProjectionResult(
        lag=grid, green_retarded=zeros, green_lesser=zeros, green_greater=zeros,
        energy=grid, green_retarded_energy=zeros, green_lesser_energy=zeros,
        occupation=0.0, r_green_retarded=0.0, r_green_lesser=0.0,
        q_green_retarded_time=np.zeros(3), q_green_lesser_time=np.zeros(3),
    )
    kernel = StationaryKernel(
        energy=grid, sigma_retarded=zeros, sigma_lesser=zeros, sigma_hartree=0.0,
        sigma_dynamic_retarded=zeros, spectral_width=np.zeros(3), phonon_occupation=0.0,
    )
    trajectory = FixedKernelTrajectory(
        time=grid, energy=grid, projected=projection, observable_time=np.asarray([0.0, 1.0, 2.0]),
        currents={"L": np.asarray([0.0, 1.0, 2.0])},
    )
    result = PSSCBAResult(True, kernel, trajectory, ())
    path = tmp_path / "current_L_physical.svg"
    publication_current(config, result, "L", path, physical=True)
    text = path.read_text(encoding="utf-8")
    assert r"$t\,[\mathrm{ps}]$" in text
    assert r"$J_L(t)\,[\mu\mathrm{A}]$" in text


def test_natural_current_plot_is_explicitly_named_and_labeled(tmp_path):
    config = load_config("examples/example.yaml")
    grid = np.asarray([-1.0, 0.0, 1.0])
    zeros = np.zeros(3, dtype=complex)
    projection = ProjectionResult(
        lag=grid, green_retarded=zeros, green_lesser=zeros, green_greater=zeros,
        energy=grid, green_retarded_energy=zeros, green_lesser_energy=zeros,
        occupation=0.0, r_green_retarded=0.0, r_green_lesser=0.0,
        q_green_retarded_time=np.zeros(3), q_green_lesser_time=np.zeros(3),
    )
    kernel = StationaryKernel(
        energy=grid, sigma_retarded=zeros, sigma_lesser=zeros, sigma_hartree=0.0,
        sigma_dynamic_retarded=zeros, spectral_width=np.zeros(3), phonon_occupation=0.0,
    )
    trajectory = FixedKernelTrajectory(
        time=grid, energy=grid, projected=projection,
        observable_time=np.asarray([0.0, 1.0, 2.0]), currents={"L": np.asarray([0.0, 1.0, 2.0])},
    )
    result = PSSCBAResult(True, kernel, trajectory, ())
    path = tmp_path / "current_L_natural.svg"
    publication_current(config, result, "L", path, physical=False)
    text = path.read_text(encoding="utf-8")
    assert r"$t\,[\hbar/\Gamma]$" in text
    assert r"$J_L(t)\,[e\Gamma/\hbar]$" in text


def test_amplitude_surface_products_are_rendered_in_natural_units(tmp_path):
    config = load_config("examples/example.yaml")
    time = np.linspace(-0.5, 1.0, 5)
    energy = np.linspace(-2.0, 2.0, 7)
    zeros_t = np.zeros(len(time), dtype=complex)
    zeros_e = np.zeros(len(energy), dtype=complex)
    projection = ProjectionResult(
        lag=time, green_retarded=zeros_t, green_lesser=zeros_t,
        green_greater=zeros_t, energy=energy,
        green_retarded_energy=zeros_e, green_lesser_energy=zeros_e,
        occupation=0.0, r_green_retarded=0.0, r_green_lesser=0.0,
        q_green_retarded_time=np.zeros(len(time)),
        q_green_lesser_time=np.zeros(len(time)),
    )
    kernel = StationaryKernel(
        energy=energy, sigma_retarded=zeros_e, sigma_lesser=zeros_e,
        sigma_hartree=0.0, sigma_dynamic_retarded=zeros_e,
        spectral_width=np.zeros(len(energy)), phonon_occupation=0.0,
    )
    trajectory = FixedKernelTrajectory(
        time=time, energy=energy, projected=projection,
        amplitude_a_squared={"L": np.ones((len(time), len(energy)))},
        amplitude_c_squared=2.0 * np.ones((len(time), len(energy))),
    )
    result = PSSCBAResult(True, kernel, trajectory, ())
    write_amplitude_surfaces(config, result, tmp_path)
    output = tmp_path / "plots" / "publication"
    assert (output / "amplitude_A2_L.svg").exists()
    assert (output / "amplitude_C2.svg").exists()
    assert r"t\,[\hbar/\Gamma]" in (output / "amplitude_A2_L.svg").read_text()
    assert r"E'/\Gamma" in (output / "amplitude_C2.svg").read_text()


def test_amplitude_surface_render_window_clips_only_the_energy_axis():
    time = np.linspace(-1.0, 1.0, 3)
    energy = np.linspace(-30.0, 30.0, 13)
    values = np.arange(time.size * energy.size, dtype=float).reshape(time.size, energy.size)
    clipped_time, clipped_energy, clipped_values = _surface_subsample(
        time, energy, values, energy_window=(-20.0, 20.0)
    )
    np.testing.assert_array_equal(clipped_time, time)
    np.testing.assert_array_equal(clipped_energy, energy[2:11])
    np.testing.assert_array_equal(clipped_values, values[:, 2:11])


def test_plot_only_mpm_failure_does_not_reject_completed_result(tmp_path, monkeypatch):
    """A diagnostic MPM fit must not turn a completed solve into a failed run."""
    config = load_config("examples/example.yaml")
    grid = np.asarray([-1.0, 0.0, 1.0])
    zeros = np.zeros(3, dtype=complex)
    projection = ProjectionResult(
        lag=grid, green_retarded=zeros, green_lesser=zeros, green_greater=zeros,
        energy=grid, green_retarded_energy=zeros, green_lesser_energy=zeros,
        occupation=0.0, r_green_retarded=0.0, r_green_lesser=0.0,
        q_green_retarded_time=np.zeros(3), q_green_lesser_time=np.zeros(3),
    )
    kernel = StationaryKernel(
        energy=grid, sigma_retarded=zeros, sigma_lesser=zeros, sigma_hartree=0.0,
        sigma_dynamic_retarded=zeros, spectral_width=np.zeros(3), phonon_occupation=0.0,
    )
    trajectory = FixedKernelTrajectory(
        time=grid, energy=grid, projected=projection, observable_time=grid,
        currents={"L": np.zeros(3), "R": np.zeros(3)},
        occupation=np.zeros(3), continuity_residual=np.zeros(3),
        collision_source=np.zeros(3),
    )
    result = PSSCBAResult(True, kernel, trajectory, ())

    def fail(*args, **kwargs):
        raise RuntimeError("intentional plot-only MPM failure")

    monkeypatch.setattr("psscba.backend.plotting.qa.build_pole_cache", fail)
    write_qa_plots(config, result, tmp_path)
    status = json.loads((tmp_path / "qa/03_mpm_reconstruction.json").read_text())
    assert status["status"] == "unavailable"
    assert "plot-only MPM failure" in status["reason"]
    assert (tmp_path / "qa/03_mpm_reconstruction.svg").exists()
