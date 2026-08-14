import numpy as np

import runner
import runner_downward_full_scba
import runner_square_full_scba
import runner_upward_full_scba


def test_protocol_specific_run_directory_and_default_system(tmp_path):
    run_dir = runner.make_run_dir(tmp_path, protocol="square")
    assert run_dir.name.startswith("run_square_")
    system = runner.make_sys(20.0, 0.0)
    assert system.pulse_protocol == runner.PULSE_PROTOCOL == "square"
    assert system.pulse_duration == runner.SQUARE_DURATION
    assert system.pulse_duration >= 0.0
    assert not system.reference_is_biased
    assert runner.make_sys(1.0, 20.0).n_w_scba == 40001


def test_fake_runner_serial_parallel_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "USE_FAKE_SOLVER", True)
    monkeypatch.setattr(runner, "VERBOSE", False)
    monkeypatch.setattr(runner, "SQUARE_DURATION", 0.05)
    w_grid = np.array([1.0, 2.0])
    g_grid = np.array([0.0])

    serial_logs = tmp_path / "serial-logs"
    serial_logs.mkdir()
    t_serial, j_serial = runner.precompute_currents_serial(
        w_grid, g_grid, t_max=0.1, n_t=5, log_dir=serial_logs
    )

    parallel_logs = tmp_path / "parallel-logs"
    parallel_logs.mkdir()
    t_parallel, j_parallel = runner.precompute_currents_parallel(
        w_grid,
        g_grid,
        t_max=0.1,
        n_t=5,
        max_workers=2,
        log_dir=parallel_logs,
    )
    assert np.allclose(t_serial, t_parallel)
    assert np.allclose(j_serial, j_parallel)
    assert np.isrealobj(j_serial)
    assert len(list(serial_logs.glob("*.quality.json"))) == 2
    assert len(list(parallel_logs.glob("*.quality.json"))) == 2

    run_dir = tmp_path / "run"
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True)
    runner.save_currents_npy(run_dir, t_serial, j_serial, w_grid, g_grid, "L")
    runner.save_all_current_plots_svg(
        t_serial, j_serial, w_grid, g_grid, "L", figure_dir
    )
    assert (run_dir / "data" / "t_ps.npy").exists()
    assert len(list(figure_dir.glob("*.svg"))) == 2


def test_serial_grid_records_failure_and_continues(tmp_path, monkeypatch):
    def calculate(W, g_q, alpha, t_max, n_t):
        if g_q > 0.0:
            raise RuntimeError("intentional parameter failure")
        return np.linspace(0.0, 1.0, n_t), np.ones(n_t), {"ok": True}

    monkeypatch.setattr(runner, "_compute_current_with_diagnostics", calculate)
    monkeypatch.setattr(runner, "CONTINUE_ON_FAILURE", True)
    monkeypatch.setattr(runner, "SQUARE_DURATION", 0.01)
    logs = tmp_path / "logs"
    logs.mkdir()
    time, currents = runner.precompute_currents_serial(
        np.array([20.0]),
        np.array([0.0, 2.5]),
        n_t=3,
        log_dir=logs,
    )
    assert np.allclose(time, [0.0, 0.5, 1.0])
    assert np.all(np.isfinite(currents[0, 0]))
    assert np.all(np.isnan(currents[0, 1]))
    assert len(list(logs.glob("*.quality.json"))) == 1
    assert len(list(logs.glob("*.failed.json"))) == 1


def test_all_failed_grid_returns_time_axis(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("all failed")

    monkeypatch.setattr(runner, "_compute_current_with_diagnostics", fail)
    monkeypatch.setattr(runner, "CONTINUE_ON_FAILURE", True)
    monkeypatch.setattr(runner, "SQUARE_DURATION", 0.1)
    logs = tmp_path / "logs"
    logs.mkdir()
    time, currents = runner.precompute_currents_serial(
        np.array([20.0]), np.array([2.5]), t_max=0.2, n_t=3, log_dir=logs
    )
    assert np.allclose(time, runner.time_to_ps([0.0, 0.1, 0.2], runner.GAMMA))
    assert np.all(np.isnan(currents))


def test_full_scba_protocol_runner_configurations():
    original = {
        "pulse": runner.PULSE_PROTOCOL,
        "mode": runner.STATIONARY_MODE,
        "duration": runner.SQUARE_DURATION,
        "t_max": runner.T_MAX,
        "n_t": runner.N_T,
        "parallel": runner.PARALLEL,
        "workers": runner.MAX_WORKERS,
    }
    try:
        runner_downward_full_scba.configure()
        assert runner.PULSE_PROTOCOL == "downward"
        assert runner.STATIONARY_MODE == "self_consistent"
        assert runner.T_MAX == 2.0
        assert runner.N_T == 201
        assert runner.PARALLEL is False
        assert runner.MAX_WORKERS == 1

        runner_upward_full_scba.configure()
        assert runner.PULSE_PROTOCOL == "upward"
        assert runner.STATIONARY_MODE == "self_consistent"
        assert runner.T_MAX == 3.0
        assert runner.N_T == 201
        assert runner.PARALLEL is False

        runner_square_full_scba.configure()
        assert runner.PULSE_PROTOCOL == "square"
        assert runner.STATIONARY_MODE == "self_consistent"
        assert runner.SQUARE_DURATION == 3.0
        assert runner.T_MAX == 6.0
        assert runner.N_T == 201
        assert runner.PARALLEL is False
    finally:
        runner.PULSE_PROTOCOL = original["pulse"]
        runner.STATIONARY_MODE = original["mode"]
        runner.SQUARE_DURATION = original["duration"]
        runner.T_MAX = original["t_max"]
        runner.N_T = original["n_t"]
        runner.PARALLEL = original["parallel"]
        runner.MAX_WORKERS = original["workers"]
