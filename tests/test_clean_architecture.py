import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from psscba import CaseWorkspace, FixedKernelSolver, SCBAIteration, load_config
from psscba.backend.plotting import RunPlotter
from psscba.backend.protocols import DownwardProtocol, SquareProtocol, UpwardProtocol
from psscba.backend.stationary import initial_zero_pulse_kernel
from psscba.frontend.campaign import resume_campaign
from psscba.frontend.dashboard import BANNER, CampaignDashboard
from psscba.frontend.tracking import RunTracker


def test_production_package_has_no_former_backend_dependency():
    repository = Path(__file__).resolve().parents[1]
    former = repository / "backend"
    assert not list(former.glob("*.py"))
    for source in (repository / "psscba").rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "from backend" not in text
        assert "import backend" not in text


def test_source_checkout_entrypoint_and_public_example():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "runner.py", "--help"], cwd=repository,
        check=True, capture_output=True, text=True,
    )
    assert "NEGF TDP PS-SCBA" in result.stdout
    assert (repository / "examples" / "example.yaml").is_file()
    assert not (repository / "pyproject.toml").exists()
    assert not (repository / "requirements.txt").exists()
    assert not (repository / "run.sh").exists()


def test_case_workspace_is_immutable_and_fixed_kernel_cache_is_local():
    config = load_config("examples/example.yaml").case
    workspace = CaseWorkspace.build(config)
    with pytest.raises(ValueError):
        workspace.energy[0] = 123.0
    kernel = initial_zero_pulse_kernel(config)
    first = FixedKernelSolver(workspace, kernel)
    second = FixedKernelSolver(workspace, kernel)
    assert first.pole_cache is None and second.pole_cache is None
    first.solve_trajectory(dense=False)
    assert first.pole_cache is not None
    assert second.pole_cache is None
    assert workspace.estimated_peak_bytes > 0
    assert set(workspace.lead_linewidths) == {"L", "R"}


def test_each_scba_iteration_has_fresh_state():
    config = load_config("examples/example.yaml").case
    workspace = CaseWorkspace.build(config)
    kernel = initial_zero_pulse_kernel(config)
    first = SCBAIteration(workspace, kernel, 0)
    second = SCBAIteration(workspace, kernel, 1)
    assert first is not second
    assert first.index == 0 and second.index == 1
    assert first.workspace is second.workspace


def test_protocol_types_are_explicit():
    assert UpwardProtocol.name == "upward"
    assert DownwardProtocol.name == "downward"
    assert SquareProtocol.name == "square"


def test_manuscript_stage_order_is_reported():
    config = load_config("examples/example.yaml")
    kernel = initial_zero_pulse_kernel(config.case)
    workspace = CaseWorkspace.build(config.case)
    events = []
    FixedKernelSolver(workspace, kernel).solve_trajectory(
        dense=False,
        progress=lambda phase, details: events.append(phase),
    )
    expected = [
        "mpm_fit",
        "endpoint_poles",
        "amplitude_a",
        "amplitude_c",
        "history_b",
        "history_d",
        "psi",
        "retarded_reconstruction",
        "lesser_sources",
        "streaming_projection",
    ]
    positions = [events.index(name) for name in expected]
    assert positions == sorted(positions)


def test_campaign_dashboard_is_atomic_and_ansi_free(tmp_path):
    run = tmp_path / "jobs" / "case"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"status":"running"}\n', encoding="utf-8")
    (run / "progress.json").write_text(json.dumps({
        "state": "running",
        "phase": "mpm_fit",
        "phase_label": "MPM rational-kernel fit",
        "outer_iteration_display": 1,
        "outer_max_iterations": 20,
        "last_completed_loss": None,
        "elapsed_seconds": 2.0,
    }), encoding="utf-8")
    manifest = {"jobs": [{
        "case_id": "case",
        "run_directory": str(run),
        "status": "created",
        "protocol": "upward",
        "bandwidth": 20.0,
        "phonon_energy": 0.2,
        "coupling_meV": 1.0,
    }]}
    dashboard = CampaignDashboard(tmp_path, manifest)
    dashboard.start()
    dashboard.refresh(reason="test")
    main = (tmp_path / "main.log").read_text(encoding="utf-8")
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")
    assert BANNER in main and "\x1b[" not in main
    assert "NEGF TDP PS-SCBA" in html
    assert "MPM rational-kernel fit" in main
    assert not list(tmp_path.glob(".dashboard.html.*"))


def test_run_plotter_is_the_public_plotting_owner():
    assert RunPlotter.__module__ == "psscba.backend.plotting.service"


def test_campaign_resume_skips_completed_jobs(tmp_path):
    run = tmp_path / "jobs" / "complete"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    manifest = {
        "config_format": "negf-tdp-ps-scba",
        "jobs": [{
            "case_id": "complete",
            "run_directory": str(run),
            "status": "complete",
            "method": "psscba",
        }],
    }
    (tmp_path / "campaign.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert resume_campaign(tmp_path) == [run]
    assert json.loads((tmp_path / "summary.json").read_text())["states"] == {"complete": 1}


def test_job_logs_are_separated(tmp_path):
    config = load_config("examples/example.yaml")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    tracker_a = RunTracker(first, config)
    tracker_b = RunTracker(second, config)
    tracker_a.log_message("only first")
    tracker_b.log_message("only second")
    assert "only first" in (first / "solver.log").read_text()
    assert "only second" not in (first / "solver.log").read_text()
    assert "only second" in (second / "solver.log").read_text()
