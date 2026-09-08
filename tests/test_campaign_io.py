import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from psscba import load_config
from psscba.frontend.campaign import aggregate, campaign_cases, case_config, run_case
from psscba.frontend.storage import create_run_directory, load_saved_observables, save_result
from psscba.types import FixedKernelTrajectory, ProjectionResult, PSSCBAResult, StationaryKernel
from psscba.backend.model.units import current_to_uA, time_to_ps
from psscba.frontend.validation import run_validation_program
from psscba.frontend.validation import validation_cases
from psscba.backend.reconstruction.engine import _observable_indices


def _smoke(tmp_path):
    config = load_config("examples/example.yaml")
    return replace(config, output_root=str(tmp_path))


def test_example_campaign_expands_to_all_protocols(tmp_path):
    cases = list(campaign_cases(_smoke(tmp_path)))
    assert len(cases) == 6
    assert {case.protocol.name for case in cases} == {"upward", "downward", "square"}


def test_validation_uses_configured_thermal_and_hot_references(tmp_path):
    config = _smoke(tmp_path)
    thermal = [
        (name, case)
        for name, case, _preparation in validation_cases(config)
        if name == "numerical_reference_thermal_dense"
    ]
    assert len(thermal) == 1
    name, case = thermal[0]
    assert case.model.phonon_occupation is None
    assert case.model.phonon_energy == pytest.approx(config.model.phonon_energy)
    assert case.projection.dense_validation is True
    assert case.numerics.energy_max <= config.validation.compact_energy_bound
    assert all(
        not item.config.projection.dense_validation
        or item.config.numerics.energy_max <= 40.0
        for item in validation_cases(config)
    )
    hot = next(
        case for name, case, _ in validation_cases(config)
        if name == f"hot_frequency_coupling_f{config.model.phonon_energy:g}_g1"
    )
    assert hot.model.phonon_occupation == pytest.approx(1.0)
    assert hot.model.beta_ph == pytest.approx(np.log(2.0) / config.model.phonon_energy)
    assert hot.model.phonon_energy != pytest.approx(np.pi / 3.0)


def test_validation_time_step_cases_respect_nyquist_bound(tmp_path):
    config = _smoke(tmp_path)
    cases = dict((name, case) for name, case, _ in validation_cases(config))
    assert (cases["numerical_dt_0.04"].numerics.energy_min, cases["numerical_dt_0.04"].numerics.energy_max) == (-40.0, 40.0)
    assert (cases["numerical_dt_0.02"].numerics.energy_min, cases["numerical_dt_0.02"].numerics.energy_max) == (-40.0, 40.0)
    assert (cases["numerical_dt_0.01"].numerics.energy_min, cases["numerical_dt_0.01"].numerics.energy_max) == (-40.0, 40.0)
    for case in cases.values():
        case.validate()


def test_validation_cases_have_tiered_labels_and_occupation_modes(tmp_path):
    config = _smoke(tmp_path)
    cases = validation_cases(config)
    foundation = next(item for item in cases if item.name.startswith("foundation_g0_"))
    assert foundation.tier == "foundation"
    assert foundation.gating is True
    assert foundation.config.model.phonon_occupation is None
    hot = next(item for item in cases if item.family == "hot_N1")
    assert hot.tier == "stress"
    assert hot.gating is False
    assert hot.config.model.phonon_occupation == pytest.approx(1.0)


def test_validation_prehistory_resolution_axes_are_materialized(tmp_path):
    config = _smoke(tmp_path)
    cases = validation_cases(config)
    orders = {
        item.config.numerics.collision_prehistory_quadrature_order
        for item in cases if item.family == "collision"
    }
    scales = {
        item.config.numerics.collision_prehistory_scale
        for item in cases if item.family == "collision"
    }
    assert orders == {32, 64, 128}
    assert scales == {0.5, 1.0, 2.0}


def test_current_time_points_selects_nested_nodes_and_rejects_non_nested():
    config = _smoke(Path("/tmp"))
    grid = np.linspace(-0.2, 0.4, 7)
    selected = _observable_indices(
        replace(config, numerics=replace(config.numerics, current_time_points=3)),
        grid,
    )
    assert selected.tolist() == [2, 4, 6]
    with pytest.raises(ValueError, match="nested subset"):
        _observable_indices(
            replace(config, numerics=replace(config.numerics, current_time_points=4)),
            grid,
        )


def test_campaign_grid_is_declared_by_the_configuration(tmp_path):
    base = _smoke(tmp_path)
    wide = case_config(base, "upward", 100.0, 5.0)
    square = case_config(base, "square", 20.0, 1.0)
    assert (wide.numerics.energy_min, wide.numerics.energy_max) == (
        base.numerics.energy_min,
        base.numerics.energy_max,
    )
    assert wide.projection.time_step == base.projection.time_step
    assert square.protocol.duration == base.campaign.square_duration
    assert square.projection.time_max >= square.protocol.duration
    assert np.isclose(
        (square.projection.time_max - square.projection.time_min)
        / square.projection.time_step,
        round(
            (square.projection.time_max - square.projection.time_min)
            / square.projection.time_step
        ),
    )


def test_run_directory_contains_resolved_versioned_configuration(tmp_path):
    config = _smoke(tmp_path)
    run_dir = create_run_directory(config, label="unit")
    with open(run_dir / "resolved.yaml", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle)
    assert resolved["config_format"] == "negf-tdp-ps-scba"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "created"


def test_result_storage_matches_original_physical_convention(tmp_path):
    config = _smoke(tmp_path)
    run_dir = create_run_directory(config, label="units")
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
    natural_time = np.asarray([0.0, 1.0, 2.0])
    natural_current = np.asarray([0.0, 1.0, -2.0])
    trajectory = FixedKernelTrajectory(
        time=grid, energy=grid, projected=projection,
        observable_time=natural_time, currents={"L": natural_current},
        corrected_currents={"L": natural_current + 0.25},
        raw_continuity_residual=np.asarray([0.1, 0.2, 0.3]),
    )
    save_result(run_dir, PSSCBAResult(True, kernel, trajectory, ()))
    units = json.loads((run_dir / "units.json").read_text())
    assert units["storage"]["time"] == "ps"
    assert units["storage"]["current"] == "uA"
    assert np.allclose(
        np.load(run_dir / "data/observable_time.npy"),
        time_to_ps(natural_time, config.model.gamma_eV),
    )
    assert np.allclose(
        np.load(run_dir / "data/current_L.npy"),
        current_to_uA(natural_current, config.model.gamma_eV),
    )
    assert np.allclose(
        np.load(run_dir / "data/current_corrected_L.npy"),
        current_to_uA(natural_current + 0.25, config.model.gamma_eV),
    )
    assert np.allclose(
        np.load(run_dir / "data/continuity_residual_raw.npy"),
        [0.1, 0.2, 0.3],
    )
    loaded_time, loaded_currents = load_saved_observables(run_dir, config)
    assert np.allclose(loaded_time, natural_time)
    assert np.allclose(loaded_currents["L"], natural_current)


def test_run_case_fails_closed_and_records_exception(tmp_path, monkeypatch):
    config = _smoke(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional campaign failure")

    monkeypatch.setattr("psscba.frontend.campaign.solve_psscba", fail)
    with pytest.raises(RuntimeError, match="intentional campaign failure"):
        run_case(config)
    run_dir = next(tmp_path.iterdir())
    failure = json.loads((run_dir / "failure.json").read_text())
    assert failure["type"] == "RuntimeError"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"


def test_aggregate_collects_only_complete_runs(tmp_path):
    config = _smoke(tmp_path)
    complete = create_run_directory(config, label="complete")
    failed = create_run_directory(config, label="failed")
    (complete / "status.json").write_text('{"status":"complete"}\n')
    (failed / "status.json").write_text('{"status":"failed"}\n')
    (complete / "diagnostics.json").write_text(
        '{"method":"psscba","r_green_retarded":0.0}\n'
    )
    (failed / "diagnostics.json").write_text(
        '{"method":"psscba","r_green_retarded":1.0}\n'
    )
    output = aggregate(tmp_path)
    records = json.loads(output.read_text())
    assert len(records) == 1
    assert records[0]["run"] == complete.name


def test_validation_manifest_is_incremental_and_resume_skips_accepted(tmp_path, monkeypatch):
    config = _smoke(tmp_path)
    planned = [("first", config, False), ("second", config, False)]
    monkeypatch.setattr("psscba.frontend.validation.validation_cases", lambda _config: planned)
    monkeypatch.setattr(
        "psscba.frontend.validation.run_oracle_preflight",
        lambda _config, **_kwargs: {},
    )
    calls = []

    def fake_run(_config, *, preparation_fixed=False, run_dir=None):
        run = Path(run_dir)
        calls.append(run)
        return run

    monkeypatch.setattr("psscba.frontend.validation.run_case", fake_run)
    assert len(run_validation_program(config, session_path=None)) == 2
    created_session = next(tmp_path.glob("validation_*"))
    manifest = created_session / "campaign.json"
    payload = json.loads(manifest.read_text())
    assert [item["status"] for item in payload["jobs"]] == ["accepted", "accepted"]
    assert (created_session / "main.log").exists()
    assert (created_session / "dashboard.html").exists()
    calls.clear()
    assert len(run_validation_program(config, session_path=created_session)) == 2
    assert calls == []


def test_validation_continues_after_case_failure_and_accumulates_failures(tmp_path, monkeypatch):
    config = _smoke(tmp_path)
    planned = [
        ("first", config, False),
        ("second", config, False),
        ("third", config, False),
    ]
    monkeypatch.setattr("psscba.frontend.validation.validation_cases", lambda _config: planned)
    monkeypatch.setattr(
        "psscba.frontend.validation.run_oracle_preflight",
        lambda _config, **_kwargs: {},
    )
    calls = []

    def fake_run(_config, *, preparation_fixed=False, run_dir=None):
        run = Path(run_dir)
        calls.append(run.name)
        if run.name.startswith("000_"):
            raise RuntimeError("intentional grid failure")
        return run

    monkeypatch.setattr("psscba.frontend.validation.run_case", fake_run)
    assert len(run_validation_program(config, session_path=None)) == 2

    session = next(tmp_path.glob("validation_*"))
    payload = json.loads((session / "campaign.json").read_text())
    assert calls == ["000_first", "001_second", "002_third"]
    assert [item["status"] for item in payload["jobs"]] == [
        "failed", "accepted", "accepted"
    ]
    assert payload["failure_summary"]["gating_count"] == 0
    assert payload["failure_summary"]["diagnostic_count"] == 1
    assert payload["failure_summary"]["diagnostic_case_ids"] == ["000_first"]
    assert payload["status"] == "complete_with_diagnostic_failures"


def test_validation_raw_continuity_is_reported_without_blocking(tmp_path, monkeypatch):
    config = _smoke(tmp_path)
    monkeypatch.setattr(
        "psscba.frontend.validation.validation_cases",
        lambda _config: [],
    )
    monkeypatch.setattr(
        "psscba.frontend.validation.run_oracle_preflight",
        lambda _config, **_kwargs: {
            "upward_raw_continuity_scaled_error": 0.25,
            "upward_corrected_continuity_scaled_error": 0.0,
        },
    )

    assert run_validation_program(config) == []
    session = next(tmp_path.glob("validation_*"))
    manifest = json.loads((session / "campaign.json").read_text())
    assert manifest["status"] == "complete"
    main = (session / "main.log").read_text(encoding="utf-8")
    assert "raw g=0 continuity WARNING" in main


def test_validation_status_update_normalizes_relative_and_absolute_paths(tmp_path):
    from psscba.frontend.storage import atomic_json
    from psscba.frontend.validation import update_validation_run_status

    run = tmp_path / "run"
    run.mkdir()
    session = tmp_path / "validation" / "campaign.json"
    atomic_json(session, {"config_format": "negf-tdp-ps-scba", "status": "interrupted", "jobs": [{
        "run_directory": str(run), "status": "interrupted",
    }]})
    update_validation_run_status(session, run, "running")
    assert json.loads(session.read_text())["jobs"][0]["status"] == "running"
