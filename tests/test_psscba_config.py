from dataclasses import replace
from pathlib import Path

import pytest

from psscba.frontend.configuration import (
    ModelConfig,
    NumericsConfig,
    ProjectionConfig,
    ProtocolConfig,
    RunConfig,
    ValidationConfig,
    load_config,
    require_shared_production_grid,
)
from psscba.frontend import cli


def test_yaml_schema_and_unknown_keys(tmp_path):
    path = tmp_path / "case.yaml"
    path.write_text(
        """config_format: negf-tdp-ps-scba
model:
  bandwidth: 20.0
  coupling_meV: 1.0
protocol:
  name: square
  duration: 0.4
projection:
  time_min: -0.2
  time_max: 0.6
  time_step: 0.1
numerics:
  energy_min: -6.0
  energy_max: 6.0
  energy_step: 0.2
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.protocol.duration == 0.4
    assert len(config.projection.time_grid()) == 9
    path.write_text("config_format: wrong\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown RunConfig"):
        load_config(path)


def test_config_rejects_nyquist_and_missing_switch_window():
    config = RunConfig(
        protocol=ProtocolConfig(name="upward"),
        projection=ProjectionConfig(time_min=-0.2, time_max=0.2, time_step=1.0),
        numerics=NumericsConfig(energy_min=-6, energy_max=6, energy_step=0.2),
    )
    with pytest.raises(ValueError):
        config.validate()


def test_current_representation_is_explicitly_validated():
    config = RunConfig(
        numerics=NumericsConfig(current_representation="unsupported"),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="current_representation"):
        config.validate()


def test_validation_schema_has_declared_frequency_and_collision_axes():
    config = load_config("configs/production.yaml")
    assert config.model.phonon_energy == pytest.approx(0.2)
    assert config.validation.frequencies == (0.1, 0.2, 0.5, 1.0, 2.0)
    assert config.numerics.collision_prehistory_quadrature_order == 64
    assert config.numerics.collision_prehistory_scale == pytest.approx(1.0)
    with pytest.raises(ValueError, match="streamed window"):
        replace(config, validation=ValidationConfig(
            streamed_window_mins=(-2.0,), streamed_window_maxs=(3.0, 6.0)
        )).validate()


def test_production_validation_requires_one_shared_energy_grid():
    base = RunConfig(
        numerics=NumericsConfig(
            energy_min=-2.0,
            energy_max=2.0,
            energy_step=0.1,
        )
    )
    # Null current-grid fields are the canonical production spelling.
    require_shared_production_grid(base)

    matching = RunConfig(
        numerics=NumericsConfig(
            energy_min=-2.0,
            energy_max=2.0,
            energy_step=0.1,
            current_energy_min=-2.0,
            current_energy_max=2.0,
            current_energy_points=41,
        )
    )
    require_shared_production_grid(matching)

    split = RunConfig(
        numerics=NumericsConfig(
            energy_min=-2.0,
            energy_max=2.0,
            energy_step=0.1,
            current_energy_min=-1.0,
            current_energy_max=1.0,
            current_energy_points=21,
        )
    )
    with pytest.raises(ValueError, match="require the current quadrature"):
        require_shared_production_grid(split)


def test_plain_cli_validate_never_launches_a_numerical_case(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "case.yaml"
    path.write_text("config_format: negf-tdp-ps-scba\n", encoding="utf-8")

    def reject_run(*_args, **_kwargs):
        raise AssertionError("plain validate must not launch run_case")

    monkeypatch.setattr(cli, "run_case", reject_run)
    assert cli.main(["validate", str(path)]) == 0
    assert "stationary_error" in capsys.readouterr().out
