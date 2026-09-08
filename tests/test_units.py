import numpy as np
import pytest

from psscba.backend.model.units import (
    current_from_uA,
    current_to_uA,
    energy_ev_to_gamma,
    energy_gamma_to_ev,
    energy_gamma_to_mev,
    energy_mev_to_gamma,
    time_from_ps,
    time_to_ps,
)


def test_energy_conversions_use_gamma_as_the_backend_unit():
    gamma_eV = 0.01
    assert energy_mev_to_gamma(20.0, gamma_eV) == pytest.approx(2.0)
    assert energy_ev_to_gamma(0.005, gamma_eV) == pytest.approx(0.5)
    assert energy_gamma_to_mev(2.0, gamma_eV) == pytest.approx(20.0)
    assert energy_gamma_to_ev(0.5, gamma_eV) == pytest.approx(0.005)
    values = np.array([0.0, 0.5, 20.0])
    assert np.allclose(energy_mev_to_gamma(values, gamma_eV), [0.0, 0.05, 2.0])


def test_energy_conversion_rejects_invalid_gamma():
    with pytest.raises(ValueError):
        energy_mev_to_gamma(1.0, 0.0)


def test_original_runner_physical_unit_factors_and_round_trips():
    gamma_eV = 0.01
    assert time_to_ps(1.0, gamma_eV) == pytest.approx(0.0658211957, rel=1e-10)
    assert current_to_uA(1.0, gamma_eV) == pytest.approx(2.4341348058, rel=1e-10)
    times = np.asarray([0.0, 0.25, 3.0])
    currents = np.asarray([-0.5, 0.0, 2.0])
    assert np.allclose(time_from_ps(time_to_ps(times, gamma_eV), gamma_eV), times)
    assert np.allclose(current_from_uA(current_to_uA(currents, gamma_eV), gamma_eV), currents)
