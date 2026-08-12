import numpy as np
import pytest

from backend.units import (
    energy_ev_to_gamma,
    energy_gamma_to_ev,
    energy_gamma_to_mev,
    energy_mev_to_gamma,
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
