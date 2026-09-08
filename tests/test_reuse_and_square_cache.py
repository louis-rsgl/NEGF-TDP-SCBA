from dataclasses import replace

import numpy as np

from psscba.backend.poles.minimal import build_pole_cache
from psscba.backend.protocols.square import (
    SquareKernelCacheStore,
    analytic_lower_half_plane_residue,
    build_square_kernel_cache,
)
from psscba.backend.stationary.initializer import initial_zero_pulse_kernel, make_system
from tests.test_psscba_solver import small_config


def test_aaa_basis_reuse_is_validated_and_marked():
    config = small_config(dense=False)
    config = replace(
        config,
        model=replace(config.model, coupling_meV=0.1),
        numerics=replace(config.numerics, mpm_aaa_initial_terms=8, mpm_aaa_max_terms=16),
    )
    system = make_system(config)
    kernel = initial_zero_pulse_kernel(config)
    first = build_pole_cache(system, kernel)
    second = build_pole_cache(system, kernel, previous_basis=first.basis)
    assert first.basis is not None
    assert second.pole_basis_reused
    assert second.fit_method == "reused_causal_aaa_basis"
    assert np.max(np.min(np.abs(second.zeta[:, None] - first.basis.zeta[None, :]), axis=1)) < 1e-12


def test_square_cache_store_keeps_independent_energy_grids():
    config = replace(
        small_config(dense=False),
        protocol=replace(small_config(dense=False).protocol, name="square", duration=0.1),
    )
    system = make_system(config)
    kernel = initial_zero_pulse_kernel(config)
    poles = build_pole_cache(system, kernel)
    store = SquareKernelCacheStore()
    coarse = np.linspace(-1.0, 1.0, 5)
    fine = np.linspace(-1.0, 1.0, 7)
    first = build_square_kernel_cache(system, coarse, cache=poles, store=store)
    again = build_square_kernel_cache(system, coarse, cache=poles, store=store)
    other = build_square_kernel_cache(system, fine, cache=poles, store=store)
    assert first is again
    assert other is not first
    assert store.size == 2
    assert set(other.S_C) == {-system.w_q, 0.0, system.w_q}


def test_analytic_residue_handles_simple_and_removable_candidates():
    pole = 0.2 - 0.4j
    value = analytic_lower_half_plane_residue(
        lambda z: 2.5 / (z - pole), [pole], analytic_step=1e-6
    )
    assert value == complex(-2.5) or abs(value + 2.5) < 1e-10
    removable = analytic_lower_half_plane_residue(
        lambda z: (z - pole) / (z - pole), [pole], analytic_step=1e-6
    )
    assert abs(removable) < 1e-8
