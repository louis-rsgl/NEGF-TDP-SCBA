from __future__ import annotations

import os
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from psscba.frontend.configuration import RunConfig


def estimate_case_memory_bytes(config: RunConfig) -> int:
    n_time = len(config.projection.time_grid())
    n_energy = len(config.numerics.energy_grid())
    complex_bytes = np.dtype(np.complex128).itemsize
    real_bytes = np.dtype(np.float64).itemsize
    return int(
        12 * n_time * n_time * complex_bytes
        + 7 * n_time * n_energy * complex_bytes
        + 8 * (n_time + n_energy) * real_bytes
    )


def profile_blas(config: RunConfig) -> dict:
    rng = np.random.default_rng(20260828)
    size = 1024
    inner = 512
    left = rng.standard_normal((size, inner)) + 1j * rng.standard_normal((size, inner))
    right = rng.standard_normal((inner, size)) + 1j * rng.standard_normal((inner, size))
    host_cores = int(os.cpu_count() or 1)
    memory_total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    memory_limit = int(0.8 * min(memory_total, 128 * 1024**3))
    from psscba.frontend.campaign import campaign_cases

    case_memory = max(
        estimate_case_memory_bytes(case) for case in campaign_cases(config)
    )
    records = []
    for threads in (1, 4, 8, 16):
        with threadpool_limits(limits=threads, user_api="blas"):
            _ = left @ right
            start = perf_counter()
            for _ in range(3):
                product = left @ right
            elapsed = perf_counter() - start
        del product
        throughput = 3.0 / elapsed
        workers = max(
            1,
            min(host_cores // threads, memory_limit // max(case_memory, 1)),
        )
        records.append(
            {
                "blas_threads": threads,
                "multiply_per_second": throughput,
                "admissible_workers": workers,
                "aggregate_score": throughput * workers,
            }
        )
    selected = max(records, key=lambda item: item["aggregate_score"])
    return {
        "host_cores": host_cores,
        "host_memory_bytes": memory_total,
        "memory_limit_bytes": memory_limit,
        "estimated_case_peak_bytes": case_memory,
        "records": records,
        "recommended_blas_threads": selected["blas_threads"],
        "recommended_workers": selected["admissible_workers"],
    }
