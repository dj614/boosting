from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

_SINGLE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def force_single_thread_numerics() -> None:
    for name in _SINGLE_THREAD_ENV_VARS:
        os.environ[name] = "1"



def resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    resolved = int(n_jobs)
    if resolved <= 0:
        return max(1, os.cpu_count() or 1)
    return resolved



def make_process_pool(max_workers: int) -> ProcessPoolExecutor:
    resolved = resolve_n_jobs(max_workers)
    if resolved <= 1:
        raise ValueError("make_process_pool requires max_workers > 1")
    return ProcessPoolExecutor(
        max_workers=resolved,
        mp_context=mp.get_context("spawn"),
        initializer=force_single_thread_numerics,
    )