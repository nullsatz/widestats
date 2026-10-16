"""Profile each widestats function on the GPU at a realistic size.

Prints, per function, the wall time and the operations taking the most
time (CPU and CUDA) from torch.profiler, plus how much of the wall time the
GPU was busy, so hotspots outside the GPU (host-side preparation, copies)
show up.

    python bench/profile.py [function ...]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import widestats as ws

DEV = "cuda"


def cases():
    """Build each function's profiling case at a realistic size."""
    rng = np.random.default_rng(0)
    x20k = rng.standard_normal((100, 20000))
    xp = rng.exponential(size=(50, 3000))
    yp = rng.exponential(size=(50, 3000))
    genes = rng.standard_normal((100, 20000))
    group = np.repeat([0, 1], 50)
    walks = rng.standard_normal((1000, 1000)).cumsum(axis=0)
    xmi = rng.standard_normal((1000, 3000))
    return {
        "cor_pairs": lambda: ws.cor_pairs(x20k, threshold=0.8, device=DEV),
        "cor_perm": lambda: ws.cor_perm(xp, yp, B=1000, seed=1, device=DEV),
        "ttest_perm": lambda: ws.ttest_perm(genes, group, B=10000, seed=1,
                                            device=DEV),
        "granger": lambda: ws.granger(walks, lag=2, device=DEV),
        "mutual_info": lambda: ws.mutual_info(xmi, device=DEV),
    }


def main(names):
    """Profile the named cases (all by default) and print the tables."""
    all_cases = cases()
    for name in names or all_cases:
        f = all_cases[name]
        f()                                   # warm up
        torch.cuda.synchronize()
        t = time.perf_counter()
        f()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                     ) as prof:
            f()
            torch.cuda.synchronize()
        ka = prof.key_averages()
        gpu_busy = sum(e.self_device_time_total for e in ka) / 1e6
        print(f"\n===== {name}: wall {wall:.3f} s, GPU busy {gpu_busy:.3f} s "
              f"({100 * gpu_busy / wall:.0f}%)")
        print(ka.table(sort_by="self_cpu_time_total", row_limit=8,
                       max_name_column_width=50))
        print(ka.table(sort_by="self_device_time_total", row_limit=8,
                       max_name_column_width=50))


if __name__ == "__main__":
    main(sys.argv[1:])
