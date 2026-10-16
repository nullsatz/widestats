"""Turn bench/results/results.csv into a markdown table of speedups.

For each function and size: widestats on the GPU (float32, and float64 for
Granger, its default on CUDA), widestats on the CPU, and the fastest CPU
baseline, with the GPU's speedup over that baseline.

    python bench/report.py [bench/results]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def fmt(s):
    """Format seconds for the table."""
    return "—" if s is None else (f"{s:.3f} s" if s < 10 else f"{s:.0f} s")


def main(folder="bench/results"):
    """Write results.md from results.csv and print it."""
    rows = list(csv.DictReader(open(Path(folder) / "results.csv")))
    cases = defaultdict(dict)
    order = []
    for r in rows:
        key = (r["function"], r["size"])
        if key not in cases:
            order.append(key)
        t = float(r["seconds"])
        if t == t:                                       # not NaN
            cases[key][r["implementation"]] = t
    out = ["| function | size | best CPU code | its time | widestats CPU "
           "| widestats GPU | GPU speedup |",
           "|---|---|---|---|---|---|---|"]
    for fn, size in order:
        c = cases[(fn, size)]
        gpu = c.get("widestats GPU float64" if fn == "granger"
                    else "widestats GPU float32")
        cpu = c.get("widestats CPU float64")
        base = {k: v for k, v in c.items() if not k.startswith("widestats")}
        if base:
            name, t = min(base.items(), key=lambda kv: kv[1])
            speed = f"**{t / gpu:,.0f}×**"  # noqa: RUF001
        else:
            name, t, speed = "not run (too slow)", None, "—"
        out.append(f"| `{fn}` | {size} | {name} | {fmt(t)} | {fmt(cpu)} "
                   f"| {fmt(gpu)} | {speed} |")
    text = "\n".join(out) + "\n"
    (Path(folder) / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:])
