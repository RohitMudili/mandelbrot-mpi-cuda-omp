#!/usr/bin/env python3
"""Generate Part B plots from the unified benchmark CSV + schedule study CSV.

Reads:
  docs/results/benchmark.csv             (from ./build/benchmark)
  docs/results/schedule_comparison.csv   (from scripts/run_schedule_study.sh)

Writes:
  docs/results/openmp_strong_scaling.png
  docs/results/openmp_schedule_comparison.png
  docs/results/throughput_comparison.png
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "docs", "results")

bench_csv = os.path.join(RES, "benchmark.csv")
sched_csv = os.path.join(RES, "schedule_comparison.csv")

if not os.path.exists(bench_csv):
    sys.exit(f"missing {bench_csv}; run ./build/benchmark first")

bench = pd.read_csv(bench_csv)
serial = bench[bench.implementation == "serial"]
omp    = bench[bench.implementation == "openmp"]
cuda   = bench[bench.implementation == "cuda"]

serial_avg = serial.time_ms.mean()
omp_agg = omp.groupby("workers", as_index=False).time_ms.mean().sort_values("workers")
omp_agg["speedup"] = serial_avg / omp_agg.time_ms

# ---- 1. OpenMP strong-scaling plot ---------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(omp_agg.workers, omp_agg.speedup, "o-", lw=2, ms=8, label="measured")
mx = max(omp_agg.workers.max(), 1)
ax.plot([1, mx], [1, mx], "k--", alpha=0.5, label="ideal (y=x)")
ax.set_xlabel("OpenMP threads")
ax.set_ylabel("speedup vs serial")
ax.set_title("OpenMP strong scaling — deep view (1920x1080, maxIter=2000)")
ax.set_xticks(omp_agg.workers)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
out = os.path.join(RES, "openmp_strong_scaling.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")

# ---- 2. Throughput comparison (Serial / OMP / CUDA) ---------------------
fig, ax = plt.subplots(figsize=(8, 5))
labels, vals = [], []
labels.append("serial")
vals.append(bench[bench.implementation == "serial"].mpixels_per_sec.mean())
for w in sorted(omp.workers.unique()):
    labels.append(f"omp({w})")
    vals.append(omp[omp.workers == w].mpixels_per_sec.mean())
labels.append("cuda")
vals.append(cuda.mpixels_per_sec.mean())
colors = ["#888"] + ["#3477eb"] * len(sorted(omp.workers.unique())) + ["#76b900"]
bars = ax.bar(labels, vals, color=colors)
ax.set_ylabel("MPixels / sec")
ax.set_title("Throughput — Serial vs OpenMP vs CUDA (deep view)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(RES, "throughput_comparison.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")

# ---- 3. Schedule comparison ---------------------------------------------
if os.path.exists(sched_csv):
    sched = pd.read_csv(sched_csv)
    s_avg = sched.groupby("schedule", as_index=False).time_ms.mean()
    order = ["static", "static-16", "dynamic-1", "dynamic-16", "dynamic-64", "guided"]
    s_avg["order"] = s_avg.schedule.map({n: i for i, n in enumerate(order)})
    s_avg = s_avg.sort_values("order")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(s_avg.schedule, s_avg.time_ms,
                  color=["#d63a3a"] + ["#3477eb"] * (len(s_avg) - 1))
    ax.set_ylabel("wall time (ms, lower is better)")
    ax.set_title("OpenMP schedule comparison — 8 threads, deep view")
    for b, v in zip(bars, s_avg.time_ms):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RES, "openmp_schedule_comparison.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
else:
    print(f"skipped schedule plot ({sched_csv} not found)")
