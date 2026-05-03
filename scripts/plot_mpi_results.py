#!/usr/bin/env python3
"""Generate Part A scaling plots from CSV inputs.

Inputs:
  docs/results/benchmark_mpi.csv          (strong scaling, fixed problem size)
  docs/results/benchmark_mpi_weak.csv     (weak scaling, pixels-per-rank held)
  docs/results/imbalance.csv              (contiguous vs cyclic, deep view)

Outputs (PNG, written to docs/results/):
  strong_scaling.png
  weak_scaling.png
  imbalance.png
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "docs", "results")
os.makedirs(RESULTS, exist_ok=True)


def avg_by_rank(csv_path):
    """Read benchmark_mpi CSV; return {workers: avg_time_ms}."""
    bucket = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bucket[int(row["workers"])].append(float(row["time_ms"]))
    return {w: sum(v) / len(v) for w, v in bucket.items()}


def plot_strong(csv_path, out_path):
    data = avg_by_rank(csv_path)
    ranks = sorted(data.keys())
    times = [data[r] for r in ranks]
    t1 = data[ranks[0]]
    speedup = [t1 / data[r] for r in ranks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ranks, times, "o-", label="measured", color="#1f77b4")
    ax1.set_xlabel("MPI ranks")
    ax1.set_ylabel("avg total time (ms)")
    ax1.set_title("Strong scaling: time vs ranks")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xticks(ranks); ax1.set_xticklabels(ranks)

    ax2.plot(ranks, speedup, "o-", label="measured", color="#1f77b4")
    ax2.plot(ranks, ranks, "k--", alpha=0.5, label="ideal (linear)")
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("speedup vs 1 rank")
    ax2.set_title("Strong scaling: speedup")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(ranks); ax2.set_xticklabels(ranks)

    fig.suptitle(f"Strong scaling — {os.path.basename(csv_path)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_weak(csv_path, out_path):
    data = avg_by_rank(csv_path)
    ranks = sorted(data.keys())
    times = [data[r] for r in ranks]
    t1 = data[ranks[0]]
    eff = [t1 / data[r] for r in ranks]  # ideal weak: time stays ~constant -> eff=1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ranks, times, "o-", color="#2ca02c")
    ax1.set_xlabel("MPI ranks (problem grows with ranks)")
    ax1.set_ylabel("avg total time (ms)")
    ax1.set_title("Weak scaling: time per step")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ranks); ax1.set_xticklabels(ranks)

    ax2.plot(ranks, eff, "o-", color="#2ca02c", label="measured")
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="ideal (1.0)")
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("efficiency  (T_1 / T_N)")
    ax2.set_title("Weak scaling: efficiency")
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(ranks); ax2.set_xticklabels(ranks)

    fig.suptitle(f"Weak scaling — {os.path.basename(csv_path)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def plot_imbalance(csv_path, out_path):
    """imbalance.csv schema: distribution,wall_time_ms,imbalance_ratio"""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["distribution"],
                         float(row["wall_time_ms"]),
                         float(row["imbalance_ratio"])))
    labels = [r[0] for r in rows]
    walls  = [r[1] for r in rows]
    ratios = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bars = ax1.bar(labels, walls, color=["#d62728", "#1f77b4"])
    ax1.set_ylabel("wall time (ms)")
    ax1.set_title("Wall time")
    for b, v in zip(bars, walls):
        ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom")

    bars2 = ax2.bar(labels, ratios, color=["#d62728", "#1f77b4"])
    ax2.set_ylabel("imbalance (max/min per-rank time)")
    ax2.set_title("Per-rank imbalance ratio")
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ideal (1.0)")
    for b, v in zip(bars2, ratios):
        ax2.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}x", ha="center", va="bottom")
    ax2.legend()

    fig.suptitle("Load imbalance — contiguous vs block-cyclic (deep view, 4 ranks)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main():
    strong = os.path.join(RESULTS, "benchmark_mpi.csv")
    weak   = os.path.join(RESULTS, "benchmark_mpi_weak.csv")
    imb    = os.path.join(RESULTS, "imbalance.csv")

    if os.path.exists(strong):
        plot_strong(strong, os.path.join(RESULTS, "strong_scaling.png"))
    else:
        print(f"skip: {strong} missing", file=sys.stderr)

    if os.path.exists(weak):
        plot_weak(weak, os.path.join(RESULTS, "weak_scaling.png"))
    else:
        print(f"skip: {weak} missing", file=sys.stderr)

    if os.path.exists(imb):
        plot_imbalance(imb, os.path.join(RESULTS, "imbalance.png"))
    else:
        print(f"skip: {imb} missing", file=sys.stderr)


if __name__ == "__main__":
    main()
