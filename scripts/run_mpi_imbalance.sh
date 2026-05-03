#!/usr/bin/env bash
# Load-imbalance comparison: contiguous vs block-cyclic row distribution.
# Runs both at the SAME rank count and same view, prints per-rank times.
# Usage:  bash scripts/run_mpi_imbalance.sh [ranks] [view]

set -e
cd "$(dirname "$0")/.."

N="${1:-4}"
VIEW="${2:---deep}"
W="${W:-1920}"
H="${H:-1080}"

echo "===== contiguous, ranks=$N, view=$VIEW ====="
mpirun -np "$N" --oversubscribe ./build/mandelbrot_mpi \
    "$VIEW" --width "$W" --height "$H" --no-output --runs 3 --contiguous \
    2>&1 | tail -20
echo

echo "===== block-cyclic (chunk=8), ranks=$N, view=$VIEW ====="
mpirun -np "$N" --oversubscribe ./build/mandelbrot_mpi \
    "$VIEW" --width "$W" --height "$H" --no-output --runs 3 --chunk 8 \
    2>&1 | tail -20
echo
