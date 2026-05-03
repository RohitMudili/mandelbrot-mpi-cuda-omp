#!/usr/bin/env bash
# Schedule comparison study (Part B §B.5).
# Runs mandelbrot_omp at 8 threads on the deep view across six OpenMP schedules.
# Emits docs/results/schedule_comparison.csv

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="docs/results/schedule_comparison.csv"
mkdir -p docs/results
echo "schedule,run,time_ms" > "$OUT"

THREADS=8
RUNS=3

for sched in static static-16 dynamic-1 dynamic-16 dynamic-64 guided; do
  echo "=== schedule=$sched threads=$THREADS ==="
  for r in $(seq 1 $RUNS); do
    line=$(./build/mandelbrot_omp --deep --threads $THREADS --schedule "$sched" \
                                  --runs 1 --no-output 2>/dev/null \
           | grep "^trial 1:" | awk '{print $3}')
    echo "  run $r: ${line} ms"
    echo "$sched,$r,$line" >> "$OUT"
  done
done

echo
echo "Wrote $OUT"
