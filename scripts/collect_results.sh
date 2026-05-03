#!/usr/bin/env bash
# Run all Part A scaling/benchmark sweeps and produce CSVs + plots.
# Outputs land in docs/results/.

set -e
cd "$(dirname "$0")/.."
mkdir -p docs/results

echo "============================================================"
echo "[1/4] Strong scaling (1920x1080 shallow, ranks 1 2 4 8)"
echo "============================================================"
STRONG_CSV="docs/results/benchmark_mpi.csv"
rm -f "$STRONG_CSV"
FIRST=1
for N in 1 2 4 8; do
    EXTRA=""
    if [ "$FIRST" = "1" ]; then EXTRA="--header"; FIRST=0; fi
    echo "===== ranks=$N ====="
    mpirun -np "$N" --oversubscribe ./build/benchmark_mpi \
        --shallow --width 1920 --height 1080 --runs 5 \
        --csv "$STRONG_CSV" --impl mpi $EXTRA
done

echo
echo "============================================================"
echo "[2/4] Weak scaling (pixels-per-rank constant, ranks 1 2 4 8)"
echo "============================================================"
# Use benchmark_mpi for weak scaling too: per-rank invocation with growing W,H.
CSV="docs/results/benchmark_mpi_weak.csv"
rm -f "$CSV"
declare -A WIDTH_FOR
declare -A HEIGHT_FOR
WIDTH_FOR[1]=960;  HEIGHT_FOR[1]=540
WIDTH_FOR[2]=1358; HEIGHT_FOR[2]=763
WIDTH_FOR[4]=1920; HEIGHT_FOR[4]=1080
WIDTH_FOR[8]=2715; HEIGHT_FOR[8]=1527
FIRST=1
for N in 1 2 4 8; do
    W="${WIDTH_FOR[$N]}"
    H="${HEIGHT_FOR[$N]}"
    EXTRA=""
    if [ "$FIRST" = "1" ]; then EXTRA="--header"; FIRST=0; fi
    echo "===== ranks=$N  resolution=${W}x${H} ====="
    mpirun -np "$N" --oversubscribe ./build/benchmark_mpi \
        --shallow --width "$W" --height "$H" --runs 5 \
        --csv "$CSV" --impl mpi_weak $EXTRA
done

echo
echo "============================================================"
echo "[3/4] Imbalance: contiguous vs cyclic (deep, 4 ranks)"
echo "============================================================"
IMB_CSV="docs/results/imbalance.csv"
echo "distribution,wall_time_ms,imbalance_ratio" > "$IMB_CSV"
for DIST in contiguous cyclic; do
    if [ "$DIST" = "contiguous" ]; then
        OUT=$(mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi \
              --deep --width 1920 --height 1080 --no-output --runs 3 --contiguous)
    else
        OUT=$(mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi \
              --deep --width 1920 --height 1080 --no-output --runs 3 --chunk 8)
    fi
    AVG=$(echo "$OUT" | awk '/avg total/{print $3}')
    RATIO=$(echo "$OUT" | awk '/imbalance ratio/{print $4}' | tr -d 'x')
    echo "$DIST,$AVG,$RATIO" >> "$IMB_CSV"
    echo "  $DIST: avg=${AVG}ms  imbalance=${RATIO}x"
done

echo
echo "============================================================"
echo "[4/4] Generate plots"
echo "============================================================"
python3 scripts/plot_mpi_results.py

echo
echo "Done. Results in docs/results/:"
ls -la docs/results/
