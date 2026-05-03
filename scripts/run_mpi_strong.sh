#!/usr/bin/env bash
# Strong-scaling sweep: fixed problem size, varying ranks.
# Usage:  bash scripts/run_mpi_strong.sh [view] [width] [height]
#   view = --shallow | --deep   (default --shallow)
#   width, height in pixels    (default 1920 1080)

set -e
cd "$(dirname "$0")/.."

VIEW="${1:---shallow}"
W="${2:-1920}"
H="${3:-1080}"
RUNS="${RUNS:-3}"
RANKS_LIST="${RANKS_LIST:-1 2 4 8}"

echo "view=$VIEW  resolution=${W}x${H}  runs=$RUNS"
echo

for N in $RANKS_LIST; do
    echo "===== ranks=$N ====="
    mpirun -np "$N" --oversubscribe ./build/mandelbrot_mpi \
        "$VIEW" --width "$W" --height "$H" --no-output --runs "$RUNS" \
        2>&1 | tail -10
    echo
done
