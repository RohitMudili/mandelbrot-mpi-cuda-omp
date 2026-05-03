#!/usr/bin/env bash
# Comprehensive test of Part A implementation.
# Runs everything we'd want to verify before declaring Part A done.

set -e
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
RESULTS=()

ok()   { echo "  PASS: $1"; PASS=$((PASS+1)); RESULTS+=("PASS  $1"); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); RESULTS+=("FAIL  $1"); }

hdr() { echo; echo "===== $1 ====="; }

# -----------------------------------------------------------
hdr "1. Build artifacts present"
# -----------------------------------------------------------
for bin in mandelbrot mandelbrot_mpi mandelbrot_mpi_cuda benchmark; do
    if [ -x "build/$bin" ]; then ok "build/$bin exists and is executable"
    else fail "build/$bin missing"; fi
done

# -----------------------------------------------------------
hdr "2. Untouched-files invariant"
# -----------------------------------------------------------
SRC=/mnt/c/CODE/mandelbrot-cuda-main/mandelbrot-cuda-main
for f in src/mandelbrot.cu include/mandelbrot.cuh src/main.cpp src/benchmark.cpp; do
    if cmp -s "$SRC/$f" "$f"; then ok "$f unchanged from original"
    else fail "$f DIFFERS from original"; fi
done

# -----------------------------------------------------------
hdr "3. mandelbrot_mpi runs at 1, 2, 4 ranks (shallow, 400x300)"
# -----------------------------------------------------------
for N in 1 2 4; do
    if mpirun -np $N --oversubscribe ./build/mandelbrot_mpi \
            --shallow --width 400 --height 300 --no-output --runs 1 \
            > /tmp/mpi_$N.log 2>&1; then
        TIME=$(grep -E "^avg total" /tmp/mpi_$N.log | head -1 | awk '{print $3}')
        ok "mandelbrot_mpi ranks=$N (avg=${TIME} ms)"
    else
        fail "mandelbrot_mpi ranks=$N — see /tmp/mpi_$N.log"
    fi
done

# -----------------------------------------------------------
hdr "4. mandelbrot_mpi_cuda runs at 1, 2 ranks (shallow, 400x300)"
# -----------------------------------------------------------
for N in 1 2; do
    if mpirun -np $N --oversubscribe ./build/mandelbrot_mpi_cuda \
            --shallow --width 400 --height 300 --no-output --runs 1 \
            > /tmp/mpicuda_$N.log 2>&1; then
        TIME=$(grep -E "^avg" /tmp/mpicuda_$N.log | head -1 | awk '{print $2}')
        ok "mandelbrot_mpi_cuda ranks=$N (avg=${TIME} ms)"
    else
        fail "mandelbrot_mpi_cuda ranks=$N — see /tmp/mpicuda_$N.log"
    fi
done

# -----------------------------------------------------------
hdr "5. Visual equivalence: CPU vs GPU MPI (800x600 shallow)"
# -----------------------------------------------------------
mpirun -np 1 ./build/mandelbrot_mpi      --shallow --width 800 --height 600 --out /tmp/cpu.ppm --runs 1 > /dev/null 2>&1
mpirun -np 1 ./build/mandelbrot_mpi_cuda --shallow --width 800 --height 600 --out /tmp/gpu.ppm --runs 1 > /dev/null 2>&1
DIFF_OUT=$(python3 scripts/compare_ppm.py /tmp/cpu.ppm /tmp/gpu.ppm)
echo "$DIFF_OUT" | sed 's/^/  /'
SUMDIFF=$(echo "$DIFF_OUT" | awk '/sum-abs-diff/{print $2}')
MAXDIFF=$(echo "$DIFF_OUT" | awk '/max-byte-diff/{print $2}')
# Tolerance: a few hundred ULP-level bytes is fine; max per-byte must be tiny.
if [ "${SUMDIFF:-99999999}" -lt 1000 ] && [ "${MAXDIFF:-99}" -le 2 ]; then
    ok "CPU/GPU output equivalent within ULP tolerance (sum=$SUMDIFF max=$MAXDIFF)"
else
    fail "CPU/GPU output diverges (sum=$SUMDIFF max=$MAXDIFF)"
fi

# -----------------------------------------------------------
hdr "6. MPI determinism: 1 rank vs 4 ranks (cyclic) bit-exact"
# -----------------------------------------------------------
mpirun -np 1 ./build/mandelbrot_mpi --shallow --width 800 --height 600 --out /tmp/r1.ppm --runs 1 > /dev/null 2>&1
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --shallow --width 800 --height 600 --out /tmp/r4.ppm --chunk 8 --runs 1 > /dev/null 2>&1
SUMDIFF=$(python3 scripts/compare_ppm.py /tmp/r1.ppm /tmp/r4.ppm | awk '/sum-abs-diff/{print $2}')
if [ "$SUMDIFF" = "0" ]; then
    ok "1-rank vs 4-rank cyclic bit-exact (sum-abs-diff=0)"
else
    fail "1-rank vs 4-rank cyclic NOT bit-exact (sum-abs-diff=$SUMDIFF)"
fi

# -----------------------------------------------------------
hdr "7. MPI determinism: contiguous vs cyclic same final image"
# -----------------------------------------------------------
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --shallow --width 400 --height 300 --out /tmp/cont.ppm --contiguous --runs 1 > /dev/null 2>&1
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --shallow --width 400 --height 300 --out /tmp/cyc.ppm  --chunk 8 --runs 1     > /dev/null 2>&1
SUMDIFF=$(python3 scripts/compare_ppm.py /tmp/cont.ppm /tmp/cyc.ppm | awk '/sum-abs-diff/{print $2}')
if [ "$SUMDIFF" = "0" ]; then
    ok "contiguous vs cyclic produce identical image (sum-abs-diff=0)"
else
    fail "contiguous vs cyclic differ (sum-abs-diff=$SUMDIFF)"
fi

# -----------------------------------------------------------
hdr "8. Strong scaling shows speedup (1->4 ranks at 1920x1080 shallow)"
# -----------------------------------------------------------
T1=$(mpirun -np 1               ./build/mandelbrot_mpi --shallow --width 1920 --height 1080 --no-output --runs 2 \
     | awk '/^avg total/{print $3}')
T4=$(mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --shallow --width 1920 --height 1080 --no-output --runs 2 \
     | awk '/^avg total/{print $3}')
SPEEDUP=$(python3 -c "print(f'{$T1/$T4:.2f}')")
echo "  T(1)=${T1} ms  T(4)=${T4} ms  speedup=${SPEEDUP}x"
# Speedup must be > 2x to call this a real win on 4 ranks.
if python3 -c "import sys; sys.exit(0 if $T1/$T4 > 2.0 else 1)"; then
    ok "4-rank speedup ${SPEEDUP}x > 2.0x"
else
    fail "4-rank speedup ${SPEEDUP}x not credible"
fi

# -----------------------------------------------------------
hdr "9. Block-cyclic improves load balance vs contiguous (deep, 4 ranks)"
# -----------------------------------------------------------
RATIO_CONT=$(mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --deep \
             --width 800 --height 600 --no-output --runs 2 --contiguous \
             | awk '/imbalance ratio/{print $4}' | tr -d 'x' | tail -1)
RATIO_CYC=$(mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --deep \
             --width 800 --height 600 --no-output --runs 2 --chunk 8 \
             | awk '/imbalance ratio/{print $4}' | tr -d 'x' | tail -1)
echo "  contiguous imbalance: ${RATIO_CONT}x"
echo "  cyclic     imbalance: ${RATIO_CYC}x"
if python3 -c "import sys; sys.exit(0 if $RATIO_CYC < $RATIO_CONT else 1)"; then
    ok "cyclic (${RATIO_CYC}x) better balanced than contiguous (${RATIO_CONT}x)"
else
    fail "cyclic not better than contiguous"
fi

# -----------------------------------------------------------
hdr "Summary"
# -----------------------------------------------------------
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo
echo "passed: $PASS    failed: $FAIL"
[ "$FAIL" -eq 0 ]
