# Mandelbrot Parallelization Plan — OpenMP + CUDA + MPI

## 0. Context

**Assignment requirement:** select a problem statement and parallelize it using **OpenMP, CUDA, and MPI**.

**Current state of the project:** A CUDA-accelerated Mandelbrot renderer with an SDL2 interactive viewer and a CPU-vs-GPU benchmark.

**Current score: 4.5 / 10**

| Aspect | Status |
|---|---|
| CUDA implementation | Done (kernel with smooth iter, DEM, Blinn-Phong shading) |
| OpenMP implementation | **Missing** — `computeMandelbrotCPU` in `benchmark.cpp` is single-threaded |
| MPI implementation | **Missing entirely** |
| Fair benchmark across paradigms | Missing — CPU baseline uses different precision/escape radius/coloring than GPU |
| Scaling study (strong/weak) | Missing |
| Report / writeup | Missing |

The Mandelbrot set is embarrassingly parallel, so it cleanly fits all three paradigms. The CUDA half is genuine and non-trivial; the gap is the other two paradigms and a fair, unified evaluation.

---

## 1. Goals

1. Add a real **OpenMP** CPU implementation.
2. Add a real **MPI** implementation, ideally with a hybrid **MPI+CUDA** or **MPI+OpenMP** mode.
3. Rewrite the benchmark so all four implementations (serial, OpenMP, CUDA, MPI) compute the **same algorithm** and can be compared fairly.
4. Produce a **scaling study** (strong + weak scaling) with plots.
5. Write a short **report** explaining design choices, load balancing, and results.

---

## 2. Deliverable Layout

```
mandelbrot-cuda-main/
├── CMakeLists.txt                 (updated: OpenMP, MPI targets)
├── README.md                      (updated: build/run instructions for all targets)
├── plan.md                        (this file)
├── include/
│   └── mandelbrot.cuh             (existing; may add shared algorithm header)
├── src/
│   ├── mandelbrot.cu              (existing CUDA kernel)
│   ├── mandelbrot_serial.cpp      (NEW — reference serial implementation, matches CUDA algorithm)
│   ├── mandelbrot_omp.cpp         (NEW — OpenMP parallel CPU)
│   ├── mandelbrot_mpi.cpp         (NEW — MPI distributed; can call CUDA or OpenMP per rank)
│   ├── benchmark.cpp              (REWRITTEN — unified fair benchmark)
│   └── main.cpp                   (existing SDL viewer; minor fixes)
├── scripts/
│   ├── run_strong_scaling.sh      (NEW)
│   ├── run_weak_scaling.sh        (NEW)
│   └── plot_results.py            (NEW — generates speedup/efficiency plots)
└── docs/
    ├── report.pdf                 (NEW — final writeup)
    └── results/                   (CSV outputs + PNG plots)
```

---

## 3. Work Breakdown

### Phase 1 — Make the comparison fair (foundation)

**Task 1.1 — Extract a shared reference algorithm**
- Create `include/mandelbrot_core.hpp` with an inline/templated function implementing the *same* iteration as the CUDA kernel:
  - `double` precision
  - escape radius² = `1e20`
  - derivative tracking for DEM
  - smooth iteration count, Blinn-Phong, overlay blend, sigmoid DEM softening
- Goal: serial / OpenMP / MPI / CUDA all produce visually identical images for the same view.

**Task 1.2 — Serial reference (`mandelbrot_serial.cpp`)**
- Single-threaded reference using the shared core.
- This is the baseline for *all* speedup numbers.

**Task 1.3 — Pick an evaluation view**
- Use a deep-zoom view where iteration cost dominates and parallel speedup is visible:
  - `centerX = -0.743643887037151`
  - `centerY =  0.131825904205330`
  - `zoom   = 1e6` (and a second view at `1e3` for comparison)
  - `maxIter = 2000`
- Standard resolution: 1920×1080 for benchmark; 3840×2160 for weak-scaling top end.

---

### Phase 2 — OpenMP implementation

**Task 2.1 — `mandelbrot_omp.cpp`**
- Wrap the per-pixel loop with:
  ```cpp
  #pragma omp parallel for collapse(2) schedule(dynamic, 16)
  ```
- **Why dynamic schedule:** the Mandelbrot interior is a horizontal band of high-iteration pixels; static scheduling would leave threads near the top/bottom idle while interior threads work much longer. Dynamic with chunk=16 balances load without excessive scheduling overhead.

**Task 2.2 — CMake integration**
```cmake
find_package(OpenMP REQUIRED)
add_executable(mandelbrot_omp src/mandelbrot_omp.cpp)
target_link_libraries(mandelbrot_omp PRIVATE OpenMP::OpenMP_CXX)
```

**Task 2.3 — Thread-count sweep**
- Run with `OMP_NUM_THREADS = 1, 2, 4, 8, 16, …` up to physical core count.
- Record wall time per thread count → strong-scaling plot.

---

### Phase 3 — MPI implementation

**Task 3.1 — `mandelbrot_mpi.cpp` — process model**
- Rank 0 (master): owns view parameters, gathers final image, optionally writes PNG/PPM.
- Ranks 0..N-1 (workers, including rank 0): each computes an assigned subset of pixels.

**Task 3.2 — Work distribution: block-cyclic rows**
- Naive contiguous row blocks → severe load imbalance (interior band lands on one rank).
- **Block-cyclic by rows:** rank `r` owns rows where `(row / chunk) % numRanks == r`, with `chunk = 8` or `16`.
- Each rank computes only its rows, packs them into a local buffer.

**Task 3.3 — Communication**
- `MPI_Gatherv` with displacements computed from the cyclic mapping to assemble the full image on rank 0.
- Alternatively: `MPI_Reduce` is unnecessary — pixels are partitioned, not summed.
- Time only the compute + gather phase (exclude MPI_Init / image write).

**Task 3.4 — Hybrid modes (bonus, strongly recommended)**
- **MPI + OpenMP:** each rank uses `#pragma omp parallel for` internally; sweep ranks × threads.
- **MPI + CUDA:** each rank claims a GPU via `cudaSetDevice(rank % deviceCount)` and computes its rows by calling the existing `computeMandelbrotCUDA` on its subregion. This reuses the CUDA kernel cleanly and is the most impressive demonstration.

**Task 3.5 — CMake integration**
```cmake
find_package(MPI REQUIRED)
add_executable(mandelbrot_mpi src/mandelbrot_mpi.cpp src/mandelbrot.cu)
target_link_libraries(mandelbrot_mpi PRIVATE MPI::MPI_CXX)
set_target_properties(mandelbrot_mpi PROPERTIES CUDA_ARCHITECTURES "75;86")
```

**Task 3.6 — Run modes**
- Single-node oversubscription: `mpirun -np 4 ./mandelbrot_mpi`.
- Multi-node (if cluster available): hostfile + `mpirun --hostfile hosts -np 8`.

---

### Phase 4 — Unified Benchmark

**Task 4.1 — Rewrite `benchmark.cpp`**
- All four paths use the *same* core algorithm from Task 1.1.
- Same view parameters, same iteration cap, same resolution.
- Verify visual equivalence (e.g. compute SSD between serial and CUDA outputs; should be ~0 modulo float vs double rounding).

**Task 4.2 — Metrics reported**
For each implementation and each configuration:
- Wall time (avg, min, max over N=5 runs)
- Speedup vs serial baseline: `T_serial / T_parallel`
- Parallel efficiency: `speedup / num_workers`
- Throughput: MPixels/sec
- Output as CSV → `docs/results/benchmark.csv`.

---

### Phase 5 — Scaling Study

**Task 5.1 — Strong scaling**
- Fixed problem size (e.g. 1920×1080, 2000 iters).
- Vary worker count: OpenMP threads {1,2,4,8,16}; MPI ranks {1,2,4,8}.
- Plot: speedup vs workers, with ideal `y=x` reference line.
- Expect sub-linear at high counts (memory bandwidth, scheduling overhead).

**Task 5.2 — Weak scaling**
- Problem size grows with workers: pixels per worker held constant.
- E.g. 1 worker → 960×540; 4 workers → 1920×1080; 16 workers → 3840×2160.
- Plot: efficiency vs workers (ideal is flat at 1.0).

**Task 5.3 — Load imbalance analysis**
- Compare static vs dynamic OpenMP scheduling; show measurable difference at the chosen view.
- Compare contiguous vs block-cyclic MPI distribution; show why cyclic wins.

**Task 5.4 — Plot script**
- `scripts/plot_results.py` reads `benchmark.csv` with pandas/matplotlib, emits:
  - `strong_scaling.png`
  - `weak_scaling.png`
  - `efficiency.png`
  - `throughput_comparison.png`

---

### Phase 6 — Report

**Task 6.1 — `docs/report.pdf` sections**
1. Problem statement (Mandelbrot, why it's parallelization-friendly).
2. Algorithm (iteration, smooth coloring, DEM, lighting).
3. Three parallel implementations — design rationale for each:
   - OpenMP: dynamic scheduling and why.
   - CUDA: thread block layout (16×16), per-pixel independence.
   - MPI: block-cyclic row distribution and why.
4. Hybrid MPI+CUDA design.
5. Experimental setup (hardware, compiler, MPI distribution).
6. Results: tables + scaling plots.
7. Discussion: load balancing, GPU vs CPU crossover, communication overhead.
8. Conclusion + per-paradigm tradeoffs.

---

### Phase 7 — Minor cleanups (do alongside the above)

- `src/main.cpp:58` returns `false` from `int main` on renderer-create failure — change to `return 1;`.
- `computeMandelbrotCUDA` re-uploads the color table to the GPU on every call. Cache it in a static device pointer that's reallocated only when the table changes. Mention this optimization in the report.
- `.gitignore` already excludes `build/`; ensure `docs/results/*.csv` is tracked but build artifacts are not.

---

## 4. Implementation Order (suggested)

| Step | Task | Effort | Why this order |
|---|---|---|---|
| 1 | Phase 1 (shared core + serial) | ~2h | Everything else depends on a fair baseline |
| 2 | Phase 2 (OpenMP) | ~1h | Smallest delta; biggest immediate score lift |
| 3 | Phase 4 (rewrite benchmark) | ~2h | Validates correctness early |
| 4 | Phase 3.1–3.3 (pure MPI) | ~3h | Core MPI requirement |
| 5 | Phase 3.4 (hybrid MPI+CUDA) | ~2h | Bonus, but reuses existing CUDA |
| 6 | Phase 5 (scaling study) | ~2h | Run on whatever hardware you have |
| 7 | Phase 6 (report) | ~3h | Final writeup |
| 8 | Phase 7 (cleanups) | ~30m | Polish |

**Estimated total: ~15h of focused work.**

---

## 5. Expected Score Progression

| After completing | Expected score |
|---|---|
| Current state | 4.5 / 10 |
| + OpenMP + fair benchmark (Phase 1, 2, 4) | ~7 / 10 |
| + Pure MPI (Phase 3.1–3.3) | ~8.5 / 10 |
| + Hybrid MPI+CUDA + scaling study + report (Phase 3.4, 5, 6) | 9–10 / 10 |

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| No multi-node cluster available for MPI | Run multi-process on single node (`-np 4` with oversubscribe); still demonstrates partitioning + communication |
| Multi-GPU not available for hybrid MPI+CUDA | Have all ranks share one GPU via `cudaSetDevice(0)` and serialize kernel launches; or fall back to MPI+OpenMP only |
| MPI on Windows is awkward | Use Microsoft MPI (MS-MPI) or run that part in WSL2 |
| Floating-point output differs slightly across implementations | Compare with tolerance (e.g. SSD < threshold), not bitwise equality |
| Dynamic scheduling overhead dominates at small chunk sizes | Tune chunk size empirically (8–64) |

---

## 7. Definition of Done

- [ ] `mandelbrot_serial`, `mandelbrot_omp`, `mandelbrot_mpi`, `mandelbrot` (CUDA viewer), and `benchmark` all build from a single `cmake ..; make` invocation.
- [ ] All four compute paths produce visually equivalent images for the same view.
- [ ] `benchmark` runs all four and writes CSV results.
- [ ] At least one scaling plot (strong scaling) is produced from real measurements.
- [ ] `docs/report.pdf` exists and covers the sections in Phase 6.
- [ ] README updated with build + run instructions for every executable.
