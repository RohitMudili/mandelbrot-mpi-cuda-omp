# Part A — Shared Core + MPI Track

## Owner deliverables

This part owns:

1. **`include/mandelbrot_core.hpp`** — the shared per-pixel algorithm header that every other implementation (serial, OpenMP, CUDA, MPI) consumes. Whoever builds Part A creates this file first; Part B blocks on it.
2. **Pure MPI implementation** — distributed rendering across processes.
3. **Hybrid MPI + CUDA implementation** — bonus deliverable that reuses the existing CUDA kernel.
4. **MPI scaling study** — strong + weak scaling, plus a load-imbalance comparison.
5. **Part A mini-report** — design rationale, plots, results.

This part can be developed and graded **independently** of Part B. The only handoff to Part B is `mandelbrot_core.hpp`.

---

## A.0 — Prerequisites (already done)

✅ WSL2 Ubuntu 24.04
✅ CUDA Toolkit 12.0 (`nvcc`)
✅ CMake 3.28
✅ SDL2 dev libs
✅ OpenMPI (`mpicc`, `mpirun` — verify with `which mpirun`)
✅ RTX 3050 Ti visible inside WSL

---

## A.1 — Shared algorithm core (`include/mandelbrot_core.hpp`)

**This is the foundation everyone else builds on. Get this right and commit it before starting anything else.**

### Requirements

- Single header, inline / `__host__ __device__` annotated so it compiles in both regular C++ and CUDA translation units.
- Implements the **exact same** per-pixel algorithm currently spread across `mandelbrot.cu` so that all four implementations produce visually identical images.
- `double` precision throughout.
- Escape radius² = 1e20 (for smooth coloring).
- Returns enough state for color computation: `iter`, final `zx, zy, dzx, dzy`.

### File contents (skeleton)

```cpp
// include/mandelbrot_core.hpp
#ifndef MANDELBROT_CORE_HPP
#define MANDELBROT_CORE_HPP

#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
  #define MB_HD __host__ __device__
#else
  #define MB_HD
  #ifndef M_PI
    #define M_PI 3.14159265358979323846
  #endif
#endif

struct MandelResult {
    int iter;
    double zx, zy;
    double dzx, dzy;
};

MB_HD inline MandelResult mandelbrotIterate(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    double dzx = 1.0, dzy = 0.0;
    int iter = 0;
    const double escapeRadius2 = 1e20;

    while (zx*zx + zy*zy < escapeRadius2 && iter < maxIter) {
        double dzx_new = 2.0 * (zx * dzx - zy * dzy) + 1.0;
        double dzy_new = 2.0 * (zx * dzy + zy * dzx);
        dzx = dzx_new; dzy = dzy_new;
        double xtemp = zx*zx - zy*zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xtemp;
        iter++;
    }
    return { iter, zx, zy, dzx, dzy };
}

MB_HD inline float smoothIterCount(double zx, double zy, int iter, int maxIter) {
    if (iter == maxIter) return 0.0f;
    const double escapeRadius = 1e10;
    const double logEscapeRadius = log(escapeRadius);
    double modz = sqrt(zx * zx + zy * zy);
    double logRatio = 2.0 * log(modz) / logEscapeRadius;
    float smoothVal = 1.0f - (float)(log(logRatio) / log(2.0));
    return (float)iter + smoothVal;
}

// Map a pixel (x, y) to complex plane c.
MB_HD inline void pixelToComplex(int x, int y, int width, int height,
                                 double centerX, double centerY, double zoom,
                                 double& cx, double& cy) {
    cx = centerX + ((x - width/2.0) / (width/2.0)) * (1.5 / zoom);
    cy = centerY + ((y - height/2.0) / (height/2.0)) * (1.0 / zoom);
}

// Distance estimate for shading falloff.
MB_HD inline float distanceEstimate(const MandelResult& r) {
    double modz  = sqrt(r.zx*r.zx + r.zy*r.zy);
    double moddz = sqrt(r.dzx*r.dzx + r.dzy*r.dzy);
    return (float)(modz * log(modz) / moddz / 2.0);
}

#endif // MANDELBROT_CORE_HPP
```

### Refactor existing CUDA kernel

`src/mandelbrot.cu` currently inlines the iteration loop, smoothing, etc. Replace those bodies with calls into the new header so the GPU and CPU sides cannot drift apart. Keep `getColor()` and `blinnPhong()` as `__device__`-only helpers in `mandelbrot.cu` (they're GPU-specific to the rendering pipeline; do not need to be shared).

### Validation gate

Before merging this header, confirm: build the existing `benchmark` and `mandelbrot` viewer using the refactored kernel, compare the rendered output against a saved reference PNG from the current code, SSD must be near zero (allow tiny rounding drift). Only then unblock Part B.

---

## A.2 — Pure MPI implementation (`src/mandelbrot_mpi.cpp`)

### Process model

- Rank 0 (master): owns view parameters, gathers final image, optionally writes a PPM/PNG.
- All ranks (including 0) compute their assigned subset of pixels.
- No idle "manager-only" rank — every rank does compute work.

### Work distribution: block-cyclic rows

Naive contiguous row blocks are bad: the dense interior of the Mandelbrot set is a horizontal band, so contiguous chunking puts most of the expensive pixels on a few unlucky ranks while others finish early.

**Use block-cyclic rows with chunk size 8 or 16:**

```cpp
// Rank r owns rows where (row / chunk) % numRanks == r
const int CHUNK = 8;
for (int row = 0; row < height; row++) {
    int chunkIdx = row / CHUNK;
    int ownerRank = chunkIdx % numRanks;
    if (ownerRank == myRank) {
        // compute this row
    }
}
```

This interleaves "interior" rows across all ranks while keeping cache locality within a chunk.

### Communication

- Each rank packs its computed rows into a contiguous local buffer (only its own rows, in row-major order with row index metadata).
- Use `MPI_Gatherv` to assemble on rank 0. Compute `recvcounts[]` and `displs[]` from the cyclic mapping.
- Alternative: each rank writes into a sparse global-sized buffer and `MPI_Reduce(MPI_BOR)` it — simpler but doubles memory and bandwidth. Prefer Gatherv.

### Timing

- Time only the compute + gather phase. Exclude `MPI_Init`, `MPI_Finalize`, image write.
- Use `MPI_Barrier` + `MPI_Wtime` on rank 0 around the timed section.

### CMake

```cmake
find_package(MPI REQUIRED)
add_executable(mandelbrot_mpi src/mandelbrot_mpi.cpp)
target_link_libraries(mandelbrot_mpi PRIVATE MPI::MPI_CXX)
target_include_directories(mandelbrot_mpi PRIVATE ${PROJECT_SOURCE_DIR}/include)
```

### Run command

```bash
mpirun -np 4 ./build/mandelbrot_mpi
# or oversubscribe on a single node:
mpirun --oversubscribe -np 8 ./build/mandelbrot_mpi
```

---

## A.3 — Hybrid MPI + CUDA (`src/mandelbrot_mpi_cuda.cpp`)

### Design

- Each MPI rank claims a GPU via `cudaSetDevice(rank % deviceCount)`.
- Each rank computes its assigned **row range** (contiguous block this time, since the GPU internally parallelizes — load imbalance dissolves at the kernel level) by calling a slightly modified `computeMandelbrotCUDA` that accepts `(yStart, yEnd)`.
- Gather to rank 0 with `MPI_Gatherv`.

### Modification needed in `mandelbrot.cu`

Add an overload that renders only rows `[yStart, yEnd)`:

```cpp
void computeMandelbrotCUDARows(uint8_t* pixels,
                               int width, int yStart, int yEnd, int fullHeight,
                               double centerX, double centerY, double zoom,
                               int maxIter, ColorTable* colorTable);
```

The kernel skips threads with `y < yStart || y >= yEnd`.

### Single-GPU fallback

On a single-GPU laptop (your case — only one RTX 3050 Ti), all ranks share one device. CUDA serializes their kernel launches automatically. The hybrid still demonstrates correctness and the rank-distribution code; the *parallel* speedup over single-rank single-GPU will be small or negative (overhead dominates) — call this out honestly in the report. The design is what matters; multi-GPU clusters would show real gains.

### CMake

```cmake
add_executable(mandelbrot_mpi_cuda
    src/mandelbrot_mpi_cuda.cpp
    src/mandelbrot.cu
)
target_link_libraries(mandelbrot_mpi_cuda PRIVATE MPI::MPI_CXX)
set_target_properties(mandelbrot_mpi_cuda PROPERTIES CUDA_ARCHITECTURES "75;86")
target_include_directories(mandelbrot_mpi_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
```

---

## A.4 — MPI-aware benchmark (`src/benchmark_mpi.cpp`)

Separate executable launched via `mpirun`. Each rank participates; rank 0 prints results.

- Runs N=5 trials per configuration.
- Sweeps over rank counts via outer shell loop (one `mpirun` invocation per rank count), not internally.
- Emits one CSV row per (impl, ranks, run) tuple.
- Configurations to measure:
  - Pure MPI: 1, 2, 4, 8 ranks
  - Hybrid MPI+CUDA: 1, 2, 4 ranks (since single GPU)

CSV schema (shared with Part B):

```
implementation,workers,run,time_ms,mpixels_per_sec
mpi,1,1,623,3.33
mpi,1,2,619,3.35
...
mpi_cuda,1,1,82,25.3
...
```

---

## A.5 — Strong + weak scaling (MPI)

### Strong scaling
- Fixed 1920×1080, deep-zoom view: `centerX = -0.743643887037151`, `centerY = 0.131825904205330`, `zoom = 1e6`, `maxIter = 2000`.
- Sweep ranks ∈ {1, 2, 4, 8} (oversubscribe past physical cores).
- Plot speedup vs ranks, with ideal `y = x` reference line.

### Weak scaling
- Pixels-per-rank held constant.
- 1 rank → 960×540, 2 → 1357×764 (≈), 4 → 1920×1080, 8 → 2715×1527, 16 → 3840×2160.
- Plot efficiency (T_1 / T_N) vs ranks (ideal flat at 1.0).

### Load-imbalance comparison (the meaty story for the report)
- Run pure MPI with **contiguous** row blocks at 4 ranks → record per-rank elapsed time → max/min ratio.
- Run pure MPI with **block-cyclic** row blocks at 4 ranks → record per-rank elapsed time → max/min ratio.
- Show the contiguous version has e.g. 3× imbalance while cyclic has <1.2×. This is the design justification.

### Scripts

```
scripts/
├── run_mpi_strong.sh     # mpirun loop over rank counts
├── run_mpi_weak.sh       # mpirun loop over (ranks, resolution) pairs
├── run_imbalance.sh      # compares contiguous vs cyclic
└── plot_mpi_results.py   # pandas + matplotlib → PNGs in docs/results/
```

---

## A.6 — Part A mini-report (`docs/report_partA.md` → PDF)

Sections:

1. **Shared algorithm core** — what's in `mandelbrot_core.hpp`, why a single header, validation that all impls produce identical images.
2. **MPI design**
   - Process model (no dedicated master)
   - Block-cyclic row distribution and why
   - `MPI_Gatherv` choice over `MPI_Reduce(MPI_BOR)`
3. **Hybrid MPI + CUDA** — design, single-GPU caveat, what multi-GPU would show.
4. **Experimental setup** — RTX 3050 Ti, WSL2 Ubuntu 24.04, OpenMPI version, view params.
5. **Results**
   - Strong scaling table + plot
   - Weak scaling table + plot
   - Load-imbalance comparison table (contiguous vs cyclic)
6. **Discussion**
   - When MPI helps vs when overhead dominates
   - Why hybrid MPI+CUDA shines on multi-node clusters but not on a single laptop GPU
7. **Conclusion**

Keep it ~3 pages.

---

## A.7 — Definition of Done (Part A only)

- [ ] `include/mandelbrot_core.hpp` exists, is consumed by the existing CUDA kernel, refactored kernel still produces visually identical output.
- [ ] `mandelbrot_mpi` builds and runs with `mpirun -np N`.
- [ ] `mandelbrot_mpi_cuda` builds and runs.
- [ ] `benchmark_mpi` produces a CSV.
- [ ] At least one strong-scaling plot and one weak-scaling plot in `docs/results/`.
- [ ] Load-imbalance comparison numbers exist (contiguous vs cyclic).
- [ ] `docs/report_partA.pdf` exists and covers sections 1–7 above.

---

## A.8 — Suggested order

| Step | Task | Effort |
|---|---|---|
| 1 | A.1 — write `mandelbrot_core.hpp`, refactor `mandelbrot.cu` to use it, verify output unchanged | ~2h |
| 2 | A.2 — pure MPI implementation | ~3h |
| 3 | A.4 — MPI-aware benchmark | ~1h |
| 4 | A.3 — hybrid MPI + CUDA | ~2h |
| 5 | A.5 — scaling runs + plots | ~2h |
| 6 | A.6 — write report | ~2h |

**Estimated total: ~12h.**

**Score impact in isolation: 4.5 → ~7.5 / 10.**
**Combined with Part B: → 9–10 / 10.**

---

## A.9 — Risks specific to Part A

| Risk | Mitigation |
|---|---|
| MPI on WSL2 has quirks with process binding | Use `mpirun --bind-to none --oversubscribe` if you see weird CPU contention |
| Single GPU shared by N ranks just serializes (no speedup) | Document this as expected; the *design* is the deliverable, not the speedup. Consider running pure MPI (no CUDA) on more ranks for a better speedup story. |
| `MPI_Gatherv` displacements get tricky with cyclic distribution | Have each rank send rows tagged with their global row index; rank 0 scatters them into the output buffer post-gather. Or use `MPI_Type_create_indexed_block` for a cleaner gather. |
| Floating-point output differs slightly from CUDA | Compare with SSD threshold, not bitwise — `double` on CPU vs `double` in PTX can differ in last few ULPs. |
