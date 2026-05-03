# Part B — OpenMP + Polish + Unified Benchmark

## Owner deliverables

This part owns:

1. **Serial reference implementation** — single-threaded CPU baseline using the shared core.
2. **OpenMP implementation** — parallel CPU using `#pragma omp parallel for`.
3. **Unified benchmark** — fair comparison of Serial vs OpenMP vs CUDA on identical algorithm + view.
4. **OpenMP scaling study** — strong scaling with thread-count sweep, dynamic-vs-static schedule comparison.
5. **Code-quality polish** — fix the bugs flagged in the code review, add color-table caching on the GPU.
6. **Part B mini-report** — design rationale, plots, results.

This part **depends on Part A's `include/mandelbrot_core.hpp`**. Once that header is committed, Part B can proceed completely independently of Part A's MPI work.

---

## B.0 — Prerequisites

✅ All Part A.0 prerequisites (toolchain already installed in WSL).
⏳ **Part A.1 must be merged first** — `include/mandelbrot_core.hpp` exists and the existing CUDA build still works against it.

If Part A is delayed, Part B can build a **temporary** local copy of the core header to unblock itself, then delete it and switch to Part A's version when ready. This avoids serial blocking.

---

## B.1 — Serial reference (`src/mandelbrot_serial.cpp`)

### Purpose

This is the **baseline T_serial** for every speedup number reported anywhere in the project. It must be correct, fair, and use the shared core.

### Structure

```cpp
#include "mandelbrot_core.hpp"

void renderSerial(uint8_t* pixels, int width, int height,
                  double centerX, double centerY, double zoom, int maxIter,
                  ColorTable* colorTable) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double cx, cy;
            pixelToComplex(x, y, width, height, centerX, centerY, zoom, cx, cy);
            MandelResult r = mandelbrotIterate(cx, cy, maxIter);
            // ... color computation matching the CUDA path ...
            int idx = (y * width + x) * 4;
            pixels[idx + 0] = R;
            pixels[idx + 1] = G;
            pixels[idx + 2] = B;
            pixels[idx + 3] = 255;
        }
    }
}
```

### Color path

Replicate the CUDA `getColor()` logic on the CPU — Blinn-Phong, overlay blend, DEM sigmoid. Either:
- (a) Lift those functions into `mandelbrot_core.hpp` as `MB_HD inline` (cleanest), or
- (b) Duplicate them into `mandelbrot_serial.cpp` (fastest, but risks drift).

**Prefer (a).** It also helps the OpenMP version. Add the `getColor()`, `blinnPhong()`, `overlay()` helpers from `mandelbrot.cu` to the shared header.

### Standalone executable

Build `mandelbrot_serial` as a standalone executable that renders a single image and writes a PPM file (no SDL dependency — easier to validate on a headless machine):

```cpp
int main() {
    // hard-coded view, write output.ppm
}
```

---

## B.2 — OpenMP implementation (`src/mandelbrot_omp.cpp`)

### The actual change

Same code as the serial version, with one annotation:

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
        // ... identical body ...
    }
}
```

### Why `schedule(dynamic, 16)` and not `static`

The Mandelbrot interior is a horizontal band of high-iteration pixels. With `schedule(static)`, threads handling rows that pass through the interior take far longer than threads handling top/bottom rows that escape quickly. Result: 8 threads run as fast as the slowest thread.

`schedule(dynamic, 16)` hands out 16-pixel chunks on demand. Slow chunks just delay one thread; fast threads pick up the next available chunk. Chunk size 16 balances scheduling overhead vs granularity.

**Document this in the report with measurements** — see B.5.

### CMake

```cmake
find_package(OpenMP REQUIRED)
add_executable(mandelbrot_omp src/mandelbrot_omp.cpp)
target_link_libraries(mandelbrot_omp PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(mandelbrot_omp PRIVATE ${PROJECT_SOURCE_DIR}/include)
```

### Run

```bash
OMP_NUM_THREADS=8 ./build/mandelbrot_omp
```

---

## B.3 — Unified benchmark (rewrite `src/benchmark.cpp`)

The current `benchmark.cpp` compares an **unfair** simplified CPU to the full GPU pipeline. Rewrite it.

### Requirements

- All paths (Serial, OpenMP, CUDA) compute the **exact same algorithm** from `mandelbrot_core.hpp`.
- Same view, same iter cap, same resolution.
- After computing, do a quick visual-equivalence check: SSD between Serial and CUDA outputs. Print "OK" if below threshold, "DRIFT" otherwise.

### Configurations measured

- Serial (1 thread)
- OpenMP, threads = 1, 2, 4, 8 (and physical core count if higher)
- CUDA

### Metrics per configuration

- N = 5 runs
- Wall time avg / min / max
- Speedup vs serial baseline
- Parallel efficiency (`speedup / num_workers`)
- Throughput (MPixels/sec)

### Output

- Pretty-print to stdout (like the current benchmark does).
- Append CSV row to `docs/results/benchmark.csv` with schema:
  ```
  implementation,workers,run,time_ms,mpixels_per_sec
  ```
  (Same schema Part A uses for `benchmark_mpi`. The two CSVs combine into one master plot.)

### Headless mode

Add a `--no-output` flag that skips PPM writing — used for tight benchmark loops.

---

## B.4 — Strong scaling study (OpenMP)

### Setup

- Fixed 1920×1080.
- Deep-zoom view: `centerX = -0.743643887037151`, `centerY = 0.131825904205330`, `zoom = 1e6`, `maxIter = 2000`.
- Sweep `OMP_NUM_THREADS` ∈ {1, 2, 4, 8, 16} (16 if hyperthreaded).

### Plot

- X-axis: thread count
- Y-axis: speedup vs 1-thread baseline
- Reference line: ideal `y = x`
- Save to `docs/results/openmp_strong_scaling.png`

### Expected shape

Near-ideal up to physical core count, sub-linear past that (hyperthreading gives ~30% extra, not 100%), then flat or slightly worse beyond hardware threads.

---

## B.5 — Schedule comparison study

This is a small but high-value study for the report — it justifies the design choice.

### Configurations

At 8 threads, on the deep-zoom view, measure:

| Schedule | Notes |
|---|---|
| `static` | default contiguous chunks |
| `static, 16` | smaller static chunks |
| `dynamic, 1` | maximum scheduling overhead |
| `dynamic, 16` | the chosen default |
| `dynamic, 64` | larger chunks |
| `guided` | exponentially decreasing chunks |

### Output

A small table in the report showing wall time for each. Expected winner: `dynamic, 16` or `guided`. The `static` numbers should be noticeably worse — that's the point.

---

## B.6 — Code polish (small but worth mentioning in report)

These are the items flagged in the original code review.

### B.6.1 — Fix `int main` returning `false`
File: `src/main.cpp:58`
Change `return false;` (which is `0` in `int`-context — actually wrong, since the renderer-create branch is the **failure** branch) to `return 1;`.

Same fix at `main.cpp:71`.

### B.6.2 — Cache the color table on the GPU
File: `src/mandelbrot.cu`, function `computeMandelbrotCUDA`.

Current code re-uploads the color table to the GPU on every call. For interactive rendering at 60 FPS that's 60 redundant uploads per second.

Add a static cache:

```cpp
static float* d_colorTableCached = nullptr;
static int    d_colorTableNumColors = 0;
static const ColorTable* lastTable = nullptr;

if (colorTable != lastTable) {
    if (d_colorTableCached) cudaFree(d_colorTableCached);
    cudaMalloc(&d_colorTableCached, colorTable->numColors * 3 * sizeof(float));
    cudaMemcpy(d_colorTableCached, colorTable->colors,
               colorTable->numColors * 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    lastTable = colorTable;
    d_colorTableNumColors = colorTable->numColors;
}
```

Add a `freeMandelbrotCudaResources()` cleanup function called from `main()` before exit.

Measure interactive viewer FPS before vs after; report the difference.

### B.6.3 — Optional: cache the device pixel buffer too
Same idea — the pixel buffer is reallocated every frame. Keep a static buffer sized to the largest request. Smaller win than the color-table fix but easy.

### B.6.4 — `.gitignore`
Make sure `build/` is ignored, `docs/results/*.csv` is tracked.

---

## B.7 — Part B mini-report (`docs/report_partB.md` → PDF)

Sections:

1. **Serial reference** — what it does, why it's the fair baseline.
2. **OpenMP design**
   - Pragma used and why
   - Schedule choice with measured comparison (B.5)
3. **Unified benchmark methodology** — same algorithm, same view, SSD validation.
4. **Experimental setup** — CPU model, core/thread count, OS, compiler, OpenMP version.
5. **Results**
   - Serial vs OpenMP vs CUDA table
   - Strong scaling plot
   - Schedule comparison table
6. **Discussion**
   - Where OpenMP catches up to CUDA (small workloads, deep zoom variance)
   - Why CUDA pulls ahead at high arithmetic intensity
   - Hyperthreading drop-off
7. **Code polish notes** — fixes from B.6, FPS impact of color-table caching.
8. **Conclusion**

Keep it ~3 pages.

---

## B.8 — Definition of Done (Part B only)

- [ ] `mandelbrot_serial` builds and produces a PPM matching CUDA output (SSD < threshold).
- [ ] `mandelbrot_omp` builds, scales with `OMP_NUM_THREADS`.
- [ ] `benchmark` rewritten — all paths use shared core, fair comparison, CSV emitted.
- [ ] OpenMP strong-scaling plot in `docs/results/`.
- [ ] Schedule comparison table in the report.
- [ ] `main.cpp:58` and `:71` return-type bugs fixed.
- [ ] Color-table caching in `computeMandelbrotCUDA`, FPS improvement measured.
- [ ] `docs/report_partB.pdf` exists and covers sections 1–8 above.

---

## B.9 — Suggested order

| Step | Task | Effort |
|---|---|---|
| 1 | Wait for Part A.1 (`mandelbrot_core.hpp`) — or build a stub locally | ~30m |
| 2 | B.1 — serial reference + standalone executable | ~1.5h |
| 3 | B.2 — OpenMP version | ~30m |
| 4 | B.3 — rewrite benchmark, validate SSD | ~2h |
| 5 | B.4 — strong-scaling runs + plot | ~1h |
| 6 | B.5 — schedule comparison study | ~1h |
| 7 | B.6 — code polish (return-bug fix, color-table cache) | ~1h |
| 8 | B.7 — write report | ~2h |

**Estimated total: ~9.5h.**

**Score impact in isolation (Part B + existing CUDA, without MPI): 4.5 → ~7 / 10.**
**Combined with Part A: → 9–10 / 10.**

---

## B.10 — Risks specific to Part B

| Risk | Mitigation |
|---|---|
| `mandelbrot_core.hpp` doesn't yet exist when Part B starts | Build a temporary stub locally; switch to Part A's version on merge. Don't let this block you. |
| Color computation drifts between CPU and CUDA paths | After moving `getColor()` etc. into the shared header, validate by SSD against a saved reference image from the original code. |
| WSL CPU thread count mis-detected | Set `OMP_NUM_THREADS` explicitly; don't rely on `omp_get_max_threads()` defaults. Verify with `nproc` and `lscpu`. |
| Benchmarks vary run-to-run because WSL is sharing CPU with Windows | Run with N=5+ trials, report min as well as mean, run with the laptop on AC power and other apps closed. |
| OpenMP at thread count > physical cores looks bad | Expected. Hyperthreaded scaling tops out around physical-core count + 30%. Document, don't try to "fix". |

---

## B.11 — Handoff notes to whoever does Part A

The only thing Part B asks of Part A is:

1. `include/mandelbrot_core.hpp` exists.
2. It exposes (at minimum):
   - `mandelbrotIterate(cx, cy, maxIter) -> MandelResult`
   - `pixelToComplex(...)`
   - `smoothIterCount(...)`
   - `distanceEstimate(...)`
   - The lighting + color helpers (`blinnPhong`, `overlay`, `getColor`) — moved from `mandelbrot.cu` into the shared header so the CPU path uses identical code.
3. The existing CUDA kernel still produces visually identical output after refactoring — verified by SSD against a saved reference PNG.

If those three boxes are checked, Part B can proceed with zero further coordination.
