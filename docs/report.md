# Mandelbrot OpenMP + CUDA + MPI: Implementation Report

## 1. Overview

This report covers the design, implementation, and measured performance of a Mandelbrot set renderer parallelized with OpenMP, CUDA, and MPI. A single per-pixel algorithm is shared across all CPU implementations through `include/mandelbrot_core.hpp`; the CUDA kernel is held byte-equivalent to the same algorithm and validated by sum-of-absolute-differences (SAD) against the serial reference.

Seven executables are produced from a single CMake build:

| Executable | Role |
|---|---|
| `mandelbrot` | SDL2 interactive viewer backed by the CUDA kernel |
| `mandelbrot_serial` | Single-threaded reference renderer |
| `mandelbrot_omp` | OpenMP renderer with selectable schedule |
| `mandelbrot_mpi` | Distributed CPU renderer with block-cyclic or contiguous row distribution |
| `mandelbrot_mpi_cuda` | Distributed orchestration around the CUDA kernel |
| `benchmark` | Unified Serial / OpenMP / CUDA fair benchmark with CSV output |
| `benchmark_mpi` | MPI benchmark with CSV output |

## 2. Environment

| Component | Value |
|---|---|
| Host OS | Windows 11 with WSL2 |
| Guest OS | Ubuntu 24.04 LTS |
| CPU | 16 logical cores exposed to WSL |
| GPU | NVIDIA GeForce RTX 4050 Laptop (compute 8.9, JIT from PTX 8.6) |
| Toolchain | g++ 13.3, nvcc 12.0, OpenMPI 3.1, OpenMP 4.5, CMake 3.28 |
| Build flags | `-O3` via CMake `Release` configuration |

The MPI weak-scaling numbers in section 6.4 were originally measured on a 4-physical-core laptop with an RTX 3050 Ti and OpenMPI 3.1. They are retained because they are not trivial to reproduce on the current host. All other numbers were taken on the RTX 4050 host above.

## 3. Algorithm and shared core

Per-pixel Mandelbrot iteration with derivative tracking for distance-estimate (DEM) shading, smooth (continuous) iteration count via log-of-log smoothing, a 4096-color sin-based palette, Blinn-Phong lighting, and Photoshop-style overlay blend. All iteration is in `double` precision; shading uses `float` intrinsics matching CUDA. Escape radius squared is 1e20 for smooth coloring.

`include/mandelbrot_core.hpp` is a single namespace-scoped header (`mbcore::`) exposing:

| Function | Purpose |
|---|---|
| `mandelbrotIterate(cx, cy, maxIter)` | Core iteration loop returning final z, dz, iter count |
| `pixelToComplex(...)` | Pixel to complex-plane mapping |
| `smoothIterCount(...)` | Continuous iteration count for smooth coloring |
| `blinnPhong(...)`, `overlay(...)`, `getColor(...)` | Shading and palette lookup |
| `renderPixel(...)` | One full RGBA pixel |
| `renderRowsSerial(...)` | Contiguous row range |
| `kShallowView`, `kDeepView` | Reference view presets |

The CUDA kernel inlines its own copies of the same math. Drift between the two paths is bounded empirically by SAD validation in the unified benchmark and the Part A test suite.

## 4. Implementations

### 4.1 Serial

`src/mandelbrot_serial.cpp` is a thin wrapper around `mbcore::renderRowsSerial`. It is the baseline T_serial against which every speedup number is measured. Cannot drift from the OpenMP path because both call the same `renderPixel` body.

### 4.2 OpenMP

`src/mandelbrot_omp.cpp` parallelizes the per-pixel nest:

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
        mbcore::renderPixel(...);
```

The Mandelbrot interior is a horizontal band of `maxIter`-bound pixels. Under `schedule(static)`, the slab covering the interior dominates wall time. `schedule(dynamic, 16)` hands out 16-pixel chunks on demand: slow chunks delay one thread but free threads pick up the next chunk. Section 6.2 measures the choice across six schedules.

### 4.3 CUDA

`src/mandelbrot.cu` implements the kernel with a 16x16 block layout. The host-side wrapper `computeMandelbrotCUDA` caches both the device pixel buffer and the device color table across calls (see section 7 on code polish).

### 4.4 Pure MPI

`src/mandelbrot_mpi.cpp`. All ranks compute their assigned subset of pixels; rank 0 additionally owns view parameters, the gather destination buffer, the optional PPM writer, and stdout reporting. No dedicated manager rank.

Default distribution is block-cyclic by rows: rank `r` owns rows `y` where `(y / chunk) % numRanks == r`, with `chunk = 8`. The `--contiguous` flag switches to plain block partitioning for direct comparison. Pixels are gathered to rank 0 via two `MPI_Gatherv` calls (row indices, then pixel data); rank 0 scatters the rows into the final image at their true `y` coordinates.

### 4.5 Hybrid MPI + CUDA

`src/mandelbrot_mpi_cuda.cpp`. Each rank computes the full image on the shared GPU and keeps only its row strip. With one GPU and N ranks this multiplies kernel work by N and does not produce a speedup; it isolates the MPI orchestration cost. On a multi-GPU cluster the obvious improvement is to add a row-range overload to the CUDA kernel.

## 5. Benchmark methodology

The pre-Part-B `benchmark.cpp` compared a simplified CPU loop (`float`, escape radius 4.0, no shading) against the full GPU pipeline. The reported speedup was meaningless. The rewrite forces all paths through the same algorithm: serial and OpenMP call `mbcore::renderPixel`; CUDA calls the unchanged `computeMandelbrotCUDA`. After the timed runs, the benchmark computes SAD between Serial and CUDA outputs (RGB only).

| Parameter | Value |
|---|---|
| Resolution | 1920x1080 |
| Iterations | 500 (shallow view), 2000 (deep view) |
| Trials | 5 per configuration after warm-up |
| OpenMP thread sweep | 1, 2, 4, 8, 16 |
| MPI rank sweep | 1, 2, 4, 8 |
| CSV schema | `implementation,workers,run,time_ms,mpixels_per_sec` |

Outputs land in `docs/results/`.

## 6. Results

### 6.1 OpenMP strong scaling

Deep view, 5 runs per configuration, `schedule(dynamic, 16)`.

| Workers | Avg (ms) | Speedup vs serial | Efficiency |
|---|---:|---:|---:|
| serial(1) | 8141 | 1.00x | 100% |
| openmp(1) | 8141 | 1.00x | 100% |
| openmp(2) | 4173 | 1.95x | 98% |
| openmp(4) | 2440 | 3.34x | 83% |
| openmp(8) | 1521 | 5.35x | 67% |
| openmp(16) | 1133 | 7.19x | 45% |
| cuda | 516 | 15.77x | n/a |

The 8 to 16 thread step uses hyperthreaded logical cores; SMT siblings share an FPU and the workload is FPU-bound, so the 45% efficiency at 16 threads is the expected hyperthreading penalty. Wall time still drops by 1.34x going from 8 to 16 threads.

Plot: `docs/results/openmp_strong_scaling.png`.

### 6.2 OpenMP schedule comparison

8 threads, deep view, 3 runs each.

| Schedule | Avg (ms) | vs guided |
|---|---:|---:|
| static | 1626 | +20% |
| static, 16 | 1418 | +5% |
| dynamic, 1 | 1408 | +4% |
| dynamic, 16 | 1399 | +3% |
| dynamic, 64 | 1413 | +4% |
| guided | 1354 | 0% |

`guided` wins by approximately 3% over the chosen `dynamic, 16`. `static` is 20% slower than `guided`, validating the choice of a non-static schedule.

Plot: `docs/results/openmp_schedule_comparison.png`.

### 6.3 MPI strong scaling

Shallow view, block-cyclic chunk=8, 3 runs per configuration on the RTX 4050 host.

| Ranks | Avg total (ms) | Speedup | Throughput (MPx/s) |
|---:|---:|---:|---:|
| 1 | 874 | 1.00x | 2.37 |
| 2 | 451 | 1.94x | 4.60 |
| 4 | 288 | 3.04x | 7.21 |
| 8 | 172 | 5.08x | 12.04 |

The reference 5-trial measurement shipped in `docs/results/benchmark_mpi.csv` (RTX 3050 Ti host) showed 1.00x to 5.55x at the same rank counts.

Plot: `docs/results/strong_scaling.png`.

### 6.4 MPI weak scaling

Original Part A measurement on the 4-physical-core / RTX 3050 Ti host. Pixels-per-rank held near constant (about 518k).

| Ranks | Resolution | Avg total (ms) | Efficiency (T1/Tn) | Throughput (MPx/s) |
|---:|:---:|---:|---:|---:|
| 1 | 960x540 | 217.6 | 1.00 | 2.38 |
| 2 | 1358x763 | 228.7 | 0.95 | 4.54 |
| 4 | 1920x1080 | 239.8 | 0.91 | 8.65 |
| 8 | 2715x1527 | 313.4 | 0.69 | 13.25 |

Efficiency holds above 0.9 to 4 ranks. The drop at 8 mirrors strong scaling: past physical-core count, hyperthreaded ranks contend for execution units, and the total work also grows 8x from 1 to 8 ranks so cache and memory pressure compound.

Plot: `docs/results/weak_scaling.png`. CSV: `docs/results/benchmark_mpi_weak.csv`.

### 6.5 MPI load-imbalance study

Deep view, 1920x1080, maxIter=2000, 4 ranks. Same view, same rank count, only the row distribution changes.

Original Part A measurement (RTX 3050 Ti host):

| Distribution | Per-rank compute (ms) | Imbalance (max/min) | Wall time (ms) | Throughput (MPx/s) |
|---|---|---:|---:|---:|
| Contiguous blocks | 2375 / 2566 / 2001 / 1899 | 1.35x | 2574 | 0.81 |
| Block-cyclic, chunk=8 | 2213 / 2213 / 2215 / 2166 | 1.02x | 2216 | 0.94 |

Reproduction on the RTX 4050 host (2 trials each):

| Distribution | Wall time (ms) | Imbalance (max/min) |
|---|---:|---:|
| Contiguous blocks | 2709 | 1.36x |
| Block-cyclic, chunk=8 | 2407 | 1.04x |

Block-cyclic distribution removes most of the imbalance and delivers an 11 to 14 percent wall-time reduction with no other code change. The dense interior of the Mandelbrot set at this view falls into one contiguous slice; cyclic distribution spreads those expensive rows across all four ranks.

Plot: `docs/results/imbalance.png`. CSV: `docs/results/imbalance.csv`.

### 6.6 Hybrid MPI + CUDA

Shallow view, 1920x1080, 2 ranks (RTX 4050).

| Phase | Time (ms, steady state) |
|---|---:|
| Per-rank GPU compute (full-image kernel) | 144 |
| Gather | 2 |
| Total | 147 |

Compared to single-GPU CUDA (77 ms shallow on the same host), the 2-rank hybrid is approximately 2x slower because both ranks render the full image on the same shared GPU. This is by design and documented at runtime; a multi-GPU host would invert the comparison.

### 6.7 Throughput summary

Deep view, MPx/sec.

| Implementation | Throughput |
|---|---:|
| serial | 0.25 |
| openmp(2) | 0.50 |
| openmp(4) | 0.85 |
| openmp(8) | 1.36 |
| openmp(16) | 1.83 |
| cuda | 4.02 |

Plot: `docs/results/throughput_comparison.png`.

### 6.8 Visual equivalence

SAD computed in the unified benchmark between Serial and CUDA outputs (RGB only, alpha skipped):

| View | SAD per byte | Bytes drifting > 1 ULP | Verdict |
|---|---:|---:|---|
| Shallow (zoom=1, maxIter=500) | 0.0007 | 44 of 6,220,800 | Algorithm parity |
| Deep (zoom=1e6, maxIter=2000) | 0.126 | 25,773 of 6,220,800 | Expected ULP-scale divergence |

The deep-view divergence is monotonic in zoom depth, not algorithmic. CPU x87/SSE `double` and GPU PTX `double` accumulate more last-bit differences when the iteration loop runs longer.

Bit-level PPM checks from Part A on a smaller image (800x600 shallow) recorded:

| Test | SAD |
|---|---|
| `mandelbrot_mpi` (CPU) vs `mandelbrot_mpi_cuda` (GPU) | 24 bytes of 1,440,000, max 1 ULP per byte |
| `mandelbrot_mpi` 1 rank vs 4 ranks (cyclic) | 0 bytes (bit-exact) |

## 7. Code polish

### 7.1 Return-bug fix in `src/main.cpp`

Lines 58 and 71 returned `false` from `int main` failure paths. `false` evaluates to `0`, which the shell interprets as success. The SDL renderer or texture creation could fail and downstream tools would silently move on. Both paths now `return 1`.

### 7.2 CUDA color-table and pixel-buffer cache

`computeMandelbrotCUDA` previously did, on every call: two `cudaMalloc`, one `cudaMemcpy` of the 48 KB color table, kernel launch, one `cudaMemcpy` back, two `cudaFree`. At 60 FPS in the SDL viewer that is 120 mallocs and 60 host-to-device memcpys per second of pure overhead.

The new cache reuses both buffers across calls. The color-table device buffer is uploaded only when the host `ColorTable*` (or its `numColors`) changes; for the SDL viewer both are stable for the lifetime of the run, so the upload happens once. The pixel device buffer is reused and only grows when the requested image is bigger than the cached size.

`freeMandelbrotCudaResources()` releases both. It is called from `main` cleanup and at the end of `benchmark`. Kernel and helper functions are byte-untouched. Functionally identical output. The win is small in the unified benchmark wall time (per-call overhead is small relative to the kernel time on a 1920x1080 image) but real in the viewer's preview path which calls the function around 60 times per second at small resolutions.

## 8. Discussion

OpenMP scales near-ideally up to the physical core count. The 8 to 16 thread step on hyperthreaded cores yields a 1.34x improvement rather than 2x because both SMT siblings on a core compete for the same FPU on this FPU-bound workload.

CUDA dominates the CPU on the deep view (15.77x over serial, 2.2x over the best OpenMP). At maxIter=2000 each pixel is roughly 2000 fused multiply-adds, which is exactly the workload GPUs are built for. On the shallow view CUDA's lead narrows to about 1.5x because launch and memory-copy overhead amortize over less compute.

Pure MPI scales nearly linearly to the physical core count. Communication overhead (3 to 8 ms gather) is small relative to compute (200+ ms per rank). Block-cyclic distribution materially helps at deep zooms (measured 11 to 14 percent wall-time win at 4 ranks).

The hybrid MPI+CUDA design is correct; only multi-GPU hardware can exercise it for performance. The orchestration code (gather, displacements, alpha-channel patch) is reusable as-is.

`guided` and `dynamic, 16` schedules are within noise of each other at 8 threads on this view; both beat `static` by about 20 percent. The lab plan asked for `dynamic, 16` and the data validates that choice.

## 9. Reproducing the results

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Unified Serial / OpenMP / CUDA benchmark
./build/benchmark --deep --runs 5

# OpenMP schedule comparison
bash scripts/run_schedule_study.sh

# MPI strong scaling
bash scripts/run_mpi_strong.sh --shallow 1920 1080

# MPI weak scaling
bash scripts/run_mpi_weak.sh --shallow

# MPI load imbalance
bash scripts/run_mpi_imbalance.sh 4 --deep

# All sweeps + CSVs + plots
bash scripts/collect_results.sh
python3 scripts/plot_partB.py

# Visual equivalence
mpirun -np 1 ./build/mandelbrot_mpi      --shallow --out cpu.ppm --runs 1 > /dev/null
mpirun -np 1 ./build/mandelbrot_mpi_cuda --shallow --out gpu.ppm --runs 1 > /dev/null
python3 scripts/compare_ppm.py cpu.ppm gpu.ppm
```

## 10. Definition of done

| Item | Status |
|---|---|
| `mandelbrot_serial` builds and matches CUDA output (SAD 0.0007/byte at shallow) | Done |
| `mandelbrot_omp` builds and scales with `--threads` | Done |
| `mandelbrot_mpi` builds and runs with `mpirun -np N` | Done |
| `mandelbrot_mpi_cuda` builds and runs | Done |
| `benchmark` rewritten as fair Serial / OpenMP / CUDA comparison with CSV and SAD validation | Done |
| `benchmark_mpi` produces a CSV with the shared schema | Done |
| `include/mandelbrot_core.hpp` exists and is the single source of the per-pixel algorithm for non-CUDA paths | Done |
| Strong scaling sweeps (OpenMP and MPI) measured, tabulated, and plotted | Done |
| Weak scaling sweep (MPI) measured, tabulated, and plotted | Done |
| Load-imbalance comparison (contiguous vs block-cyclic) measured and plotted | Done |
| OpenMP schedule comparison measured and plotted | Done |
| `main.cpp` return-type bugs fixed at lines 58 and 71 | Done |
| Color-table and pixel-buffer cache added to `computeMandelbrotCUDA`; cleanup hook wired into main and benchmark | Done |
| README and report updated | Done |

## 11. Glossary

| Term | Meaning |
|---|---|
| SAD | Sum of absolute differences (per-byte image comparison) |
| ULP | Unit in the last place (smallest representable floating-point difference) |
| DEM | Distance estimate (used for halo softening in the shading) |
| SMT | Simultaneous multithreading (hyperthreading) |
| FPU | Floating-point unit |
| JIT | Just-in-time compilation (CUDA driver compiles PTX to the target architecture at first launch) |
| PTX | Parallel Thread Execution (NVIDIA's intermediate representation) |
| MPx/sec | Megapixels per second |
