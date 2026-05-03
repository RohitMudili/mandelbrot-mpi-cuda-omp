# Mandelbrot — CUDA + MPI (OpenMP track in progress)

A Mandelbrot set renderer parallelized across multiple paradigms for an HPC lab assignment. **Part A** delivers the CUDA renderer, the shared CPU algorithm core, and pure-MPI plus hybrid MPI+CUDA executables. **Part B** (OpenMP + unified benchmark + code polish) is being implemented separately — see `plan_partB.md`.

## Status

| Track | Status |
|---|---|
| CUDA renderer + SDL2 viewer | ✅ Done |
| Shared algorithm core (`mandelbrot_core.hpp`) | ✅ Done — Part A |
| Pure MPI with block-cyclic distribution | ✅ Done — Part A |
| Hybrid MPI + CUDA | ✅ Done — Part A |
| Scaling + load-imbalance study | ✅ Done — Part A |
| OpenMP implementation | ⏳ Part B |
| Unified Serial / OpenMP / CUDA benchmark | ⏳ Part B |
| Code-polish fixes (return-bug, color-table caching) | ⏳ Part B |

## Highlight results (Part A)

Measured on RTX 3050 Ti Laptop GPU + 4-physical-core CPU, WSL2 Ubuntu 24.04, OpenMPI 3.1.

**Strong scaling, pure MPI, 1920×1080 shallow view:**

| Ranks | Time (ms) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 849 | 1.00× | 100% |
| 2 | 441 | 1.93× | 96% |
| 4 | 243 | 3.49× | 87% |
| 8 | 153 | 5.55× | 69% |

**Load-imbalance study, deep view, 4 ranks:**

| Distribution | Imbalance ratio | Wall time |
|---|---:|---:|
| Contiguous rows | 1.35× | 2574 ms |
| Block-cyclic, chunk=8 | **1.02×** | **2216 ms** (-14%) |

**Visual equivalence:** CPU MPI and GPU MPI outputs differ by 24 bytes out of 1.44 MB (max ±1 ULP per byte). 1-rank vs 4-rank MPI is bit-exact.

Full numbers in [`docs/report_partA.md`](docs/report_partA.md).

## Targets

| Executable | Purpose |
|---|---|
| `mandelbrot` | Interactive SDL2 viewer (CUDA) |
| `benchmark` | CPU vs GPU benchmark |
| `mandelbrot_mpi` | Distributed CPU renderer with block-cyclic / contiguous row distribution |
| `mandelbrot_mpi_cuda` | Distributed orchestration around the CUDA kernel |

## Prerequisites

- CUDA Toolkit 12.x (compute capability 7.5 or 8.6)
- CMake ≥ 3.18
- C++17 compiler
- SDL2 development libraries (for the viewer only)
- An MPI implementation (OpenMPI tested; MPICH should also work)

On Ubuntu 24.04 / WSL2:

```bash
sudo apt install build-essential cmake libsdl2-dev libopenmpi-dev openmpi-bin nvidia-cuda-toolkit
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake skips the MPI targets gracefully if MPI isn't found, so `mandelbrot` and `benchmark` still build on machines without MPI.

## Run

### Interactive viewer (CUDA)

```bash
./build/mandelbrot
```

Left-click + drag to pan, scroll to zoom.

### CPU vs GPU benchmark

```bash
./build/benchmark
```

### Pure MPI

```bash
# 4 ranks, deep zoom, block-cyclic (default chunk=8)
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --deep

# Same but contiguous row partitioning, for comparison
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --deep --contiguous

# Headless benchmark mode
mpirun -np 8 --oversubscribe ./build/mandelbrot_mpi --shallow --no-output --runs 5
```

CLI flags: `--shallow` / `--deep`, `--width W`, `--height H`, `--chunk K`, `--contiguous`, `--out file.ppm`, `--no-output`, `--runs N`, `--max-iter M`.

### Hybrid MPI + CUDA

```bash
mpirun -np 2 --oversubscribe ./build/mandelbrot_mpi_cuda --shallow
```

**Caveat:** because the CUDA function renders the entire image in one kernel launch, each rank computes the full image on the shared GPU and keeps only its row strip. This isolates the MPI orchestration cost; it does not produce a speedup on a single-GPU machine. On multi-GPU clusters, `cudaSetDevice(rank % deviceCount)` would give real gains. See the source comments and `docs/report_partA.md` §3.4.

## Scripts

| Script | What it does |
|---|---|
| `scripts/run_mpi_strong.sh [view] [W] [H]` | Strong-scaling sweep across 1, 2, 4, 8 ranks |
| `scripts/run_mpi_imbalance.sh [ranks] [view]` | Compares contiguous vs block-cyclic distributions |
| `scripts/test_partA.sh` | Full Part A test suite (18 checks) |
| `scripts/compare_ppm.py a.ppm b.ppm` | Byte-level PPM diff for visual-equivalence checks |

Examples:

```bash
bash scripts/run_mpi_strong.sh --shallow 1920 1080
bash scripts/run_mpi_imbalance.sh 4 --deep
bash scripts/test_partA.sh
```

## Project layout

```
.
├── CMakeLists.txt
├── README.md                          (this file)
├── plan.md                            (overall combined plan)
├── plan_partA.md                      (Part A spec — implemented)
├── plan_partB.md                      (Part B spec — in progress)
├── include/
│   ├── mandelbrot.cuh                 (CUDA header)
│   └── mandelbrot_core.hpp            (shared CPU mirror of the CUDA math)
├── src/
│   ├── mandelbrot.cu                  (CUDA kernel)
│   ├── main.cpp                       (SDL viewer)
│   ├── benchmark.cpp                  (CPU vs GPU benchmark)
│   ├── mandelbrot_mpi.cpp             (pure MPI)
│   └── mandelbrot_mpi_cuda.cpp        (hybrid MPI+CUDA)
├── scripts/
│   ├── run_mpi_strong.sh
│   ├── run_mpi_imbalance.sh
│   ├── test_partA.sh
│   └── compare_ppm.py
└── docs/
    └── report_partA.md                (Part A results writeup)
```

## Algorithm

Per-pixel Mandelbrot iteration with:
- `double` precision throughout the iteration loop
- Escape radius² = 1e20 (enables smooth coloring)
- Derivative tracking (`dz = 2·z·dz + 1`) for distance-estimate shading
- Smooth (continuous) iteration count via log-of-log smoothing
- Sin-based 4096-color palette
- Blinn-Phong lighting (azimuth 60°, elevation 50°)
- Photoshop-style overlay blend
- Distance-estimate sigmoid for halo softening

The CPU implementations in `mandelbrot_core.hpp` mirror the CUDA math byte-for-byte (validated to ≤1 ULP per byte across a 1.44 MB rendered image).

## For Part B contributors

Read `plan_partB.md` first. The shared header `include/mandelbrot_core.hpp` exposes:

- `mbcore::mandelbrotIterate(cx, cy, maxIter)` → `MandelResult`
- `mbcore::pixelToComplex(...)`, `smoothIterCount(...)`, `getColor(...)`
- `mbcore::renderPixel(...)` and `mbcore::renderRowsSerial(...)` — drop-in for serial / OpenMP wrappers
- View presets: `mbcore::kShallowView`, `mbcore::kDeepView`

Adding an OpenMP version is mostly:

```cpp
#include "mandelbrot_core.hpp"
// ...
#pragma omp parallel for collapse(2) schedule(dynamic, 16)
for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
        mbcore::renderPixel(pixels, x, y, width, height, ...);
```

See `plan_partB.md` for the full Definition of Done.
