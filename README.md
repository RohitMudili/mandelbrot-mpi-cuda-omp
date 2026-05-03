# Mandelbrot: OpenMP + CUDA + MPI

Mandelbrot renderer with serial, OpenMP, CUDA, pure-MPI, and hybrid MPI+CUDA implementations. All paths share one per-pixel algorithm (`include/mandelbrot_core.hpp`) so output matches across implementations to within floating-point ULP.

## Status

| Track | Status |
|---|---|
| CUDA renderer + SDL2 viewer | Done |
| Shared algorithm core (`mandelbrot_core.hpp`) | Done |
| Serial reference (`mandelbrot_serial`) | Done |
| OpenMP implementation (`mandelbrot_omp`) | Done |
| Pure MPI with block-cyclic distribution | Done |
| Hybrid MPI + CUDA | Done |
| Unified Serial / OpenMP / CUDA benchmark | Done |
| MPI strong, weak, and load-imbalance studies | Done |
| OpenMP strong-scaling and schedule comparison | Done |
| Code-polish fixes (return-bug, color-table caching) | Done |

## Targets

| Executable | What it is |
|---|---|
| `mandelbrot` | SDL2 viewer (CUDA) |
| `mandelbrot_serial` | Single-threaded reference |
| `mandelbrot_omp` | OpenMP renderer |
| `mandelbrot_mpi` | Distributed CPU (block-cyclic or contiguous rows) |
| `mandelbrot_mpi_cuda` | Distributed orchestration around the CUDA kernel |
| `benchmark` | Serial / OpenMP / CUDA fair benchmark, CSV out |
| `benchmark_mpi` | MPI benchmark, CSV out |

## Prerequisites

CUDA Toolkit 12.x, CMake 3.18 or newer, a C++17 compiler, OpenMP, an MPI implementation, SDL2 (viewer only).

On Ubuntu 24.04 / WSL2:

```bash
sudo apt install build-essential cmake libsdl2-dev libopenmpi-dev openmpi-bin nvidia-cuda-toolkit
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

MPI targets are skipped if MPI is not found; OpenMP is required.

## Run

```bash
./build/mandelbrot                                       # SDL viewer
./build/mandelbrot_serial --deep
./build/mandelbrot_omp --deep --threads 8
./build/benchmark --deep --runs 5                        # writes docs/results/benchmark.csv
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --deep
mpirun -np 4 --oversubscribe ./build/mandelbrot_mpi --deep --contiguous
mpirun -np 2 --oversubscribe ./build/mandelbrot_mpi_cuda --shallow
```

Common flags: `--shallow` / `--deep`, `--width W`, `--height H`, `--max-iter M`, `--runs N`, `--no-output`, `--out file.ppm`. MPI-specific: `--chunk K`, `--contiguous`. OpenMP-specific: `--threads T`, `--schedule {static|static-16|dynamic-1|dynamic-16|dynamic-64|guided}`.

## Results

Headline numbers (RTX 4050 + 16 logical cores, deep view 1920x1080, maxIter=2000, 5 runs):

| Implementation | Avg (ms) | Speedup vs serial |
|---|---:|---:|
| serial | 8141 | 1.00x |
| openmp(2) | 4173 | 1.95x |
| openmp(4) | 2440 | 3.34x |
| openmp(8) | 1521 | 5.35x |
| openmp(16) | 1133 | 7.19x |
| cuda | 516 | 15.77x |

MPI strong scaling (shallow view, block-cyclic chunk=8): 1 to 8 ranks gives 5.1x speedup. MPI load imbalance: contiguous 1.36x, block-cyclic 1.04x.

OpenMP schedule comparison at 8 threads (deep view): `static` 1626 ms, `dynamic-16` 1399 ms, `guided` 1354 ms.

Plots and CSVs in `docs/results/`. Full writeup in `docs/report.md`.

## Layout

```
include/   mandelbrot.cuh, mandelbrot_core.hpp
src/       mandelbrot.cu, main.cpp, benchmark.cpp,
           mandelbrot_serial.cpp, mandelbrot_omp.cpp,
           mandelbrot_mpi.cpp, mandelbrot_mpi_cuda.cpp, benchmark_mpi.cpp
scripts/   run_mpi_strong.sh, run_mpi_imbalance.sh, run_schedule_study.sh,
           plot_mpi_results.py, plot_partB.py, compare_ppm.py, test_partA.sh
docs/      report.md, results/
```

## Algorithm

Per-pixel iteration with `double` precision, escape radius squared = 1e20, derivative tracking for distance-estimate shading, log-of-log smooth iteration count, 4096-color sin-based palette, Blinn-Phong lighting, overlay blend. CPU and CUDA paths produce visually equivalent output (SAD under 1 ULP per byte at shallow view; small ULP-scale drift at deep zoom).
