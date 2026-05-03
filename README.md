# Mandelbrot CUDA

A CUDA-accelerated Mandelbrot set renderer with SDL2 visualization.

## Prerequisites

- CUDA Toolkit (compatible with compute capability 7.5 and 8.6)
- SDL2 library
- CMake 3.18 or higher
- C++17 compatible compiler

## Building

   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

## Running

### Interactive Viewer
Run the main application:
```bash
./mandelbrot
```
- Use mouse to pan
- Scroll to zoom in/out

### Benchmark
Run the benchmark to compare CPU vs GPU performance:
```bash
./benchmark
```
This will output timing information for rendering a 1920x1080 image.