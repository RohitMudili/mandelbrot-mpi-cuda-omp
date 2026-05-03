#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#include <stdint.h>

// Structure to hold color table
struct ColorTable {
    float* colors;     // RGB colors (3 floats per color)
    int numColors;     // Number of colors in table
};

// Color table creation
ColorTable* createSinColorTable(int numColors);
void freeColorTable(ColorTable* table);

// Original direct computation (for fallback)
void computeMandelbrotCUDA(
    uint8_t* pixels,      // Output pixel buffer
    int width,            // Image width
    int height,           // Image height
    double centerX,       // Center of view (real)
    double centerY,       // Center of view (imaginary)
    double zoom,          // Zoom level
    int maxIter,          // Max iterations
    ColorTable* colorTable // Color table (optional, NULL for grayscale)
);

// Release internally-cached device buffers (color table, pixel buffer).
// Call once before exit. Safe to call multiple times.
#ifdef __cplusplus
extern "C" {
#endif
void freeMandelbrotCudaResources();
#ifdef __cplusplus
}
#endif

#endif