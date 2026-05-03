// mandelbrot_core.hpp
// Shared per-pixel Mandelbrot algorithm used by serial / OpenMP / MPI
// implementations. This is a faithful CPU mirror of the math in src/mandelbrot.cu
// — same formulas, same constants, same operation order, same precision mix
// (double for iteration, float for shading). DO NOT diverge from the CUDA
// version; if the kernel ever changes, mirror the change here.
//
// This header is intentionally CPU-only. The CUDA kernel in src/mandelbrot.cu
// is unchanged and continues to use its own internal copies of these helpers.

#ifndef MANDELBROT_CORE_HPP
#define MANDELBROT_CORE_HPP

#include <stdint.h>
#include <math.h>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mbcore {

struct ColorTableHost {
    float* colors;   // RGB triples, 3 floats per color, length = 3 * numColors
    int    numColors;
};

struct MandelResult {
    int    iter;
    double zx, zy;
    double dzx, dzy;
};

// Build the same sin-based palette as createSinColorTable() in mandelbrot.cu.
inline ColorTableHost* createSinColorTable(int numColors) {
    ColorTableHost* table = new ColorTableHost;
    table->numColors = numColors;
    table->colors = new float[numColors * 3];

    for (int i = 0; i < numColors; i++) {
        float t = (float)i / (numColors - 1);
        float r, g, b;

        if (t < 0.33f) {
            float s = t / 0.33f;
            r = 0.0f;
            g = 0.2f * s;
            b = 0.3f + 0.7f * s;
        } else if (t < 0.66f) {
            float s = (t - 0.33f) / 0.33f;
            r = 0.0f;
            g = 0.2f + 0.8f * s;
            b = 1.0f - 0.3f * s;
        } else {
            float s = (t - 0.66f) / 0.34f;
            r = 0.4f * s;
            g = 1.0f - 0.8f * s;
            b = 0.7f + 0.3f * s;
        }

        table->colors[i * 3 + 0] = fmaxf(0.0f, fminf(1.0f, r));
        table->colors[i * 3 + 1] = fmaxf(0.0f, fminf(1.0f, g));
        table->colors[i * 3 + 2] = fmaxf(0.0f, fminf(1.0f, b));
    }
    return table;
}

inline void freeColorTable(ColorTableHost* table) {
    if (table) {
        delete[] table->colors;
        delete table;
    }
}

// Map a pixel to its complex-plane coordinate.
// Matches the kernel's exact formula (mandelbrot.cu:205-206).
inline void pixelToComplex(int x, int y, int width, int height,
                           double centerX, double centerY, double zoom,
                           double& cx, double& cy) {
    cx = centerX + ((x - width / 2.0) / (width / 2.0)) * (1.5 / zoom);
    cy = centerY + ((y - height / 2.0) / (height / 2.0)) * (1.0 / zoom);
}

// Per-pixel Mandelbrot iteration with derivative tracking for distance
// estimation. Mirror of mandelbrotDevice() in mandelbrot.cu:154-184.
inline MandelResult mandelbrotIterate(double cx, double cy, int maxIter) {
    double zx = 0.0, zy = 0.0;
    double dzx = 1.0, dzy = 0.0;
    int iter = 0;

    const double escapeRadius2 = 1e20;

    while (zx * zx + zy * zy < escapeRadius2 && iter < maxIter) {
        double dzx_new = 2.0 * (zx * dzx - zy * dzy) + 1.0;
        double dzy_new = 2.0 * (zx * dzy + zy * dzx);
        dzx = dzx_new;
        dzy = dzy_new;

        double xtemp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xtemp;
        iter++;
    }

    return MandelResult{ iter, zx, zy, dzx, dzy };
}

// Smooth (continuous) iteration count.
// Mirror of smoothIterCount() in mandelbrot.cu:95-107.
inline float smoothIterCount(double zx, double zy, int iter, int maxIter) {
    if (iter == maxIter) return 0.0f;

    const double escapeRadius = 1e10;
    const double logEscapeRadius = log(escapeRadius);

    double modz = sqrt(zx * zx + zy * zy);
    double logRatio = 2.0 * log(modz) / logEscapeRadius;
    float smoothVal = 1.0f - (float)(log(logRatio) / log(2.0));

    return (float)iter + smoothVal;
}

// Blinn-Phong shading. Mirror of blinnPhong() in mandelbrot.cu:48-81.
// Uses the f-suffixed math intrinsics to match CUDA's cosf/sinf/powf precision.
inline float blinnPhong(double normalX, double normalY,
                        float lightAzimuth, float lightElevation,
                        float intensity, float kAmbient, float kDiffuse,
                        float kSpecular, float shininess) {
    double normalMag = sqrt(normalX * normalX + normalY * normalY + 1.0);
    double nx = normalX / normalMag;
    double ny = normalY / normalMag;
    double nz = 1.0 / normalMag;

    float lx = cosf(lightAzimuth) * cosf(lightElevation);
    float ly = sinf(lightAzimuth) * cosf(lightElevation);
    float lz = sinf(lightElevation);

    float ldiff = (float)(nx * lx + ny * ly + nz * lz);
    ldiff = ldiff / (1.0f + (float)nz * lz);

    float phiHalf = (M_PI / 2.0f + lightElevation) / 2.0f;
    float lspec = (float)(nx * cosf(lightAzimuth) * sinf(phiHalf) +
                          ny * sinf(lightAzimuth) * sinf(phiHalf) +
                          nz * cosf(phiHalf));
    lspec = lspec / (1.0f + (float)nz * cosf(phiHalf));
    lspec = powf(fmaxf(0.0f, lspec), shininess);

    float bright = kAmbient + kDiffuse * ldiff + kSpecular * lspec;
    bright = bright * intensity + (1.0f - intensity) / 2.0f;

    return bright;
}

// Photoshop-style overlay blend. Mirror of overlay() in mandelbrot.cu:84-92.
inline float overlay(float x, float y, float gamma) {
    float out;
    if (2.0f * y < 1.0f) {
        out = 2.0f * x * y;
    } else {
        out = 1.0f - 2.0f * (1.0f - x) * (1.0f - y);
    }
    return out * gamma + x * (1.0f - gamma);
}

// Full per-pixel color computation: palette lookup + Blinn-Phong + DEM softening.
// Mirror of getColor() in mandelbrot.cu:110-152.
inline void getColor(const float* colorTable, int numColors,
                     float niter, float ncycle,
                     double normalX, double normalY, float dem, float diag,
                     uint8_t& r, uint8_t& g, uint8_t& b) {
    float normalized = fmodf(sqrtf(niter), ncycle) / ncycle;

    int colorIdx = (int)roundf(normalized * (numColors - 1));
    if (colorIdx < 0) colorIdx = 0;
    if (colorIdx > numColors - 1) colorIdx = numColors - 1;

    float rf = colorTable[colorIdx * 3 + 0];
    float gf = colorTable[colorIdx * 3 + 1];
    float bf = colorTable[colorIdx * 3 + 2];

    float lightAzimuth   = 60.0f * 2.0f * (float)M_PI / 360.0f;
    float lightElevation = 50.0f * (float)M_PI / 180.0f;
    float intensity = 0.65f;
    float kAmbient  = 0.3f;
    float kDiffuse  = 0.6f;
    float kSpecular = 0.35f;
    float shininess = 30.0f;

    float bright = blinnPhong(normalX, normalY, lightAzimuth, lightElevation,
                              intensity, kAmbient, kDiffuse, kSpecular, shininess);

    dem = dem / diag;
    dem = -logf(dem + 1e-10f) / 12.0f;
    dem = 1.0f / (1.0f + expf(-10.0f * (dem - 0.5f)));

    rf = overlay(rf, bright, 1.0f) * (1.0f - dem) + rf * dem;
    gf = overlay(gf, bright, 1.0f) * (1.0f - dem) + gf * dem;
    bf = overlay(bf, bright, 1.0f) * (1.0f - dem) + bf * dem;

    r = (uint8_t)(fminf(fmaxf(rf, 0.0f), 1.0f) * 255.0f);
    g = (uint8_t)(fminf(fmaxf(gf, 0.0f), 1.0f) * 255.0f);
    b = (uint8_t)(fminf(fmaxf(bf, 0.0f), 1.0f) * 255.0f);
}

// Compute a single RGBA pixel. Mirrors the body of mandelbrotKernel()
// in mandelbrot.cu:186-251 for one (x, y).
inline void renderPixel(uint8_t* pixels, int x, int y, int width, int height,
                        double centerX, double centerY, double zoom, int maxIter,
                        const float* colorTable, int numColors, float ncycle) {
    double cx, cy;
    pixelToComplex(x, y, width, height, centerX, centerY, zoom, cx, cy);

    float diag = sqrtf((1.5f / (float)zoom) * (1.5f / (float)zoom) +
                       (1.0f / (float)zoom) * (1.0f / (float)zoom));

    MandelResult mr = mandelbrotIterate(cx, cy, maxIter);

    int pixelIndex = (y * width + x) * 4;

    if (mr.iter == maxIter) {
        pixels[pixelIndex + 0] = 0;
        pixels[pixelIndex + 1] = 0;
        pixels[pixelIndex + 2] = 0;
        pixels[pixelIndex + 3] = 255;
        return;
    }

    if (colorTable != nullptr && numColors > 0) {
        float smoothIter = smoothIterCount(mr.zx, mr.zy, mr.iter, maxIter);

        double modz  = sqrt(mr.zx * mr.zx + mr.zy * mr.zy);
        double moddz = sqrt(mr.dzx * mr.dzx + mr.dzy * mr.dzy);
        float dem = (float)(modz * log(modz) / moddz / 2.0);

        double normalX = mr.zx / moddz;
        double normalY = mr.zy / moddz;

        uint8_t r, g, b;
        getColor(colorTable, numColors, smoothIter, ncycle,
                 normalX, normalY, dem, diag, r, g, b);

        pixels[pixelIndex + 0] = r;
        pixels[pixelIndex + 1] = g;
        pixels[pixelIndex + 2] = b;
        pixels[pixelIndex + 3] = 255;
    } else {
        uint8_t color = (uint8_t)(255 * mr.iter / maxIter);
        pixels[pixelIndex + 0] = color;
        pixels[pixelIndex + 1] = color;
        pixels[pixelIndex + 2] = color;
        pixels[pixelIndex + 3] = 255;
    }
}

// Convenience: render a contiguous row range [yStart, yEnd) into a buffer
// sized as if it were full image (so each pixel writes at (y * width + x) * 4).
// Used by both pure MPI (per-rank rows) and the hybrid MPI+CUDA path.
inline void renderRowsSerial(uint8_t* pixels, int width, int height,
                             int yStart, int yEnd,
                             double centerX, double centerY, double zoom,
                             int maxIter,
                             const float* colorTable, int numColors,
                             float ncycle) {
    for (int y = yStart; y < yEnd; y++) {
        for (int x = 0; x < width; x++) {
            renderPixel(pixels, x, y, width, height,
                        centerX, centerY, zoom, maxIter,
                        colorTable, numColors, ncycle);
        }
    }
}

// Default cycle constant matching mandelbrot.cu:270.
constexpr float kDefaultNcycle = 32.0f;

// Default palette size matching main.cpp:81.
constexpr int kDefaultNumColors = 4096;

// Reference deep-zoom view used for benchmarks/scaling studies.
struct ViewParams {
    double centerX;
    double centerY;
    double zoom;
    int    maxIter;
};

constexpr ViewParams kShallowView{ -0.5,                0.0,                1.0,    500  };
constexpr ViewParams kDeepView   { -0.743643887037151,  0.131825904205330,  1.0e6,  2000 };

} // namespace mbcore

#endif // MANDELBROT_CORE_HPP
