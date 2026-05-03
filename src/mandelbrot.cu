#include "mandelbrot.cuh"
#include <cuda_runtime.h>
#include <math.h>

ColorTable* createSinColorTable(int numColors) {
    ColorTable* table = new ColorTable;
    table->numColors = numColors;
    table->colors = new float[numColors * 3];
    
    for (int i = 0; i < numColors; i++) {
        float t = (float)i / (numColors - 1);  // 0 to 1
        
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

void freeColorTable(ColorTable* table) {
    if (table) {
        delete[] table->colors;
        delete table;
    }
}

// Blinn-Phong shading for lighting effects
__device__ float blinnPhong(double normalX, double normalY, 
                            float lightAzimuth, float lightElevation,
                            float intensity, float kAmbient, float kDiffuse, 
                            float kSpecular, float shininess) {
    // Normalize the normal vector
    double normalMag = sqrt(normalX * normalX + normalY * normalY + 1.0);
    double nx = normalX / normalMag;
    double ny = normalY / normalMag;
    double nz = 1.0 / normalMag;
    
    // Light vector: [cos(theta)cos(phi), sin(theta)cos(phi), sin(phi)]
    float lx = cosf(lightAzimuth) * cosf(lightElevation);
    float ly = sinf(lightAzimuth) * cosf(lightElevation);
    float lz = sinf(lightElevation);
    
    // Diffuse light = dot(light, normal)
    float ldiff = nx * lx + ny * ly + nz * lz;
    ldiff = ldiff / (1.0f + nz * lz); // Normalization
    
    // Specular light: Blinn-Phong
    float phiHalf = (M_PI / 2.0f + lightElevation) / 2.0f;
    float lspec = nx * cosf(lightAzimuth) * sinf(phiHalf) +
                  ny * sinf(lightAzimuth) * sinf(phiHalf) +
                  nz * cosf(phiHalf);
    lspec = lspec / (1.0f + nz * cosf(phiHalf)); // Normalization
    lspec = powf(fmaxf(0.0f, lspec), shininess);
    
    // Brightness = ambient + diffuse + specular
    float bright = kAmbient + kDiffuse * ldiff + kSpecular * lspec;
    // Add intensity
    bright = bright * intensity + (1.0f - intensity) / 2.0f;
    
    return bright;
}

// Overlay blend mode for combining colors
__device__ float overlay(float x, float y, float gamma) {
    float out;
    if (2.0f * y < 1.0f) {
        out = 2.0f * x * y;
    } else {
        out = 1.0f - 2.0f * (1.0f - x) * (1.0f - y);
    }
    return out * gamma + x * (1.0f - gamma);
}

// Smooth iteration count with escape radius
__device__ float smoothIterCount(double zx, double zy, int iter, int maxIter) {
    if (iter == maxIter) return 0.0f;
    
    // Large escape radius for better smoothing
    const double escapeRadius = 1e10;
    const double logEscapeRadius = log(escapeRadius);
    
    double modz = sqrt(zx * zx + zy * zy);
    double logRatio = 2.0 * log(modz) / logEscapeRadius;
    float smoothVal = 1.0f - log(logRatio) / log(2.0);
    
    return (float)iter + smoothVal;
}

// Device function to get color from table with shading
__device__ void getColor(float* colorTable, int numColors, float niter, float ncycle,
                        double normalX, double normalY, float dem, float diag,
                        uint8_t& r, uint8_t& g, uint8_t& b) {
    // Apply power transform and map to [0,1]
    float normalized = fmodf(sqrtf(niter), ncycle) / ncycle;
    
    // Get color index
    int colorIdx = (int)roundf(normalized * (numColors - 1));
    colorIdx = min(max(colorIdx, 0), numColors - 1);
    
    // Extract RGB from color table (as floats 0-1)
    float rf = colorTable[colorIdx * 3 + 0];
    float gf = colorTable[colorIdx * 3 + 1];
    float bf = colorTable[colorIdx * 3 + 2];
    
    // Blinn-Phong lighting parameters - adjusted for better HD appearance
    float lightAzimuth = 60.0f * 2.0f * M_PI / 360.0f;     // 60 degrees for better angle
    float lightElevation = 50.0f * M_PI / 180.0f;          // 50 degrees elevation
    float intensity = 0.65f;      // Slightly reduced intensity for subtlety
    float kAmbient = 0.3f;        // More ambient light for better visibility
    float kDiffuse = 0.6f;        // Stronger diffuse for definition
    float kSpecular = 0.35f;      // Reduced specular for less harsh highlights
    float shininess = 30.0f;      // Slightly higher for tighter highlights
    
    // Calculate brightness from normal
    float bright = blinnPhong(normalX, normalY, lightAzimuth, lightElevation,
                             intensity, kAmbient, kDiffuse, kSpecular, shininess);
    
    // Distance estimate: log transform and sigmoid
    dem = dem / diag;  // Normalize by diagonal
    dem = -logf(dem + 1e-10f) / 12.0f;
    dem = 1.0f / (1.0f + expf(-10.0f * (dem - 0.5f)));
    
    // Apply brightness with overlay blend, modulated by distance estimate
    rf = overlay(rf, bright, 1.0f) * (1.0f - dem) + rf * dem;
    gf = overlay(gf, bright, 1.0f) * (1.0f - dem) + gf * dem;
    bf = overlay(bf, bright, 1.0f) * (1.0f - dem) + bf * dem;
    
    // Clamp and convert to uint8
    r = (uint8_t)(fminf(fmaxf(rf, 0.0f), 1.0f) * 255.0f);
    g = (uint8_t)(fminf(fmaxf(gf, 0.0f), 1.0f) * 255.0f);
    b = (uint8_t)(fminf(fmaxf(bf, 0.0f), 1.0f) * 255.0f);
}

__device__ int mandelbrotDevice(double cx, double cy, int maxIter, 
                               double* outZx, double* outZy, 
                               double* outDzx, double* outDzy) {
    double zx = 0.0;
    double zy = 0.0;
    double dzx = 1.0;  // Derivative starts at 1
    double dzy = 0.0;
    int iter = 0;
    
    const double escapeRadius2 = 1e20;  // Large escape radius squared
    
    while (zx*zx + zy*zy < escapeRadius2 && iter < maxIter) {
        // Update derivative: dz = 2*z*dz + 1
        double dzx_new = 2.0 * (zx * dzx - zy * dzy) + 1.0;
        double dzy_new = 2.0 * (zx * dzy + zy * dzx);
        dzx = dzx_new;
        dzy = dzy_new;
        
        // Update z: z = z^2 + c
        double xtemp = zx*zx - zy*zy + cx;
        zy = 2.0*zx*zy + cy;
        zx = xtemp;
        iter++;
    }
    
    *outZx = zx;
    *outZy = zy;
    *outDzx = dzx;
    *outDzy = dzy;
    return iter;
}

__global__ void mandelbrotKernel(
    uint8_t* pixels,
    int width,
    int height,
    double centerX,
    double centerY,
    double zoom,
    int maxIter,
    float* colorTable,
    int numColors,
    float ncycle
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    double cx = centerX + ((x - width/2.0) / (width/2.0)) * (1.5 / zoom);
    double cy = centerY + ((y - height/2.0) / (height/2.0)) * (1.0 / zoom);
    
    // Calculate diagonal for DEM normalization
    float diag = sqrtf((1.5f / zoom) * (1.5f / zoom) + (1.0f / zoom) * (1.0f / zoom));
    
    double zx, zy, dzx, dzy;
    int iter = mandelbrotDevice(cx, cy, maxIter, &zx, &zy, &dzx, &dzy);

    int pixelIndex = (y * width + x) * 4;
    
    if (iter == maxIter) {
        // Inside - black
        pixels[pixelIndex] = 0;
        pixels[pixelIndex + 1] = 0; 
        pixels[pixelIndex + 2] = 0; 
    } else {
        if (colorTable != nullptr && numColors > 0) {
            // Smooth iteration count
            float smoothIter = smoothIterCount(zx, zy, iter, maxIter);
            
            // Calculate distance estimate (DEM)
            double modz = sqrt(zx * zx + zy * zy);
            double moddz = sqrt(dzx * dzx + dzy * dzy);
            float dem = modz * log(modz) / moddz / 2.0;
            
            // Calculate normal vector for lighting
            double normalX = zx / moddz;
            double normalY = zy / moddz;
            
            // Get color from table with shading
            uint8_t r, g, b;
            getColor(colorTable, numColors, smoothIter, ncycle, 
                    normalX, normalY, dem, diag, r, g, b);
            
            pixels[pixelIndex] = r;
            pixels[pixelIndex + 1] = g;
            pixels[pixelIndex + 2] = b;
        } else {
            // Fallback to grayscale
            uint8_t color = static_cast<uint8_t>(255 * iter / maxIter);
            pixels[pixelIndex] = color;
            pixels[pixelIndex + 1] = color;
            pixels[pixelIndex + 2] = color;
        }
    }
}

// Cached device buffers. The color table re-uploaded on every call (48 KB at
// the default palette size) and the pixel buffer was malloc'd and freed every
// frame; at 60 FPS that's significant overhead. Both are now reused across
// calls. freeMandelbrotCudaResources() releases them before exit.
static float*       g_d_colorTable      = nullptr;
static int          g_d_colorTableCount = 0;
static const ColorTable* g_lastTablePtr = nullptr;
static uint8_t*     g_d_pixels          = nullptr;
static size_t       g_d_pixelsBytes     = 0;

extern "C" void freeMandelbrotCudaResources() {
    if (g_d_colorTable) { cudaFree(g_d_colorTable); g_d_colorTable = nullptr; }
    g_d_colorTableCount = 0;
    g_lastTablePtr = nullptr;
    if (g_d_pixels) { cudaFree(g_d_pixels); g_d_pixels = nullptr; }
    g_d_pixelsBytes = 0;
}

void computeMandelbrotCUDA(
    uint8_t* pixels,
    int width,
    int height,
    double centerX,
    double centerY,
    double zoom,
    int maxIter,
    ColorTable* colorTable
) {
    size_t pixelSize = (size_t)width * height * 4 * sizeof(uint8_t);

    // Reuse the device pixel buffer when possible; only grow.
    if (pixelSize > g_d_pixelsBytes) {
        if (g_d_pixels) cudaFree(g_d_pixels);
        cudaMalloc(&g_d_pixels, pixelSize);
        g_d_pixelsBytes = pixelSize;
    }

    int numColors = 0;
    float ncycle = 32.0f; // Default cycle value from Python
    float* d_colorTable = nullptr;

    if (colorTable != nullptr) {
        numColors = colorTable->numColors;
        // Pointer equality is enough — callers hold a stable ColorTable* for
        // the run.
        if (colorTable != g_lastTablePtr || numColors != g_d_colorTableCount) {
            if (g_d_colorTable) cudaFree(g_d_colorTable);
            size_t colorSize = (size_t)numColors * 3 * sizeof(float);
            cudaMalloc(&g_d_colorTable, colorSize);
            cudaMemcpy(g_d_colorTable, colorTable->colors, colorSize,
                       cudaMemcpyHostToDevice);
            g_lastTablePtr      = colorTable;
            g_d_colorTableCount = numColors;
        }
        d_colorTable = g_d_colorTable;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y
    );

    mandelbrotKernel<<<gridDim, blockDim>>>(
        g_d_pixels,
        width,
        height,
        centerX,
        centerY,
        zoom,
        maxIter,
        d_colorTable,
        numColors,
        ncycle
    );

    cudaDeviceSynchronize();

    cudaMemcpy(pixels, g_d_pixels, pixelSize, cudaMemcpyDeviceToHost);
}
