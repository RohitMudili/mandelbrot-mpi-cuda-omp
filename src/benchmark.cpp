#include <iostream>
#include <chrono>
#include <climits>
#include <algorithm>
#include "mandelbrot.cuh"

const int WIDTH = 1920;
const int HEIGHT = 1080;
const int MAX_ITER = 500;

int mandelbrotCPU(float cx, float cy, int maxIter) {
    float zx = 0.0f;
    float zy = 0.0f;
    int iter = 0;
    
    while (zx*zx + zy*zy < 4.0f && iter < maxIter) {
        float xtemp = zx*zx - zy*zy + cx;
        zy = 2.0f*zx*zy + cy;
        zx = xtemp;
        iter++;
    }
    
    return iter;
}

void computeMandelbrotCPU(uint8_t* pixels, int width, int height,
                            double centerX, double centerY, double zoom,
                            int maxIter, ColorTable* colorTable) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Map pixel to complex plane
            double cx = centerX + (x - width/2.0) / (width/2.0) * (1.5 / zoom);
            double cy = centerY + (y - height/2.0) / (height/2.0) * (1.0 / zoom);
            
            int iter = mandelbrotCPU(cx, cy, maxIter);
            
            int idx = (y * width + x) * 4;
            
            if (iter == maxIter) {
                // Inside set - black
                pixels[idx + 0] = 0;
                pixels[idx + 1] = 0;
                pixels[idx + 2] = 0;
            } else {
                // Use color table if available
                if (colorTable) {
                    int colorIdx = iter % colorTable->numColors;
                    pixels[idx + 0] = (uint8_t)(colorTable->colors[colorIdx * 3 + 0] * 255);
                    pixels[idx + 1] = (uint8_t)(colorTable->colors[colorIdx * 3 + 1] * 255);
                    pixels[idx + 2] = (uint8_t)(colorTable->colors[colorIdx * 3 + 2] * 255);
                } else {
                    // Grayscale
                    uint8_t color = (uint8_t)((iter * 255) / maxIter);
                    pixels[idx + 0] = color;
                    pixels[idx + 1] = color;
                    pixels[idx + 2] = color;
                }
            }
            pixels[idx + 3] = 255;
        }
    }
}

int main() {
    const int NUM_RUNS = 5;
    
    std::cout << "\n=== Mandelbrot CPU vs GPU Benchmark ===" << std::endl;
    std::cout << "Resolution: " << WIDTH << "x" << HEIGHT << std::endl;
    std::cout << "Iterations: " << MAX_ITER << std::endl;
    std::cout << "Benchmark runs: " << NUM_RUNS << std::endl;
    
    uint8_t* cpuPixels = new uint8_t[WIDTH * HEIGHT * 4];
    uint8_t* gpuPixels = new uint8_t[WIDTH * HEIGHT * 4];
    
    double testCenterX = -0.5;
    double testCenterY = 0.0;
    double testZoom = 1.0;
    
    // Create color table
    ColorTable* colorTable = createSinColorTable(4096);
    
    // Warm up GPU
    std::cout << "\nWarming up GPU..." << std::endl;
    computeMandelbrotCUDA(gpuPixels, WIDTH, HEIGHT, testCenterX, testCenterY, testZoom, MAX_ITER, colorTable);
    
    // CPU Benchmark - multiple runs
    std::cout << "Running CPU benchmark..." << std::endl;
    long long cpuTotalTime = 0;
    long long cpuMinTime = LLONG_MAX;
    long long cpuMaxTime = 0;
    
    for (int i = 0; i < NUM_RUNS; i++) {
        auto cpuStart = std::chrono::high_resolution_clock::now();
        computeMandelbrotCPU(cpuPixels, WIDTH, HEIGHT, testCenterX, testCenterY, testZoom, MAX_ITER, colorTable);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        long long cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();
        
        cpuTotalTime += cpuDuration;
        cpuMinTime = std::min(cpuMinTime, cpuDuration);
        cpuMaxTime = std::max(cpuMaxTime, cpuDuration);
        
        std::cout << "  Run " << (i+1) << ": " << cpuDuration << " ms" << std::endl;
    }
    
    double cpuAvgTime = (double)cpuTotalTime / NUM_RUNS;
    
    // GPU Benchmark - multiple runs
    std::cout << "\nRunning GPU benchmark..." << std::endl;
    long long gpuTotalTime = 0;
    long long gpuMinTime = LLONG_MAX;
    long long gpuMaxTime = 0;
    
    for (int i = 0; i < NUM_RUNS; i++) {
        auto gpuStart = std::chrono::high_resolution_clock::now();
        computeMandelbrotCUDA(gpuPixels, WIDTH, HEIGHT, testCenterX, testCenterY, testZoom, MAX_ITER, colorTable);
        auto gpuEnd = std::chrono::high_resolution_clock::now();
        long long gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(gpuEnd - gpuStart).count();
        
        gpuTotalTime += gpuDuration;
        gpuMinTime = std::min(gpuMinTime, gpuDuration);
        gpuMaxTime = std::max(gpuMaxTime, gpuDuration);
        
        std::cout << "  Run " << (i+1) << ": " << gpuDuration << " ms" << std::endl;
    }
    
    double gpuAvgTime = (double)gpuTotalTime / NUM_RUNS;
    
    std::cout << "\n=== CPU Results ===" << std::endl;
    std::cout << "  Average: " << cpuAvgTime << " ms" << std::endl;
    std::cout << "  Min:     " << cpuMinTime << " ms" << std::endl;
    std::cout << "  Max:     " << cpuMaxTime << " ms" << std::endl;
    
    std::cout << "\n=== GPU Results ===" << std::endl;
    std::cout << "  Average: " << gpuAvgTime << " ms" << std::endl;
    std::cout << "  Min:     " << gpuMinTime << " ms" << std::endl;
    std::cout << "  Max:     " << gpuMaxTime << " ms" << std::endl;
    
    std::cout << "\n=== Comparison ===" << std::endl;
    if (gpuAvgTime > 0) {
        double speedup = cpuAvgTime / gpuAvgTime;
        std::cout << "  Speedup: " << speedup << "x" << std::endl;
    }
    
    double cpuMPixelsPerSec = (WIDTH * HEIGHT / 1000000.0) / (cpuAvgTime / 1000.0);
    double gpuMPixelsPerSec = (WIDTH * HEIGHT / 1000000.0) / (gpuAvgTime / 1000.0);
    
    std::cout << "\n=== Throughput ===" << std::endl;
    std::cout << "  CPU: " << cpuMPixelsPerSec << " MPixels/sec" << std::endl;
    std::cout << "  GPU: " << gpuMPixelsPerSec << " MPixels/sec" << std::endl;
    std::cout << std::endl;
    
    // Cleanup
    freeColorTable(colorTable);
    delete[] cpuPixels;
    delete[] gpuPixels;
    
    return 0;
}
