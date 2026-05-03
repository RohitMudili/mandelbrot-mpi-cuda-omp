// Single-threaded reference renderer. Baseline for every speedup number.
// Uses mbcore::renderRowsSerial from the shared header so output matches the
// OpenMP / CUDA / MPI paths.
//
// CLI: [--shallow|--deep] [--width W --height H] [--max-iter M]
//      [--out file.ppm] [--no-output] [--runs N]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include "mandelbrot_core.hpp"

using namespace mbcore;

struct Cfg {
    int width = 1920, height = 1080;
    ViewParams view = kDeepView;
    bool writeOutput = true;
    std::string outPath = "mandelbrot_serial.ppm";
    int runs = 1;
};

static void parseArgs(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--shallow") c.view = kShallowView;
        else if (a == "--deep")    c.view = kDeepView;
        else if (a == "--width"  && i + 1 < argc) c.width  = atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) c.height = atoi(argv[++i]);
        else if (a == "--max-iter" && i + 1 < argc) c.view.maxIter = atoi(argv[++i]);
        else if (a == "--out"    && i + 1 < argc) c.outPath = argv[++i];
        else if (a == "--no-output") c.writeOutput = false;
        else if (a == "--runs"   && i + 1 < argc) c.runs = atoi(argv[++i]);
    }
}

static void writePPM(const std::string& path, const uint8_t* pixels,
                     int width, int height) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "failed to open %s\n", path.c_str()); return; }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    std::vector<uint8_t> rgb((size_t)width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = pixels[i * 4 + 0];
        rgb[i * 3 + 1] = pixels[i * 4 + 1];
        rgb[i * 3 + 2] = pixels[i * 4 + 2];
    }
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    Cfg cfg;
    parseArgs(argc, argv, cfg);

    printf("=== mandelbrot_serial ===\n");
    printf("resolution: %dx%d  view: center=(%g,%g) zoom=%g maxIter=%d  runs=%d\n",
           cfg.width, cfg.height,
           cfg.view.centerX, cfg.view.centerY, cfg.view.zoom, cfg.view.maxIter,
           cfg.runs);

    ColorTableHost* palette = createSinColorTable(kDefaultNumColors);
    std::vector<uint8_t> pixels((size_t)cfg.width * cfg.height * 4, 0);

    std::vector<double> trialMs;
    trialMs.reserve(cfg.runs);
    for (int t = 0; t < cfg.runs; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        renderRowsSerial(pixels.data(), cfg.width, cfg.height,
                         0, cfg.height,
                         cfg.view.centerX, cfg.view.centerY, cfg.view.zoom,
                         cfg.view.maxIter,
                         palette->colors, palette->numColors, kDefaultNcycle);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        trialMs.push_back(ms);
        printf("trial %d: %.2f ms\n", t + 1, ms);
    }

    if (cfg.writeOutput) {
        writePPM(cfg.outPath, pixels.data(), cfg.width, cfg.height);
        printf("wrote %s\n", cfg.outPath.c_str());
    }

    if (!trialMs.empty()) {
        double sum = 0, mn = trialMs[0], mx = trialMs[0];
        for (double v : trialMs) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
        double avg = sum / trialMs.size();
        double mpx = (cfg.width * cfg.height / 1.0e6) / (avg / 1000.0);
        printf("\n=== summary ===\navg: %.2f ms  min: %.2f ms  max: %.2f ms\n", avg, mn, mx);
        printf("throughput: %.2f MPixels/sec\n", mpx);
    }

    freeColorTable(palette);
    return 0;
}
