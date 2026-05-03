// OpenMP renderer. Default schedule is dynamic,16; the interior band of the
// Mandelbrot set is a hot strip of maxIter pixels, so static would leave most
// threads idle while one thread chews through it. --schedule lets the choice
// be measured (see scripts/run_schedule_study.sh).
//
// CLI: [--shallow|--deep] [--width W --height H] [--max-iter M]
//      [--threads T] [--schedule {static|static-16|dynamic-1|dynamic-16|dynamic-64|guided}]
//      [--out file.ppm] [--no-output] [--runs N]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include "mandelbrot_core.hpp"

using namespace mbcore;

enum class Sched { Static, Static16, Dynamic1, Dynamic16, Dynamic64, Guided };

struct Cfg {
    int width = 1920, height = 1080;
    ViewParams view = kDeepView;
    int threads = 0;            // 0 → use OMP default
    Sched sched = Sched::Dynamic16;
    bool writeOutput = true;
    std::string outPath = "mandelbrot_omp.ppm";
    int runs = 1;
};

static Sched parseSched(const std::string& s) {
    if (s == "static")     return Sched::Static;
    if (s == "static-16")  return Sched::Static16;
    if (s == "dynamic-1")  return Sched::Dynamic1;
    if (s == "dynamic-16") return Sched::Dynamic16;
    if (s == "dynamic-64") return Sched::Dynamic64;
    if (s == "guided")     return Sched::Guided;
    fprintf(stderr, "unknown schedule '%s', using dynamic-16\n", s.c_str());
    return Sched::Dynamic16;
}

static const char* schedName(Sched s) {
    switch (s) {
        case Sched::Static:    return "static";
        case Sched::Static16:  return "static,16";
        case Sched::Dynamic1:  return "dynamic,1";
        case Sched::Dynamic16: return "dynamic,16";
        case Sched::Dynamic64: return "dynamic,64";
        case Sched::Guided:    return "guided";
    }
    return "?";
}

static void parseArgs(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--shallow") c.view = kShallowView;
        else if (a == "--deep")    c.view = kDeepView;
        else if (a == "--width"  && i + 1 < argc) c.width  = atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) c.height = atoi(argv[++i]);
        else if (a == "--max-iter" && i + 1 < argc) c.view.maxIter = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) c.threads = atoi(argv[++i]);
        else if (a == "--schedule" && i + 1 < argc) c.sched = parseSched(argv[++i]);
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

// Six near-identical loops because the schedule clause needs a compile-time
// literal when combined with collapse(2).
static void renderOMP(uint8_t* pixels, const Cfg& cfg,
                      const ColorTableHost* palette) {
    const int W = cfg.width, H = cfg.height;
    const ViewParams V = cfg.view;
    const float* tbl = palette->colors;
    const int    nC  = palette->numColors;

    switch (cfg.sched) {
    case Sched::Static:
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, kDefaultNcycle);
        break;
    case Sched::Static16:
        #pragma omp parallel for collapse(2) schedule(static, 16)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, kDefaultNcycle);
        break;
    case Sched::Dynamic1:
        #pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, kDefaultNcycle);
        break;
    case Sched::Dynamic16:
        #pragma omp parallel for collapse(2) schedule(dynamic, 16)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, kDefaultNcycle);
        break;
    case Sched::Dynamic64:
        #pragma omp parallel for collapse(2) schedule(dynamic, 64)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, kDefaultNcycle);
        break;
    case Sched::Guided:
        #pragma omp parallel for collapse(2) schedule(guided)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, kDefaultNcycle);
        break;
    }
}

int main(int argc, char** argv) {
    Cfg cfg;
    parseArgs(argc, argv, cfg);

    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
    int actualThreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        actualThreads = omp_get_num_threads();
    }

    printf("=== mandelbrot_omp ===\n");
    printf("threads:    %d  (omp_get_max_threads=%d)\n",
           actualThreads, omp_get_max_threads());
    printf("schedule:   %s\n", schedName(cfg.sched));
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
        renderOMP(pixels.data(), cfg, palette);
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
