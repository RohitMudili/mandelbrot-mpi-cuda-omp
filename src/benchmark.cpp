// Unified Serial / OpenMP / CUDA benchmark. All three paths render the same
// algorithm at the same view; SAD between Serial and CUDA validates parity.
// CSV schema matches benchmark_mpi: implementation,workers,run,time_ms,mpixels_per_sec
//
// CLI: [--shallow|--deep] [--width W --height H] [--max-iter M]
//      [--runs N] [--csv path] [--no-csv] [--ppm]
//      [--threads "1,2,4,8"]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <omp.h>
#include "mandelbrot_core.hpp"
#include "mandelbrot.cuh"

// CUDA cleanup hook implemented in mandelbrot.cu.
extern "C" void freeMandelbrotCudaResources();

namespace mb = mbcore;

struct Cfg {
    int width = 1920, height = 1080;
    mb::ViewParams view = mb::kDeepView;
    int runs = 5;
    std::string csv = "docs/results/benchmark.csv";
    bool writeCSV = true;
    bool writePPM = false;
    std::vector<int> threadsSweep;     // populated below
};

static std::vector<int> defaultThreadsSweep() {
    std::vector<int> t = {1, 2, 4, 8};
    int hw = omp_get_max_threads();
    if (hw > 8 && std::find(t.begin(), t.end(), hw) == t.end()) t.push_back(hw);
    return t;
}

static void parseArgs(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--shallow") c.view = mb::kShallowView;
        else if (a == "--deep")    c.view = mb::kDeepView;
        else if (a == "--width"  && i + 1 < argc) c.width  = atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) c.height = atoi(argv[++i]);
        else if (a == "--max-iter" && i + 1 < argc) c.view.maxIter = atoi(argv[++i]);
        else if (a == "--runs" && i + 1 < argc)    c.runs = atoi(argv[++i]);
        else if (a == "--csv"  && i + 1 < argc)    c.csv = argv[++i];
        else if (a == "--no-csv")                  c.writeCSV = false;
        else if (a == "--ppm")                     c.writePPM = true;
        else if (a == "--threads" && i + 1 < argc) {
            c.threadsSweep.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                std::string tok = s.substr(pos, comma == std::string::npos
                                                 ? std::string::npos : comma - pos);
                if (!tok.empty()) c.threadsSweep.push_back(atoi(tok.c_str()));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
    }
    if (c.threadsSweep.empty()) c.threadsSweep = defaultThreadsSweep();
}

static double sumAbsDiff(const std::vector<uint8_t>& a,
                         const std::vector<uint8_t>& b) {
    double s = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) s += std::abs((int)a[i] - (int)b[i]);
    return s;
}

// One row of CSV output.
struct CsvRow {
    std::string impl;
    int workers;
    int run;
    double timeMs;
    double mpx;
};

static void appendCsv(FILE* f, const CsvRow& r) {
    fprintf(f, "%s,%d,%d,%.3f,%.3f\n",
            r.impl.c_str(), r.workers, r.run, r.timeMs, r.mpx);
    fflush(f);
}

// Render one image with the chosen schedule; used internally by runOmp.
static void renderOmpDynamic16(uint8_t* pixels, int W, int H,
                               const mb::ViewParams& V,
                               const float* tbl, int nC) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            mb::renderPixel(pixels, x, y, W, H,
                            V.centerX, V.centerY, V.zoom, V.maxIter,
                            tbl, nC, mb::kDefaultNcycle);
}

int main(int argc, char** argv) {
    Cfg cfg;
    parseArgs(argc, argv, cfg);

    printf("=== unified benchmark ===\n");
    printf("resolution: %dx%d   view: center=(%g,%g) zoom=%g maxIter=%d   runs=%d\n",
           cfg.width, cfg.height,
           cfg.view.centerX, cfg.view.centerY, cfg.view.zoom, cfg.view.maxIter,
           cfg.runs);
    printf("OpenMP thread sweep:");
    for (int t : cfg.threadsSweep) printf(" %d", t);
    printf("   (omp_get_max_threads=%d)\n", omp_get_max_threads());

    // CSV setup
    FILE* csvFile = nullptr;
    if (cfg.writeCSV) {
        // Open in write mode (truncate) so each invocation produces a fresh CSV.
        csvFile = fopen(cfg.csv.c_str(), "w");
        if (!csvFile) {
            fprintf(stderr, "warning: could not open %s for writing\n", cfg.csv.c_str());
        } else {
            fprintf(csvFile, "implementation,workers,run,time_ms,mpixels_per_sec\n");
        }
    }

    mb::ColorTableHost* hostPalette = mb::createSinColorTable(mb::kDefaultNumColors);
    // The CUDA path uses its own legacy ColorTable struct; build it once.
    ColorTable* cudaPalette = createSinColorTable(mb::kDefaultNumColors);

    const int nPix    = cfg.width * cfg.height;
    const size_t nBytes = (size_t)nPix * 4;
    const double mpx_const = nPix / 1.0e6;

    std::vector<uint8_t> serialPixels(nBytes, 0);
    std::vector<uint8_t> ompPixels(nBytes, 0);
    std::vector<uint8_t> cudaPixels(nBytes, 0);

    auto timed = [](auto fn) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    // -------- Serial --------
    printf("\n--- Serial (1 thread) ---\n");
    double serialAvg = 0;
    {
        std::vector<double> ms;
        for (int r = 1; r <= cfg.runs; r++) {
            double t = timed([&] {
                mb::renderRowsSerial(serialPixels.data(), cfg.width, cfg.height,
                                     0, cfg.height,
                                     cfg.view.centerX, cfg.view.centerY, cfg.view.zoom,
                                     cfg.view.maxIter,
                                     hostPalette->colors, hostPalette->numColors,
                                     mb::kDefaultNcycle);
            });
            ms.push_back(t);
            double mpx = mpx_const / (t / 1000.0);
            printf("  run %d: %.2f ms  (%.2f MPx/s)\n", r, t, mpx);
            if (csvFile) appendCsv(csvFile, {"serial", 1, r, t, mpx});
        }
        serialAvg = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
        double mn = *std::min_element(ms.begin(), ms.end());
        printf("  avg: %.2f ms  min: %.2f ms\n", serialAvg, mn);
    }

    // -------- OpenMP sweep --------
    std::vector<std::pair<int,double>> ompAvgByThreads;
    for (int t : cfg.threadsSweep) {
        omp_set_num_threads(t);
        printf("\n--- OpenMP (threads=%d, schedule=dynamic,16) ---\n", t);
        std::vector<double> ms;
        for (int r = 1; r <= cfg.runs; r++) {
            double tm = timed([&] {
                renderOmpDynamic16(ompPixels.data(), cfg.width, cfg.height,
                                   cfg.view, hostPalette->colors,
                                   hostPalette->numColors);
            });
            ms.push_back(tm);
            double mpx = mpx_const / (tm / 1000.0);
            printf("  run %d: %.2f ms  (%.2f MPx/s)\n", r, tm, mpx);
            if (csvFile) appendCsv(csvFile, {"openmp", t, r, tm, mpx});
        }
        double avg = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
        double mn = *std::min_element(ms.begin(), ms.end());
        printf("  avg: %.2f ms  min: %.2f ms  speedup vs serial: %.2fx  efficiency: %.0f%%\n",
               avg, mn, serialAvg / avg, 100.0 * (serialAvg / avg) / t);
        ompAvgByThreads.emplace_back(t, avg);
    }

    // -------- CUDA --------
    printf("\n--- CUDA ---\n");
    // Warm up (kernel JIT, color-table cache load).
    computeMandelbrotCUDA(cudaPixels.data(), cfg.width, cfg.height,
                          cfg.view.centerX, cfg.view.centerY, cfg.view.zoom,
                          cfg.view.maxIter, cudaPalette);
    double cudaAvg = 0;
    {
        std::vector<double> ms;
        for (int r = 1; r <= cfg.runs; r++) {
            double t = timed([&] {
                computeMandelbrotCUDA(cudaPixels.data(), cfg.width, cfg.height,
                                      cfg.view.centerX, cfg.view.centerY, cfg.view.zoom,
                                      cfg.view.maxIter, cudaPalette);
            });
            ms.push_back(t);
            double mpx = mpx_const / (t / 1000.0);
            printf("  run %d: %.2f ms  (%.2f MPx/s)\n", r, t, mpx);
            if (csvFile) appendCsv(csvFile, {"cuda", 0, r, t, mpx});
        }
        cudaAvg = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
        double mn = *std::min_element(ms.begin(), ms.end());
        printf("  avg: %.2f ms  min: %.2f ms  speedup vs serial: %.2fx\n",
               cudaAvg, mn, serialAvg / cudaAvg);
    }

    // SAD between Serial and CUDA. Skip alpha bytes — CUDA doesn't write them.
    double sad = 0;
    int    drift = 0;
    for (size_t i = 0; i < serialPixels.size(); i++) {
        if ((i & 3) == 3) continue;  // skip alpha
        int d = std::abs((int)serialPixels[i] - (int)cudaPixels[i]);
        sad += d;
        if (d > 1) drift++;
    }
    double sadPerByte = sad / (double)(serialPixels.size() * 3 / 4);
    printf("\n=== Visual equivalence (Serial vs CUDA) ===\n");
    printf("sum-abs-diff: %.0f bytes over %zu RGB bytes  (%.4f / byte)\n",
           sad, serialPixels.size() * 3 / 4, sadPerByte);
    printf("bytes drifting > 1 ULP: %d  ->  %s\n",
           drift,
           sadPerByte < 0.05 ? "OK (within ULP tolerance)"
                             : "DRIFT (algorithms diverged)");

    // -------- Summary table --------
    printf("\n=== summary (avg ms) ===\n");
    printf("  serial(1):  %.2f ms   1.00x\n", serialAvg);
    for (auto [t, avg] : ompAvgByThreads) {
        printf("  openmp(%d): %.2f ms   %.2fx\n", t, avg, serialAvg / avg);
    }
    printf("  cuda:       %.2f ms   %.2fx\n", cudaAvg, serialAvg / cudaAvg);

    // -------- Optional PPM dumps for visual inspection --------
    if (cfg.writePPM) {
        auto dump = [&](const char* path, const std::vector<uint8_t>& px) {
            FILE* f = fopen(path, "wb");
            if (!f) return;
            fprintf(f, "P6\n%d %d\n255\n", cfg.width, cfg.height);
            std::vector<uint8_t> rgb((size_t)cfg.width * cfg.height * 3);
            for (int i = 0; i < cfg.width * cfg.height; i++) {
                rgb[i*3+0] = px[i*4+0]; rgb[i*3+1] = px[i*4+1]; rgb[i*3+2] = px[i*4+2];
            }
            fwrite(rgb.data(), 1, rgb.size(), f);
            fclose(f);
        };
        dump("benchmark_serial.ppm", serialPixels);
        dump("benchmark_omp.ppm",    ompPixels);
        dump("benchmark_cuda.ppm",   cudaPixels);
        printf("\nwrote benchmark_{serial,omp,cuda}.ppm for visual inspection\n");
    }

    if (csvFile) { fclose(csvFile); printf("CSV: %s\n", cfg.csv.c_str()); }

    mb::freeColorTable(hostPalette);
    freeColorTable(cudaPalette);
    freeMandelbrotCudaResources();
    return 0;
}
