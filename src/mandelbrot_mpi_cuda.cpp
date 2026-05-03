// mandelbrot_mpi_cuda.cpp
// Hybrid MPI + CUDA renderer.
//
// Design constraint: do NOT modify the existing CUDA code (mandelbrot.cu /
// mandelbrot.cuh). The existing computeMandelbrotCUDA() renders a full image
// in a single kernel launch; it does not accept a row range. Within that
// constraint there are two honest options:
//
//   (a) Have each rank compute the FULL image with computeMandelbrotCUDA(),
//       then keep only its assigned row range. Simple; correct; wastes GPU
//       work. With one shared GPU and N ranks this is N× the kernel work.
//
//   (b) Have each rank compute its OWN sub-image at a SHIFTED view such that
//       its sub-image lines up with one row strip of the global image. This
//       requires custom math because the kernel's pixel-to-complex mapping
//       uses (width, height) as global, not local — translating the view
//       analytically while keeping the math identical is fragile and easy to
//       get subtly wrong.
//
// We choose (a). It is mathematically equivalent to the single-rank CUDA
// path and uses computeMandelbrotCUDA verbatim. The performance number it
// produces is the gather/assemble overhead on top of single-GPU CUDA — i.e.
// it shows the *cost* of MPI orchestration, not a speedup. On a multi-GPU
// system you would partition by row range and see real gains; on this
// single-GPU laptop the design is the deliverable, not the wall time.
//
// Distribution:
//   - Block partitioning by rows: rank r owns rows [yStart_r, yEnd_r).
//   - Each rank runs computeMandelbrotCUDA() on the full image, copies its
//     row range into a local dense buffer, and gathers to rank 0.
//
// CLI mirrors mandelbrot_mpi (use --shallow / --deep / --width / --height).

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include "mandelbrot_core.hpp"
#include "mandelbrot.cuh"   // unchanged CUDA header

// Intentionally NOT `using namespace mbcore` here — mandelbrot.cuh exposes a
// global createSinColorTable / freeColorTable that would collide with the
// namespaced versions in mandelbrot_core.hpp. We use the CUDA-side palette
// (since we call computeMandelbrotCUDA verbatim) and only pull view types
// from mbcore.
using mbcore::ViewParams;
using mbcore::kShallowView;
using mbcore::kDeepView;
using mbcore::kDefaultNumColors;

struct Config {
    int width = 1920, height = 1080;
    ViewParams view = kShallowView;
    bool writeOutput = true;
    std::string outPath = "mandelbrot_mpi_cuda.ppm";
    int runs = 1;
};

static void parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--shallow")        cfg.view = kShallowView;
        else if (a == "--deep")           cfg.view = kDeepView;
        else if (a == "--width"  && i + 1 < argc) cfg.width  = atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) cfg.height = atoi(argv[++i]);
        else if (a == "--out"    && i + 1 < argc) cfg.outPath = argv[++i];
        else if (a == "--no-output")      cfg.writeOutput = false;
        else if (a == "--runs"   && i + 1 < argc) cfg.runs   = atoi(argv[++i]);
        else if (a == "--max-iter" && i + 1 < argc) cfg.view.maxIter = atoi(argv[++i]);
    }
}

static void rowRange(int rank, int numRanks, int height, int& yStart, int& yEnd) {
    int base = height / numRanks;
    int rem  = height % numRanks;
    yStart = rank * base + std::min(rank, rem);
    int count = base + (rank < rem ? 1 : 0);
    yEnd = yStart + count;
}

static void writePPM(const std::string& path, const uint8_t* pixels,
                     int width, int height) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    std::vector<uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        rgb[i * 3 + 0] = pixels[i * 4 + 0];
        rgb[i * 3 + 1] = pixels[i * 4 + 1];
        rgb[i * 3 + 2] = pixels[i * 4 + 2];
    }
    fwrite(rgb.data(), 1, rgb.size(), f);
    fclose(f);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Config cfg;
    parseArgs(argc, argv, cfg);

    if (rank == 0) {
        printf("=== mandelbrot_mpi_cuda ===\n");
        printf("ranks: %d  resolution: %dx%d  view: center=(%g,%g) zoom=%g maxIter=%d\n",
               numRanks, cfg.width, cfg.height,
               cfg.view.centerX, cfg.view.centerY, cfg.view.zoom, cfg.view.maxIter);
        printf("note: each rank renders the full image on its (shared) GPU and keeps only its row strip.\n");
        printf("      this isolates the MPI orchestration cost; multi-GPU would give real speedup.\n");
        fflush(stdout);
    }

    // Build the same palette used by the CUDA kernel internally.
    ColorTable* palette = createSinColorTable(kDefaultNumColors);

    int yStart, yEnd;
    rowRange(rank, numRanks, cfg.height, yStart, yEnd);
    const int myRowCount = yEnd - yStart;
    const int rowBytes   = cfg.width * 4;

    std::vector<uint8_t> fullImage((size_t)cfg.width * cfg.height * 4, 0);
    std::vector<uint8_t> localStrip((size_t)myRowCount * rowBytes, 0);

    std::vector<double> trialTimes;
    trialTimes.reserve(cfg.runs);

    for (int trial = 0; trial < cfg.runs; trial++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // Call the unchanged CUDA function. It renders the full image.
        computeMandelbrotCUDA(fullImage.data(), cfg.width, cfg.height,
                              cfg.view.centerX, cfg.view.centerY, cfg.view.zoom,
                              cfg.view.maxIter, palette);

        // Set alpha (the kernel does not write the alpha channel).
        for (int i = 0; i < cfg.width * cfg.height; i++) {
            fullImage[i * 4 + 3] = 255;
        }

        // Extract our row strip.
        memcpy(localStrip.data(),
               fullImage.data() + (size_t)yStart * rowBytes,
               (size_t)myRowCount * rowBytes);

        double tCompute = MPI_Wtime();

        // Gather all strips back to rank 0.
        std::vector<int> recvCounts(numRanks), recvDispls(numRanks);
        for (int r = 0; r < numRanks; r++) {
            int s, e;
            rowRange(r, numRanks, cfg.height, s, e);
            recvCounts[r] = (e - s) * rowBytes;
            recvDispls[r] = s * rowBytes;
        }

        std::vector<uint8_t> gathered;
        if (rank == 0) gathered.assign((size_t)cfg.width * cfg.height * 4, 0);

        MPI_Gatherv(localStrip.data(), myRowCount * rowBytes, MPI_BYTE,
                    rank == 0 ? gathered.data() : nullptr,
                    recvCounts.data(), recvDispls.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        if (rank == 0) {
            double total = (t1 - t0) * 1000.0;
            trialTimes.push_back(total);
            printf("trial %d: compute(rank-local)=%.2f ms  gather=%.2f ms  total=%.2f ms\n",
                   trial + 1,
                   (tCompute - t0) * 1000.0,
                   (t1 - tCompute) * 1000.0,
                   total);
            fflush(stdout);

            if (trial == cfg.runs - 1 && cfg.writeOutput) {
                writePPM(cfg.outPath, gathered.data(), cfg.width, cfg.height);
                printf("wrote %s\n", cfg.outPath.c_str());
            }
        }
    }

    if (rank == 0 && !trialTimes.empty()) {
        double sum = 0, mn = trialTimes[0], mx = trialTimes[0];
        for (double t : trialTimes) { sum += t; mn = std::min(mn, t); mx = std::max(mx, t); }
        double avg = sum / trialTimes.size();
        double mpx = (cfg.width * cfg.height / 1.0e6) / (avg / 1000.0);
        printf("\n=== summary ===\navg: %.2f ms  min: %.2f ms  max: %.2f ms\nthroughput: %.2f MPixels/sec\n",
               avg, mn, mx, mpx);
    }

    freeColorTable(palette);
    MPI_Finalize();
    return 0;
}
