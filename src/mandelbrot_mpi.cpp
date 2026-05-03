// mandelbrot_mpi.cpp
// Distributed Mandelbrot renderer using MPI with block-cyclic row distribution.
//
// Process model:
//   - All ranks compute their assigned rows.
//   - Rank 0 also gathers and writes the final image.
//
// Distribution:
//   - Block-cyclic by rows with chunk = 8 (configurable via --chunk).
//   - Rank r owns rows where (row / chunk) % numRanks == r.
//   - Rationale: the dense interior of the Mandelbrot set is a horizontal
//     band; contiguous row blocks would dump most of the expensive pixels
//     onto a few unlucky ranks. Cyclic interleaving evens that out.
//   - For comparison studies, --contiguous switches to plain block partitioning.
//
// Communication:
//   - Each rank packs its rows densely (no holes) into a send buffer along
//     with the row indices it owns, and rank 0 reassembles via MPI_Gatherv.
//
// CLI:
//   mpirun -np N ./mandelbrot_mpi [--shallow|--deep] [--width W --height H]
//                                 [--chunk K] [--contiguous] [--out file.ppm]
//                                 [--no-output] [--runs R]

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>
#include "mandelbrot_core.hpp"

using namespace mbcore;

struct Config {
    int   width   = 1920;
    int   height  = 1080;
    ViewParams view = kShallowView;
    int   chunk   = 8;
    bool  contiguous = false;
    bool  writeOutput = true;
    std::string outPath = "mandelbrot_mpi.ppm";
    int   runs = 1;
};

static void parseArgs(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--shallow")        cfg.view = kShallowView;
        else if (a == "--deep")      cfg.view = kDeepView;
        else if (a == "--width"  && i + 1 < argc) cfg.width  = atoi(argv[++i]);
        else if (a == "--height" && i + 1 < argc) cfg.height = atoi(argv[++i]);
        else if (a == "--chunk"  && i + 1 < argc) cfg.chunk  = atoi(argv[++i]);
        else if (a == "--contiguous") cfg.contiguous = true;
        else if (a == "--out"    && i + 1 < argc) cfg.outPath = argv[++i];
        else if (a == "--no-output")  cfg.writeOutput = false;
        else if (a == "--runs"   && i + 1 < argc) cfg.runs   = atoi(argv[++i]);
        else if (a == "--max-iter" && i + 1 < argc) cfg.view.maxIter = atoi(argv[++i]);
    }
}

// Build the list of row indices owned by `rank` under the chosen distribution.
static std::vector<int> ownedRows(int rank, int numRanks, int height,
                                  int chunk, bool contiguous) {
    std::vector<int> rows;
    if (contiguous) {
        // Plain block partitioning, balanced as evenly as possible.
        int base = height / numRanks;
        int rem  = height % numRanks;
        int start = rank * base + std::min(rank, rem);
        int count = base + (rank < rem ? 1 : 0);
        rows.reserve(count);
        for (int y = start; y < start + count; y++) rows.push_back(y);
    } else {
        // Block-cyclic: rank r owns rows where (y / chunk) % numRanks == r.
        rows.reserve(height / numRanks + chunk);
        for (int y = 0; y < height; y++) {
            int chunkIdx = y / chunk;
            if (chunkIdx % numRanks == rank) rows.push_back(y);
        }
    }
    return rows;
}

// Write an RGBA buffer as a binary PPM (P6, RGB only — alpha is dropped).
static void writePPM(const std::string& path, const uint8_t* pixels,
                     int width, int height) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        return;
    }
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
        printf("=== mandelbrot_mpi ===\n");
        printf("ranks:           %d\n", numRanks);
        printf("resolution:      %dx%d\n", cfg.width, cfg.height);
        printf("view:            center=(%g, %g) zoom=%g maxIter=%d\n",
               cfg.view.centerX, cfg.view.centerY, cfg.view.zoom, cfg.view.maxIter);
        printf("distribution:    %s%s\n",
               cfg.contiguous ? "contiguous" : "block-cyclic",
               cfg.contiguous ? "" : (std::string(" (chunk=") +
                                      std::to_string(cfg.chunk) + ")").c_str());
        printf("runs:            %d\n", cfg.runs);
        fflush(stdout);
    }

    // Build palette (every rank — small, deterministic).
    ColorTableHost* palette = createSinColorTable(kDefaultNumColors);

    // Determine row ownership for this rank.
    std::vector<int> myRows = ownedRows(rank, numRanks, cfg.height,
                                        cfg.chunk, cfg.contiguous);
    int myRowCount = (int)myRows.size();

    // Local buffer holds only the rows we own, packed densely (no per-row gaps).
    // Each row is width*4 bytes (RGBA).
    const int rowBytes = cfg.width * 4;
    std::vector<uint8_t> localPixels((size_t)myRowCount * rowBytes, 0);

    // Helper: render one of our rows into local row slot `localIdx`.
    auto renderLocalRow = [&](int localIdx) {
        int y = myRows[localIdx];
        // We render into a single-row staging buffer using a "virtual" coord
        // system where the row sits at offset 0 inside localPixels[localIdx].
        // renderPixel() expects (y * width + x) * 4 indexing into a full-image
        // buffer, so we instead compute pixels for this y directly and write
        // them to the dense local slot.
        for (int x = 0; x < cfg.width; x++) {
            // Reuse renderPixel by giving it a temporary 1-row buffer trick:
            // pretend height==1 and y==0. But that breaks pixelToComplex which
            // depends on the *actual* y. So inline the body instead, mirroring
            // renderPixel() exactly with the real y.
            double cx, cy;
            pixelToComplex(x, y, cfg.width, cfg.height,
                           cfg.view.centerX, cfg.view.centerY, cfg.view.zoom,
                           cx, cy);
            float diag = sqrtf((1.5f / (float)cfg.view.zoom) * (1.5f / (float)cfg.view.zoom) +
                               (1.0f / (float)cfg.view.zoom) * (1.0f / (float)cfg.view.zoom));
            MandelResult mr = mandelbrotIterate(cx, cy, cfg.view.maxIter);

            uint8_t* px = &localPixels[(size_t)localIdx * rowBytes + x * 4];

            if (mr.iter == cfg.view.maxIter) {
                px[0] = 0; px[1] = 0; px[2] = 0; px[3] = 255;
                continue;
            }

            float smoothIter = smoothIterCount(mr.zx, mr.zy, mr.iter, cfg.view.maxIter);
            double modz  = sqrt(mr.zx * mr.zx + mr.zy * mr.zy);
            double moddz = sqrt(mr.dzx * mr.dzx + mr.dzy * mr.dzy);
            float dem = (float)(modz * log(modz) / moddz / 2.0);
            double normalX = mr.zx / moddz;
            double normalY = mr.zy / moddz;

            uint8_t r, g, b;
            getColor(palette->colors, palette->numColors, smoothIter, kDefaultNcycle,
                     normalX, normalY, dem, diag, r, g, b);
            px[0] = r; px[1] = g; px[2] = b; px[3] = 255;
        }
    };

    // Run timed compute trials.
    std::vector<double> trialTimes;
    trialTimes.reserve(cfg.runs);
    double localComputeMaxAcrossTrials = 0.0;

    for (int trial = 0; trial < cfg.runs; trial++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int i = 0; i < myRowCount; i++) renderLocalRow(i);

        double tCompute = MPI_Wtime();

        // Per-rank compute time, max across ranks (this is the wall time).
        double localCompute = tCompute - t0;
        double globalCompute = 0.0;
        MPI_Reduce(&localCompute, &globalCompute, 1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        // Gather to rank 0. Two pieces of metadata travel with the pixels:
        //  (1) recvcounts: bytes from each rank (rowsOwned * rowBytes)
        //  (2) recvdispls: where they land in a contiguous "gathered" buffer
        // After gather, rank 0 reassembles into the final image using each
        // rank's row-index list.
        std::vector<int> recvCounts, recvDispls;
        std::vector<int> allRowCounts(numRanks, 0);
        std::vector<int> allRowOffsets(numRanks, 0);

        // Tell rank 0 how many rows each rank has.
        MPI_Gather(&myRowCount, 1, MPI_INT,
                   allRowCounts.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            recvCounts.resize(numRanks);
            recvDispls.resize(numRanks);
            int offset = 0;
            for (int r = 0; r < numRanks; r++) {
                recvCounts[r]  = allRowCounts[r] * rowBytes;
                recvDispls[r]  = offset;
                allRowOffsets[r] = offset / rowBytes;
                offset += recvCounts[r];
            }
        }

        // Gather row indices each rank owns, into one flat array on rank 0.
        std::vector<int> allRowIndices;
        if (rank == 0) allRowIndices.resize(cfg.height); // sum of counts == height
        std::vector<int> idxRecvCounts(numRanks, 0), idxRecvDispls(numRanks, 0);
        if (rank == 0) {
            int off = 0;
            for (int r = 0; r < numRanks; r++) {
                idxRecvCounts[r] = allRowCounts[r];
                idxRecvDispls[r] = off;
                off += allRowCounts[r];
            }
        }
        MPI_Gatherv(myRows.data(), myRowCount, MPI_INT,
                    rank == 0 ? allRowIndices.data() : nullptr,
                    idxRecvCounts.data(), idxRecvDispls.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        // Gather pixel rows (densely packed) into a flat receive buffer.
        std::vector<uint8_t> gathered;
        if (rank == 0) gathered.resize((size_t)cfg.height * rowBytes);

        MPI_Gatherv(localPixels.data(),
                    myRowCount * rowBytes,
                    MPI_BYTE,
                    rank == 0 ? gathered.data() : nullptr,
                    recvCounts.data(),
                    recvDispls.data(),
                    MPI_BYTE,
                    0, MPI_COMM_WORLD);

        double tGather = MPI_Wtime();

        // Rank 0: scatter the gathered rows into final image at their true y.
        std::vector<uint8_t> finalImage;
        if (rank == 0) {
            finalImage.assign((size_t)cfg.width * cfg.height * 4, 0);
            int rowsSeen = 0;
            for (int r = 0; r < numRanks; r++) {
                int count = allRowCounts[r];
                for (int j = 0; j < count; j++) {
                    int globalY = allRowIndices[idxRecvDispls[r] + j];
                    const uint8_t* src = &gathered[((size_t)idxRecvDispls[r] + j) * rowBytes];
                    uint8_t* dst = &finalImage[(size_t)globalY * rowBytes];
                    memcpy(dst, src, rowBytes);
                }
                rowsSeen += count;
                (void)rowsSeen;
            }
        }

        double tAssemble = MPI_Wtime();

        if (rank == 0) {
            double total = tAssemble - t0;
            trialTimes.push_back(total * 1000.0); // ms
            printf("trial %d: compute(max-rank)=%.2f ms  gather=%.2f ms  assemble=%.2f ms  total=%.2f ms\n",
                   trial + 1,
                   globalCompute * 1000.0,
                   (tGather - tCompute) * 1000.0,
                   (tAssemble - tGather) * 1000.0,
                   total * 1000.0);
            fflush(stdout);

            if (trial == cfg.runs - 1 && cfg.writeOutput) {
                writePPM(cfg.outPath, finalImage.data(), cfg.width, cfg.height);
                printf("wrote %s\n", cfg.outPath.c_str());
            }
        }

        if (localCompute > localComputeMaxAcrossTrials)
            localComputeMaxAcrossTrials = localCompute;
    }

    // Per-rank load-imbalance dump (useful for the contiguous-vs-cyclic study).
    {
        // Collect per-rank max-trial compute time at rank 0.
        std::vector<double> perRank(numRanks, 0.0);
        MPI_Gather(&localComputeMaxAcrossTrials, 1, MPI_DOUBLE,
                   perRank.data(), 1, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        if (rank == 0) {
            double mn = perRank[0], mx = perRank[0];
            for (double v : perRank) { mn = std::min(mn, v); mx = std::max(mx, v); }
            printf("\nper-rank compute time (max trial):\n");
            for (int r = 0; r < numRanks; r++) {
                printf("  rank %2d: %.2f ms  (%d rows)\n",
                       r, perRank[r] * 1000.0,
                       (int)ownedRows(r, numRanks, cfg.height, cfg.chunk, cfg.contiguous).size());
            }
            printf("imbalance ratio (max/min): %.2fx\n", mx / std::max(mn, 1e-9));
        }
    }

    if (rank == 0 && !trialTimes.empty()) {
        double sum = 0, mn = trialTimes[0], mx = trialTimes[0];
        for (double t : trialTimes) { sum += t; mn = std::min(mn, t); mx = std::max(mx, t); }
        double avg = sum / trialTimes.size();
        double mpx = (cfg.width * cfg.height / 1.0e6) / (avg / 1000.0);
        printf("\n=== summary ===\n");
        printf("avg total: %.2f ms   min: %.2f ms   max: %.2f ms\n", avg, mn, mx);
        printf("throughput: %.2f MPixels/sec\n", mpx);
    }

    freeColorTable(palette);
    MPI_Finalize();
    return 0;
}
