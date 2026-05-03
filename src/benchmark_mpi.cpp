// benchmark_mpi.cpp
// MPI-aware benchmark for Part A. Runs the pure-MPI Mandelbrot render N times
// and emits one CSV row per trial:
//
//   implementation,workers,run,time_ms,mpixels_per_sec
//
// Only rank 0 writes to the CSV / stdout. The outer harness invokes this
// once per rank-count via mpirun.
//
// CLI:
//   mpirun -np N ./benchmark_mpi [--shallow|--deep] [--width W --height H]
//                                [--runs R] [--csv path] [--impl name]
//                                [--chunk K] [--contiguous]
//                                [--header]   (rank 0 prints CSV header first)

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include "mandelbrot_core.hpp"

using namespace mbcore;

struct Cfg {
    int   width = 1920;
    int   height = 1080;
    ViewParams view = kShallowView;
    int   runs = 5;
    int   chunk = 8;
    bool  contiguous = false;
    std::string csv = "";
    std::string impl = "mpi";
    bool  header = false;
};

static void parseArgs(int argc, char** argv, Cfg& c) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--shallow")        c.view = kShallowView;
        else if (a == "--deep")      c.view = kDeepView;
        else if (a == "--width" && i+1 < argc)  c.width = atoi(argv[++i]);
        else if (a == "--height" && i+1 < argc) c.height = atoi(argv[++i]);
        else if (a == "--runs"  && i+1 < argc)  c.runs  = atoi(argv[++i]);
        else if (a == "--csv"   && i+1 < argc)  c.csv   = argv[++i];
        else if (a == "--impl"  && i+1 < argc)  c.impl  = argv[++i];
        else if (a == "--chunk" && i+1 < argc)  c.chunk = atoi(argv[++i]);
        else if (a == "--contiguous") c.contiguous = true;
        else if (a == "--header") c.header = true;
        else if (a == "--max-iter" && i+1 < argc) c.view.maxIter = atoi(argv[++i]);
    }
}

static std::vector<int> ownedRows(int rank, int numRanks, int height,
                                  int chunk, bool contiguous) {
    std::vector<int> rows;
    if (contiguous) {
        int base = height / numRanks;
        int rem  = height % numRanks;
        int start = rank * base + std::min(rank, rem);
        int count = base + (rank < rem ? 1 : 0);
        rows.reserve(count);
        for (int y = start; y < start + count; y++) rows.push_back(y);
    } else {
        rows.reserve(height / numRanks + chunk);
        for (int y = 0; y < height; y++) {
            int chunkIdx = y / chunk;
            if (chunkIdx % numRanks == rank) rows.push_back(y);
        }
    }
    return rows;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Cfg cfg;
    parseArgs(argc, argv, cfg);

    ColorTableHost* palette = createSinColorTable(kDefaultNumColors);

    std::vector<int> myRows = ownedRows(rank, numRanks, cfg.height, cfg.chunk, cfg.contiguous);
    int myRowCount = (int)myRows.size();

    const int rowBytes = cfg.width * 4;
    std::vector<uint8_t> localPixels((size_t)myRowCount * rowBytes, 0);

    auto renderLocalRow = [&](int localIdx) {
        int y = myRows[localIdx];
        for (int x = 0; x < cfg.width; x++) {
            double cx, cy;
            pixelToComplex(x, y, cfg.width, cfg.height,
                           cfg.view.centerX, cfg.view.centerY, cfg.view.zoom, cx, cy);
            float diag = sqrtf((1.5f / (float)cfg.view.zoom) * (1.5f / (float)cfg.view.zoom) +
                               (1.0f / (float)cfg.view.zoom) * (1.0f / (float)cfg.view.zoom));
            MandelResult mr = mandelbrotIterate(cx, cy, cfg.view.maxIter);
            uint8_t* px = &localPixels[(size_t)localIdx * rowBytes + x * 4];
            if (mr.iter == cfg.view.maxIter) {
                px[0] = 0; px[1] = 0; px[2] = 0; px[3] = 255; continue;
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

    FILE* csvFile = nullptr;
    if (rank == 0 && !cfg.csv.empty()) {
        // Append mode — outer harness controls header writing via --header.
        csvFile = fopen(cfg.csv.c_str(), "a");
        if (!csvFile) fprintf(stderr, "warning: could not open %s\n", cfg.csv.c_str());
    }
    if (rank == 0 && cfg.header) {
        const char* hdr = "implementation,workers,run,time_ms,mpixels_per_sec\n";
        if (csvFile) fputs(hdr, csvFile);
        fputs(hdr, stdout);
    }

    for (int trial = 1; trial <= cfg.runs; trial++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int i = 0; i < myRowCount; i++) renderLocalRow(i);

        // Gather (timing includes communication).
        std::vector<int> allRowCounts(numRanks, 0);
        MPI_Gather(&myRowCount, 1, MPI_INT,
                   allRowCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> recvCounts, recvDispls, idxRecvCounts, idxRecvDispls;
        if (rank == 0) {
            recvCounts.resize(numRanks);
            recvDispls.resize(numRanks);
            idxRecvCounts.resize(numRanks);
            idxRecvDispls.resize(numRanks);
            int off = 0, ioff = 0;
            for (int r = 0; r < numRanks; r++) {
                recvCounts[r] = allRowCounts[r] * rowBytes;
                recvDispls[r] = off;        off  += recvCounts[r];
                idxRecvCounts[r] = allRowCounts[r];
                idxRecvDispls[r] = ioff;    ioff += allRowCounts[r];
            }
        }

        std::vector<int> allRowIndices;
        if (rank == 0) allRowIndices.resize(cfg.height);
        MPI_Gatherv(myRows.data(), myRowCount, MPI_INT,
                    rank == 0 ? allRowIndices.data() : nullptr,
                    idxRecvCounts.data(), idxRecvDispls.data(), MPI_INT,
                    0, MPI_COMM_WORLD);

        std::vector<uint8_t> gathered;
        if (rank == 0) gathered.resize((size_t)cfg.height * rowBytes);
        MPI_Gatherv(localPixels.data(), myRowCount * rowBytes, MPI_BYTE,
                    rank == 0 ? gathered.data() : nullptr,
                    recvCounts.data(), recvDispls.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);

        // Assemble (counted in wall time so the CSV reflects end-to-end render).
        if (rank == 0) {
            std::vector<uint8_t> finalImage((size_t)cfg.width * cfg.height * 4, 0);
            for (int r = 0; r < numRanks; r++) {
                int count = allRowCounts[r];
                for (int j = 0; j < count; j++) {
                    int globalY = allRowIndices[idxRecvDispls[r] + j];
                    const uint8_t* src = &gathered[((size_t)idxRecvDispls[r] + j) * rowBytes];
                    memcpy(&finalImage[(size_t)globalY * rowBytes], src, rowBytes);
                }
            }
        }
        double t1 = MPI_Wtime();

        if (rank == 0) {
            double ms = (t1 - t0) * 1000.0;
            double mpx = (cfg.width * cfg.height / 1.0e6) / ((t1 - t0));
            char line[256];
            snprintf(line, sizeof(line), "%s,%d,%d,%.3f,%.3f\n",
                     cfg.impl.c_str(), numRanks, trial, ms, mpx);
            if (csvFile) fputs(line, csvFile);
            fputs(line, stdout);
            fflush(stdout);
        }
    }

    if (csvFile) fclose(csvFile);
    freeColorTable(palette);
    MPI_Finalize();
    return 0;
}
