cmake_minimum_required(VERSION 3.18)
project(mandelbrot_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# OpenMP — required by Part B targets (benchmark, mandelbrot_omp)
find_package(OpenMP REQUIRED)

# Add main executable
add_executable(mandelbrot
    src/main.cpp
    src/mandelbrot.cu
)

# Add benchmark executable (Part B: unified Serial vs OpenMP vs CUDA)
add_executable(benchmark
    src/benchmark.cpp
    src/mandelbrot.cu
)

# Link libraries
target_link_libraries(mandelbrot ${SDL2_LIBRARIES})
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)

# CUDA architecture
set_target_properties(mandelbrot PROPERTIES
    CUDA_ARCHITECTURES "75;86"
)
set_target_properties(benchmark PROPERTIES
    CUDA_ARCHITECTURES "75;86"
)

# ---- Part B: Serial + OpenMP standalone executables ----
add_executable(mandelbrot_serial src/mandelbrot_serial.cpp)
target_include_directories(mandelbrot_serial PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(mandelbrot_omp src/mandelbrot_omp.cpp)
target_include_directories(mandelbrot_omp PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(mandelbrot_omp PRIVATE OpenMP::OpenMP_CXX)

# ---- Part A: MPI targets (additive — does not touch existing executables) ----

find_package(MPI)
if(MPI_CXX_FOUND)
    add_executable(mandelbrot_mpi src/mandelbrot_mpi.cpp)
    target_link_libraries(mandelbrot_mpi PRIVATE MPI::MPI_CXX)
    target_include_directories(mandelbrot_mpi PRIVATE ${PROJECT_SOURCE_DIR}/include)

    add_executable(benchmark_mpi src/benchmark_mpi.cpp)
    target_link_libraries(benchmark_mpi PRIVATE MPI::MPI_CXX)
    target_include_directories(benchmark_mpi PRIVATE ${PROJECT_SOURCE_DIR}/include)

    add_executable(mandelbrot_mpi_cuda
        src/mandelbrot_mpi_cuda.cpp
        src/mandelbrot.cu
    )
    target_link_libraries(mandelbrot_mpi_cuda PRIVATE MPI::MPI_CXX)
    target_include_directories(mandelbrot_mpi_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(mandelbrot_mpi_cuda PROPERTIES
        CUDA_ARCHITECTURES "75;86"
    )
else()
    message(STATUS "MPI not found - skipping mandelbrot_mpi / mandelbrot_mpi_cuda targets")
endif()