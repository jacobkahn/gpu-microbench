cmake_minimum_required(VERSION 3.18)

project(gpu-microbench LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(bench Benchmark.cpp Utils.cpp Kernels.cu)

target_include_directories(bench PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(bench PRIVATE CUDA::cudart)
