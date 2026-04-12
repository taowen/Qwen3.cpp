cmake_minimum_required(VERSION 3.16)
project(ggml C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

option(BUILD_SHARED_LIBS "ggml: build shared libraries" OFF)
option(GGML_BACKEND_DL "ggml: build backends as dynamic libraries" OFF)
option(GGML_CPU "ggml: enable CPU backend" ON)
option(GGML_SYCL "ggml: enable SYCL backend" OFF)
option(GGML_VULKAN "ggml: enable Vulkan backend" OFF)
option(GGML_OPENMP "ggml: enable OpenMP in CPU backend" ON)
option(GGML_ALL_WARNINGS "ggml: enable compiler warnings" ON)
option(GGML_FATAL_WARNINGS "ggml: enable -Werror" OFF)
option(GGML_SCHED_NO_REALLOC "ggml: disallow reallocations in ggml-alloc" OFF)
set(GGML_SCHED_MAX_COPIES "4" CACHE STRING "ggml: max input copies")

add_subdirectory(src)
