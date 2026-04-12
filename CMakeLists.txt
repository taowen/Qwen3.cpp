cmake_minimum_required(VERSION 3.16)
project(qwen3-cpp C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QWEN3_CPP_BUILD_CLI "Build minimal qwen3 CLI" OFF)

if (NOT TARGET ggml)
    add_subdirectory(ggml)
endif()

add_subdirectory(src)

if (QWEN3_CPP_BUILD_CLI)
    add_executable(qwen3-cli
        tools/cli/cli.cpp
    )

    target_include_directories(qwen3-cli PRIVATE include)
    target_compile_features(qwen3-cli PRIVATE cxx_std_17)
    target_link_libraries(qwen3-cli PRIVATE qwen3-llama)
endif()
