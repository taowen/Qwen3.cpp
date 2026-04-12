set(GGML_CPU_NAME ggml-cpu)

ggml_add_backend_library(${GGML_CPU_NAME}
    ggml-cpu-kernel.c
    ggml-cpu.cpp
    repack.cpp
    repack.h
    hbm.cpp
    hbm.h
    quants.c
    quants.h
    traits.cpp
    traits.h
    ggml-cpu-impl.h
    common.h
    binary-ops.h
    binary-ops.cpp
    unary-ops.h
    unary-ops.cpp
    simd-mappings.h
    vec.h
    vec.cpp
    ops.h
    ops.cpp
    arch/x86/quants.c
    arch/x86/repack.cpp
    arch/x86/cpu-feats.cpp
    ../../include/ggml-cpu.h)

target_include_directories(${GGML_CPU_NAME} PRIVATE .. ../.. .)
target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)

if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
endif()
