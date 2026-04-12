include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)
if (NOT SUPPORTS_SYCL)
    message(FATAL_ERROR "C++ compiler lacks SYCL support")
endif()

ggml_add_backend_library(ggml-sycl
    ggml-sycl.cpp
    ../../include/ggml-sycl.h)

target_compile_options(ggml-sycl PRIVATE "-fsycl")
target_link_options(ggml-sycl PRIVATE "-fsycl")
