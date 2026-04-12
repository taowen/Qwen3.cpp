add_compile_definitions(GGML_SCHED_MAX_COPIES=${GGML_SCHED_MAX_COPIES})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (GGML_BACKEND_DL AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "GGML_BACKEND_DL requires BUILD_SHARED_LIBS")
endif()

add_library(ggml-base
    ../include/ggml.h
    ../include/ggml-alloc.h
    ../include/ggml-backend.h
    ../include/ggml-cpp.h
    ../include/gguf.h
    ggml-core.c
    ggml-graph.c
    ggml-ops.c
    ggml-stubs.c
    ggml.cpp
    ggml-alloc.c
    ggml-backend-core.cpp
    ggml-backend-sched.cpp
    ggml-backend-cpu.cpp
    ggml-threading.cpp
    ggml-threading.h
    ggml-minimal-stubs.cpp
    ggml-quants.c
    ggml-quants.h
    gguf.cpp)

if (GGML_SCHED_NO_REALLOC)
    target_compile_definitions(ggml-base PUBLIC GGML_SCHED_NO_REALLOC)
endif()

target_compile_definitions(ggml-base PRIVATE
    GGML_VERSION="0.0.0"
    GGML_COMMIT="unknown"
)

add_library(ggml
    ggml-backend-dl.cpp
    ggml-backend-reg.cpp)
add_library(ggml::ggml ALIAS ggml)

target_link_libraries(ggml PUBLIC ggml-base)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml PRIVATE dl)
endif()

function(ggml_add_backend_library backend)
    if (GGML_BACKEND_DL)
        add_library(${backend} MODULE ${ARGN})
        set_target_properties(${backend} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_DL)
        add_dependencies(ggml ${backend})
    else()
        add_library(${backend} ${ARGN})
        target_link_libraries(ggml PUBLIC ${backend})
    endif()

    target_link_libraries(${backend} PRIVATE ggml-base)
    target_include_directories(${backend} PRIVATE ..)

    if (${BUILD_SHARED_LIBS})
        target_compile_definitions(${backend} PRIVATE GGML_BACKEND_BUILD)
        target_compile_definitions(${backend} PUBLIC GGML_BACKEND_SHARED)
    endif()
endfunction()

if (GGML_CPU)
    add_subdirectory(ggml-cpu)
    if (NOT GGML_BACKEND_DL)
        target_compile_definitions(ggml PUBLIC GGML_USE_CPU)
    endif()
endif()

if (GGML_SYCL)
    add_subdirectory(ggml-sycl)
    if (NOT GGML_BACKEND_DL)
        target_compile_definitions(ggml PUBLIC GGML_USE_SYCL)
    endif()
endif()

if (GGML_VULKAN)
    add_subdirectory(ggml-vulkan)
    if (NOT GGML_BACKEND_DL)
        target_compile_definitions(ggml PUBLIC GGML_USE_VULKAN)
    endif()
endif()

foreach (target ggml-base ggml)
    target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
    target_compile_features(${target} PRIVATE c_std_11 cxx_std_17)
endforeach()

target_link_libraries(ggml-base PRIVATE Threads::Threads)

if (BUILD_SHARED_LIBS)
    foreach (target ggml-base ggml)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_compile_definitions(${target} PRIVATE GGML_BUILD)
        target_compile_definitions(${target} PUBLIC GGML_SHARED)
    endforeach()
endif()
