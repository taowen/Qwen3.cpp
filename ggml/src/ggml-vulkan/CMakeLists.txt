find_package(Vulkan QUIET)

if (Vulkan_FOUND)
    ggml_add_backend_library(ggml-vulkan
        ggml-vulkan.cpp
        ../../include/ggml-vulkan.h)

    target_link_libraries(ggml-vulkan PRIVATE Vulkan::Vulkan)
    target_include_directories(ggml-vulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else()
    message(WARNING "Vulkan not found; ggml-vulkan backend disabled")
endif()
