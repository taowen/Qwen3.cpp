add_executable(qwen3-quantize quantize.cpp)
target_link_libraries(qwen3-quantize PRIVATE qwen3-llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(qwen3-quantize PRIVATE ../../include)
target_compile_features(qwen3-quantize PRIVATE cxx_std_17)
