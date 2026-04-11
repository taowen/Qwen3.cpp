add_library(qwen3-llama STATIC
    ../include/llama.h
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-kv-cache.cpp
    llama-memory.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
    models/qwen3.cpp
)

target_include_directories(qwen3-llama PRIVATE .)
target_include_directories(qwen3-llama PUBLIC ../include)
target_compile_features(qwen3-llama PRIVATE cxx_std_17)

target_link_libraries(qwen3-llama PUBLIC ggml)
