#include "llama.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

int32_t llama_split_path(
    char * split_path,
    size_t maxlen,
    const char * path_prefix,
    int32_t split_no,
    int32_t split_count) {

    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";

    const int written = snprintf(
        split_path,
        maxlen,
        SPLIT_PATH_FORMAT,
        path_prefix,
        split_no + 1,
        split_count
    );

    if (written < 0 || (size_t) written >= maxlen) {
        return 0;
    }

    return (int32_t) written;
}

int32_t llama_split_prefix(
    char * split_prefix,
    size_t maxlen,
    const char * split_path,
    int32_t split_no,
    int32_t split_count) {

    const std::string str_split_path(split_path);

    char postfix[32];
    snprintf(postfix, sizeof(postfix), "-%05d-of-%05d.gguf", split_no + 1, split_count);

    const std::string str_postfix(postfix);
    if (str_split_path.size() <= str_postfix.size()) {
        return 0;
    }

    const size_t size_prefix = str_split_path.size() - str_postfix.size();

    if (str_split_path.compare(size_prefix, std::string::npos, str_postfix) == 0) {
        const size_t copy_len = std::min(size_prefix + 1, maxlen);
        snprintf(split_prefix, copy_len, "%s", split_path);

        return (int32_t) size_prefix;
    }

    return 0;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits,
        FILE * file,
        llama_model & model,
        llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.use_mmap, params.use_direct_io,
            params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;
        model.hparams.no_alloc   = params.no_alloc;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }

    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    llama_model * model = new llama_model(params);

    const int status = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}
