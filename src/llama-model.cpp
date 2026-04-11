#include "llama-model.h"

#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache.h"

#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"

// TODO: tmp until the ggml meta backend matures and becomes public
#include "../../ggml/src/ggml-ext.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_32B:           return "32B";
        default:                     return "?B";
    }
}
static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void llama_model::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model::load_arch(llama_model_loader & ml) {
    arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
    if (!devices.empty() && devices[0].is_meta && !llm_arch_supports_sm_tensor(arch)) {
        throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
    }
}

void llama_model::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.swa_layers.begin(), hparams.swa_layers.end(), 0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for differentiating model types
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // arch-specific KVs
    switch (arch) {
        case LLM_ARCH_QWEN3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
                    case 36: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B; break;
                    case 40: type = LLM_TYPE_14B; break;
                    case 64: type = LLM_TYPE_32B; break;
                    default: type = LLM_TYPE_UNKNOWN;
                }
            } break;
        default: throw std::runtime_error("unsupported model architecture: " + arch_name());
    }

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer      = hparams.n_layer;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s, direct_io = %s)\n",
        __func__, ml.use_mmap ? "true" : "false", ml.use_direct_io ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, int(n_layer) + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < int(hparams.n_layer) && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    const auto TENSOR_DUPLICATED      = llama_model_loader::TENSOR_DUPLICATED;
    const auto TENSOR_NOT_REQUIRED    = llama_model_loader::TENSOR_NOT_REQUIRED;
    const auto TENSOR_SKIP            = llama_model_loader::TENSOR_SKIP;
    const auto TENSOR_SKIP_IF_VIRTUAL = llama_model_loader::TENSOR_SKIP_IF_VIRTUAL;

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        const int64_t n_head        = hparams.n_head();
        const int64_t n_head_kv     = hparams.n_head_kv();
        const int64_t n_embd        = hparams.n_embd;
        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa();
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
        const int64_t n_embd_head_k = hparams.n_embd_head_k();
        const int64_t n_embd_head_v = hparams.n_embd_head_v();
        const int64_t n_ff          = hparams.n_ff();
        const int64_t n_embd_gqa    = n_embd_v_gqa;
        const int64_t n_vocab       = vocab.n_tokens();
        const int64_t n_token_types = vocab.n_token_types();
        const int64_t n_rot         = hparams.n_rot();
        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;
        const int64_t n_ctx_train   = hparams.n_ctx_train;

        if (n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        auto create_tensor = [&](const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) -> ggml_tensor * {
            const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
            return ml.create_tensor(
                hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
                tn, ne, flags);
        };

        layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        switch (arch) {
            case LLM_ARCH_QWEN3:
                {
                    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

                    // output
                    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
                    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
                    // if output is NULL, init from the input tok embed
                    if (output == NULL) {
                        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
                    }

                    // output rerank head
                    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

                    for (int i = 0; i < n_layer; ++i) {
                        auto & layer = layers[i];

                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

                        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
                        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
                        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

                        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
                        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

                        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
                        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
                        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
                        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
                    }
                } break;
            default: throw std::runtime_error("unsupported model architecture: " + arch_name());
        }
    }

    ml.done_getting_tensors();

    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ml.ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ml.ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ml.ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t llama_model::n_gpu_layers() const {
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer + 1;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, hparams.no_alloc);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps            = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv           = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (auto label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }
    }

    LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
    if (pimpl->n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
    } else if (pimpl->n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
    } else if (pimpl->n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        case LLM_ARCH_QWEN3:
            {
                GGML_ASSERT(!hparams.is_swa_any());

                res = new llama_kv_cache(
                        *this,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        1,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        nullptr);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm;

    switch (arch) {
        case LLM_ARCH_QWEN3:
            {
                llm = std::make_unique<llm_build_qwen3>(*this, params);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs();

    return llm->res->get_gf();
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ false,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    return model->hparams.n_swa;
}

uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_QWEN3:
            return LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    GGML_UNUSED(model);
    return false;
}

bool llama_model_has_decoder(const llama_model * model) {
    GGML_UNUSED(model);
    return true;
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    GGML_UNUSED(model);
    return LLAMA_TOKEN_NULL;
}

