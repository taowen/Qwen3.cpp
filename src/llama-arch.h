#pragma once

#include "ggml.h" // ggml_op

#include <string>
#include <vector>

//
// gguf constants (sync with gguf.py)
//

enum llm_arch {
    LLM_ARCH_QWEN3,
    LLM_ARCH_UNKNOWN,
};

enum llm_kv {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_FILE_TYPE,
    LLM_KV_GENERAL_NAME,

    LLM_KV_VOCAB_SIZE,
    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_EMBEDDING_LENGTH_OUT,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_EXPERT_COUNT,
    LLM_KV_EXPERT_USED_COUNT,
    LLM_KV_POOLING_TYPE,
    LLM_KV_CLASSIFIER_OUTPUT_LABELS,

    LLM_KV_ATTENTION_CAUSAL,
    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_KEY_LENGTH,
    LLM_KV_ATTENTION_VALUE_LENGTH,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
    LLM_KV_ATTENTION_KEY_LENGTH_SWA,
    LLM_KV_ATTENTION_VALUE_LENGTH_SWA,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_DIMENSION_COUNT_SWA,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,
    LLM_KV_ROPE_SCALING_TYPE,
    LLM_KV_ROPE_SCALING_FACTOR,
    LLM_KV_ROPE_SCALING_ATTN_FACTOR,
    LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
    LLM_KV_ROPE_SCALING_FINETUNED,

    LLM_KV_SPLIT_NO,
    LLM_KV_SPLIT_COUNT,
    LLM_KV_SPLIT_TENSORS_COUNT,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_PRE,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_EOT_ID,
    LLM_KV_TOKENIZER_EOM_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_MASK_ID,
    LLM_KV_TOKENIZER_ADD_BOS,
    LLM_KV_TOKENIZER_ADD_EOS,
    LLM_KV_TOKENIZER_ADD_SEP,
    LLM_KV_TOKENIZER_ADD_PREFIX,
    LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,
    LLM_KV_TOKENIZER_FIM_PRE_ID,
    LLM_KV_TOKENIZER_FIM_SUF_ID,
    LLM_KV_TOKENIZER_FIM_MID_ID,
    LLM_KV_TOKENIZER_FIM_PAD_ID,
    LLM_KV_TOKENIZER_FIM_REP_ID,
    LLM_KV_TOKENIZER_FIM_SEP_ID,

    // deprecated:
    LLM_KV_TOKENIZER_PREFIX_ID,
    LLM_KV_TOKENIZER_SUFFIX_ID,
    LLM_KV_TOKENIZER_MIDDLE_ID,
};

enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
    LLM_TENSOR_CLS_OUT,
};

enum llm_tensor_layer {
    LLM_TENSOR_LAYER_INPUT,
    LLM_TENSOR_LAYER_REPEATING,
    LLM_TENSOR_LAYER_OUTPUT,
};

struct LLM_KV {
    LLM_KV(llm_arch arch, const char * suffix = nullptr);

    llm_arch arch;
    const char * suffix;

    std::string operator()(llm_kv kv) const;
};

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_QWEN3);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN_IMPL {
    const llm_arch arch;
    const llm_tensor tensor;
    const char * const suffix;
    const int bid;
    const int xid;

    LLM_TN_IMPL(llm_arch arch, llm_tensor tensor, const char * suffix, int bid, int xid);

    std::string str() const;

    operator std::string() const {
        return str();
    }

    friend bool operator==(const std::string & str, const LLM_TN_IMPL & tn) {
        return str == tn.str();
    }

    friend bool operator!=(const std::string & str, const LLM_TN_IMPL & tn) {
        return str != tn.str();
    }
};

struct LLM_TN {
    LLM_TN(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    LLM_TN_IMPL operator()(llm_tensor tensor, const char * suffix, int bid = -1, int xid = -1) const {
        return LLM_TN_IMPL(arch, tensor, suffix, bid, xid);
    }

    LLM_TN_IMPL operator()(llm_tensor tensor, int bid = -1, int xid = -1) const {
        return LLM_TN_IMPL(arch, tensor, nullptr, bid, xid);
    }
};

struct llm_tensor_info {
    llm_tensor_layer layer;
    ggml_op op;
};

std::vector<llm_arch> llm_arch_all();

const char * llm_arch_name(llm_arch arch);

llm_arch llm_arch_from_string(const std::string & name);

const llm_tensor_info & llm_tensor_info_for(llm_tensor tensor);

bool llm_arch_is_recurrent      (const llm_arch & arch);
bool llm_arch_is_hybrid         (const llm_arch & arch);
bool llm_arch_is_diffusion      (const llm_arch & arch);
bool llm_arch_supports_sm_tensor(const llm_arch & arch);
