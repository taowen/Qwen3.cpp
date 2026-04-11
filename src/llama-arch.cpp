#include "llama-arch.h"

#include "llama-impl.h"

#include <map>
#include <vector>

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_QWEN3,            "qwen3"            },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_ARCHITECTURE,             "general.architecture"                  },
    { LLM_KV_GENERAL_FILE_TYPE,                "general.file_type"                     },
    { LLM_KV_GENERAL_NAME,                     "general.name"                          },

    { LLM_KV_VOCAB_SIZE,                        "%s.vocab_size"                        },
    { LLM_KV_CONTEXT_LENGTH,                    "%s.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "%s.embedding_length"                  },
    { LLM_KV_EMBEDDING_LENGTH_OUT,              "%s.embedding_length_out"              },
    { LLM_KV_BLOCK_COUNT,                       "%s.block_count"                       },
    { LLM_KV_FEED_FORWARD_LENGTH,               "%s.feed_forward_length"               },
    { LLM_KV_EXPERT_COUNT,                      "%s.expert_count"                      },
    { LLM_KV_EXPERT_USED_COUNT,                 "%s.expert_used_count"                 },
    { LLM_KV_POOLING_TYPE,                      "%s.pooling_type"                      },
    { LLM_KV_CLASSIFIER_OUTPUT_LABELS,          "%s.classifier.output_labels"          },

    { LLM_KV_ATTENTION_HEAD_COUNT,                   "%s.attention.head_count"                   },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,                "%s.attention.head_count_kv"                },
    { LLM_KV_ATTENTION_KEY_LENGTH,                   "%s.attention.key_length"                   },
    { LLM_KV_ATTENTION_VALUE_LENGTH,                 "%s.attention.value_length"                 },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,            "%s.attention.layer_norm_rms_epsilon"       },
    { LLM_KV_ATTENTION_CAUSAL,                       "%s.attention.causal"                       },
    { LLM_KV_ATTENTION_KEY_LENGTH_SWA,               "%s.attention.key_length_swa"               },
    { LLM_KV_ATTENTION_VALUE_LENGTH_SWA,             "%s.attention.value_length_swa"             },

    { LLM_KV_ROPE_DIMENSION_COUNT,           "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_DIMENSION_COUNT_SWA,       "%s.rope.dimension_count_swa"             },
    { LLM_KV_ROPE_FREQ_BASE,                 "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_SCALE_LINEAR,              "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,              "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,            "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,       "%s.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,      "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,         "%s.rope.scaling.finetuned"               },

    { LLM_KV_SPLIT_NO,            "split.no"            },
    { LLM_KV_SPLIT_COUNT,         "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT, "split.tensors.count" },

    { LLM_KV_TOKENIZER_MODEL,                "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                  "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                 "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,           "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,     "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,               "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,               "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,               "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,               "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_EOT_ID,               "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,               "tokenizer.ggml.eom_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,               "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,               "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,               "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_MASK_ID,              "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,              "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,              "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_SEP,              "tokenizer.ggml.add_sep_token"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,           "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,      "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_FIM_PRE_ID,           "tokenizer.ggml.fim_pre_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SUF_ID,           "tokenizer.ggml.fim_suf_token_id"         },
    { LLM_KV_TOKENIZER_FIM_MID_ID,           "tokenizer.ggml.fim_mid_token_id"         },
    { LLM_KV_TOKENIZER_FIM_PAD_ID,           "tokenizer.ggml.fim_pad_token_id"         },
    { LLM_KV_TOKENIZER_FIM_REP_ID,           "tokenizer.ggml.fim_rep_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SEP_ID,           "tokenizer.ggml.fim_sep_token_id"         },

    // deprecated
    { LLM_KV_TOKENIZER_PREFIX_ID, "tokenizer.ggml.prefix_token_id" },
    { LLM_KV_TOKENIZER_SUFFIX_ID, "tokenizer.ggml.suffix_token_id" },
    { LLM_KV_TOKENIZER_MIDDLE_ID, "tokenizer.ggml.middle_token_id" },
};

static const std::map<llm_tensor, const char *> LLM_TENSOR_NAMES = {
    { LLM_TENSOR_TOKEN_EMBD,                             "token_embd" },
    { LLM_TENSOR_OUTPUT_NORM,                            "output_norm" },
    { LLM_TENSOR_OUTPUT,                                 "output" },
    { LLM_TENSOR_ATTN_NORM,                              "blk.%d.attn_norm" },
    { LLM_TENSOR_ATTN_Q,                                 "blk.%d.attn_q" },
    { LLM_TENSOR_ATTN_K,                                 "blk.%d.attn_k" },
    { LLM_TENSOR_ATTN_V,                                 "blk.%d.attn_v" },
    { LLM_TENSOR_ATTN_OUT,                               "blk.%d.attn_output" },
    { LLM_TENSOR_FFN_NORM,                               "blk.%d.ffn_norm" },
    { LLM_TENSOR_FFN_GATE,                               "blk.%d.ffn_gate" },
    { LLM_TENSOR_FFN_DOWN,                               "blk.%d.ffn_down" },
    { LLM_TENSOR_FFN_UP,                                 "blk.%d.ffn_up" },
    { LLM_TENSOR_ATTN_Q_NORM,                            "blk.%d.attn_q_norm" },
    { LLM_TENSOR_ATTN_K_NORM,                            "blk.%d.attn_k_norm" },
    { LLM_TENSOR_CLS_OUT,                                "cls.output" },
};

static const std::map<llm_tensor, llm_tensor_info> LLM_TENSOR_INFOS = {
    {LLM_TENSOR_TOKEN_EMBD,                 {LLM_TENSOR_LAYER_INPUT,     GGML_OP_GET_ROWS}},
    {LLM_TENSOR_OUTPUT_NORM,                {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL}},
    {LLM_TENSOR_OUTPUT,                     {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_K,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_OUT,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_NORM,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_NORM,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_Q_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_K_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_CLS_OUT,                    {LLM_TENSOR_LAYER_OUTPUT,    GGML_OP_MUL_MAT}},
};

LLM_KV::LLM_KV(llm_arch arch, const char * suffix) : arch(arch), suffix(suffix) {}

std::string LLM_KV::operator()(llm_kv kv) const {
    std::string name = ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));

    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

LLM_TN_IMPL::LLM_TN_IMPL(llm_arch arch, llm_tensor tensor, const char * suffix, int bid, int xid)
    : arch(arch), tensor(tensor), suffix(suffix), bid(bid), xid(xid) {}

std::string LLM_TN_IMPL::str() const {
    if (LLM_TENSOR_NAMES.find(tensor) == LLM_TENSOR_NAMES.end()) {
        GGML_ABORT("unknown tensor name for tensor id %d", static_cast<int>(tensor));
    }

    std::string name = ::format(LLM_TENSOR_NAMES.at(tensor), bid, xid);
    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

std::vector<llm_arch> llm_arch_all() {
    std::vector<llm_arch> ret;
    ret.reserve(LLM_ARCH_NAMES.size());
    for (const auto & [arch, _] : LLM_ARCH_NAMES) {
        ret.push_back(arch);
    }
    return ret;
}

const char * llm_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

const llm_tensor_info & llm_tensor_info_for(llm_tensor tensor) {
    return LLM_TENSOR_INFOS.at(tensor);
}

bool llm_arch_is_recurrent(const llm_arch & arch) {
    switch (arch) {
        default:
            return false;
    }
}

bool llm_arch_is_hybrid(const llm_arch & arch) {
    switch (arch) {
        default:
            return false;
    }
}

bool llm_arch_is_diffusion(const llm_arch & arch) {
    switch (arch) {
        default:
            return false;
    }
}

bool llm_arch_supports_sm_tensor(const llm_arch & arch) {
    switch (arch) {
        default:
            return true;
    }
}
