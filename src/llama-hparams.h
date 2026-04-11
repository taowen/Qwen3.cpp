#pragma once

#include "llama.h"

#include <array>
#include <cassert>

// bump if necessary
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 512 // Qwen3 Next

enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
};

struct llama_hparams {
    bool vocab_only;
    bool no_alloc;
    bool rope_finetuned;
    bool use_par_res;
    bool swin_norm;

    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_layer;
    int32_t n_layer_kv_from_start = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_rel_attn_bkts = 0;

    // different head size for full_attention and SWA layers
    uint32_t n_embd_head_k_full; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v_full; // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;

    // different RoPE dimensions for full_attention and SWA layers
    uint32_t n_rot_full;
    uint32_t n_rot_swa;

    // note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
    uint32_t n_embd_head_k_mla_impl = 0;
    uint32_t n_embd_head_v_mla_impl = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;
    float f_attn_logit_softcapping   = 50.0f;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa  = 10000.0f;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa = 1.0f;

    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul = 0.0f;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;

    std::array<int, 4> rope_sections;

    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;
    // if swa_layers[il] == 1, then layer il is SWA
    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_layers;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;
    bool use_kq_norm   = false;

    uint32_t n_cls_out = 1;
    uint32_t n_embd_out_impl = 0;

    // qwen3vl deepstack
    uint32_t n_deepstack_layers = 0;


    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    uint32_t n_gqa(uint32_t il = 0) const;

    uint32_t n_rot(uint32_t il = 0) const;

    // dimension of main + auxiliary input embeddings
    uint32_t n_embd_inp() const;

    // dimension of output embeddings
    uint32_t n_embd_out() const;

    // dimension of key/value embeddings for each head (per layer)
    uint32_t n_embd_head_k(uint32_t il = 0) const;
    uint32_t n_embd_head_v(uint32_t il = 0) const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    // true if any layer has a different n_embd_k_gqa/n_embd_v_gqa
    bool is_n_embd_k_gqa_variable() const;
    bool is_n_embd_v_gqa_variable() const;

    // return the maximum n_embd_k_gqa/n_embd_v_gqa across all layers
    uint32_t n_embd_k_gqa_max() const;
    uint32_t n_embd_v_gqa_max() const;

    uint32_t n_pos_per_embd() const;
    bool is_swa(uint32_t il) const;
    bool is_swa_any() const;

    // note: currently only support if either all or none of the layers are MLA
    bool is_mla() const;

    uint32_t n_embd_head_k_mla() const;
    uint32_t n_embd_head_v_mla() const;

    bool has_kv(uint32_t il) const;

    // number of layers for which has_kv() returns true
    uint32_t n_layer_kv() const;

    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
        assert(p0 >= 0 && p1 >= 0);

        switch (swa_type) {
            case LLAMA_SWA_TYPE_NONE:
                {
                } break;
            case LLAMA_SWA_TYPE_STANDARD:
                {
                    if (p1 - p0 >= (int32_t) n_swa) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_CHUNKED:
                {
                    const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                    if (p0 < pos_chunk_start) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                {
                    const int32_t half_n_swa = (int32_t) n_swa / 2;
                    const int32_t pos_diff = p1 - p0;

                    // Mask if outside the symmetric window
                    if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                        return true;
                    }
                } break;
        }

        return false;
    }

    bool use_mrope() const;
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
