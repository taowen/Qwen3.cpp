#include "llama-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

bool llama_hparams::is_swa_any() const {
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (swa_layers[il]) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head/n_head_kv;
}

uint32_t llama_hparams::n_rot(uint32_t il) const {
    if (il < n_layer) {
        return is_swa(il) ? n_rot_swa : n_rot_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_inp() const {
    uint32_t n_embd_inp = n_embd;

    if (n_deepstack_layers > 0) {
        n_embd_inp += n_embd * n_deepstack_layers;
    }

    return n_embd_inp;
}

uint32_t llama_hparams::n_embd_out() const {
    return n_embd_out_impl > 0 ? n_embd_out_impl : n_embd;
}

uint32_t llama_hparams::n_embd_head_k(uint32_t il) const {
    if (il < n_layer) {
        return is_swa(il) ? n_embd_head_k_swa : n_embd_head_k_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_head_v(uint32_t il) const {
    if (il < n_layer) {
        return is_swa(il) ? n_embd_head_v_swa : n_embd_head_v_full;
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k(il) * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_v(il) * n_head_kv;
}

bool llama_hparams::is_n_embd_k_gqa_variable() const {
    const uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (val != n_embd_k_gqa(il)) {
            return true;
        }
    }

    return false;
}

bool llama_hparams::is_n_embd_v_gqa_variable() const {
    const uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (val != n_embd_v_gqa(il)) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_embd_k_gqa_max() const {
    uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        val = std::max(val, n_embd_k_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_v_gqa_max() const {
    uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        val = std::max(val, n_embd_v_gqa(il));
    }

    return val;
}

bool llama_hparams::is_mla() const {
    assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
           (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}

uint32_t llama_hparams::n_embd_head_k_mla() const {
    return is_mla() ? n_embd_head_k_mla_impl : n_embd_head_k();
}

uint32_t llama_hparams::n_embd_head_v_mla() const {
    return is_mla() ? n_embd_head_v_mla_impl : n_embd_head_v();
}

bool llama_hparams::has_kv(uint32_t il) const {
    if (n_layer_kv_from_start >= 0) {
        if (il < (uint32_t) n_layer_kv_from_start) {
            return true;
        }

        return false;
    }

    // by default, all layers have kv
    return true;
}

uint32_t llama_hparams::n_layer_kv() const {
    uint32_t res = 0;

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (has_kv(il)) {
            res++;
        }
    }

    return res;
}
uint32_t llama_hparams::n_pos_per_embd() const {
    return rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
}

bool llama_hparams::is_swa(uint32_t il) const {
    if (il < n_layer) {
        return swa_layers[il];
    }

    GGML_ABORT("fatal error");
}
bool llama_hparams::use_mrope() const {
    return rope_sections[0] > 0 && rope_sections[1] > 0;
}
