#include "llama-sampler.h"

#include "llama-impl.h"
#include "llama-vocab.h"

#include <algorithm>
#include <vector>

struct llama_sampler * llama_sampler_init(struct llama_sampler_i * iface, llama_sampler_context_t ctx) {
    return new llama_sampler {
        /* .iface = */ iface,
        /* .ctx   = */ ctx,
    };
}

const char * llama_sampler_name(const struct llama_sampler * smpl) {
    if (!smpl) {
        return "(null)";
    }

    if (!smpl->iface) {
        return "(null)";
    }

    return smpl->iface->name(smpl);
}

void llama_sampler_accept(struct llama_sampler * smpl, llama_token token) {
    if (!smpl) {
        return;
    }

    if (smpl->iface->accept) {
        smpl->iface->accept(smpl, token);
    }
}

void llama_sampler_apply(struct llama_sampler * smpl, struct llama_token_data_array * cur_p) {
    if (!smpl) {
        return;
    }

    GGML_ASSERT(smpl->iface->apply);
    smpl->iface->apply(smpl, cur_p);
}

void llama_sampler_reset(struct llama_sampler * smpl) {
    if (!smpl) {
        return;
    }

    if (smpl->iface->reset) {
        smpl->iface->reset(smpl);
    }
}

struct llama_sampler * llama_sampler_clone(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    if (smpl->iface->clone) {
        return smpl->iface->clone(smpl);
    }

    if (smpl->ctx == nullptr) {
        return llama_sampler_init(
            /* .iface = */ smpl->iface,
            /* .ctx   = */ nullptr
        );
    }

    GGML_ABORT("the sampler does not support cloning");
}

void llama_sampler_free(struct llama_sampler * smpl) {
    if (smpl == nullptr) {
        return;
    }

    if (smpl->iface->free) {
        smpl->iface->free(smpl);
    }

    delete smpl;
}

//
// sampler chain
//

static const char * llama_sampler_chain_name(const struct llama_sampler * smpl) {
    auto * ctx = (llama_sampler_chain *) smpl->ctx;

    static std::string result;
    result = "chain(";
    for (size_t i = 0; i < ctx->samplers.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += ctx->samplers[i].ptr ? llama_sampler_name(ctx->samplers[i].ptr) : "(null)";
    }
    result += ")";

    return result.c_str();
}

static void llama_sampler_chain_accept(struct llama_sampler * smpl, llama_token token) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    for (auto & cur : chain->samplers) {
        llama_sampler_accept(cur.ptr, token);
    }
}

static void llama_sampler_chain_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;

    const int64_t t_start = ggml_time_us();

    for (auto & cur : chain->samplers) {
        llama_sampler_apply(cur.ptr, cur_p);
    }

    chain->t_sample_us += ggml_time_us() - t_start;
    chain->n_sample++;
}

static void llama_sampler_chain_reset(struct llama_sampler * smpl) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    for (auto & cur : chain->samplers) {
        llama_sampler_reset(cur.ptr);
    }
}

static struct llama_sampler * llama_sampler_chain_clone(const struct llama_sampler * smpl) {
    const auto * chain_src = (const llama_sampler_chain *) smpl->ctx;
    auto * result = llama_sampler_chain_init(chain_src->params);

    auto * chain_dst = (llama_sampler_chain *) result->ctx;
    chain_dst->samplers.clear();
    for (const auto & cur : chain_src->samplers) {
        chain_dst->samplers.push_back({
            /* .is_backend = */ false,
            /* .ptr        = */ llama_sampler_clone(cur.ptr),
        });
    }

    return result;
}

static void llama_sampler_chain_free(struct llama_sampler * smpl) {
    auto * chain = (llama_sampler_chain *) smpl->ctx;
    for (auto & cur : chain->samplers) {
        llama_sampler_free(cur.ptr);
    }
    delete chain;
}

static struct llama_sampler_i llama_sampler_chain_i = {
    /* .name              = */ llama_sampler_chain_name,
    /* .accept            = */ llama_sampler_chain_accept,
    /* .apply             = */ llama_sampler_chain_apply,
    /* .reset             = */ llama_sampler_chain_reset,
    /* .clone             = */ llama_sampler_chain_clone,
    /* .free              = */ llama_sampler_chain_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params) {
    return llama_sampler_init(
        /* .iface = */ &llama_sampler_chain_i,
        /* .ctx   = */ new llama_sampler_chain {
            /* .params      = */ params,
            /* .samplers    = */ {},
            /* .cur         = */ {},
            /* .t_sample_us = */ 0,
            /* .n_sample    = */ 0,
        }
    );
}

void llama_sampler_chain_add(struct llama_sampler * chain, struct llama_sampler * smpl) {
    auto * p = (llama_sampler_chain *) chain->ctx;
    p->samplers.push_back({
        /* .is_backend = */ false,
        /* .ptr        = */ smpl,
    });
}

struct llama_sampler * llama_sampler_chain_get(struct llama_sampler * chain, int32_t i) {
    if (chain == nullptr) {
        return nullptr;
    }

    if (chain->iface != &llama_sampler_chain_i) {
        return nullptr;
    }

    if (i == -1) {
        return chain;
    }

    const auto * p = (const llama_sampler_chain *) chain->ctx;

    if (i < 0 || (size_t) i >= p->samplers.size()) {
        return nullptr;
    }

    return p->samplers[i].ptr;
}

int llama_sampler_chain_n(const struct llama_sampler * chain) {
    const auto * p = (const llama_sampler_chain *) chain->ctx;

    return p->samplers.size();
}

struct llama_sampler * llama_sampler_chain_remove(struct llama_sampler * chain, int32_t i) {
    auto * p = (llama_sampler_chain *) chain->ctx;

    if (i < 0 || (size_t) i >= p->samplers.size()) {
        return nullptr;
    }

    auto * result = p->samplers[i].ptr;
    p->samplers.erase(p->samplers.begin() + i);

    return result;
}

//
// greedy sampler
//

struct llama_sampler_greedy {
};

static const char * llama_sampler_greedy_name(const struct llama_sampler * smpl) {
    GGML_UNUSED(smpl);
    return "greedy";
}

static void llama_sampler_greedy_reset(struct llama_sampler * smpl) {
    GGML_UNUSED(smpl);
}

static struct llama_sampler * llama_sampler_greedy_clone(const struct llama_sampler * smpl) {
    GGML_UNUSED(smpl);
    return llama_sampler_init_greedy();
}

static void llama_sampler_greedy_free(struct llama_sampler * smpl) {
    delete (llama_sampler_greedy *) smpl->ctx;
}

static void llama_sampler_greedy_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    GGML_UNUSED(smpl);
    cur_p->selected = 0;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
            cur_p->selected = i;
        }
    }
}

static struct llama_sampler_i llama_sampler_greedy_i = {
    /* .name              = */ llama_sampler_greedy_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ llama_sampler_greedy_apply,
    /* .reset             = */ llama_sampler_greedy_reset,
    /* .clone             = */ llama_sampler_greedy_clone,
    /* .free              = */ llama_sampler_greedy_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct llama_sampler * llama_sampler_init_greedy() {
    return llama_sampler_init(
        /* .iface = */ &llama_sampler_greedy_i,
        /* .ctx   = */ new llama_sampler_greedy {}
    );
}

llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const float * logits = llama_get_logits_ith(ctx, idx);
    GGML_ASSERT(logits != nullptr);

    std::vector<llama_token_data> cur(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    llama_token_data_array cur_p = {
        /* .data       = */ cur.data(),
        /* .size       = */ cur.size(),
        /* .selected   = */ -1,
        /* .sorted     = */ false,
    };

    llama_sampler_apply(smpl, &cur_p);

    GGML_ASSERT(cur_p.selected >= 0 && cur_p.selected < (int32_t) cur_p.size);

    auto token = cur_p.data[cur_p.selected].id;
    llama_sampler_accept(smpl, token);

    return token;
}

struct llama_perf_sampler_data llama_perf_sampler(const struct llama_sampler * chain) {
    struct llama_perf_sampler_data data = {};

    if (chain == nullptr || chain->iface != &llama_sampler_chain_i) {
        GGML_ABORT("%s: invalid sampler passed - requires a sampler created with llama_sampler_chain_init()\n", __func__);
    }

    const auto * ctx = (const struct llama_sampler_chain *) chain->ctx;

    data.t_sample_ms = 1e-3 * ctx->t_sample_us;
    data.n_sample    = std::max(0, ctx->n_sample);

    return data;
}

void llama_perf_sampler_print(const struct llama_sampler * chain) {
    const auto data = llama_perf_sampler(chain);

    LLAMA_LOG_INFO("%s:    samplers time = %10.2f ms / %5d runs\n", __func__, data.t_sample_ms, data.n_sample);
}

void llama_perf_sampler_reset(struct llama_sampler * chain) {
    if (chain == nullptr || chain->iface != &llama_sampler_chain_i) {
        GGML_ABORT("%s: invalid sampler passed - requires a sampler created with llama_sampler_chain_init()\n", __func__);
    }

    auto * ctx = (struct llama_sampler_chain *) chain->ctx;

    ctx->t_sample_us = 0;
    ctx->n_sample    = 0;
}
