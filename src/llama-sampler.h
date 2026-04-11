#pragma once

#include "llama.h"

#include <vector>

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    struct info {
        bool is_backend;
        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing
    mutable int64_t t_sample_us = 0;
    mutable int32_t n_sample = 0;
};
