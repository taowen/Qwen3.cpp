//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_MUL_MAT_DISPATCH_HPP
#define GGML_SYCL_MUL_MAT_DISPATCH_HPP

#include "common.hpp"

void ggml_sycl_mul_mat_run_dmmv(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

void ggml_sycl_mul_mat_run_mmvq(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    bool use_reordered_quant);

void ggml_sycl_mul_mat_run_mmq(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

void ggml_sycl_mul_mat_run_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

#endif // GGML_SYCL_MUL_MAT_DISPATCH_HPP
