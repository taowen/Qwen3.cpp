//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "ggml-sycl/common.hpp"
#include "ggml-sycl/convert.hpp"
#include "ggml-sycl/dmmv.hpp"
#include "ggml-sycl/gemm.hpp"
#include "ggml-sycl/mmq.hpp"
#include "ggml-sycl/mmvq.hpp"
#include "ggml-sycl/mul-mat-dispatch.hpp"
#include "ggml-sycl/quantize.hpp"

// Kept layout-compatible with the split-buffer context stored in buft->context.
struct ggml_backend_sycl_split_buffer_type_context {
    std::array<float, GGML_SYCL_MAX_DEVICES> tensor_split;
};

void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i, const char * src1_ddq_i,
    float * dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr & stream);

static void dev2dev_memcpy(sycl::queue & q_dst, sycl::queue & q_src, void * ptr_dst,
                           const void * ptr_src, size_t size) {
    char * host_buf = static_cast<char *>(malloc(size));
    q_src.memcpy(host_buf, static_cast<const char *>(ptr_src), size).wait();
    q_dst.memcpy(static_cast<char *>(ptr_dst), host_buf, size).wait();
    free(host_buf);
}

static bool ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
    const char * name = buffer->buft->iface.get_name(buffer->buft);
    return std::strcmp(name, GGML_SYCL_NAME "_Split") == 0;
}

static bool ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    const char * name = buffer->buft->iface.get_name(buffer->buft);
    const size_t sycl_name_len = std::strlen(GGML_SYCL_NAME);
    return std::strncmp(name, GGML_SYCL_NAME, sycl_name_len) == 0 &&
           std::strcmp(name, GGML_SYCL_NAME "_Split") != 0;
}

static int64_t get_row_rounding(ggml_type type, const std::array<float, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t max_compute_capability = std::numeric_limits<int64_t>::min();
    for (int i = 0; i < ggml_sycl_info().device_count; ++i) {
        if (tensor_split[i] < (i + 1 < ggml_sycl_info().device_count ? tensor_split[i + 1] : 1.0f)) {
            if (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 64;
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return 1;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_IQ3_S:
            return max_compute_capability >= VER_GEN9 ? 128 : 64;
        case GGML_TYPE_Q6_K:
            return 64;
        default:
            GGML_ABORT("fatal error");
    }
}

static dpct::err0 ggml_sycl_cpy_tensor_2d(void * dst,
                                          const struct ggml_tensor * src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_low, int64_t i1_high,
                                          queue_ptr stream) try {
    dpct::memcpy_direction kind;
    char * src_ptr;
    if (ggml_backend_buffer_is_host(src->buffer)) {
        kind    = dpct::host_to_device;
        src_ptr = static_cast<char *>(src->data);
    } else if (ggml_backend_buffer_is_sycl(src->buffer)) {
        kind    = dpct::device_to_device;
        src_ptr = static_cast<char *>(src->data);
    } else if (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        GGML_ASSERT(i1_low == 0 && i1_high == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src->extra);
        int id;
        SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
        src_ptr = static_cast<char *>(extra->data_device[id]);
    } else {
        GGML_ABORT("fatal error");
    }

    char * dst_ptr = static_cast<char *>(dst);
    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);

    const ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low * nb1 + i2 * nb2 + i3 * nb3;
    if (nb0 == ts && nb1 == ts * ne0 / bs) {
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1, kind, *stream));
    } else if (nb0 == ts) {
        return CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
            dst_ptr, ts * ne0 / bs, x, nb1, ts * ne0 / bs, i1_diff, kind, *stream));
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; ++i1) {
            const void * rx = static_cast<const void *>(x + i1 * nb1);
            void * rd = static_cast<void *>(dst_ptr + i1 * ts * ne0 / bs);
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            if (r != 0) {
                return r;
            }
        }
        return 0;
    }
}
catch (sycl::exception const & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

#include "ggml-sycl/mul-mat-template.hpp"

void ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i, const char * src1_ddq_i,
    float * dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const queue_ptr & stream) try {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    const int64_t row_diff = row_high - row_low;

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));

    const int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#else
    bool use_fp16 = false;
#endif
    if ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        if (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff * ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        const sycl::half * src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (const sycl::half *) src0_dd_i
                                         : src0_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        if (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            const to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols * ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        const sycl::half * src1_ptr = src1->type == GGML_TYPE_F16
                ? (const sycl::half *) src1->data + src1_padded_row_size
                                         : src1_as_f16.get();

#if GGML_SYCL_DNNL
        if (!g_ggml_sycl_disable_dnn) {
            DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ptr,
                                      DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr,
                                      DnnlGemmWrapper::to_dt<sycl::half>(), dst_dd_i,
                                      DnnlGemmWrapper::to_dt<float>(), stream);
        } else
#endif
        {
            ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool(), row_diff * src1_ncols);

            const sycl::half alpha_f16 = 1.0f;
            const sycl::half beta_f16  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha_f16, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta_f16,
                dst_f16.get(), dpct::library_data_t::real_half, ldc,
                dpct::library_data_t::real_half)));
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting dst to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(GGML_TYPE_F16, dst);
            to_fp32_sycl(dst_f16.get(), dst_dd_i, row_diff * src1_ncols, stream);
        }
    } else {
        ggml_sycl_pool_alloc<float> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<float> src1_ddq_as_f32(ctx.pool());
        if (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff * ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff * ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            const to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols * ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols * ne10, stream);
        }
        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

#if GGML_SYCL_DNNL
        if (!g_ggml_sycl_disable_dnn) {
            DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                      DnnlGemmWrapper::to_dt<float>(), src1_ddf1_i,
                                      DnnlGemmWrapper::to_dt<float>(), dst_dd_i,
                                      DnnlGemmWrapper::to_dt<float>(), stream);
        } else
#endif
        {
            const float alpha = 1.0f;
            const float beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                src1_ncols, ne10, dpct::get_value(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                dpct::get_value(&beta, *stream), dst_dd_i, ldc)));
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
catch (sycl::exception const & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

void ggml_sycl_mul_mat_run_dmmv(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {
    ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
}

void ggml_sycl_mul_mat_run_mmvq(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst,
    bool use_reordered_quant) {
    if (use_reordered_quant) {
        ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
    } else {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
    }
}

void ggml_sycl_mul_mat_run_mmq(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {
    ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
}

void ggml_sycl_mul_mat_run_sycl(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst) {
    ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
}
