#pragma once

#include "ggml.h"

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE std::hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cpy(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif