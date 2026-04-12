#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-opt.h"

extern "C" {

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    GGML_UNUSED(buf);
    return false;
}

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(buft);
    return nullptr;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    GGML_UNUSED(opt_ctx);
}

}
