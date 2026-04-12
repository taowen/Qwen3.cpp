#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uintptr_t ggml_stub_unreachable(const char * name) {
    fprintf(stderr, "qwen3-cpp stub hit: %s\n", name);
    abort();
    return 0;
}

#define GGML_STUB(name) uintptr_t name(void) { return ggml_stub_unreachable(#name); }

GGML_STUB(ggml_norm)
GGML_STUB(ggml_pad)
GGML_STUB(ggml_print_backtrace)
GGML_STUB(ggml_quantize_chunk)
GGML_STUB(ggml_soft_max)
GGML_STUB(ggml_soft_max_add_sinks)
GGML_STUB(ggml_soft_max_ext)
