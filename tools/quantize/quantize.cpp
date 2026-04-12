#include "llama.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

static void usage(const char * prog) {
    std::printf("usage: %s model-f16.gguf model-q4_k_m.gguf [Q4_K_M] [nthreads]\n", prog);
    std::printf("notes:\n");
    std::printf("  - default type is Q4_K_M\n");
    std::printf("  - accepted type aliases: Q4_K_M, Q4_K, 15\n");
}

static bool is_integer(const std::string & s) {
    if (s.empty()) {
        return false;
    }
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') {
        i = 1;
    }
    if (i >= s.size()) {
        return false;
    }
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
            return false;
        }
    }
    return true;
}

static std::string to_upper(std::string s) {
    for (char & c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

static bool parse_ftype(const std::string & v, llama_ftype & out) {
    const std::string t = to_upper(v);
    if (t == "Q4_K_M" || t == "Q4_K" || t == "15") {
        out = LLAMA_FTYPE_MOSTLY_Q4_K_M;
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }

    const std::string inp = argv[1];
    const std::string out = argv[2];

    llama_ftype ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
    int nthreads = 0;

    if (argc >= 4) {
        const std::string arg3 = argv[3];
        if (is_integer(arg3)) {
            nthreads = std::stoi(arg3);
        } else if (!parse_ftype(arg3, ftype)) {
            std::fprintf(stderr, "unsupported quant type: %s\n", arg3.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (argc >= 5) {
        const std::string arg4 = argv[4];
        if (!is_integer(arg4)) {
            std::fprintf(stderr, "invalid nthreads: %s\n", arg4.c_str());
            usage(argv[0]);
            return 1;
        }
        nthreads = std::stoi(arg4);
    }

    if (argc > 5) {
        usage(argv[0]);
        return 1;
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.ftype = ftype;
    params.nthread = nthreads;

    std::printf("quantizing: %s -> %s (type=Q4_K_M, nthreads=%d)\n", inp.c_str(), out.c_str(), params.nthread);

    const uint32_t rc = llama_model_quantize(inp.c_str(), out.c_str(), &params);

    if (rc != 0) {
        std::fprintf(stderr, "quantization failed with code %u\n", rc);
        return 1;
    }

    std::printf("done\n");
    return 0;
}
