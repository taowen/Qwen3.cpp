#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

static void print_usage(const char * prog) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-p prompt] [-t temperature] [prompt]\n", prog);
    printf("\n");
}

#ifdef _WIN32
static std::string utf8_from_wide(const wchar_t * wstr) {
    if (wstr == nullptr) {
        return {};
    }

    const int size = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        return {};
    }

    std::string out(size, '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), size, nullptr, nullptr) <= 0) {
        return {};
    }

    out.pop_back(); // remove trailing '\0'
    return out;
}
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::vector<std::string> args;
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);

    if (__wargv != nullptr && __argc > 0) {
        args.reserve(__argc);
        for (int i = 0; i < __argc; ++i) {
            args.push_back(utf8_from_wide(__wargv[i]));
        }
    }
#endif
    if (args.empty()) {
        args.reserve(argc);
        for (int i = 0; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }
    }

    const int argc_u = static_cast<int>(args.size());

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    std::string prompt = "Hello my name is";
    bool prompt_set = false;
    // number of tokens to predict
    int n_predict = 32;
    // sampling temperature (<= 0 means greedy)
    float temperature = 0.0f;

    // parse command line arguments
    {
        int i = 1;
        for (; i < argc_u; i++) {
            const std::string & arg = args[i];
            if (arg == "-m" || arg == "--model") {
                if (i + 1 < argc_u) {
                    model_path = args[++i];
                } else {
                    print_usage(args[0].c_str());
                    return 1;
                }
            } else if (arg == "-n" || arg == "--n-predict") {
                if (i + 1 < argc_u) {
                    try {
                        n_predict = std::stoi(args[++i]);
                    } catch (...) {
                        print_usage(args[0].c_str());
                        return 1;
                    }
                } else {
                    print_usage(args[0].c_str());
                    return 1;
                }
            } else if (arg == "-p" || arg == "--prompt") {
                if (i + 1 < argc_u) {
                    prompt = args[++i];
                    prompt_set = true;
                } else {
                    print_usage(args[0].c_str());
                    return 1;
                }
            } else if (arg == "-t" || arg == "--temp" || arg == "--temperature") {
                if (i + 1 < argc_u) {
                    try {
                        temperature = std::stof(args[++i]);
                    } catch (...) {
                        print_usage(args[0].c_str());
                        return 1;
                    }
                } else {
                    print_usage(args[0].c_str());
                    return 1;
                }
            } else if (arg == "-h" || arg == "--help") {
                print_usage(args[0].c_str());
                return 0;
            } else if (!arg.empty() && arg[0] == '-') {
                fprintf(stderr, "unknown argument: %s\n", arg.c_str());
                print_usage(args[0].c_str());
                return 1;
            } else {
                // positional prompt starts here
                prompt = arg;
                prompt_set = true;
                for (++i; i < argc_u; i++) {
                    prompt += " ";
                    prompt += args[i];
                }
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(args[0].c_str());
            return 1;
        }
        if (!prompt_set) {
            prompt = "Hello my name is";
        }
        if (n_predict < 1) {
            fprintf(stderr, "n_predict must be >= 1\n");
            return 1;
        }
    if (temperature < 0.0f) {
        fprintf(stderr, "temperature must be >= 0\n");
        return 1;
    }
    if (temperature > 0.0f) {
        fprintf(stderr, "temperature sampling is not available in this minimal build\n");
        return 1;
    }
    }

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU-only path

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // apply a minimal ChatML prompt format for Qwen chat models
    const std::string prompt_chat =
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt_chat.c_str(), prompt_chat.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt_chat.c_str(), prompt_chat.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    ctx_params.n_batch = n_prompt;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
