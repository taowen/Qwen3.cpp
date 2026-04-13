/*
 * Vendor runtime chatbot entry for qwen3.cpp.
 *
 * This keeps llama runner code local so you can customize behavior without
 * editing third_party/executorch.
 */

#include "runner.h"

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/runtime/core/error.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model_path;
  std::string tokenizer_path;
  int max_new_tokens = 160;
  float temperature = 0.0f;
  bool ignore_eos = true;
};

void print_usage(const char* exe) {
  std::cout
      << "Usage: " << exe << " --model_path <model.pte> --tokenizer_path <tokenizer.json> [options]\n"
      << "Options:\n"
      << "  --max_new_tokens <int>   default: 160\n"
      << "  --temperature <float>    default: 0\n"
      << "  --ignore_eos <0|1>       default: 1\n"
      << "  --help\n\n"
      << "Runtime commands:\n"
      << "  /exit   quit chatbot\n"
      << "  /reset  reset KV cache / runner state\n";
}

bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need_val = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (k == "--model_path") {
      args.model_path = need_val("--model_path");
    } else if (k == "--tokenizer_path") {
      args.tokenizer_path = need_val("--tokenizer_path");
    } else if (k == "--max_new_tokens") {
      args.max_new_tokens = std::stoi(need_val("--max_new_tokens"));
    } else if (k == "--temperature") {
      args.temperature = std::stof(need_val("--temperature"));
    } else if (k == "--ignore_eos") {
      args.ignore_eos = std::stoi(need_val("--ignore_eos")) != 0;
    } else if (k == "--help" || k == "-h") {
      print_usage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown arg: " << k << "\n";
      print_usage(argv[0]);
      return false;
    }
  }

  if (args.model_path.empty() || args.tokenizer_path.empty()) {
    print_usage(argv[0]);
    return false;
  }

  if (args.max_new_tokens <= 0) {
    std::cerr << "--max_new_tokens must be > 0\n";
    return false;
  }

  return true;
}

} // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return args.model_path.empty() || args.tokenizer_path.empty() ? 2 : 0;
  }

  auto runner = example::create_llama_runner(
      args.model_path,
      args.tokenizer_path,
      std::vector<std::string>{},
      args.temperature,
      nullptr,
      "forward");

  if (!runner) {
    std::cerr << "Failed to create llama runner\n";
    return 1;
  }

  std::cout << "qwen3.cpp chatbot ready. Type /exit to quit, /reset to reset KV cache.\n";

  std::string line;
  while (true) {
    std::cout << "\nYou> ";
    if (!std::getline(std::cin, line)) {
      break;
    }

    if (line == "/exit") {
      break;
    }
    if (line == "/reset") {
      runner->reset();
      std::cout << "[state reset]\n";
      continue;
    }
    if (line.empty()) {
      continue;
    }

    executorch::extension::llm::GenerationConfig cfg{.temperature = args.temperature};
    cfg.max_new_tokens = args.max_new_tokens;
    cfg.ignore_eos = args.ignore_eos;

    std::cout << "Assistant> ";
    const auto err = runner->generate(line, cfg);
    if (err != executorch::runtime::Error::Ok) {
      std::cerr << "\n[generate failed, error=" << static_cast<int>(err) << "]\n";
    }
  }

  return 0;
}
