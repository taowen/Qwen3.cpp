@{
  Paths = @{
    ExecuTorchRepoRoot = "C:/Apps/qwen3-export"
    ExecuTorchBuildDir = "build/executorch-win-vulkan-llm-clangcl"
    ExecuTorchInstallPrefix = "C:/Apps/qwen3-export/build/executorch-win-vulkan-llm-clangcl/install"
    ExecuTorchRoot = "C:/Apps/qwen3-export/third_party/executorch"
    RuntimeRoot = "runtime"
    RuntimePromptFile = "runtime/prompts/perf_prompt.txt"
    DefaultModelArtifact = "qwen3_0_6b_vulkan_8da4w_kv_sdpa_dynamic.pte"
    TokenizerPath = "C:/Users/taowen/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json"
    RunnerRelativePath = "examples/models/llama/Release/llama_main.exe"
    PromptSourceFile = "perf_prompt.txt"
  }

  Export = @{
    ModelId = "qwen3_0_6b"
    ParamsRelativePath = "third_party/executorch/examples/models/qwen3/config/0_6b_config.json"
    QuantMode = "8da4w"
    GroupSize = 128
    EnableVulkan = $true
    EnableKVCache = $true
    EnableSDPAWithKVCache = $true
  }

  Runtime = @{
    DefaultMaxNewTokens = 80
    DefaultTemperature = 0.0
    DefaultIgnoreEos = $true
  }

  Bench = @{
    DefaultRuns = 3
  }

  Build = @{
    ChatbotBuildDir = "build/qwen3-chatbot-clangcl"
    Generator = "Visual Studio 17 2022"
    Toolset = "ClangCL"
  }

  Python = @{
    Version = "3.11"
  }
}
