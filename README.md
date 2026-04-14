# Qwen3.cpp

Refactored Windows workflow for Qwen3 + ExecuTorch Vulkan.

## Design

- Single command entrypoint: `scripts/qwen3.ps1`
- Centralized config: `config/default.psd1`
- Runtime bundle is artifact-only: `runtime/`
- Editable runtime app layer: `vendor-runtime/`
- Editable Vulkan backend/shaders: `vendor-executorch/backends/vulkan/`

## Repository Layout

- `config/`: default and environment-specific settings
- `docs/`: baseline and architecture snapshots
- `scripts/`: entrypoint, wrappers, health checks
- `scripts/lib/`: shared PowerShell helpers
- `runtime/`: local runtime artifacts (`bin`, `models`, `tokenizer`, `prompts`, `bench`)
- `vendor-runtime/`: vendored C++ runtime app sources
- `vendor-executorch/backends/vulkan/`: vendored ExecuTorch Vulkan backend subset
- `vendor-executorch/extension/pybindings/`: vendored Python binding patches for Vulkan overlay hooks

## Quick Start

1. Setup environment:

```powershell
.\scripts\setup_env.ps1 -Recreate
```

2. Validate environment:

```powershell
.\scripts\doctor.ps1 -SkipRuntimeArtifacts
```

3. Export model from external ExecuTorch workspace:

```powershell
.\scripts\export_sdpa_dynamic.ps1
```

4. Pack runtime artifacts:

```powershell
.\scripts\pack_runtime.ps1
```

5. Run:

```powershell
.\scripts\run.ps1
```

## Python Usage

Recommended Python entry is `scripts/vk_pure_python.py`.
All commands below assume you are in repo root (`C:\Apps\Qwen3.cpp`).

### 1. Prepare Python env (uv)

```powershell
.\scripts\qwen3.ps1 env
```

Or recreate cleanly:

```powershell
.\scripts\qwen3.ps1 env -Recreate
```

### 2. Verify Vulkan backend in Python

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py check-backends
```

If output does not include `VulkanBackend`, bootstrap once:

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py bootstrap-vulkan --config-path .\config\isolated.psd1
```

Force rebuild when backend/runtime code changed:

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py bootstrap-vulkan --config-path .\config\isolated.psd1 --force-rebuild
```

### 3. Run pure Python export + inference

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py pipeline --config-path .\config\isolated.psd1
```

Common options:

- `--fusion-profile none|kv|sdpa_kv`
- `--prompt "your prompt"`
- `--max-len 128`
- `--temperature 0.0`
- `--export-only` (only export `.pte`, skip inference)

Example:

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py pipeline --config-path .\config\isolated.psd1 --fusion-profile sdpa_kv --prompt "Write one sentence about Vulkan."
```

### 4. Shader/fusion control during export

Dynamic shader overlay:

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py pipeline --config-path .\config\isolated.psd1 --export-only --shader-manifest .\runtime\vk_jit_cache\manifest.txt
```

Operator partition boundary control:

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_pure_python.py pipeline --config-path .\config\isolated.psd1 --export-only --vk-op-blocklist aten.add.Tensor --vk-op-blocklist aten.relu.default
```

### 5. Python-driven shader iteration loop (with C++ runtime rebuild)

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_jit.py run
```

Useful variants:

```powershell
.\.venv\Scripts\python.exe .\scripts\vk_jit.py run --dry-run --no-run
.\.venv\Scripts\python.exe .\scripts\vk_jit.py check-runtime
.\.venv\Scripts\python.exe .\scripts\vk_jit.py run --skip-export --skip-pack
```

## Experimental Python Vulkan JIT Runtime

If you want to bypass ET-VK AOT runtime and iterate in a Python-only JIT loop
(fusion pass + GLSL generation + runtime compile), use:

This path uses Vulkan SDK `glslc` only (no `pyshaderc` support).
It does not include a Python Vulkan dispatch backend.

```powershell
python .\scripts\vk_runtime_jit.py doctor
python .\scripts\vk_runtime_jit.py demo --noop-compiler
```

Dynamic overlay e2e smoke test:

```powershell
python .\runtime\tests\vk_overlay_e2e.py --build-dir C:\Apps\qwen3-export\build\executorch-win-vulkan-pybind-cp311
```

## C++ vs Python Perf Compare

Use unified benchmark compare (same model/prompt/token settings for both paths):

```powershell
python .\scripts\vk_perf_compare.py --config-path .\config\isolated.psd1 --runs 1 --max-new-tokens 16
```

PowerShell wrapper:

```powershell
.\scripts\perf_compare.ps1 -ConfigPath .\config\isolated.psd1 -Runs 1 -MaxNewTokens 16
```

Outputs:

- `runtime/bench_compare/runs.csv` (per-run metrics)
- `runtime/bench_compare/summary.csv` (backend aggregates)
- `runtime/bench_compare/ratio.json` (Python/C++ ratios)

## Unified Command

All wrappers call `scripts/qwen3.ps1`:

```powershell
.\scripts\qwen3.ps1 env
.\scripts\qwen3.ps1 doctor
.\scripts\qwen3.ps1 export
.\scripts\qwen3.ps1 pack
.\scripts\qwen3.ps1 run
.\scripts\qwen3.ps1 bench
.\scripts\qwen3.ps1 bench-compare
.\scripts\qwen3.ps1 chatbot-build
.\scripts\qwen3.ps1 chatbot-run
.\scripts\qwen3.ps1 vendor-sync-runtime
.\scripts\qwen3.ps1 vendor-sync-vulkan -Mode pull
.\scripts\qwen3.ps1 baseline
```

## Configuration

Defaults are in `config/default.psd1`.

Common overrides:

- `Paths.ExecuTorchRepoRoot`
- `Paths.ExecuTorchRoot`
- `Paths.ExecuTorchInstallPrefix`
- `Paths.TokenizerPath`
- `Paths.DefaultModelArtifact`

You can pass an alternate config file to any command:

```powershell
.\scripts\qwen3.ps1 run -ConfigPath .\config\my_machine.psd1
```

## Vendor Sync

Runtime sync (from external ExecuTorch llama sources):

```powershell
.\scripts\vendor_sync_runtime.ps1
```

Vulkan sync:

```powershell
.\scripts\vendor_sync_vulkan.ps1 -Mode pull
.\scripts\vendor_sync_vulkan.ps1 -Mode push
```

The pull mode keeps a curated Vulkan subset and prunes noisy directories.

## Notes

- Runtime binaries/models/tokenizer files are gitignored.
- `scripts/*.ps1` wrappers are compatibility shims; new logic lives in `scripts/qwen3.ps1`.
