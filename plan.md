# Plan: Reuse ExecuTorch Assets with Python Flexibility (Vulkan)

## 1. Goal

- Reuse the official ExecuTorch pipeline end-to-end (export, partition, fusion, runtime).
- Add Python-side control for fast iteration:
  - dynamic shader manifest injection at export time
  - operator-level partition boundary control
- Keep runtime constraints clear:
  - JIT compiles Vulkan shaders only
  - no C++ JIT
- Compiler policy: `glslc` only (no `pyshaderc` support).

## 2. Top Technical Risks

1. Python Vulkan runtime availability
   - `portable_lib` Vulkan backend can mismatch Python ABI (for example `cp311` vs `cp312`).
   - If pybindings changes live only in an external repo, this repo cannot fully reproduce/debug.

2. Dynamic shader overlay correctness
   - Compile spec must flow from export to `VulkanBackend::init`.
   - Overlay load / clear / AOT fallback behavior must be deterministic.

3. Fusion control depth
   - Fusion control is mostly export-time, not runtime-time.
   - Need explicit allow/block controls for operator partition boundaries.

## 3. Implemented Changes

### 3.1 Runtime (C++)

- Added dynamic overlay support in `ShaderRegistry`.
- Added `ShaderPackLoader` for manifest-based SPIR-V loading.
- Added Vulkan compile spec keys in backend init path:
  - `dynamic_shader_manifest_path`
  - `clear_dynamic_shader_overlay`

### 3.2 Export / Tooling (Python)

- `VulkanPartitioner.parse_compile_options` now supports string values.
- `scripts/vk_pure_python.py` now supports:
  - `--shader-manifest`
  - `--keep-dynamic-shader-overlay`
  - `--vk-op-blocklist`
  - `--vk-op-allowlist`
- Added patched export path that overrides Vulkan partitioner in Python
  (no C++ rebuild needed for this control plane).
- Refactored `bootstrap-vulkan` to deterministic CMake path:
  - build `portable_lib` and `data_loader` in ABI-matched build dir
  - install generated `.pyd` into current `.venv`
  - verify `VulkanBackend` registration after install
- Added source-sync hardening before build/export:
  - sync `vendor-executorch/backends/vulkan` -> upstream clone
  - sync `vendor-executorch/extension/pybindings` -> upstream clone
  - sync uses non-preserving-mtime copy to force rebuild when sources change
  - skip `.git` metadata during overlay copy (Windows-safe)
  - sync now runs as content-delta copy with summary stats
  - excludes high-churn/non-critical dirs (`backends/vulkan/third-party`, `test`)

### 3.3 Asset Hardening in This Repo

- Vendored key pybindings files:
  - `vendor-executorch/extension/pybindings/pybindings.cpp`
  - `vendor-executorch/extension/pybindings/portable_lib.py`
  - `vendor-executorch/extension/pybindings/pybindings.pyi`
- Kept external patch artifact for upstream sync:
  - `runtime/patches/executorch_pybind_vulkan_overlay.patch`

## 4. Execution Strategy

1. Reproducibility first:
   - Keep source/patches in this repo, not only in external workspace.
2. Control second:
   - Keep export-time knobs in Python for fast iteration.
3. Verification third:
   - Always run smoke/e2e checks (including failure-path checks).

## 5. Self-Test Checklist

1. Python script sanity
   - `py_compile` passes for modified scripts.
2. Compiler path
   - `vk_runtime_jit.py` works with `glslc` path and produces correct demo output.
3. Export control path
   - `vk_pure_python.py` dry-run shows patched partitioner path when manifest/op controls are provided.
4. Runtime overlay e2e
   - `runtime/tests/vk_overlay_e2e.py` passes:
      - invalid manifest fails as expected
      - valid manifest runs and matches numeric expectations
   - test hardened for Windows:
      - temp `.pte` cleanup uses retry/best-effort unlink
      - manifest overrides both add buffer + add texture kernels

## 6. Latest Validation (2026-04-14)

1. `uv run python scripts/vk_pure_python.py bootstrap-vulkan --config-path config/isolated.psd1 --force-rebuild` passed.
   - Verified runtime backend list contains `VulkanBackend`.
   - Verified partitioner compile-option parsing includes dynamic shader keys.
2. `uv run python runtime/tests/vk_overlay_e2e.py --build-dir C:\\Apps\\Qwen3.cpp\\upstream\\build\\executorch-win-vulkan-pybind-cp311` passed (`vk_overlay_e2e: OK`).
3. After sync refactor, `bootstrap-vulkan --force-rebuild` and `vk_overlay_e2e` re-ran and passed.

## 7. Open Items

1. Add optional fast bootstrap mode to skip full CMake reconfigure when no source deltas are detected.
2. Improve Python prefill throughput in benchmark path (currently much lower than C++ baseline in smoke run).

## 8. Perf Compare Baseline (2026-04-14)

Added:

- `scripts/vk_python_bench.py`:
  - Python-native benchmark runner with explicit prefill/decode timing.
  - Emits `PyPythonObserver { ... }` per run.
- `scripts/vk_perf_compare.py`:
  - Runs C++ (`llama_main.exe`) and Python (`vk_python_bench.py`) under same settings.
  - Emits `runs.csv`, `summary.csv`, `ratio.json`.
- `scripts/perf_compare.ps1` wrapper.
- `scripts/qwen3.ps1 bench-compare` unified entrypoint.

Smoke run:

```powershell
python .\scripts\vk_perf_compare.py --config-path .\config\isolated.psd1 --runs 1 --max-new-tokens 16 --out-dir runtime/bench_compare_smoke
```

Observed summary:

- C++: prefill `132.353` tok/s, decode `34.5622` tok/s, wall `3.654` s
- Python: prefill `3.8487` tok/s, decode `24.4769` tok/s, wall `2.9576` s
- Ratio (Python/C++): prefill `0.0291`, decode `0.7082`, wall `0.8094`
