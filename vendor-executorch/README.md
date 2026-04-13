# vendor-executorch

Vendored ExecuTorch sources used for local customization.

## Included now

- `backends/vulkan/` (full backend snapshot, including shader GLSL and SPIR-V codegen script)

Key shader paths:

- `backends/vulkan/runtime/graph/ops/glsl/*.glsl`
- `backends/vulkan/runtime/gen_vulkan_spv.py`

## Sync workflow

From repo root (`C:\Apps\qwen3.cpp`):

```powershell
.\scripts\vendor_sync_vulkan.ps1 -Mode pull
```

This refreshes vendor copy from external ExecuTorch source (default `C:\Apps\qwen3-export\third_party\executorch`).

After editing vendored Vulkan sources, push back to build tree:

```powershell
.\scripts\vendor_sync_vulkan.ps1 -Mode push
```

Then rebuild ExecuTorch Vulkan LLM binaries from repo root:

```powershell
cmake --build build\executorch-win-vulkan-llm-clangcl --config Release --target install -- /m
```
