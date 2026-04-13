# vendor-executorch

Vendored ExecuTorch sources used for local customization.

## Included now

- `backends/vulkan/` (curated backend subset used for shader/runtime work)

Key shader paths:

- `backends/vulkan/runtime/graph/ops/glsl/*.glsl`
- `backends/vulkan/runtime/gen_vulkan_spv.py`

## Sync workflow

From repo root (`C:\Apps\qwen3.cpp`):

```powershell
.\scripts\qwen3.ps1 vendor-sync-vulkan -Mode pull
```

This refreshes vendor copy from external ExecuTorch source (default `C:\Apps\qwen3-export\third_party\executorch`).

After editing vendored Vulkan sources, push back to upstream tree:

```powershell
.\scripts\qwen3.ps1 vendor-sync-vulkan -Mode push
```

Then rebuild external ExecuTorch Vulkan LLM binaries from the external workspace:

```powershell
cmake --build build\executorch-win-vulkan-llm-clangcl --config Release --target install -- /m
```
