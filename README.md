# Qwen3.cpp

Windows-focused repo for ExecuTorch-based Qwen3 workflow:

- export uses upstream ExecuTorch (Python)
- runtime app layer is vendored locally (C++)
- Vulkan backend/shaders are vendored locally for custom optimization

## Repository layout

- `scripts/`: export, pack, benchmark, sync, and chatbot helper scripts
- `runtime/`: local runtime bundle workspace (`bin`, `models`, `tokenizer`, `prompts`)
- `vendor-runtime/`: vendored runner/tokenizer/chatbot C++ app sources
- `vendor-executorch/backends/vulkan/`: vendored ExecuTorch Vulkan backend + shaders

## Quick start

1. Setup Python env:

```powershell
.\scripts\setup_env.ps1 -Recreate
```

2. Put runtime artifacts into:
   - `runtime/bin/llama_main.exe`
   - `runtime/models/<model>.pte`
   - `runtime/tokenizer/tokenizer.json`
2. Run:

```powershell
.\scripts\run.ps1
```

## Export (dynamic-shape + SDPA with KV cache)

Export uses local `.venv` plus external ExecuTorch workspace source (default `C:\Apps\qwen3-export`):

```powershell
.\scripts\export_sdpa_dynamic.ps1
```

## Vendor runtime app build

```powershell
.\scripts\build_chatbot.ps1
.\scripts\run_chatbot.ps1
```

## Vendor sync helpers

- Sync llama runner/tokenizer from upstream ExecuTorch:

```powershell
.\scripts\vendor_sync_runtime.ps1
```

- Sync Vulkan backend:

```powershell
.\scripts\vendor_sync_vulkan.ps1 -Mode pull
.\scripts\vendor_sync_vulkan.ps1 -Mode push
```

## Notes

- Large runtime artifacts are intentionally gitignored.
- Keep model/tokenizer/runner binaries local under `runtime/` and do not commit them.
