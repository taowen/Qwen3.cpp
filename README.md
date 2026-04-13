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

## Unified Command

All wrappers call `scripts/qwen3.ps1`:

```powershell
.\scripts\qwen3.ps1 env
.\scripts\qwen3.ps1 doctor
.\scripts\qwen3.ps1 export
.\scripts\qwen3.ps1 pack
.\scripts\qwen3.ps1 run
.\scripts\qwen3.ps1 bench
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
