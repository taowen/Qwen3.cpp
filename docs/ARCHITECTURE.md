# Architecture

## Principles

1. Keep operational defaults in one config file.
2. Keep script behavior in one entrypoint.
3. Keep vendor code editable but scoped.
4. Keep runtime artifacts outside source control.

## Command Flow

- Wrappers under `scripts/*.ps1` call `scripts/qwen3.ps1`.
- `qwen3.ps1` loads `config/default.psd1` (or override config).
- Common helpers are implemented in `scripts/lib/common.ps1`.

## Boundaries

- Export path: external ExecuTorch workspace.
- Runtime app: local `vendor-runtime`.
- Vulkan backend customization: local `vendor-executorch/backends/vulkan`.

## CI Scope

- Environment lockfile validation (`uv sync --locked`)
- Doctor checks (`-SkipExternal -SkipRuntimeArtifacts`)
- Command dry-runs for critical paths
