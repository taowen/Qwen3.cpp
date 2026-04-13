# Refactor Plan

## Phase 0: Baseline Freeze

- [x] Tag baseline commit as `baseline-pre-refactor`
- [x] Snapshot baseline metadata in `docs/BASELINE.md`

## Phase 1: Configuration Center

- [x] Create `config/default.psd1`
- [x] Move path/build/runtime defaults out of ad-hoc scripts

## Phase 2: Script Library

- [x] Add `scripts/lib/common.ps1` with shared helpers
- [x] Replace repeated path/command/check logic

## Phase 3: Unified Entrypoint

- [x] Add `scripts/qwen3.ps1`
- [x] Convert old scripts into compatibility wrappers

## Phase 4: Structure + Vendor Convergence

- [x] Keep `runtime/` as artifact zone
- [x] Add curated vendor sync behavior for Vulkan backend
- [x] Keep runtime and Vulkan customization boundaries explicit

## Phase 5: Quality Gates

- [x] Add doctor command (`scripts/doctor.ps1`)
- [x] Add baseline command wrapper (`scripts/freeze_baseline.ps1`)
- [x] Add Windows CI workflow (`.github/workflows/windows-ci.yml`)
