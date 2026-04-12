# Qwen3.cpp

Minimal `llama.cpp`-style runtime for Qwen3, with local tools for:

- `PyTorch/HF -> GGUF (Q4_K_M)` export via AutoRound

## Quick Start (PyTorch -> Q4_K_M GGUF)

Default export path is **AutoRound**.

### 1) One-click uv env setup

```powershell
.\scripts\setup-uv-env.ps1 -Backend xpu
```

Or:

```cmd
scripts\setup-uv-env.cmd -Backend xpu
```

This installs dependencies from `pyproject.toml` and creates `.venv`.

### 2) Run export (no extra params)

From repo root:

```powershell
.\scripts\convert-pytorch-to-q4_k_m.ps1
```

Or:

```cmd
scripts\convert-pytorch-to-q4_k_m.cmd
```

Defaults:

- model: `Qwen/Qwen3-0.6B`
- format: `gguf:q4_k_m`
- output dir: `models\qwen3-0.6b-q4_k_m\`
- device map: `xpu`

Custom model example:

```powershell
.\scripts\convert-pytorch-to-q4_k_m.ps1 -Model "Qwen/Qwen3-1.7B"
```

## Notes

- AutoRound is now a default dependency in `pyproject.toml`.
- `convert-pytorch-to-q4_k_m.ps1` always uses AutoRound export.
- You can still pass `-PythonExe` to use an existing Python env.
- If Python env is missing, the script auto-runs `scripts/setup-uv-env.ps1`.
- You can pass a local HF model directory via `-Model "C:\path\to\model"`.

## Local Model Paths

Current local copies:

- original HF weights: `models\Qwen3-0.6B-hf\`
- quantized GGUF: `models\Qwen3-0.6B-gguf\Qwen3-0.6B-752M-Q4_K_M.gguf`

These model artifacts are intentionally excluded from git via `.gitignore`.
