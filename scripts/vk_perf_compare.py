#!/usr/bin/env python3
"""Compare C++ llama_main Vulkan runtime vs Python native runner metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RepoConfig:
    repo_root: Path
    runtime_root: Path
    cpp_model_path: Path
    cpp_tokenizer_path: Path
    prompt_path: Path
    python_model_id: str
    python_params_path: Path
    python_tokenizer_path: Path
    python_tokenizer_config_path: Path | None
    default_runs: int
    default_max_new_tokens: int
    default_temperature: float


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_checked(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _load_repo_config(repo_root: Path, config_path: str) -> RepoConfig:
    common_ps1 = repo_root / "scripts" / "lib" / "common.ps1"
    ps_script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f". {_ps_quote(str(common_ps1))}",
            f"$repoRoot = {_ps_quote(str(repo_root))}",
            f"$cfgBundle = Import-Qwen3Config -RepoRoot $repoRoot -ConfigPath {_ps_quote(config_path)}",
            "$cfg = $cfgBundle.Data",
            "$runtimeRoot = Resolve-FullPath -Path $cfg.Paths.RuntimeRoot -BaseDir $repoRoot",
            "$upstreamRepo = Resolve-FullPath -Path $cfg.Paths.ExecuTorchRepoRoot -BaseDir $repoRoot",
            "$cppModelPath = Join-Path $runtimeRoot ('models\\' + $cfg.Paths.DefaultModelArtifact)",
            "$cppTokenizerPath = Join-Path $runtimeRoot 'tokenizer\\tokenizer.json'",
            "$promptPath = Resolve-FullPath -Path $cfg.Paths.RuntimePromptFile -BaseDir $repoRoot",
            "$pyTok = Resolve-FullPath -Path $cfg.Paths.TokenizerPath -BaseDir $repoRoot",
            "$pyTokCfg = Join-Path (Split-Path -Parent $pyTok) 'tokenizer_config.json'",
            "if (!(Test-Path $pyTokCfg)) { $pyTokCfg = '' }",
            "$pyParams = Join-Path $upstreamRepo $cfg.Export.ParamsRelativePath",
            "$obj = [pscustomobject]@{",
            "  repo_root = $repoRoot",
            "  runtime_root = $runtimeRoot",
            "  cpp_model_path = $cppModelPath",
            "  cpp_tokenizer_path = $cppTokenizerPath",
            "  prompt_path = $promptPath",
            "  python_model_id = $cfg.Export.ModelId",
            "  python_params_path = $pyParams",
            "  python_tokenizer_path = $pyTok",
            "  python_tokenizer_config_path = $pyTokCfg",
            "  default_runs = [int]$cfg.Bench.DefaultRuns",
            "  default_max_new_tokens = [int]$cfg.Runtime.DefaultMaxNewTokens",
            "  default_temperature = [double]$cfg.Runtime.DefaultTemperature",
            "}",
            "$obj | ConvertTo-Json -Compress",
        ]
    )
    proc = _run_checked(
        ["powershell", "-NoProfile", "-Command", ps_script],
        cwd=repo_root,
    )
    data = json.loads(proc.stdout)
    tok_cfg = data["python_tokenizer_config_path"] or None
    cfg = RepoConfig(
        repo_root=Path(data["repo_root"]),
        runtime_root=Path(data["runtime_root"]),
        cpp_model_path=Path(data["cpp_model_path"]),
        cpp_tokenizer_path=Path(data["cpp_tokenizer_path"]),
        prompt_path=Path(data["prompt_path"]),
        python_model_id=str(data["python_model_id"]),
        python_params_path=Path(data["python_params_path"]),
        python_tokenizer_path=Path(data["python_tokenizer_path"]),
        python_tokenizer_config_path=Path(tok_cfg) if tok_cfg else None,
        default_runs=int(data["default_runs"]),
        default_max_new_tokens=int(data["default_max_new_tokens"]),
        default_temperature=float(data["default_temperature"]),
    )
    return cfg


def _parse_last_json_line(text: str, prefix: str) -> dict[str, Any] | None:
    pattern = re.compile(rf"{re.escape(prefix)}\s+(\{{.*\}})")
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1)
    return json.loads(raw)


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_cpp_bench(
    cfg: RepoConfig,
    *,
    runs: int,
    max_new_tokens: int,
    temperature: float,
    prompt_file: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    exe = cfg.runtime_root / "bin" / "llama_main.exe"
    if not exe.exists():
        raise FileNotFoundError(f"Missing C++ runner: {exe}")

    cpp_dir = out_dir / "cpp"
    cpp_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i in range(1, runs + 1):
        cmd = [
            str(exe),
            "--model_path",
            str(cfg.cpp_model_path),
            "--tokenizer_path",
            str(cfg.cpp_tokenizer_path),
            "--prompt_file",
            str(prompt_file),
            "--temperature",
            str(temperature),
            "--max_new_tokens",
            str(max_new_tokens),
            "--ignore_eos",
        ]
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=cfg.repo_root,
            text=True,
            capture_output=True,
        )
        wall_seconds = time.perf_counter() - start
        (cpp_dir / f"run{i}_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (cpp_dir / f"run{i}_stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"C++ bench failed at run={i} (exit={proc.returncode}). "
                f"See logs under {cpp_dir}"
            )

        obs = _parse_last_json_line(proc.stdout, "PyTorchObserver")
        row: dict[str, Any] = {
            "backend": "cpp",
            "run": i,
            "exit_code": proc.returncode,
            "wall_seconds": _round4(wall_seconds),
            "prefill_token_per_sec": None,
            "decode_token_per_sec": None,
            "prompt_tokens": None,
            "generated_tokens": None,
        }
        if obs:
            row["prefill_token_per_sec"] = float(obs.get("prefill_token_per_sec"))
            row["decode_token_per_sec"] = float(obs.get("decode_token_per_sec"))
            row["prompt_tokens"] = int(obs.get("prompt_tokens"))
            row["generated_tokens"] = int(obs.get("generated_tokens"))
        rows.append(row)
    return rows


def _run_python_bench(
    cfg: RepoConfig,
    *,
    runs: int,
    max_new_tokens: int,
    temperature: float,
    prompt_file: Path,
    out_dir: Path,
) -> list[dict[str, Any]]:
    py_exe = cfg.repo_root / ".venv" / "Scripts" / "python.exe"
    if not py_exe.exists():
        raise FileNotFoundError(f"Missing Python executable: {py_exe}")

    cmd = [
        str(py_exe),
        str(cfg.repo_root / "scripts" / "vk_python_bench.py"),
        "--pte",
        str(cfg.cpp_model_path),
        "--params",
        str(cfg.python_params_path),
        "--tokenizer",
        str(cfg.python_tokenizer_path),
        "--model-id",
        cfg.python_model_id,
        "--prompt-file",
        str(prompt_file),
        "--runs",
        str(runs),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--max-len",
        "2048",
        "--kv-cache",
        "--out-dir",
        str(out_dir / "python"),
    ]
    if cfg.python_tokenizer_config_path is not None:
        cmd.extend(["--tokenizer-config", str(cfg.python_tokenizer_config_path)])

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cfg.repo_root, text=True, capture_output=True)
    wall_seconds = time.perf_counter() - start

    py_dir = out_dir / "python"
    py_dir.mkdir(parents=True, exist_ok=True)
    (py_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (py_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Python bench failed (exit={proc.returncode})\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"PyPythonObserver\s+(\{.*\})")
    for m in pattern.finditer(proc.stdout):
        obs = json.loads(m.group(1))
        rows.append(
            {
                "backend": "python",
                "run": int(obs["run"]),
                "exit_code": 0,
                "wall_seconds": _round4(float(obs["wall_seconds"])),
                "prefill_token_per_sec": (
                    float(obs["prefill_token_per_sec"])
                    if obs.get("prefill_token_per_sec") is not None
                    else None
                ),
                "decode_token_per_sec": (
                    float(obs["decode_token_per_sec"])
                    if obs.get("decode_token_per_sec") is not None
                    else None
                ),
                "prompt_tokens": int(obs["prompt_tokens"]),
                "generated_tokens": int(obs["generated_tokens"]),
            }
        )
    if len(rows) != runs:
        raise RuntimeError(
            f"Expected {runs} PyPythonObserver lines, got {len(rows)}. Check {py_dir / 'stdout.log'}"
        )

    # Track full process wall-time for reference.
    (py_dir / "process_wall_seconds.txt").write_text(
        f"{wall_seconds:.6f}\n", encoding="utf-8"
    )
    return rows


def _build_summary(
    *,
    cpp_rows: list[dict[str, Any]],
    py_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def collect(rows: list[dict[str, Any]], key: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            out.append(float(v))
        return out

    cpp_prefill = _avg(collect(cpp_rows, "prefill_token_per_sec"))
    cpp_decode = _avg(collect(cpp_rows, "decode_token_per_sec"))
    cpp_wall = _avg(collect(cpp_rows, "wall_seconds"))
    py_prefill = _avg(collect(py_rows, "prefill_token_per_sec"))
    py_decode = _avg(collect(py_rows, "decode_token_per_sec"))
    py_wall = _avg(collect(py_rows, "wall_seconds"))

    by_backend = [
        {
            "backend": "cpp",
            "runs": len(cpp_rows),
            "avg_prefill_token_per_sec": _round4(cpp_prefill),
            "avg_decode_token_per_sec": _round4(cpp_decode),
            "avg_wall_seconds": _round4(cpp_wall),
        },
        {
            "backend": "python",
            "runs": len(py_rows),
            "avg_prefill_token_per_sec": _round4(py_prefill),
            "avg_decode_token_per_sec": _round4(py_decode),
            "avg_wall_seconds": _round4(py_wall),
        },
    ]

    ratio = {
        "python_vs_cpp_prefill_ratio": _round4(
            (py_prefill / cpp_prefill) if (py_prefill and cpp_prefill) else None
        ),
        "python_vs_cpp_decode_ratio": _round4(
            (py_decode / cpp_decode) if (py_decode and cpp_decode) else None
        ),
        "python_vs_cpp_wall_ratio": _round4(
            (py_wall / cpp_wall) if (py_wall and cpp_wall) else None
        ),
    }
    return by_backend, ratio


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare Vulkan performance: C++ vs Python runner.")
    p.add_argument("--config-path", default="config/default.psd1")
    p.add_argument("--runs", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.add_argument("--temperature", type=float, default=float("nan"))
    p.add_argument("--prompt-file", default="")
    p.add_argument("--out-dir", default="runtime/bench_compare")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_repo_config(repo_root, args.config_path)

    runs = args.runs if args.runs > 0 else cfg.default_runs
    max_new_tokens = (
        args.max_new_tokens if args.max_new_tokens > 0 else cfg.default_max_new_tokens
    )
    temperature = (
        args.temperature
        if not (args.temperature != args.temperature)
        else cfg.default_temperature
    )
    prompt_file = (
        Path(args.prompt_file).resolve()
        if args.prompt_file
        else cfg.prompt_path.resolve()
    )
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path_like in (
        cfg.cpp_model_path,
        cfg.cpp_tokenizer_path,
        cfg.python_params_path,
        cfg.python_tokenizer_path,
        prompt_file,
    ):
        if not path_like.exists():
            raise FileNotFoundError(f"Missing input path: {path_like}")

    cpp_rows = _run_cpp_bench(
        cfg,
        runs=runs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_file=prompt_file,
        out_dir=out_dir,
    )
    py_rows = _run_python_bench(
        cfg,
        runs=runs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        prompt_file=prompt_file,
        out_dir=out_dir,
    )

    merged = cpp_rows + py_rows
    _write_csv(out_dir / "runs.csv", merged)
    summary_rows, ratio = _build_summary(cpp_rows=cpp_rows, py_rows=py_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "ratio.json").write_text(
        json.dumps(ratio, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("vk_perf_compare summary:")
    print(json.dumps({"summary": summary_rows, "ratio": ratio}, indent=2, ensure_ascii=False))
    print(f"runs_csv={out_dir / 'runs.csv'}")
    print(f"summary_csv={out_dir / 'summary.csv'}")
    print(f"ratio_json={out_dir / 'ratio.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
