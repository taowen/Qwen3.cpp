#!/usr/bin/env python3
"""
Python-driven Vulkan shader iteration workflow for this repo.

Goal:
- edit Vulkan GLSL under vendor-executorch/
- run one Python command
- auto sync -> incremental rebuild -> export -> pack -> run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence


def _log(message: str) -> None:
    print(f"[vk-jit] {message}")


def _format_cmd(cmd: Sequence[str]) -> str:
    quoted: List[str] = []
    for part in cmd:
        if any(ch.isspace() for ch in part):
            quoted.append(f'"{part}"')
        else:
            quoted.append(part)
    return " ".join(quoted)


def _run(cmd: Sequence[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    _log(f"$ {_format_cmd(cmd)}{where}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _powershell_exe() -> str:
    # This project targets Windows PowerShell workflows.
    return "powershell"


def _resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _read_psd1_as_json(psd1_path: Path) -> dict[str, Any]:
    escaped = str(psd1_path).replace("'", "''")
    ps_expr = (
        f"$cfg=Import-PowerShellDataFile -Path '{escaped}'; "
        "$cfg | ConvertTo-Json -Depth 12 -Compress"
    )
    result = subprocess.run(
        [_powershell_exe(), "-NoProfile", "-Command", ps_expr],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _qwen3_cmd(repo_root: Path, args: Iterable[str]) -> List[str]:
    return [
        _powershell_exe(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(repo_root / "scripts" / "qwen3.ps1"),
        *args,
    ]


def _run_pipeline(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    config_path = _resolve_path(args.config_path, repo_root)
    cfg = _read_psd1_as_json(config_path)

    paths = cfg["Paths"]
    upstream_repo = Path(paths["ExecuTorchRepoRoot"]).resolve()
    build_dir = _resolve_path(paths["ExecuTorchBuildDir"], upstream_repo)

    _log(f"Config: {config_path}")
    _log(f"External ExecuTorch repo: {upstream_repo}")
    _log(f"External build dir: {build_dir}")

    if not args.skip_sync:
        _run(
            _qwen3_cmd(
                repo_root,
                [
                    "vendor-sync-vulkan",
                    "-Mode",
                    "push",
                    "-ConfigPath",
                    str(config_path),
                ],
            ),
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_build:
        _run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "install",
                "--",
                "/m",
            ],
            cwd=upstream_repo,
            dry_run=args.dry_run,
        )

    if not args.skip_export:
        _run(
            _qwen3_cmd(repo_root, ["export", "-ConfigPath", str(config_path)]),
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.skip_pack:
        _run(
            _qwen3_cmd(repo_root, ["pack", "-ConfigPath", str(config_path)]),
            cwd=repo_root,
            dry_run=args.dry_run,
        )

    if not args.no_run:
        run_args: List[str] = ["run", "-ConfigPath", str(config_path)]
        if args.model:
            run_args.extend(["-Model", args.model])
        if args.prompt:
            run_args.extend(["-Prompt", args.prompt])
        if args.prompt_file:
            run_args.extend(["-PromptFile", args.prompt_file])
        if args.max_new_tokens is not None:
            run_args.extend(["-MaxNewTokens", str(args.max_new_tokens)])
        if args.temperature is not None:
            run_args.extend(["-Temperature", str(args.temperature)])
        _run(_qwen3_cmd(repo_root, run_args), cwd=repo_root, dry_run=args.dry_run)

    _log("Done.")
    return 0


def _check_runtime_backends(_: argparse.Namespace) -> int:
    # Run in this Python interpreter/environment.
    try:
        from executorch.runtime import Runtime
    except Exception as exc:
        print(f"[vk-jit] failed to import executorch.runtime: {exc}")
        return 2

    runtime = Runtime.get()
    names = sorted(runtime.backend_registry.registered_backend_names)
    print("[vk-jit] registered backends:")
    for name in names:
        print(f"  - {name}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python-driven Vulkan shader rebuild + run workflow."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run",
        help="Sync Vulkan shaders, incrementally rebuild, export/pack, then run.",
    )
    run_p.add_argument("--config-path", default="config/default.psd1")
    run_p.add_argument("--skip-sync", action="store_true")
    run_p.add_argument("--skip-build", action="store_true")
    run_p.add_argument("--skip-export", action="store_true")
    run_p.add_argument("--skip-pack", action="store_true")
    run_p.add_argument("--no-run", action="store_true")
    run_p.add_argument("--model", default="")
    run_p.add_argument("--prompt", default="")
    run_p.add_argument("--prompt-file", default="")
    run_p.add_argument("--max-new-tokens", type=int, default=None)
    run_p.add_argument("--temperature", type=float, default=None)
    run_p.add_argument("--dry-run", action="store_true")
    run_p.set_defaults(func=_run_pipeline)

    check_p = sub.add_parser(
        "check-runtime",
        help="Print registered ExecuTorch runtime backends in current Python env.",
    )
    check_p.set_defaults(func=_check_runtime_backends)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
