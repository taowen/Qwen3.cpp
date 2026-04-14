#!/usr/bin/env python3
"""
Pure-Python Vulkan workflow for this repo.

Features:
1) Operator-fusion profile control at export time.
2) Python-native Llama runner execution (no llama_main.exe required).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Set, Tuple


def _log(msg: str) -> None:
    print(f"[vk-py] {msg}")


DEFAULT_VULKAN_BUFFER_LIMIT = 512 * 1024 * 1024


def _quote(parts: Sequence[str]) -> str:
    out: List[str] = []
    for p in parts:
        if any(ch.isspace() for ch in p):
            out.append(f'"{p}"')
        else:
            out.append(p)
    return " ".join(out)


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    where = f" (cwd={cwd})" if cwd else ""
    _log(f"$ {_quote(cmd)}{where}")
    if dry_run:
        return
    subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        check=True,
    )


def _powershell() -> str:
    return "powershell"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(repo_root: Path) -> Path:
    return repo_root / ".venv" / "Scripts" / "python.exe"


def _full_path(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _load_psd1(psd1_path: Path) -> dict[str, Any]:
    escaped = str(psd1_path).replace("'", "''")
    expr = (
        f"$cfg=Import-PowerShellDataFile -Path '{escaped}'; "
        "$cfg | ConvertTo-Json -Depth 12 -Compress"
    )
    out = subprocess.run(
        [_powershell(), "-NoProfile", "-Command", expr],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


@dataclass
class Context:
    repo_root: Path
    config_path: Path
    cfg: dict[str, Any]
    venv_python: Path
    upstream_repo: Path
    upstream_executorch: Path
    build_dir: Path
    tokenizer_path: Path
    tokenizer_config_path: Path | None
    params_path: Path
    default_model_artifact: str
    default_model_id: str


def _build_context(config_path: str) -> Context:
    repo_root = _repo_root()
    cfg_path = _full_path(config_path, repo_root)
    cfg = _load_psd1(cfg_path)
    paths = cfg["Paths"]
    export = cfg["Export"]

    upstream_repo = Path(paths["ExecuTorchRepoRoot"]).resolve()
    upstream_executorch = Path(paths["ExecuTorchRoot"]).resolve()
    build_dir = _full_path(paths["ExecuTorchBuildDir"], upstream_repo)
    tokenizer_path = _full_path(paths["TokenizerPath"], repo_root)

    tok_cfg = tokenizer_path.parent / "tokenizer_config.json"
    tokenizer_config_path = tok_cfg if tok_cfg.exists() else None

    params_path = _full_path(export["ParamsRelativePath"], upstream_repo)

    return Context(
        repo_root=repo_root,
        config_path=cfg_path,
        cfg=cfg,
        venv_python=_venv_python(repo_root),
        upstream_repo=upstream_repo,
        upstream_executorch=upstream_executorch,
        build_dir=build_dir,
        tokenizer_path=tokenizer_path,
        tokenizer_config_path=tokenizer_config_path,
        params_path=params_path,
        default_model_artifact=paths["DefaultModelArtifact"],
        default_model_id=export["ModelId"],
    )


def _compute_fusion_flags(profile: str) -> tuple[bool, bool]:
    if profile == "none":
        return False, False
    if profile == "kv":
        return True, False
    if profile == "sdpa_kv":
        return True, True
    raise ValueError(f"Unsupported fusion profile: {profile}")


def _flatten_csv_args(raw_values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for part in raw.split(","):
            token = part.strip()
            if not token or token in seen:
                continue
            out.append(token)
            seen.add(token)
    return out


def _build_export_llama_args(
    *,
    output_name: str,
    model_id: str,
    quant_mode: str,
    group_size: int,
    enable_kv: bool,
    enable_sdpa_kv: bool,
) -> list[str]:
    args: list[str] = [
        "--model",
        model_id,
        "--params",
        "__QWEN3CPP_PARAMS_PLACEHOLDER__",
        "-qmode",
        quant_mode,
        "-G",
        str(group_size),
        "--output_name",
        output_name,
    ]
    args.append("-V")
    if enable_kv:
        args.append("-kv")
    if enable_sdpa_kv:
        args.append("--use_sdpa_with_kv_cache")
    return args


def _export_vulkan_pte_with_patched_partitioner(
    ctx: Context,
    *,
    env: dict[str, str],
    export_llama_args: Sequence[str],
    compile_options: dict[str, Any],
    operator_blocklist: Sequence[str],
    operator_allowlist: Sequence[str],
    dry_run: bool,
) -> None:
    payload = {
        "export_args": list(export_llama_args),
        "compile_options": compile_options,
        "operator_blocklist": list(operator_blocklist),
        "operator_allowlist": list(operator_allowlist),
    }
    export_env = dict(env)
    export_env["VK_PURE_PYTHON_EXPORT_PATCH_PAYLOAD"] = json.dumps(payload)

    patch_code = r"""
import json
import os
import torch

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import executorch.backends.vulkan.utils as vk_utils
from executorch.devtools.backend_debug import get_delegation_info
from executorch.examples.models.llama.export_llama_lib import build_args_parser, export_llama
import executorch.examples.models.llama.export_llama_lib as export_llama_lib


def _resolve_op_key(op_name):
    cur = torch.ops
    for segment in op_name.split("."):
        if not segment:
            continue
        if not hasattr(cur, segment):
            raise RuntimeError(f"Unknown op key segment '{segment}' in '{op_name}'")
        cur = getattr(cur, segment)
    return cur


payload = json.loads(os.environ["VK_PURE_PYTHON_EXPORT_PATCH_PAYLOAD"])
compile_options = dict(payload.get("compile_options", {}))
operator_blocklist = [_resolve_op_key(x) for x in payload.get("operator_blocklist", [])]
operator_allowlist = [_resolve_op_key(x) for x in payload.get("operator_allowlist", [])]

# Some Vulkan operator checks (e.g. embedding weight size) read this global limit.
if "buffer_limit" in compile_options:
    try:
        vk_utils.DEFAULT_BUFFER_LIMIT = int(compile_options["buffer_limit"])
    except Exception:
        pass


def _custom_get_vulkan_partitioner(dtype_override=None, enable_dynamic_shape=False, force_fp16=False):
    if dtype_override not in ("fp32", None):
        raise AssertionError("Vulkan backend does not support non-fp32 dtype override")
    options = {
        "require_dynamic_shapes": bool(enable_dynamic_shape),
        "force_fp16": bool(force_fp16),
    }
    options.update(compile_options)
    return VulkanPartitioner(
        options,
        operator_blocklist=operator_blocklist or None,
        operator_allowlist=operator_allowlist or None,
    )


export_llama_lib.get_vulkan_partitioner = _custom_get_vulkan_partitioner


def _strict_print_delegation_info(graph_module):
    info = get_delegation_info(graph_module)
    print(info.get_summary(), end="")
    non_delegated = [
        (k, v.non_delegated)
        for k, v in info.delegation_by_operator.items()
        if v.non_delegated > 0
    ]
    # `getitem` is a tuple/list plumbing op and not a compute kernel fallback.
    non_delegated = [(k, v) for (k, v) in non_delegated if k != "getitem"]
    if non_delegated:
        non_delegated.sort(key=lambda x: x[1], reverse=True)
        top = ", ".join(f"{k}:{v}" for k, v in non_delegated[:12])
        raise RuntimeError(
            "CPU fallback is forbidden: found non-delegated call_function nodes after Vulkan partition. "
            f"non_delegated={sum(v for _, v in non_delegated)}; top_ops=[{top}]"
        )


export_llama_lib.print_delegation_info = _strict_print_delegation_info

parser = build_args_parser()
namespace = parser.parse_args(payload["export_args"])
namespace.verbose = True
export_llama(namespace)
"""

    _run(
        [
            str(ctx.venv_python),
            "-c",
            patch_code,
        ],
        cwd=ctx.upstream_repo,
        env=export_env,
        dry_run=dry_run,
    )


def _export_vulkan_pte(
    ctx: Context,
    *,
    output_name: str,
    model_id: str,
    quant_mode: str,
    group_size: int,
    enable_kv: bool,
    enable_sdpa_kv: bool,
    dynamic_shader_manifest_path: Path | None,
    clear_dynamic_shader_overlay: bool,
    operator_blocklist: Sequence[str],
    operator_allowlist: Sequence[str],
    dry_run: bool,
) -> Path:
    _sync_vulkan_backend_sources(ctx, dry_run=dry_run)
    _sync_llama_export_sources(ctx, dry_run=dry_run)

    flatc = ctx.build_dir / "third-party" / "flatc_ep" / "bin" / "flatc.exe"
    if not flatc.exists():
        raise RuntimeError(
            f"flatc not found at {flatc}. Build external ExecuTorch once before export."
        )
    _ensure_upstream_custom_ops_lib(ctx, dry_run=dry_run)
    _ensure_upstream_serialize_schema_files(ctx, dry_run=dry_run)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(ctx.upstream_repo / "third_party")
    env["FLATC_EXECUTABLE"] = str(flatc)

    export_llama_args = _build_export_llama_args(
        output_name=output_name,
        model_id=model_id,
        quant_mode=quant_mode,
        group_size=group_size,
        enable_kv=enable_kv,
        enable_sdpa_kv=enable_sdpa_kv,
    )

    export_llama_args = [
        str(ctx.params_path) if x == "__QWEN3CPP_PARAMS_PLACEHOLDER__" else x
        for x in export_llama_args
    ]

    compile_options: dict[str, Any] = {
        "buffer_limit": DEFAULT_VULKAN_BUFFER_LIMIT,
    }
    if dynamic_shader_manifest_path is not None:
        compile_options["dynamic_shader_manifest_path"] = str(
            dynamic_shader_manifest_path
        )
        compile_options["clear_dynamic_shader_overlay"] = bool(
            clear_dynamic_shader_overlay
        )

    _log("Using patched Vulkan partitioner export path (strict no-CPU-fallback).")
    _export_vulkan_pte_with_patched_partitioner(
        ctx,
        env=env,
        export_llama_args=export_llama_args,
        compile_options=compile_options,
        operator_blocklist=operator_blocklist,
        operator_allowlist=operator_allowlist,
        dry_run=dry_run,
    )
    return ctx.upstream_repo / output_name


def _run_python_runner(
    ctx: Context,
    *,
    pte_path: Path,
    model_id: str,
    prompt: str,
    max_len: int,
    temperature: float,
    use_kv_cache: bool,
    tokenizer_path: Path | None,
    tokenizer_config_path: Path | None,
    dry_run: bool,
) -> None:
    tok = tokenizer_path if tokenizer_path is not None else ctx.tokenizer_path
    tok_cfg = tokenizer_config_path if tokenizer_config_path is not None else ctx.tokenizer_config_path
    if tok.suffix.lower() == ".json" and tok_cfg is None:
        raise RuntimeError(
            f"Tokenizer is json but tokenizer_config.json not found beside it: {tok}"
        )

    args: list[str] = [
        str(ctx.venv_python),
        "-m",
        "executorch.examples.models.llama.runner.native",
        "--model",
        model_id,
        "--pte",
        str(pte_path),
        "--params",
        str(ctx.params_path),
        "--tokenizer",
        str(tok),
        "--prompt",
        prompt,
        "--max_len",
        str(max_len),
        "--temperature",
        str(temperature),
    ]
    if tok_cfg is not None:
        args.extend(["--tokenizer_config", str(tok_cfg)])
    if use_kv_cache:
        args.append("--kv_cache")

    _run(args, cwd=ctx.upstream_repo, dry_run=dry_run)


def _check_backends(_: argparse.Namespace) -> int:
    try:
        from executorch.extension.pybindings import portable_lib
    except Exception as exc:
        print(f"[vk-py] failed to import portable_lib: {exc}")
        return 2
    names = portable_lib._get_registered_backend_names()
    print("[vk-py] portable_lib registered backends:")
    for n in names:
        print(f"  - {n}")
    return 0


def _has_vulkan_backend(python_exe: Path) -> bool:
    probe_code = r"""
import json
try:
    from executorch.extension.pybindings import portable_lib
    names = list(portable_lib._get_registered_backend_names())
    print(json.dumps({"ok": True, "names": names}))
except Exception:
    print(json.dumps({"ok": False, "names": []}))
"""
    out = subprocess.run(
        [str(python_exe), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(out.stdout)
    if not bool(data.get("ok", False)):
        return False
    names = set(data.get("names", []))
    return "VulkanBackend" in names


def _query_python_env(python_exe: Path) -> dict[str, Any]:
    probe_code = r"""
import json
import site
import sys
import torch

site_packages = ""
for p in site.getsitepackages():
    if p.endswith("site-packages"):
        site_packages = p
        break
if not site_packages and site.getsitepackages():
    site_packages = site.getsitepackages()[0]

print(json.dumps({
    "major": int(sys.version_info.major),
    "minor": int(sys.version_info.minor),
    "site_packages": site_packages,
    "torch_cmake_prefix": torch.utils.cmake_prefix_path,
}))
"""
    out = subprocess.run(
        [str(python_exe), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


def _find_built_module(build_dir: Path, module_basename: str, py_tag: str) -> Path:
    direct = build_dir / "Release" / f"{module_basename}.{py_tag}-win_amd64.pyd"
    if direct.exists():
        return direct
    matches = list(build_dir.rglob(f"{module_basename}.{py_tag}-win_amd64.pyd"))
    if not matches:
        raise RuntimeError(f"Built module not found: {module_basename}.{py_tag}-win_amd64.pyd in {build_dir}")
    return matches[0]


def _find_custom_ops_aot_lib(build_dir: Path) -> Path:
    preferred = (
        build_dir
        / "extension"
        / "llm"
        / "custom_ops"
        / "Release"
        / "custom_ops_aot_lib.dll"
    )
    if preferred.exists():
        return preferred
    matches = list(build_dir.rglob("custom_ops_aot_lib.*"))
    for match in matches:
        if match.suffix.lower() in {".dll", ".so", ".dylib"}:
            return match
    raise RuntimeError(
        f"Built custom ops library not found in {build_dir} (custom_ops_aot_lib.*)."
    )


def _ensure_upstream_custom_ops_lib(
    ctx: Context,
    *,
    site_packages: Path | None = None,
    dry_run: bool,
) -> None:
    upstream_custom_ops_dir = (
        ctx.upstream_executorch / "extension" / "llm" / "custom_ops"
    )
    existing = list(upstream_custom_ops_dir.glob("*custom_ops_aot_lib.*"))
    if existing:
        return

    candidates: list[Path] = []
    if site_packages is not None:
        candidates.extend(
            [
                site_packages
                / "executorch"
                / "extension"
                / "llm"
                / "custom_ops"
                / "custom_ops_aot_lib.dll",
                site_packages
                / "executorch"
                / "extension"
                / "llm"
                / "custom_ops"
                / "libcustom_ops_aot_lib.so",
                site_packages
                / "executorch"
                / "extension"
                / "llm"
                / "custom_ops"
                / "libcustom_ops_aot_lib.dylib",
            ]
        )
    if ctx.build_dir.exists():
        try:
            candidates.append(_find_custom_ops_aot_lib(ctx.build_dir))
        except RuntimeError:
            pass

    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise RuntimeError(
            "Missing custom_ops_aot_lib in upstream ExecuTorch source tree. "
            "Run `scripts/vk_pure_python.py bootstrap-vulkan --force-rebuild` first."
        )

    dst = upstream_custom_ops_dir / src.name
    if dry_run:
        _log(f"Would install upstream custom ops lib: {src} -> {dst}")
        return
    _copy_binary_with_retry(src, dst)
    _log(f"Installed upstream custom ops lib: {dst}")


def _ensure_upstream_serialize_schema_files(ctx: Context, *, dry_run: bool) -> None:
    target_dir = ctx.upstream_executorch / "exir" / "_serialize"
    required = ["program.fbs", "scalar_type.fbs"]
    missing = [name for name in required if not (target_dir / name).exists()]
    if not missing:
        return

    py_info = _query_python_env(ctx.venv_python)
    site_packages = Path(str(py_info["site_packages"]))
    source_dir = site_packages / "executorch" / "exir" / "_serialize"

    for name in missing:
        src = source_dir / name
        dst = target_dir / name
        if not src.exists():
            raise RuntimeError(
                f"Missing required schema resource: {src}. "
                "Cannot hydrate upstream ExecuTorch serialize schemas."
            )
        if dry_run:
            _log(f"Would install serialize schema: {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _log(f"Installed serialize schema: {dst}")


def _resolve_cmake_executable() -> str:
    known_candidates = [
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        ),
        Path(r"C:\Program Files\CMake\bin\cmake.exe"),
    ]
    for c in known_candidates:
        if c.exists():
            return str(c)

    where_out = subprocess.run(
        ["where.exe", "cmake"],
        check=False,
        capture_output=True,
        text=True,
    )
    if where_out.returncode == 0:
        for line in where_out.stdout.splitlines():
            p = Path(line.strip())
            if not p.exists():
                continue
            lowered = str(p).lower().replace("/", "\\")
            if "\\.venv\\scripts\\cmake" in lowered:
                continue
            if "\\.local\\bin\\cmake" in lowered:
                continue
            return str(p)

    cmake_from_path = shutil.which("cmake")
    if cmake_from_path:
        return cmake_from_path
    raise RuntimeError("cmake executable not found")


def _files_identical(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    try:
        src_stat = src.stat()
        dst_stat = dst.stat()
    except OSError:
        return False
    if src_stat.st_size != dst_stat.st_size:
        return False

    buf_size = 65536
    with src.open("rb") as fs, dst.open("rb") as fd:
        while True:
            bs = fs.read(buf_size)
            bd = fd.read(buf_size)
            if bs != bd:
                return False
            if not bs:
                return True


def _copy_if_exists(
    src: Path, dst: Path, *, dry_run: bool, log_each: bool = False
) -> bool:
    if not src.exists():
        return False
    if _files_identical(src, dst):
        return False
    if log_each:
        _log(f"Sync file: {src} -> {dst}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Intentionally avoid preserving source mtime so build systems detect updates.
    shutil.copyfile(src, dst)
    return True


def _copy_tree_overlay(
    src: Path,
    dst: Path,
    *,
    dry_run: bool,
    exclude_dir_names: Set[str] | None = None,
    log_each: bool = False,
) -> Tuple[int, int]:
    if not src.exists():
        return (0, 0)
    excluded = exclude_dir_names or set()
    total = 0
    changed = 0
    for entry in src.rglob("*"):
        if not entry.is_file():
            continue
        rel = entry.relative_to(src)
        rel_parts = rel.parts
        if ".git" in rel_parts:
            continue
        if any(part in excluded for part in rel_parts[:-1]):
            continue
        total += 1
        if _copy_if_exists(entry, dst / rel, dry_run=dry_run, log_each=log_each):
            changed += 1
    return (total, changed)


def _sync_llama_export_sources(ctx: Context, *, dry_run: bool) -> None:
    """
    Sync locally maintained llama export sources into upstream ExecuTorch tree.
    """
    local_models_root = ctx.repo_root / "vendor-executorch" / "examples" / "models"
    if not local_models_root.exists():
        return
    upstream_models_root = ctx.upstream_executorch / "examples" / "models"
    _log(
        "Sync local export sources: "
        f"{local_models_root} -> {upstream_models_root}"
    )
    total, changed = _copy_tree_overlay(
        local_models_root, upstream_models_root, dry_run=dry_run
    )
    _log(f"Llama export source sync: {changed}/{total} files updated.")


def _sync_vulkan_backend_sources(ctx: Context, *, dry_run: bool) -> None:
    """
    Sync locally maintained Vulkan backend sources into upstream ExecuTorch tree.
    """
    local_vk_root = ctx.repo_root / "vendor-executorch" / "backends" / "vulkan"
    if not local_vk_root.exists():
        return
    upstream_vk_root = ctx.upstream_executorch / "backends" / "vulkan"
    _log(
        "Sync local Vulkan backend sources: "
        f"{local_vk_root} -> {upstream_vk_root}"
    )
    total, changed = _copy_tree_overlay(
        local_vk_root,
        upstream_vk_root,
        dry_run=dry_run,
        exclude_dir_names={"third-party", "test", "__pycache__"},
    )
    _log(f"Vulkan backend source sync: {changed}/{total} files updated.")


def _sync_extension_pybindings_sources(ctx: Context, *, dry_run: bool) -> None:
    """
    Sync locally maintained pybindings C++/Python sources into upstream ExecuTorch tree.
    """
    local_pybindings_root = (
        ctx.repo_root / "vendor-executorch" / "extension" / "pybindings"
    )
    if not local_pybindings_root.exists():
        return
    upstream_pybindings_root = ctx.upstream_executorch / "extension" / "pybindings"
    _log(
        "Sync local pybindings sources: "
        f"{local_pybindings_root} -> {upstream_pybindings_root}"
    )
    total, changed = _copy_tree_overlay(
        local_pybindings_root,
        upstream_pybindings_root,
        dry_run=dry_run,
        exclude_dir_names={"test", "__pycache__"},
    )
    _log(f"Pybindings source sync: {changed}/{total} files updated.")


def _copy_binary_with_retry(
    src: Path, dst: Path, *, retries: int = 8, delay_sec: float = 0.5
) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_sec)
    raise RuntimeError(
        f"Failed to install binary {src} -> {dst}. "
        "Destination may be locked by another Python process using ExecuTorch."
    ) from last_error


def _sync_python_overlays(ctx: Context, *, site_packages: Path, dry_run: bool) -> None:
    pybindings_dst = site_packages / "executorch" / "extension" / "pybindings"
    pybindings_src_candidates = [
        ctx.repo_root / "vendor-executorch" / "extension" / "pybindings",
        ctx.upstream_executorch / "extension" / "pybindings",
    ]
    pybindings_src = next((p for p in pybindings_src_candidates if p.exists()), None)
    if pybindings_src is not None:
        _copy_if_exists(
            pybindings_src / "portable_lib.py",
            pybindings_dst / "portable_lib.py",
            dry_run=dry_run,
        )
        _copy_if_exists(
            pybindings_src / "pybindings.pyi",
            pybindings_dst / "pybindings.pyi",
            dry_run=dry_run,
        )
        _copy_if_exists(
            pybindings_src / "data_loader.pyi",
            pybindings_dst / "data_loader.pyi",
            dry_run=dry_run,
        )

    vulkan_partitioner_src_candidates = [
        ctx.repo_root
        / "vendor-executorch"
        / "backends"
        / "vulkan"
        / "partitioner"
        / "vulkan_partitioner.py",
        ctx.upstream_executorch
        / "backends"
        / "vulkan"
        / "partitioner"
        / "vulkan_partitioner.py",
    ]
    vulkan_partitioner_src = next(
        (p for p in vulkan_partitioner_src_candidates if p.exists()), None
    )
    if vulkan_partitioner_src is not None:
        _copy_if_exists(
            vulkan_partitioner_src,
            site_packages
            / "executorch"
            / "backends"
            / "vulkan"
            / "partitioner"
            / "vulkan_partitioner.py",
            dry_run=dry_run,
        )


def _verify_python_overlays(ctx: Context, *, dry_run: bool) -> None:
    verify_code = r"""
from executorch.backends.vulkan.partitioner.vulkan_partitioner import parse_compile_options
specs = parse_compile_options({"dynamic_shader_manifest_path": "x.txt", "clear_dynamic_shader_overlay": True})
keys = [s.key for s in specs]
print(keys)
assert "dynamic_shader_manifest_path" in keys
"""
    _run([str(ctx.venv_python), "-c", verify_code], cwd=ctx.repo_root, dry_run=dry_run)


def _bootstrap_vulkan_pybind_cmake(
    ctx: Context,
    *,
    build_dir: Path,
    dry_run: bool,
) -> None:
    py_info = _query_python_env(ctx.venv_python)
    py_tag = f"cp{py_info['major']}{py_info['minor']}"
    torch_cmake_prefix = str(py_info["torch_cmake_prefix"])
    site_packages = Path(str(py_info["site_packages"]))

    if not site_packages.exists():
        raise RuntimeError(f"site-packages path does not exist: {site_packages}")

    cmake_exe = _resolve_cmake_executable()

    configure_cmd = [
        cmake_exe,
        "-S",
        str(ctx.upstream_executorch),
        "-B",
        str(build_dir),
        "--preset",
        "pybind",
        "-T",
        "ClangCL",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DPYTHON_EXECUTABLE={ctx.venv_python}",
        f"-DCMAKE_PREFIX_PATH={torch_cmake_prefix}",
        "-DEXECUTORCH_BUILD_PYBIND=ON",
        "-DEXECUTORCH_BUILD_VULKAN=ON",
        "-DEXECUTORCH_BUILD_CUDA=OFF",
        "-DEXECUTORCH_BUILD_OPENVINO=OFF",
    ]
    build_portable_cmd = [
        cmake_exe,
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        "portable_lib",
        "--",
        "/m",
    ]
    build_data_loader_cmd = [
        cmake_exe,
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        "data_loader",
        "--",
        "/m",
    ]
    build_custom_ops_cmd = [
        cmake_exe,
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "--target",
        "custom_ops_aot_lib",
        "--",
        "/m",
    ]

    _run(configure_cmd, cwd=ctx.upstream_executorch, dry_run=dry_run)
    _run(build_portable_cmd, cwd=ctx.upstream_executorch, dry_run=dry_run)
    _run(build_data_loader_cmd, cwd=ctx.upstream_executorch, dry_run=dry_run)
    _run(build_custom_ops_cmd, cwd=ctx.upstream_executorch, dry_run=dry_run)

    if dry_run:
        return

    portable_src = _find_built_module(build_dir, "_portable_lib", py_tag)
    data_loader_src = _find_built_module(build_dir, "data_loader", py_tag)
    custom_ops_src = _find_custom_ops_aot_lib(build_dir)

    dst_dir = site_packages / "executorch" / "extension" / "pybindings"
    dst_dir.mkdir(parents=True, exist_ok=True)

    _copy_binary_with_retry(portable_src, dst_dir / portable_src.name)
    _copy_binary_with_retry(data_loader_src, dst_dir / data_loader_src.name)
    custom_ops_dst = (
        site_packages
        / "executorch"
        / "extension"
        / "llm"
        / "custom_ops"
        / custom_ops_src.name
    )
    _copy_binary_with_retry(custom_ops_src, custom_ops_dst)

    _sync_python_overlays(ctx, site_packages=site_packages, dry_run=dry_run)
    _ensure_upstream_custom_ops_lib(ctx, site_packages=site_packages, dry_run=dry_run)

    verify_cmd = [
        str(ctx.venv_python),
        "-c",
        "from executorch.extension.pybindings import portable_lib; "
        "names=portable_lib._get_registered_backend_names(); "
        "print(names); "
        "assert 'VulkanBackend' in names",
    ]
    _run(verify_cmd, cwd=ctx.repo_root, dry_run=dry_run)
    _verify_python_overlays(ctx, dry_run=dry_run)


def _bootstrap_vulkan_pybind(args: argparse.Namespace) -> int:
    ctx = _build_context(args.config_path)
    _sync_vulkan_backend_sources(ctx, dry_run=args.dry_run)
    _sync_extension_pybindings_sources(ctx, dry_run=args.dry_run)
    py_info = _query_python_env(ctx.venv_python)
    site_packages = Path(str(py_info["site_packages"]))
    if (not args.force_rebuild) and _has_vulkan_backend(ctx.venv_python):
        _sync_python_overlays(ctx, site_packages=site_packages, dry_run=args.dry_run)
        _ensure_upstream_custom_ops_lib(
            ctx, site_packages=site_packages, dry_run=args.dry_run
        )
        _verify_python_overlays(ctx, dry_run=args.dry_run)
        _log("VulkanBackend is already available in current Python runtime; skipping rebuild.")
        return 0

    py_tag = f"cp{py_info['major']}{py_info['minor']}"
    build_dir = (
        _full_path(args.build_dir, ctx.repo_root)
        if args.build_dir
        else (ctx.upstream_repo / "build" / f"executorch-win-vulkan-pybind-{py_tag}")
    )
    _log(f"Bootstrap build dir: {build_dir}")
    _bootstrap_vulkan_pybind_cmake(
        ctx,
        build_dir=build_dir,
        dry_run=args.dry_run,
    )
    _log("Bootstrap finished via CMake portable_lib build/install.")
    return 0


def _pipeline(args: argparse.Namespace) -> int:
    ctx = _build_context(args.config_path)
    enable_kv, enable_sdpa_kv = _compute_fusion_flags(args.fusion_profile)
    shader_manifest = (
        _full_path(args.shader_manifest, ctx.repo_root) if args.shader_manifest else None
    )
    op_blocklist = _flatten_csv_args(args.vk_op_blocklist)
    op_allowlist = _flatten_csv_args(args.vk_op_allowlist)

    _log(f"Config: {ctx.config_path}")
    _log(f"Fusion profile: {args.fusion_profile}")
    if shader_manifest is not None:
        _log(f"Dynamic shader manifest: {shader_manifest}")
    if op_blocklist:
        _log(f"Vulkan operator blocklist: {op_blocklist}")
    if op_allowlist:
        _log(f"Vulkan operator allowlist: {op_allowlist}")
    if not _has_vulkan_backend(ctx.venv_python):
        raise RuntimeError(
            "Python runtime has no VulkanBackend. Run `bootstrap-vulkan` first."
        )

    out_name = args.output_name or ctx.default_model_artifact
    pte_path = _export_vulkan_pte(
        ctx,
        output_name=out_name,
        model_id=args.model_id or ctx.default_model_id,
        quant_mode=args.quant_mode,
        group_size=args.group_size,
        enable_kv=enable_kv,
        enable_sdpa_kv=enable_sdpa_kv,
        dynamic_shader_manifest_path=shader_manifest,
        clear_dynamic_shader_overlay=not args.keep_dynamic_shader_overlay,
        operator_blocklist=op_blocklist,
        operator_allowlist=op_allowlist,
        dry_run=args.dry_run,
    )

    if args.export_only:
        _log(f"Export done: {pte_path}")
        return 0

    tokenizer_override = _full_path(args.tokenizer, ctx.repo_root) if args.tokenizer else None
    tokenizer_cfg_override = (
        _full_path(args.tokenizer_config, ctx.repo_root) if args.tokenizer_config else None
    )

    _run_python_runner(
        ctx,
        pte_path=pte_path,
        model_id=args.model_id or ctx.default_model_id,
        prompt=args.prompt,
        max_len=args.max_len,
        temperature=args.temperature,
        use_kv_cache=enable_kv,
        tokenizer_path=tokenizer_override,
        tokenizer_config_path=tokenizer_cfg_override,
        dry_run=args.dry_run,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pure-Python Vulkan workflow (no CMake build step in pipeline)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    chk = sub.add_parser("check-backends", help="Show Python portable_lib backend registry.")
    chk.set_defaults(func=_check_backends)

    boot = sub.add_parser(
        "bootstrap-vulkan",
        help="Build/install Vulkan-enabled portable_lib into current venv using CMake.",
    )
    boot.add_argument("--config-path", default="config/default.psd1")
    boot.add_argument("--build-dir", default="")
    boot.add_argument("--force-rebuild", action="store_true")
    boot.add_argument("--dry-run", action="store_true")
    boot.set_defaults(func=_bootstrap_vulkan_pybind)

    run = sub.add_parser(
        "pipeline",
        help="Pure Python export + Python native runner inference.",
    )
    run.add_argument("--config-path", default="config/default.psd1")
    run.add_argument("--model-id", default="")
    run.add_argument("--output-name", default="")
    run.add_argument("--quant-mode", default="8da4w")
    run.add_argument("--group-size", type=int, default=128)
    run.add_argument("--fusion-profile", choices=["none", "kv", "sdpa_kv"], default="sdpa_kv")
    run.add_argument("--shader-manifest", default="")
    run.add_argument("--keep-dynamic-shader-overlay", action="store_true")
    run.add_argument(
        "--vk-op-blocklist",
        action="append",
        default=[],
        help="Comma-separated op keys, e.g. aten.add.Tensor,aten.relu.default",
    )
    run.add_argument(
        "--vk-op-allowlist",
        action="append",
        default=[],
        help="Comma-separated op keys, e.g. aten.add.Tensor,aten.relu.default",
    )
    run.add_argument("--export-only", action="store_true")
    run.add_argument("--prompt", default="Please briefly introduce Vulkan in one sentence.")
    run.add_argument("--max-len", type=int, default=128)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--tokenizer", default="")
    run.add_argument("--tokenizer-config", default="")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=_pipeline)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
