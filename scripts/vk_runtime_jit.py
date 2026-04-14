#!/usr/bin/env python3
"""
Experimental pure-Python Vulkan JIT runtime.

What this script does:
1) Export a model graph via torch.export (optionally to_edge).
2) Apply a Python fusion pass (example: add + relu -> add_relu).
3) Generate GLSL compute shader source at runtime.
4) JIT-compile GLSL to SPIR-V via glslc (with cache).
5) Validate outputs with CPU eager reference only (no Vulkan dispatch path).

This is intentionally independent from ExecuTorch ET-VK AOT runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch import nn


def _log(msg: str) -> None:
    print(f"[vk-runtime-jit] {msg}")


@dataclasses.dataclass
class JITOp:
    op_type: str
    name: str
    in_names: List[str]
    out_name: str
    meta: Dict[str, Any]


def _target_name(target: Any) -> str:
    if hasattr(target, "__name__"):
        return str(target.__name__)
    return str(target)


def _is_add_target(target: Any) -> bool:
    s = _target_name(target).lower()
    return "add" in s


def _is_relu_target(target: Any) -> bool:
    s = _target_name(target).lower()
    return "relu" in s


def _graph_nodes_to_ops(graph_module: torch.fx.GraphModule) -> List[JITOp]:
    """
    Build linear JIT ops from FX graph with a tiny fusion pass:
    add -> relu  => add_relu.
    """
    nodes = list(graph_module.graph.nodes)
    fused_add: set[torch.fx.Node] = set()

    # Pass 1: find relu(add(...)) patterns.
    for n in nodes:
        if n.op != "call_function" or not _is_relu_target(n.target):
            continue
        if not n.args:
            continue
        arg0 = n.args[0]
        if not isinstance(arg0, torch.fx.Node):
            continue
        prev = arg0
        if prev.op == "call_function" and _is_add_target(prev.target):
            fused_add.add(prev)

    ops: List[JITOp] = []

    for n in nodes:
        if n.op != "call_function":
            continue

        if _is_relu_target(n.target) and len(n.args) >= 1:
            arg0 = n.args[0]
            if isinstance(arg0, torch.fx.Node):
                prev = arg0
                if prev in fused_add:
                    in_names = []
                    for a in prev.args[:2]:
                        if isinstance(a, torch.fx.Node):
                            in_names.append(a.name)
                    if len(in_names) == 2:
                        ops.append(
                            JITOp(
                                op_type="add_relu",
                                name=f"fused_{prev.name}_{n.name}",
                                in_names=in_names,
                                out_name=n.name,
                                meta={"src_add": prev.name, "src_relu": n.name},
                            )
                        )
                        continue

        if _is_add_target(n.target):
            if n in fused_add:
                continue
            in_names = []
            for a in n.args[:2]:
                if isinstance(a, torch.fx.Node):
                    in_names.append(a.name)
            if len(in_names) == 2:
                ops.append(
                    JITOp(
                        op_type="add",
                        name=n.name,
                        in_names=in_names,
                        out_name=n.name,
                        meta={},
                    )
                )
            continue

        if _is_relu_target(n.target):
            in_names = []
            a0 = n.args[0] if n.args else None
            if isinstance(a0, torch.fx.Node):
                in_names.append(a0.name)
            if len(in_names) == 1:
                ops.append(
                    JITOp(
                        op_type="relu",
                        name=n.name,
                        in_names=in_names,
                        out_name=n.name,
                        meta={},
                    )
                )
            continue

    return ops


def _shader_for_op(op: JITOp, n_elem: int) -> str:
    preamble = (
        "#version 450\n"
        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
        f"const uint N = {int(n_elem)}u;\n"
    )
    if op.op_type == "add":
        body = """
layout(set=0, binding=0) buffer In0 { float data[]; } in0;
layout(set=0, binding=1) buffer In1 { float data[]; } in1;
layout(set=0, binding=2) buffer Out { float data[]; } out0;
void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= N) { return; }
  out0.data[idx] = in0.data[idx] + in1.data[idx];
}
"""
    elif op.op_type == "relu":
        body = """
layout(set=0, binding=0) buffer In0 { float data[]; } in0;
layout(set=0, binding=1) buffer Out { float data[]; } out0;
void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= N) { return; }
  float a = in0.data[idx];
  out0.data[idx] = max(a, 0.0);
}
"""
    elif op.op_type == "add_relu":
        body = """
layout(set=0, binding=0) buffer In0 { float data[]; } in0;
layout(set=0, binding=1) buffer In1 { float data[]; } in1;
layout(set=0, binding=2) buffer Out { float data[]; } out0;
void main() {
  uint idx = gl_GlobalInvocationID.x;
  if (idx >= N) { return; }
  float a = in0.data[idx];
  float b = in1.data[idx];
  out0.data[idx] = max(a + b, 0.0);
}
"""
    else:
        raise ValueError(f"Unsupported op for shader generation: {op.op_type}")
    return preamble + body


class ShaderCompiler:
    def __init__(self, cache_dir: Path, use_noop: bool = False):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_noop = use_noop
        self._backend = "noop" if use_noop else "glslc"
        if not use_noop:
            glslc = subprocess.run(
                ["where", "glslc"],
                check=False,
                capture_output=True,
                text=True,
            )
            if glslc.returncode != 0:
                raise RuntimeError(
                    "No shader compiler available. Install Vulkan SDK and ensure `glslc` is in PATH, "
                    "or use --noop-compiler."
                )

    def _cache_key(self, source: str) -> str:
        h = hashlib.sha256()
        h.update(source.encode("utf-8"))
        h.update(b"|shader=compute")
        return h.hexdigest()

    def compile_compute(self, source: str, virtual_name: str) -> bytes:
        key = self._cache_key(source)
        cache_file = self.cache_dir / f"{key}.spv"
        if cache_file.exists():
            return cache_file.read_bytes()

        if self.use_noop:
            data = source.encode("utf-8")
            cache_file.write_bytes(data)
            return data

        if self._backend == "glslc":
            src_path = self.cache_dir / f"{key}.comp"
            src_path.write_text(source, encoding="utf-8")
            cmd = [
                "glslc",
                "-fshader-stage=compute",
                "-O",
                str(src_path),
                "-o",
                str(cache_file),
            ]
            subprocess.run(cmd, check=True)
            return cache_file.read_bytes()

        raise RuntimeError(f"Unsupported compiler backend: {self._backend}")


def _run_reference_op(op: JITOp, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    if op.op_type == "add":
        return state[op.in_names[0]] + state[op.in_names[1]]
    if op.op_type == "relu":
        return torch.relu(state[op.in_names[0]])
    if op.op_type == "add_relu":
        return torch.relu(state[op.in_names[0]] + state[op.in_names[1]])
    raise ValueError(f"Unsupported op type: {op.op_type}")


def _compile_and_validate(
    compiler: ShaderCompiler,
    ops: Iterable[JITOp],
    named_tensors: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, int]:
    state: Dict[str, torch.Tensor] = dict(named_tensors)
    compiled_bytes = 0
    last_out: torch.Tensor | None = None
    for op in ops:
        in0 = state[op.in_names[0]]
        n_elem = int(in0.numel())
        glsl = _shader_for_op(op, n_elem)
        spv = compiler.compile_compute(glsl, f"{op.name}.comp")
        compiled_bytes += len(spv)
        out = _run_reference_op(op, state)
        state[op.out_name] = out
        last_out = out
    if last_out is None:
        raise RuntimeError("No executable ops were produced from graph.")
    return last_out, compiled_bytes


class DemoAddRelu(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + y)


def _export_graph_module(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    use_to_edge: bool,
) -> torch.fx.GraphModule:
    ep = torch.export.export(model, example_inputs)
    if not use_to_edge:
        return ep.graph_module

    try:
        from executorch.exir import to_edge  # type: ignore
    except Exception as exc:
        raise RuntimeError("use_to_edge=True requires executorch.exir.to_edge") from exc

    edge = to_edge(ep)
    # API-compatible probing.
    if hasattr(edge, "exported_program"):
        p = edge.exported_program()
        if hasattr(p, "graph_module"):
            return p.graph_module
    if hasattr(edge, "graph_module"):
        return edge.graph_module
    raise RuntimeError("Cannot extract graph_module from to_edge(...) result.")


def _collect_named_inputs(
    gm: torch.fx.GraphModule, inputs: Tuple[torch.Tensor, ...]
) -> Dict[str, torch.Tensor]:
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(inputs):
        raise RuntimeError(
            f"Input mismatch: graph expects {len(placeholders)} placeholders, got {len(inputs)} tensors."
        )
    out: Dict[str, torch.Tensor] = {}
    for n, t in zip(placeholders, inputs):
        out[n.name] = t.detach().clone().to(dtype=torch.float32)
    return out


def _run_demo(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    model = DemoAddRelu().eval()

    x = torch.randn(args.size, dtype=torch.float32)
    y = torch.randn(args.size, dtype=torch.float32)
    gm = _export_graph_module(model, (x, y), use_to_edge=args.use_to_edge)
    ops = _graph_nodes_to_ops(gm)
    if not ops:
        raise RuntimeError("No supported ops found. Demo expects add/relu pattern.")

    _log(f"Extracted ops: {[o.op_type for o in ops]}")

    named_inputs = _collect_named_inputs(gm, (x, y))
    cache_dir = Path(args.cache_dir).resolve()
    compiler = ShaderCompiler(cache_dir=cache_dir, use_noop=args.noop_compiler)
    out, compiled_bytes = _compile_and_validate(compiler=compiler, ops=ops, named_tensors=named_inputs)
    ref = model(x, y)
    max_diff = (out - ref).abs().max().item()
    _log(f"max abs diff vs torch reference: {max_diff:.6f}")
    _log(f"compiled kernels: {len(ops)}, total bytes: {compiled_bytes}")

    if args.dump_json:
        report = {
            "ops": [dataclasses.asdict(o) for o in ops],
            "size": int(args.size),
            "max_abs_diff": float(max_diff),
            "compiled_kernels": len(ops),
            "compiled_bytes": int(compiled_bytes),
            "noop_compiler": bool(args.noop_compiler),
        }
        dump_path = Path(args.dump_json).resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _log(f"Wrote report: {dump_path}")

    return 0


def _doctor(_: argparse.Namespace) -> int:
    mods = ["torch", "executorch"]
    result = {}
    for m in mods:
        try:
            mod = __import__(m)
            result[m] = getattr(mod, "__file__", str(mod))
        except Exception as exc:
            result[m] = f"ERR:{type(exc).__name__}:{exc}"
    glslc = subprocess.run(
        ["where", "glslc"],
        check=False,
        capture_output=True,
        text=True,
    )
    result["glslc"] = (
        glslc.stdout.strip().splitlines() if glslc.returncode == 0 else "ERR:glslc not found"
    )
    print(json.dumps(result, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experimental pure-Python Vulkan JIT runtime.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Check python dependency availability.")
    d.set_defaults(func=_doctor)

    demo = sub.add_parser("demo", help="Run add+relu fusion demo.")
    demo.add_argument("--size", type=int, default=4096)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--cache-dir", default="runtime/vk_jit_cache")
    demo.add_argument("--noop-compiler", action="store_true")
    demo.add_argument("--use-to-edge", action="store_true")
    demo.add_argument("--dump-json", default="")
    demo.set_defaults(func=_run_demo)

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
