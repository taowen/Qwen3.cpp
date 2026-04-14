#!/usr/bin/env python3
"""
End-to-end smoke test for Vulkan dynamic shader manifest overlay.

Checks:
1) Invalid manifest path should fail during runtime execute.
2) Valid manifest path can override AOT kernel behavior.
3) Overlay keep/clear semantics are deterministic across module loads.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.extension.pybindings import portable_lib


class AddModel(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def _unlink_with_retry(path: Path, *, retries: int = 10, delay_sec: float = 0.1) -> None:
    for _ in range(retries):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay_sec)
    # Best-effort cleanup: do not fail functional test because of Windows file lock timing.
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def _build_module(
    x: torch.Tensor, y: torch.Tensor, compile_options: dict[str, Any]
) -> object:
    model = AddModel().eval()
    exported = torch.export.export(model, (x, y))
    partitioner = VulkanPartitioner(compile_options=compile_options)
    prog = to_edge_transform_and_lower(exported, partitioner=[partitioner]).to_executorch()
    pte_path = (
        Path(tempfile.gettempdir())
        / ("vk_overlay_e2e_" + next(tempfile._get_candidate_names()) + ".pte")
    )
    prog.save(str(pte_path))
    try:
        return portable_lib._load_for_executorch(str(pte_path))
    finally:
        _unlink_with_retry(pte_path)


def _run_add(module: object, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = module.forward([x, y])[0]
    assert isinstance(out, torch.Tensor)
    return out


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _assert_close(a: torch.Tensor, b: torch.Tensor, atol: float, message: str) -> None:
    diff = _max_diff(a, b)
    if diff > atol:
        raise RuntimeError(f"{message}, max diff={diff}")


def _assert_not_close(a: torch.Tensor, b: torch.Tensor, atol: float, message: str) -> None:
    diff = _max_diff(a, b)
    if diff <= atol:
        raise RuntimeError(f"{message}, max diff={diff}")


def _expect_failure(x: torch.Tensor, y: torch.Tensor, compile_options: dict[str, Any]) -> None:
    failed = False
    try:
        module = _build_module(x, y, compile_options)
        _run_add(module, x, y)
    except Exception:
        failed = True
    if not failed:
        raise RuntimeError("Expected failure, but operation succeeded.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Vulkan dynamic shader overlay e2e smoke test.")
    parser.add_argument(
        "--build-dir",
        default=r"C:\Apps\qwen3-export\build\executorch-win-vulkan-pybind-cp311",
        help="ExecuTorch build dir that contains flatc and shader artifacts.",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    flatc_path = build_dir / "third-party" / "flatc_ep" / "bin" / "flatc.exe"
    if not flatc_path.exists():
        raise RuntimeError(f"flatc not found: {flatc_path}")
    add_buffer_shader_path = (
        build_dir / "vulkan_compute_shaders" / "binary_add_buffer_float.spv"
    )
    add_texture_shader_path = (
        build_dir / "vulkan_compute_shaders" / "binary_add_texture3d_float.spv"
    )
    mul_buffer_shader_path = (
        build_dir / "vulkan_compute_shaders" / "binary_mul_buffer_float.spv"
    )
    mul_texture_shader_path = (
        build_dir / "vulkan_compute_shaders" / "binary_mul_texture3d_float.spv"
    )
    for p in [
        add_buffer_shader_path,
        add_texture_shader_path,
        mul_buffer_shader_path,
        mul_texture_shader_path,
    ]:
        if not p.exists():
            raise RuntimeError(f"SPV not found: {p}")

    os.environ["FLATC_EXECUTABLE"] = str(flatc_path)

    torch.manual_seed(0)
    x = torch.randn(64, dtype=torch.float32)
    y = torch.randn(64, dtype=torch.float32)
    ref_add = x + y
    ref_mul = x * y

    _expect_failure(
        x,
        y,
        {
            "dynamic_shader_manifest_path": r"C:\__not_exist__\manifest.txt",
            "clear_dynamic_shader_overlay": True,
        },
    )

    with tempfile.TemporaryDirectory(prefix="vk_overlay_manifest_") as td:
        override_manifest = Path(td) / "override_mul_manifest.txt"
        # Override both buffer and texture add kernels to mul SPIR-V variants.
        # Runtime may pick either path based on layout decisions.
        override_manifest.write_text(
            "\n".join(
                [
                    f"binary_add_buffer_float={mul_buffer_shader_path}",
                    f"binary_add_texture3d_float={mul_texture_shader_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        # Case A: load override + clear first => output should match mul, not add.
        m_override = _build_module(
            x,
            y,
            {
                "dynamic_shader_manifest_path": str(override_manifest),
                "clear_dynamic_shader_overlay": True,
            },
        )
        out_override = _run_add(m_override, x, y)
        _assert_close(out_override, ref_mul, 1e-6, "override load should match mul result")
        _assert_not_close(
            out_override,
            ref_add,
            1e-4,
            "override load unexpectedly matched add result",
        )

        # Case B: no manifest + keep overlay => still use previous override.
        m_keep = _build_module(x, y, {"clear_dynamic_shader_overlay": False})
        out_keep = _run_add(m_keep, x, y)
        _assert_close(out_keep, ref_mul, 1e-6, "overlay keep should still match mul result")

        # Case C: clear-only (no manifest) => fallback to AOT add kernel.
        m_clear = _build_module(x, y, {"clear_dynamic_shader_overlay": True})
        out_clear = _run_add(m_clear, x, y)
        _assert_close(out_clear, ref_add, 1e-6, "clear-only should fallback to AOT add result")

    print("vk_overlay_e2e: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
