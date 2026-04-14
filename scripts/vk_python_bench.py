#!/usr/bin/env python3
"""Python-side benchmark runner for ExecuTorch Llama native runner.

Prints one line per run:
  PyPythonObserver { ...json... }
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from executorch.examples.models.llama.runner.generation import next_token
from executorch.examples.models.llama.runner.native import NativeLlamaRunner


@dataclass
class RunMetrics:
    run: int
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    wall_seconds: float
    prefill_token_per_sec: float | None
    decode_token_per_sec: float | None
    end_to_end_token_per_sec: float | None


def _read_prompt(prompt: str, prompt_file: Path | None) -> str:
    if prompt:
        return prompt
    if prompt_file is None:
        raise ValueError("Either --prompt or --prompt-file must be provided.")
    return prompt_file.read_text(encoding="utf-8")


def _build_runner(args: argparse.Namespace) -> NativeLlamaRunner:
    ns = SimpleNamespace(
        model=args.model_id,
        pte=str(args.pte),
        params=str(args.params),
        tokenizer=str(args.tokenizer),
        tokenizer_config=str(args.tokenizer_config) if args.tokenizer_config else None,
        prompt="",
        temperature=float(args.temperature),
        kv_cache=bool(args.kv_cache),
        max_len=int(args.max_len),
    )
    return NativeLlamaRunner(ns)


def _is_stop_token(tokenizer: Any, token: int) -> bool:
    if token == tokenizer.eos_id:
        return True
    if hasattr(tokenizer, "stop_tokens") and token in tokenizer.stop_tokens:
        return True
    return False


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def _run_once(
    runner: NativeLlamaRunner,
    *,
    run_index: int,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> RunMetrics:
    prompt_tokens = runner.tokenizer.encode(prompt, bos=True, eos=False)
    target_max_len = min(runner.max_seq_len, len(prompt_tokens) + max_new_tokens)
    if target_max_len <= len(prompt_tokens):
        raise ValueError(
            "max_new_tokens too small for current prompt/max_len; no decode step can run."
        )

    wall_start = time.perf_counter()
    prefill_start = time.perf_counter()
    logits = runner.forward(
        tokens=torch.tensor([prompt_tokens], dtype=torch.long, device=runner.device),
        input_pos=(
            torch.tensor([0], dtype=torch.long, device=runner.device)
            if runner.use_kv_cache
            else None
        ),
    )
    prefill_seconds = time.perf_counter() - prefill_start

    current_token = next_token(logits, temperature, top_p)
    generated_tokens = 1
    tokens = prompt_tokens + [current_token]

    decode_seconds = 0.0
    if not _is_stop_token(runner.tokenizer, current_token):
        decode_start = time.perf_counter()
        while len(tokens) < target_max_len:
            if runner.use_kv_cache:
                logits = runner.forward(
                    tokens=torch.tensor([[current_token]], dtype=torch.long, device=runner.device),
                    input_pos=torch.tensor(
                        [len(tokens) - 1], dtype=torch.long, device=runner.device
                    ),
                )
            else:
                logits = runner.forward(
                    tokens=torch.tensor([tokens], dtype=torch.long, device=runner.device)
                )
            current_token = next_token(logits, temperature, top_p)
            tokens.append(current_token)
            generated_tokens += 1
            if _is_stop_token(runner.tokenizer, current_token):
                break
        decode_seconds = time.perf_counter() - decode_start

    wall_seconds = time.perf_counter() - wall_start

    prefill_tps = _safe_div(float(len(prompt_tokens)), prefill_seconds)
    decode_tps = _safe_div(float(max(generated_tokens - 1, 0)), decode_seconds)
    end_to_end_tps = _safe_div(float(generated_tokens), wall_seconds)

    return RunMetrics(
        run=run_index,
        prompt_tokens=len(prompt_tokens),
        generated_tokens=generated_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        wall_seconds=wall_seconds,
        prefill_token_per_sec=prefill_tps,
        decode_token_per_sec=decode_tps,
        end_to_end_token_per_sec=end_to_end_tps,
    )


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark Python ExecuTorch native runner.")
    p.add_argument("--pte", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--tokenizer-config", type=Path, default=None)
    p.add_argument("--model-id", default="qwen3_0_6b")
    p.add_argument("--prompt", default="")
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--kv-cache", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")

    for path_like in (args.pte, args.params, args.tokenizer):
        if not path_like.exists():
            raise FileNotFoundError(f"Missing input file: {path_like}")
    if args.tokenizer_config and not args.tokenizer_config.exists():
        raise FileNotFoundError(f"Missing tokenizer config: {args.tokenizer_config}")

    prompt = _read_prompt(args.prompt, args.prompt_file)
    runner = _build_runner(args)

    run_rows: list[RunMetrics] = []
    for idx in range(1, args.runs + 1):
        row = _run_once(
            runner,
            run_index=idx,
            prompt=prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        run_rows.append(row)
        print(f"PyPythonObserver {json.dumps(asdict(row), ensure_ascii=False)}")

    prefill = [r.prefill_token_per_sec for r in run_rows if r.prefill_token_per_sec]
    decode = [r.decode_token_per_sec for r in run_rows if r.decode_token_per_sec]
    wall = [r.wall_seconds for r in run_rows]
    end_to_end = [
        r.end_to_end_token_per_sec for r in run_rows if r.end_to_end_token_per_sec
    ]
    summary = {
        "runs": args.runs,
        "ok_runs": len(run_rows),
        "avg_prefill_token_per_sec": _avg([float(v) for v in prefill]) if prefill else None,
        "avg_decode_token_per_sec": _avg([float(v) for v in decode]) if decode else None,
        "avg_end_to_end_token_per_sec": _avg([float(v) for v in end_to_end]) if end_to_end else None,
        "avg_wall_seconds": _avg([float(v) for v in wall]) if wall else None,
    }
    print(f"PyPythonSummary {json.dumps(summary, ensure_ascii=False)}")

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run_dicts = [asdict(r) for r in run_rows]
        _write_csv(args.out_dir / "runs.csv", run_dicts)
        _write_csv(args.out_dir / "summary.csv", [summary])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
