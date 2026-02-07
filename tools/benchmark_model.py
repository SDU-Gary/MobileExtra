#!/usr/bin/env python3
"""Unified forward-only benchmark.

Policy (project-wide): **benchmark latency must be forward-only**.
That means we measure *only* `model(x)` and explicitly exclude any
pre/post-processing (log normalize, mask ring, exp restore, tone mapping, etc.).

For end-to-end pipeline timing, create a separate script (not benchmark_model).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.common.bench_utils import measure_forward_latency
from tools.common.model_loading import load_model_from_ckpt, load_model_from_pth


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args():
    ap = argparse.ArgumentParser(description="Forward-only model benchmark")
    ap.add_argument("--model", type=str, required=True, help="Path to model (.pth or Lightning .ckpt)")
    ap.add_argument("--from-ckpt", action="store_true", help="Load as Lightning checkpoint")
    ap.add_argument(
        "--network-type",
        type=str,
        default="v2",
        help="v1|v2|student_s1|student_s2|extranet (used to construct the module)",
    )
    ap.add_argument("--base-channels", type=int, default=24)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="Use torch.autocast FP16 on CUDA")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 2

    try:
        if args.from_ckpt:
            model = load_model_from_ckpt(model_path, args.network_type, args.base_channels, device)
        else:
            model = load_model_from_pth(model_path, args.network_type, args.base_channels, device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise

    total, trainable = count_params(model)
    input_shape = (int(args.batch), 7, int(args.height), int(args.width))
    result = measure_forward_latency(
        model,
        input_shape,
        device=device,
        warmup=int(args.warmup),
        iters=int(args.iters),
        amp=bool(args.amp),
    )

    print("=" * 70)
    print("Forward-only Benchmark")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Network type: {args.network_type}")
    print(f"Device: {device}")
    print(f"Input: {input_shape[0]}x{input_shape[1]}x{input_shape[2]}x{input_shape[3]}")
    print(f"Params: total={total:,} (trainable={trainable:,})")
    print(f"Latency: {result.latency_ms:.3f} ms ({result.fps:.1f} FPS)")
    print("NOTE: This excludes any pre/post-processing by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

