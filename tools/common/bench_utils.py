from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class BenchResult:
    latency_ms: float
    fps: float


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_forward_latency(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],
    device: str,
    warmup: int,
    iters: int,
    amp: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> BenchResult:
    """Forward-only latency: measures only model(x).

    This intentionally excludes any pre/post-processing.
    """
    if dtype is None:
        dtype = torch.float16 if (amp and device.startswith("cuda")) else torch.float32

    x = torch.randn(*input_shape, device=device, dtype=dtype)
    model = model.to(device).eval()

    # Warmup
    with torch.no_grad():
        if amp and device.startswith("cuda") and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for _ in range(max(0, warmup)):
                    _ = model(x)
        else:
            for _ in range(max(0, warmup)):
                _ = model(x)
    _sync_if_cuda(device)

    # Benchmark (batched timing to reduce overhead)
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        if amp and device.startswith("cuda") and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for _ in range(max(1, iters)):
                    _ = model(x)
        else:
            for _ in range(max(1, iters)):
                _ = model(x)
    _sync_if_cuda(device)
    dt = (time.perf_counter() - t0) * 1000.0 / max(1, iters)
    return BenchResult(latency_ms=float(dt), fps=float(1000.0 / dt if dt > 0 else 0.0))

