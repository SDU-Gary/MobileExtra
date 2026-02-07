#!/usr/bin/env python3
"""Export a *fused* ONNX graph for deployment.

Fused graph includes the log-domain pre/post processing so that TensorRT/ONNXRuntime
sees the exact same math as training/inference pipeline.

This script is intentionally separate from benchmarking scripts.
All export artifacts must go to an artifacts directory (outside repo by default).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.common.artifacts import default_artifacts_dir  # noqa: E402
from tools.common.model_loading import load_model_from_ckpt, load_model_from_pth  # noqa: E402


class FusedPatchWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        log_eps: float = 1e-6,
        log_delta_abs_max: float = 16.0,
        log_delta_alpha: float = 1.2,
        residual_scale: float = 1.0,
        ring_kernel: int = 3,
        ring_scale: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.log_eps = float(log_eps)
        self.log_delta_abs_max = float(log_delta_abs_max)
        self.log_delta_alpha = float(log_delta_alpha)
        self.residual_scale = float(residual_scale)
        self.ring_kernel = int(ring_kernel)
        self.ring_scale = float(ring_scale)

    def _log_normalize(self, warped_rgb: torch.Tensor):
        warped_pos = torch.clamp(warped_rgb, min=0.0)
        log_img = torch.log(warped_pos + self.log_eps)

        B = log_img.shape[0]
        min_log = torch.amin(log_img.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)
        max_log = torch.amax(log_img.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)
        denom = torch.clamp(max_log - min_log, min=1e-6)
        Xn = (log_img - min_log) / denom
        return log_img, Xn

    def _hole_ring_weight(self, holes_mask: torch.Tensor, target_h: int, target_w: int):
        if holes_mask.shape[2] != target_h or holes_mask.shape[3] != target_w:
            holes_mask = F.interpolate(holes_mask, size=(target_h, target_w), mode="nearest")
        pad = self.ring_kernel // 2
        ring = F.max_pool2d(holes_mask, kernel_size=self.ring_kernel, stride=1, padding=pad) - holes_mask
        ring = torch.clamp(ring, 0.0, 1.0)
        return torch.clamp(holes_mask + self.ring_scale * ring, 0.0, 1.0)

    def forward(self, x: torch.Tensor):
        # x: [B,7,H,W]
        warped_rgb = x[:, :3]
        holes_mask = x[:, 3:4]

        log_img, Xn = self._log_normalize(warped_rgb)
        model_in = torch.cat([Xn, x[:, 3:]], dim=1)

        residual_pred_log = self.model(model_in)
        delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * self.log_delta_abs_max

        mask_weight = self._hole_ring_weight(holes_mask, log_img.shape[2], log_img.shape[3])
        delta_log = delta_log * mask_weight

        log_output = log_img + delta_log
        residual_linear = torch.exp(log_output) - self.log_eps
        reconstructed = warped_rgb + residual_linear * self.residual_scale
        return residual_linear, reconstructed


def parse_args():
    ap = argparse.ArgumentParser(description="Export fused Patch-style model to ONNX")
    ap.add_argument("--model", required=True, help="ckpt or pth path")
    ap.add_argument("--from-ckpt", action="store_true", help="load as lightning checkpoint")
    ap.add_argument(
        "--network-type",
        default="v2",
        help="v1|v2|student_s1|student_s2|extranet",
    )
    ap.add_argument("--base-channels", type=int, default=24)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true", help="enable dynamic H/W axes")
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=None, help="output ONNX path")
    ap.add_argument("--artifacts-dir", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else default_artifacts_dir(ROOT)
    export_dir = artifacts_dir / "exports" / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else (export_dir / f"{args.network_type}_{args.height}x{args.width}_fused.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_ckpt:
        model = load_model_from_ckpt(args.model, args.network_type, args.base_channels, device)
    else:
        model = load_model_from_pth(args.model, args.network_type, args.base_channels, device)

    wrapper = FusedPatchWrapper(model, residual_scale=args.residual_scale).to(device).eval()
    dummy = torch.randn(1, 7, args.height, args.width, device=device)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "residual": {0: "batch", 2: "height", 3: "width"},
            "reconstructed": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["residual", "reconstructed"],
        opset_version=int(args.opset),
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    print(f"✅ Exported fused ONNX: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

