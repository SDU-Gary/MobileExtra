#!/usr/bin/env python3
"""Build a TensorRT engine from ONNX via `trtexec`.

This script exists to keep export/deployment logic separate from training and
benchmarking.

Artifacts are written to an artifacts directory outside the source tree by
default.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.common.artifacts import default_artifacts_dir  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Build TensorRT engine from ONNX using trtexec")
    ap.add_argument("--onnx", required=True, help="Input ONNX path")
    ap.add_argument("--out", default=None, help="Output engine path (.trt/.plan)")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16")
    ap.add_argument("--workspace-mib", type=int, default=2048, help="Workspace pool size (MiB)")

    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--dynamic", action="store_true", help="Build with explicit dynamic profile")
    ap.add_argument("--min-shapes", type=str, default=None, help="e.g. input:1x7x256x256")
    ap.add_argument("--opt-shapes", type=str, default=None, help="e.g. input:1x7x256x256")
    ap.add_argument("--max-shapes", type=str, default=None, help="e.g. input:1x7x256x256")

    ap.add_argument("--artifacts-dir", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    trtexec = shutil.which("trtexec")
    if not trtexec:
        print("[ERROR] trtexec not found in PATH")
        return 2

    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.exists():
        print(f"[ERROR] ONNX not found: {onnx_path}")
        return 2

    artifacts_dir = Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else default_artifacts_dir(ROOT)
    export_dir = artifacts_dir / "exports" / "trt"
    export_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out).expanduser().resolve() if args.out else (export_dir / (onnx_path.stem + ("_fp16" if args.fp16 else "_fp32") + ".trt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={out_path}",
        f"--memPoolSize=workspace:{int(args.workspace_mib)}",
    ]
    if args.fp16:
        cmd.append("--fp16")

    # Shape/profile options
    if args.height is not None and args.width is not None:
        shape = f"input:1x7x{int(args.height)}x{int(args.width)}"
        if args.dynamic:
            cmd.extend([f"--minShapes={shape}", f"--optShapes={shape}", f"--maxShapes={shape}"])
        else:
            # For static ONNX, do NOT pass explicit shapes.
            pass
    else:
        # Explicit shape strings
        if args.dynamic:
            if not (args.min_shapes and args.opt_shapes and args.max_shapes):
                print("[ERROR] --dynamic requires --min-shapes/--opt-shapes/--max-shapes or --height/--width")
                return 2
            cmd.extend([f"--minShapes={args.min_shapes}", f"--optShapes={args.opt_shapes}", f"--maxShapes={args.max_shapes}"])

    print(" ".join(str(x) for x in cmd))
    if args.dry_run:
        return 0

    p = subprocess.run(cmd)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

