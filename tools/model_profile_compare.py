#!/usr/bin/env python3
"""
Compare model variants (V1, V2, Student S1/S2, ExtraNet): params, MACs (if torchinfo available), and latency.

Usage:
  # PyTorch FP32 inference (default)
  python tools/model_profile_compare.py --models v1 v2 student_s1 student_s2 extranet \
      --height 256 --width 256 --batch 1 --device cuda --warmup 10 --iters 50

  # TensorRT FP16 inference via ONNX export
  python tools/model_profile_compare.py --models student_s1_enhanced extranet \
      --height 256 --width 256 --batch 1 --device cuda --warmup 10 --iters 50 \
      --use-tensorrt --fp16

  # TensorRT FP8 quantization (Model Optimizer PTQ)
  python tools/model_profile_compare.py --models student_s1_enhanced extranet \
      --height 256 --width 256 --batch 1 --device cuda --warmup 10 --iters 50 \
      --use-tensorrt --fp8

Notes:
- MACs estimation uses torchinfo.summary (requires torchinfo>=1.8). If unavailable, MACs will be skipped.
- Latency is FORWARD-ONLY: measures pure model(input) time, no pre/post-processing.
  - NOT included: log normalize, tanh scaling, hole mask weight, exp restore
  - For pipeline timing (pre/post included), use a dedicated pipeline benchmark script (not provided here).
- Latency is measured with random input, no grad, torch.cuda.synchronize() when on CUDA.
- ExtraNet and students are from src/npu/networks/patch.
- TensorRT mode: Exports ONNX (FP16), builds TRT engine, and benchmarks with TensorRT.
- NOTE: TRT artifacts should be stored in an artifacts directory (see docs/PIPELINE.md).
"""
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.common.bench_utils import measure_forward_latency
from tools.common.model_loading import build_model

try:
    from torchinfo import summary as torchinfo_summary
    TORCHINFO_AVAILABLE = True
except Exception:
    TORCHINFO_AVAILABLE = False

# TensorRT imports
try:
    import tensorrt as trt
    import numpy as np
    TENSORRT_AVAILABLE = True
except Exception:
    TENSORRT_AVAILABLE = False

# Model Optimizer imports for FP8 quantization
try:
    import modelopt.torch.quantization as mtq
    MODELOPT_AVAILABLE = True
except Exception:
    MODELOPT_AVAILABLE = False


def build_model_for_name(name: str, device: str) -> torch.nn.Module:
    name_l = name.lower()
    if name_l in {"v1", "patchnetwork"}:
        return build_model("v1", base_channels=24, device=device)
    if name_l in {"v2", "patchnetworkv2", "patch_network_v2"}:
        return build_model("v2", base_channels=24, device=device)
    if name_l in {"student_s1", "student"}:
        return build_model("student_s1", base_channels=16, device=device)
    if name_l == "student_s1_enhanced":
        return build_model("student_s1_enhanced", base_channels=20, device=device)
    if name_l == "student_s1_asymmetric":
        return build_model("student_s1_asymmetric", base_channels=20, device=device)
    if name_l == "s1_symmetric_bc32":
        return build_model("s1_symmetric_bc32", base_channels=32, device=device)
    if name_l in {"student_s2"}:
        return build_model("student_s2", base_channels=12, device=device)
    if name_l in {"extranet"}:
        return build_model("extranet", base_channels=32, device=device)
    raise ValueError(f"Unknown model {name}")


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_macs(model, input_shape, device):
    if not TORCHINFO_AVAILABLE:
        return None
    try:
        # torchinfo counts multiply-adds; summary returns total_mult_adds
        info = torchinfo_summary(model, input_size=input_shape, col_names=("output_size", "num_params", "mult_adds"), verbose=0, device=device)
        return info.total_mult_adds
    except Exception:
        return None


# REMOVED: Full pipeline with pre/post-processing (use benchmark_model.py for full pipeline testing)
# def model_forward_pipeline(model, x):
#     # Complete log-domain pipeline with log normalize, tanh, hole mask, exp restore
#     # This function is kept for reference but not used in latency measurement
#     pass


def quantize_fp8_ptq(model: torch.nn.Module, input_shape: Tuple[int, ...], num_calib_samples: int = 32):
    """Apply FP8 PTQ quantization using NVIDIA Model Optimizer

    Args:
        model: PyTorch model to quantize
        input_shape: Input tensor shape for calibration
        num_calib_samples: Number of calibration samples

    Returns:
        Quantized model
    """
    if not MODELOPT_AVAILABLE:
        raise RuntimeError("nvidia-modelopt not available. Install with: pip install nvidia-modelopt")

    device = next(model.parameters()).device
    model.eval()

    # Prepare calibration data (random tensors)
    calib_data = [torch.randn(*input_shape, device=device) for _ in range(num_calib_samples)]

    # Define forward loop for calibration
    def forward_loop():
        with torch.no_grad():
            for data in calib_data:
                _ = model(data)

    # FP8 quantization configuration
    print("🔧 Applying FP8 PTQ quantization...")
    quant_cfg = mtq.FP8_DEFAULT_CFG
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    print("✅ FP8 quantization complete")
    return model


def export_onnx(model: torch.nn.Module, input_shape: Tuple[int, ...], output_path: str, fp16: bool = False, fp8: bool = False, opset: int = 18):
    """Export PyTorch model to ONNX format

    Args:
        model: PyTorch model to export
        input_shape: Input tensor shape (B, C, H, W)
        output_path: Output ONNX file path
        fp16: Export with FP16 precision
        fp8: Apply FP8 quantization before export
        opset: ONNX opset version (default 18 for better compatibility)
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    # Apply FP8 quantization if requested
    if fp8:
        model = quantize_fp8_ptq(model, input_shape)
        precision_str = "FP8"
    elif fp16:
        model = model.half()
        dummy_input = dummy_input.half()
        precision_str = "FP16"
    else:
        precision_str = "FP32"

    # Use legacy TorchScript exporter for compatibility (disable dynamo)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # Force legacy TorchScript exporter
    )
    print(f"✅ Exported {precision_str} ONNX to {output_path}")


def build_tensorrt_engine(onnx_path: str, engine_path: str, fp16: bool = True, fp8: bool = False, workspace_gb: int = 2):
    """Build TensorRT engine from ONNX file

    Args:
        onnx_path: Input ONNX file path
        engine_path: Output TRT engine file path
        fp16: Enable FP16 precision
        fp8: Enable FP8 precision (overrides fp16)
        workspace_gb: Max workspace size in GB

    Returns:
        TensorRT engine
    """
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT not available. Install with: pip install tensorrt")

    # Check if engine already exists
    if os.path.exists(engine_path):
        print(f"✅ Loading cached TRT engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    # Build new engine
    print(f"🔨 Building TRT engine from {onnx_path}...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('❌ Failed to parse ONNX file:')
            for error_idx in range(parser.num_errors):
                print(parser.get_error(error_idx))
            raise RuntimeError("ONNX parsing failed")

    # Build configuration
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # Set precision mode
    if fp8:
        # FP8 requires specific TensorRT version and GPU support
        try:
            config.set_flag(trt.BuilderFlag.FP8)
            print("✅ FP8 mode enabled")
        except AttributeError:
            print("⚠️ FP8 not supported, falling back to FP16")
            config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 mode enabled")

    # Build engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"✅ Saved TRT engine to {engine_path}")

    # Deserialize for use
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


def measure_tensorrt_latency(engine, input_shape: Tuple[int, ...], warmup: int, iters: int, fp16: bool = False):
    """Measure TensorRT inference latency using PyTorch CUDA tensors

    Args:
        engine: TensorRT engine
        input_shape: Input tensor shape (B, C, H, W)
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        fp16: Use FP16 precision

    Returns:
        Average latency in milliseconds
    """
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT not available")

    # Create execution context
    context = engine.create_execution_context()

    # Prepare input/output tensors on CUDA
    dtype = torch.float16 if fp16 else torch.float32
    device = torch.device('cuda')
    input_tensor = torch.randn(*input_shape, dtype=dtype, device=device)
    output_shape = (input_shape[0], 3, input_shape[2], input_shape[3])  # RGB output
    output_tensor = torch.empty(output_shape, dtype=dtype, device=device)

    # Get device pointers
    bindings = [input_tensor.data_ptr(), output_tensor.data_ptr()]

    # Warmup
    for _ in range(warmup):
        context.execute_v2(bindings=bindings)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        context.execute_v2(bindings=bindings)
    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1000.0 / iters

    return dt


def measure_latency_forward_only(model, input_shape: Tuple[int, int, int, int], device: str, warmup: int, iters: int) -> float:
    res = measure_forward_latency(model, input_shape, device=device, warmup=warmup, iters=iters, amp=False)
    return res.latency_ms


def main():
    ap = argparse.ArgumentParser(description="Profile model variants")
    ap.add_argument('--models', nargs='+', default=['v1','v2','student_s1','student_s2','extranet'],
                    help='model names: v1 v2 student_s1 student_s2 extranet')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--use-tensorrt', action='store_true', help='Use TensorRT for inference (requires ONNX export)')
    ap.add_argument('--fp16', action='store_true', help='Use FP16 precision (for TensorRT)')
    ap.add_argument('--fp8', action='store_true', help='Use FP8 quantization (requires Model Optimizer)')
    ap.add_argument('--onnx-dir', type=str, default='onnx/benchmark', help='Directory to save ONNX/TRT files')
    ap.add_argument('--opset', type=int, default=18, help='ONNX opset version (18 recommended for better compatibility)')
    args = ap.parse_args()

    device = args.device
    shapestr = f"{args.batch}x7x{args.height}x{args.width}"
    input_shape = (args.batch, 7, args.height, args.width)

    # Validate precision settings
    if args.fp8 and args.fp16:
        print("⚠️  Warning: Both --fp8 and --fp16 specified. FP8 takes precedence.")
        args.fp16 = False

    # Mode header
    mode_str = "TensorRT" if args.use_tensorrt else "PyTorch"
    if args.fp8:
        precision_str = "FP8"
    elif args.fp16:
        precision_str = "FP16"
    else:
        precision_str = "FP32"

    print("="*70)
    print(f"Model Variants Comparison - FORWARD-ONLY Benchmark ({mode_str} {precision_str})")
    print("="*70)
    print(f"Device: {device}, input: {shapestr}, warmup={args.warmup}, iters={args.iters}")
    print("⚠️  NOTE: Measuring pure model(input) time only (no pre/post-processing)")
    if args.use_tensorrt:
        if not TENSORRT_AVAILABLE:
            print("❌ ERROR: TensorRT not available. Install with: pip install tensorrt")
            return
        print(f"🔧 TensorRT mode: ONNX export → TRT engine build → FP16 inference")
        print(f"📁 ONNX/TRT files will be saved to: {args.onnx_dir}")
    print("="*70)

    if not torch.cuda.is_available() and device.startswith('cuda'):
        print("[WARN] CUDA not available, fallback to CPU")
        device = 'cpu'

    # Create ONNX directory if using TensorRT
    if args.use_tensorrt:
        Path(args.onnx_dir).mkdir(parents=True, exist_ok=True)

    for name in args.models:
        try:
            model = build_model_for_name(name, device)
        except Exception as e:
            print(f"\n[name={name}] build failed: {e}")
            continue

        total_params, trainable_params = count_params(model)
        macs = estimate_macs(model, input_shape, device)

        # Benchmark latency
        if args.use_tensorrt:
            # Determine precision suffix for file names
            if args.fp8:
                precision_suffix = 'fp8'
            elif args.fp16:
                precision_suffix = 'fp16'
            else:
                precision_suffix = 'fp32'

            # Export ONNX
            onnx_path = os.path.join(args.onnx_dir, f"{name}_{args.height}x{args.width}_{precision_suffix}.onnx")
            engine_path = os.path.join(args.onnx_dir, f"{name}_{args.height}x{args.width}_{precision_suffix}.trt")

            try:
                if not os.path.exists(onnx_path):
                    export_onnx(model, input_shape, onnx_path, fp16=args.fp16, fp8=args.fp8, opset=args.opset)
                else:
                    print(f"✅ Using cached ONNX: {onnx_path}")

                # Build TRT engine
                engine = build_tensorrt_engine(onnx_path, engine_path, fp16=args.fp16, fp8=args.fp8)

                # Measure latency
                latency = measure_tensorrt_latency(engine, input_shape, args.warmup, args.iters, fp16=(args.fp16 or args.fp8))
            except Exception as e:
                print(f"❌ TensorRT benchmark failed for {name}: {e}")
                continue
        else:
            # PyTorch inference
            latency = measure_latency_forward_only(model, input_shape, device, args.warmup, args.iters)

        print(f"\n=== {name} ===")
        print(f"Params: total={total_params:,} (trainable={trainable_params:,})")
        if macs is not None:
            gmac = macs / 1e9
            print(f"MACs (torchinfo est.): {gmac:.3f} GMAC")
        else:
            print("MACs: n/a (torchinfo not available)")

        backend_str = f"TensorRT {precision_str}" if args.use_tensorrt else f"PyTorch {precision_str}"
        print(f"Latency (forward-only): {latency:.3f} ms ({1000/latency:.1f} FPS) on {backend_str}")

if __name__ == '__main__':
    main()
