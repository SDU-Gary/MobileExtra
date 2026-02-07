#!/usr/bin/env python3
"""
网络性能分析工具 - 找出推理瓶颈

用法:
    python tools/profile_network.py --model models/colleague/last.ckpt --from-ckpt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os
from pathlib import Path
import time
from typing import Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.common.model_loading import load_model_from_ckpt, load_model_from_pth

# NOTE: This profiler includes pre/post processing timing. Keep its logic local
# to avoid benchmarking window mismatches in forward-only scripts.

def fused_log_normalize(x: torch.Tensor, eps: float = 1e-6):
    warped_pos = torch.clamp(x, min=0.0)
    log_img = torch.log(warped_pos + eps)
    B = log_img.shape[0]
    min_log = torch.amin(log_img.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)
    max_log = torch.amax(log_img.view(B, -1), dim=1, keepdim=True).view(B, 1, 1, 1)
    denom = torch.clamp(max_log - min_log, min=1e-6)
    Xn = (log_img - min_log) / denom
    return log_img, Xn, min_log, denom


def fused_exp_restore(log_output: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.exp(log_output) - eps


def fused_hole_mask_weight(
    holes_mask: torch.Tensor,
    target_h: int,
    target_w: int,
    kernel_size: int = 3,
    ring_scale: float = 0.5,
) -> torch.Tensor:
    if holes_mask.shape[2] != target_h or holes_mask.shape[3] != target_w:
        holes_mask = F.interpolate(holes_mask, size=(target_h, target_w), mode='nearest')
    pad = kernel_size // 2
    ring = F.max_pool2d(holes_mask, kernel_size=kernel_size, stride=1, padding=pad) - holes_mask
    ring = torch.clamp(ring, 0.0, 1.0)
    return torch.clamp(holes_mask + ring_scale * ring, 0.0, 1.0)

# Detect PatchNetworkV2 availability to pick correct architecture
try:
    from src.npu.networks.patch.patch_network_v2 import PatchNetworkV2
except Exception:
    PatchNetworkV2 = None

try:
    from torch.profiler import profile, record_function, ProfilerActivity
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False
    print("警告: torch.profiler不可用，将使用简单计时")


def simple_layer_profiling(model: nn.Module, input_tensor: torch.Tensor, device: str = 'cuda'):
    """简单的逐层计时分析"""
    print("\n" + "="*70)
    print("🔍 逐层性能分析（简单计时）")
    print("="*70)

    eps = 1e-6
    log_delta_abs_max = 16.0
    log_delta_alpha = 1.2

    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

        # 分阶段计时
        timings = {}

        # 1. 预处理阶段
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        warped_rgb = input_tensor[:, :3]
        log_img, Xn, min_log, denom = fused_log_normalize(warped_rgb, eps)
        input_norm = torch.cat([Xn, input_tensor[:, 3:]], dim=1)

        if device == 'cuda':
            torch.cuda.synchronize()
        timings['预处理（对数化+归一化）'] = (time.time() - t0) * 1000

        # 2. 网络推理阶段 - 总体
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        residual_pred_log = model(input_norm)

        if device == 'cuda':
            torch.cuda.synchronize()
        timings['网络推理（总体）'] = (time.time() - t0) * 1000

        # 3. 后处理阶段
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        delta_log = log_delta_alpha * torch.tanh(residual_pred_log) * log_delta_abs_max
        holes_mask = input_tensor[:, 3:4]
        mask_weight = fused_hole_mask_weight(
            holes_mask, log_img.shape[2], log_img.shape[3],
            kernel_size=3, ring_scale=0.5
        )
        delta_log = delta_log * mask_weight
        log_output = log_img + delta_log
        output_rgb = fused_exp_restore(log_output, eps)

        if device == 'cuda':
            torch.cuda.synchronize()
        timings['后处理（mask+指数还原）'] = (time.time() - t0) * 1000

    # 打印结果
    total_time = sum(timings.values())
    print(f"\n各阶段耗时统计:")
    for stage, t in timings.items():
        percentage = (t / total_time) * 100
        print(f"  {stage:30s}: {t:6.2f} ms ({percentage:5.1f}%)")
    print(f"  {'='*30}   {'='*6}    {'='*7}")
    print(f"  {'总计':30s}: {total_time:6.2f} ms (100.0%)")

    return timings


def detailed_module_profiling(model: nn.Module, input_tensor: torch.Tensor, device: str = 'cuda'):
    """详细的模块级性能分析"""
    print("\n" + "="*70)
    print("🔍 详细模块级分析（逐模块计时）")
    print("="*70)

    # 准备输入 (保持与推理流水一致)
    eps = 1e-6
    warped_rgb = input_tensor[:, :3]
    log_img, Xn, min_log, denom = fused_log_normalize(warped_rgb, eps)
    input_norm = torch.cat([Xn, input_tensor[:, 3:]], dim=1)

    model.eval()
    module_timings = {}

    def time_module(module, x, name, iterations=50):
        times = []
        out = None
        for _ in range(iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            out = module(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
        return out, sum(times) / len(times)

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_norm)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Branch: v1 具有 encoder_blocks/decoder_blocks；v2 采用显式模块名
        if hasattr(model, 'encoder_blocks') and hasattr(model, 'decoder_blocks'):
            print("\n分析 Encoder...")
            x = input_norm
            for i, enc_block in enumerate(model.encoder_blocks):
                x, t = time_module(enc_block, x, f'encoder_{i+1}')
                module_timings[f'Encoder Block {i+1}'] = t
                print(f"  Encoder Block {i+1}: {t:.3f} ms")

            print("\n分析 Bottleneck...")
            x, t = time_module(model.bottleneck, x, 'bottleneck')
            module_timings['Bottleneck (含Attention)'] = t
            print(f"  Bottleneck: {t:.3f} ms")

            # Decoder
            print("\n分析 Decoder...")
            encoder_outputs = []
            x_dec = input_norm
            for enc_block in model.encoder_blocks:
                x_dec = enc_block(x_dec)
                encoder_outputs.append(x_dec)

            x_dec = model.bottleneck(x_dec)

            for i, dec_block in enumerate(model.decoder_blocks):
                skip_idx = len(model.decoder_blocks) - 1 - i
                skip_connection = encoder_outputs[skip_idx]

                if device == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                x_up = F.interpolate(x_dec, size=skip_connection.shape[2:],
                                    mode='bilinear', align_corners=False)
                if device == 'cuda':
                    torch.cuda.synchronize()
                t_upsample = (time.time() - t0) * 1000

                x_concat = torch.cat([x_up, skip_connection], dim=1)
                x_dec, t = time_module(dec_block, x_concat, f'decoder_{i+1}')

                module_timings[f'Decoder Block {i+1} (Upsample)'] = t_upsample
                module_timings[f'Decoder Block {i+1} (Conv)'] = t
                print(f"  Decoder Block {i+1} Upsample: {t_upsample:.3f} ms")
                print(f"  Decoder Block {i+1} Conv: {t:.3f} ms")

            print("\n分析 Output Conv...")
            x_dec, t = time_module(model.output_conv, x_dec, 'output_conv')
            module_timings['Output Conv'] = t
            print(f"  Output Conv: {t:.3f} ms")

        elif PatchNetworkV2 is not None and isinstance(model, PatchNetworkV2):
            print("\n分析 Encoder (V2)...")
            boundary_mask = model._generate_boundary_mask(input_tensor)

            # encoder
            x1, t = time_module(lambda x: model.encoder1(x, boundary_mask), model.input_proj(model.input_padding(input_tensor), boundary_mask), 'encoder1')
            module_timings['Encoder1 (DS+Hardswish)'] = t
            d1, _ = time_module(lambda x: model.down1(x, boundary_mask), x1, 'down1', iterations=10)

            x2, t = time_module(lambda x: model.encoder2(x, boundary_mask), d1, 'encoder2')
            module_timings['Encoder2'] = t
            d2, _ = time_module(lambda x: model.down2(x, boundary_mask), x2, 'down2', iterations=10)

            x3, t = time_module(lambda x: model.encoder3(x, boundary_mask), d2, 'encoder3')
            module_timings['Encoder3'] = t
            d3, _ = time_module(lambda x: model.down3(x, boundary_mask), x3, 'down3', iterations=10)

            x4, t = time_module(lambda x: model.encoder4(x, boundary_mask), d3, 'encoder4')
            module_timings['Encoder4'] = t
            d4, _ = time_module(lambda x: model.down4(x, boundary_mask), x4, 'down4', iterations=10)

            x5, t = time_module(model.encoder5, d4, 'encoder5')
            module_timings['Encoder5'] = t

            print("\n分析 Bottleneck (V2)...")
            bn_out, t = time_module(model.bottleneck, x5, 'bottleneck')
            module_timings['Bottleneck (Attention)'] = t

            print("\n分析 Decoder (V2)...")
            # Decoder1
            u1, _ = time_module(model.up1, bn_out, 'up1', iterations=10)
            if u1.shape[2:] != x4.shape[2:]:
                u1 = F.interpolate(u1, size=x4.shape[2:], mode='bilinear', align_corners=False)
            u1_cat = torch.cat([u1, x4], dim=1)
            u1_conv, t = time_module(lambda x: model.up_conv1(x, boundary_mask), u1_cat, 'up_conv1')
            module_timings['UpConv1'] = t
            u1_dec, t = time_module(lambda x: model.decoder1(x, boundary_mask), u1_conv, 'decoder1')
            module_timings['Decoder1'] = t

            # Decoder2
            u2, _ = time_module(model.up2, u1_dec, 'up2', iterations=10)
            if u2.shape[2:] != x3.shape[2:]:
                u2 = F.interpolate(u2, size=x3.shape[2:], mode='bilinear', align_corners=False)
            u2_cat = torch.cat([u2, x3], dim=1)
            u2_conv, t = time_module(lambda x: model.up_conv2(x, boundary_mask), u2_cat, 'up_conv2')
            module_timings['UpConv2'] = t
            u2_dec, t = time_module(lambda x: model.decoder2(x, boundary_mask), u2_conv, 'decoder2')
            module_timings['Decoder2'] = t

            # Decoder3
            u3, _ = time_module(model.up3, u2_dec, 'up3', iterations=10)
            if u3.shape[2:] != x2.shape[2:]:
                u3 = F.interpolate(u3, size=x2.shape[2:], mode='bilinear', align_corners=False)
            u3_cat = torch.cat([u3, x2], dim=1)
            u3_conv, t = time_module(lambda x: model.up_conv3(x, boundary_mask), u3_cat, 'up_conv3')
            module_timings['UpConv3'] = t
            u3_dec, t = time_module(lambda x: model.decoder3(x, boundary_mask), u3_conv, 'decoder3')
            module_timings['Decoder3'] = t

            # Decoder4
            u4, _ = time_module(model.up4, u3_dec, 'up4', iterations=10)
            if u4.shape[2:] != x1.shape[2:]:
                u4 = F.interpolate(u4, size=x1.shape[2:], mode='bilinear', align_corners=False)
            u4_cat = torch.cat([u4, x1], dim=1)
            u4_conv, t = time_module(lambda x: model.up_conv4(x, boundary_mask), u4_cat, 'up_conv4')
            module_timings['UpConv4'] = t
            u4_dec, t = time_module(lambda x: model.decoder4(x, boundary_mask), u4_conv, 'decoder4 (DS+Hardswish)')
            module_timings['Decoder4'] = t

            # Output projection
            out_expand, t = time_module(model.output_expand, u4_dec, 'output_expand')
            module_timings['Output Expand (24->8)'] = t
            out_act, t = time_module(model.output_expand_act, out_expand, 'output_expand_act', iterations=50)
            module_timings['Output Expand Act'] = t
            out_final, t = time_module(model.output_conv, out_act, 'output_conv')
            module_timings['Output Conv (8->3)'] = t
            print(f"  Output Conv: {t:.3f} ms")

        else:
            print("当前模型结构未适配详细模块分析，跳过")
            return module_timings

    # 打印总结
    print("\n" + "="*70)
    print("📊 模块耗时总结（按耗时排序）")
    print("="*70)

    sorted_timings = sorted(module_timings.items(), key=lambda x: x[1], reverse=True)
    total_time = sum(module_timings.values())

    print(f"\n{'模块名称':40s}  {'耗时(ms)':>10s}  {'占比(%)':>8s}")
    print("-" * 70)
    for name, t in sorted_timings:
        percentage = (t / total_time) * 100
        print(f"{name:40s}  {t:10.3f}  {percentage:8.1f}%")
    print("-" * 70)
    print(f"{'总计':40s}  {total_time:10.3f}  {100.0:8.1f}%")

    return module_timings


def pytorch_profiler_analysis(model: nn.Module, input_tensor: torch.Tensor, device: str = 'cuda'):
    """使用PyTorch Profiler进行详细分析"""
    if not PROFILER_AVAILABLE:
        print("\n⚠️  PyTorch Profiler不可用，跳过此分析")
        return None

    print("\n" + "="*70)
    print("🔍 PyTorch Profiler 详细分析")
    print("="*70)

    # 准备输入
    eps = 1e-6
    warped_rgb = input_tensor[:, :3]
    log_img, Xn, min_log, denom = fused_log_normalize(warped_rgb, eps)
    input_norm = torch.cat([Xn, input_tensor[:, 3:]], dim=1)

    model.eval()

    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = model(input_norm)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Profiling
        activities = [ProfilerActivity.CPU]
        if device == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
        ) as prof:
            with record_function("model_inference"):
                for _ in range(10):
                    _ = model(input_norm)
                    if device == 'cuda':
                        torch.cuda.synchronize()

        # 打印结果
        print("\n按CUDA时间排序的Top 20操作:")
        print(prof.key_averages().table(sort_by="cuda_time_total" if device == 'cuda' else "cpu_time_total", row_limit=20))

        print("\n按CPU时间排序的Top 20操作:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        # 保存详细trace
        trace_path = "output/profile_trace.json"
        Path("output").mkdir(exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"\n✅ 详细trace已保存至: {trace_path}")
        print(f"   使用Chrome浏览器打开 chrome://tracing 查看")

        return prof


def analyze_convolution_cost(model: nn.Module, input_shape: tuple):
    """分析卷积操作的计算成本"""
    print("\n" + "="*70)
    print("🔍 卷积操作FLOPs分析")
    print("="*70)

    B, C, H, W = input_shape

    # 当前FLOPs估计主要针对V1（拥有 encoder_channels/decoder_channels 和 PatchGatedConvBlock）。
    if not hasattr(model, 'encoder_channels') or not hasattr(model, 'decoder_channels'):
        print("\n⚠️  FLOPs 分析暂未适配该模型结构（可能是 PatchNetworkV2），已跳过")
        return None, {}

    def conv2d_flops(in_c, out_c, kernel_size, h, w, groups=1):
        """计算Conv2d的FLOPs"""
        # FLOPs = 2 * C_in * C_out * K_h * K_w * H_out * W_out / groups
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        return 2 * in_c * out_c * k_h * k_w * h * w / groups

    def gated_conv_flops(in_c, out_c, kernel_size, h, w):
        """计算Gated Conv的FLOPs (feature + mask)"""
        feature_flops = conv2d_flops(in_c, out_c, kernel_size, h, w)
        mask_flops = conv2d_flops(in_c, out_c, kernel_size, h, w)
        return feature_flops + mask_flops

    total_flops = 0
    layer_flops = {}

    # 分析网络结构
    print(f"\n输入尺寸: {input_shape}")
    print(f"\nbase_channels = {model.base_channels}")

    # Encoder
    print("\nEncoder FLOPs:")
    current_h, current_w = H, W
    current_c = C

    for i, ch in enumerate(model.encoder_channels):
        # PatchGatedConvBlock: 两个gated conv
        flops1 = gated_conv_flops(current_c, ch, 3, current_h, current_w)
        flops2 = gated_conv_flops(ch, ch, 3, current_h, current_w)
        block_flops = flops1 + flops2

        layer_flops[f'Encoder {i+1}'] = block_flops
        total_flops += block_flops

        print(f"  Encoder {i+1} ({current_c}→{ch}, {current_h}x{current_w}): {block_flops/1e9:.3f} GFLOPs")

        # Downsample
        if i < len(model.encoder_channels) - 1:
            current_h //= 2
            current_w //= 2
        current_c = ch

    # Bottleneck
    print("\nBottleneck FLOPs:")
    bn_ch = model.encoder_channels[-1]
    bn_flops = gated_conv_flops(bn_ch, bn_ch, 3, current_h, current_w)

    # Attention (SeparableAttention2D)
    # H attention: AdaptiveAvgPool + Conv1x1 (hidden_dim) + Conv1x1 (ch)
    hidden_dim = max(bn_ch // 8, 8)
    h_att_flops = conv2d_flops(bn_ch, hidden_dim, 1, current_h, 1) + \
                  conv2d_flops(hidden_dim, bn_ch, 1, current_h, 1)
    w_att_flops = conv2d_flops(bn_ch, hidden_dim, 1, 1, current_w) + \
                  conv2d_flops(hidden_dim, bn_ch, 1, 1, current_w)
    c_att_flops = conv2d_flops(bn_ch, hidden_dim, 1, 1, 1) + \
                  conv2d_flops(hidden_dim, bn_ch, 1, 1, 1)

    # Feature fusion + output gate
    fusion_flops = conv2d_flops(bn_ch, bn_ch//2, 1, current_h, current_w) + \
                   conv2d_flops(bn_ch//2, bn_ch, 1, current_h, current_w)
    gate_flops = conv2d_flops(bn_ch, bn_ch, 1, current_h, current_w)

    attention_total = h_att_flops + w_att_flops + c_att_flops + fusion_flops + gate_flops

    layer_flops['Bottleneck Conv'] = bn_flops
    layer_flops['Bottleneck Attention'] = attention_total
    total_flops += bn_flops + attention_total

    print(f"  Bottleneck Conv: {bn_flops/1e9:.3f} GFLOPs")
    print(f"  Bottleneck Attention: {attention_total/1e9:.3f} GFLOPs")

    # Decoder
    print("\nDecoder FLOPs:")
    for i, ch in enumerate(model.decoder_channels):
        if i == 0:
            in_ch = bn_ch * 2  # concat with skip
        else:
            in_ch = model.decoder_channels[i-1] * 2

        # Upsample (bilinear没有learnable params，计算量小)
        current_h *= 2
        current_w *= 2

        # PatchGatedConvBlock
        flops1 = gated_conv_flops(in_ch, ch, 3, current_h, current_w)
        flops2 = gated_conv_flops(ch, ch, 3, current_h, current_w)
        block_flops = flops1 + flops2

        layer_flops[f'Decoder {i+1}'] = block_flops
        total_flops += block_flops

        print(f"  Decoder {i+1} ({in_ch}→{ch}, {current_h}x{current_w}): {block_flops/1e9:.3f} GFLOPs")

    # Output Conv
    output_flops = gated_conv_flops(model.decoder_channels[-1], 3, 3, current_h, current_w)
    layer_flops['Output Conv'] = output_flops
    total_flops += output_flops
    print(f"\nOutput Conv: {output_flops/1e9:.3f} GFLOPs")

    # 总结
    print("\n" + "="*70)
    print(f"总FLOPs: {total_flops/1e9:.3f} GFLOPs")
    print("="*70)

    # 按FLOPs排序
    print("\n各层FLOPs占比（按降序）:")
    sorted_flops = sorted(layer_flops.items(), key=lambda x: x[1], reverse=True)
    for name, flops in sorted_flops[:10]:
        percentage = (flops / total_flops) * 100
        print(f"  {name:25s}: {flops/1e9:8.3f} GFLOPs ({percentage:5.1f}%)")

    return total_flops, layer_flops


def main():
    parser = argparse.ArgumentParser(description='网络性能分析工具')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (.pth or .ckpt)')
    parser.add_argument('--from-ckpt', action='store_true',
                       help='从checkpoint加载')
    parser.add_argument('--network-type', type=str, default='v1', choices=['v1', 'v2', 'patchnetworkv2', 'patch_network_v2'],
                       help='选择网络架构：v1=PatchNetwork，v2=PatchNetworkV2')
    parser.add_argument('--base-channels', type=int, default=24,
                       help='网络base_channels (default: 24)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[1080, 1920],
                       metavar=('H', 'W'), help='输入尺寸 (default: 1080 1920)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--skip-pytorch-profiler', action='store_true',
                       help='跳过PyTorch Profiler（如果太慢）')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        args.device = 'cpu'

    print(f"🔧 设备: {args.device.upper()}")
    if args.device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    try:
        if args.from_ckpt:
            model = load_model_from_ckpt(str(model_path), args.base_channels, args.device, args.network_type)
        else:
            model = load_model_from_pth(str(model_path), args.base_channels, args.device, args.network_type)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create test input
    H, W = args.input_size
    input_tensor = torch.randn(1, 7, H, W, device=args.device)

    print(f"\n测试配置:")
    print(f"  输入尺寸: 1 × 7 × {H} × {W}")
    print(f"  总像素数: {H * W:,}")
    print(f"  Base channels: {args.base_channels}")

    # 1. 简单分阶段计时
    simple_layer_profiling(model, input_tensor, args.device)

    # 2. 详细模块级分析
    detailed_module_profiling(model, input_tensor, args.device)

    # 3. FLOPs分析
    analyze_convolution_cost(model, input_tensor.shape)

    # 4. PyTorch Profiler (可选)
    if not args.skip_pytorch_profiler:
        pytorch_profiler_analysis(model, input_tensor, args.device)
    else:
        print("\n⏭️  跳过PyTorch Profiler分析")

    print("\n" + "="*70)
    print("✅ 性能分析完成")
    print("="*70)


if __name__ == "__main__":
    main()
