#!/usr/bin/env python3
"""
Half-Chain HDR Debug Tool (No Network)

目的：在不引入网络学习的前提下，复现训练管线中“线性HDR→规范化→（对数或统计域）残差→重建→tone-map”的数据流，
逐步导出中间结果（输入/目标/重建与关键中间量），用于定位视觉不一致问题。

特性：
- 从 YAML 读取 hdr_processing 与 normalization 统一配置
- 使用 ColleagueDatasetAdapter 的线性 HDR 预处理（与训练一致）
- 三种规范化路径模拟：log / global / per_patch（与训练 PatchFrameInterpolationTrainer 对齐）
- 残差使用“理想GT残差”（无网络）：
  * log: delta_log = target_log - log_img（可选仅在洞+边缘的掩码内生效）
  * global: residual_norm = (target-mu)/sigma - (warped-mu)/sigma
  * per_patch: residual_norm = target/b - warped/b
- 将 warped、target、reconstruct 分别按 hdr_vis.tone_map（支持 mulaw）保存为 LDR PNG

用法：
  python tools/test_mulaw.py --config ./configs/colleague_training_config.yaml \
                             --out-dir ./logs/mulaw_half_chain
"""

import os
import sys
import argparse
import yaml

import torch
import json


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """Clamp to [0,1] and convert to uint8 CHW tensor."""
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img


def save_png(chw01: torch.Tensor, path: str) -> None:
    """Save CHW [0,1] to PNG using PIL."""
    from PIL import Image
    img_u8 = to_uint8(chw01).cpu()  # [3,H,W]
    img_hwc = img_u8.permute(1, 2, 0).contiguous().numpy()
    Image.fromarray(img_hwc, mode='RGB').save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--out-dir', type=str, default='./logs/mulaw_half_chain')
    # Step-2 debugging switches for log normalization path
    p.add_argument('--log-disable-mask-weighting', action='store_true',
                   help='In log mode, do NOT weight delta by holes+ring mask (apply delta everywhere)')
    p.add_argument('--log-disable-minmax-norm', action='store_true',
                   help='In log mode, do NOT use per-image min/max normalization of log_img')
    # Tone-map probing
    p.add_argument('--no-gamma', action='store_true', help='Force gamma=1.0 to isolate gamma influence')
    p.add_argument('--override-mu', type=float, default=None, help='Override mu for mu-law to probe curve effect')
    p.add_argument('--tm-mode', type=str, default=None, choices=['mulaw','mulaw_luma','reinhard'],
                   help='Force tone-map mode for probing; mulaw_luma applies mu-law on luminance then rescales RGB')
    args = p.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_root = cfg.get('data', {}).get('data_root', './data')
    hdr = cfg.get('hdr_processing', {})
    tm_method = str(hdr.get('tone_mapping_for_display', 'mulaw')).lower()
    mu = float(hdr.get('mulaw_mu', 500.0))
    gamma = float(hdr.get('gamma', 2.2))
    exposure = float(hdr.get('exposure', 1.0))
    adaptive = hdr.get('adaptive_exposure', {'enable': False})
    # Keep preprocessing flags consistent with training pipeline
    enable_linear_preprocessing = bool(hdr.get('enable_linear_preprocessing', True))
    enable_srgb_linear = bool(hdr.get('enable_srgb_linear', False))
    scale_factor = float(hdr.get('scale_factor', 0.70))

    # Import dataset adapter and tone-map
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from train.colleague_dataset_adapter import ColleagueDatasetAdapter
    try:
        from src.npu.utils.hdr_vis import tone_map as tone_map
    except Exception:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
        from hdr_vis import tone_map as tone_map

    print('=== Half-Chain HDR Debug (No Network) ===')
    print(f' data_root: {data_root}')
    # Apply overrides for probing
    if args.no_gamma:
        gamma = 1.0
    if args.override_mu is not None:
        mu = float(args.override_mu)
    if args.tm_mode is not None:
        tm_method = args.tm_mode
    print(f' tone_mapping_for_display: {tm_method}, mu={mu}, exposure={exposure}, gamma={gamma}, adaptive={adaptive}')
    print(f' preprocessing: enable_linear_preprocessing={enable_linear_preprocessing}, enable_srgb_linear={enable_srgb_linear}, scale_factor={scale_factor}')
    norm = cfg.get('normalization', {}) or {}
    print(' normalization:', norm)

    # Build dataset adapter (train split)
    adapter = ColleagueDatasetAdapter(
        data_root=data_root,
        split='train',
        # preprocessing flags (match training adapter_kwargs)
        enable_linear_preprocessing=enable_linear_preprocessing,
        enable_srgb_linear=enable_srgb_linear,
        scale_factor=scale_factor,
        # display/tone-map params (for potential internal denorm usage)
        tone_mapping_for_display=tm_method,
        gamma=gamma,
        exposure=exposure,
        adaptive_exposure=adaptive,
        mulaw_mu=mu,
        augmentation=False,
    )

    # Get first sample: returns (input, target_residual, target_rgb)
    input_tensor, target_residual, target_rgb = adapter[0]
    H, W = target_rgb.shape[1], target_rgb.shape[2]
    print(f' first sample: target_rgb {tuple(target_rgb.shape)} range=({float(target_rgb.min()):.4f}, {float(target_rgb.max()):.4f}), '
          f'input_rgb range=({float(input_tensor[:3].min()):.4f}, {float(input_tensor[:3].max()):.4f})')

    # Prepare tone-mapper
    try:
        from src.npu.utils.hdr_vis import tone_map
    except Exception:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
        from hdr_vis import tone_map

    def tm(x: torch.Tensor) -> torch.Tensor:
        # x: [3,H,W] linear HDR
        if tm_method == 'mulaw_luma':
            # Luminance-based mu-law: preserve chroma by mapping Y and scaling RGB
            R, G, B = x[0], x[1], x[2]
            Y = 0.2126*R + 0.7152*G + 0.0722*B
            # exposure
            exp = float(exposure)
            if isinstance(adaptive, dict) and adaptive.get('enable', False):
                # Simple adaptive exposure on luminance
                try:
                    target_lum = float(adaptive.get('target_luminance', 0.18))
                    perc = float(adaptive.get('percentile', 0.9))
                    q = torch.quantile(Y.view(-1), perc)
                    exp = exp * float(target_lum / max(q.item(), 1e-6))
                except Exception:
                    pass
            Y_exp = torch.clamp(Y * exp, min=0.0)
            # mu-law on luminance only
            denom = torch.log(torch.tensor(1.0 + mu, dtype=Y.dtype))
            Y_ldr = torch.log(1.0 + mu * Y_exp) / denom
            # optional gamma after mapping
            if gamma != 1.0:
                Y_ldr = torch.pow(torch.clamp(Y_ldr, 1e-8, 1.0), 1.0/gamma)
            Y_ldr = torch.clamp(Y_ldr, 0.0, 1.0)
            # scale RGB to match new luminance
            eps = 1e-6
            s = (Y_ldr / torch.clamp(Y, min=eps)).clamp(0.0, 10.0)
            out = torch.stack([R*s, G*s, B*s], dim=0)
            out = torch.clamp(out, 0.0, 1.0)
            return out
        # default: use library tone_map
        y = tone_map(x, method=tm_method, mu=mu, gamma=gamma, exposure=exposure, adaptive_exposure=adaptive)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.clamp(y, 0.0, 1.0)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Split components
    warped = input_tensor[:3].clone()            # linear HDR
    holes = input_tensor[3:4].clone()            # [1,H,W]
    target = target_rgb.clone()                  # linear HDR

    # Baseline stats before any tone-map (sanity: should often exceed 1.0 for HDR)
    def _stats(name: str, t: torch.Tensor):
        print(f"[STATS] {name}: min={float(t.min()):.6f}, max={float(t.max()):.6f}, mean={float(t.mean()):.6f}")
        if float(t.max()) <= 1.1 and float(t.min()) >= 0.0:
            print(f"[WARN] {name} looks like LDR/normalized already (0..~1) before tone-map; risk of double tone-map")

    _stats('warped_linear_HDR', warped)
    _stats('target_linear_HDR', target)

    # Save baseline tone-mapped images
    save_png(tm(warped), os.path.join(out_dir, '01_warped_tm.png'))
    save_png(tm(target), os.path.join(out_dir, '02_target_tm.png'))
    save_png(torch.clamp(holes.expand(3, H, W), 0.0, 1.0), os.path.join(out_dir, '00_holes.png'))

    # Reconstruct via normalization path (GT residual, no network)
    norm_type = str(norm.get('type', 'none')).lower()

    recon = None
    if norm_type == 'global':
        # Global standardization
        stats_path = norm.get('stats_path') or cfg.get('training', {}).get('global_standardization', {}).get('stats_path')
        mu_g = 0.0; sigma_g = 1.0
        if stats_path and os.path.exists(stats_path):
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    s = json.load(f)
                mu_g = float(s.get('mu', 0.0)); sigma_g = float(s.get('sigma', 1.0))
            except Exception as e:
                print('[WARN] failed to read stats_path:', e)
        mu_t = torch.as_tensor(mu_g, dtype=warped.dtype)
        sigma_t = torch.as_tensor(max(sigma_g, 1e-6), dtype=warped.dtype)
        Xn = (warped - mu_t) / sigma_t
        Yn = (target - mu_t) / sigma_t
        residual_norm_gt = Yn - Xn
        recon = (Xn + residual_norm_gt) * sigma_t + mu_t
        print('[INFO] global recon range:', float(recon.min()), float(recon.max()))

    elif norm_type == 'per_patch':
        # Per-patch percentile scale (single image)
        perc = float(norm.get('percentile', 0.99))
        b = torch.quantile(warped.view(-1), perc)
        b = float(max(min(b.item(), float(norm.get('max_scale', 512.0))), float(norm.get('min_scale', 1.0))))
        b_t = torch.as_tensor(b, dtype=warped.dtype)
        Xn = warped / b_t
        Yn = target / b_t
        residual_norm_gt = Yn - Xn
        recon = (Xn + residual_norm_gt) * b_t
        print('[INFO] per_patch recon range:', float(recon.min()), float(recon.max()))

    elif norm_type == 'log':
        # True log-space path
        log_epsilon = float(norm.get('log_epsilon', 1.0e-6))
        eps_t = torch.as_tensor(log_epsilon, dtype=warped.dtype)
        warped_pos = torch.clamp(warped, min=0.0)
        target_pos = torch.clamp(target, min=0.0)
        log_img = torch.log(warped_pos + eps_t)
        target_log = torch.log(target_pos + eps_t)
        # Per-image min/max like training (can be disabled via flag)
        if not args.log_disable_minmax_norm:
            min_log = torch.amin(log_img.view(-1), dim=0)
            max_log = torch.amax(log_img.view(-1), dim=0)
            denom = torch.clamp(max_log - min_log, min=1e-6)
            Xn = (log_img - min_log) / denom  # for debug visualization
        gt_delta_log = target_log - log_img

        # Optional: only apply delta in holes (+ ring) (can be disabled via flag)
        if (not args.log_disable_mask_weighting) and bool(norm.get('log_apply_delta_in_holes', True)):
            try:
                ring_scale = float(norm.get('log_delta_mask_ring_scale', 0.5))
                k = int(norm.get('log_delta_mask_ring_kernel', 3))
                k = max(3, k if k % 2 == 1 else k + 1)
                import torch.nn.functional as F
                pad = k // 2
                ring = F.max_pool2d(holes.unsqueeze(0), kernel_size=k, stride=1, padding=pad) - holes.unsqueeze(0)
                ring = torch.clamp(ring, 0.0, 1.0)
                mask_weight = torch.clamp(holes.unsqueeze(0) + ring_scale * ring, 0.0, 1.0).squeeze(0)
                gt_delta_log = gt_delta_log * mask_weight
            except Exception as e:
                print('[WARN] log holes mask weighting failed:', e)
        Ln_hat = log_img + gt_delta_log
        recon = torch.exp(Ln_hat) - eps_t
        print('[INFO] log recon range:', float(recon.min()), float(recon.max()))

    else:
        # None: identity residual (for reference)
        recon = warped.clone()
        print('[INFO] none recon (identity)')

    # Save recon tone-mapped + intermediate debug maps
    save_png(tm(recon), os.path.join(out_dir, '03_recon_tm.png'))

    # Also dump Xn visualization (if available) as 0..1 PNG for reference
    try:
        if 'Xn' in locals():
            Xn_v = torch.clamp((Xn - float(Xn.min())) / max(float((Xn.max() - Xn.min())), 1e-6), 0.0, 1.0)
            save_png(Xn_v, os.path.join(out_dir, '10_Xn_debug.png'))
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
