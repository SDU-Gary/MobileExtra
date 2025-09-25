#!/usr/bin/env python3
"""
HDR Data Range Probe

Purpose:
- Dry-check the numeric ranges along the full HDR pipeline, from raw EXR read →
  linear HDR preprocessing → residual target → 4x4 grid patches → network forward
  → reconstructed patch → tone-mapped LDR for perceptual.

What it prints:
- Per-frame: raw EXR ranges (ref/warp_hole/correct), linear-HDR ranges for input/target
- Residual target range and reconstruction consistency
- Per-patch (summarized + first K patches details): input channels, masks, residual pred,
  reconstructed, and tone-mapped ranges

Notes:
- This script does NOT require training; it uses current configs and uninitialized network
  weights to traverse the data flow and report ranges.
- If PyYAML is unavailable, it falls back to defaults and prints a warning.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# Local project imports (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'train'))

# Optional YAML config loading
def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        print(f"[WARN] PyYAML not available: {e}. Using built-in defaults.")
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Failed to read YAML at {path}: {e}. Using built-in defaults.")
        return None


def _get(d: Dict[str, Any], path: str, default: Any) -> Any:
    """Small helper to read nested config values, e.g., _get(cfg, 'hdr_processing.scale_factor', 0.7)."""
    cur: Any = d
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def format_range(t: torch.Tensor) -> str:
    t = t.detach()
    return f"[{t.min().item():.6f}, {t.max().item():.6f}]"


def main():
    # Load config (optional)
    cfg_path = REPO_ROOT / 'configs' / 'colleague_training_config.yaml'
    cfg = _load_yaml(cfg_path) or {}

    # Resolve basic settings
    data_root = _get(cfg, 'data.data_root', './data')
    processed_bistro_path = _get(cfg, 'data.processed_bistro_path', './data/processed_bistro')
    hdr_scale = float(_get(cfg, 'hdr_processing.scale_factor', 0.70))
    enable_srgb_linear = bool(_get(cfg, 'hdr_processing.enable_srgb_linear', False))
    tm_method = str(_get(cfg, 'hdr_processing.tone_mapping_for_display', 'reinhard'))
    tm_gamma = float(_get(cfg, 'hdr_processing.gamma', 2.2))
    tm_exposure = float(_get(cfg, 'hdr_processing.exposure', 1.0))
    tm_adapt = _get(cfg, 'hdr_processing.adaptive_exposure', {'enable': False})

    grid_rows = int(_get(cfg, 'patch.simple_grid_rows', 4))
    grid_cols = int(_get(cfg, 'patch.simple_grid_cols', 4))
    exp_H = int(_get(cfg, 'patch.simple_expected_height', 1080))
    exp_W = int(_get(cfg, 'patch.simple_expected_width', 1920))

    print("=== HDR Data Range Probe ===")
    print(f"Config: data_root={data_root}, processed_bistro={processed_bistro_path}")
    print(f"HDR: scale_factor={hdr_scale}, srgb→linear={enable_srgb_linear}")
    print(f"ToneMap: method={tm_method}, gamma={tm_gamma}, exposure={tm_exposure}, adaptive={tm_adapt}")
    print(f"Grid: {grid_rows}x{grid_cols}, expected={exp_H}x{exp_W}")
    # Input normalization (training-aligned)
    norm_cfg = _get(cfg, 'training.input_normalization', {})
    norm_enable = bool(_get(cfg, 'training.input_normalization.enable', False))
    norm_pct = float(_get(cfg, 'training.input_normalization.percentile', 0.99))
    norm_min = float(_get(cfg, 'training.input_normalization.min_scale', 1.0))
    norm_max = float(_get(cfg, 'training.input_normalization.max_scale', 512.0))
    print(f"InputNorm: enable={norm_enable}, pct={norm_pct}, min={norm_min}, max={norm_max}")
    # Global standardization
    gs_enable = bool(_get(cfg, 'training.global_standardization.enable', False))
    gs_stats_path = _get(cfg, 'training.global_standardization.stats_path', './configs/hdr_global_stats.json')
    gs_mu = None
    gs_sigma = None
    if gs_enable and Path(gs_stats_path).exists():
        try:
            import json
            with open(gs_stats_path, 'r', encoding='utf-8') as f:
                s = json.load(f)
            gs_mu = float(s.get('mu', 0.0))
            gs_sigma = float(s.get('sigma', 1.0))
        except Exception as e:
            print(f"[WARN] failed to read GS stats: {e}")
            gs_enable = False
    print(f"GlobalStd: enable={gs_enable}, mu={gs_mu}, sigma={gs_sigma}")
    # Log normalization (config-only display)
    norm_type = _get(cfg, 'normalization.type', 'none')
    log_eps = float(_get(cfg, 'normalization.log_epsilon', 1e-8))
    print(f"NormType: {norm_type}, log_epsilon={log_eps}")

    # Imports for pipeline
    from train.colleague_dataset_adapter import ColleagueDatasetAdapter
    from train.residual_learning_helper import ResidualLearningHelper
    from simple_patch_extractor import SimplePatchExtractor, SimpleGridConfig
    from src.npu.utils.hdr_vis import tone_map
    from src.npu.networks.patch.patch_network import PatchNetwork

    # Instantiate dataset adapter with HDR preprocessing enabled
    adapter = ColleagueDatasetAdapter(
        data_root=data_root,
        split='train',
        enable_linear_preprocessing=True,
        enable_srgb_linear=enable_srgb_linear,
        scale_factor=hdr_scale,
        tone_mapping_for_display=tm_method,
        gamma=tm_gamma,
        exposure=tm_exposure,
        adaptive_exposure=tm_adapt,
    )

    if len(adapter) == 0:
        print("No frames available.")
        return

    # Use the first frame for probing
    frame_id = adapter.frame_list[0]
    fp = adapter.file_mapping[frame_id]
    print(f"\n--- Probing Frame ID: {frame_id} ---")
    print(f"Paths: ref={fp['ref']}, warp_hole={fp['warp_hole']}, correct={fp['correct']}")

    # 1) Read raw EXR
    ref_raw = adapter._load_exr(fp['ref'])           # [3,H,W]
    warp_raw = adapter._load_exr(fp['warp_hole'])    # [3,H,W]
    correct_raw = adapter._load_exr(fp['correct'])   # [1,H,W] (or more → squeezed inside __getitem__)

    print(f"Raw ref      shape={tuple(ref_raw.shape)},  range={format_range(ref_raw)}")
    print(f"Raw warp     shape={tuple(warp_raw.shape)}, range={format_range(warp_raw)}")
    print(f"Raw correct  shape={tuple(correct_raw.shape)}, range={format_range(correct_raw)}")

    # 2) Build 7ch input (pre-preprocess), clamp correct to [0,1]
    H, W = ref_raw.shape[1], ref_raw.shape[2]
    input_raw = torch.zeros(7, H, W)
    input_raw[:3] = warp_raw[:3]
    corr1 = correct_raw[:1]
    input_raw[3:4] = torch.clamp(corr1, 0.0, 1.0)
    # occlusion + residual_mv zeros by design

    print(f"Input RAW 7ch: warped_rgb={format_range(input_raw[:3])}, holes={format_range(input_raw[3:4])}, occ={format_range(input_raw[4:5])}, mv={format_range(input_raw[5:7])}")

    # 3) Linear HDR preprocessing (same as adapter)
    input_lin = adapter._linear_hdr_preprocessing(input_raw)
    target_lin = adapter._linear_hdr_preprocessing(ref_raw)
    print(f"Input LIN 7ch: warped_rgb={format_range(input_lin[:3])}, holes={format_range(input_lin[3:4])}, occ={format_range(input_lin[4:5])}, mv={format_range(input_lin[5:7])}")
    print(f"Target LIN RGB: shape={tuple(target_lin.shape)}, range={format_range(target_lin)}")

    # 4) Residual target and reconstruction consistency
    warped_rgb = input_lin[:3]
    target_residual = ResidualLearningHelper.compute_residual_target(target_lin, warped_rgb)
    reconstructed = ResidualLearningHelper.reconstruct_from_residual(warped_rgb, target_residual)
    rec_err = torch.mean(torch.abs(reconstructed - target_lin)).item()
    print(f"Residual target range={format_range(target_residual)}, recon_err={rec_err:.8f}")

    # 5) 4x4 grid patches (resized to 128x128 in training, but here we keep native 270x480 per grid cell)
    # For consistency with training, we will resize to 128x128 per patch after extraction.
    grid_cfg = SimpleGridConfig(
        grid_rows=grid_rows, grid_cols=grid_cols,
        expected_height=exp_H, expected_width=exp_W,
        patch_height=exp_H // grid_rows, patch_width=exp_W // grid_cols,
        enable_size_validation=True, enable_debug_info=False,
    )
    extractor = SimplePatchExtractor(grid_cfg)

    inp_np = input_lin.numpy()           # [7,H,W]
    tgt_np = target_lin.numpy()          # [3,H,W]
    holes_np = input_lin[3].numpy()      # [H,W]

    inp_patches, positions = extractor.extract_patches(inp_np)
    tgt_patches, _ = extractor.extract_patches(tgt_np)

    # Convert to tensors and resize to [N,*,128,128] like training
    inp_ts = []
    tgt_ts = []
    for ip, tp in zip(inp_patches, tgt_patches):
        ipt = torch.from_numpy(ip).permute(2, 0, 1).float()   # [7,h,w]
        tpt = torch.from_numpy(tp).permute(2, 0, 1).float()   # [3,h,w]
        if ipt.shape[1:] != (128, 128):
            ipt = F.interpolate(ipt.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
            tpt = F.interpolate(tpt.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
        inp_ts.append(ipt)
        tgt_ts.append(tpt)
    inp_patches_t = torch.stack(inp_ts)  # [N,7,128,128]
    tgt_patches_t = torch.stack(tgt_ts)  # [N,3,128,128]

    print(f"\nPatches: N={inp_patches_t.shape[0]}, shape per patch={tuple(inp_patches_t.shape[2:])}")
    print(f"Patch input warped_rgb range={format_range(inp_patches_t[:, :3])}")
    print(f"Patch holes_mask range={format_range(inp_patches_t[:, 3:4])}")
    print(f"Patch target RGB range={format_range(tgt_patches_t)}")

    # 6) Patch network forward (untrained) + reconstruction
    net = PatchNetwork(input_channels=7, output_channels=3, base_channels=64)
    # Build boundary_override from target edges (like training_step)
    tgt_gray = torch.mean(tgt_patches_t, dim=1, keepdim=True)
    kernel = net._create_boundary_kernel()
    bmap = torch.sigmoid(torch.abs(F.conv2d(tgt_gray, kernel, padding=1)) * 2.0)
    with torch.no_grad():
        residual_pred = net(inp_patches_t, boundary_override=bmap)        # [N,3,128,128]
        reconstructed = inp_patches_t[:, :3] + residual_pred              # linear HDR
    print(f"Residual pred (no-norm) range={format_range(residual_pred)}")
    print(f"Reconstructed pred (no-norm) range={format_range(reconstructed)}")

    print("\n===== PER-PATCH NORMALIZATION (percentile) =====")
    # 6b) Per-patch normalization forward (always compute using config params)
    try:
        warped_rgb_p = inp_patches_t[:, :3]
        B = warped_rgb_p.shape[0]
        with torch.no_grad():
            try:
                q = torch.quantile(warped_rgb_p.view(B, -1), norm_pct, dim=1)
            except Exception:
                q = warped_rgb_p.view(B, -1).amax(dim=1)
            b = q.view(-1, 1, 1, 1)
            b = torch.clamp(b, min=norm_min, max=norm_max)
        # Stats for b
        b_vals = b.view(B)
        b_min = float(b_vals.min())
        b_max = float(b_vals.max())
        b_mean = float(b_vals.mean())
        print(f"Norm scales b stats: min={b_min:.6f}, mean={b_mean:.6f}, max={b_max:.6f}")
        # Normalized inputs
        inp_patches_norm = inp_patches_t.clone()
        inp_patches_norm[:, :3] = inp_patches_norm[:, :3] / b
        print(f"Patch input warped_rgb (norm) range={format_range(inp_patches_norm[:, :3])}")

        with torch.no_grad():
            residual_pred_norm = net(inp_patches_norm, boundary_override=bmap)
            reconstructed_norm = (warped_rgb_p / b + residual_pred_norm) * b
        print(f"Residual pred (norm) range={format_range(residual_pred_norm)}")
        print(f"Reconstructed pred (norm) range={format_range(reconstructed_norm)}")
        tm_pred_norm = tone_map(reconstructed_norm, method=tm_method, gamma=tm_gamma, exposure=tm_exposure, adaptive_exposure=tm_adapt)
        print(f"Tone-mapped pred LDR (norm) range={format_range(tm_pred_norm)} (expected within [0,1])")
    except Exception as e:
        print(f"[WARN] per-patch normalization path failed: {e}")

    print("\n===== GLOBAL STANDARDIZATION (mu/sigma) =====")
    # 6c) Global standardization forward (matches new training)
    if gs_mu is not None and gs_sigma is not None:
        mu = torch.tensor(gs_mu, dtype=inp_patches_t.dtype)
        sigma = torch.tensor(gs_sigma, dtype=inp_patches_t.dtype)
        warped_rgb_p = inp_patches_t[:, :3]
        Xn = (warped_rgb_p - mu) / torch.clamp(sigma, min=1e-6)
        inp_patches_gs = inp_patches_t.clone()
        inp_patches_gs[:, :3] = Xn
        with torch.no_grad():
            residual_pred_gs = net(inp_patches_gs, boundary_override=bmap)
            reconstructed_gs = (Xn + residual_pred_gs) * sigma + mu
        print(f"Residual pred (gs) range={format_range(residual_pred_gs)}")
        print(f"Reconstructed pred (gs) range={format_range(reconstructed_gs)}")
        tm_pred_gs = tone_map(reconstructed_gs, method=tm_method, gamma=tm_gamma, exposure=tm_exposure, adaptive_exposure=tm_adapt)
        print(f"Tone-mapped pred LDR (gs) range={format_range(tm_pred_gs)} (expected within [0,1])")
    else:
        print("[WARN] global stats not available; skip GS path.")

    print("\n===== LOG NORMALIZATION =====")
    # 6d) Log normalization forward (Scheme B: predict bounded delta in log space)
    try:
        eps = torch.tensor(log_eps, dtype=inp_patches_t.dtype)
        warped_rgb_p = inp_patches_t[:, :3]
        warped_pos = torch.clamp(warped_rgb_p, min=0.0)
        log_img = torch.log(warped_pos + eps)
        B = warped_rgb_p.shape[0]
        min_log = torch.amin(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
        max_log = torch.amax(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
        denom = torch.clamp(max_log - min_log, min=1e-6)
        Xn = (log_img - min_log) / denom
        inp_patches_log = inp_patches_t.clone()
        inp_patches_log[:, :3] = Xn
        with torch.no_grad():
            residual_pred_log = net(inp_patches_log, boundary_override=bmap)  # delta_log proxy
            beta = float(_get(cfg, 'normalization.log_delta_scale', 0.1))
            delta_log = (beta * denom) * torch.tanh(residual_pred_log)
            log_hat = log_img + delta_log
            reconstructed_log = torch.exp(log_hat) - eps
        print(f"Residual pred (log) range={format_range(residual_pred_log)}")
        print(f"Reconstructed pred (log) range={format_range(reconstructed_log)}")
        tm_pred_log = tone_map(reconstructed_log, method=tm_method, gamma=tm_gamma, exposure=tm_exposure, adaptive_exposure=tm_adapt)
        print(f"Tone-mapped pred LDR (log) range={format_range(tm_pred_log)} (expected within [0,1])")
    except Exception as e:
        print(f"[WARN] log normalization path failed: {e}")

    # 7) Tone-mapped LDR for perceptual (pred and target)
    tm_pred = tone_map(reconstructed, method=tm_method, gamma=tm_gamma, exposure=tm_exposure, adaptive_exposure=tm_adapt)
    tm_tgt = tone_map(tgt_patches_t, method=tm_method, gamma=tm_gamma, exposure=tm_exposure, adaptive_exposure=tm_adapt)
    print(f"Tone-mapped pred LDR (no-norm) range={format_range(tm_pred)} (expected within [0,1])")
    print(f"Tone-mapped tgt  LDR range={format_range(tm_tgt)} (expected within [0,1])")
    print("\n===== SUMMARY =====")
    print("Sections above show: \n- Baseline (no-norm)\n- Per-Patch Normalization\n- Global Standardization\n- Log Normalization")

    # 8) Optional: print first K patches detailed ranges
    K = min(4, inp_patches_t.shape[0])
    print(f"\nFirst {K} patch details:")
    for i in range(K):
        pi = inp_patches_t[i]
        tt = tgt_patches_t[i]
        rp = residual_pred[i]
        rc = reconstructed[i]
        print(f"- Patch {i:02d} @ pos=({positions[i].x},{positions[i].y}) {positions[i].width}x{positions[i].height}")
        print(f"  input warped_rgb: {format_range(pi[:3])}; holes: {format_range(pi[3:4])}; mv: {format_range(pi[5:7])}")
        print(f"  target RGB     : {format_range(tt)}")
        print(f"  residual pred  : {format_range(rp)}; reconstructed: {format_range(rc)}")

    print("\n[Done] HDR data range probe completed.")


if __name__ == '__main__':
    main()
