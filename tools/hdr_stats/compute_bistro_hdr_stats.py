#!/usr/bin/env python3
"""
Compute HDR pixel-distribution statistics for processed_bistro.

- Scans EXR files under data_root/processed_bistro/{ref,warp_hole}
- Loads RGB as float32 [C,H,W]
- Computes per-file and aggregated stats with subsampling
- Outputs JSON and optional PNG histograms to logs/hdr_stats

Usage:
  python tools/hdr_stats/compute_bistro_hdr_stats.py \
    --data-root ./data \
    --max-files 200 \
    --sample-per-file 200000 \
    --save-hist

Notes:
- Uses OpenEXR if available; falls back to OpenCV (cv2) if installed.
- Assumes EXR RGB is linear HDR; if not sure, this still provides useful relative stats.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

# Try optional deps
try:
    import OpenEXR  # type: ignore
    import Imath   # type: ignore
    import array   # type: ignore
    OPENEXR_AVAILABLE = True
except Exception:
    OPENEXR_AVAILABLE = False

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# Matplotlib is optional, only for histograms
try:
    import matplotlib.pyplot as plt  # type: ignore
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False


def load_exr(path: Path) -> np.ndarray:
    """Load EXR as [C,H,W] float32, prefer RGB order if present."""
    if not path.exists():
        raise FileNotFoundError(path)

    if OPENEXR_AVAILABLE:
        try:
            return _load_exr_openexr(path)
        except Exception as e:
            print(f"[WARN] OpenEXR failed for {path}: {e}")

    if CV2_AVAILABLE:
        try:
            return _load_exr_opencv(path)
        except Exception as e:
            print(f"[WARN] OpenCV failed for {path}: {e}")

    raise RuntimeError(f"No EXR backend available to read: {path}")


def _load_exr_openexr(path: Path) -> np.ndarray:
    f = OpenEXR.InputFile(str(path))
    header = f.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = list(header['channels'].keys())
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    data = []
    if all(c in channels for c in ['R','G','B']):
        order = ['R','G','B']
    else:
        order = sorted(channels)

    for c in order[:3]:
        raw = f.channel(c, pixel_type)
        arr = np.array(array.array('f', raw), dtype=np.float32).reshape(height, width)
        data.append(arr)

    f.close()
    if not data:
        raise ValueError(f"No channels in {path}")

    return np.stack(data, axis=0)


def _load_exr_opencv(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"cv2.imread failed: {path}")
    if img.ndim == 2:
        return img[np.newaxis, ...].astype(np.float32)
    # BGR -> RGB, HWC -> CHW
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chw = np.transpose(img, (2,0,1)).astype(np.float32)
    return chw


def luminance_from_rgb(rgb_chw: np.ndarray) -> np.ndarray:
    """Compute luminance Y from linear RGB: Y = 0.2126 R + 0.7152 G + 0.0722 B"""
    if rgb_chw.shape[0] < 3:
        # Fallback: single channel
        return rgb_chw[0]
    R, G, B = rgb_chw[0], rgb_chw[1], rgb_chw[2]
    return 0.2126 * R + 0.7152 * G + 0.0722 * B


def robust_percentiles(arr: np.ndarray, ps: List[float]) -> Dict[str, float]:
    vals = {}
    flat = arr.reshape(-1)
    for p in ps:
        vals[f"p{int(p)}"] = float(np.percentile(flat, p))
    return vals


def file_stats(rgb: np.ndarray, rng: np.random.RandomState, sample_per_file: int) -> Dict:
    """Compute per-file stats and return subsamples for aggregation."""
    C, H, W = rgb.shape
    total = H * W
    flat = rgb.reshape(C, -1)

    # Channel-wise stats
    stats_c = []
    for c in range(min(3, C)):
        ch = flat[c]
        neg = float((ch < 0).mean())
        zero = float((ch == 0).mean())
        s = {
            'min': float(ch.min()),
            'max': float(ch.max()),
            'mean': float(ch.mean()),
            'std': float(ch.std()),
            'neg_frac': neg,
            'zero_frac': zero,
            **robust_percentiles(ch, [50, 90, 95, 99, 99.9])
        }
        stats_c.append(s)

    # Luminance stats
    Y = luminance_from_rgb(rgb)
    Yf = Y.reshape(-1)
    lum_stats = {
        'min': float(Yf.min()),
        'max': float(Yf.max()),
        'mean': float(Yf.mean()),
        'std': float(Yf.std()),
        'neg_frac': float((Yf < 0).mean()),
        'zero_frac': float((Yf == 0).mean()),
        **robust_percentiles(Yf, [50, 90, 95, 99, 99.9])
    }

    # Subsample pixels for global aggregation
    if sample_per_file > 0 and total > 0:
        idx = rng.choice(total, size=min(sample_per_file, total), replace=False)
        subs_rgb = flat[:, idx]  # shape [C, N]
        subs_Y = Yf[idx]
    else:
        subs_rgb = np.empty((C,0), dtype=np.float32)
        subs_Y = np.empty((0,), dtype=np.float32)

    return {
        'channels': stats_c,
        'luminance': lum_stats,
        'subs_rgb': subs_rgb,
        'subs_Y': subs_Y,
    }


def aggregate_stats(samples_rgb: List[np.ndarray], samples_Y: List[np.ndarray]) -> Dict:
    """Aggregate subsamples to compute global percentiles and suggestions."""
    if samples_rgb:
        rgb_concat = np.concatenate(samples_rgb, axis=1)  # [C, N]
    else:
        rgb_concat = np.empty((3,0), dtype=np.float32)
    if samples_Y:
        Y_concat = np.concatenate(samples_Y, axis=0)
    else:
        Y_concat = np.empty((0,), dtype=np.float32)

    agg = {'channels': [], 'luminance': {}}

    for c in range(min(3, rgb_concat.shape[0])):
        ch = rgb_concat[c]
        if ch.size == 0:
            agg['channels'].append({})
            continue
        agg['channels'].append({
            **robust_percentiles(ch, [50, 90, 95, 99, 99.9]),
            'mean': float(ch.mean()),
            'std': float(ch.std()),
        })

    if Y_concat.size > 0:
        agg['luminance'] = {
            **robust_percentiles(Y_concat, [50, 90, 95, 99, 99.9]),
            'mean': float(Y_concat.mean()),
            'std': float(Y_concat.std()),
        }

    # Suggested scale factors so that p95 ~ 1.0 after division
    suggestions = {}
    if agg['luminance']:
        suggestions['scale_factor_luminance_p95'] = float(agg['luminance']['p95'])
        suggestions['scale_factor_luminance_p99'] = float(agg['luminance']['p99'])
    if agg['channels'] and all('p95' in ch for ch in agg['channels'] if ch):
        ch_p95 = [ch['p95'] for ch in agg['channels'] if ch]
        suggestions['scale_factor_max_channel_p95'] = float(max(ch_p95))

    agg['suggestions'] = suggestions
    agg['sample_counts'] = {
        'rgb_total_samples': int(rgb_concat.shape[1]) if rgb_concat.ndim == 2 else 0,
        'luminance_total_samples': int(Y_concat.size),
    }
    return agg


def save_histograms(output_dir: Path, name: str, samples_rgb: List[np.ndarray], samples_Y: List[np.ndarray]):
    if not MPL_AVAILABLE:
        print("[INFO] Matplotlib not available, skip histograms")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    if samples_rgb:
        rgb = np.concatenate(samples_rgb, axis=1)
        for c, label in enumerate(['R','G','B'][:rgb.shape[0]]):
            plt.figure(figsize=(6,4))
            data = np.clip(rgb[c], 0, np.percentile(rgb[c], 99.9))
            plt.hist(data, bins=200, log=True)
            plt.title(f"{name} {label} distribution (clipped 99.9p)")
            plt.xlabel("Value"); plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(output_dir / f"{name}_{label}_hist.png")
            plt.close()

    if samples_Y:
        Y = np.concatenate(samples_Y, axis=0)
        plt.figure(figsize=(6,4))
        data = np.clip(Y, 0, np.percentile(Y, 99.9))
        plt.hist(data, bins=200, log=True)
        plt.title(f"{name} luminance distribution (clipped 99.9p)")
        plt.xlabel("Value"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_luminance_hist.png")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data', help='Root folder containing processed_bistro')
    ap.add_argument('--max-files', type=int, default=500, help='Max files per subset to process')
    ap.add_argument('--sample-per-file', type=int, default=200_000, help='Random pixel samples per file for aggregation')
    ap.add_argument('--save-hist', action='store_true', help='Save histogram PNGs')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    data_root = Path(args.data_root) / 'processed_bistro'
    ref_dir = data_root / 'ref'
    warp_dir = data_root / 'warp_hole'

    if not ref_dir.exists():
        print(f"[ERROR] Missing dir: {ref_dir}")
        sys.exit(1)
    if not warp_dir.exists():
        print(f"[ERROR] Missing dir: {warp_dir}")
        sys.exit(1)

    rng = np.random.RandomState(args.seed)

    subsets = {
        'ref': sorted([p for p in ref_dir.glob('*.exr')]),
        'warp_hole': sorted([p for p in warp_dir.glob('*.exr')]),
    }

    results = {
        'env': {
            'openexr_available': OPENEXR_AVAILABLE,
            'opencv_available': CV2_AVAILABLE,
            'matplotlib_available': MPL_AVAILABLE,
        },
        'settings': {
            'max_files': args.max_files,
            'sample_per_file': args.sample_per_file,
            'seed': args.seed,
        },
        'subsets': {}
    }

    out_dir = Path('logs/hdr_stats')
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, files in subsets.items():
        print(f"[INFO] Subset {name}: {len(files)} files")
        files = files[: args.max_files]
        per_file_stats = []
        samples_rgb: List[np.ndarray] = []
        samples_Y: List[np.ndarray] = []

        for i, fpath in enumerate(files):
            try:
                rgb = load_exr(fpath)
                s = file_stats(rgb, rng, args.sample_per_file)
                per_file_stats.append({
                    'file': fpath.name,
                    'channels': s['channels'],
                    'luminance': s['luminance'],
                })
                if s['subs_rgb'].size:
                    samples_rgb.append(s['subs_rgb'])
                if s['subs_Y'].size:
                    samples_Y.append(s['subs_Y'])
                if (i+1) % 20 == 0:
                    print(f"  processed {i+1}/{len(files)}")
            except Exception as e:
                print(f"[WARN] Failed {fpath.name}: {e}")

        agg = aggregate_stats(samples_rgb, samples_Y)
        if args.save_hist:
            save_histograms(out_dir, name, samples_rgb, samples_Y)

        results['subsets'][name] = {
            'file_count': len(files),
            'per_file_stats_sampled': per_file_stats[: min(10, len(per_file_stats))],  # keep a small sample
            'aggregate': agg,
        }

    # Suggest global recommendation from ref subset if available, else warp_hole
    rec_src = 'ref' if 'ref' in results['subsets'] else 'warp_hole'
    rec = results['subsets'][rec_src]['aggregate'].get('suggestions', {})
    results['recommendation'] = {
        'scale_factor_prefer': rec.get('scale_factor_luminance_p95', None),
        'alternatives': rec,
        'rationale': 'Choose scale so that p95 luminance maps to ~1.0 after division; adjust per visualization.'
    }

    out_json = out_dir / 'processed_bistro_stats.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved stats: {out_json}")

    # Also emit a short markdown summary skeleton
    md = out_dir / 'RESULTS_TEMPLATE.md'
    md.write_text(RESULTS_TEMPLATE, encoding='utf-8')
    print(f"[OK] Wrote template: {md}")


RESULTS_TEMPLATE = """# processed_bistro HDR 像素分布统计报告（模板）

## 基本信息
- 数据根目录: ./data/processed_bistro
- 统计设置: max_files=<填写>, sample_per_file=<填写>, seed=<填写>
- 读取后端: OpenEXR=<填写>, OpenCV=<填写>

## 全局建议
- 推荐 SCALE_FACTOR（优先按亮度p95）: <填写>
- 备选: { 亮度p99: <填写>, 通道最大p95: <填写> }
- 依据: 使 p95 ≈ 1.0，有利于大多数像素落在[0,1]附近，同时保留高光>1

## 子集统计

### ref
- 文件数: <填写>
- 亮度统计（聚合样本）:
  - p50: <填写> | p90: <填写> | p95: <填写> | p99: <填写> | p99.9: <填写>
  - mean: <填写> | std: <填写>
- 通道统计（聚合样本）:
  - R: p95=<填写> p99=<填写>
  - G: p95=<填写> p99=<填写>
  - B: p95=<填写> p99=<填写>
- 负值占比（示例文件）: <填写>
- 直方图: logs/hdr_stats/ref_*.png（若保存）

### warp_hole
- 文件数: <填写>
- 亮度统计（聚合样本）:
  - p50: <填写> | p90: <填写> | p95: <填写> | p99: <填写> | p99.9: <填写>
  - mean: <填写> | std: <填写>
- 通道统计（聚合样本）:
  - R: p95=<填写> p99=<填写>
  - G: p95=<填写> p99=<填写>
  - B: p95=<填写> p99=<填写>
- 负值占比（示例文件）: <填写>
- 直方图: logs/hdr_stats/warp_hole_*.png（若保存）

## 结论与选型
- 初始 SCALE_FACTOR: <填写>
- 说明: 选择亮度p95作为缩放因子，兼顾稳定性与细节；若训练初期梯度较大，可尝试 p90；若高光保留仍不足，可增大 SCALE_FACTOR 或在显示端调高曝光

## 附：采样文件概览（可选）
- 列举若干代表文件的 min/max/neg_frac/高分位，以便核对异常值

"""

if __name__ == '__main__':
    main()
