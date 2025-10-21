#!/usr/bin/env python3
"""
Compute Global HDR Mean/Std (mu, sigma) after linear preprocessing.

Usage:
  python tools/compute_hdr_global_stats.py --config configs/colleague_training_config.yaml \
      --split train --include-holes false --out configs/hdr_global_stats.json

Notes:
  - We compute statistics on input RGB (warped_rgb) AFTER linear HDR preprocessing.
  - To avoid hole bias, default excludes hole pixels (holes_mask<0.5).
  - Outputs a JSON with {mu, sigma, count, include_holes, computed_at}.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch

# Local imports
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'train'))


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load YAML {path}: {e}")
        return {}


def get_cfg(d: Dict[str, Any], key: str, default: Any) -> Any:
    cur = d
    for k in key.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def compute_stats(args) -> Dict[str, Any]:
    from train.colleague_dataset_adapter import ColleagueDatasetAdapter

    cfg = load_yaml(Path(args.config)) if args.config else {}
    data_root = get_cfg(cfg, 'data.data_root', './data')

    # Construct dataset with linear preprocessing enabled
    dataset = ColleagueDatasetAdapter(
        data_root=data_root,
        split=args.split,
        enable_linear_preprocessing=True,
        enable_srgb_linear=bool(get_cfg(cfg, 'hdr_processing.enable_srgb_linear', False)),
        scale_factor=float(get_cfg(cfg, 'hdr_processing.scale_factor', 0.70)),
        tone_mapping_for_display=str(get_cfg(cfg, 'hdr_processing.tone_mapping_for_display', 'reinhard')),
        gamma=float(get_cfg(cfg, 'hdr_processing.gamma', 2.2)),
        exposure=float(get_cfg(cfg, 'hdr_processing.exposure', 1.0)),
        adaptive_exposure=get_cfg(cfg, 'hdr_processing.adaptive_exposure', {'enable': False}),
    )

    include_holes = args.include_holes

    # Welford accumulators
    n = 0
    mean = 0.0
    M2 = 0.0

    for idx in range(len(dataset)):
        input_tensor, _, _ = dataset[idx]  # [7,H,W], with linear HDR preprocessing applied
        rgb = input_tensor[:3]  # [3,H,W]
        if include_holes:
            mask = torch.ones(1, rgb.shape[1], rgb.shape[2], dtype=torch.bool)
        else:
            holes_mask = input_tensor[3:4]  # [1,H,W]
            mask = holes_mask < 0.5

        # Select valid pixels and flatten
        valid = rgb[:, mask[0]].reshape(-1).double()
        if valid.numel() == 0:
            continue
        # Update Welford statistics
        batch_n = valid.numel()
        batch_mean = valid.mean().item()
        batch_var = valid.var(unbiased=False).item()

        # Combine two groups' stats: existing (n,mean,M2) with new batch
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        delta = batch_mean - mean
        tot_n = n + batch_n
        mean = mean + delta * batch_n / max(tot_n, 1)
        M2 = M2 + batch_var * batch_n + delta * delta * n * batch_n / max(tot_n, 1)
        n = tot_n

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(dataset)} samples...")

    if n == 0:
        raise RuntimeError("No valid pixels found to compute global stats.")

    var = M2 / n
    sigma = float(max(var, 0.0) ** 0.5)
    mu = float(mean)

    stats = {
        'mu': mu,
        'sigma': sigma,
        'count': int(n),
        'include_holes': bool(include_holes),
        'computed_at': datetime.now().isoformat(timespec='seconds'),
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/colleague_training_config.yaml')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--include-holes', type=lambda x: str(x).lower() in ['1','true','yes'], default=False)
    parser.add_argument('--out', type=str, default='configs/hdr_global_stats.json')
    args = parser.parse_args()

    stats = compute_stats(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved global HDR stats to {out_path} \nmu={stats['mu']:.6f}, sigma={stats['sigma']:.6f}, count={stats['count']}")


if __name__ == '__main__':
    main()

