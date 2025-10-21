#!/usr/bin/env python3
"""
Inspect overlap-crop hole coverage statistics.

This utility reuses the existing ColleaguePatchDataset overlap-crop logic
to report how crops are filtered and ranked by hole coverage.

Example:
    python tools/inspect_overlap_crops.py \
        --config configs/colleague_training_config.yaml \
        --split train \
        --max-frames 10
"""

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

# Ensure project modules are importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "train") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "train"))

from train.patch_training_framework import ColleaguePatchDataset
from train.patch_aware_dataset import PatchTrainingConfig as BasePatchTrainingConfig


def _build_patch_config(cfg_dict: Dict[str, Any]) -> BasePatchTrainingConfig:
    """Construct PatchTrainingConfig from a raw dict."""
    cfg = BasePatchTrainingConfig()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)
    return cfg


def _collect_overlap_stats(dataset: ColleaguePatchDataset, max_frames: int = None) -> Dict[str, Any]:
    """Replicate overlap-crop candidate gathering to collect statistics."""
    total_frames = len(dataset.base_dataset)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    all_candidates: List[float] = []
    frame_candidate_counts: List[int] = []

    crop_size = dataset._crop_size
    stride = dataset._crop_stride
    min_frac = dataset._min_hole_frac

    for img_idx in range(total_frames):
        try:
            input_tensor, _, _ = dataset.base_dataset[img_idx]
        except Exception as exc:  # pragma: no cover - dataset access errors
            print(f"[WARN] Skip frame {img_idx}: {exc}")
            continue

        holes = input_tensor[3] if torch.is_tensor(input_tensor) else torch.from_numpy(input_tensor[3]).float()
        H, W = holes.shape[-2], holes.shape[-1]
        P = min(crop_size, H, W)

        sx = stride
        xs = list(range(0, max(1, W - P + 1), sx))
        ys = list(range(0, max(1, H - P + 1), sx))
        if xs and xs[-1] != (W - P):
            xs.append(max(0, W - P))
        if ys and ys[-1] != (H - P):
            ys.append(max(0, H - P))
        if not xs:
            xs = [0]
        if not ys:
            ys = [0]

        frame_candidates: List[float] = []
        for y in ys:
            for x in xs:
                cell = holes[y:y + P, x:x + P]
                if cell.numel() == 0:
                    continue
                frac = float(cell.float().mean().item())
                if min_frac > 0.0 and frac < min_frac:
                    continue
                frame_candidates.append(frac)

        if not frame_candidates:
            # fallback to center crop (matches dataset logic)
            cx = max(0, (W - P) // 2)
            cy = max(0, (H - P) // 2)
            cell = holes[cy:cy + P, cx:cx + P]
            frac = float(cell.float().mean().item()) if cell.numel() > 0 else 0.0
            frame_candidates.append(frac)

        all_candidates.extend(frame_candidates)
        frame_candidate_counts.append(len(frame_candidates))

    selected = dataset._overlap_crops
    selected_fracs = [
        float(rec.get('hole_frac', 0.0))
        for rec in selected
        if rec.get("image_idx", 0) < total_frames
    ]

    def _summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        sorted_vals = sorted(values)
        return {
            "count": len(sorted_vals),
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": statistics.fmean(sorted_vals),
            "median": statistics.median(sorted_vals),
            "p95": sorted_vals[int(0.95 * (len(sorted_vals) - 1))],
        }

    stats = {
        "frames_scanned": total_frames,
        "candidates": _summary(all_candidates),
        "selected": _summary(selected_fracs),
        "selected_top5": sorted(selected_fracs, reverse=True)[:5],
        "selected_bottom5": sorted(selected_fracs)[:5],
        "per_frame_candidate_count": _summary(frame_candidate_counts),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect overlap-crop ranking statistics.")
    parser.add_argument("--config", type=Path, default=Path("configs/colleague_training_config.yaml"),
                        help="YAML config to load.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to analyse.")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data_root from config.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit on frames scanned for statistics.")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    patch_cfg_dict = cfg.get("patch", {})
    patch_cfg = _build_patch_config(patch_cfg_dict)
    if not getattr(patch_cfg, "use_overlapping_crops", False):
        raise RuntimeError("Config does not enable overlapping crops (patch.use_overlapping_crops=false).")

    data_cfg = cfg.get("data", {})
    data_root = str(args.data_root or data_cfg.get("data_root", "./data"))

    hdr_cfg = cfg.get("hdr_processing", {})
    adapter_kwargs = dict(
        enable_linear_preprocessing=hdr_cfg.get("enable_linear_preprocessing", True),
        enable_srgb_linear=hdr_cfg.get("enable_srgb_linear", True),
        scale_factor=hdr_cfg.get("scale_factor", 0.70),
        tone_mapping_for_display=hdr_cfg.get("tone_mapping_for_display", "reinhard"),
        gamma=hdr_cfg.get("gamma", 2.2),
        exposure=hdr_cfg.get("exposure", 1.0),
        adaptive_exposure=hdr_cfg.get("adaptive_exposure", {"enable": False}),
        mulaw_mu=hdr_cfg.get("mulaw_mu", 500.0),
    )

    dataset = ColleaguePatchDataset(
        data_root=data_root,
        split=args.split,
        patch_config=patch_cfg,
        adapter_kwargs=adapter_kwargs,
    )

    if not getattr(dataset, "_use_overlap", False):
        raise RuntimeError("Dataset did not enter overlapping-crop mode; ensure patch_config.use_overlapping_crops=True.")

    stats = _collect_overlap_stats(dataset, max_frames=args.max_frames)

    print("=== Overlap Crop Statistics ===")
    print(f"Frames scanned: {stats['frames_scanned']}")
    print(f"Total selected crops: {stats['selected'].get('count', 0)}")
    print("\nAll candidate hole fraction stats:")
    for k, v in stats["candidates"].items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nSelected crop hole fraction stats:")
    for k, v in stats["selected"].items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nTop 5 selected hole fractions:", [f"{x:.6f}" for x in stats["selected_top5"]])
    print("Bottom 5 selected hole fractions:", [f"{x:.6f}" for x in stats["selected_bottom5"]])

    print("\nPer-frame candidate count stats:")
    for k, v in stats["per_frame_candidate_count"].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
