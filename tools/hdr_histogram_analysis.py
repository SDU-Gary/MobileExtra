#!/usr/bin/env python3
"""
HDR Histogram Analysis Tool

功能：
1. 统计原始 HDR EXR 数据与预处理后输入网络的 HDR 数据直方图
2. 使用对数刻度直观展示宽动态范围分布
3. 输出均值、标准差、最小值、最大值等统计指标
4. 支持批量采样，生成对比图与 JSON 统计结果

用法示例：
    python tools/hdr_histogram_analysis.py \
        --config configs/colleague_training_config.yaml \
        --data-root ./data \
        --split train \
        --max-samples 32 \
        --output-dir ./logs/hdr_hist \
        --num-bins 128
"""

import argparse
import json
import os
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# 将仓库根目录加入 sys.path，确保脚本可从命令行独立运行
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matplotlib import font_manager  # noqa: E402
from train.patch_training_framework import ColleaguePatchDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HDR 数据直方图分析工具")
    parser.add_argument("--config", type=Path, default=Path("configs/colleague_training_config.yaml"),
                        help="训练配置文件路径（用于读取 HDR 预处理参数）")
    parser.add_argument("--data-root", type=Path, default=Path("./data"),
                        help="数据根目录（包含 processed_bistro）")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="数据集划分")
    parser.add_argument("--max-samples", type=int, default=32,
                        help="参与统计的最大样本数")
    parser.add_argument("--num-bins", type=int, default=128,
                        help="直方图分箱数（log 间隔）")
    parser.add_argument("--output-dir", type=Path, default=Path("./logs/hdr_hist"),
                        help="输出目录（图像与统计 JSON）")
    parser.add_argument("--channel", type=str, default="warp",
                        choices=["warp", "ref"],
                        help="统计的 RGB 通道来源：warp=warped_hole，ref=参考图像")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="对数刻度时避免 log(0) 的最小值")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_patch_config(patch_cfg: Dict) -> SimpleNamespace:
    defaults = {
        "enable_patch_mode": True,
        "patch_size": 256,
        "patch_mode_probability": 1.0,
        "min_patches_per_image": 16,
        "max_patches_per_image": 16,
        "patch_overlap_threshold": 0.3,
        "enable_patch_cache": False,
        "cache_size": 1000,
        "cache_hit_threshold": 0.8,
        "patch_augmentation": False,
        "augmentation_probability": 0.0,
        "min_hole_area": 0,
        "max_hole_area": 1000000,
        "boundary_margin": 0,
        "use_optimized_patches": False,
        "use_simple_grid_patches": True,
        "simple_grid_rows": 4,
        "simple_grid_cols": 4,
        "simple_expected_height": 1080,
        "simple_expected_width": 1920,
        "use_overlapping_crops": False,
        "crop_size": 256,
        "crop_stride": 128,
        "keep_top_frac": 0.5,
        "min_hole_frac": 0.005,
        "sample_hole_weighted": True,
        "enforce_min_hole_frac": True,
        "patch_height": 270,
        "patch_width": 480,
    }
    merged = {**defaults, **(patch_cfg or {})}
    return SimpleNamespace(**merged)


def load_global_stats(stats_path: str) -> Tuple[float, float]:
    if not stats_path:
        return 0.0, 1.0
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mu = float(stats.get("mu", 0.0))
        sigma = float(stats.get("sigma", 1.0))
        if sigma <= 1e-8:
            sigma = 1.0
        return mu, sigma
    except Exception:
        return 0.0, 1.0


def compute_network_features(patch_input: torch.Tensor,
                             norm_cfg: Dict,
                             global_stats: Tuple[float, float],
                             epsilon: float,
                             log_stats: Tuple[float, float],
                             percentile_pair: Tuple[float, float],
                             use_dither: bool,
                             dither_scale: float) -> Dict[str, torch.Tensor]:
    warped_rgb = patch_input[:3].clone()
    norm_type = str(norm_cfg.get("type", "none")).lower()

    if norm_type == "log":
        eps = float(norm_cfg.get("log_epsilon", epsilon))
        warped_pos = torch.clamp(warped_rgb, min=0.0)

        log_img_clean = torch.log(warped_pos + eps)

        if use_dither:
            noise = (torch.rand_like(warped_pos) - 0.5) * dither_scale
            warped_pos_dither = torch.clamp(warped_pos + noise, min=0.0)
            log_img_dither = torch.log(warped_pos_dither + eps)
        else:
            log_img_dither = log_img_clean

        log_img = log_img_clean
        min_log_clean = float(log_img_clean.min())
        max_log_clean = float(log_img_clean.max())
        denom_clean = max(max_log_clean - min_log_clean, 1e-6)
        baseline_clean = (log_img_clean - min_log_clean) / denom_clean

        min_log_dither = float(log_img_dither.min())
        max_log_dither = float(log_img_dither.max())
        denom_dither = max(max_log_dither - min_log_dither, 1e-6)
        baseline_dither = (log_img_dither - min_log_dither) / denom_dither

        mu_g, sigma_g = log_stats
        sigma_g = max(sigma_g, 1e-6)
        global_z = (log_img - mu_g) / sigma_g

        p_low = max(0.0, min(1.0, percentile_pair[0]))
        p_high = max(0.0, min(1.0, percentile_pair[1]))
        if p_high <= p_low:
            p_low, p_high = 0.005, 0.995
        values = log_img.flatten()
        try:
            q_low = torch.quantile(values, p_low)
            q_high = torch.quantile(values, p_high)
        except Exception:
            q_low, q_high = values.min(), values.max()
        denom_p = max(float(q_high - q_low), 1e-6)
        percentile_scaled = torch.clamp((log_img - q_low) / denom_p, 0.0, 1.0)

        return {
            "baseline": baseline_clean,
            "baseline_dither": baseline_dither,
            "global_z": global_z,
            "percentile_scaled": percentile_scaled,
            "log_img": log_img_clean,
            "warped_rgb": warped_rgb,
        }

    # Non-log path fallback to simple representations
    mu, sigma = global_stats
    sigma = max(sigma, 1e-6)
    values = warped_rgb.flatten()
    try:
        q = torch.quantile(values, 0.99)
    except Exception:
        q = values.abs().max()
    scale = max(float(q), 1e-6)

    return {
        "baseline": warped_rgb / scale,
        "baseline_dither": warped_rgb / scale,
        "global_z": (warped_rgb - mu) / sigma,
        "percentile_scaled": warped_rgb / scale,
        "log_img": torch.log(torch.clamp(warped_rgb, min=0.0) + epsilon),
        "warped_rgb": warped_rgb,
    }


def get_patch_coordinates(idx: int,
                          metadata: Dict,
                          patch_config: SimpleNamespace,
                          patch_shape: Tuple[int, int]) -> Tuple[int, int]:
    if metadata and "crop_rect" in metadata:
        x, y, _, _ = metadata["crop_rect"]
        return int(x), int(y)

    patches_per_image = int(patch_config.max_patches_per_image)
    simple_cols = int(patch_config.simple_grid_cols)
    patch_h, patch_w = patch_shape

    patch_idx = idx % patches_per_image
    row = patch_idx // simple_cols
    col = patch_idx % simple_cols
    y = row * patch_h
    x = col * patch_w
    return int(x), int(y)


def collect_patch_values(patch_dataset: ColleaguePatchDataset,
                         patch_config: SimpleNamespace,
                         norm_cfg: Dict,
                         global_stats: Tuple[float, float],
                         log_stats: Tuple[float, float],
                         percentile_pair: Tuple[float, float],
                         channel: str,
                         max_samples: int,
                         epsilon: float,
                         tone_cfg: Dict) -> Dict[str, np.ndarray]:
    raw_values: List[np.ndarray] = []
    proc_baseline: List[np.ndarray] = []
    proc_baseline_dither: List[np.ndarray] = []
    proc_global: List[np.ndarray] = []
    proc_percentile: List[np.ndarray] = []
    log_values: List[np.ndarray] = []
    tone_values: List[np.ndarray] = []

    total = min(max_samples, len(patch_dataset))
    for idx in range(total):
        sample = patch_dataset[idx]
        patch_input = sample["patch_input"]
        metadata = sample.get("metadata", {})
        patch_h, patch_w = patch_input.shape[1], patch_input.shape[2]

        # Determine source frame index
        if metadata and "source_index" in metadata:
            src_idx = int(metadata["source_index"])
        else:
            patches_per_image = int(patch_config.max_patches_per_image)
            src_idx = idx // patches_per_image

        frame_id = patch_dataset.base_dataset.frame_list[src_idx]
        file_paths = patch_dataset.base_dataset.file_mapping[frame_id]
        src_key = "warp_hole" if channel == "warp" else "ref"
        raw_full = patch_dataset.base_dataset._load_exr(file_paths[src_key])[:3]  # type: ignore[attr-defined]

        x, y = get_patch_coordinates(idx, metadata, patch_config, (patch_h, patch_w))
        raw_patch = raw_full[:, y:y + patch_h, x:x + patch_w]

        raw_values.append(raw_patch.flatten().detach().cpu().numpy())
        features = compute_network_features(
            patch_input.clone(),
            norm_cfg,
            global_stats,
            epsilon,
            log_stats,
            percentile_pair,
            use_dither=bool(norm_cfg.get("log_dither_enable", False)),
            dither_scale=float(norm_cfg.get("log_dither_scale", 1.0e-6)),
        )
        proc_baseline.append(features["baseline"].flatten().detach().cpu().numpy())
        proc_baseline_dither.append(features["baseline_dither"].flatten().detach().cpu().numpy())
        proc_global.append(features["global_z"].flatten().detach().cpu().numpy())
        proc_percentile.append(features["percentile_scaled"].flatten().detach().cpu().numpy())
        log_values.append(features["log_img"].flatten().detach().cpu().numpy())

        tone_mapped = tone_map_tensor(features["warped_rgb"], tone_cfg)
        tone_values.append(tone_mapped.flatten().detach().cpu().numpy())

    return {
        "raw": np.concatenate(raw_values),
        "baseline": np.concatenate(proc_baseline),
        "baseline_dither": np.concatenate(proc_baseline_dither),
        "global_z": np.concatenate(proc_global),
        "percentile": np.concatenate(proc_percentile),
        "log_img": np.concatenate(log_values),
        "tone_mapped": np.concatenate(tone_values),
    }


def tone_map_tensor(rgb_tensor: torch.Tensor, tone_cfg: Dict) -> torch.Tensor:
    method = str(tone_cfg.get("tone_mapping_for_display", "reinhard")).lower()
    gamma = float(tone_cfg.get("gamma", 1.0))
    exposure = float(tone_cfg.get("exposure", 1.0))
    mu = float(tone_cfg.get("mulaw_mu", 500.0))

    rgb = torch.clamp(rgb_tensor, min=0.0)
    if method == "mulaw":
        mu_tensor = rgb.new_tensor(mu)
        mapped = torch.log1p(mu_tensor * exposure * rgb) / torch.log1p(mu_tensor)
    elif method == "reinhard":
        mapped = (exposure * rgb) / (1.0 + exposure * rgb)
    else:
        mapped = torch.clamp(exposure * rgb, 0.0, None)

    if gamma != 1.0:
        mapped = torch.clamp(mapped, min=0.0) ** (1.0 / max(gamma, 1e-6))
    return torch.clamp(mapped, 0.0, 1.0)


def compute_histograms(data_map: Dict[str, np.ndarray],
                       num_bins: int,
                       epsilon: float) -> Dict[str, Dict[str, np.ndarray]]:
    combined = np.concatenate(list(data_map.values()))
    min_val = combined.min()
    max_val = combined.max()

    if max_val <= 0:
        raise ValueError("数据全部为非正值，无法构建对数直方图。")

    if min_val > 0:
        log_min = np.log10(max(min_val, epsilon))
        log_max = np.log10(max_val)
        bin_edges = np.logspace(log_min, log_max, num_bins + 1)
    else:
        max_abs = max(abs(min_val), abs(max_val))
        pos_edges = np.logspace(np.log10(epsilon), np.log10(max_abs), num_bins // 2 + 1)
        neg_edges = -pos_edges[::-1]
        bin_edges = np.concatenate((neg_edges, [0.0], pos_edges[1:]))

    histograms = {}
    for key, values in data_map.items():
        if values.min() >= 0:
            values_clipped = np.clip(values, epsilon, None)
        else:
            values_clipped = values
        hist, _ = np.histogram(values_clipped, bins=bin_edges, density=True)
        histograms[key] = {
            "density": hist,
            "bin_edges": bin_edges,
            "min_value": float(values_clipped.min()),
            "max_value": float(values_clipped.max()),
        }
    histograms["combined"] = {
        "bin_edges": bin_edges,
        "min_value": min_val,
        "max_value": max_val,
    }
    return histograms


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p1": float(np.percentile(values, 1)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
    }


LABEL_MAP = {
    "raw": ("原始 HDR", "Raw HDR"),
    "baseline": ("现有归一化（log+min/max）", "Baseline (log + min/max)"),
    "baseline_dither": ("log+min/max（带抖动）", "Baseline (log + min/max + dither)"),
    "global_z": ("全局 Z-score (log)", "Global Z-score (log)"),
    "percentile": ("分位缩放 (log)", "Percentile scaling (log)"),
    "log_img": ("log 域原值", "Raw log-domain"),
    "tone_mapped": ("Tone-mapped LDR", "Tone-mapped LDR"),
}

def plot_histogram(histograms: Dict[str, Dict[str, np.ndarray]],
                   output_path: Path,
                   title_cn: str,
                   title_en: str,
                   use_chinese: bool) -> None:
    combined = histograms["combined"]
    bin_edges = combined["bin_edges"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    plt.figure(figsize=(11, 7))
    xlabel = "像素强度 / 特征值（对数刻度）" if use_chinese else "Intensity / Feature (log scale)"
    ylabel = "概率密度（对数刻度）" if use_chinese else "Probability Density (log scale)"
    title = title_cn if use_chinese else title_en

    colors = {
        "raw": "#1f77b4",
        "baseline": "#ff7f0e",
        "baseline_dither": "#ffbb78",
        "global_z": "#2ca02c",
        "percentile": "#d62728",
        "log_img": "#9467bd",
        "tone_mapped": "#8c564b",
    }

    for key, hist in histograms.items():
        if key == "combined":
            continue
        label_cn, label_en = LABEL_MAP.get(key, (key, key))
        label = label_cn if use_chinese else label_en
        density = hist["density"]
        plt.plot(bin_centers, density, label=label, lw=2, alpha=0.9, color=colors.get(key))

    if combined["min_value"] > 0:
        plt.xscale("log")
    else:
        plt.xscale("symlog", linthresh=1e-6)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_single_histogram(key: str,
                          histograms: Dict[str, Dict[str, np.ndarray]],
                          output_path: Path,
                          use_chinese: bool) -> None:
    combined = histograms["combined"]
    if key == "combined" or key not in histograms:
        return

    hist = histograms[key]
    bin_edges = combined["bin_edges"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    label_cn, label_en = LABEL_MAP.get(key, (key, key))
    xlabel = "像素强度 / 特征值（对数刻度）" if use_chinese else "Intensity / Feature (log scale)"
    ylabel = "概率密度（对数刻度）" if use_chinese else "Probability Density (log scale)"
    title = label_cn if use_chinese else label_en

    plt.figure(figsize=(9, 6))
    plt.plot(bin_centers, hist["density"], lw=2, color="#1f77b4")
    if combined["min_value"] > 0:
        plt.xscale("log")
    else:
        plt.xscale("symlog", linthresh=1e-6)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def configure_fonts() -> bool:
    """配置 matplotlib 字体以支持中文；若无中文字体则回退到默认并提示。

    Returns:
        bool: 是否成功设置中文字体
    """
    candidate_fonts = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Source Han Sans CN",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    # 回退：使用默认字体并提示
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("⚠️ 未找到中文字体，已回退至 DejaVu Sans，将使用英文标签展示。")
    return False


def load_log_stats(norm_cfg: Dict) -> Tuple[float, float]:
    stats_path = norm_cfg.get("log_stats_path")
    if not stats_path:
        return 0.0, 1.0
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mu = float(stats.get("log_mu", 0.0))
        sigma = float(stats.get("log_sigma", 1.0))
        if sigma <= 1e-8:
            sigma = 1.0
        return mu, sigma
    except Exception:
        return 0.0, 1.0


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    use_chinese = configure_fonts()

    cfg = load_config(args.config)
    patch_config = build_patch_config(cfg.get("patch", {}))

    hdr_cfg = cfg.get("hdr_processing", {})
    adapter_kwargs = dict(
        enable_linear_preprocessing=hdr_cfg.get("enable_linear_preprocessing", True),
        enable_srgb_linear=hdr_cfg.get("enable_srgb_linear", False),
        scale_factor=float(hdr_cfg.get("scale_factor", 1.0)),
        tone_mapping_for_display=hdr_cfg.get("tone_mapping_for_display", "reinhard"),
        gamma=float(hdr_cfg.get("gamma", 1.0)),
        exposure=float(hdr_cfg.get("exposure", 1.0)),
        adaptive_exposure=hdr_cfg.get("adaptive_exposure", {"enable": False}),
        mulaw_mu=float(hdr_cfg.get("mulaw_mu", 500.0)),
    )

    patch_dataset = ColleaguePatchDataset(
        data_root=str(args.data_root),
        split=args.split,
        patch_config=patch_config,
        adapter_kwargs=adapter_kwargs
    )

    sample_count = min(args.max_samples, len(patch_dataset))
    if sample_count == 0:
        raise RuntimeError("数据集为空，无法统计。")

    norm_cfg = cfg.get("normalization", {})
    stats_path = norm_cfg.get("stats_path") or cfg.get("training", {}).get("global_standardization", {}).get("stats_path", "")
    global_stats = load_global_stats(stats_path if stats_path else "")
    log_stats = load_log_stats(norm_cfg)
    percentile_pair = (
        float(norm_cfg.get("percentile_low", 0.005)),
        float(norm_cfg.get("percentile_high", 0.995)),
    )
    tone_cfg = cfg.get("hdr_processing", {})

    print(f"统计样本数: {sample_count} / {len(patch_dataset)} (split={args.split})")

    data_map = collect_patch_values(
        patch_dataset=patch_dataset,
        patch_config=patch_config,
        norm_cfg=norm_cfg,
        global_stats=global_stats,
        log_stats=log_stats,
        percentile_pair=percentile_pair,
        channel=args.channel,
        max_samples=sample_count,
        epsilon=args.epsilon,
        tone_cfg=tone_cfg,
    )

    stats = {k: summarize(v) for k, v in data_map.items()}
    histograms = compute_histograms(data_map, args.num_bins, args.epsilon)

    plot_path = args.output_dir / f"hdr_hist_{args.channel}_{args.split}.png"
    plot_title_cn = f"HDR 分布多视角对比 - {args.channel.upper()} ({args.split})"
    plot_title_en = f"HDR Distribution Comparison - {args.channel.upper()} ({args.split})"
    plot_histogram(histograms, plot_path, plot_title_cn, plot_title_en, use_chinese)

    for key in histograms.keys():
        if key == "combined":
            continue
        single_path = args.output_dir / f"hdr_hist_{args.channel}_{args.split}_{key}.png"
        plot_single_histogram(key, histograms, single_path, use_chinese)

    stats_path = args.output_dir / f"hdr_hist_{args.channel}_{args.split}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": str(args.config),
            "data_root": str(args.data_root),
            "split": args.split,
            "channel": args.channel,
            "max_samples": sample_count,
            "num_bins": args.num_bins,
            "statistics": stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"直方图已保存: {plot_path}")
    print(f"统计信息已保存: {stats_path}")
    for key, stat in stats.items():
        print(f"{key} 统计: {stat}")


if __name__ == "__main__":
    main()
