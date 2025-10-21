#!/usr/bin/env python3
"""
Validate linear-HDR preprocessing + tone-mapped display pipeline.
- Loads EXR from processed_bistro (ref/warp_hole)
- Optional sRGB->Linear conversion (Kornia)
- Non-negative clamp, divide by SCALE_FACTOR
- Tone mapping (Reinhard) + gamma to [0,1]
- Saves PNG grids for quick visual check

Usage:
  # 单子集预览（ref 或 warp_hole）
  python tools/hdr_stats/validate_linear_hdr_pipeline.py \
    --data-root ./data \
    --subset ref --max-files 6 \
    --scale-factor 0.70 --gamma 2.2 \
    --enable-srgb-linear

  # 生成 ref vs warp_hole 并排对比网格（按相同 frame_id 配对）
  python tools/hdr_stats/validate_linear_hdr_pipeline.py \
    --data-root ./data \
    --pair-grid --max-pairs 8 \
    --scale-factor 0.70 --gamma 2.2 \
    --enable-srgb-linear
"""
import argparse
from pathlib import Path
import sys
import numpy as np

# Backends
try:
    import OpenEXR  # type: ignore
    import Imath    # type: ignore
    import array    # type: ignore
    OPENEXR_AVAILABLE = True
except Exception:
    OPENEXR_AVAILABLE = False

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    import torch
except Exception as e:
    print(f"[ERROR] PyTorch not available: {e}")
    sys.exit(1)

# Kornia for sRGB<->Linear
try:
    import kornia
    import kornia.color as Kcolor
    KORNIA_AVAILABLE = True
except Exception:
    KORNIA_AVAILABLE = False


def load_exr(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    if OPENEXR_AVAILABLE:
        try:
            return _load_exr_openexr(path)
        except Exception as e:
            print(f"[WARN] OpenEXR failed: {path}, {e}")
    if CV2_AVAILABLE:
        try:
            return _load_exr_opencv(path)
        except Exception as e:
            print(f"[WARN] OpenCV failed: {path}, {e}")
    raise RuntimeError(f"No EXR backend available to read: {path}")


def _load_exr_openexr(path: Path) -> torch.Tensor:
    f = OpenEXR.InputFile(str(path))
    header = f.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = list(header['channels'].keys())
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    data = []
    order = ['R','G','B'] if all(c in channels for c in ['R','G','B']) else sorted(channels)
    for c in order[:3]:
        raw = f.channel(c, pixel_type)
        arr = np.array(array.array('f', raw), dtype=np.float32).reshape(height, width)
        data.append(arr)
    f.close()
    if not data:
        raise ValueError(f"No channels in {path}")
    chw = np.stack(data, axis=0)
    return torch.from_numpy(chw).float()


def _load_exr_opencv(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"cv2.imread failed: {path}")
    if img.ndim == 2:
        chw = img[np.newaxis, ...].astype(np.float32)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        chw = np.transpose(img, (2,0,1)).astype(np.float32)
    return torch.from_numpy(chw).float()


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    # rgb: [3,H,W] in [0, +inf) here, but sRGB->Linear expects [0,1] typical range.
    # If EXR is already linear, enabling this might have negligible effect.
    if not KORNIA_AVAILABLE:
        return rgb
    # Clamp to [0,1] for conversion safety
    x = torch.clamp(rgb, 0.0, 1.0).unsqueeze(0)
    y = Kcolor.rgb_to_linear_rgb(x)
    return y.squeeze(0)


def reinhard_tonemap(hdr: torch.Tensor, gamma: float) -> torch.Tensor:
    ldr = hdr / (1.0 + hdr)
    if gamma != 1.0:
        ldr = torch.pow(torch.clamp(ldr, 1e-8, 1.0), 1.0 / gamma)
    return torch.clamp(ldr, 0.0, 1.0)


def to_hwc_u8(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(x, 0.0, 1.0)
    x = (x * 255.0 + 0.5).to(torch.uint8)
    return x.permute(1,2,0).cpu().numpy()


def make_grid_three(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Concatenate three images horizontally, each [3,H,W]
    H = min(a.shape[1], b.shape[1], c.shape[1])
    W = min(a.shape[2], b.shape[2], c.shape[2])
    a = a[:, :H, :W]
    b = b[:, :H, :W]
    c = c[:, :H, :W]
    return torch.cat([a, b, c], dim=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--subset', type=str, default='ref', choices=['ref','warp_hole'])
    ap.add_argument('--max-files', type=int, default=6)
    ap.add_argument('--pair-grid', action='store_true', help='Create side-by-side grids for ref vs warp_hole')
    ap.add_argument('--max-pairs', type=int, default=8)
    ap.add_argument('--scale-factor', type=float, default=0.70)
    ap.add_argument('--gamma', type=float, default=2.2)
    ap.add_argument('--enable-srgb-linear', action='store_true')
    ap.add_argument('--save-dir', type=str, default='logs/hdr_stats/validate')
    args = ap.parse_args()

    root = Path(args.data_root) / 'processed_bistro'
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Kornia available: {KORNIA_AVAILABLE}, sRGB→Linear={'ON' if args.enable_srgb_linear else 'OFF'}")
    print(f"[INFO] Using Reinhard tone-mapping, gamma={args.gamma}")
    print(f"[INFO] SCALE_FACTOR={args.scale_factor}")

    if args.pair_grid:
        ref_dir = root / 'ref'
        warp_dir = root / 'warp_hole'
        if not ref_dir.exists() or not warp_dir.exists():
            print(f"[ERROR] Missing ref or warp_hole dir: {ref_dir} | {warp_dir}")
            sys.exit(1)

        def extract_id(stem: str) -> str:
            # ref-123 -> 123, warped_hole-123 -> 123
            if '-' in stem:
                return stem.split('-')[-1]
            return stem

        ref_map = {extract_id(p.stem): p for p in sorted(ref_dir.glob('*.exr'))}
        warp_map = {extract_id(p.stem): p for p in sorted(warp_dir.glob('*.exr'))}

        common_ids = sorted(set(ref_map.keys()) & set(warp_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        if not common_ids:
            print("[ERROR] No common frame ids found between ref and warp_hole")
            sys.exit(1)

        for fid in common_ids[: args.max_pairs]:
            rf = ref_map[fid]
            wf = warp_map[fid]
            try:
                r = torch.clamp(load_exr(rf), min=0.0)
                w = torch.clamp(load_exr(wf), min=0.0)
                r_lin = srgb_to_linear(r) if args.enable_srgb_linear else r
                w_lin = srgb_to_linear(w) if args.enable_srgb_linear else w
                r_scaled = r_lin / args.scale_factor
                w_scaled = w_lin / args.scale_factor
                r_disp = reinhard_tonemap(r_scaled, args.gamma)
                w_disp = reinhard_tonemap(w_scaled, args.gamma)

                grid = make_grid_three(w_disp, r_disp, r_disp)[:, :, : (w_disp.shape[2] + r_disp.shape[2])]  # use first two panels
                grid_u8 = to_hwc_u8(grid)
                cv2.imwrite(str(out_dir / f"pair-{fid}_warp_ref_tm.png"), cv2.cvtColor(grid_u8, cv2.COLOR_RGB2BGR))

                print(f"[OK] pair {fid}: warp lin[{w_lin.min():.4f},{w_lin.max():.4f}] -> disp[{w_disp.min():.4f},{w_disp.max():.4f}] | ref lin[{r_lin.min():.4f},{r_lin.max():.4f}] -> disp[{r_disp.min():.4f},{r_disp.max():.4f}]")
            except Exception as e:
                print(f"[WARN] Pair {fid} failed: {e}")

        print(f"[DONE] Saved pair grids to: {out_dir}")
        return

    # Single-subset mode
    data_dir = root / args.subset
    if not data_dir.exists():
        print(f"[ERROR] Missing dir: {data_dir}")
        sys.exit(1)
    files = sorted([p for p in data_dir.glob('*.exr')])[: args.max_files]
    for f in files:
        try:
            rgb = load_exr(f)  # [3,H,W]
            rgb = torch.clamp(rgb, min=0.0)
            lin = srgb_to_linear(rgb) if args.enable_srgb_linear else rgb
            scaled = lin / args.scale_factor
            disp = reinhard_tonemap(scaled, args.gamma)
            disp_u8 = to_hwc_u8(disp)
            cv2.imwrite(str(out_dir / f"{f.stem}_tm.png"), cv2.cvtColor(disp_u8, cv2.COLOR_RGB2BGR))
            print(f"[OK] {f.name}: range lin=[{lin.min():.4f},{lin.max():.4f}] scaled=[{scaled.min():.4f},{scaled.max():.4f}] disp=[{disp.min():.4f},{disp.max():.4f}]")
        except Exception as e:
            print(f"[WARN] Failed {f.name}: {e}")

    print(f"[DONE] Saved tone-mapped PNGs to: {out_dir}")


if __name__ == '__main__':
    main()
