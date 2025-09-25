#!/usr/bin/env python3
"""
Dry Start Checker

Purpose: Validate config-driven pipeline wiring without running training.

Checks:
1) Load YAML config and print key sections (hdr_processing, loss.weights/robust, training.gradient_clip_val)
2) Tone-map utility smoke test (src/npu/utils/hdr_vis.tone_map)
3) Initialize Patch TensorBoard Visualizer with hdr_processing + visualization.grid
4) (Optional) Initialize PatchAwareLoss (robust + weights + tone-mapped perceptual)
5) (Optional) Forward dummy tensors through PatchAwareLoss

Usage examples:
  python tools/dry_check.py --config configs/colleague_training_config.yaml --logdir ./logs/dry_check
  python tools/dry_check.py --config configs/colleague_training_config.yaml --init-loss --forward-loss

Notes:
- If torchvision tries to download VGG16 weights and your environment is offline, omit --init-loss/--forward-loss
  or ensure weights are cached (set TORCH_HOME).
"""

import argparse
import os
import sys
import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/colleague_training_config.yaml')
    ap.add_argument('--logdir', type=str, default='./logs/dry_check')
    ap.add_argument('--init-loss', action='store_true', help='Initialize PatchAwareLoss (may require VGG weights)')
    ap.add_argument('--forward-loss', action='store_true', help='Run a dummy forward through PatchAwareLoss (implies --init-loss)')
    args = ap.parse_args()

    # Resolve project root and ensure import paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Load YAML
    cfg_path = args.config
    if not os.path.exists(cfg_path):
        print(f'[ERROR] Config not found: {cfg_path}')
        return 2
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print('=== Config Snapshot ===')
    print('hdr_processing:', cfg.get('hdr_processing'))
    print('loss.weights:', cfg.get('loss', {}).get('weights'))
    print('loss.robust:', cfg.get('loss', {}).get('robust'))
    print('training.gradient_clip_val:', cfg.get('training', {}).get('gradient_clip_val'))
    print('visualization.grid:', cfg.get('visualization', {}).get('grid'))

    # 1) Tone-map utility smoke test
    try:
        import torch
        from src.npu.utils.hdr_vis import tone_map
        hp = cfg.get('hdr_processing', {}) or {}
        x = torch.tensor([[[0.0, 0.5], [2.0, 10.0]]], dtype=torch.float32).repeat(3, 1, 1)
        y = tone_map(
            x,
            method=str(hp.get('tone_mapping_for_display', 'reinhard')).lower(),
            gamma=float(hp.get('gamma', 2.2)),
            exposure=float(hp.get('exposure', 1.0)),
            adaptive_exposure=hp.get('adaptive_exposure', {'enable': False}),
        )
        print(f'[OK] tone_map: out_range=[{float(y.min()):.4f}, {float(y.max()):.4f}]')
    except Exception as e:
        print(f'[WARN] tone_map test failed: {e}')

    # 2) Initialize Patch TensorBoard Visualizer
    try:
        from train.patch_tensorboard_logger import create_patch_visualizer
        viz_cfg = {
            'visualization_frequency': cfg.get('visualization', {}).get('visualization_frequency', 100),
            'save_frequency': cfg.get('visualization', {}).get('save_frequency', 500),
            'enable_visualization': True,
            'hdr_processing': cfg.get('hdr_processing', {}),
            'grid': cfg.get('visualization', {}).get('grid', {}),
        }
        viz = create_patch_visualizer(args.logdir, viz_cfg)
        # Read back logger params
        logger = viz.patch_logger
        print('[OK] PatchTensorBoardLogger:')
        print('  tone_mapping =', logger.tone_mapping)
        print('  gamma       =', logger.gamma)
        print('  exposure    =', getattr(logger, 'exposure', None))
        print('  adaptive_exposure =', getattr(logger, 'adaptive_exposure', None))
        print('  grid panels =', logger.grid_panel_order, 'max_patches =', logger.grid_max_patches)
        viz.close()
    except Exception as e:
        print(f'[WARN] Visualizer init failed: {e}')

    # 3) (Optional) Initialize PatchAwareLoss
    if args.init_loss or args.forward_loss:
        try:
            import torch
            from train.patch_training_framework import PatchAwareLoss
            robust_cfg = cfg.get('loss', {}).get('robust', {})
            weights_cfg = cfg.get('loss', {}).get('weights', {})
            hdr_vis_cfg = cfg.get('hdr_processing', {})
            loss_fn = PatchAwareLoss(
                robust_config=robust_cfg,
                weights_config=weights_cfg,
                hdr_vis_config=hdr_vis_cfg,
            )
            print('[OK] PatchAwareLoss initialized')
            if args.forward_loss:
                pred = torch.rand(2, 3, 128, 128) * 2.0
                tgt = torch.rand(2, 3, 128, 128) * 2.0
                total, ldict = loss_fn({'patch': pred}, {'patch': tgt}, mode='patch', epoch=0)
                print('[OK] PatchAwareLoss forward: total =', float(total.item()))
        except Exception as e:
            print(f'[WARN] PatchAwareLoss init/forward failed (check torchvision/VGG weights): {e}')

    print('\n[DONE] Dry check completed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
