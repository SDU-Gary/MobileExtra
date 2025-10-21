#!/usr/bin/env python3
"""
Dry Check Pipeline - 快速连通性自检脚本（不正式训练）

功能：
- 读取配置，打印关键训练/采样/显示域参数
- 构建数据集（简单网格或重叠crop），抓取一个小batch并打印shape/洞占比
- 构建Patch训练器与DataLoader
- 使用Lightning fast_dev_run=1 进行一次train/val快速跑通（前向+损失），验证全链路可用
"""

import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print("=== DRY CHECK: CONFIG SUMMARY ===")
    patch = cfg.get('patch', {})
    hdr = cfg.get('hdr_processing', {})
    train = cfg.get('training', {})
    norm = cfg.get('normalization', {})
    lossw = cfg.get('loss', {}).get('weights', {})
    print(f" data_root: {cfg.get('data', {}).get('data_root', './data')}")
    if patch.get('use_overlapping_crops', False):
        print(f" patch: overlapping crops ON (size={patch.get('crop_size',256)}, stride={patch.get('crop_stride',128)}, keep_top_frac={patch.get('keep_top_frac',0.5)})")
    else:
        print(f" patch: simple grid ({patch.get('simple_grid_rows',4)}x{patch.get('simple_grid_cols',4)})")
    print(f" training: batch_size={train.get('batch_size',4)}, num_workers={train.get('num_workers',0)}, accumulate_grad_batches={train.get('accumulate_grad_batches',1)}")
    print(f" hdr: tone_mapping={hdr.get('tone_mapping_for_display','reinhard')}, mu={hdr.get('mulaw_mu','NA')}, exposure={hdr.get('exposure',1.0)}, gamma={hdr.get('gamma',2.2)}")
    print(f" norm: type={norm.get('type','none')}, log_abs_max={norm.get('log_delta_abs_max',0)}, log_alpha={norm.get('log_delta_alpha',1.0)}, log_eps={norm.get('log_epsilon','NA')}")
    print(f" loss weights: domain={lossw.get('domain','ldr')}, ldr_for_pixel={lossw.get('ldr_for_pixel',True)}, ldr_for_edge={lossw.get('ldr_for_edge',True)}")

    # 动态导入训练框架
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from train.patch_training_framework import (
        create_patch_trainer,
        setup_data_loaders,
    )

    # 组装trainer_config与patch_config
    net_cfg = cfg.get('network', {})
    trainer_cfg = {
        'optimizer': {
            'learning_rate': train.get('learning_rate', 1e-4),
            'weight_decay': train.get('weight_decay', 1e-5)
        },
        'scheduler': {
            'T_max': train.get('max_epochs', 100),
            'eta_min': 1e-6
        },
        'log_dir': cfg.get('monitoring', {}).get('tensorboard_log_dir', './logs/patch_training'),
        'batch_size': train.get('batch_size', 4),
        'max_epochs': max(1, int(train.get('max_epochs', 1))),
        'gradient_clip_val': train.get('gradient_clip_val', 0.5),
        'accumulate_grad_batches': train.get('accumulate_grad_batches', 1),
    }
    patch_cfg = cfg.get('patch', {})

    # 创建训练器与数据集
    trainer = create_patch_trainer(
        model_config={'inpainting_network': {'input_channels': net_cfg.get('input_channels',7), 'output_channels': net_cfg.get('output_channels',3), 'base_channels': net_cfg.get('base_channels',24)}},
        training_config=trainer_cfg,
        patch_config=None,
        full_config=cfg,
    )
    ok = setup_data_loaders(trainer, cfg.get('data', {}), type('PC', (), patch_cfg), full_config=cfg)
    if not ok:
        print('[DRY] 数据加载器设置失败')
        return 1

    # 抓取一个batch检查
    try:
        batch = next(iter(trainer.train_loader))
        x = batch['patch_input']
        holes = x[:,3:4]
        hole_frac = float(holes.mean().item())
        print(f"[DRY] train batch shapes: input={tuple(x.shape)}, target={tuple(batch['patch_target_rgb'].shape)}, hole_frac_mean={hole_frac:.6f}")
    except Exception as e:
        print('[DRY] 取训练batch失败:', e)
        return 1

    # fast_dev_run=1 跑通一次（train/val各一步）
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    tb_logger = TensorBoardLogger(save_dir=trainer_cfg['log_dir'], name='dry_check')
    pl_trainer = pl.Trainer(
        max_epochs=1,
        logger=tb_logger,
        callbacks=[ModelCheckpoint(dirpath='./models/dry_check', save_top_k=0)],
        accelerator='auto', devices='auto', precision=32,
        fast_dev_run=True,
        accumulate_grad_batches=trainer_cfg.get('accumulate_grad_batches',1),
        gradient_clip_val=trainer_cfg.get('gradient_clip_val',0.5),
        log_every_n_steps=1,
    )
    print('[DRY] 开始 fast_dev_run=1 ...')
    pl_trainer.fit(model=trainer, train_dataloaders=trainer.train_loader, val_dataloaders=trainer.val_loader)
    print('[DRY] fast_dev_run 完成')
    return 0

if __name__ == '__main__':
    sys.exit(main())

