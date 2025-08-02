#!/usr/bin/env python3
"""
@file train.py
@brief 移动端实时帧外插与空洞补全系统训练脚本

功能描述：
- 完整的端到端训练流程
- 支持多游戏数据集联合训练
- 知识蒸馏和量化感知训练
- 分布式训练和混合精度训练
- 实时监控和日志记录

训练阶段：
1. 数据加载和预处理
2. 模型初始化和配置
3. 训练循环执行
4. 验证和性能评估
5. 模型保存和部署准备

性能监控：
- 实时损失曲线
- SSIM/PSNR指标跟踪
- GPU内存和计算利用率
- 训练速度和ETA估算

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import argparse
import yaml
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import wandb

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'networks'))

# 导入训练组件
from training_framework import FrameInterpolationTrainer, create_trainer
from dataset import GameSceneDataModule, create_data_module
from datasets.cod_mobile_dataset import create_cod_mobile_dataset
from datasets.honor_of_kings_dataset import create_honor_of_kings_dataset
from mobile_inpainting_network import create_mobile_inpainting_network

# 设置随机种子
def set_random_seeds(seed: int = 42):
    """设置所有随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)
    print(f"Random seeds set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载训练配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        config: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        raise


def create_multi_game_dataset(data_configs: Dict[str, Any], 
                             batch_size: int = 32,
                             num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    创建多游戏联合数据集
    
    Args:
        data_configs: 多游戏数据配置
        batch_size: 批次大小
        num_workers: 工作线程数
    
    Returns:
        dataloaders: 数据加载器字典
    """
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # 遍历所有游戏配置
    for game_name, game_config in data_configs.items():
        print(f"\n=== Loading {game_name} Dataset ===")
        
        try:
            # 根据游戏类型创建对应数据集
            if game_name == 'cod_mobile':
                train_ds = create_cod_mobile_dataset(
                    data_root=game_config['data_root'],
                    split_file=game_config['train_split'],
                    patch_size=game_config['patch_size'],
                    mode='train'
                )
                val_ds = create_cod_mobile_dataset(
                    data_root=game_config['data_root'],
                    split_file=game_config['val_split'],
                    patch_size=game_config['patch_size'],
                    mode='val'
                )
                
            elif game_name == 'honor_of_kings':
                train_ds = create_honor_of_kings_dataset(
                    data_root=game_config['data_root'],
                    split_file=game_config['train_split'],
                    patch_size=game_config['patch_size'],
                    mode='train'
                )
                val_ds = create_honor_of_kings_dataset(
                    data_root=game_config['data_root'],
                    split_file=game_config['val_split'],
                    patch_size=game_config['patch_size'],
                    mode='val'
                )
            
            # TODO: 添加其他游戏数据集
            # elif game_name == 'qq_speed':
            #     ...
            # elif game_name == 'genshin_impact':
            #     ...
            
            else:
                print(f"Warning: Unknown game type {game_name}, skipping...")
                continue
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            
            print(f"{game_name} - Train: {len(train_ds)}, Val: {len(val_ds)}")
            
        except Exception as e:
            print(f"Error loading {game_name} dataset: {e}")
            continue
    
    if not train_datasets:
        raise ValueError("No valid datasets loaded!")
    
    # 合并数据集
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    # 创建数据加载器
    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"\n=== Combined Dataset Statistics ===")
    print(f"Total Train Samples: {len(combined_train)}")
    print(f"Total Val Samples: {len(combined_val)}")
    print(f"Batch Size: {batch_size}")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")
    
    return {
        'train': train_loader,
        'val': val_loader
    }


def setup_logging(config: Dict[str, Any]) -> List[pl.loggers.Logger]:
    """
    设置日志记录器
    
    Args:
        config: 训练配置
    
    Returns:
        loggers: 日志记录器列表
    """
    loggers = []
    
    # TensorBoard日志
    if config.get('tensorboard', {}).get('enabled', True):
        tb_logger = TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name='mobile_inpainting',
            version=f"v{config['experiment']['version']}",
            default_hp_metric=False
        )
        loggers.append(tb_logger)
        print(f"TensorBoard logging enabled: {tb_logger.log_dir}")
    
    # W&B日志
    if config.get('wandb', {}).get('enabled', False):
        wandb_logger = WandbLogger(
            project=config['wandb']['project'],
            name=config['experiment']['name'],
            version=config['experiment']['version'],
            save_dir=config['logging']['log_dir']
        )
        loggers.append(wandb_logger)
        print(f"W&B logging enabled: {config['wandb']['project']}")
    
    return loggers


def setup_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """
    设置训练回调函数
    
    Args:
        config: 训练配置
    
    Returns:
        callbacks: 回调函数列表
    """
    callbacks = []
    
    # 模型检查点
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='mobile_inpainting-{epoch:02d}-{val/ssim:.4f}',
        monitor='val/ssim',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    if config.get('early_stopping', {}).get('enabled', True):
        early_stop_callback = EarlyStopping(
            monitor='val/total_loss',
            patience=config['early_stopping']['patience'],
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks


def create_trainer_instance(config: Dict[str, Any],
                           loggers: List[pl.loggers.Logger],
                           callbacks: List[pl.Callback]) -> pl.Trainer:
    """
    创建PyTorch Lightning训练器
    
    Args:
        config: 训练配置
        loggers: 日志记录器
        callbacks: 回调函数
    
    Returns:
        trainer: Lightning训练器
    """
    trainer_config = config['trainer']
    
    # 分布式训练策略
    strategy = None
    if trainer_config.get('distributed', False):
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=trainer_config['max_epochs'],
        accelerator=trainer_config.get('accelerator', 'gpu'),
        devices=trainer_config.get('devices', 1),
        strategy=strategy,
        precision=trainer_config.get('precision', 32),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=loggers,
        callbacks=callbacks,
        deterministic=True
    )
    
    return trainer


def main():
    """训练主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Mobile Frame Interpolation Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子 
    set_random_seeds(config.get('seed', 42))
    
    # 创建输出目录
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    print("="*60)
    print("🚀 Mobile Frame Interpolation Training Started")
    print("="*60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Version: {config['experiment']['version']}")
    print(f"Config: {args.config}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("="*60)
    
    try:
        # 创建数据集
        print("\n📊 Setting up datasets...")
        dataloaders = create_multi_game_dataset(
            data_configs=config['datasets'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
        
        # 创建模型
        print("\n🧠 Creating model...")
        model_trainer = create_trainer(
            model_config=config['model'],
            training_config=config['training'],
            teacher_model_path=config['training'].get('teacher_model_path')
        )
        
        # 设置日志和回调
        print("\n📝 Setting up logging and callbacks...")
        loggers = setup_logging(config)
        callbacks = setup_callbacks(config)
        
        # 创建训练器
        print("\n⚡ Creating PyTorch Lightning trainer...")
        trainer = create_trainer_instance(config, loggers, callbacks)
        
        # 开始训练
        if not args.test_only:
            print("\n🏋️ Starting training...")
            start_time = time.time()
            
            trainer.fit(
                model=model_trainer,
                train_dataloaders=dataloaders['train'],
                val_dataloaders=dataloaders['val'],
                ckpt_path=args.resume
            )
            
            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
        
        # 运行测试
        if 'test' in dataloaders or args.test_only:
            print("\n🧪 Running final testing...")
            if args.test_only and args.resume:
                # 从检查点加载模型进行测试
                model_trainer = model_trainer.load_from_checkpoint(args.resume)
            
            test_results = trainer.test(
                model=model_trainer,
                dataloaders=dataloaders.get('test', dataloaders['val'])
            )
            print(f"Test Results: {test_results}")
        
        print("\n🎉 All tasks completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    main()