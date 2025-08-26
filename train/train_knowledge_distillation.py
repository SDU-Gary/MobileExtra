#!/usr/bin/env python3
"""
@file train_knowledge_distillation.py
@brief 知识蒸馏专用训练脚本

功能描述：
- 专门用于Teacher-Student知识蒸馏训练
- 支持多阶段蒸馏：特征蒸馏 + 输出蒸馏
- 渐进式蒸馏策略和温度调度
- 蒸馏效果评估和对比分析

蒸馏流程：
1. 加载预训练的Teacher模型
2. 初始化轻量级Student模型
3. 设计蒸馏损失函数组合
4. 执行渐进式蒸馏训练
5. 评估蒸馏效果和性能对比

技术特点：
- 多层次特征蒸馏
- 自适应温度调度
- 蒸馏-微调联合优化
- 性能保持率实时监控

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'networks'))

from training_framework import FrameInterpolationTrainer
from mobile_inpainting_network import MobileInpaintingNetwork
from train import create_multi_game_dataset, load_config, set_random_seeds


class KnowledgeDistillationTrainer(pl.LightningModule):
    """
    知识蒸馏训练器
    
    专门用于Teacher-Student框架的蒸馏训练
    """
    
    def __init__(self,
                 teacher_model_path: str,
                 student_config: Dict[str, Any],
                 distillation_config: Dict[str, Any],
                 training_config: Dict[str, Any]):
        """
        初始化知识蒸馏训练器
        
        Args:
            teacher_model_path: Teacher模型路径
            student_config: Student模型配置
            distillation_config: 蒸馏配置
            training_config: 训练配置
        """
        super(KnowledgeDistillationTrainer, self).__init__()
        
        self.save_hyperparameters()
        
        # 加载Teacher模型
        print(f"Loading teacher model from {teacher_model_path}")
        self.teacher_model = self._load_teacher_model(teacher_model_path)
        self.teacher_model.eval()
        
        # 冻结Teacher模型参数
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 创建Student模型
        print("Creating student model...")
        self.student_model = MobileInpaintingNetwork(
            input_channels=student_config.get('input_channels', 7),
            output_channels=student_config.get('output_channels', 3)
        )
        
        # 蒸馏配置
        self.temperature = distillation_config.get('temperature', 4.0)
        self.alpha = distillation_config.get('alpha', 0.7)  # 蒸馏损失权重
        self.beta = distillation_config.get('beta', 0.3)    # 硬标签损失权重
        
        # 温度调度
        self.temperature_schedule = distillation_config.get('temperature_schedule', {})
        self.current_temperature = self.temperature
        
        # 损失函数
        self.hard_loss = nn.L1Loss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        self.feature_loss = nn.MSELoss()
        
        # 训练配置
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        
        # 性能监控
        self.teacher_performance = {}
        self.student_performance = {}
        self.distillation_metrics = {}
        
        print(f"=== Knowledge Distillation Trainer ===")
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"Student parameters: {sum(p.numel() for p in self.student_model.parameters()):,}")
        print(f"Initial temperature: {self.temperature}")
        print(f"Alpha (distillation weight): {self.alpha}")
        print(f"Beta (hard loss weight): {self.beta}")
    
    def _load_teacher_model(self, model_path: str) -> nn.Module:
        """
        加载Teacher模型
        
        Args:
            model_path: 模型路径
        
        Returns:
            teacher_model: Teacher模型
        """
        if model_path.endswith('.ckpt'):
            # 从Lightning检查点加载
            teacher_trainer = FrameInterpolationTrainer.load_from_checkpoint(model_path)
            return teacher_trainer.student_model
        else:
            # 从PyTorch检查点加载
            teacher_model = MobileInpaintingNetwork()
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                teacher_model.load_state_dict(checkpoint['state_dict'])
            else:
                teacher_model.load_state_dict(checkpoint)
            
            return teacher_model
    
    def _update_temperature(self, epoch: int):
        """
        更新蒸馏温度
        
        Args:
            epoch: 当前epoch
        """
        if 'type' in self.temperature_schedule:
            schedule_type = self.temperature_schedule['type']
            
            if schedule_type == 'linear':
                # 线性衰减
                start_temp = self.temperature_schedule.get('start', self.temperature)
                end_temp = self.temperature_schedule.get('end', 1.0)
                total_epochs = self.temperature_schedule.get('epochs', 100)
                
                progress = min(epoch / total_epochs, 1.0)
                self.current_temperature = start_temp + (end_temp - start_temp) * progress
                
            elif schedule_type == 'cosine':
                # 余弦衰减
                start_temp = self.temperature_schedule.get('start', self.temperature)
                end_temp = self.temperature_schedule.get('end', 1.0)
                total_epochs = self.temperature_schedule.get('epochs', 100)
                
                progress = min(epoch / total_epochs, 1.0)
                self.current_temperature = end_temp + (start_temp - end_temp) * \
                                         (1 + np.cos(np.pi * progress)) / 2
            
            self.log('distillation/temperature', self.current_temperature)
    
    def forward(self, x):
        """前向传播"""
        return self.student_model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        input_data, target, input_prev, target_prev = batch
        
        # Student模型预测
        student_output = self.student_model(input_data)
        
        # Teacher模型预测（无梯度）
        with torch.no_grad():
            teacher_output = self.teacher_model(input_data)
        
        # 计算损失
        losses = self._compute_distillation_losses(
            student_output, teacher_output, target
        )
        
        total_loss = losses['total']
        
        # 记录损失
        for key, value in losses.items():
            self.log(f'train/{key}_loss', value, prog_bar=(key=='total'))
        
        # 记录性能指标
        with torch.no_grad():
            student_ssim = self._calculate_ssim(student_output, target)
            teacher_ssim = self._calculate_ssim(teacher_output, target)
            
            self.log('train/student_ssim', student_ssim)
            self.log('train/teacher_ssim', teacher_ssim)
            self.log('train/performance_ratio', student_ssim / (teacher_ssim + 1e-8))
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        input_data, target, input_prev, target_prev = batch
        
        # 模型预测
        student_output = self.student_model(input_data)
        
        with torch.no_grad():
            teacher_output = self.teacher_model(input_data)
        
        # 计算损失
        losses = self._compute_distillation_losses(
            student_output, teacher_output, target
        )
        
        # 计算性能指标
        student_ssim = self._calculate_ssim(student_output, target)
        teacher_ssim = self._calculate_ssim(teacher_output, target)
        student_psnr = self._calculate_psnr(student_output, target)
        teacher_psnr = self._calculate_psnr(teacher_output, target)
        
        # 记录验证指标
        self.log('val/total_loss', losses['total'], prog_bar=True)
        self.log('val/student_ssim', student_ssim, prog_bar=True)
        self.log('val/teacher_ssim', teacher_ssim)
        self.log('val/student_psnr', student_psnr)
        self.log('val/teacher_psnr', teacher_psnr)
        self.log('val/ssim_retention', student_ssim / (teacher_ssim + 1e-8), prog_bar=True)
        self.log('val/psnr_retention', student_psnr / (teacher_psnr + 1e-8))
        
        return {
            'val_loss': losses['total'],
            'student_ssim': student_ssim,
            'teacher_ssim': teacher_ssim,
            'retention_rate': student_ssim / (teacher_ssim + 1e-8)
        }
    
    def _compute_distillation_losses(self, 
                                   student_output: torch.Tensor,
                                   teacher_output: torch.Tensor,
                                   target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        
        Args:
            student_output: Student输出
            teacher_output: Teacher输出
            target: 真值目标
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 硬标签损失（与真值的L1损失）
        hard_loss = self.hard_loss(student_output, target)
        losses['hard'] = hard_loss
        
        # 软标签蒸馏损失（KL散度）
        student_soft = F.log_softmax(student_output / self.current_temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / self.current_temperature, dim=1)
        
        kl_loss = self.kl_div_loss(student_soft, teacher_soft) * (self.current_temperature ** 2)
        losses['kl_divergence'] = kl_loss
        
        # 特征蒸馏损失（输出特征的MSE）
        feature_loss = self.feature_loss(student_output, teacher_output.detach())
        losses['feature'] = feature_loss
        
        # 总损失
        total_loss = (self.beta * hard_loss + 
                     self.alpha * kl_loss + 
                     0.1 * feature_loss)
        losses['total'] = total_loss
        
        return losses
    
    def on_train_epoch_start(self):
        """训练epoch开始时的回调"""
        # 更新温度调度
        self._update_temperature(self.current_epoch)
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss'
            }
        }
    
    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算SSIM指标"""
        # 简化的SSIM计算
        mu1 = pred.mean()
        mu2 = target.mean() 
        sigma1_sq = ((pred - mu1) ** 2).mean()
        sigma2_sq = ((target - mu2) ** 2).mean()
        sigma12 = ((pred - mu1) * (target - mu2)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim
    
    def _calculate_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算PSNR指标"""
        mse = F.mse_loss(pred, target)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # 输入范围[-1,1]
        return psnr


def create_distillation_trainer(teacher_model_path: str,
                               config: Dict[str, Any]) -> KnowledgeDistillationTrainer:
    """
    创建知识蒸馏训练器
    
    Args:
        teacher_model_path: Teacher模型路径
        config: 训练配置
    
    Returns:
        trainer: 蒸馏训练器
    """
    return KnowledgeDistillationTrainer(
        teacher_model_path=teacher_model_path,
        student_config=config['model'],
        distillation_config=config['distillation'],
        training_config=config['training']
    )


def main():
    """知识蒸馏训练主函数"""
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to distillation config file')
    parser.add_argument('--teacher', type=str, required=True,
                       help='Path to teacher model')
    parser.add_argument('--output-dir', type=str, default='./distillation_output',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置随机种子
    set_random_seeds(config.get('seed', 42))
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🎓 Knowledge Distillation Training Started")
    print("="*60)
    print(f"Teacher Model: {args.teacher}")
    print(f"Config: {args.config}")
    print(f"Output Dir: {output_dir}")
    print("="*60)
    
    try:
        # 创建数据集
        print("\n📊 Setting up datasets...")
        dataloaders = create_multi_game_dataset(
            data_configs=config['datasets'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
        
        # 创建蒸馏训练器
        print("\n🎓 Creating distillation trainer...")
        distillation_trainer = create_distillation_trainer(
            teacher_model_path=args.teacher,
            config=config
        )
        
        # 设置Logger和Callbacks
        logger = TensorBoardLogger(
            save_dir=str(output_dir / 'logs'),
            name='knowledge_distillation'
        )
        
        callbacks = [
            ModelCheckpoint(
                dirpath=str(output_dir / 'checkpoints'),
                filename='student-{epoch:02d}-{val/ssim_retention:.4f}',
                monitor='val/ssim_retention',
                mode='max',
                save_top_k=3,
                save_last=True
            ),
            EarlyStopping(
                monitor='val/total_loss',
                patience=20,
                mode='min'
            )
        ]
        
        # 创建Lightning训练器
        trainer = pl.Trainer(
            max_epochs=config['trainer']['max_epochs'],
            accelerator='gpu',
            devices=1,
            precision=16,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            deterministic=True
        )
        
        # 开始蒸馏训练
        print("\n🏋️ Starting knowledge distillation...")
        start_time = time.time()
        
        trainer.fit(
            model=distillation_trainer,
            train_dataloaders=dataloaders['train'],
            val_dataloaders=dataloaders['val']
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Distillation completed in {training_time/3600:.2f} hours")
        
        # 保存最终的Student模型
        student_model_path = output_dir / 'student_model.pth'
        torch.save(distillation_trainer.student_model.state_dict(), student_model_path)
        print(f"Student model saved to {student_model_path}")
        
        # 性能对比测试
        print("\n📊 Running performance comparison...")
        test_results = trainer.test(
            model=distillation_trainer,
            dataloaders=dataloaders['val']
        )
        
        print(f"Final Test Results: {test_results}")
        print("\n🎉 Knowledge distillation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Distillation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()