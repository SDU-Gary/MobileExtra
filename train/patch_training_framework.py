#!/usr/bin/env python3
"""
Patch-based training framework for PyTorch Lightning.

Features:
- Patch-aware training loops with boundary handling
- Multi-scale loss computation
- Adaptive performance optimization  
- Integrated visualization system
- Progressive training strategies

Compatible with existing FrameInterpolationTrainer architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass

# Import existing components
try:
    from training_framework import (
        PerceptualLoss, DistillationLoss, FrameInterpolationTrainer,
        CombinedLoss, EdgeLoss, TemporalConsistencyLoss
    )
except ImportError as e:
    print(f"Training framework import warning: {e}")
    # Provide basic loss function placeholders
    class PerceptualLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, pred, target):
            return torch.tensor(0.0)
    
    class DistillationLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, student, teacher, target):
            return torch.tensor(0.0), {}
    
    class EdgeLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, pred, target):
            return torch.tensor(0.0)
    
    class FrameInterpolationTrainer:
        pass
    class CombinedLoss:
        pass
    class TemporalConsistencyLoss:
        pass

try:
    from training_monitor import TrainingMonitor, create_training_monitor
except ImportError:
    try:
        from train.training_monitor import TrainingMonitor, create_training_monitor
    except ImportError:
        print("Training monitor module import warning")
        class TrainingMonitor:
            def __init__(self, log_dir, config):
                pass
            def log_training_progress(self, progress_dict):
                pass
        def create_training_monitor(config, model=None):
            return TrainingMonitor(config.get('log_dir', './logs'), config)

# Import patch components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu'))

# Use absolute imports to avoid relative import issues
try:
    from src.npu.networks.patch_inpainting import PatchBasedInpainting, PatchInpaintingConfig
    from src.npu.networks.mobile_inpainting_network import MobileInpaintingNetwork
except ImportError:
    # Alternative import method
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.npu.networks.patch_inpainting import PatchBasedInpainting, PatchInpaintingConfig
    from src.npu.networks.mobile_inpainting_network import MobileInpaintingNetwork

# 导入patch数据集
try:
    from patch_aware_dataset import PatchAwareDataset, PatchTrainingConfig
except ImportError:
    try:
        from train.patch_aware_dataset import PatchAwareDataset, PatchTrainingConfig
    except ImportError:
        print("Patch数据集导入警告")
        class PatchAwareDataset:
            pass
        class PatchTrainingConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

# 导入patch可视化组件
try:
    from patch_tensorboard_logger import create_patch_visualizer, PatchTensorBoardLogger
except ImportError:
    try:
        from train.patch_tensorboard_logger import create_patch_visualizer, PatchTensorBoardLogger
    except ImportError:
        print("Patch TensorBoard可视化导入警告")
        def create_patch_visualizer(log_dir, config=None):
            return None
        class PatchTensorBoardLogger:
            def __init__(self, *args, **kwargs):
                pass
            def log_patch_visualization(self, *args, **kwargs):
                pass
            def log_patch_comparison(self, *args, **kwargs):
                pass
            def log_training_step(self, *args, **kwargs):
                pass
            def log_validation_step(self, *args, **kwargs):
                pass


@dataclass
class PatchTrainingScheduleConfig:
    """Patch训练调度配置"""
    # 训练阶段配置
    patch_warmup_epochs: int = 20           # patch模式热身阶段
    mixed_training_epochs: int = 50         # 混合训练阶段
    full_fine_tuning_epochs: int = 30       # 全图fine-tuning阶段
    
    # 模式切换策略
    initial_patch_probability: float = 0.9   # 初始patch概率
    final_patch_probability: float = 0.3     # 最终patch概率
    
    # 自适应调度
    enable_adaptive_scheduling: bool = True  # 启用自适应调度
    loss_patience: int = 5                   # 损失无改善容忍epochs
    
    # 性能阈值
    patch_efficiency_threshold: float = 1.5  # patch效率阈值（相比full模式）
    quality_drop_threshold: float = 0.05     # 质量下降容忍阈值


class PatchAwareLoss(nn.Module):
    """
    Patch-Aware损失函数系统
    
    功能：
    1. Multi-scale损失：patch-level + image-level
    2. Boundary-aware处理：patch边界的特殊损失
    3. Consistency损失：patch融合一致性
    4. 自适应权重：基于训练阶段的动态权重调整
    """
    
    def __init__(self, 
                 lambda_patch: float = 1.0,
                 lambda_full: float = 1.0,
                 lambda_boundary: float = 0.5,
                 lambda_consistency: float = 0.3,
                 enable_adaptive_weights: bool = True):
        super(PatchAwareLoss, self).__init__()
        
        # 损失权重
        self.lambda_patch = lambda_patch
        self.lambda_full = lambda_full
        self.lambda_boundary = lambda_boundary
        self.lambda_consistency = lambda_consistency
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # 基础损失函数
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()
        
        # patch专用损失
        self.boundary_kernel = self._create_boundary_kernel()
        
    def _create_boundary_kernel(self) -> torch.Tensor:
        """创建边界检测卷积核"""
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                mode: str,
                epoch: int = 0,
                metadata: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算patch-aware损失
        
        Args:
            predictions: 预测结果字典
            targets: 目标结果字典
            mode: 'patch' (现在只支持patch模式)
            epoch: 当前epoch（用于自适应权重）
            metadata: 额外元数据
            
        Returns:
            total_loss: 总损失
            loss_dict: 分项损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # 计算自适应权重
        if self.enable_adaptive_weights:
            weights = self._compute_adaptive_weights(epoch, mode)
        else:
            weights = {
                'patch': self.lambda_patch,
                'full': self.lambda_full,
                'boundary': self.lambda_boundary,
                'consistency': self.lambda_consistency
            }
        
        # Patch模式损失
        if 'patch' in predictions:
            patch_pred = predictions['patch']  # [N, 3, 128, 128]
            patch_target = targets['patch']    # [N, 3, 128, 128]
            
            # 基础patch损失
            patch_l1 = self.l1_loss(patch_pred, patch_target)
            patch_perceptual = self.perceptual_loss(patch_pred, patch_target)
            patch_edge = self.edge_loss(patch_pred, patch_target)
            
            # 边界感知损失
            boundary_loss = self._compute_boundary_loss(
                patch_pred, patch_target, metadata.get('patch_metadata', []) if metadata else []
            )
            
            losses['patch_l1'] = patch_l1
            losses['patch_perceptual'] = patch_perceptual
            losses['patch_edge'] = patch_edge
            losses['patch_boundary'] = boundary_loss
            
            # patch总损失 - 🔧 增强感知损失和边缘损失权重
            patch_total = (patch_l1 + 0.8 * patch_perceptual + 
                          0.5 * patch_edge + weights['boundary'] * boundary_loss)
            total_loss += weights['patch'] * patch_total
        
        # 注意：删除了Full模式损失和一致性损失，现在只专注于patch训练
        
        losses['total'] = total_loss
        return total_loss, losses
    
    def _compute_adaptive_weights(self, epoch: int, mode: str) -> Dict[str, float]:
        """计算自适应损失权重 - 简化为只支持patch模式"""
        # 基于训练进度调整权重
        progress = min(epoch / 100.0, 1.0)  # 假设100 epoch为完整训练
        
        # 现在只有patch模式：early重patch，later重boundary
        patch_weight = 1.0
        boundary_weight = 0.5 - 0.2 * progress  # 训练后期减少边界权重
        
        return {
            'patch': patch_weight,
            'boundary': boundary_weight,
            'consistency': 0.0,  # 不再使用一致性损失
            'full': 0.0  # 不再使用全图损失
        }
    
    def _compute_boundary_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                              patch_metadata: List[Dict]) -> torch.Tensor:
        """计算边界感知损失"""
        if len(patch_metadata) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # 简化版边界损失：在patch边界区域加权
        boundary_loss = 0.0
        
        for i, metadata in enumerate(patch_metadata):
            if i >= pred.shape[0]:
                break
                
            # 检测边界区域（简单的边缘检测）
            if self.boundary_kernel.device != pred.device:
                self.boundary_kernel = self.boundary_kernel.to(pred.device)
            
            # 对每个通道分别处理
            patch_pred = pred[i:i+1]  # [1, 3, 128, 128]
            patch_target = target[i:i+1]  # [1, 3, 128, 128]
            
            # 边界检测（使用灰度化后的结果）
            pred_gray = torch.mean(patch_pred, dim=1, keepdim=True)  # [1, 1, 128, 128]
            boundary_map = F.conv2d(pred_gray, self.boundary_kernel, padding=1)
            boundary_map = torch.sigmoid(boundary_map * 0.1)
            
            # 在边界区域加权L1损失
            boundary_weighted_loss = torch.mean(
                F.l1_loss(patch_pred, patch_target, reduction='none') * 
                (1.0 + boundary_map)  # 边界区域权重增强
            )
            boundary_loss += boundary_weighted_loss
        
        return boundary_loss / len(patch_metadata) if patch_metadata else torch.tensor(0.0, device=pred.device)
    


class PatchTrainingScheduler:
    """
    Patch训练调度器
    
    功能：
    1. 训练阶段管理
    2. 模式概率动态调整
    3. 自适应性能优化
    4. 早停和质量监控
    """
    
    def __init__(self, config: PatchTrainingScheduleConfig):
        self.config = config
        self.current_epoch = 0
        self.patch_performance_history = []
        self.full_performance_history = []
        
    def get_patch_probability(self, epoch: int, recent_losses: Optional[List[float]] = None) -> float:
        """获取当前epoch应该使用的patch模式概率"""
        self.current_epoch = epoch
        
        # 基础调度策略
        if epoch < self.config.patch_warmup_epochs:
            # 热身阶段：高patch概率
            base_prob = self.config.initial_patch_probability
        elif epoch < self.config.patch_warmup_epochs + self.config.mixed_training_epochs:
            # 混合训练阶段：线性下降
            progress = (epoch - self.config.patch_warmup_epochs) / self.config.mixed_training_epochs
            base_prob = (self.config.initial_patch_probability - 
                        (self.config.initial_patch_probability - self.config.final_patch_probability) * progress)
        else:
            # Fine-tuning阶段：低patch概率
            base_prob = self.config.final_patch_probability
        
        # 自适应调整
        if self.config.enable_adaptive_scheduling and recent_losses:
            base_prob = self._adaptive_adjustment(base_prob, recent_losses)
        
        return np.clip(base_prob, 0.1, 0.9)
    
    def _adaptive_adjustment(self, base_prob: float, recent_losses: List[float]) -> float:
        """基于最近损失历史的自适应调整"""
        if len(recent_losses) < 3:
            return base_prob
        
        # 检查损失趋势
        recent_trend = recent_losses[-1] - recent_losses[-3]
        
        if recent_trend > 0:  # 损失上升
            # 增加patch概率（patch训练通常更稳定）
            adjustment = 0.1
        else:  # 损失下降
            # 保持当前策略
            adjustment = 0.0
        
        return base_prob + adjustment
    
    def should_switch_strategy(self, current_metrics: Dict[str, float]) -> bool:
        """判断是否应该切换训练策略"""
        if not self.config.enable_adaptive_scheduling:
            return False
        
        # 基于性能指标决定策略切换
        # 这里可以添加更复杂的逻辑
        return False


class PatchFrameInterpolationTrainer(pl.LightningModule):
    """
    Patch-Based帧插值训练器
    
    继承自PyTorch Lightning，支持patch-aware训练：
    1. 智能模式调度
    2. 动态batch处理  
    3. 多尺度损失优化
    4. 自适应性能调优
    5. 完整的监控和日志
    """
    
    def __init__(self,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 patch_config: Optional[PatchTrainingConfig] = None,
                 schedule_config: Optional[PatchTrainingScheduleConfig] = None,
                 teacher_model_path: Optional[str] = None,
                 full_config: Optional[Dict[str, Any]] = None):
        super(PatchFrameInterpolationTrainer, self).__init__()
        
        self.save_hyperparameters()
        
        # 配置
        self.model_config = model_config
        self.training_config = training_config
        self.patch_config = patch_config or PatchTrainingConfig()
        self.schedule_config = schedule_config or PatchTrainingScheduleConfig()
        
        # 创建patch-based网络
        inpainting_config = model_config.get('inpainting_network', {})
        patch_inpainting_config = PatchInpaintingConfig(
            enable_patch_mode=self.patch_config.enable_patch_mode,
            patch_network_channels=24  # 轻量化配置
        )
        
        self.student_model = PatchBasedInpainting(
            input_channels=inpainting_config.get('input_channels', 7),
            output_channels=inpainting_config.get('output_channels', 3),
            config=patch_inpainting_config
        )
        
        # 注意：删除了fallback全图网络，现在只专注patch训练
        
        # 加载教师网络（知识蒸馏）
        self.teacher_model = None
        if teacher_model_path and os.path.exists(teacher_model_path):
            self.teacher_model = MobileInpaintingNetwork(
                input_channels=inpainting_config.get('input_channels', 7),
                output_channels=inpainting_config.get('output_channels', 3)
            )
            checkpoint = torch.load(teacher_model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.teacher_model.load_state_dict(checkpoint['state_dict'])
            else:
                self.teacher_model.load_state_dict(checkpoint)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        # 损失函数
        self.patch_loss = PatchAwareLoss()
        
        if self.teacher_model is not None:
            self.distillation_loss = DistillationLoss()
        
        # 训练调度器
        self.scheduler = PatchTrainingScheduler(self.schedule_config)
        
        # 性能监控
        monitor_config = (full_config or {}).copy()
        monitor_config['log_dir'] = training_config.get('log_dir', './logs')
        self.training_monitor = create_training_monitor(
            config=monitor_config,
            model=self.student_model
        )
        
        # Patch专用可视化系统
        viz_config = {
            'visualization_frequency': full_config.get('visualization', {}).get('visualization_frequency', 100),
            'save_frequency': full_config.get('visualization', {}).get('save_frequency', 500),
            'enable_visualization': full_config.get('visualization', {}).get('enable_visualization', True)
        }
        self.patch_visualizer = create_patch_visualizer(
            log_dir=training_config.get('log_dir', './logs'),
            config=viz_config
        ) if viz_config['enable_visualization'] else None
        
        # 训练统计
        self.training_stats = {
            'patch_steps': 0,
            'recent_losses': [],
            'best_val_loss': float('inf'),
            'patience_count': 0
        }
        
        # 可视化步骤计数器（避免与Lightning的global_step冲突）
        self.visualization_step = 0
        
        print(f"Patch visualization system: {'ENABLED' if self.patch_visualizer else 'DISABLED'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.student_model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """训练步骤"""
        current_epoch = self.current_epoch
        
        # 获取当前的patch概率
        patch_probability = self.scheduler.get_patch_probability(
            current_epoch, 
            self.training_stats['recent_losses'][-10:] if self.training_stats['recent_losses'] else None
        )
        
        # 动态设置patch模式概率
        self.student_model.config.enable_patch_mode = True
        
        # 准备输入数据
        predictions = {}
        targets = {}
        
        # 处理patch数据（现在只有patch数据）
        if 'patch_input' in batch and len(batch['patch_input']) > 0:
            patch_input = batch['patch_input']    # [N, 7, 128, 128]
            patch_target = batch['patch_target']  # [N, 3, 128, 128]
            
            # Patch网络推理
            patch_pred = self.student_model.patch_network(patch_input)
            predictions['patch'] = patch_pred
            targets['patch'] = patch_target
            
            self.training_stats['patch_steps'] += 1
        else:
            # 如果没有patch数据，跳过这个batch
            return {'loss': torch.tensor(0.0, requires_grad=True)}
        
        # 现在只有patch模式
        mode = 'patch'
        
        # 计算损失
        loss, loss_dict = self.patch_loss(
            predictions, targets, mode, current_epoch,
            metadata=batch.get('batch_info', {})
        )
        
        # 注意：知识蒸馏暂时禁用，因为删除了全图模式
        # TODO: 如需知识蒸馏，需要实现patch-level的蒸馏策略
        
        # 更新统计
        self.training_stats['recent_losses'].append(loss.item())
        if len(self.training_stats['recent_losses']) > 20:
            self.training_stats['recent_losses'].pop(0)
        
        # 日志记录
        self.log_dict({
            'train_loss': loss,
            'patch_probability': patch_probability,
            'mode': 0,  # 现在只有patch模式
            **{f'train_{k}': v for k, v in loss_dict.items() if k != 'total'}
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        # Patch可视化记录（使用Lightning的global_step）
        current_global_step = self.global_step if hasattr(self, 'trainer') and self.trainer else self.visualization_step
        
        if self.patch_visualizer and self.patch_visualizer.should_visualize(current_global_step):
            try:
                # 记录patch对比（输入|目标|预测）
                if 'patch' in predictions and 'patch' in targets:
                    self.patch_visualizer.log_patch_comparison(
                        step=current_global_step,
                        patch_inputs=batch['patch_input'][:8],  # 最多显示8个patches
                        patch_targets=targets['patch'][:8],
                        patch_predictions=predictions['patch'][:8],
                        tag=f'training_epoch_{current_epoch}'
                    )
                
                # 记录训练步骤统计
                self.patch_visualizer.log_training_step(
                    step=current_global_step,
                    epoch=current_epoch,
                    mode=mode,
                    loss_dict=loss_dict,
                    batch_info=batch.get('batch_info', {}),
                    learning_rate=self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else None
                )
                
            except Exception as e:
                print(f"WARNING: Visualization logging failed: {e}")
        
        # 更新可视化步骤计数（仅在没有trainer时使用）
        if not hasattr(self, 'trainer') or not self.trainer:
            self.visualization_step += 1
        
        return {
            'loss': loss,
            'mode': mode,
            'batch_info': batch.get('batch_info', {}),
            'predictions': predictions,
            'targets': targets
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        # 验证时使用混合模式，patch概率固定为0.5
        predictions = {}
        targets = {}
        
        # 处理验证数据（与训练步骤类似，但无梯度更新）
        if 'patch_input' in batch and len(batch['patch_input']) > 0:
            patch_pred = self.student_model.patch_network(batch['patch_input'])
            predictions['patch'] = patch_pred
            targets['patch'] = batch['patch_target']
        
        # 现在只有patch模式
        mode = 'patch'
        
        # 计算验证损失
        val_loss, val_loss_dict = self.patch_loss(
            predictions, targets, mode, self.current_epoch
        )
        
        # 日志记录
        self.log_dict({
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_loss_dict.items() if k != 'total'}
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        # 验证可视化记录
        current_global_step = self.global_step if hasattr(self, 'trainer') and self.trainer else self.visualization_step
        
        if self.patch_visualizer and self.current_epoch % 5 == 0:  # 每5个epoch记录验证图像
            try:
                if 'patch' in predictions and 'patch' in targets:
                    self.patch_visualizer.log_patch_comparison(
                        step=current_global_step,
                        patch_inputs=batch['patch_input'][:4],  # 验证时显示4个patches
                        patch_targets=targets['patch'][:4],
                        patch_predictions=predictions['patch'][:4],
                        tag=f'validation_epoch_{self.current_epoch}'
                    )
                
                # 记录验证损失
                self.patch_visualizer.log_validation_step(
                    step=current_global_step,
                    val_loss_dict=val_loss_dict
                )
                
            except Exception as e:
                print(f"WARNING: Validation visualization logging failed: {e}")
        
        return {
            'val_loss': val_loss,
            'mode': mode,
            'predictions': predictions,
            'targets': targets
        }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 优化器配置
        optimizer_config = self.training_config.get('optimizer', {})
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=optimizer_config.get('learning_rate', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler_config = self.training_config.get('scheduler', {})
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_epoch_end(self) -> None:
        """Epoch结束时的处理"""
        # 更新训练监控
        if hasattr(self, 'training_monitor'):
            self.training_monitor.log_training_progress({
                'epoch': self.current_epoch,
                'patch_steps': self.training_stats['patch_steps']
            })
        
        # Patch训练进度可视化
        if self.patch_visualizer and self.current_epoch % 10 == 0:  # 每10个epoch记录训练进度
            try:
                current_global_step = self.global_step if hasattr(self, 'trainer') and self.trainer else self.visualization_step
                patch_stats = self.get_training_stats()
                self.patch_visualizer.log_training_progress(
                    step=current_global_step,
                    epoch=self.current_epoch,
                    patch_stats=patch_stats
                )
            except Exception as e:
                print(f"WARNING: Training progress visualization failed: {e}")
        
        # 自适应调整
        current_val_loss = self.trainer.callback_metrics.get('val_loss', float('inf'))
        if current_val_loss < self.training_stats['best_val_loss']:
            self.training_stats['best_val_loss'] = current_val_loss
            self.training_stats['patience_count'] = 0
        else:
            self.training_stats['patience_count'] += 1
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        total_steps = self.training_stats['patch_steps']
        
        # 获取patch网络性能统计
        patch_network_stats = {}
        if hasattr(self.student_model, 'get_performance_stats'):
            try:
                patch_network_stats = self.student_model.get_performance_stats()
            except:
                pass
        
        return {
            **self.training_stats,
            'total_steps': total_steps,
            'patch_ratio': 1.0,  # 现在只有patch模式
            'patch_mode_count': self.training_stats['patch_steps'],
            'avg_patches_per_image': patch_network_stats.get('avg_patches_per_image', 0),
            'total_patches_generated': patch_network_stats.get('total_patches_processed', 0),
            'cache_hit_rate': 0.0,  # 占位符，实际需要从数据集获取
            'processing_speed': 100.0  # 占位符
        }


def create_patch_trainer(model_config: Dict[str, Any],
                        training_config: Dict[str, Any],
                        patch_config: Optional[PatchTrainingConfig] = None,
                        schedule_config: Optional[PatchTrainingScheduleConfig] = None,
                        teacher_model_path: Optional[str] = None,
                        full_config: Optional[Dict[str, Any]] = None) -> PatchFrameInterpolationTrainer:
    """创建Patch训练器的工厂函数"""
    
    trainer = PatchFrameInterpolationTrainer(
        model_config=model_config,
        training_config=training_config,
        patch_config=patch_config,
        schedule_config=schedule_config,
        teacher_model_path=teacher_model_path,
        full_config=full_config
    )
    
    # 如果trainer有可视化系统，输出启用状态
    if hasattr(trainer, 'patch_visualizer') and trainer.patch_visualizer:
        print("SUCCESS: Patch visualization system integrated into training framework")
        print(f"   - 可视化频率: 每{trainer.patch_visualizer.vis_frequency}步")
        print(f"   - 保存频率: 每{trainer.patch_visualizer.save_frequency}步")
        print(f"   - 日志目录: {trainer.patch_visualizer.log_dir}")
    
    return trainer


def test_patch_training_framework():
    """测试Patch训练框架"""
    # 配置
    model_config = {
        'inpainting_network': {
            'input_channels': 7,
            'output_channels': 3
        }
    }
    
    training_config = {
        'optimizer': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5
        },
        'scheduler': {
            'T_max': 100,
            'eta_min': 1e-6
        },
        'log_dir': './logs'
    }
    
    # 创建训练器
    try:
        trainer = create_patch_trainer(
            model_config=model_config,
            training_config=training_config
        )
        
        print("SUCCESS: PatchFrameInterpolationTrainer created successfully")
        print(f"Patch网络参数: {sum(p.numel() for p in trainer.student_model.parameters()):,}")
        
        # 测试前向传播
        test_input = torch.randn(1, 7, 270, 480)
        with torch.no_grad():
            output = trainer(test_input)
        print(f"前向传播测试: {test_input.shape} -> {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_patch_training_framework()
    print(f"\n{'SUCCESS: Patch training framework test passed!' if success else 'ERROR: Patch training framework test failed!'}")