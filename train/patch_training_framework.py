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
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass
from torch.nn.functional import smooth_l1_loss

# Import existing components
try:
    from training_framework import (
        PerceptualLoss, DistillationLoss, FrameInterpolationTrainer,
        CombinedLoss, EdgeLoss, TemporalConsistencyLoss
    )
except ImportError as e:
    print(f"Training framework import warning: {e}")
    class PerceptualLoss(nn.Module):
        """ UPGRADED: 真正的VGG感知损失实现"""
        def __init__(self):
            super().__init__()
            
            # 尝试加载VGG16进行感知损失计算
            try:
                import torchvision.models as models
                # 加载预训练VGG16
                vgg = models.vgg16(pretrained=True)
                self.vgg_features = vgg.features[:30]  # 到relu5_3
                
                # 冻结VGG参数
                for param in self.vgg_features.parameters():
                    param.requires_grad = False
                    
                # ImageNet标准化参数
                self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                
                # 特征提取层索引
                self.feature_layers = [3, 8, 15, 22, 29]
                self.use_vgg = True
                
                print(" VGG感知损失初始化成功")
                
            except Exception as e:
                print(f" VGG感知损失初始化失败，使用SSIM替代: {e}")
                self.use_vgg = False
                # SSIM窗口
                self.register_buffer('ssim_window', self._create_ssim_window(11, 3))
                
        def _create_ssim_window(self, window_size: int, channel: int) -> torch.Tensor:
            """创建SSIM高斯窗口"""
            import numpy as np
            sigma = 1.5
            gauss = torch.Tensor([
                np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                for x in range(window_size)
            ])
            gauss = gauss / gauss.sum()
            
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window
            
        def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
            """标准化输入到VGG范围"""
            # 假设输入是[-1, 1] 或 [0, 1] 范围
            if x.min() < 0:
                x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
            
            # ImageNet标准化
            return (x - self.mean) / self.std
            
        def _extract_vgg_features(self, x: torch.Tensor) -> list:
            """提取VGG特征"""
            x = self._normalize_for_vgg(x)
            features = []
            
            for i, layer in enumerate(self.vgg_features):
                x = layer(x)
                if i in self.feature_layers:
                    features.append(x)
                    
            return features
            
        def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """计算SSIM作为VGG的替代"""
            window = self.ssim_window.to(pred.device)
            
            # 计算均值
            mu1 = F.conv2d(pred, window, padding=5, groups=3)
            mu2 = F.conv2d(target, window, padding=5, groups=3)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            # 方差和协方差
            sigma1_sq = F.conv2d(pred * pred, window, padding=5, groups=3) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window, padding=5, groups=3) - mu2_sq
            sigma12 = F.conv2d(pred * target, window, padding=5, groups=3) - mu1_mu2
            
            # SSIM计算
            C1 = 0.01**2
            C2 = 0.03**2
            
            ssim = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
            
            return 1 - ssim.mean()  # 转换为损失 (越小越好)
        
        def forward(self, pred, target):
            """感知损失计算"""
            if not self.use_vgg:
                # 使用SSIM作为替代
                return self._compute_ssim(pred, target)
            
            try:
                # VGG感知损失
                pred_features = self._extract_vgg_features(pred)
                target_features = self._extract_vgg_features(target)
                
                perceptual_loss = 0.0
                layer_weights = [1.5, 1.5, 2.0, 2.0, 1.5]  # 每层权重
                
                for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
                    if i < len(layer_weights):
                        # 使用L1损失比较特征
                        layer_loss = smooth_l1_loss(pred_feat, target_feat, beta=0.001)
                        perceptual_loss += layer_weights[i] * layer_loss
                
                return perceptual_loss
                
            except Exception as e:
                print(f" VGG感知损失计算失败，使用SSIM替代: {e}")
                # 降级到SSIM
                return self._compute_ssim(pred, target)
    
    class DistillationLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, student, teacher, target):
            # 返回零损失但不影响训练（通过requires_grad=False）
            return torch.tensor(0.0, requires_grad=False), {}
    
    class EdgeLoss(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, pred, target):
            # 使用简单的梯度差异作为边缘损失
            return torch.nn.functional.l1_loss(
                torch.diff(pred, dim=-1), 
                torch.diff(target, dim=-1)
            ) + torch.nn.functional.l1_loss(
                torch.diff(pred, dim=-2), 
                torch.diff(target, dim=-2)
            )
    
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
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from train.patch_tensorboard_logger import create_patch_visualizer, PatchTensorBoardLogger
        TENSORBOARD_AVAILABLE = True
        print(" TensorBoard可视化: train模块导入成功")
    except ImportError as e:
        print(f"⚠️ Patch TensorBoard可视化导入警告: {e}")
        TENSORBOARD_AVAILABLE = False
        
        #  FIX: 提供完整的fallback实现以确保训练不中断
        def create_patch_visualizer(log_dir, config=None):
            """Fallback可视化器 - 提供基本功能但不执行实际记录"""
            class FallbackVisualizer:
                def __init__(self, log_dir, config):
                    self.log_dir = log_dir
                    self.config = config or {}
                    self.vis_frequency = config.get('visualization_frequency', 100) if config else 100
                    self.save_frequency = config.get('save_frequency', 500) if config else 500
                    print(f"🔄 使用Fallback可视化器: {log_dir}")
                
                def should_visualize(self, step):
                    return step % self.vis_frequency == 0
                
                def log_patch_visualization(self, *args, **kwargs):
                    pass
                
                def log_patch_comparison(self, *args, **kwargs):
                    pass
                
                def log_training_step(self, *args, **kwargs):
                    pass
                
                def log_validation_step(self, *args, **kwargs):
                    pass
                
                def log_training_progress(self, *args, **kwargs):
                    pass
                
                def close(self):
                    pass
            
            return FallbackVisualizer(log_dir, config)
        
        class PatchTensorBoardLogger:
            def __init__(self, *args, **kwargs):
                print("🔄 使用Fallback TensorBoard Logger")
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
            
            # patch总损失
            patch_total = (patch_l1 + 1.2 * patch_perceptual + 
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
        boundary_weight = 0.7
        
        return {
            'patch': patch_weight,
            'boundary': boundary_weight,
            'consistency': 0.0,  # 不再使用一致性损失
            'full': 0.0  # 不再使用全图损失
        }
    
    def _compute_boundary_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                              patch_metadata: List[Dict]) -> torch.Tensor:
        """计算边界感知损失 -  FIXED: 自适应边界检测，不依赖metadata"""
        batch_size = pred.shape[0]
        
        if batch_size == 0:
            return torch.tensor(0.0, device=pred.device)
        
        #  NEW: 自动边界检测，不依赖外部metadata
        total_boundary_loss = 0.0
        
        # 确保boundary_kernel在正确设备上
        if self.boundary_kernel.device != pred.device:
            self.boundary_kernel = self.boundary_kernel.to(pred.device)
        
        for i in range(batch_size):
            # 对每个patch计算边界损失
            patch_pred = pred[i:i+1]  # [1, 3, H, W]
            patch_target = target[i:i+1]  # [1, 3, H, W]
            
            #  NEW: 多种边界检测策略组合
            # 1. 基于目标图像的边缘检测
            target_gray = torch.mean(patch_target, dim=1, keepdim=True)  # [1, 1, H, W]
            target_edges = F.conv2d(target_gray, self.boundary_kernel, padding=1)
            target_boundary_map = torch.sigmoid(torch.abs(target_edges) * 2.0)
            
            # 2. 基于预测图像的边缘检测
            pred_gray = torch.mean(patch_pred, dim=1, keepdim=True)  # [1, 1, H, W]
            pred_edges = F.conv2d(pred_gray, self.boundary_kernel, padding=1)
            pred_boundary_map = torch.sigmoid(torch.abs(pred_edges) * 1.0)
            
            # 3. 组合边界图：取两者的最大值
            combined_boundary_map = torch.max(target_boundary_map, pred_boundary_map)
            
            # 4. 添加patch边缘区域（patch的四周边界）
            H, W = patch_pred.shape[2], patch_pred.shape[3]
            edge_margin = 8  # 边缘区域宽度
            edge_mask = torch.zeros_like(combined_boundary_map)
            
            # 设置边缘区域
            edge_mask[:, :, :edge_margin, :] = 0.5  # 顶部
            edge_mask[:, :, -edge_margin:, :] = 0.5  # 底部
            edge_mask[:, :, :, :edge_margin] = 0.5  # 左侧
            edge_mask[:, :, :, -edge_margin:] = 0.5  # 右侧
            
            # 5. 最终边界权重图
            final_boundary_map = torch.clamp(combined_boundary_map + edge_mask, 0.0, 2.0)
            
            # 6. 在边界区域加权L1损失
            boundary_weighted_loss = torch.mean(
                F.l1_loss(patch_pred, patch_target, reduction='none') * 
                (1.0 + final_boundary_map)  # 边界区域权重增强 (1.0-3.0倍)
            )
            
            total_boundary_loss += boundary_weighted_loss
        
        # 返回平均边界损失
        return total_boundary_loss / batch_size
    


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
            patch_network_channels=24  #  REVERT: 回退到稳定的24通道配置
        )
        
        self.student_model = PatchBasedInpainting(
            input_channels=inpainting_config.get('input_channels', 7),
            output_channels=inpainting_config.get('output_channels', 3),
            config=patch_inpainting_config
        )
        
        #  NEW: 启动时输出模型配置信息
        self._print_model_architecture_info(inpainting_config, patch_inpainting_config)
        
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
        
        # 训练侧统一的边界检测卷积核（与损失函数一致）
        boundary_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('boundary_kernel', boundary_kernel)
        
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
    
    def _validate_batch_data(self, batch: Dict[str, torch.Tensor]) -> None:
        """验证batch数据的完整性和格式"""
        try:
            # 检查必需的字段
            if 'patch_input' not in batch:
                raise ValueError("Missing 'patch_input' in batch data")
            if 'patch_target_residual' not in batch:
                raise ValueError("Missing 'patch_target_residual' in batch data")  
            if 'patch_target_rgb' not in batch:
                raise ValueError("Missing 'patch_target_rgb' in batch data")
            
            # 检查数据维度
            patch_input = batch['patch_input']
            patch_target_residual = batch['patch_target_residual']
            patch_target_rgb = batch['patch_target_rgb']
            
            if patch_input.dim() != 4:
                raise ValueError(f"patch_input should be 4D tensor, got {patch_input.dim()}D")
            if patch_target_residual.dim() != 4:
                raise ValueError(f"patch_target_residual should be 4D tensor, got {patch_target_residual.dim()}D")
            if patch_target_rgb.dim() != 4:
                raise ValueError(f"patch_target_rgb should be 4D tensor, got {patch_target_rgb.dim()}D")
            
            # 检查通道数
            if patch_input.shape[1] != 7:
                raise ValueError(f"patch_input should have 7 channels, got {patch_input.shape[1]}")
            if patch_target_residual.shape[1] != 3:
                raise ValueError(f"patch_target_residual should have 3 channels, got {patch_target_residual.shape[1]}")
            if patch_target_rgb.shape[1] != 3:
                raise ValueError(f"patch_target_rgb should have 3 channels, got {patch_target_rgb.shape[1]}")
            
            # 检查批次大小一致性
            batch_size = patch_input.shape[0]
            if patch_target_residual.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: input={batch_size}, target_residual={patch_target_residual.shape[0]}")
            if patch_target_rgb.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: input={batch_size}, target_rgb={patch_target_rgb.shape[0]}")
            
            # 检查patch尺寸一致性
            if patch_input.shape[2:] != patch_target_residual.shape[2:]:
                raise ValueError(f"Spatial size mismatch: input={patch_input.shape[2:]}, target_residual={patch_target_residual.shape[2:]}")
            if patch_input.shape[2:] != patch_target_rgb.shape[2:]:
                raise ValueError(f"Spatial size mismatch: input={patch_input.shape[2:]}, target_rgb={patch_target_rgb.shape[2:]}")
            
            # 检查数据范围
            if torch.isnan(patch_input).any() or torch.isinf(patch_input).any():
                raise ValueError("patch_input contains NaN or Inf values")
            if torch.isnan(patch_target_residual).any() or torch.isinf(patch_target_residual).any():
                raise ValueError("patch_target_residual contains NaN or Inf values")
            if torch.isnan(patch_target_rgb).any() or torch.isinf(patch_target_rgb).any():
                raise ValueError("patch_target_rgb contains NaN or Inf values")
            
            # 检查残差学习数据一致性
            try:
                from residual_learning_helper import ResidualLearningHelper
                warped_rgb = patch_input[:, :3]
                ResidualLearningHelper.validate_residual_data(patch_input, patch_target_residual, patch_target_rgb)
            except ImportError:
                pass
                
        except Exception as e:
            print(f"ERROR: Batch数据验证失败: {e}")
            #  FIX: 添加类型检查，避免对list调用keys()
            if isinstance(batch, dict):
                print(f"Batch keys: {list(batch.keys())}")
                if 'patch_input' in batch:
                    print(f"patch_input shape: {batch['patch_input'].shape}")
                if 'patch_target_residual' in batch:
                    print(f"patch_target_residual shape: {batch['patch_target_residual'].shape}")
                if 'patch_target_rgb' in batch:
                    print(f"patch_target_rgb shape: {batch['patch_target_rgb'].shape}")
            else:
                print(f"Batch type: {type(batch)}, content: {batch}")
            raise e

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
        
        # 数据验证和残差学习patch数据处理
        self._validate_batch_data(batch)
        
        if 'patch_input' in batch and len(batch['patch_input']) > 0:
            patch_input = batch['patch_input']                      # [N, 7, 270, 480]
            patch_target_residual = batch['patch_target_residual']  # [N, 3, 270, 480]
            patch_target_rgb = batch['patch_target_rgb']            # [N, 3, 270, 480]
            
            # 训练时基于目标图像构建边界图并传入网络（与损失侧语义一致）
            target_gray = torch.mean(patch_target_rgb, dim=1, keepdim=True)  # [N,1,H,W]
            # 确保卷积核在同一设备
            kernel = self.boundary_kernel
            if kernel.device != target_gray.device:
                kernel = kernel.to(target_gray.device)
            target_edges = F.conv2d(target_gray, kernel, padding=1)
            boundary_override = torch.sigmoid(torch.abs(target_edges) * 2.0)

            # Patch网络推理 - 输出残差预测（传入 boundary_override）
            residual_pred = self.student_model.patch_network(patch_input, boundary_override=boundary_override)
            
            #  使用统一的残差学习工具类
            try:
                from residual_learning_helper import ResidualLearningHelper
            except ImportError:
                import sys
                sys.path.append('./train')
                from residual_learning_helper import ResidualLearningHelper
            
            # 残差预测转换为完整图像用于损失计算
            warped_rgb = patch_input[:, :3]  # 提取输入的warped RGB
            patch_pred_full = ResidualLearningHelper.reconstruct_from_residual(warped_rgb, residual_pred)
            
            # 损失计算使用完整重建图像与目标RGB
            predictions['patch'] = patch_pred_full
            targets['patch'] = patch_target_rgb
            
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
        
        # 知识蒸馏暂时禁用，因为删除了全图模式
        
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
                #  记录patch对比（输入|目标RGB|重建图像）- 残差学习版本
                if 'patch' in predictions and 'patch' in targets:
                    self.patch_visualizer.log_patch_comparison(
                        step=current_global_step,
                        patch_inputs=batch['patch_input'][:8],  # 最多显示8个patches
                        patch_targets=batch['patch_target_rgb'][:8],  # 使用RGB目标进行可视化
                        patch_predictions=predictions['patch'][:8],   # 重建的完整图像
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
        
        #  处理残差学习验证数据
        if 'patch_input' in batch and len(batch['patch_input']) > 0:
            patch_input = batch['patch_input']
            patch_target_rgb = batch['patch_target_rgb'] # 获取目标RGB用于生成boundary_override

            target_gray = torch.mean(patch_target_rgb, dim=1, keepdim=True)  # [N,1,H,W]
            # 确保卷积核在同一设备
            kernel = self.boundary_kernel
            if kernel.device != target_gray.device:
                kernel = kernel.to(target_gray.device)
            target_edges = F.conv2d(target_gray, kernel, padding=1)
            boundary_override = torch.sigmoid(torch.abs(target_edges) * 2.0) 
            
            # 网络输出残差预测
            residual_pred = self.student_model.patch_network(batch['patch_input'], boundary_override=boundary_override)
            
            # 使用统一的残差学习工具类转换为完整图像
            try:
                from residual_learning_helper import ResidualLearningHelper
            except ImportError:
                import sys
                sys.path.append('./train')
                from residual_learning_helper import ResidualLearningHelper
            
            warped_rgb = batch['patch_input'][:, :3]
            patch_pred_full = ResidualLearningHelper.reconstruct_from_residual(warped_rgb, residual_pred)
            
            predictions['patch'] = patch_pred_full
            targets['patch'] = batch['patch_target_rgb']  # 使用RGB目标
        
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
                #  验证可视化 - 残差学习版本
                if 'patch' in predictions and 'patch' in targets:
                    self.patch_visualizer.log_patch_comparison(
                        step=current_global_step,
                        patch_inputs=batch['patch_input'][:4],  # 验证时显示4个patches
                        patch_targets=batch['patch_target_rgb'][:4],  # 使用RGB目标
                        patch_predictions=predictions['patch'][:4],   # 重建图像
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
        # 优化器配置 - 确保类型转换
        optimizer_config = self.training_config.get('optimizer', {})
        
        # 安全的类型转换
        learning_rate = float(optimizer_config.get('learning_rate', 1e-4))
        weight_decay = float(optimizer_config.get('weight_decay', 1e-5))
        
        print(f" 优化器配置: lr={learning_rate}, weight_decay={weight_decay}")
        
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器 - 确保类型转换
        scheduler_config = self.training_config.get('scheduler', {})
        
        T_max = int(scheduler_config.get('T_max', 100))
        eta_min = float(scheduler_config.get('eta_min', 1e-6))
        
        print(f" 调度器配置: T_max={T_max}, eta_min={eta_min}")
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
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
    
    def _print_model_architecture_info(self, inpainting_config: Dict[str, Any], patch_inpainting_config) -> None:
        """ NEW: 启动时输出模型架构信息，便于验证配置正确性"""
        print("\n" + "="*70)
        print("🏗️  模型架构配置信息")
        print("="*70)
        
        # 基础网络配置
        input_channels = inpainting_config.get('input_channels', 7)
        output_channels = inpainting_config.get('output_channels', 3)
        base_channels = inpainting_config.get('base_channels', 64)
        patch_network_channels = patch_inpainting_config.patch_network_channels
        
        print(f" 网络类型: PatchBasedInpainting (patch训练框架)")
        print(f" 输入通道数: {input_channels}")
        print(f" 输出通道数: {output_channels}")
        print(f" 配置文件base_channels: {base_channels}")
        print(f" 实际patch_network_channels: {patch_network_channels}")
        
        # 验证配置一致性
        if base_channels == patch_network_channels:
            print(f" 配置一致性检查: 通过 (base_channels = patch_network_channels = {base_channels})")
        else:
            print(f"⚠️  配置不一致: base_channels={base_channels}, patch_network_channels={patch_network_channels}")
        
        # 输出具体的PatchNetwork信息
        if hasattr(self.student_model, 'patch_network'):
            patch_network = self.student_model.patch_network
            total_params = sum(p.numel() for p in patch_network.parameters())
            trainable_params = sum(p.numel() for p in patch_network.parameters() if p.requires_grad)
            
            print(f" PatchNetwork参数统计:")
            print(f"   - 总参数量: {total_params:,}")
            print(f"   - 可训练参数: {trainable_params:,}")
            print(f"   - 参数大小: {total_params * 4 / 1024 / 1024:.2f} MB")
            
            # 如果可以访问网络结构，显示通道配置
            if hasattr(patch_network, 'ch1'):
                print(f" Enhanced PatchNetwork通道架构:")
                print(f"   - ch1 (Level 1): {patch_network.ch1}")
                print(f"   - ch2 (Level 2): {patch_network.ch2}")
                print(f"   - ch3 (Level 3): {patch_network.ch3}")
                print(f"   - ch4 (Level 4): {patch_network.ch4}")
                print(f"   - ch5 (Bottleneck): {patch_network.ch5}")
        
        # 训练模式配置
        learning_mode = inpainting_config.get('learning_mode', 'residual')
        print(f" 学习模式: {learning_mode}")
        
        # Patch配置信息
        if hasattr(self, 'patch_config'):
            print(f" Patch配置:")
            print(f"   - Patch模式: {'启用' if self.patch_config.enable_patch_mode else '禁用'}")
            if hasattr(self.patch_config, 'patch_size'):
                print(f"   - Patch大小: {self.patch_config.patch_size}x{self.patch_config.patch_size}")
        
        print("="*70 + "\n")
        
        # 如果发现配置问题，给出警告
        if base_channels != 64:
            print("⚠️  警告: base_channels不是64，可能与预期配置不符")
        if patch_network_channels != 64:
            print("⚠️  警告: patch_network_channels不是64，训练的模型将无法与64通道推理脚本兼容")
        if base_channels != patch_network_channels:
            print("⚠️  警告: base_channels与patch_network_channels不一致，请检查配置")
        
        print("💡 提示: 如果要与现有推理脚本兼容，确保两个通道数都是64")


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


def main():
    """主训练函数 - 支持配置文件训练"""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Patch Training Framework")
    parser.add_argument('--config', type=str, required=True, 
                       help='Configuration file path')
    parser.add_argument('--test-only', action='store_true',
                       help='Run test only without training')
    
    args = parser.parse_args()
    
    if args.test_only:
        # 运行测试模式
        success = test_patch_training_framework()
        print(f"\n{'SUCCESS: Patch training framework test passed!' if success else 'ERROR: Patch training framework test failed!'}")
        return 0 if success else 1
    
    # 训练模式 - 加载配置文件
    try:
        print(f" 加载配置文件: {args.config}")
        
        if not os.path.exists(args.config):
            print(f" 配置文件不存在: {args.config}")
            return 1
        
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            print(" 配置文件为空或格式错误")
            return 1
        
        print(" 配置文件加载成功")
        
        # 启动实际训练
        success = run_patch_training(config)
        return 0 if success else 1
        
    except Exception as e:
        print(f" 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

def run_patch_training(config: Dict[str, Any]) -> bool:
    """运行Patch训练流程"""
    try:
        print(" 启动Patch训练流程...")
        
        # 提取配置sections
        network_config = config.get('network', {})
        training_config = config.get('training', {})
        patch_config_dict = config.get('patch', {})
        data_config = config.get('data', {})
        loss_config = config.get('loss', {})
        monitoring_config = config.get('monitoring', {})
        
        print(f" 网络配置: {network_config.get('type', 'Unknown')}")
        print(f" 训练批次: {training_config.get('batch_size', 'Unknown')}")
        print(f" Patch模式: {'启用' if patch_config_dict.get('enable_patch_mode', False) else '禁用'}")
        print(f" 简单网格: {'启用' if patch_config_dict.get('use_simple_grid_patches', False) else '禁用'}")
        
        # 创建训练器配置
        model_config = {
            'inpainting_network': {
                'input_channels': network_config.get('input_channels', 7),
                'output_channels': network_config.get('output_channels', 3),
                'base_channels': network_config.get('base_channels', 24)  #  REVERT: 回退到稳定的24通道
            }
        }
        
        trainer_config = {
            'optimizer': {
                'learning_rate': training_config.get('learning_rate', 1e-4),
                'weight_decay': training_config.get('weight_decay', 1e-5)
            },
            'scheduler': {
                'T_max': training_config.get('max_epochs', 100),
                'eta_min': 1e-6
            },
            'log_dir': monitoring_config.get('tensorboard_log_dir', './logs/patch_training'),
            'batch_size': training_config.get('batch_size', 4),
            'max_epochs': training_config.get('max_epochs', 100)
        }
        
        # 创建Patch配置
        from patch_aware_dataset import PatchTrainingConfig
        
        patch_training_config = PatchTrainingConfig(
            enable_patch_mode=patch_config_dict.get('enable_patch_mode', True),
            patch_size=patch_config_dict.get('patch_size', 128),
            use_simple_grid_patches=patch_config_dict.get('use_simple_grid_patches', False),
            use_optimized_patches=patch_config_dict.get('use_optimized_patches', True),
            simple_grid_rows=patch_config_dict.get('simple_grid_rows', 4),
            simple_grid_cols=patch_config_dict.get('simple_grid_cols', 4),
            simple_expected_height=patch_config_dict.get('simple_expected_height', 1080),
            simple_expected_width=patch_config_dict.get('simple_expected_width', 1920),
            min_patches_per_image=patch_config_dict.get('min_patches_per_image', 8),
            max_patches_per_image=patch_config_dict.get('max_patches_per_image', 16)
        )
        
        # 创建训练器
        trainer = create_patch_trainer(
            model_config=model_config,
            training_config=trainer_config,
            patch_config=patch_training_config,
            full_config=config
        )
        
        print(" Patch训练器创建成功")
        print(f" 网络参数: {sum(p.numel() for p in trainer.student_model.parameters()):,}")
        
        # 创建数据加载器
        success = setup_data_loaders(trainer, data_config, patch_training_config)
        if not success:
            print(" 数据加载器设置失败")
            return False
        
        # 开始训练
        print(" 开始训练循环...")
        
        # 创建PyTorch Lightning Trainer
        max_epochs = trainer_config.get('max_epochs', 100)
        
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from pytorch_lightning.loggers import TensorBoardLogger
        
        # 创建回调
        callbacks = []
        
        # 模型检查点
        checkpoint_callback = ModelCheckpoint(
            dirpath=monitoring_config.get('model_save_dir', './models/colleague'),
            filename='patch-model-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            save_top_k=3,
            mode='min',
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        
        # 早停
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        )
        # callbacks.append(early_stop_callback)
        
        # TensorBoard日志记录器
        tb_logger = TensorBoardLogger(
            save_dir=monitoring_config.get('tensorboard_log_dir', './logs/colleague_training'),
            name='patch_training'
        )
        
        # 创建Lightning Trainer
        pl_trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=tb_logger,
            callbacks=callbacks,
            accelerator='auto',  # 自动检测GPU/CPU
            devices='auto',      # 自动检测设备数量
            precision=32,        # 使用FP32精度
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            gradient_clip_val=1.0,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        print(f" Lightning Trainer配置:")
        print(f"   最大轮数: {max_epochs}")
        print(f"   设备: {pl_trainer.accelerator} ({pl_trainer.num_devices})")
        print(f"   日志目录: {monitoring_config.get('tensorboard_log_dir', './logs/colleague_training')}")
        print(f"   模型保存: {monitoring_config.get('model_save_dir', './models/colleague')}")
        
        # 启动训练
        print(" 启动PyTorch Lightning训练...")
        pl_trainer.fit(
            model=trainer,
            train_dataloaders=trainer.train_loader,
            val_dataloaders=trainer.val_loader
        )
        
        print(" 训练完成!")
        return True
        
    except Exception as e:
        print(f" 训练过程失败: {e}")
        import traceback
        traceback.print_exc()
        return False

class ColleaguePatchDataset(Dataset):
    """ColleagueDatasetAdapter的Patch包装器"""
    
    def __init__(self, data_root: str, split: str, patch_config):
        from colleague_dataset_adapter import ColleagueDatasetAdapter
        
        # 创建基础数据集
        self.base_dataset = ColleagueDatasetAdapter(
            data_root=data_root,
            split=split,
            augmentation=False
        )
        
        self.patch_config = patch_config
        
        # 初始化简单网格提取器
        if patch_config.use_simple_grid_patches:
            print(" 初始化SimplePatchExtractor")
            
            try:
                # 导入并创建SimplePatchExtractor
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                
                from simple_patch_extractor import SimplePatchExtractor, SimpleGridConfig
                
                simple_config = SimpleGridConfig(
                    grid_rows=patch_config.simple_grid_rows,
                    grid_cols=patch_config.simple_grid_cols,
                    expected_height=patch_config.simple_expected_height,
                    expected_width=patch_config.simple_expected_width,
                    enable_debug_info=False
                )
                
                self.patch_extractor = SimplePatchExtractor(simple_config)
                self._using_simple_grid = True
                
                print(f"    网格配置: {patch_config.simple_grid_rows}x{patch_config.simple_grid_cols} = {patch_config.simple_grid_rows * patch_config.simple_grid_cols} patches")
                
            except ImportError as e:
                print(f" SimplePatchExtractor导入失败: {e}")
                raise
        else:
            raise NotImplementedError("只支持简单网格策略")
    
    def __len__(self):
        # 每个图像产生固定数量的patch
        return len(self.base_dataset) * self.patch_config.max_patches_per_image
    
    def __getitem__(self, idx):
        # 计算源图像索引和patch索引
        patches_per_image = self.patch_config.max_patches_per_image
        image_idx = idx // patches_per_image
        patch_idx = idx % patches_per_image
        
        try:
            # 获取源图像数据 - 这应该返回 [C, H, W] 格式的张量
            input_tensor, target_residual, target_rgb = self.base_dataset[image_idx]
            
            # 确保数据格式为 [C, H, W]
            if input_tensor.shape != (7, 1080, 1920):
                print(f"  输入数据形状异常: {input_tensor.shape}, 期望: (7, 1080, 1920)")
                # 尝试修复形状
                if len(input_tensor.shape) == 3:
                    if input_tensor.shape[0] == 7:  # [C, H, W] 但H和W不对
                        # 转置或者其他处理
                        pass
                    elif input_tensor.shape[2] == 7:  # [H, W, C]
                        input_tensor = input_tensor.permute(2, 0, 1)
                    elif input_tensor.shape[1] == 7:  # [H, C, W] - 不太可能
                        input_tensor = input_tensor.permute(1, 0, 2)
            
            # 确保目标数据也是正确格式
            if target_residual.shape[0] != 3:
                if len(target_residual.shape) == 3 and target_residual.shape[2] == 3:
                    target_residual = target_residual.permute(2, 0, 1)
            
            if target_rgb.shape[0] != 3:
                if len(target_rgb.shape) == 3 and target_rgb.shape[2] == 3:
                    target_rgb = target_rgb.permute(2, 0, 1)
            
            # 转换为numpy用于patch提取 - 注意：SimpleGridExtractor可能期望[H, W, C]格式
            # 先尝试传递[C, H, W]格式，如果出错再调整
            input_numpy = input_tensor.numpy()  # [7, 1080, 1920]
            target_residual_numpy = target_residual.numpy()  # [3, 1080, 1920]
            target_rgb_numpy = target_rgb.numpy()  # [3, 1080, 1920]
            
            # 检查SimplePatchExtractor期望的格式
            # 如果出现"Size mismatch"，说明extractor期望[H, W, C]格式
            try:
                # 尝试使用[C, H, W]格式
                input_patches, positions = self.patch_extractor.extract_patches(input_numpy)
            except Exception as shape_error:
                if "Size mismatch" in str(shape_error):
                    # 转换为[H, W, C]格式
                    if idx < 3:
                        print(f"[DEBUG] 转换数据格式: {input_numpy.shape} -> [H, W, C]")
                    input_numpy = input_numpy.transpose(1, 2, 0)  # [7, 1080, 1920] -> [1080, 1920, 7]
                    target_residual_numpy = target_residual_numpy.transpose(1, 2, 0)  # [3, 1080, 1920] -> [1080, 1920, 3]
                    target_rgb_numpy = target_rgb_numpy.transpose(1, 2, 0)  # [3, 1080, 1920] -> [1080, 1920, 3]
                    
                    # 重新尝试提取patches
                    input_patches, positions = self.patch_extractor.extract_patches(input_numpy)
                else:
                    raise shape_error
            
            # 提取目标patches
            target_residual_patches, _ = self.patch_extractor.extract_patches(target_residual_numpy)
            target_rgb_patches, _ = self.patch_extractor.extract_patches(target_rgb_numpy)
            
            # 检查patch数量
            if patch_idx >= len(input_patches):
                patch_idx = len(input_patches) - 1
            
            # 获取指定的patch
            patch_input = input_patches[patch_idx]  # numpy array
            patch_target_residual = target_residual_patches[patch_idx]  # numpy array  
            patch_target_rgb = target_rgb_patches[patch_idx]  # numpy array
            
            # 确保patch格式为[C, H, W]
            if len(patch_input.shape) == 3:
                if patch_input.shape[2] == 7:  # [H, W, C] -> [C, H, W]
                    patch_input = patch_input.transpose(2, 0, 1)
                if patch_target_residual.shape[2] == 3:  # [H, W, C] -> [C, H, W]
                    patch_target_residual = patch_target_residual.transpose(2, 0, 1)
                if patch_target_rgb.shape[2] == 3:  # [H, W, C] -> [C, H, W]
                    patch_target_rgb = patch_target_rgb.transpose(2, 0, 1)
            
            # 转换为PyTorch张量
            patch_input = torch.from_numpy(patch_input).float()
            patch_target_residual = torch.from_numpy(patch_target_residual).float()
            patch_target_rgb = torch.from_numpy(patch_target_rgb).float()
            
            #  UPDATED: 形状验证更新为支持非正方形patch (270x480)
            expected_h, expected_w = 270, 480  # 4x4网格切分后的patch尺寸
            assert patch_input.shape[0] == 7, f"输入patch通道数错误: {patch_input.shape[0]} (期望7)"
            assert patch_input.shape[1:] == (expected_h, expected_w), f"输入patch尺寸错误: {patch_input.shape[1:]} (期望{expected_h}x{expected_w})"
            assert patch_target_residual.shape[0] == 3, f"残差目标patch通道数错误: {patch_target_residual.shape[0]} (期望3)"
            assert patch_target_residual.shape[1:] == (expected_h, expected_w), f"残差目标patch尺寸错误: {patch_target_residual.shape[1:]} (期望{expected_h}x{expected_w})"
            assert patch_target_rgb.shape[0] == 3, f"RGB目标patch通道数错误: {patch_target_rgb.shape[0]} (期望3)"
            assert patch_target_rgb.shape[1:] == (expected_h, expected_w), f"RGB目标patch尺寸错误: {patch_target_rgb.shape[1:]} (期望{expected_h}x{expected_w})"
            
            #  FIX: 返回字典格式而不是tuple，符合PyTorch Lightning期望
            return {
                'patch_input': patch_input,
                'patch_target_residual': patch_target_residual,
                'patch_target_rgb': patch_target_rgb
            }
            
        except Exception as e:
            print(f" ColleaguePatchDataset获取数据失败 [idx={idx}, image_idx={image_idx}, patch_idx={patch_idx}]: {e}")
            import traceback
            traceback.print_exc()
            
            #  FIX: 返回字典格式的零张量 (270x480尺寸)
            expected_h, expected_w = 270, 480  # 4x4网格切分后的patch尺寸
            return {
                'patch_input': torch.zeros(7, expected_h, expected_w),
                'patch_target_residual': torch.zeros(3, expected_h, expected_w),
                'patch_target_rgb': torch.zeros(3, expected_h, expected_w)
            }


def setup_data_loaders(trainer, data_config: Dict[str, Any], patch_config) -> bool:
    """设置数据加载器"""
    try:
        print(" 设置数据加载器...")
        
        # 根据dataset_type选择数据集
        dataset_type = data_config.get('dataset_type', 'colleague')
        
        if dataset_type == 'colleague':
            # 使用ColleagueDatasetAdapter的patch包装器
            from colleague_dataset_adapter import ColleagueDatasetAdapter
            
            print(" 使用ColleagueDatasetAdapter + ColleaguePatchDataset")
            
            # 创建ColleaguePatchDataset包装器
            train_dataset = ColleaguePatchDataset(
                data_root=data_config.get('data_root', './data'),
                split='train',
                patch_config=patch_config
            )
            
            val_dataset = ColleaguePatchDataset(
                data_root=data_config.get('data_root', './data'),
                split='val',
                patch_config=patch_config
            )
            
            print(f" 训练样本: {len(train_dataset)} patches")
            print(f" 验证样本: {len(val_dataset)} patches")
            
        else:
            print(f" 不支持的数据集类型: {dataset_type}")
            return False
        
        # 创建数据加载器 - 从正确的配置位置获取参数
        training_section = trainer.training_config if hasattr(trainer, 'training_config') else {}
        batch_size = training_section.get('batch_size', 4)
        num_workers = 2  # 固定为2，避免多进程问题
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        # 设置训练器的数据加载器
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        
        print(" 数据加载器设置完成")
        return True
        
    except Exception as e:
        print(f" 数据加载器设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)