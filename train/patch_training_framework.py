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
import os
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass
from torch.nn.functional import smooth_l1_loss

# Import existing components
try:
    from training_framework import (
        DistillationLoss, FrameInterpolationTrainer,
        CombinedLoss, TemporalConsistencyLoss
    )
except ImportError as e:
    # Silent fallback by default; enable env TRAINING_FRAMEWORK_IMPORT_WARN=1 to see this message
    if os.environ.get('TRAINING_FRAMEWORK_IMPORT_WARN', '0') == '1':
        print(f"Training framework import warning: {e}")
    class PerceptualLoss(nn.Module):
        """ UPGRADED: 真正的VGG感知损失实现"""
        def __init__(self):
            super().__init__()
            
            # 尝试加载VGG16进行感知损失计算
            try:
                import torchvision.models as models
                try:
                    weights = models.VGG16_Weights.DEFAULT
                    vgg = models.vgg16(weights=weights)
                except Exception:
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
                    print(f" 使用Fallback可视化器: {log_dir}")
                
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
                print(" 使用Fallback TensorBoard Logger")
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
                 enable_adaptive_weights: bool = True,
                 robust_config: Optional[Dict[str, Any]] = None,
                 hdr_vis_config: Optional[Dict[str, Any]] = None,
                 weights_config: Optional[Dict[str, Any]] = None,
                 masking_config: Optional[Dict[str, Any]] = None):
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

        # 鲁棒损失配置（针对像素重建项）
        self.robust_cfg = robust_config or {}
        self.robust_enable = bool(self.robust_cfg.get('enable', False))
        self.robust_type = str(self.robust_cfg.get('type', 'huber')).lower()
        self.robust_delta = float(self.robust_cfg.get('params', {}).get('delta', 0.2))
        self.robust_epsilon = float(self.robust_cfg.get('params', {}).get('epsilon', 0.003))

        # VGG感知输入的tone-mapping设置（与显示一致，默认Reinhard+gamma）
        self.hdr_vis_cfg = hdr_vis_config or {}
        self.tone_mapping = str(self.hdr_vis_cfg.get('tone_mapping_for_display', 'reinhard')).lower()
        self.gamma = float(self.hdr_vis_cfg.get('gamma', 2.2))
        self.exposure = float(self.hdr_vis_cfg.get('exposure', 1.0))
        self.adaptive_exposure_cfg = self.hdr_vis_cfg.get('adaptive_exposure', {'enable': False})

        # Import tone-mapping utility
        try:
            from src.npu.utils.hdr_vis import tone_map as _tm
        except Exception:
            import sys, os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'npu', 'utils'))
            from hdr_vis import tone_map as _tm
        self._tone_map_impl = _tm

        # Loss weights from config (fallback defaults)
        wc = weights_config or {}
        self.w_l1 = float(wc.get('l1', 1.0))
        self.w_perc = float(wc.get('perceptual', 0.02))
        self.w_edge = float(wc.get('edge', 0.1))
        self.w_boundary = float(wc.get('boundary', 0.1))
        # 洞区加权项（默认0不启用）
        self.w_hole = float(wc.get('hole', 0.0))
        # 非洞“保持”项（默认较低）
        self.w_ctx = float(wc.get('ctx', 0.1))
        # 洞边环加权（强度与核大小）
        self.hole_ring_scale = float(wc.get('hole_ring_scale', 0.5))
        self.hole_ring_kernel = int(wc.get('hole_ring_kernel', 3))  # 应为奇数，>=3

        # LDR/HDR 计算域：恢复为固定策略（perceptual/boundary 在 LDR，pixel/edge 在 HDR）

        # 掩码化选项（只对洞区/洞边施加感知/边缘）
        mc = masking_config or {}
        self.mask_perceptual = bool(mc.get('mask_perceptual', False))
        self.mask_edge = bool(mc.get('mask_edge', False))
        # NEW: 边界损失掩码化（仅在洞区/洞边生效）
        self.mask_boundary = bool(mc.get('mask_boundary', False))

        # NEW: 洞面积补偿（当洞区很小时，补偿感知/边缘/边界项量级）
        ac = (weights_config or {}).get('area_compensation', {}) if isinstance(weights_config, dict) else {}
        self.area_comp_enable = bool(ac.get('enable', False))
        self.area_comp_min_frac = float(ac.get('min_frac', 0.05))
        self.area_comp_gamma = float(ac.get('gamma', 1.0))
        self.area_comp_max_scale = float(ac.get('max_scale', 4.0))
        self.area_comp_apply_to = set((ac.get('apply_to', ['perceptual', 'edge', 'boundary']) or []))

        # NEW: 洞内色彩统计一致性（均值/方差），默认关闭
        enh = (weights_config or {}).get('enhanced', {}) if isinstance(weights_config, dict) else {}
        cs = enh.get('color_stats', {}) if isinstance(enh, dict) else {}
        # 权重优先从 weights_config.color_stats 读取；否则落到 enhanced.color_stats.weight
        self.w_color_stats = float((weights_config or {}).get('color_stats', cs.get('weight', 0.0)))
        self.color_stats_enable = bool(cs.get('enable', False)) or (self.w_color_stats > 0.0)
        self.color_stats_domain = str(cs.get('domain', 'ldr')).lower()  # 'ldr' or 'hdr'
        self.color_stats_use_std = bool(cs.get('use_std', True))
        self.color_stats_eps = float(cs.get('epsilon', 1.0e-6))

        # NEW: 局部感知（patchs-in-a-patch）配置：在洞内网格子块上单独计算VGG感知
        lp = enh.get('local_perceptual', {}) if isinstance(enh, dict) else {}
        # 权重优先从 weights.local_perceptual 读取
        self.w_local_perc = float((weights_config or {}).get('local_perceptual', lp.get('weight', 0.0)))
        self.local_perc_enable = bool(lp.get('enable', False)) or (self.w_local_perc > 0.0)
        self.local_perc_grid_rows = int(lp.get('grid_rows', 2))
        self.local_perc_grid_cols = int(lp.get('grid_cols', 4))
        self.local_perc_hole_thresh = float(lp.get('hole_threshold_frac', 0.05))
        self.local_perc_padding = int(lp.get('padding', 8))  # 像素
        self.local_perc_max_crops = int(lp.get('max_crops_per_patch', 6))

    def _robust_masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """鲁棒/非鲁棒的掩码像素平均损失。

        Args:
            pred/target: [N,3,H,W]
            mask: [N,1,H,W] or [N,3,H,W] in [0,1]
        Returns:
            scalar loss averaged over masked pixels
        """
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.expand_as(pred)
        eps = 1e-6
        if self.robust_enable:
            if self.robust_type == 'huber':
                per = F.smooth_l1_loss(pred, target, beta=self.robust_delta, reduction='none')
            elif self.robust_type == 'charbonnier':
                diff = pred - target
                per = torch.sqrt(diff * diff + self.robust_epsilon * self.robust_epsilon)
            else:
                per = torch.abs(pred - target)
        else:
            per = torch.abs(pred - target)
        num = (per * mask).sum()
        den = mask.sum() + eps
        return num / den

    def _charbonnier(self, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return torch.sqrt(x * x + eps * eps)

    def _robust_reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unmasked robust reconstruction loss (scalar)."""
        if self.robust_enable:
            if self.robust_type == 'huber':
                return F.smooth_l1_loss(pred, target, beta=self.robust_delta)
            elif self.robust_type == 'charbonnier':
                diff = pred - target
                return torch.sqrt(diff * diff + self.robust_epsilon * self.robust_epsilon).mean()
            else:
                return self.l1_loss(pred, target)
        else:
            return self.l1_loss(pred, target)

    def _tone_map_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        # 线性HDR -> tone-mapped LDR [0,1]，调用通用工具（支持 mulaw）
        return self._tone_map_impl(
            x,
            method=self.tone_mapping,
            gamma=self.gamma,
            exposure=self.exposure,
            adaptive_exposure=self.adaptive_exposure_cfg,
        )
        
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

            # 预先构建洞区mask与洞边ring用于掩码化
            mask_weight = None
            if metadata and isinstance(metadata, dict) and 'holes_mask' in metadata:
                holes_mask = torch.clamp(metadata['holes_mask'], 0.0, 1.0)
                if holes_mask.shape[2:] != patch_pred.shape[2:]:
                    holes_mask = F.interpolate(holes_mask, size=patch_pred.shape[2:], mode='nearest')
                try:
                    k = max(3, int(self.hole_ring_kernel) if self.hole_ring_kernel % 2 == 1 else int(self.hole_ring_kernel) + 1)
                    pad = k // 2
                    ring = F.max_pool2d(holes_mask, kernel_size=k, stride=1, padding=pad) - holes_mask
                    ring = torch.clamp(ring, 0.0, 1.0)
                except Exception:
                    ring = torch.zeros_like(holes_mask)
                # 掩码化权重（0~1）：洞 + scale*ring
                mask_weight = torch.clamp(holes_mask + self.hole_ring_scale * ring, 0.0, 1.0)  # [N,1,H,W]

            # LDR 缓存（mu-law）
            ldr_pred = self._tone_map_for_vgg(patch_pred)
            ldr_tgt  = self._tone_map_for_vgg(patch_target)

            # NEW: 预计算 per-sample 面积放大系数向量（area_scale_vec），供各分项使用
            area_scale_vec = None
            if self.area_comp_enable and (metadata and isinstance(metadata, dict) and 'holes_mask' in metadata):
                holes_mask_ac = torch.clamp(metadata['holes_mask'], 0.0, 1.0)
                if holes_mask_ac.shape[2:] != patch_pred.shape[2:]:
                    holes_mask_ac = F.interpolate(holes_mask_ac, size=patch_pred.shape[2:], mode='nearest')
                hole_frac = holes_mask_ac.view(holes_mask_ac.size(0), -1).mean(dim=1)  # [N]
                min_frac = max(self.area_comp_min_frac, 1e-6)
                scale_raw = 1.0 / torch.clamp(hole_frac, min=min_frac)
                area_scale_vec = torch.clamp(scale_raw, min=1.0, max=self.area_comp_max_scale).pow(self.area_comp_gamma)  # [N]

            # 基础patch损失（像素重建项，支持鲁棒损失）- 仅记录（HDR 域）
            patch_l1 = self._robust_reconstruction_loss(patch_pred, patch_target)
            patch_l1 = torch.nan_to_num(patch_l1, nan=0.0, posinf=1e6, neginf=0.0)
            # 感知损失：在tone-mapped LDR上计算（与显示一致）
            tm_pred = self._tone_map_for_vgg(patch_pred)
            tm_tgt  = self._tone_map_for_vgg(patch_target)
            # 掩码化感知：pred_m = pred*mask + tgt*(1-mask)
            if self.mask_perceptual and mask_weight is not None:
                mask3 = mask_weight.expand_as(tm_pred)
                tm_pred_m = tm_pred * mask3 + tm_tgt * (1.0 - mask3)
            else:
                tm_pred_m = tm_pred
            # 面积补偿（per-sample）：逐样本计算感知损失
            if self.area_comp_enable and ('perceptual' in self.area_comp_apply_to) and (area_scale_vec is not None):
                # per-sample perceptual
                per_vec = []
                for i in range(tm_pred_m.size(0)):
                    per_i = self.perceptual_loss(tm_pred_m[i:i+1], tm_tgt[i:i+1])
                    per_vec.append(per_i)
                per_vec = torch.stack(per_vec)  # [N]
                patch_perceptual = (area_scale_vec.to(per_vec.device) * per_vec).mean()
            else:
                patch_perceptual = self.perceptual_loss(tm_pred_m, tm_tgt)
            patch_perceptual = torch.nan_to_num(patch_perceptual, nan=0.0, posinf=1e6, neginf=0.0)
            # 边缘损失：在 HDR 上
            if self.mask_edge and mask_weight is not None:
                mask3e = mask_weight.expand_as(patch_pred)
                pred_edge_m = patch_pred * mask3e + patch_target * (1.0 - mask3e)
            else:
                pred_edge_m = patch_pred
            patch_edge = self.edge_loss(pred_edge_m, patch_target)
            patch_edge = torch.nan_to_num(patch_edge, nan=0.0, posinf=1e6, neginf=0.0)
            
            # 边界感知损失（支持面积补偿的 per-sample 放大）
            boundary_loss = self._compute_boundary_loss(
                patch_pred, patch_target,
                metadata.get('patch_metadata', []) if metadata else [],
                mask_weight,
                # 传入可选 per-sample area_scale 向量（若启用且包含 'boundary'）
                (area_scale_vec if (self.area_comp_enable and ('boundary' in self.area_comp_apply_to)) else None)
            )
            boundary_loss = torch.nan_to_num(boundary_loss, nan=0.0, posinf=1e6, neginf=0.0)

            # NEW: 边缘/边界的面积补偿：边缘 per-sample，边界在函数内处理
            if self.area_comp_enable and (area_scale_vec is not None):
                if 'edge' in self.area_comp_apply_to:
                    # 重新计算 per-sample edge 并放大（避免二次重复计算，直接用当前标量近似不符合要求，因此重算）
                    edge_vec = []
                    if self.ldr_for_edge:
                        pred_e = ldr_pred; tgt_e = ldr_tgt
                    else:
                        pred_e = patch_pred; tgt_e = patch_target
                    if self.mask_edge and mask_weight is not None:
                        mask3e = (mask_weight.expand_as(pred_e))
                        pred_mix = pred_e * mask3e + (ldr_tgt if self.ldr_for_edge else patch_target) * (1.0 - mask3e)
                    else:
                        pred_mix = pred_e
                    for i in range(pred_mix.size(0)):
                        edge_i = self.edge_loss(pred_mix[i:i+1], tgt_e[i:i+1])
                        edge_vec.append(edge_i)
                    edge_vec = torch.stack(edge_vec)  # [N]
                    patch_edge = (area_scale_vec.to(edge_vec.device) * edge_vec).mean()

            # 洞区加权 L1（可选）
            hole_l1 = torch.tensor(0.0, device=patch_pred.device)
            ctx_preserve = torch.tensor(0.0, device=patch_pred.device)
            if metadata and isinstance(metadata, dict) and 'holes_mask' in metadata:
                holes_mask = torch.clamp(metadata['holes_mask'], 0.0, 1.0)  # [N,1,H,W]
                if holes_mask.shape[2:] != patch_pred.shape[2:]:
                    holes_mask = F.interpolate(holes_mask, size=patch_pred.shape[2:], mode='nearest')
                # 洞边环：3x3 maxpool - 原mask
                try:
                    k = max(3, int(self.hole_ring_kernel) if self.hole_ring_kernel % 2 == 1 else int(self.hole_ring_kernel) + 1)
                    pad = k // 2
                    ring = F.max_pool2d(holes_mask, kernel_size=k, stride=1, padding=pad) - holes_mask
                    ring = torch.clamp(ring, 0.0, 1.0)
                except Exception:
                    ring = torch.zeros_like(holes_mask)
                hole_weight = holes_mask * (1.0 + self.hole_ring_scale * ring)
                if self.w_hole > 0.0:
                    hole_l1 = self._robust_masked_loss(patch_pred, patch_target, hole_weight)
                    hole_l1 = torch.nan_to_num(hole_l1, nan=0.0, posinf=1e6, neginf=0.0)
                # 非洞“保持”：pred 接近 target_rgb（由patch_target提供）
                if self.w_ctx > 0.0:
                    ctx_mask = torch.clamp(1.0 - holes_mask, 0.0, 1.0)
                    ctx_preserve = self._robust_masked_loss(patch_pred, patch_target, ctx_mask)
                    ctx_preserve = torch.nan_to_num(ctx_preserve, nan=0.0, posinf=1e6, neginf=0.0)
            
            losses['patch_l1'] = patch_l1
            losses['patch_perceptual'] = patch_perceptual
            losses['patch_edge'] = patch_edge
            losses['patch_boundary'] = boundary_loss
            if self.w_hole > 0.0:
                losses['patch_hole_l1'] = hole_l1
            if self.w_ctx > 0.0:
                losses['patch_ctx_preserve'] = ctx_preserve

            # NEW: 洞内色彩统计一致性（在 LDR 或 HDR 域对洞区均值/方差进行匹配）
            color_stats_loss = torch.tensor(0.0, device=patch_pred.device)
            if self.color_stats_enable and self.w_color_stats > 0.0 and (metadata and isinstance(metadata, dict) and 'holes_mask' in metadata):
                holes_mask_cs = torch.clamp(metadata['holes_mask'], 0.0, 1.0)  # [N,1,H,W]
                if holes_mask_cs.shape[2:] != patch_pred.shape[2:]:
                    holes_mask_cs = F.interpolate(holes_mask_cs, size=patch_pred.shape[2:], mode='nearest')
                # 选择域
                if self.color_stats_domain == 'ldr':
                    pred_space = self._tone_map_for_vgg(patch_pred.detach())  # detach 避免二次梯度过度耦合
                    tgt_space = self._tone_map_for_vgg(patch_target.detach())
                else:
                    pred_space = patch_pred
                    tgt_space = patch_target
                # 计算洞内均值/方差（按通道、按样本），再在batch上取平均
                eps = self.color_stats_eps
                # 扩展mask到3通道
                m3 = holes_mask_cs.expand_as(pred_space)
                # 均值
                pred_sum = (pred_space * m3).sum(dim=[2,3])  # [N,3]
                tgt_sum  = (tgt_space  * m3).sum(dim=[2,3])
                mcount   = m3.sum(dim=[2,3]).clamp_min(eps)   # [N,3] 等价，但各通道mask一致
                pred_mean = pred_sum / mcount
                tgt_mean  = tgt_sum  / mcount
                mean_l1 = torch.abs(pred_mean - tgt_mean).mean()
                if self.color_stats_use_std:
                    # 方差/标准差
                    pred_var = (((pred_space - pred_mean.unsqueeze(-1).unsqueeze(-1)) ** 2) * m3).sum(dim=[2,3]) / mcount
                    tgt_var  = (((tgt_space  - tgt_mean.unsqueeze(-1).unsqueeze(-1)) ** 2) * m3).sum(dim=[2,3]) / mcount
                    pred_std = torch.sqrt(pred_var + eps)
                    tgt_std  = torch.sqrt(tgt_var  + eps)
                    std_l1 = torch.abs(pred_std - tgt_std).mean()
                else:
                    std_l1 = torch.tensor(0.0, device=patch_pred.device)
                color_stats_loss = mean_l1 + std_l1
                losses['patch_color_stats'] = color_stats_loss

            # 局部感知（patchs-in-a-patch）已移除（回退到全局感知为主）
            
            # patch总损失（全掩码化：移除全局L1项，仅保留掩码化损失）
            patch_total = (
                self.w_perc * patch_perceptual +
                self.w_edge * patch_edge +
                self.w_boundary * boundary_loss +
                self.w_hole * hole_l1 +
                self.w_ctx * ctx_preserve +
                self.w_color_stats * color_stats_loss
            )
            patch_total = torch.nan_to_num(patch_total, nan=0.0, posinf=1e6, neginf=0.0)
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
                              patch_metadata: List[Dict],
                              mask_weight: Optional[torch.Tensor] = None,
                              area_scale_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算边界感知损失（LDR空间）

        变更：在计算边界权重图与像素差之前，先对线性HDR图像做 tone‑mapping（与VGG感知一致），
        再在 LDR 空间上进行边缘检测与加权 L1，提升与显示/感知一致性。
        """
        batch_size = pred.shape[0]
        
        if batch_size == 0:
            return torch.tensor(0.0, device=pred.device)
        
        #  NEW: 自动边界检测，不依赖外部metadata
        total_boundary_loss = 0.0
        
        # 确保boundary_kernel在正确设备上
        if self.boundary_kernel.device != pred.device:
            self.boundary_kernel = self.boundary_kernel.to(pred.device)
        
        for i in range(batch_size):
            # 对每个patch计算边界损失（先做 tone‑mapping 到 LDR）
            patch_pred_hdr = pred[i:i+1]      # [1, 3, H, W] (HDR)
            patch_target_hdr = target[i:i+1]  # [1, 3, H, W] (HDR)

            # Tone-map 到 LDR 空间（与感知损失同配置）
            patch_pred = self._tone_map_for_vgg(patch_pred_hdr)
            patch_target = self._tone_map_for_vgg(patch_target_hdr)
            
            #  NEW: 多种边界检测策略组合（在 LDR 空间上进行）
            # 1. 基于目标图像的边缘检测（LDR）
            target_gray = torch.mean(patch_target, dim=1, keepdim=True)  # [1, 1, H, W] (LDR)
            target_edges = F.conv2d(target_gray, self.boundary_kernel, padding=1)
            target_boundary_map = torch.sigmoid(torch.abs(target_edges) * 2.0)
            
            # 2. 基于预测图像的边缘检测（LDR）
            pred_gray = torch.mean(patch_pred, dim=1, keepdim=True)  # [1, 1, H, W] (LDR)
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
            
                # 6. 在边界区域加权L1损失（LDR空间，支持掩码化，仅对洞区/洞边生效）
            per = F.l1_loss(patch_pred, patch_target, reduction='none')  # [1,3,H,W] (LDR)
            w_map = (1.0 + final_boundary_map)  # [1,1,H,W]
            if self.mask_boundary and mask_weight is not None:
                m_i = mask_weight[i:i+1]  # [1,1,H,W]
                m3 = m_i.expand_as(per)
                num = (per * w_map * m3).sum()
                den = m3.sum() + 1e-6
                boundary_weighted_loss = num / den
            else:
                boundary_weighted_loss = (per * w_map).mean()
            
            # 面积补偿按样本放大
            if area_scale_vec is not None:
                try:
                    s = float(area_scale_vec.view(-1)[i].item())
                    boundary_weighted_loss = boundary_weighted_loss * s
                except Exception:
                    pass
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
        
        # 损失函数（引入鲁棒损失配置）
        robust_cfg = (full_config or {}).get('loss', {}).get('robust', {}) if full_config else {}
        hdr_vis_cfg = (full_config or {}).get('hdr_processing', {}) if full_config else {}
        weights_cfg = (full_config or {}).get('loss', {}).get('weights', {}) if full_config else {}
        masking_cfg = (full_config or {}).get('loss', {}).get('masking', {}) if full_config else {}
        self.patch_loss = PatchAwareLoss(
            robust_config=robust_cfg,
            hdr_vis_config=hdr_vis_cfg,
            weights_config=weights_cfg,
            masking_config=masking_cfg,
        )

        # 梯度范数监控参数
        self.grad_clip_val = float(training_config.get('gradient_clip_val', 0.5))
        self.grad_norm_ema = 0.0
        self.grad_ema_decay = 0.95
        
        # 训练侧统一的边界检测卷积核（与损失函数一致）
        boundary_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('boundary_kernel', boundary_kernel)
        
        # Loss weight scheduling (step-based; optional)
        lc = (full_config or {}).get('loss', {}) if full_config else {}
        self.weight_sched_cfg = lc.get('weight_schedule', {}) if isinstance(lc, dict) else {}
        self.weight_sched_enable = bool(self.weight_sched_cfg.get('enable', False))
        # Cache initial weights to use as defaults
        self._w_init = {
            'perc': float(self.patch_loss.w_perc),
            'edge': float(self.patch_loss.w_edge),
            'boundary': float(self.patch_loss.w_boundary),
            'hole': float(self.patch_loss.w_hole),
            'ctx': float(self.patch_loss.w_ctx),
            'log_sup': float(getattr(self, 'log_sup_weight', getattr(self, 'log_supervision_weight', 0.0)))
        }

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
            'enable_visualization': full_config.get('visualization', {}).get('enable_visualization', True),
            # 传递HDR显示配置，便于logger使用统一的tone-mapping
            'hdr_processing': full_config.get('hdr_processing', {}),
            # 传递数据根目录，避免路径重复 processed_bistro/processed_bistro
            'data_root': full_config.get('data', {}).get('data_root', './data')
        }
        self.patch_visualizer = create_patch_visualizer(
            log_dir=training_config.get('log_dir', './logs'),
            config=viz_config
        ) if viz_config['enable_visualization'] else None

        # 输入归一化（线性HDR→按patch尺度归一化，再反归一化）配置
        norm_cfg = (full_config or {}).get('training', {}).get('input_normalization', {}) if full_config else {}
        self.in_norm_enable = bool(norm_cfg.get('enable', False))
        self.in_norm_method = str(norm_cfg.get('method', 'per_patch_percentile')).lower()
        self.in_norm_percentile = float(norm_cfg.get('percentile', 0.99))
        self.in_norm_min_scale = float(norm_cfg.get('min_scale', 1.0))
        self.in_norm_max_scale = float(norm_cfg.get('max_scale', 512.0))

        # 全局标准化配置（单个 mu、sigma）
        gs_cfg = (full_config or {}).get('training', {}).get('global_standardization', {}) if full_config else {}
        self.gs_enable = bool(gs_cfg.get('enable', False))
        self.gs_mu = 0.0
        self.gs_sigma = 1.0
        self.gs_apply_to = str(gs_cfg.get('apply_to_channels', 'rgb')).lower()
        stats_path = gs_cfg.get('stats_path', None)
        if self.gs_enable and stats_path:
            try:
                import json
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                self.gs_mu = float(stats.get('mu', 0.0))
                self.gs_sigma = float(stats.get('sigma', 1.0))
                if not (self.gs_sigma > 1e-8):
                    print(f"[WARN] Global std too small ({self.gs_sigma}), fallback to 1.0")
                    self.gs_sigma = 1.0
                print(f" Global standardization: mu={self.gs_mu:.6f}, sigma={self.gs_sigma:.6f}")
            except Exception as e:
                print(f"[WARN] Failed to load global stats from {stats_path}: {e}. Disable GS.")
                self.gs_enable = False

        # NEW: Unified normalization config (overrides legacy flags if provided)
        norm_unified = (full_config or {}).get('normalization', {}) if full_config else {}
        self.norm_type = str(norm_unified.get('type', 'global' if self.gs_enable else ('per_patch' if self.in_norm_enable else 'none'))).lower()
        # Per-patch params override
        self.in_norm_percentile = float(norm_unified.get('percentile', self.in_norm_percentile))
        self.in_norm_min_scale = float(norm_unified.get('min_scale', self.in_norm_min_scale))
        self.in_norm_max_scale = float(norm_unified.get('max_scale', self.in_norm_max_scale))
        # Global stats override
        stats_path2 = norm_unified.get('stats_path', None)
        if stats_path2 and self.norm_type == 'global':
            try:
                import json
                with open(stats_path2, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                self.gs_mu = float(stats.get('mu', self.gs_mu))
                self.gs_sigma = float(stats.get('sigma', self.gs_sigma))
            except Exception as e:
                print(f"[WARN] Failed to load GS stats from {stats_path2}: {e}")
        # Log-norm params
        self.log_epsilon = float(norm_unified.get('log_epsilon', 1.0e-8))
        self.log_delta_scale = float(norm_unified.get('log_delta_scale', 0.1))
        # NEW: absolute cap for log-delta magnitude (overrides scale*denom when > 0)
        self.log_delta_abs_max = float(norm_unified.get('log_delta_abs_max', 0.0))  # 0: disabled
        self.log_delta_alpha = float(norm_unified.get('log_delta_alpha', 1.0))     # multiplier for tanh output
        # NEW: configurable log-delta masking and log-domain supervision
        self.log_apply_delta_in_holes = bool(norm_unified.get('log_apply_delta_in_holes', True))
        self.log_delta_mask_ring_scale = float(norm_unified.get('log_delta_mask_ring_scale', 0.5))
        self.log_delta_mask_ring_kernel = int(norm_unified.get('log_delta_mask_ring_kernel', 3))
        self.log_sup_enable = bool(norm_unified.get('log_supervision_enable', False))
        self.log_sup_weight = float(norm_unified.get('log_supervision_weight', 0.5))
        self.log_sup_type = str(norm_unified.get('log_supervision_type', 'l1')).lower()  # l1|huber|charbonnier
        self.log_sup_huber_delta = float(norm_unified.get('log_supervision_huber_delta', 0.2))
        print(f"Normalization mode: {self.norm_type}")
        
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

    def _apply_loss_weight_schedule(self, step: int) -> None:
        """Apply step-based scheduling to loss weights as requested.

        Phases (default if config not provided):
          - 0..2000: log_sup=0.9, w_perc=2.0, w_hole=0.5, others unchanged
          - 2000..3000: linearly log_sup 0.9->0.3, w_perc 2.0->8.0; w_hole=0.5
          - 2000..8000: linearly ramp w_edge/w_boundary/w_ctx to 1.0
          - >=8000..12000: linearly w_hole 0.5->0.2 (others hold)
          - >=12000: hold terminal values

        All numbers can be overridden by loss.weight_schedule in YAML (optional).
        """
        # Read config or fallback defaults
        cfg = self.weight_sched_cfg or {}
        p1_end = int(cfg.get('phase1_end', 2000))
        p2_end = int(cfg.get('phase2_end', 3000))
        p3_end = int(cfg.get('phase3_end', 8000))
        hole_decay_end = int(cfg.get('hole_decay_end', 12000))

        # Targets
        log_p1 = float(cfg.get('log_p1', 0.9))
        log_p2 = float(cfg.get('log_p2', 0.3))
        perc_p1 = float(cfg.get('perc_p1', 2.0))
        perc_p2 = float(cfg.get('perc_p2', 8.0))
        hole_p1 = float(cfg.get('hole_p1', 0.5))
        hole_p4 = float(cfg.get('hole_p4', 0.2))
        other_target = float(cfg.get('other_target', 1.0))  # edge/boundary/ctx

        # Helper: linear ramp function
        def lerp(a: float, b: float, t: float) -> float:
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            return a + (b - a) * t

        # Start from initial weights
        w_perc = self._w_init['perc']
        w_edge = self._w_init['edge']
        w_boundary = self._w_init['boundary']
        w_hole = self._w_init['hole']
        w_ctx = self._w_init['ctx']
        w_log = self._w_init['log_sup']

        if step < p1_end:
            w_log = log_p1
            w_perc = perc_p1
            w_hole = hole_p1
            # others unchanged
        elif step < p2_end:
            # phase2 transition over 1k steps
            t = (step - p1_end) / max(1, (p2_end - p1_end))
            w_log = lerp(log_p1, log_p2, t)
            w_perc = lerp(perc_p1, perc_p2, t)
            w_hole = hole_p1
            # others ramp starts in phase2 and continues till p3_end
            t_other = (step - p1_end) / max(1, (p3_end - p1_end))
            w_edge = lerp(self._w_init['edge'], other_target, t_other)
            w_boundary = lerp(self._w_init['boundary'], other_target, t_other)
            w_ctx = lerp(self._w_init['ctx'], other_target, t_other)
        elif step < p3_end:
            w_log = log_p2
            w_perc = perc_p2
            w_hole = hole_p1
            t_other = (step - p1_end) / max(1, (p3_end - p1_end))
            w_edge = lerp(self._w_init['edge'], other_target, t_other)
            w_boundary = lerp(self._w_init['boundary'], other_target, t_other)
            w_ctx = lerp(self._w_init['ctx'], other_target, t_other)
        elif step < hole_decay_end:
            w_log = log_p2
            w_perc = perc_p2
            # decay w_hole from 0.5 to 0.2 over [p3_end, hole_decay_end]
            t_h = (step - p3_end) / max(1, (hole_decay_end - p3_end))
            w_hole = lerp(hole_p1, hole_p4, t_h)
            # others fixed at targets
            w_edge = other_target
            w_boundary = other_target
            w_ctx = other_target
        else:
            w_log = log_p2
            w_perc = perc_p2
            w_hole = hole_p4
            w_edge = other_target
            w_boundary = other_target
            w_ctx = other_target

        # Apply to loss and trainer
        self.patch_loss.w_perc = float(w_perc)
        self.patch_loss.w_edge = float(w_edge)
        self.patch_loss.w_boundary = float(w_boundary)
        self.patch_loss.w_hole = float(w_hole)
        self.patch_loss.w_ctx = float(w_ctx)
        self.log_sup_weight = float(w_log)
        # Log scheduled weights for monitoring
        try:
            self.log('sched_w_perc', w_perc, on_step=True, on_epoch=False, prog_bar=False)
            self.log('sched_w_edge', w_edge, on_step=True, on_epoch=False, prog_bar=False)
            self.log('sched_w_boundary', w_boundary, on_step=True, on_epoch=False, prog_bar=False)
            self.log('sched_w_hole', w_hole, on_step=True, on_epoch=False, prog_bar=False)
            self.log('sched_w_ctx', w_ctx, on_step=True, on_epoch=False, prog_bar=False)
            self.log('sched_w_log', w_log, on_step=True, on_epoch=False, prog_bar=False)
        except Exception:
            pass
    
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
        # Apply step-based loss weight scheduling (if enabled)
        if getattr(self, 'weight_sched_enable', False):
            try:
                step = self.global_step if hasattr(self, 'trainer') and self.trainer else self.visualization_step
                self._apply_loss_weight_schedule(step)
            except Exception as _e:
                if os.environ.get('LOSS_SCHED_WARN', '0') == '1':
                    print('[WARN] weight schedule failed:', _e)
        
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

            # Patch网络推理（统一归一化入口）
            residual_pred_norm = None
            if self.norm_type == 'global':
                warped_rgb = patch_input[:, :3]
                mu = torch.as_tensor(self.gs_mu, dtype=warped_rgb.dtype, device=warped_rgb.device)
                sigma = torch.as_tensor(self.gs_sigma, dtype=warped_rgb.dtype, device=warped_rgb.device)
                Xn = (warped_rgb - mu) / torch.clamp(sigma, min=1e-6)
                patch_input_norm = patch_input.clone()
                patch_input_norm[:, :3] = Xn
                residual_pred_norm = self.student_model.patch_network(patch_input_norm, boundary_override=boundary_override)
            elif self.norm_type == 'per_patch':
                warped_rgb = patch_input[:, :3]
                B = warped_rgb.shape[0]
                with torch.no_grad():
                    try:
                        q = torch.quantile(warped_rgb.view(B, -1), self.in_norm_percentile, dim=1)
                    except Exception:
                        q = warped_rgb.view(B, -1).amax(dim=1)
                    b = q.view(-1, 1, 1, 1)
                    b = torch.clamp(b, min=self.in_norm_min_scale, max=self.in_norm_max_scale)
                patch_input_norm = patch_input.clone()
                patch_input_norm[:, :3] = patch_input_norm[:, :3] / b
                residual_pred_norm = self.student_model.patch_network(patch_input_norm, boundary_override=boundary_override)
            elif self.norm_type == 'log':
                warped_rgb = patch_input[:, :3]
                eps = torch.as_tensor(self.log_epsilon, dtype=warped_rgb.dtype, device=warped_rgb.device)
                # Ensure non-negative then log (features only)
                warped_pos = torch.clamp(warped_rgb, min=0.0)
                log_img = torch.log(warped_pos + eps)
                B = warped_rgb.shape[0]
                # Min-max normalize in log domain to [0,1] for input features
                min_log = torch.amin(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
                max_log = torch.amax(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
                denom = torch.clamp(max_log - min_log, min=1e-6)
                Xn = (log_img - min_log) / denom
                patch_input_norm = patch_input.clone()
                patch_input_norm[:, :3] = Xn
                # IMPORTANT: predict delta in log space (bounded via tanh)
                residual_pred_log = self.student_model.patch_network(patch_input_norm, boundary_override=boundary_override)
                # Scale delta in log domain
                if self.log_delta_abs_max > 0.0:
                    # Absolute-cap mode: delta limited by a global max magnitude
                    delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * self.log_delta_abs_max
                else:
                    # Legacy mode: proportional to patch's log dynamic range
                    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log)
                # Optionally restrict delta to hole regions (+ring) to stabilize colors
                if self.log_apply_delta_in_holes:
                    holes_mask = torch.clamp(batch['patch_input'][:, 3:4], 0.0, 1.0)
                    if holes_mask.shape[2:] != log_img.shape[2:]:
                        holes_mask = F.interpolate(holes_mask, size=log_img.shape[2:], mode='nearest')
                    try:
                        k = max(3, int(self.log_delta_mask_ring_kernel) if int(self.log_delta_mask_ring_kernel) % 2 == 1 else int(self.log_delta_mask_ring_kernel) + 1)
                        pad = k // 2
                        ring = F.max_pool2d(holes_mask, kernel_size=k, stride=1, padding=pad) - holes_mask
                        ring = torch.clamp(ring, 0.0, 1.0)
                    except Exception:
                        ring = torch.zeros_like(holes_mask)
                    mask_weight = torch.clamp(holes_mask + self.log_delta_mask_ring_scale * ring, 0.0, 1.0)
                    delta_log = delta_log * mask_weight
            else:
                residual_pred_norm = None
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
            if self.norm_type == 'global':
                mu = torch.as_tensor(self.gs_mu, dtype=warped_rgb.dtype, device=warped_rgb.device)
                sigma = torch.as_tensor(self.gs_sigma, dtype=warped_rgb.dtype, device=warped_rgb.device)
                Xn = patch_input_norm[:, :3]  # (warped_rgb - mu)/sigma
                patch_pred_full = (Xn + residual_pred_norm) * sigma + mu
            elif self.norm_type == 'per_patch':
                warped_rgb_norm = warped_rgb / b
                patch_pred_full = (warped_rgb_norm + residual_pred_norm) * b
            elif self.norm_type == 'log':
                # Scheme B: true log-space residual learning
                Ln_hat = log_img + delta_log
                patch_pred_full = torch.exp(Ln_hat) - eps
            else:
                patch_pred_full = ResidualLearningHelper.reconstruct_from_residual(warped_rgb, residual_pred)
            # Sanitize NaN/Inf before loss & visualization
            patch_pred_full = torch.nan_to_num(patch_pred_full, nan=0.0, posinf=1e6, neginf=0.0)
            patch_target_rgb = torch.nan_to_num(patch_target_rgb, nan=0.0, posinf=1e6, neginf=0.0)
            
            # 损失计算使用完整重建图像与目标RGB
            predictions['patch'] = patch_pred_full
            targets['patch'] = patch_target_rgb

            # NEW: optional log-domain supervision on holes (+ring), comparing delta_log to GT log residual
            log_sup_loss = None
            if self.log_sup_enable and self.norm_type == 'log':
                try:
                    eps = torch.as_tensor(self.log_epsilon, dtype=patch_target_rgb.dtype, device=patch_target_rgb.device)
                    tgt_pos = torch.clamp(patch_target_rgb, min=0.0)
                    target_log = torch.log(tgt_pos + eps)
                    gt_delta_log = target_log - log_img
                    holes_mask = torch.clamp(batch['patch_input'][:, 3:4], 0.0, 1.0)
                    if holes_mask.shape[2:] != gt_delta_log.shape[2:]:
                        holes_mask = F.interpolate(holes_mask, size=gt_delta_log.shape[2:], mode='nearest')
                    try:
                        k = max(3, int(self.log_delta_mask_ring_kernel) if int(self.log_delta_mask_ring_kernel) % 2 == 1 else int(self.log_delta_mask_ring_kernel) + 1)
                        pad = k // 2
                        ring = F.max_pool2d(holes_mask, kernel_size=k, stride=1, padding=pad) - holes_mask
                        ring = torch.clamp(ring, 0.0, 1.0)
                    except Exception:
                        ring = torch.zeros_like(holes_mask)
                    sup_mask = torch.clamp(holes_mask + self.log_delta_mask_ring_scale * ring, 0.0, 1.0)
                    if self.log_sup_type == 'huber':
                        per = F.smooth_l1_loss(delta_log, gt_delta_log, beta=self.log_sup_huber_delta, reduction='none')
                    elif self.log_sup_type == 'charbonnier':
                        diff = delta_log - gt_delta_log
                        per = torch.sqrt(diff * diff + 1.0e-6)
                    else:
                        per = torch.abs(delta_log - gt_delta_log)
                    num = (per * sup_mask).sum()
                    den = sup_mask.sum() + 1e-6
                    log_sup_loss = torch.nan_to_num(num / den, nan=0.0, posinf=1e6, neginf=0.0)
                except Exception as e:
                    print("[WARN] Failed to compute log-domain supervision:", e)
                    log_sup_loss = None
            
            self.training_stats['patch_steps'] += 1
        else:
            # 如果没有patch数据，跳过这个batch
            return {'loss': torch.tensor(0.0, requires_grad=True)}
        
        # 现在只有patch模式
        mode = 'patch'
        
        # 计算损失（传入洞区掩码用于洞区加权L1）
        hole_meta = None
        if 'patch_input' in batch:
            hole_meta = {
                'holes_mask': batch['patch_input'][:, 3:4],
                'warped_rgb': batch['patch_input'][:, :3],
            }
        loss, loss_dict = self.patch_loss(
            predictions, targets, mode, current_epoch,
            metadata=hole_meta
        )
        if self.log_sup_enable and (log_sup_loss is not None):
            loss = loss + self.log_sup_weight * log_sup_loss
            loss_dict['log_supervision'] = log_sup_loss.detach()

        # 重建误差分位数（训练）
        if 'patch' in predictions and 'patch' in targets:
            self._log_recon_error_percentiles(predictions['patch'], targets['patch'], prefix='train')
        
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
            **{f'train_{k}': v for k, v in loss_dict.items() if k != 'total' and v is not None}
        }, on_step=True, on_epoch=True, prog_bar=True)

        # NEW: batch hole stats logging
        try:
            holes = batch['patch_input'][:, 3:4]
            frac = holes.mean()
            # ratio above small threshold
            thr = 1.0e-3
            nz = (holes.view(holes.size(0), -1).mean(dim=1) > thr).float().mean()
            self.log('train_hole_frac_mean', frac, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_hole_nonzero_ratio', nz, on_step=True, on_epoch=True, prog_bar=False)
        except Exception:
            pass
        
        # Patch可视化记录（使用Lightning的global_step）
        current_global_step = self.global_step if hasattr(self, 'trainer') and self.trainer else self.visualization_step
        
        if self.patch_visualizer and self.patch_visualizer.should_visualize(current_global_step):
            try:
                #  记录patch对比（输入|目标RGB|重建图像）- 残差学习版本
                if (
                    'patch' in predictions and 'patch' in targets and
                    isinstance(predictions['patch'], torch.Tensor) and
                    'patch_input' in batch and isinstance(batch.get('patch_input'), torch.Tensor) and
                    'patch_target_rgb' in batch and isinstance(batch.get('patch_target_rgb'), torch.Tensor) and
                    predictions['patch'].shape[0] > 0 and batch['patch_input'].shape[0] > 0 and batch['patch_target_rgb'].shape[0] > 0
                ):
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
                # Debug context
                try:
                    pi = batch.get('patch_input', None)
                    pt = batch.get('patch_target_rgb', None)
                    pp = predictions.get('patch', None)
                    print("WARNING: Visualization logging failed:", e)
                    print("  types:", type(pi), type(pt), type(pp))
                    if isinstance(pi, torch.Tensor):
                        print("  patch_input shape:", tuple(pi.shape))
                    if isinstance(pt, torch.Tensor):
                        print("  patch_target_rgb shape:", tuple(pt.shape))
                    if isinstance(pp, torch.Tensor):
                        print("  predictions[patch] shape:", tuple(pp.shape))
                except Exception:
                    pass
        
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
            
            # 网络输出残差预测（统一归一化入口，与训练一致）
            if self.norm_type == 'global':
                warped_rgb = patch_input[:, :3]
                mu = torch.as_tensor(self.gs_mu, dtype=warped_rgb.dtype, device=warped_rgb.device)
                sigma = torch.as_tensor(self.gs_sigma, dtype=warped_rgb.dtype, device=warped_rgb.device)
                Xn = (warped_rgb - mu) / torch.clamp(sigma, min=1e-6)
                patch_input_norm = patch_input.clone()
                patch_input_norm[:, :3] = Xn
                residual_pred_norm = self.student_model.patch_network(patch_input_norm, boundary_override=boundary_override)
            elif self.norm_type == 'per_patch':
                warped_rgb = patch_input[:, :3]
                B = warped_rgb.shape[0]
                with torch.no_grad():
                    try:
                        q = torch.quantile(warped_rgb.view(B, -1), self.in_norm_percentile, dim=1)
                    except Exception:
                        q = warped_rgb.view(B, -1).amax(dim=1)
                    b = q.view(-1, 1, 1, 1)
                    b = torch.clamp(b, min=self.in_norm_min_scale, max=self.in_norm_max_scale)
                patch_input_norm = patch_input.clone()
                patch_input_norm[:, :3] = patch_input_norm[:, :3] / b
                residual_pred_norm = self.student_model.patch_network(patch_input_norm, boundary_override=boundary_override)
            elif self.norm_type == 'log':
                warped_rgb = patch_input[:, :3]
                eps = torch.as_tensor(self.log_epsilon, dtype=warped_rgb.dtype, device=warped_rgb.device)
                warped_pos = torch.clamp(warped_rgb, min=0.0)
                log_img = torch.log(warped_pos + eps)
                B = warped_rgb.shape[0]
                min_log = torch.amin(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
                max_log = torch.amax(log_img.view(B, -1), dim=1).view(-1, 1, 1, 1)
                denom = torch.clamp(max_log - min_log, min=1e-6)
                Xn = (log_img - min_log) / denom
                patch_input_norm = patch_input.clone()
                patch_input_norm[:, :3] = Xn
                # Predict delta in log space and reconstruct via exp
                residual_pred_log = self.student_model.patch_network(patch_input_norm, boundary_override=boundary_override)
                if self.log_delta_abs_max > 0.0:
                    delta_log = self.log_delta_alpha * torch.tanh(residual_pred_log) * self.log_delta_abs_max
                else:
                    delta_log = (self.log_delta_scale * denom) * torch.tanh(residual_pred_log)
                if self.log_apply_delta_in_holes:
                    holes_mask = torch.clamp(batch['patch_input'][:, 3:4], 0.0, 1.0)
                    if holes_mask.shape[2:] != log_img.shape[2:]:
                        holes_mask = F.interpolate(holes_mask, size=log_img.shape[2:], mode='nearest')
                    try:
                        k = max(3, int(self.log_delta_mask_ring_kernel) if int(self.log_delta_mask_ring_kernel) % 2 == 1 else int(self.log_delta_mask_ring_kernel) + 1)
                        pad = k // 2
                        ring = F.max_pool2d(holes_mask, kernel_size=k, stride=1, padding=pad) - holes_mask
                        ring = torch.clamp(ring, 0.0, 1.0)
                    except Exception:
                        ring = torch.zeros_like(holes_mask)
                    mask_weight = torch.clamp(holes_mask + self.log_delta_mask_ring_scale * ring, 0.0, 1.0)
                    delta_log = delta_log * mask_weight
                Ln_hat = log_img + delta_log
                patch_pred_full = torch.exp(Ln_hat) - eps
            else:
                residual_pred = self.student_model.patch_network(batch['patch_input'], boundary_override=boundary_override)
            
            # 使用统一的残差学习工具类转换为完整图像
            try:
                from residual_learning_helper import ResidualLearningHelper
            except ImportError:
                import sys
                sys.path.append('./train')
                from residual_learning_helper import ResidualLearningHelper
            
            warped_rgb = batch['patch_input'][:, :3]
            if self.norm_type == 'global':
                mu = torch.as_tensor(self.gs_mu, dtype=warped_rgb.dtype, device=warped_rgb.device)
                sigma = torch.as_tensor(self.gs_sigma, dtype=warped_rgb.dtype, device=warped_rgb.device)
                Xn = patch_input_norm[:, :3]
                patch_pred_full = (Xn + residual_pred_norm) * sigma + mu
            elif self.norm_type == 'per_patch':
                warped_rgb_norm = warped_rgb / b
                patch_pred_full = (warped_rgb_norm + residual_pred_norm) * b
            elif self.norm_type == 'log':
                # Already reconstructed via exp(log + delta)
                pass
            else:
                patch_pred_full = ResidualLearningHelper.reconstruct_from_residual(warped_rgb, residual_pred)
            patch_pred_full = torch.nan_to_num(patch_pred_full, nan=0.0, posinf=1e6, neginf=0.0)
            
            predictions['patch'] = patch_pred_full
            targets['patch'] = batch['patch_target_rgb']  # 使用RGB目标
        
        # 现在只有patch模式
        mode = 'patch'

        # 计算验证损失（传入洞区掩码用于洞区加权L1）
        hole_meta = None
        if 'patch_input' in batch:
            hole_meta = {
                'holes_mask': batch['patch_input'][:, 3:4],
                'warped_rgb': batch['patch_input'][:, :3],
            }
        val_loss, val_loss_dict = self.patch_loss(
            predictions, targets, mode, self.current_epoch,
            metadata=hole_meta
        )

        # 重建误差分位数（验证）
        if 'patch' in predictions and 'patch' in targets:
            self._log_recon_error_percentiles(predictions['patch'], targets['patch'], prefix='val')
        
        # 日志记录
        self.log_dict({
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_loss_dict.items() if k != 'total' and v is not None}
        }, on_step=False, on_epoch=True, prog_bar=True)

        # NEW: validation hole stats logging
        try:
            holes = batch['patch_input'][:, 3:4]
            frac = holes.mean()
            thr = 1.0e-3
            nz = (holes.view(holes.size(0), -1).mean(dim=1) > thr).float().mean()
            self.log('val_hole_frac_mean', frac, on_step=False, on_epoch=True, prog_bar=False)
            self.log('val_hole_nonzero_ratio', nz, on_step=False, on_epoch=True, prog_bar=False)
        except Exception:
            pass
        
        # 验证可视化记录
        current_global_step = self.global_step if hasattr(self, 'trainer') and self.trainer else self.visualization_step
        
        if self.patch_visualizer and self.current_epoch % 5 == 0:  # 每5个epoch记录验证图像
            try:
                #  验证可视化 - 残差学习版本
                if (
                    'patch' in predictions and 'patch' in targets and
                    isinstance(predictions['patch'], torch.Tensor) and
                    'patch_input' in batch and isinstance(batch.get('patch_input'), torch.Tensor) and
                    'patch_target_rgb' in batch and isinstance(batch.get('patch_target_rgb'), torch.Tensor) and
                    predictions['patch'].shape[0] > 0 and batch['patch_input'].shape[0] > 0 and batch['patch_target_rgb'].shape[0] > 0
                ):
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
                # Debug context
                try:
                    pi = batch.get('patch_input', None)
                    pt = batch.get('patch_target_rgb', None)
                    pp = predictions.get('patch', None)
                    print("WARNING: Validation visualization logging failed:", e)
                    print("  types:", type(pi), type(pt), type(pp))
                    if isinstance(pi, torch.Tensor):
                        print("  patch_input shape:", tuple(pi.shape))
                    if isinstance(pt, torch.Tensor):
                        print("  patch_target_rgb shape:", tuple(pt.shape))
                    if isinstance(pp, torch.Tensor):
                        print("  predictions[patch] shape:", tuple(pp.shape))
                except Exception:
                    pass
        
        return {
            'val_loss': val_loss,
            'mode': mode,
            'predictions': predictions,
            'targets': targets
        }

    def on_after_backward(self) -> None:
        """记录全局梯度范数及裁剪触发情况（TensorBoard标量）。"""
        # 计算全局L2梯度范数
        total_sq = 0.0
        for p in self.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            if g.is_sparse:
                g = g.coalesce().values()
            total_sq += float(g.pow(2).sum().item())
        global_norm = total_sq ** 0.5

        # EMA 平滑
        self.grad_norm_ema = self.grad_ema_decay * self.grad_norm_ema + (1.0 - self.grad_ema_decay) * global_norm
        clipped = 1.0 if global_norm > self.grad_clip_val else 0.0

        # 记录到TensorBoard（Lightning的 self.log）
        self.log('grad_norm', global_norm, on_step=True, on_epoch=False, prog_bar=False)
        self.log('grad_norm_ema', self.grad_norm_ema, on_step=True, on_epoch=False, prog_bar=False)
        self.log('grad_clip_val', self.grad_clip_val, on_step=True, on_epoch=False, prog_bar=False)
        self.log('grad_clipped', clipped, on_step=True, on_epoch=False, prog_bar=False)

    def _log_recon_error_percentiles(self, pred: torch.Tensor, target: torch.Tensor, prefix: str) -> None:
        """记录重建误差分位数（p50/p90/p95）与平均值到TensorBoard。

        Args:
            pred: [N,3,H,W]
            target: [N,3,H,W]
            prefix: 'train' or 'val'
        """
        try:
            err = torch.abs(pred - target)
            flat = err.view(err.size(0), -1)
            # 使用批次平均的分位数：先对每个样本求分位数，再取均值
            qs = torch.tensor([0.5, 0.9, 0.95], device=flat.device)
            per_sample = torch.quantile(flat, qs, dim=1)  # [3, N]
            vals = per_sample.mean(dim=1)  # [3]
            p50, p90, p95 = vals[0], vals[1], vals[2]
            mean_err = err.mean()
            self.log(f'{prefix}_err_p50', p50, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{prefix}_err_p90', p90, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{prefix}_err_p95', p95, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f'{prefix}_err_mean', mean_err, on_step=True, on_epoch=True, prog_bar=False)
        except Exception:
            # 兼容torch不支持quantile的情况：退化为均值
            try:
                mean_err = torch.abs(pred - target).mean()
                self.log(f'{prefix}_err_mean', mean_err, on_step=True, on_epoch=True, prog_bar=False)
            except Exception:
                pass
            return
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 优化器配置 - 确保类型转换
        optimizer_config = self.training_config.get('optimizer', {})

        # 安全的类型转换
        learning_rate = float(optimizer_config.get('learning_rate', 1e-4))
        weight_decay = float(optimizer_config.get('weight_decay', 1e-5))
        eps = float(optimizer_config.get('eps', 1e-6))
        beta2 = float(optimizer_config.get('beta2', 0.98))

        print(f" 优化器配置: lr={learning_rate}, weight_decay={weight_decay}, eps={eps}, beta2={beta2}")

        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, beta2),
            eps=eps
        )

        # 学习率调度器 - 确保类型转换
        scheduler_config = self.training_config.get('scheduler', {})

        T_max = int(scheduler_config.get('T_max', 100))
        eta_min = float(scheduler_config.get('eta_min', 1e-6))

        print(f" 调度器配置: T_max={T_max}, eta_min={eta_min}")

        # Warmup 配置（step 级）
        warmup_steps = int(self.training_config.get('warmup_steps', 0))
        warmup_start_factor = float(self.training_config.get('warmup_start_factor', 0.1))
        if warmup_steps > 0:
            print(f" Warmup: steps={warmup_steps}, start_factor={warmup_start_factor}")
            def lr_lambda(current_step: int):
                if current_step >= warmup_steps:
                    return 1.0
                start = warmup_start_factor
                return start + (1.0 - start) * (current_step / max(1, warmup_steps))
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            warmup_scheduler = None

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        result = {'optimizer': optimizer}
        schedulers = []
        # step 级 warmup
        if warmup_scheduler is not None:
            schedulers.append({
                'scheduler': warmup_scheduler,
                'interval': 'step',
                'frequency': 1
            })
        # epoch 级 Cosine
        schedulers.append({
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        })
        result['lr_scheduler'] = schedulers if len(schedulers) > 1 else schedulers[0]
        return result
    
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
            'max_epochs': training_config.get('max_epochs', 100),
            'gradient_clip_val': training_config.get('gradient_clip_val', 0.5)
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
        success = setup_data_loaders(trainer, data_config, patch_training_config, full_config=config)
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
        gc_val = training_config.get('gradient_clip_val', 0.5)
        pl_trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=tb_logger,
            callbacks=callbacks,
            accelerator='auto',  # 自动检测GPU/CPU
            devices='auto',      # 自动检测设备数量
            precision=32,        # 使用FP32精度
            log_every_n_steps=10,
            check_val_every_n_epoch=1,
            gradient_clip_val=gc_val,
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
    
    def __init__(self, data_root: str, split: str, patch_config, adapter_kwargs: Optional[Dict[str, Any]] = None):
        from colleague_dataset_adapter import ColleagueDatasetAdapter
        
        # 创建基础数据集
        self.base_dataset = ColleagueDatasetAdapter(
            data_root=data_root,
            split=split,
            augmentation=False,
            **(adapter_kwargs or {})
        )
        
        self.patch_config = patch_config

        # 回退：仅支持简单网格策略
        # NEW: hole-aware sampling controls (with safe defaults)
        self._hole_sampling_enabled = bool(getattr(patch_config, 'sample_hole_weighted', True))
        self._enforce_min_hole = bool(getattr(patch_config, 'enforce_min_hole_frac', True))
        self._min_hole_frac = float(getattr(patch_config, 'min_hole_frac', 0.005))  # 0.5%
        # For stats/debug
        self._last_hole_fracs = None
        
        # 初始化简单网格提取器（当未启用重叠crop时）
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
            raise NotImplementedError("未启用简单网格或重叠crop，至少需开启其一")
    
    def __len__(self):
        # 每个图像产生固定数量的patch
        return len(self.base_dataset) * self.patch_config.max_patches_per_image
    
    def __getitem__(self, idx):
        # 简单网格模式：计算源图像索引和patch索引
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
            
            # Hole-aware sampling: prefer patches with larger hole coverage
            try:
                # Prepare holes mask (from input)
                # Get holes channel as numpy in [H,W]
                if isinstance(input_numpy, np.ndarray):
                    if input_numpy.ndim == 3 and input_numpy.shape[-1] == 7:
                        holes_np = input_numpy[:, :, 3]
                    elif input_numpy.ndim == 3 and input_numpy.shape[0] == 7:
                        holes_np = input_numpy[3, :, :]
                    else:
                        # Fallback: derive from original tensors
                        holes_np = input_tensor[3].numpy()
                else:
                    holes_np = input_tensor[3].numpy()

                # Extract hole patches aligned with grid
                hole_patches, _ = self.patch_extractor.extract_patches(holes_np)
                # Compute hole fraction per patch
                hole_fracs = []
                for hp in hole_patches:
                    # hp shape: [h,w]
                    try:
                        hole_fracs.append(float(np.mean(hp)))
                    except Exception:
                        hole_fracs.append(0.0)
                self._last_hole_fracs = hole_fracs

                # Determine effective index
                order = np.argsort(np.asarray(hole_fracs))[::-1] if len(hole_fracs) > 0 else None
                eff_idx = patch_idx
                if self._hole_sampling_enabled and order is not None:
                    eff_idx = int(order[patch_idx % len(order)])
                if self._enforce_min_hole and order is not None:
                    # If chosen patch has too few holes, pick first above threshold if exists
                    if hole_fracs[eff_idx] < self._min_hole_frac:
                        found = None
                        for j in order:
                            if hole_fracs[int(j)] >= self._min_hole_frac:
                                found = int(j); break
                        if found is not None:
                            eff_idx = found
                # 获取指定的patch
                patch_input = input_patches[eff_idx]
                patch_target_residual = target_residual_patches[eff_idx]
                patch_target_rgb = target_rgb_patches[eff_idx]
            except Exception:
                # Fallback to original index on any error
                patch_input = input_patches[patch_idx]
                patch_target_residual = target_residual_patches[patch_idx]
                patch_target_rgb = target_rgb_patches[patch_idx]
            
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
            
            #  形状验证（维持原网格尺寸）
            expected_h, expected_w = 270, 480
            assert patch_input.shape[0] == 7 and patch_input.shape[1:] == (expected_h, expected_w), "输入patch尺寸不匹配"
            assert patch_target_residual.shape[0] == 3 and patch_target_residual.shape[1:] == (expected_h, expected_w), "残差目标patch尺寸不匹配"
            assert patch_target_rgb.shape[0] == 3 and patch_target_rgb.shape[1:] == (expected_h, expected_w), "RGB目标patch尺寸不匹配"
            
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

    # 回退：无重叠crop索引构建


def setup_data_loaders(trainer, data_config: Dict[str, Any], patch_config, full_config: Optional[Dict[str, Any]] = None) -> bool:
    """设置数据加载器"""
    try:
        print(" 设置数据加载器...")
        
        # 根据dataset_type选择数据集
        dataset_type = data_config.get('dataset_type', 'colleague')
        
        if dataset_type == 'colleague':
            # 使用ColleagueDatasetAdapter的patch包装器
            from colleague_dataset_adapter import ColleagueDatasetAdapter
            
            print(" 使用ColleagueDatasetAdapter + ColleaguePatchDataset")
            
            # 读取HDR处理配置
            hdr_cfg = (full_config or {}).get('hdr_processing', {}) if full_config else {}
            adapter_kwargs = dict(
                enable_linear_preprocessing=hdr_cfg.get('enable_linear_preprocessing', True),
                enable_srgb_linear=hdr_cfg.get('enable_srgb_linear', True),
                scale_factor=hdr_cfg.get('scale_factor', 0.70),
                tone_mapping_for_display=hdr_cfg.get('tone_mapping_for_display', 'reinhard'),
                gamma=hdr_cfg.get('gamma', 2.2),
                exposure=hdr_cfg.get('exposure', 1.0),
                adaptive_exposure=hdr_cfg.get('adaptive_exposure', {'enable': False}),
                mulaw_mu=hdr_cfg.get('mulaw_mu', 500.0),
            )

            # 创建ColleaguePatchDataset包装器
            train_dataset = ColleaguePatchDataset(
                data_root=data_config.get('data_root', './data'),
                split='train',
                patch_config=patch_config,
                adapter_kwargs=adapter_kwargs
            )
            
            val_dataset = ColleaguePatchDataset(
                data_root=data_config.get('data_root', './data'),
                split='val',
                patch_config=patch_config,
                adapter_kwargs=adapter_kwargs
            )
            
            print(f" 训练样本: {len(train_dataset)} patches")
            print(f" 验证样本: {len(val_dataset)} patches")
            
        else:
            print(f" 不支持的数据集类型: {dataset_type}")
            return False
        
        # 创建数据加载器 - 从正确的配置位置获取参数
        training_section = trainer.training_config if hasattr(trainer, 'training_config') else {}
        batch_size = training_section.get('batch_size', 4)
        # Windows 平台默认使用单进程避免pickle问题；优先从配置读取
        default_workers = 0 if os.name == 'nt' else 2
        num_workers = int(training_section.get('num_workers', default_workers))
        print(f" DataLoader num_workers: {num_workers} (platform={'Windows' if os.name=='nt' else 'Unix-like'})")
        
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
