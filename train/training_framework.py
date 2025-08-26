"""
PyTorch Lightning training framework for mobile inpainting network.

Features:
- Teacher-Student knowledge distillation
- Multi-loss optimization (L1 + Perceptual + Edge + Temporal)
- Mixed precision training with gradient clipping
- TensorBoard visualization

Target: <3M parameters, SSIM>0.95, <100 epochs convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Import network models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'networks'))
from mobile_inpainting_network import MobileInpaintingNetwork

# Import training monitor
from training_monitor import TrainingMonitor, create_training_monitor


class PerceptualLoss(nn.Module):
    """VGG19-based perceptual loss for visual quality enhancement."""
    
    def __init__(self, feature_layers=[2, 7, 12, 21, 30]):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG19 and extract specified layers
        vgg = vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        
        self.features = nn.ModuleList()
        for layer_idx in feature_layers:
            self.features.append(vgg[:layer_idx+1])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        """Compute perceptual loss using VGG features."""
        # Normalize to [0,1] range
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        
        # Convert grayscale to RGB if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0.0
        for feature_extractor in self.features:
            pred_feat = feature_extractor(pred)
            target_feat = feature_extractor(target)
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(self.features)


class EdgeLoss(nn.Module):
    """Sobel-based edge detection loss for contour preservation."""
    
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
        # Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        """Compute edge loss using Sobel operators."""
        # Convert to grayscale
        pred_gray = torch.mean(pred, dim=1, keepdim=True)
        target_gray = torch.mean(target, dim=1, keepdim=True)
        
        # Compute edges
        pred_edge_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        
        target_edge_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        return F.l1_loss(pred_edge, target_edge)


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for frame stability."""
    
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
    
    def forward(self, pred_current, pred_previous, target_current, target_previous):
        """Compute temporal consistency loss between consecutive frames."""
        # Compute frame differences
        pred_diff = pred_current - pred_previous
        target_diff = target_current - target_previous
        
        return F.l1_loss(pred_diff, target_diff)


class CombinedLoss(nn.Module):
    """Combined loss function: L1 + Perceptual + Edge + Temporal."""
    
    def __init__(self, 
                 lambda_l1=1.0,
                 lambda_perceptual=0.1, 
                 lambda_edge=0.05,
                 lambda_temporal=0.02):
        super(CombinedLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_edge = lambda_edge
        self.lambda_temporal = lambda_temporal
        
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.edge_loss = EdgeLoss()
        self.temporal_loss = TemporalConsistencyLoss()
    
    def forward(self, pred, target, pred_prev=None, target_prev=None):
        """
        计算总损失
        Args:
            pred: 预测输出 [B, 3, H, W]
            target: 真值目标 [B, 3, H, W]
            pred_prev: 前一帧预测（可选）
            target_prev: 前一帧真值（可选）
        """
        losses = {}
        
        # Reconstruction loss
        l1_loss = self.l1_loss(pred, target)
        losses['l1'] = l1_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(pred, target)
        losses['perceptual'] = perceptual_loss
        
        # Edge loss
        edge_loss = self.edge_loss(pred, target)
        losses['edge'] = edge_loss
        
        # Total loss
        total_loss = (self.lambda_l1 * l1_loss + 
                     self.lambda_perceptual * perceptual_loss + 
                     self.lambda_edge * edge_loss)
        
        # Temporal consistency loss (if previous frame provided)
        if pred_prev is not None and target_prev is not None:
            temporal_loss = self.temporal_loss(pred, pred_prev, target, target_prev)
            losses['temporal'] = temporal_loss
            total_loss += self.lambda_temporal * temporal_loss
        
        losses['total'] = total_loss
        return total_loss, losses


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for Teacher-Student training."""
    
    def __init__(self, temperature=4.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_output, teacher_output, target):
        """Compute distillation loss for regression task."""
        # Feature distillation loss (MSE for regression)
        feature_distillation_loss = self.mse_loss(student_output, teacher_output)
        
        # Hard target loss
        hard_loss = F.l1_loss(student_output, target)
        
        # Combined loss: balance feature matching and target matching
        total_loss = self.alpha * feature_distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, {
            'feature_distillation': feature_distillation_loss,
            'hard': hard_loss,
            'total': total_loss
        }


class FrameInterpolationTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for frame interpolation with knowledge distillation."""
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 teacher_model_path: Optional[str] = None,
                 full_config: Optional[Dict[str, Any]] = None):
        super(FrameInterpolationTrainer, self).__init__()
        
        self.save_hyperparameters()
        
        # Create student network
        inpainting_config = model_config.get('inpainting_network', {})
        self.student_model = MobileInpaintingNetwork(
            input_channels=inpainting_config.get('input_channels', 7),
            output_channels=inpainting_config.get('output_channels', 3)
        )
        
        # Load teacher network for knowledge distillation
        self.teacher_model = None
        if teacher_model_path and os.path.exists(teacher_model_path):
            self.teacher_model = MobileInpaintingNetwork(
                input_channels=inpainting_config.get('input_channels', 7),
                output_channels=inpainting_config.get('output_channels', 3)
            )
            checkpoint = torch.load(teacher_model_path, map_location='cpu')
            self.teacher_model.load_state_dict(checkpoint['state_dict'])
            self.teacher_model.eval()
            
            # Freeze teacher network parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        # 损失函数 - 确保类型正确
        loss_weights = training_config.get('loss_weights', {})
        self.criterion = CombinedLoss(
            lambda_l1=float(loss_weights.get('reconstruction', 1.0)),
            lambda_perceptual=float(loss_weights.get('perceptual', 0.1)),
            lambda_edge=float(loss_weights.get('temporal_consistency', 0.05)),
            lambda_temporal=float(loss_weights.get('temporal_consistency', 0.02))
        )
        
        # 蒸馏损失
        if self.teacher_model is not None:
            distillation_config = training_config.get('distillation', {})
            self.distillation_loss = DistillationLoss(
                temperature=float(distillation_config.get('temperature', 4.0)),
                alpha=float(distillation_config.get('alpha', 0.7))
            )
        
        # 训练配置 - 确保类型正确
        self.learning_rate = float(training_config.get('learning_rate', 1e-4))
        self.weight_decay = float(training_config.get('weight_decay', 1e-5))
        
        # 指标统计
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # 初始化训练监控器
        self.monitor = None
        if full_config:
            self.monitor = create_training_monitor(full_config, self.student_model)
            if self.monitor:
                print("INFO: Advanced training monitoring system enabled")
    
    def on_fit_start(self):
        """训练开始时连接Lightning日志器"""
        if self.monitor and self.logger:
            # 连接Lightning的TensorBoard日志器
            self.monitor.set_lightning_logger(self.logger)
            print("INFO: Training monitor connected to Lightning logger")
    
    def forward(self, x):
        """前向传播"""
        return self.student_model(x)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        # 开始step监控
        if self.monitor:
            self.monitor.start_step()
        
        # 解包数据 - 修正为匹配corrected_dataset.py的输出格式
        if len(batch) == 2:
            # 新格式: (input_tensor, target_tensor) from corrected_dataset.py
            input_data, target = batch
            input_prev, target_prev = None, None
        elif len(batch) == 4:
            # 旧格式: (input_data, target, input_prev, target_prev)
            input_data, target, input_prev, target_prev = batch
        else:
            raise ValueError(f"Unexpected batch format. Expected 2 or 4 items, got {len(batch)}")
        
        # 学生网络预测
        pred = self.student_model(input_data)
        pred_prev = self.student_model(input_prev) if input_prev is not None else None
        
        # 计算主要损失
        main_loss, loss_dict = self.criterion(pred, target, pred_prev, target_prev)
        
        # 知识蒸馏损失
        total_loss = main_loss
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_pred = self.teacher_model(input_data)
            
            distill_loss, distill_dict = self.distillation_loss(pred, teacher_pred, target)
            total_loss += distill_loss
            loss_dict.update({f'distill_{k}': v for k, v in distill_dict.items()})
        
        # 记录损失
        self.log('train/total_loss', total_loss, prog_bar=True)
        for key, value in loss_dict.items():
            if key != 'total':  # 避免重复记录total_loss
                self.log(f'train/{key}_loss', value)
        
        # 记录指标
        with torch.no_grad():
            ssim_score = self._calculate_ssim(pred, target)
            psnr_score = self._calculate_psnr(pred, target)
            self.log('train/ssim', ssim_score)
            self.log('train/psnr', psnr_score)
        
        # 高级监控记录
        if self.monitor:
            # 获取当前学习率
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer and self.trainer.optimizers else self.learning_rate
            
            # 构建损失字典用于监控
            monitor_loss_dict = {'total': total_loss.item()}
            for key, value in loss_dict.items():
                monitor_loss_dict[key] = value.item() if torch.is_tensor(value) else value
            
            # 结束step监控（传入损失和学习率）
            self.monitor.end_step(monitor_loss_dict, current_lr)
            
            # 梯度监控
            self.monitor.monitor_gradients()
        
        self.training_step_outputs.append(total_loss)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 解包数据 - 修正为匹配corrected_dataset.py的输出格式
        if len(batch) == 2:
            # 新格式: (input_tensor, target_tensor) from corrected_dataset.py
            input_data, target = batch
            input_prev, target_prev = None, None
        elif len(batch) == 4:
            # 旧格式: (input_data, target, input_prev, target_prev)
            input_data, target, input_prev, target_prev = batch
        else:
            raise ValueError(f"Unexpected batch format. Expected 2 or 4 items, got {len(batch)}")
        
        # 预测
        pred = self.student_model(input_data)
        pred_prev = self.student_model(input_prev) if input_prev is not None else None
        
        # 计算损失
        val_loss, loss_dict = self.criterion(pred, target, pred_prev, target_prev)
        
        # 计算指标
        ssim_score = self._calculate_ssim(pred, target)
        psnr_score = self._calculate_psnr(pred, target)
        
        # 记录
        self.log('val/total_loss', val_loss, prog_bar=True)
        self.log('val/ssim', ssim_score, prog_bar=True)
        self.log('val/psnr', psnr_score, prog_bar=True)
        
        for key, value in loss_dict.items():
            if key != 'total':  # 避免重复记录total_loss
                self.log(f'val/{key}_loss', value)
        
        # 图像可视化保存（只在第一个batch保存，避免过多I/O）
        if self.monitor and batch_idx == 0:
            self.monitor.save_training_images(input_data, pred, target, prefix="val")
        
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'ssim': ssim_score,
            'psnr': psnr_score
        })
        
        return val_loss
    
    def on_train_epoch_start(self):
        """训练epoch开始"""
        if self.monitor:
            self.monitor.start_epoch(self.current_epoch)
    
    def on_train_epoch_end(self):
        """训练epoch结束"""
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log('epoch/train_loss', avg_loss)
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        """验证epoch结束"""
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_ssim = torch.stack([x['ssim'] for x in self.validation_step_outputs]).mean()
            avg_psnr = torch.stack([x['psnr'] for x in self.validation_step_outputs]).mean()
            
            self.log('epoch/val_loss', avg_loss)
            self.log('epoch/val_ssim', avg_ssim)
            self.log('epoch/val_psnr', avg_psnr)
            
            # 高级监控epoch结束
            if self.monitor:
                # 获取训练损失
                train_loss = torch.stack(self.training_step_outputs).mean().item() if self.training_step_outputs else 0.0
                self.monitor.end_epoch(
                    train_loss,
                    avg_loss.item(),
                    avg_ssim.item(),
                    avg_psnr.item()
                )
            
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss'
            }
        }
    
    def _calculate_ssim(self, pred, target):
        """
        计算SSIM指标 - 修复为正确的SSIM计算
        Args:
            pred: 预测图像 [B, C, H, W], 范围[-1, 1]
            target: 目标图像 [B, C, H, W], 范围[-1, 1]
        """
        # 转换到[0, 1]范围进行SSIM计算
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        
        # 确保数值稳定性
        pred_01 = torch.clamp(pred_01, 0.0, 1.0)
        target_01 = torch.clamp(target_01, 0.0, 1.0)
        
        # SSIM常数
        c1 = (0.01 * 1.0) ** 2  # 数据范围为1.0
        c2 = (0.03 * 1.0) ** 2
        
        # 计算均值、方差和协方差
        mu1 = F.avg_pool2d(pred_01, 3, 1, 1)
        mu2 = F.avg_pool2d(target_01, 3, 1, 1)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(pred_01 ** 2, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(target_01 ** 2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(pred_01 * target_01, 3, 1, 1) - mu1_mu2
        
        # SSIM计算
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_map = numerator / (denominator + 1e-8)  # 避免除零
        ssim = ssim_map.mean()
        
        # 确保SSIM在合理范围内
        ssim = torch.clamp(ssim, -1.0, 1.0)
        
        return ssim
    
    def _calculate_psnr(self, pred, target):
        """
        计算PSNR指标 - 修复为正确的PSNR计算
        Args:
            pred: 预测图像 [B, C, H, W], 范围[-1, 1]
            target: 目标图像 [B, C, H, W], 范围[-1, 1]
        """
        # 转换到[0, 1]范围进行PSNR计算
        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        
        # 确保数值稳定性
        pred_01 = torch.clamp(pred_01, 0.0, 1.0)
        target_01 = torch.clamp(target_01, 0.0, 1.0)
        
        # 计算MSE
        mse = F.mse_loss(pred_01, target_01)
        
        # 避免log(0)，设置最小MSE值
        mse = torch.clamp(mse, min=1e-10)
        
        # PSNR计算，最大像素值为1.0
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # 限制PSNR范围，避免异常值
        psnr = torch.clamp(psnr, 0.0, 100.0)
        
        return psnr


def create_trainer(model_config: Dict[str, Any], 
                  training_config: Dict[str, Any],
                  teacher_model_path: Optional[str] = None,
                  full_config: Optional[Dict[str, Any]] = None) -> FrameInterpolationTrainer:
    """
    创建训练器实例
    
    Args:
        model_config: 模型配置
        training_config: 训练配置
        teacher_model_path: 教师模型路径（用于知识蒸馏）
        full_config: 完整配置（包含监控设置）
    
    Returns:
        trainer: 训练器实例
    """
    trainer = FrameInterpolationTrainer(
        model_config=model_config,
        training_config=training_config,
        teacher_model_path=teacher_model_path,
        full_config=full_config
    )
    
    print(f"=== Frame Interpolation Trainer ===")
    print(f"Student Model Parameters: {trainer.student_model.get_parameter_count():,}")
    print(f"Teacher Model: {'Loaded' if trainer.teacher_model else 'None'}")
    print(f"Learning Rate: {trainer.learning_rate}")
    print(f"Weight Decay: {trainer.weight_decay}")
    
    return trainer


if __name__ == "__main__":
    # 测试训练器
    model_config = {
        'input_channels': 7,
        'output_channels': 3
    }
    
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'loss_weights': {
            'reconstruction': 1.0,
            'perceptual': 0.1,
            'temporal_consistency': 0.05
        }
    }
    
    trainer = create_trainer(model_config, training_config)
    print("Training framework test completed successfully!")