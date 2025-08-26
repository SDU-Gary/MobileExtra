#!/usr/bin/env python3
"""
Patch Training TensorBoard Visualization System

Core features:
1. Patch detection and processing visualization 
2. Training quality comparison (patch vs full)
3. Real-time performance and resource monitoring
4. Mode switching and scheduling visualization 
5. Loss function decomposition display
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import cv2
from train.unified_dataset import UnifiedNoiseBaseDataset

# TensorBoard和可视化
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# 导入现有监控系统
try:
    from training_monitor import TrainingMonitor
except ImportError:
    try:
        from train.training_monitor import TrainingMonitor
    except ImportError:
        # 提供简化的TrainingMonitor占位符
        class TrainingMonitor:
            def __init__(self, log_dir, config):
                self.log_dir = log_dir
                self.config = config
            
            def log_training_progress(self, progress_dict):
                pass

# 导入patch组件
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src', 'npu', 'networks'))

try:
    from src.npu.networks.patch import PatchInfo, PatchPosition
except ImportError:
    try:
        from patch import PatchInfo, PatchPosition
    except ImportError:
        # 直接导入模块文件
        patch_dir = os.path.join(project_root, 'src', 'npu', 'networks', 'patch')
        sys.path.insert(0, patch_dir)
        from hole_detector import PatchInfo
        from patch_extractor import PatchPosition


class PatchVisualizationHelper:
    """Patch visualization helper for hole detection and processing display"""
    
    def __init__(self):
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]
    
    def visualize_hole_detection(self, 
                                original_image: np.ndarray,
                                holes_mask: np.ndarray,
                                patch_infos: List[PatchInfo]) -> np.ndarray:
        """Visualize hole detection results with patch boundaries"""
        # Convert to HWC format
        if len(original_image.shape) == 3 and original_image.shape[0] == 3:
            image = np.transpose(original_image, (1, 2, 0))
        else:
            image = original_image.copy()
        
        # Ensure uint8 range
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        H, W = image.shape[:2]
        vis_image = image.copy()
        
        # Draw hole mask overlay
        hole_overlay = np.zeros_like(vis_image)
        hole_overlay[:, :, 0] = holes_mask * 255
        vis_image = cv2.addWeighted(vis_image, 0.7, hole_overlay, 0.3, 0)
        
        # Draw patch boundaries and centers
        for i, patch_info in enumerate(patch_infos):
            color = self.colors[i % len(self.colors)]
            
            x1 = max(0, patch_info.center_x - 64)
            y1 = max(0, patch_info.center_y - 64)
            x2 = min(W, patch_info.center_x + 64)
            y2 = min(H, patch_info.center_y + 64)
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_image, (patch_info.center_x, patch_info.center_y), 5, color, -1)
            
            label = f"P{patch_info.patch_id}({patch_info.hole_area})"
            cv2.putText(vis_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_image
    
    def create_patch_grid(self, 
                         patch_inputs: torch.Tensor,
                         patch_targets: torch.Tensor,
                         patch_predictions: torch.Tensor,
                         max_patches: int = 16) -> torch.Tensor:
        """Create patch grid: input|target|prediction"""
        N = min(patch_inputs.shape[0], max_patches)
        
        inputs_rgb = patch_inputs[:N, :3]
        targets = patch_targets[:N]
        predictions = patch_predictions[:N]
        
        comparison_patches = []
        for i in range(N):
            patch_row = torch.cat([
                inputs_rgb[i], targets[i], predictions[i]
            ], dim=2)
            comparison_patches.append(patch_row)
        
        if len(comparison_patches) > 0:
            grid = torch.cat(comparison_patches, dim=1)
        else:
            grid = torch.zeros(3, 128, 384)
        
        grid = self._denormalize_hdr_for_display(grid)
        
        return grid
    
    def _denormalize_hdr_for_display(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """HDR denormalization for TensorBoard display"""
        # [-1, 1] -> [0, 1]
        rgb_01 = (normalized_tensor + 1.0) / 2.0
        
        # [0, 1] -> log space [0, 5.024]
        log_min_val = 0.0
        log_max_val = 5.023574285781275  # log1p(151.0)
        log_values = rgb_01 * (log_max_val - log_min_val) + log_min_val
        
        # Log space -> HDR space
        hdr_rgb = torch.expm1(log_values)
        
        # HDR -> LDR tone mapping
        ldr_rgb = hdr_rgb / (1.0 + hdr_rgb)
        display_rgb = torch.clamp(ldr_rgb, 0.0, 1.0)
        
        return display_rgb
    
    def visualize_training_progress(self,
                                   epoch: int,
                                   patch_stats: Dict[str, Any],
                                   loss_history: List[float]) -> plt.Figure:
        """Create training progress visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Mode usage statistics
        ax1 = axes[0, 0]
        modes = ['Patch', 'Full', 'Mixed']
        counts = [
            patch_stats.get('patch_mode_count', 0),
            patch_stats.get('full_mode_count', 0),
            patch_stats.get('mixed_steps', 0)
        ]
        ax1.pie(counts, labels=modes, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Training Mode Distribution')
        
        # Loss curve
        ax2 = axes[0, 1]
        if loss_history:
            ax2.plot(loss_history)
            ax2.set_title('Training Loss History')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # Patch statistics
        ax3 = axes[1, 0]
        patch_metrics = {
            'Avg Patches/Image': patch_stats.get('avg_patches_per_image', 0),
            'Total Patches': patch_stats.get('total_patches_generated', 0),
            'Cache Hit Rate': patch_stats.get('cache_hit_rate', 0) * 100
        }
        bars = ax3.bar(patch_metrics.keys(), patch_metrics.values())
        ax3.set_title('Patch Statistics')
        ax3.set_ylabel('Count / Percentage')
        
        # Performance metrics
        ax4 = axes[1, 1]
        if 'patch_ratio' in patch_stats:
            performance_data = {
                'Patch Ratio': patch_stats.get('patch_ratio', 0) * 100,
                'Cache Efficiency': patch_stats.get('cache_hit_rate', 0) * 100,
                'Processing Speed': min(patch_stats.get('processing_speed', 100), 100)
            }
            bars = ax4.bar(performance_data.keys(), performance_data.values())
            ax4.set_title('Performance Metrics')
            ax4.set_ylabel('Percentage')
            ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        return fig


class PatchTensorBoardLogger:
    """
    Patch训练专用TensorBoard日志记录器
    
    功能：
    1. 扩展现有TrainingMonitor
    2. Patch特定的可视化和监控
    3. 实时性能统计
    4. 模式切换分析
    """
    
    def __init__(self, 
                 log_dir: str,
                 config: Optional[Dict[str, Any]] = None,
                 base_monitor: Optional[TrainingMonitor] = None):
        """
        初始化Patch TensorBoard日志记录器
        
        Args:
            log_dir: 日志目录
            config: 配置字典
            base_monitor: 基础训练监控器
        """
        self.log_dir = Path(log_dir)
        self.config = config or {}
        self.base_monitor = base_monitor
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(self.log_dir / 'patch_tensorboard')
        
        # 🔧 创建数据集实例用于反归一化（修复可视化问题）
        # 在测试环境中可能会失败，所以用try-catch包装
        try:
            self.dataset_for_denorm = UnifiedNoiseBaseDataset(
                data_root="./output_motion_fix",  # 这个路径不会被实际使用，只是为了初始化
                split='train',
                augmentation=False
            )
        except Exception as e:
            print(f"WARNING: Dataset initialization failed (normal in test environment): {e}")
            self.dataset_for_denorm = None
        
        # 可视化助手
        self.viz_helper = PatchVisualizationHelper()
        
        # 统计信息
        self.step_count = 0
        self.epoch_count = 0
        self.loss_history = []
        self.performance_history = []
        
        # 图像保存目录
        self.images_dir = self.log_dir / 'visualization_images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PatchTensorBoardLogger initialized at: {self.log_dir}")
    
    def log_training_step(self,
                         step: int,
                         epoch: int,
                         mode: str,
                         loss_dict: Dict[str, torch.Tensor],
                         batch_info: Dict[str, Any],
                         learning_rate: Optional[float] = None):
        """
        记录训练步骤
        
        Args:
            step: 全局步骤数
            epoch: 当前epoch
            mode: 训练模式 ('patch', 'full', 'mixed')
            loss_dict: 损失字典
            batch_info: batch信息
            learning_rate: 学习率
        """
        self.step_count = step
        self.epoch_count = epoch
        
        # 1. 记录损失
        for loss_name, loss_value in loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                self.writer.add_scalar(f'Loss/{loss_name}', loss_value.item(), step)
            else:
                self.writer.add_scalar(f'Loss/{loss_name}', loss_value, step)
        
        # 更新损失历史
        if 'total' in loss_dict:
            total_loss = loss_dict['total']
            if isinstance(total_loss, torch.Tensor):
                self.loss_history.append(total_loss.item())
            else:
                self.loss_history.append(total_loss)
        
        # 保持历史长度
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
        
        # 2. 记录学习率
        if learning_rate is not None:
            self.writer.add_scalar('Training/LearningRate', learning_rate, step)
        
        # 3. 记录模式统计
        mode_value = {'patch': 0, 'full': 1, 'mixed': 2}.get(mode, 0)
        self.writer.add_scalar('Training/Mode', mode_value, step)
        
        # 4. 记录batch信息
        if batch_info:
            if 'patch_count' in batch_info:
                self.writer.add_scalar('Batch/PatchCount', batch_info['patch_count'], step)
            if 'full_count' in batch_info:
                self.writer.add_scalar('Batch/FullCount', batch_info['full_count'], step)
            if 'total_patches' in batch_info:
                self.writer.add_scalar('Batch/TotalPatches', batch_info['total_patches'], step)
    
    def log_validation_step(self,
                           step: int,
                           val_loss_dict: Dict[str, torch.Tensor],
                           val_metrics: Optional[Dict[str, float]] = None):
        """记录验证步骤"""
        # 验证损失
        for loss_name, loss_value in val_loss_dict.items():
            if isinstance(loss_value, torch.Tensor):
                self.writer.add_scalar(f'Validation/{loss_name}', loss_value.item(), step)
            else:
                self.writer.add_scalar(f'Validation/{loss_name}', loss_value, step)
        
        # 验证指标
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, step)
    
    def log_patch_visualization(self,
                               step: int,
                               original_image: torch.Tensor,
                               holes_mask: torch.Tensor,
                               patch_infos: List[PatchInfo],
                               tag: str = 'patch_detection'):
        """
        记录patch检测可视化
        
        Args:
            step: 步骤数
            original_image: 原始图像 [B, C, H, W]
            holes_mask: 空洞掩码 [B, 1, H, W]
            patch_infos: patch信息列表
            tag: 标签名
        """
        if original_image.shape[0] == 0:
            return
        
        # 取第一个sample进行可视化
        img = original_image[0, :3].cpu().numpy()  # [3, H, W] -> [H, W, 3]
        mask = holes_mask[0, 0].cpu().numpy()      # [H, W]
        
        # 创建可视化
        vis_result = self.viz_helper.visualize_hole_detection(img, mask, patch_infos)
        
        # 转换为tensor并添加到TensorBoard
        vis_tensor = torch.from_numpy(vis_result).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        self.writer.add_image(f'Visualization/{tag}', vis_tensor, step)
        
        # 保存到磁盘
        save_path = self.images_dir / f'{tag}_step_{step:06d}.png'
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR))
    
    def log_patch_comparison(self,
                           step: int,
                           patch_inputs: torch.Tensor,
                           patch_targets: torch.Tensor,
                           patch_predictions: torch.Tensor,
                           tag: str = 'patch_comparison'):
        """
        记录patch对比可视化
        
        Args:
            step: 步骤数
            patch_inputs: 输入patches [N, 7, 128, 128]
            patch_targets: 目标patches [N, 3, 128, 128]
            patch_predictions: 预测patches [N, 3, 128, 128]
            tag: 标签名
        """
        if patch_inputs.shape[0] == 0:
            return
        
        # 创建对比网格
        grid_tensor = self.viz_helper.create_patch_grid(
            patch_inputs.cpu(), 
            patch_targets.cpu(), 
            patch_predictions.cpu(),
            max_patches=8  # 最多显示8个patches
        )
        
        # 添加到TensorBoard
        self.writer.add_image(f'Comparison/{tag}', grid_tensor, step)
    
    def log_training_progress(self,
                             step: int,
                             epoch: int,
                             patch_stats: Dict[str, Any]):
        """
        记录训练进度可视化
        
        Args:
            step: 步骤数
            epoch: epoch数
            patch_stats: patch统计信息
        """
        # 创建进度图
        fig = self.viz_helper.visualize_training_progress(
            epoch, patch_stats, self.loss_history
        )
        
        # 添加到TensorBoard
        self.writer.add_figure('Progress/TrainingStats', fig, step)
        plt.close(fig)  # 释放内存
    
    def log_performance_metrics(self,
                               step: int,
                               gpu_memory_mb: float,
                               cpu_usage: float,
                               training_speed_samples_per_sec: float,
                               patch_processing_time_ms: float):
        """记录性能指标"""
        self.writer.add_scalar('Performance/GPU_Memory_MB', gpu_memory_mb, step)
        self.writer.add_scalar('Performance/CPU_Usage', cpu_usage, step)
        self.writer.add_scalar('Performance/Training_Speed_SPS', training_speed_samples_per_sec, step)
        self.writer.add_scalar('Performance/Patch_Processing_Time_MS', patch_processing_time_ms, step)
    
    def log_model_graph(self, model: nn.Module, input_shape: Tuple[int, ...]):
        """记录模型计算图"""
        try:
            dummy_input = torch.randn(1, *input_shape)
            self.writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"无法记录模型计算图: {e}")
    
    def log_histogram(self, step: int, model: nn.Module):
        """记录权重和梯度直方图"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
            self.writer.add_histogram(f'Weights/{name}', param.data, step)
    
    def close(self):
        """关闭日志记录器"""
        if self.writer:
            self.writer.close()
        
        # 如果有基础监控器，也关闭它
        if self.base_monitor:
            # 这里可以调用base_monitor的清理方法
            pass


class PatchTrainingVisualizer:
    """
    Patch训练完整可视化系统
    
    整合多个可视化组件，提供统一接口
    """
    
    def __init__(self, 
                 log_dir: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化可视化系统
        
        Args:
            log_dir: 日志目录
            config: 配置参数
        """
        self.log_dir = Path(log_dir)
        self.config = config or {}
        
        # 创建patch专用logger
        self.patch_logger = PatchTensorBoardLogger(log_dir, config)
        
        # 可视化频率控制
        self.vis_frequency = config.get('visualization_frequency', 100)  # 每100步可视化一次
        self.save_frequency = config.get('save_frequency', 500)          # 每500步保存一次
        
        print(f"PatchTrainingVisualizer initialized at: {log_dir}")
    
    def log_training_step(self, **kwargs):
        """代理训练步骤记录"""
        return self.patch_logger.log_training_step(**kwargs)
    
    def log_validation_step(self, **kwargs):
        """代理验证步骤记录"""
        return self.patch_logger.log_validation_step(**kwargs)
    
    def log_patch_visualization(self, **kwargs):
        """代理patch可视化记录"""
        return self.patch_logger.log_patch_visualization(**kwargs)
    
    def log_patch_comparison(self, **kwargs):
        """代理patch对比记录"""
        return self.patch_logger.log_patch_comparison(**kwargs)
    
    def log_training_progress(self, **kwargs):
        """代理训练进度记录"""
        return self.patch_logger.log_training_progress(**kwargs)
    
    def should_visualize(self, step: int) -> bool:
        """判断是否应该进行可视化"""
        return step % self.vis_frequency == 0
    
    def should_save(self, step: int) -> bool:
        """判断是否应该保存"""
        return step % self.save_frequency == 0
    
    def close(self):
        """关闭可视化系统"""
        self.patch_logger.close()


def create_patch_visualizer(log_dir: str, 
                          config: Optional[Dict[str, Any]] = None) -> PatchTrainingVisualizer:
    """创建patch训练可视化器的工厂函数"""
    return PatchTrainingVisualizer(log_dir, config)


def test_patch_tensorboard_logger():
    """测试Patch TensorBoard日志记录器"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建可视化器
        visualizer = create_patch_visualizer(
            log_dir=temp_dir,
            config={
                'visualization_frequency': 10,
                'save_frequency': 50
            }
        )
        
        # 模拟训练数据
        batch_size = 2
        patch_count = 4
        
        # 测试基础日志记录
        loss_dict = {
            'total': torch.tensor(0.5),
            'patch_l1': torch.tensor(0.3),
            'full_l1': torch.tensor(0.2)
        }
        
        batch_info = {
            'patch_count': patch_count,
            'full_count': batch_size,
            'total_patches': patch_count
        }
        
        visualizer.log_training_step(
            step=100,
            epoch=5,
            mode='mixed',
            loss_dict=loss_dict,
            batch_info=batch_info,
            learning_rate=1e-4
        )
        
        print("SUCCESS: Basic logging test passed")
        
        # 测试可视化
        if visualizer.should_visualize(100):
            # 模拟图像数据
            original_image = torch.randn(1, 7, 270, 480)
            holes_mask = torch.zeros(1, 1, 270, 480)
            holes_mask[0, 0, 50:100, 100:200] = 1
            
            from patch import PatchInfo
            patch_infos = [
                PatchInfo(center_x=150, center_y=75, hole_area=2500, patch_id=0)
            ]
            
            visualizer.log_patch_visualization(
                step=100,
                original_image=original_image,
                holes_mask=holes_mask,
                patch_infos=patch_infos
            )
            
            print("SUCCESS: Patch visualization test passed")
        
        # 测试patch对比
        patch_inputs = torch.randn(4, 7, 128, 128)
        patch_targets = torch.randn(4, 3, 128, 128)
        patch_predictions = torch.randn(4, 3, 128, 128)
        
        visualizer.log_patch_comparison(
            step=100,
            patch_inputs=patch_inputs,
            patch_targets=patch_targets,
            patch_predictions=patch_predictions
        )
        
        print("SUCCESS: Patch comparison visualization test passed")
        
        # 关闭可视化器
        visualizer.close()
        
        print("SUCCESS: All tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # 运行测试
    success = test_patch_tensorboard_logger()
    print(f"\n{'SUCCESS: Patch TensorBoard Logger test passed!' if success else 'ERROR: Test failed!'}")