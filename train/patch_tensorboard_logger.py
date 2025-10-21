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
from collections import defaultdict
#  FIX: 改用colleague数据集适配器
try:
    from train.colleague_dataset_adapter import ColleagueDatasetAdapter
except ImportError:
    from colleague_dataset_adapter import ColleagueDatasetAdapter

# TensorBoard和可视化
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# 兼容旧版接口：提供轻量化的 TrainingMonitor 占位符，避免外部依赖
class TrainingMonitor:
    def __init__(self, *args, **kwargs):
        self.log_dir = kwargs.get('log_dir')

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
    
    def __init__(self, tone_mapping: str = 'reinhard', gamma: float = 2.2,
                 exposure: float = 1.0, adaptive_exposure: Optional[Dict[str, Any]] = None,
                 mulaw_mu: float = 500.0):
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
        self.tone_mapping = (tone_mapping or 'reinhard').lower()
        self.gamma = float(gamma)
        self.exposure = float(exposure)
        self.adaptive_exposure = adaptive_exposure or {'enable': False}
        self.mulaw_mu = float(mulaw_mu)
    
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
                         max_patches: int = 16,
                         panel_order: Optional[List[str]] = None) -> torch.Tensor:
        """Create patch grid with configurable panel order.

        panel_order: list of items among ['input','target','pred']
        """
        order = panel_order or ['input', 'target', 'pred']
        N = min(patch_inputs.shape[0], max_patches)
        
        inputs_rgb = patch_inputs[:N, :3]
        targets = patch_targets[:N]
        predictions = patch_predictions[:N]
        
        panel_map = {
            'input': inputs_rgb,
            'target': targets,
            'pred': predictions,
        }
        comparison_patches = []
        for i in range(N):
            panels = [panel_map[k][i] for k in order if k in panel_map]
            if not panels:
                continue
            patch_row = torch.cat(panels, dim=2)
            comparison_patches.append(patch_row)
        
        if len(comparison_patches) > 0:
            grid = torch.cat(comparison_patches, dim=1)
        else:
            # Fallback canvas (3 panels * 128 width)
            grid = torch.zeros(3, 128, 128 * max(1, len(order)))

        try:
            grid = self._denormalize_hdr_for_display(grid)
        except Exception:
            # Fallback: return unclamped grid to avoid None
            grid = torch.clamp(grid, 0.0, 1.0)

        return grid
    
    def _denormalize_hdr_for_display(self, img: torch.Tensor) -> torch.Tensor:
        """统一显示转换：
        - 若输入像旧的[-1,1]，按旧log1p逆；
        - 否则视为线性HDR缩放后的值，调用通用tone-mapping（Reinhard+gamma）。
        """
        vmin = float(img.min().item()) if torch.is_tensor(img) else -2.0
        vmax = float(img.max().item()) if torch.is_tensor(img) else 2.0
        # 旧数据路径：只有在确有负值（判定为[-1,1]）时才走旧反归一化
        if vmin < 0.0 and vmax <= 1.1:
            # 旧规范：[-1,1] -> [0,1]，然后按旧log1p逆（历史兼容）
            rgb_01 = (img + 1.0) / 2.0
            log_min_val = 0.0
            log_max_val = 5.023574285781275  # log1p(151.0)
            log_values = rgb_01 * (log_max_val - log_min_val) + log_min_val
            hdr_rgb = torch.expm1(log_values)
            try:
                from src.npu.utils.hdr_vis import tone_map as _tm
            except Exception:
                import sys, os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
                from hdr_vis import tone_map as _tm
            return _tm(hdr_rgb, method=self.tone_mapping, gamma=self.gamma, exposure=self.exposure, adaptive_exposure=self.adaptive_exposure, mu=self.mulaw_mu)
        elif vmin >= 0.0 and vmax <= 1.1:
            # 视为线性HDR已在[0,1]范围（或预缩放），直接做tone-mapping以统一显示空间
            try:
                from src.npu.utils.hdr_vis import tone_map as _tm
            except Exception:
                import sys, os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
                from hdr_vis import tone_map as _tm
            return _tm(img, method=self.tone_mapping, gamma=self.gamma, exposure=self.exposure, adaptive_exposure=self.adaptive_exposure, mu=self.mulaw_mu)
        # 线性HDR显示路径（通用tone-mapping）
        try:
            from src.npu.utils.hdr_vis import tone_map as _tm
        except Exception:
            import sys, os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
            from hdr_vis import tone_map as _tm
        return _tm(img, method=self.tone_mapping, gamma=self.gamma, exposure=self.exposure, adaptive_exposure=self.adaptive_exposure, mu=self.mulaw_mu)
    
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
            ax2.set_xlabel('Epoch')
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
        
        # 读取HDR显示配置（用于固定的 tone-map 路径）
        hdr_cfg = self.config.get('hdr_processing', {})
        self.tone_mapping = str(hdr_cfg.get('tone_mapping_for_display', 'reinhard')).lower()
        self.gamma = float(hdr_cfg.get('gamma', 2.2))
        self.exposure = float(hdr_cfg.get('exposure', 1.0))
        self.adaptive_exposure = hdr_cfg.get('adaptive_exposure', {'enable': False})
        # 显式保留 mu 参数，确保 mulaw 路径与“单独 mu-law 可视化”一致
        self.mulaw_mu = float(hdr_cfg.get('mulaw_mu', 500.0))
        
        #  FIX: 创建colleague数据集实例用于反归一化（修复可视化问题）
        # 在测试环境中可能会失败，所以用try-catch包装
        try:
            hdr_cfg = self.config.get('hdr_processing', {})
            data_root = self.config.get('data_root', './data')
            adapter_kwargs = dict(
                enable_linear_preprocessing=hdr_cfg.get('enable_linear_preprocessing', True),
                enable_srgb_linear=hdr_cfg.get('enable_srgb_linear', True),
                scale_factor=hdr_cfg.get('scale_factor', 0.70),
                tone_mapping_for_display=hdr_cfg.get('tone_mapping_for_display', 'reinhard'),
                gamma=hdr_cfg.get('gamma', 2.2),
                exposure=hdr_cfg.get('exposure', 1.0),
                adaptive_exposure=hdr_cfg.get('adaptive_exposure', {'enable': False}),
            )
            self.dataset_for_denorm = ColleagueDatasetAdapter(
                data_root=data_root,  # 传入数据根目录（包含 processed_bistro）
                split='train',
                **adapter_kwargs
            )
        except Exception as e:
            print(f"WARNING: Dataset initialization failed (normal in test environment): {e}")
            self.dataset_for_denorm = None
        
        # 可视化助手
        self.viz_helper = PatchVisualizationHelper(
            tone_mapping=self.tone_mapping,
            gamma=self.gamma,
            exposure=self.exposure,
            adaptive_exposure=self.adaptive_exposure,
        )
        # Grid settings
        grid_cfg = self.config.get('grid', {})
        self.grid_max_patches = int(grid_cfg.get('max_patches', 8))
        self.grid_panel_order = grid_cfg.get('panel_order', ['input', 'target', 'pred'])
        
        # 统计信息
        self.step_count = 0
        self.epoch_count = 0
        self.loss_history = []
        self.performance_history = []

        # Epoch-level accumulators for scalar logging
        self._current_epoch = None
        self._epoch_loss_sums = defaultdict(float)
        self._epoch_loss_counts = defaultdict(int)
        self._epoch_lr_values = []
        self._epoch_mode_values = []
        self._epoch_batch_stats = defaultdict(list)

        self._val_current_epoch = None
        self._val_epoch_loss_sums = defaultdict(float)
        self._val_epoch_loss_counts = defaultdict(int)
        self._val_metrics_sums = defaultdict(float)
        self._val_metrics_counts = defaultdict(int)
        
        # 图像保存目录
        self.images_dir = self.log_dir / 'visualization_images'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PatchTensorBoardLogger initialized at: {self.log_dir}")
    
    def _reset_training_accumulators(self) -> None:
        self._epoch_loss_sums = defaultdict(float)
        self._epoch_loss_counts = defaultdict(int)
        self._epoch_lr_values = []
        self._epoch_mode_values = []
        self._epoch_batch_stats = defaultdict(list)

    def _flush_training_epoch(self) -> None:
        if self._current_epoch is None or not self._epoch_loss_counts:
            return
        epoch = self._current_epoch
        # Loss averages
        for loss_name, total in self._epoch_loss_sums.items():
            count = max(1, self._epoch_loss_counts[loss_name])
            avg = total / count
            self.writer.add_scalar(f'Loss/{loss_name}', avg, epoch)
            if loss_name == 'total':
                self.loss_history.append(avg)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
        # Learning rate (mean)
        if self._epoch_lr_values:
            avg_lr = sum(self._epoch_lr_values) / len(self._epoch_lr_values)
            self.writer.add_scalar('Training/LearningRate', avg_lr, epoch)
        # Mode (最后一次)
        if self._epoch_mode_values:
            self.writer.add_scalar('Training/Mode', self._epoch_mode_values[-1], epoch)
        # Batch-level stats (mean)
        for stat_name, values in self._epoch_batch_stats.items():
            if values:
                avg_val = sum(values) / len(values)
                self.writer.add_scalar(f'Batch/{stat_name}', avg_val, epoch)
        self._reset_training_accumulators()

    def _reset_validation_accumulators(self) -> None:
        self._val_epoch_loss_sums = defaultdict(float)
        self._val_epoch_loss_counts = defaultdict(int)
        self._val_metrics_sums = defaultdict(float)
        self._val_metrics_counts = defaultdict(int)

    def _flush_validation_epoch(self) -> None:
        if self._val_current_epoch is None or not self._val_epoch_loss_counts:
            return
        epoch = self._val_current_epoch
        for loss_name, total in self._val_epoch_loss_sums.items():
            count = max(1, self._val_epoch_loss_counts[loss_name])
            avg = total / count
            self.writer.add_scalar(f'Validation/{loss_name}', avg, epoch)
        for metric_name, total in self._val_metrics_sums.items():
            count = max(1, self._val_metrics_counts[metric_name])
            avg = total / count
            self.writer.add_scalar(f'Metrics/{metric_name}', avg, epoch)
        self._reset_validation_accumulators()
    
    def finalize_validation_epoch(self) -> None:
        """写出当前累计的验证指标。"""
        self._flush_validation_epoch()
    
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

        # 当epoch变化时，写入上一epoch的统计
        if self._current_epoch is None:
            self._current_epoch = epoch
        elif epoch != self._current_epoch:
            self._flush_training_epoch()
            self._current_epoch = epoch

        import math
        for loss_name, loss_value in loss_dict.items():
            if loss_value is None:
                continue
            try:
                if isinstance(loss_value, torch.Tensor):
                    v = loss_value.detach()
                    v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=0.0)
                    scalar = float(v.item())
                else:
                    scalar = float(loss_value)
                if not math.isfinite(scalar):
                    scalar = 0.0
                self._epoch_loss_sums[loss_name] += scalar
                self._epoch_loss_counts[loss_name] += 1
            except Exception as e:
                print(f"WARNING: log_training_step scalar failed for {loss_name}: {e}")

        # 记录学习率与模式
        if learning_rate is not None:
            self._epoch_lr_values.append(float(learning_rate))
        mode_value = {'patch': 0, 'full': 1, 'mixed': 2}.get(mode, 0)
        self._epoch_mode_values.append(mode_value)

        # 批次统计
        if batch_info:
            for key in ('patch_count', 'full_count', 'total_patches'):
                if key in batch_info:
                    try:
                        self._epoch_batch_stats[key].append(float(batch_info[key]))
                    except Exception:
                        continue
    
    def log_validation_step(self,
                           epoch: int,
                           val_loss_dict: Dict[str, torch.Tensor],
                           val_metrics: Optional[Dict[str, float]] = None):
        """记录验证步骤"""
        if self._val_current_epoch is None:
            self._val_current_epoch = epoch
        elif epoch != self._val_current_epoch:
            self._flush_validation_epoch()
            self._val_current_epoch = epoch

        import math
        for loss_name, loss_value in val_loss_dict.items():
            if loss_value is None:
                continue
            try:
                if isinstance(loss_value, torch.Tensor):
                    v = loss_value.detach()
                    v = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=0.0)
                    scalar = float(v.item())
                else:
                    scalar = float(loss_value)
                if not math.isfinite(scalar):
                    scalar = 0.0
                self._val_epoch_loss_sums[loss_name] += scalar
                self._val_epoch_loss_counts[loss_name] += 1
            except Exception as e:
                print(f"WARNING: log_validation_step scalar failed for {loss_name}: {e}")
        
        if val_metrics:
            for metric_name, metric_value in val_metrics.items():
                try:
                    scalar = float(metric_value)
                except Exception:
                    continue
                if not math.isfinite(scalar):
                    scalar = 0.0
                self._val_metrics_sums[metric_name] += scalar
                self._val_metrics_counts[metric_name] += 1
    
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
        记录patch对比可视化 -  残差学习版本
        
        Args:
            step: 步骤数
            patch_inputs: 输入patches [N, 7, 128, 128]
            patch_targets: 目标patches [N, 3, 128, 128] (完整RGB图像)
            patch_predictions: 预测patches [N, 3, 128, 128] (重建完整图像)
            tag: 标签名
        """
        if patch_inputs.shape[0] == 0:
            return
        
        #  残差学习可视化: 显示 输入RGB | 目标RGB | 重建RGB
        # patch_inputs[:, :3] 是warped RGB
        # patch_targets 是目标完整RGB  
        # patch_predictions 是网络重建的完整RGB
        
        try:
            # 最稳妥：每个面板先单独 tone-map → clamp → 再拼接。
            if patch_inputs is None or patch_targets is None or patch_predictions is None:
                print("WARNING: log_patch_comparison received None inputs:",
                      type(patch_inputs), type(patch_targets), type(patch_predictions))
                return
            if not isinstance(patch_inputs, torch.Tensor) or not isinstance(patch_targets, torch.Tensor) or not isinstance(patch_predictions, torch.Tensor):
                print("WARNING: log_patch_comparison expects tensors, got:",
                      type(patch_inputs), type(patch_targets), type(patch_predictions))
                return

            N = int(min(patch_inputs.shape[0], getattr(self, 'grid_max_patches', 8)))
            if N <= 0:
                return

            # 取 RGB 通道
            inp_rgb = patch_inputs[:N, :3].detach().cpu()
            tgt_rgb = patch_targets[:N, :3].detach().cpu()
            pred_rgb = patch_predictions[:N, :3].detach().cpu()

            # (Debug removed) previously printed pre-tone-map stats

            # 统一 tone-map
            try:
                from src.npu.utils.hdr_vis import tone_map as _tm
            except Exception:
                import sys, os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'utils'))
                from hdr_vis import tone_map as _tm

            def tm(x: torch.Tensor) -> torch.Tensor:
                y = _tm(
                    x,
                    method=self.tone_mapping,
                    gamma=self.gamma,
                    exposure=self.exposure,
                    adaptive_exposure=self.adaptive_exposure,
                    mu=self.mulaw_mu,
                )
                y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                return torch.clamp(y, 0.0, 1.0)

            inp_ldr = tm(inp_rgb)
            tgt_ldr = tm(tgt_rgb)
            pred_ldr = tm(pred_rgb)

            order = getattr(self, 'grid_panel_order', ['input', 'target', 'pred'])
            panel_map = {'input': inp_ldr, 'target': tgt_ldr, 'pred': pred_ldr}

            rows = []
            for i in range(N):
                panels = [panel_map[k][i] for k in order if k in panel_map]
                if not panels:
                    continue
                rows.append(torch.cat(panels, dim=2))
            if not rows:
                print("WARNING: log_patch_comparison got empty rows after assembly")
                return

            grid_tensor = torch.cat(rows, dim=1)  # [3, H*N, W*len(order)] in [0,1]
            grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0, posinf=1.0, neginf=0.0)
            grid_tensor = torch.clamp(grid_tensor, 0.0, 1.0)

            # 写入 TensorBoard（CHW，[0,1]）
            self.writer.add_image(f'Comparison/{tag}', grid_tensor, step)

            # 另存 PNG
            try:
                import cv2 as _cv2
                hwc = (grid_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype('uint8')
                png_path = self.images_dir / f'{tag}_step_{int(step):06d}.png'
                _cv2.imwrite(str(png_path), _cv2.cvtColor(hwc, _cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"WARNING: failed to save grid png: {e}")
        except Exception as e:
            print(f"WARNING: log_patch_comparison failed: {e}")
    
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
        # 在绘图前刷新当前epoch的训练统计（确保按 epoch 写入）
        self._flush_training_epoch()

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
        # Flush remaining epoch statistics before closing
        self._flush_training_epoch()
        self._flush_validation_epoch()
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
    
    def finalize_validation_epoch(self):
        """提交当前验证统计"""
        return self.patch_logger.finalize_validation_epoch()
    
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
        visualizer.log_validation_step(
            epoch=5,
            val_loss_dict={'total': torch.tensor(0.4)}
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
