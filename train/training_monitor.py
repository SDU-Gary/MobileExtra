#!/usr/bin/env python3
"""
@file training_monitor.py
@brief 高级训练监控和日志系统

功能：
- TensorBoard高级记录（梯度、权重、图像）
- 结构化文件日志输出
- 内存和性能监控
- 训练图像可视化保存
- 移动端特化监控（参数效率、量化准备）

@author AI算法团队
@date 2025-08-16
@version 1.0
"""

import os
import time
import psutil
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


class FileLoggerManager:
    """文件日志管理器 - 结构化日志输出"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建多个日志记录器
        self.loggers = {}
        self._setup_loggers()
        
    def _setup_loggers(self):
        """设置多个分类日志记录器"""
        log_config = self.config.get('file_logging', {})
        log_files = log_config.get('log_files', {})
        
        for log_type, filename in log_files.items():
            logger = logging.getLogger(f'train_{log_type}')
            logger.setLevel(getattr(logging, log_config.get('log_level', 'INFO')))
            
            # 清除已有handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # 文件handler
            file_path = self.log_dir / filename
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=self._parse_size(log_config.get('max_file_size', '50MB')),
                backupCount=log_config.get('backup_count', 5),
                encoding='utf-8'
            )
            
            # 控制台handler
            console_handler = logging.StreamHandler()
            
            # 格式化器
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            # 只有training日志输出到控制台，避免重复
            if log_type == 'training':
                logger.addHandler(console_handler)
                
            logger.propagate = False
            self.loggers[log_type] = logger
    
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串为字节数"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def log(self, log_type: str, level: str, message: str, extra_data: Optional[Dict] = None):
        """记录日志"""
        if log_type not in self.loggers:
            log_type = 'training'  # 默认类型
            
        logger = self.loggers[log_type]
        
        # 格式化消息
        if extra_data:
            extra_str = " | ".join([f"{k}={v}" for k, v in extra_data.items()])
            message = f"{message} | {extra_str}"
        
        getattr(logger, level.lower())(message)


class PerformanceTracker:
    """性能和资源监控器"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置计时器"""
        self.start_time = time.time()
        self.step_times = []
        self.epoch_start_time = None
        
    def start_epoch(self):
        """开始epoch计时"""
        self.epoch_start_time = time.time()
        
    def start_step(self):
        """开始step计时"""
        self.step_start_time = time.time()
        
    def end_step(self):
        """结束step计时"""
        if hasattr(self, 'step_start_time'):
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            return step_time
        return 0.0
    
    def get_epoch_time(self) -> float:
        """获取epoch耗时"""
        if self.epoch_start_time:
            return time.time() - self.epoch_start_time
        return 0.0
    
    def get_average_step_time(self) -> float:
        """获取平均step耗时"""
        if self.step_times:
            return sum(self.step_times) / len(self.step_times)
        return 0.0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        memory_info = {}
        
        # CPU内存
        cpu_memory = psutil.virtual_memory()
        memory_info['cpu_used_gb'] = cpu_memory.used / (1024**3)
        memory_info['cpu_total_gb'] = cpu_memory.total / (1024**3)
        memory_info['cpu_percent'] = cpu_memory.percent
        
        # GPU内存
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            memory_info['gpu_allocated_gb'] = gpu_memory_allocated
            memory_info['gpu_reserved_gb'] = gpu_memory_reserved
            memory_info['gpu_total_gb'] = gpu_memory_total
            memory_info['gpu_percent'] = (gpu_memory_allocated / gpu_memory_total) * 100
        
        return memory_info


class TrainingMonitor:
    """高级训练监控器 - 集成TensorBoard和文件日志"""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module):
        self.config = config
        self.model = model
        self.monitoring_config = config.get('advanced_monitoring', {})
        
        # 初始化组件
        self.file_logger = FileLoggerManager(config)
        self.performance_tracker = PerformanceTracker()
        
        # TensorBoard writer - 协调PyTorch Lightning日志器
        self.tb_writer = None
        self.use_lightning_logger = True  # 默认使用Lightning的日志器
        
        # 只有在需要额外详细监控时才创建独立writer
        if (config.get('advanced_monitoring', {}).get('enabled', True) and 
            not self.use_lightning_logger):
            log_dir = Path(config['log_dir']) / 'mobile_inpainting' / f"v{config.get('version', '1.0')}"
            self.tb_writer = SummaryWriter(log_dir=str(log_dir / 'detailed_metrics'))
        
        # Lightning日志器引用（将在set_lightning_logger中设置）
        self.lightning_logger = None
        
        # 监控状态
        self.global_step = 0
        self.current_epoch = 0
        self.gradient_norms = []
        self.weight_updates = {}
        
        # 图像保存目录
        self.image_save_dir = Path(config['log_dir']) / 'images'
        self.image_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_training("监控系统初始化完成", {
            'use_lightning_logger': self.use_lightning_logger,
            'independent_tb_writer': self.tb_writer is not None,
            'monitoring_enabled': self.monitoring_config.get('enabled', False)
        })
    
    def set_lightning_logger(self, lightning_logger):
        """设置PyTorch Lightning的日志器引用"""
        self.lightning_logger = lightning_logger
        if lightning_logger and hasattr(lightning_logger, 'experiment'):
            self.log_training("Lightning TensorBoard日志器已连接", {
                'log_dir': str(lightning_logger.log_dir) if hasattr(lightning_logger, 'log_dir') else 'unknown'
            })
    
    def log_training(self, message: str, extra_data: Optional[Dict] = None):
        """记录训练日志"""
        self.file_logger.log('training', 'info', message, extra_data)
    
    def log_monitoring(self, message: str, extra_data: Optional[Dict] = None):
        """记录监控日志"""
        self.file_logger.log('monitoring', 'info', message, extra_data)
    
    def log_performance(self, message: str, extra_data: Optional[Dict] = None):
        """记录性能日志"""
        self.file_logger.log('performance', 'info', message, extra_data)
    
    def log_error(self, message: str, extra_data: Optional[Dict] = None):
        """记录错误日志"""
        self.file_logger.log('errors', 'error', message, extra_data)
    
    def start_epoch(self, epoch: int):
        """开始epoch监控"""
        self.current_epoch = epoch
        self.performance_tracker.start_epoch()
        self.gradient_norms.clear()
        
        self.log_training(f"开始训练Epoch {epoch}")
    
    def start_step(self):
        """开始step监控"""
        self.performance_tracker.start_step()
    
    def end_step(self, loss_dict: Dict[str, float], learning_rate: float):
        """结束step监控"""
        step_time = self.performance_tracker.end_step()
        self.global_step += 1
        
        # 记录基础训练指标到TensorBoard
        log_config = self.config.get('tensorboard', {})
        should_log = self.global_step % log_config.get('log_every_n_steps', 50) == 0
        
        # 优先使用Lightning日志器，避免重复记录
        if should_log and self.use_lightning_logger and self.lightning_logger:
            try:
                # 使用Lightning的experiment (TensorBoard SummaryWriter)
                tb_logger = self.lightning_logger.experiment
                
                # 记录详细监控指标（Lightning不会记录的）
                tb_logger.add_scalar('detailed/step_time', step_time, self.global_step)
                tb_logger.add_scalar('detailed/global_step_rate', 1.0/step_time if step_time > 0 else 0, self.global_step)
                
            except Exception as e:
                self.log_error(f"Lightning TensorBoard记录失败: {str(e)}")
        
        # 备用：使用独立TensorBoard writer
        elif should_log and self.tb_writer:
            # 基础损失记录到TensorBoard
            for loss_name, loss_value in loss_dict.items():
                self.tb_writer.add_scalar(f'train/{loss_name}', loss_value, self.global_step)
            
            self.tb_writer.add_scalar('train/learning_rate', learning_rate, self.global_step)
            self.tb_writer.add_scalar('train/step_time', step_time, self.global_step)
        
        # 性能监控
        if (self.monitoring_config.get('performance_monitoring', {}).get('enabled') and
            self.global_step % self.monitoring_config['performance_monitoring'].get('log_every_n_steps', 200) == 0):
            self._log_performance_metrics(step_time)
    
    def monitor_gradients(self):
        """监控梯度信息"""
        grad_config = self.monitoring_config.get('gradient_monitoring', {})
        if not grad_config.get('enabled', False):
            return
        
        if self.global_step % grad_config.get('log_every_n_steps', 100) != 0:
            return
        
        total_norm = 0.0
        param_count = 0
        gradient_dict = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # 记录各层梯度范数
                layer_name = name.split('.')[0]  # 获取层名
                if layer_name not in gradient_dict:
                    gradient_dict[layer_name] = []
                gradient_dict[layer_name].append(param_norm.item())
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            # TensorBoard记录梯度信息
            tb_logger = None
            if self.use_lightning_logger and self.lightning_logger:
                try:
                    tb_logger = self.lightning_logger.experiment
                except Exception:
                    tb_logger = self.tb_writer
            else:
                tb_logger = self.tb_writer
            
            if tb_logger:
                tb_logger.add_scalar('gradients/global_norm', total_norm, self.global_step)
                
                # 各层梯度范数
                for layer_name, norms in gradient_dict.items():
                    avg_norm = sum(norms) / len(norms)
                    tb_logger.add_scalar(f'gradients/{layer_name}_norm', avg_norm, self.global_step)
                
                # 梯度分布直方图
                if grad_config.get('track_gradient_distribution', False):
                    all_grads = torch.cat([
                        param.grad.data.flatten() 
                        for param in self.model.parameters() 
                        if param.grad is not None
                    ])
                    tb_logger.add_histogram('gradients/distribution', all_grads, self.global_step)
            
            # 文件日志记录
            self.log_monitoring(f"梯度监控 Step {self.global_step}", {
                'global_norm': f'{total_norm:.6f}',
                'param_count': param_count,
                'avg_layer_norms': {k: f'{sum(v)/len(v):.6f}' for k, v in gradient_dict.items()}
            })
    
    def monitor_weights(self):
        """监控权重信息"""
        weight_config = self.monitoring_config.get('weight_monitoring', {})
        if not weight_config.get('enabled', False):
            return
        
        if self.current_epoch == 0 or self.current_epoch % weight_config.get('log_every_n_epochs', 1) != 0:
            return
        
        weight_stats = {}
        param_count_total = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weight_data = param.data
                
                # 权重统计
                stats = {
                    'mean': weight_data.mean().item(),
                    'std': weight_data.std().item(),
                    'min': weight_data.min().item(),
                    'max': weight_data.max().item(),
                    'shape': list(weight_data.shape),
                    'param_count': weight_data.numel()
                }
                
                weight_stats[name] = stats
                param_count_total += weight_data.numel()
                
                # TensorBoard权重分布
                if self.tb_writer and weight_config.get('track_weight_distribution', False):
                    self.tb_writer.add_histogram(f'weights/{name}', weight_data, self.current_epoch)
                    self.tb_writer.add_scalar(f'weights/{name}_mean', stats['mean'], self.current_epoch)
                    self.tb_writer.add_scalar(f'weights/{name}_std', stats['std'], self.current_epoch)
        
        # 参数总数监控（移动端3M预算）
        param_count_m = param_count_total / 1e6
        budget_usage = (param_count_total / 3e6) * 100  # 3M参数预算
        
        if self.tb_writer:
            self.tb_writer.add_scalar('model/param_count_millions', param_count_m, self.current_epoch)
            self.tb_writer.add_scalar('model/budget_usage_percent', budget_usage, self.current_epoch)
        
        # 文件日志记录
        self.log_monitoring(f"权重监控 Epoch {self.current_epoch}", {
            'total_params': f'{param_count_m:.2f}M',
            'budget_usage': f'{budget_usage:.1f}%',
            'layers_monitored': len(weight_stats)
        })
    
    def save_training_images(self, input_batch: torch.Tensor, output_batch: torch.Tensor, 
                           target_batch: torch.Tensor, prefix: str = "train"):
        """保存训练图像对比"""
        image_config = self.monitoring_config.get('image_visualization', {})
        if not image_config.get('enabled', False):
            return
        
        # 检查保存频率
        save_freq = image_config.get('save_every_n_epochs', 5)
        if self.current_epoch % save_freq != 0:
            return
        
        max_samples = min(image_config.get('max_samples_per_epoch', 4), input_batch.size(0))
        save_individual = image_config.get('save_individual_images', True)  # 新增选项
        
        try:
            # 选择样本
            indices = torch.randperm(input_batch.size(0))[:max_samples]
            input_samples = input_batch[indices]
            output_samples = output_batch[indices]
            target_samples = target_batch[indices]
            
            # 处理7通道输入 - 只显示前3通道(RGB)
            if input_samples.size(1) == 7:
                input_rgb = input_samples[:, :3]  # 外推RGB
                holes_mask = input_samples[:, 3:4]  # 语义空洞掩码
                occlusion_mask = input_samples[:, 4:5]  # 遮挡掩码
                # residual_mv = input_samples[:, 5:7]  # 残差MV不可视化
            else:
                input_rgb = input_samples
                holes_mask = None
                occlusion_mask = None
            
            # HDR图像转换为显示格式 - 使用Reinhard色调映射
            def hdr_to_display(tensor):
                """
                将HDR图像转换为可显示的LDR图像
                使用Reinhard色调映射算法
                """
                tensor = tensor.clone()
                # 确保数据为正值
                tensor = torch.clamp(tensor, min=0.0)
                
                # Reinhard色调映射: x / (1 + x)
                # 对于非常亮的像素，这会将它们压缩到[0,1]范围
                tone_mapped = tensor / (1.0 + tensor)
                
                # 应用伽马校正以适应显示器
                gamma = 2.2
                tone_mapped = torch.pow(tone_mapped, 1.0 / gamma)
                
                # 确保在[0,1]范围内
                return torch.clamp(tone_mapped, 0.0, 1.0)
            
            input_rgb = hdr_to_display(input_rgb)
            output_norm = hdr_to_display(output_samples)
            target_norm = hdr_to_display(target_samples)
            
            # 创建对比图像网格
            comparison_images = []
            
            for i in range(max_samples):
                row_images = [input_rgb[i], output_norm[i], target_norm[i]]
                
                # 添加掩码可视化（如果存在）
                if holes_mask is not None:
                    holes_vis = holes_mask[i].expand(3, -1, -1)  # 转为3通道显示
                    row_images.append(holes_vis)
                    
                if occlusion_mask is not None:
                    occlusion_vis = occlusion_mask[i].expand(3, -1, -1)
                    row_images.append(occlusion_vis)
                
                comparison_images.extend(row_images)
            
            if save_individual:
                # 保存单张完整图像（网络输出）
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for i in range(max_samples):
                    # 保存输出图像
                    output_filename = f"{prefix}_output_epoch_{self.current_epoch:03d}_sample_{i}_{timestamp}.png"
                    output_filepath = self.image_save_dir / output_filename
                    vutils.save_image(output_norm[i], output_filepath)
                    
                    # 保存输入图像（带空洞的外推图）
                    input_filename = f"{prefix}_input_epoch_{self.current_epoch:03d}_sample_{i}_{timestamp}.png"
                    input_filepath = self.image_save_dir / input_filename
                    vutils.save_image(input_rgb[i], input_filepath)
                    
                    # 保存目标图像
                    target_filename = f"{prefix}_target_epoch_{self.current_epoch:03d}_sample_{i}_{timestamp}.png"
                    target_filepath = self.image_save_dir / target_filename
                    vutils.save_image(target_norm[i], target_filepath)
            
            # 可选：继续保存对比网格图
            if image_config.get('save_comparison_grid', True):
                # 创建网格
                cols = 5 if holes_mask is not None and occlusion_mask is not None else 3
                grid = vutils.make_grid(comparison_images, nrow=cols, padding=2, normalize=False)
                
                # 保存到文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_grid_epoch_{self.current_epoch:03d}_{timestamp}.png"
                filepath = self.image_save_dir / filename
                
                vutils.save_image(grid, filepath)
                
                # TensorBoard记录
                if self.tb_writer:
                    self.tb_writer.add_image(f'{prefix}/comparison', grid, self.current_epoch)
            
            # 日志记录
            self.log_training(f"保存训练图像 Epoch {self.current_epoch}", {
                'samples': max_samples,
                'file': filename,
                'input_shape': str(input_samples.shape),
                'channels': f"{cols}_column_layout"
            })
            
        except Exception as e:
            self.log_error(f"保存训练图像失败: {str(e)}")
    
    def _log_performance_metrics(self, step_time: float):
        """记录性能指标"""
        memory_info = self.performance_tracker.get_memory_usage()
        avg_step_time = self.performance_tracker.get_average_step_time()
        
        # TensorBoard记录
        if self.tb_writer:
            self.tb_writer.add_scalar('performance/step_time', step_time, self.global_step)
            self.tb_writer.add_scalar('performance/avg_step_time', avg_step_time, self.global_step)
            
            for key, value in memory_info.items():
                self.tb_writer.add_scalar(f'memory/{key}', value, self.global_step)
        
        # 文件日志记录
        self.log_performance(f"性能监控 Step {self.global_step}", {
            'step_time': f'{step_time:.3f}s',
            'avg_step_time': f'{avg_step_time:.3f}s',
            'cpu_memory': f'{memory_info.get("cpu_used_gb", 0):.1f}/{memory_info.get("cpu_total_gb", 0):.1f}GB ({memory_info.get("cpu_percent", 0):.1f}%)',
            'gpu_memory': f'{memory_info.get("gpu_allocated_gb", 0):.1f}GB ({memory_info.get("gpu_percent", 0):.1f}%)'
        })
    
    def end_epoch(self, avg_train_loss: float, avg_val_loss: float, 
                  avg_val_ssim: float, avg_val_psnr: float):
        """结束epoch监控"""
        epoch_time = self.performance_tracker.get_epoch_time()
        
        # TensorBoard记录
        if self.tb_writer:
            self.tb_writer.add_scalar('epoch/train_loss', avg_train_loss, self.current_epoch)
            self.tb_writer.add_scalar('epoch/val_loss', avg_val_loss, self.current_epoch)
            self.tb_writer.add_scalar('epoch/val_ssim', avg_val_ssim, self.current_epoch)
            self.tb_writer.add_scalar('epoch/val_psnr', avg_val_psnr, self.current_epoch)
            self.tb_writer.add_scalar('epoch/epoch_time', epoch_time, self.current_epoch)
        
        # 权重监控
        self.monitor_weights()
        
        # 梯度统计
        if self.gradient_norms:
            avg_grad_norm = sum(self.gradient_norms) / len(self.gradient_norms)
            if self.tb_writer:
                self.tb_writer.add_scalar('epoch/avg_gradient_norm', avg_grad_norm, self.current_epoch)
        
        # 文件日志记录
        self.log_training(f"完成Epoch {self.current_epoch}", {
            'epoch_time': f'{epoch_time:.1f}s',
            'train_loss': f'{avg_train_loss:.6f}',
            'val_loss': f'{avg_val_loss:.6f}',
            'val_ssim': f'{avg_val_ssim:.4f}',
            'val_psnr': f'{avg_val_psnr:.2f}dB',
            'avg_grad_norm': f'{sum(self.gradient_norms)/len(self.gradient_norms):.6f}' if self.gradient_norms else 'N/A'
        })
    
    def close(self):
        """关闭监控器"""
        if self.tb_writer:
            self.tb_writer.close()
        
        self.log_training("训练监控系统关闭")


# 工具函数
def create_training_monitor(config: Dict[str, Any], model: nn.Module) -> Optional[TrainingMonitor]:
    """创建训练监控器"""
    if not config.get('logging', {}).get('advanced_monitoring', {}).get('enabled', False):
        return None
    
    try:
        # 合并配置，包含experiment版本信息
        monitor_config = config['logging'].copy()
        monitor_config['version'] = config.get('experiment', {}).get('version', '1.0')
        return TrainingMonitor(monitor_config, model)
    except Exception as e:
        print(f"创建训练监控器失败: {e}")
        return None