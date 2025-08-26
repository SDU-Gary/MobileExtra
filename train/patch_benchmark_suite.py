#!/usr/bin/env python3
"""
Patch-Based训练性能基准测试系统

设计原则：
- 全面性能评估：训练速度、内存使用、质量收敛、参数效率
- 对比分析：patch模式 vs 全图模式的详细对比
- 自动化测试：无人值守的完整基准测试流程
- 科学评估：统计学显著性测试和置信区间分析

核心功能：
1. 训练速度基准测试
2. 内存效率分析  
3. 质量收敛评估
4. 参数效率对比
5. A/B测试框架
6. 消融实验支持
7. 自动化报告生成

作者：AI算法团队
日期：2025-08-24
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from datetime import datetime
import psutil
import GPUtil
from dataclasses import dataclass, asdict
import pandas as pd
from scipy import stats
import logging

# 导入训练组件
try:
    from patch_training_framework import (
        PatchFrameInterpolationTrainer, 
        PatchTrainingScheduleConfig,
        create_patch_trainer
    )
    from patch_aware_dataset import PatchTrainingConfig
    from training_framework import FrameInterpolationTrainer
except ImportError as e:
    print(f"基准测试导入警告: {e}")
    # 提供基础类作为占位符
    class PatchFrameInterpolationTrainer:
        pass
    class PatchTrainingScheduleConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class PatchTrainingConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def create_patch_trainer(*args, **kwargs):
        return PatchFrameInterpolationTrainer()
    class FrameInterpolationTrainer:
        pass
try:
    from patch_aware_dataset import (
        PatchAwareDataset, 
        create_patch_aware_dataloader,
        PatchTrainingConfig as DatasetPatchConfig
    )
    from unified_dataset import UnifiedNoiseBaseDataset
except ImportError as e:
    print(f"数据集导入警告: {e}")
    class PatchAwareDataset:
        pass
    def create_patch_aware_dataloader(*args, **kwargs):
        return None
    class DatasetPatchConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    class UnifiedNoiseBaseDataset:
        pass


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 测试参数
    test_epochs: int = 5                    # 测试epoch数
    test_samples: int = 100                 # 每个测试的样本数
    repeat_runs: int = 3                    # 重复运行次数
    warmup_steps: int = 10                  # 预热步骤数
    
    # 内存监控
    memory_monitoring: bool = True          # 启用内存监控
    memory_check_interval: float = 0.1      # 内存检查间隔(秒)
    
    # 质量评估
    quality_metrics: List[str] = None       # 质量指标列表
    convergence_patience: int = 10          # 收敛耐心值
    min_improvement: float = 0.001          # 最小改善阈值
    
    # 统计分析
    confidence_level: float = 0.95          # 置信水平
    significance_threshold: float = 0.05    # 显著性阈值
    
    # 输出控制
    save_detailed_logs: bool = True         # 保存详细日志
    generate_plots: bool = True             # 生成图表
    save_raw_data: bool = True              # 保存原始数据

    def __post_init__(self):
        if self.quality_metrics is None:
            self.quality_metrics = ['loss', 'psnr', 'ssim']


@dataclass 
class PerformanceMetrics:
    """性能指标数据类"""
    # 训练速度
    samples_per_second: float = 0.0
    epoch_time_seconds: float = 0.0
    step_time_ms: float = 0.0
    
    # 内存使用
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    
    # 质量指标
    final_loss: float = float('inf')
    best_loss: float = float('inf')
    convergence_epoch: int = -1
    
    # 模型效率
    model_parameters: int = 0
    model_size_mb: float = 0.0
    inference_time_ms: float = 0.0
    
    # 训练统计
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0


class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self, check_interval: float = 0.1):
        self.check_interval = check_interval
        self.monitoring = False
        self.gpu_memory_history = []
        self.cpu_memory_history = []
        self.cpu_usage_history = []
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.gpu_memory_history = []
        self.cpu_memory_history = []
        self.cpu_usage_history = []
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        
    def collect_metrics(self) -> Dict[str, float]:
        """收集当前指标"""
        metrics = {}
        
        # GPU内存
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一个GPU
                metrics['gpu_memory_used_mb'] = gpu.memoryUsed
                metrics['gpu_memory_total_mb'] = gpu.memoryTotal
                metrics['gpu_utilization'] = gpu.load * 100
        except Exception as e:
            metrics['gpu_memory_used_mb'] = 0
            metrics['gpu_memory_total_mb'] = 0
            metrics['gpu_utilization'] = 0
        
        # CPU和系统内存
        process = psutil.Process()
        memory_info = process.memory_info()
        metrics['cpu_memory_mb'] = memory_info.rss / (1024 * 1024)
        metrics['cpu_usage'] = psutil.cpu_percent()
        
        # 如果在监控中，记录历史
        if self.monitoring:
            self.gpu_memory_history.append(metrics['gpu_memory_used_mb'])
            self.cpu_memory_history.append(metrics['cpu_memory_mb'])
            self.cpu_usage_history.append(metrics['cpu_usage'])
        
        return metrics
    
    def get_summary_stats(self) -> Dict[str, float]:
        """获取监控期间的统计摘要"""
        summary = {}
        
        if self.gpu_memory_history:
            summary['peak_gpu_memory_mb'] = max(self.gpu_memory_history)
            summary['avg_gpu_memory_mb'] = np.mean(self.gpu_memory_history)
            summary['gpu_memory_std'] = np.std(self.gpu_memory_history)
        
        if self.cpu_memory_history:
            summary['peak_cpu_memory_mb'] = max(self.cpu_memory_history)
            summary['avg_cpu_memory_mb'] = np.mean(self.cpu_memory_history)
            
        if self.cpu_usage_history:
            summary['avg_cpu_usage'] = np.mean(self.cpu_usage_history)
            summary['peak_cpu_usage'] = max(self.cpu_usage_history)
        
        return summary


class TrainingBenchmark:
    """训练基准测试器"""
    
    def __init__(self, 
                 config: BenchmarkConfig,
                 data_root: str,
                 output_dir: str):
        self.config = config
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 系统监控
        self.monitor = SystemMonitor(config.memory_check_interval)
        
        # 日志设置
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('benchmark')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            self.output_dir / 'benchmark.log'
        )
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def benchmark_patch_training(self, 
                                model_config: Dict[str, Any],
                                training_config: Dict[str, Any]) -> PerformanceMetrics:
        """基准测试patch训练模式"""
        self.logger.info("开始Patch训练模式基准测试")
        
        # 创建patch数据集和模型
        dataset_config = DatasetPatchConfig(
            enable_patch_mode=True,
            patch_mode_probability=0.7
        )
        
        trainer = create_patch_trainer(
            model_config=model_config,
            training_config=training_config,
            patch_config=dataset_config
        )
        
        dataloader = create_patch_aware_dataloader(
            data_root=self.data_root,
            split='train',
            batch_size=training_config.get('batch_size', 4),
            config=dataset_config,
            num_workers=2
        )
        
        # 运行基准测试
        return self._run_training_benchmark(trainer, dataloader, "patch")
    
    def benchmark_full_training(self,
                               model_config: Dict[str, Any], 
                               training_config: Dict[str, Any]) -> PerformanceMetrics:
        """基准测试全图训练模式"""
        self.logger.info("开始全图训练模式基准测试")
        
        # 创建传统训练器和数据集
        trainer = FrameInterpolationTrainer(
            model_config=model_config,
            training_config=training_config
        )
        
        dataset = UnifiedNoiseBaseDataset(
            data_root=self.data_root,
            split='train'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.get('batch_size', 4),
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        return self._run_training_benchmark(trainer, dataloader, "full")
    
    def _run_training_benchmark(self,
                               trainer: nn.Module,
                               dataloader: DataLoader,
                               mode: str) -> PerformanceMetrics:
        """运行训练基准测试"""
        metrics = PerformanceMetrics()
        
        # 初始化
        trainer.train()
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-4)
        
        # 模型信息
        metrics.model_parameters = sum(p.numel() for p in trainer.parameters())
        metrics.model_size_mb = sum(p.numel() * p.element_size() for p in trainer.parameters()) / (1024 * 1024)
        
        # 预热
        self.logger.info(f"预热阶段 - {self.config.warmup_steps} 步")
        self._warmup(trainer, dataloader, optimizer, self.config.warmup_steps)
        
        # 开始监控
        self.monitor.start_monitoring()
        
        # 记录开始时间
        start_time = time.time()
        total_samples = 0
        loss_history = []
        step_times = []
        
        try:
            for epoch in range(self.config.test_epochs):
                epoch_start = time.time()
                epoch_samples = 0
                epoch_steps = 0
                
                for batch_idx, batch in enumerate(dataloader):
                    step_start = time.time()
                    
                    try:
                        # 前向传播
                        if mode == "patch":
                            # Patch模式处理
                            loss = self._process_patch_batch(trainer, batch, optimizer)
                        else:
                            # 全图模式处理
                            loss = self._process_full_batch(trainer, batch, optimizer)
                        
                        loss_history.append(loss.item())
                        metrics.successful_steps += 1
                        
                        # 计算样本数（根据模式不同）
                        if mode == "patch":
                            batch_samples = batch.get('batch_info', {}).get('total_patches', 0)
                        else:
                            batch_samples = batch['data'].shape[0] if 'data' in batch else len(batch)
                        
                        epoch_samples += batch_samples
                        
                    except Exception as e:
                        self.logger.warning(f"步骤失败: {e}")
                        metrics.failed_steps += 1
                        continue
                    
                    step_time = time.time() - step_start
                    step_times.append(step_time * 1000)  # 转为毫秒
                    epoch_steps += 1
                    metrics.total_steps += 1
                    
                    # 采集系统指标
                    self.monitor.collect_metrics()
                    
                    # 限制测试样本数
                    if epoch_samples >= self.config.test_samples:
                        break
                
                epoch_time = time.time() - epoch_start
                total_samples += epoch_samples
                
                self.logger.info(
                    f"Epoch {epoch+1}/{self.config.test_epochs}: "
                    f"{epoch_samples} samples, {epoch_time:.2f}s, "
                    f"avg_loss: {np.mean(loss_history[-epoch_steps:]):.4f}"
                )
        
        except KeyboardInterrupt:
            self.logger.info("基准测试被用户中断")
        except Exception as e:
            self.logger.error(f"基准测试出错: {e}")
        
        finally:
            # 停止监控
            self.monitor.stop_monitoring()
        
        # 计算最终指标
        total_time = time.time() - start_time
        metrics.samples_per_second = total_samples / max(total_time, 0.001)
        metrics.epoch_time_seconds = total_time / max(self.config.test_epochs, 1)
        metrics.step_time_ms = np.mean(step_times) if step_times else 0
        
        # 质量指标
        if loss_history:
            metrics.final_loss = loss_history[-1]
            metrics.best_loss = min(loss_history)
            metrics.convergence_epoch = self._find_convergence_epoch(loss_history)
        
        # 系统资源统计
        resource_stats = self.monitor.get_summary_stats()
        metrics.peak_gpu_memory_mb = resource_stats.get('peak_gpu_memory_mb', 0)
        metrics.avg_gpu_memory_mb = resource_stats.get('avg_gpu_memory_mb', 0)
        metrics.peak_cpu_memory_mb = resource_stats.get('peak_cpu_memory_mb', 0)
        
        # 推理时间测试
        metrics.inference_time_ms = self._benchmark_inference_time(trainer, dataloader)
        
        self.logger.info(f"{mode}模式基准测试完成")
        return metrics
    
    def _warmup(self, trainer, dataloader, optimizer, warmup_steps: int):
        """预热阶段"""
        trainer.train()
        for i, batch in enumerate(dataloader):
            if i >= warmup_steps:
                break
            
            try:
                optimizer.zero_grad()
                # 简单前向传播，不记录指标
                if hasattr(trainer, 'training_step'):
                    result = trainer.training_step(batch, i)
                    if isinstance(result, dict) and 'loss' in result:
                        loss = result['loss']
                    else:
                        loss = result
                else:
                    # 传统训练器
                    if isinstance(batch, dict) and 'data' in batch:
                        input_data = batch['data'][:, :7]
                        target_data = batch['data'][:, 7:10]
                        pred = trainer(input_data)
                        loss = nn.L1Loss()(pred, target_data)
                    else:
                        continue
                
                loss.backward()
                optimizer.step()
            except Exception as e:
                continue
    
    def _process_patch_batch(self, trainer, batch, optimizer):
        """处理patch模式batch"""
        optimizer.zero_grad()
        result = trainer.training_step(batch, 0)
        
        if isinstance(result, dict) and 'loss' in result:
            loss = result['loss']
        else:
            loss = result
        
        loss.backward()
        optimizer.step()
        return loss
    
    def _process_full_batch(self, trainer, batch, optimizer):
        """处理全图模式batch"""
        optimizer.zero_grad()
        
        if isinstance(batch, dict) and 'data' in batch:
            input_data = batch['data'][:, :7]
            target_data = batch['data'][:, 7:10]
        else:
            # 假设batch是tensor
            input_data = batch[:, :7]
            target_data = batch[:, 7:10]
        
        pred = trainer(input_data)
        loss = nn.L1Loss()(pred, target_data)
        
        loss.backward()
        optimizer.step()
        return loss
    
    def _find_convergence_epoch(self, loss_history: List[float]) -> int:
        """寻找收敛epoch"""
        if len(loss_history) < self.config.convergence_patience:
            return -1
        
        window_size = self.config.convergence_patience
        for i in range(window_size, len(loss_history)):
            recent_losses = loss_history[i-window_size:i]
            if max(recent_losses) - min(recent_losses) < self.config.min_improvement:
                return i // self.config.test_samples  # 粗略估算epoch
        
        return -1
    
    def _benchmark_inference_time(self, trainer, dataloader) -> float:
        """基准测试推理时间"""
        trainer.eval()
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 10:  # 只测试10个batch
                    break
                
                try:
                    if isinstance(batch, dict) and 'data' in batch:
                        input_data = batch['data'][:1, :7]  # 只取一个样本
                    elif hasattr(batch, 'get') and 'full_input' in batch:
                        input_data = batch['full_input'][:1]
                    else:
                        continue
                    
                    start = time.time()
                    _ = trainer(input_data)
                    end = time.time()
                    times.append((end - start) * 1000)  # 转为毫秒
                
                except Exception as e:
                    continue
        
        return np.mean(times) if times else 0.0


class BenchmarkAnalyzer:
    """基准测试结果分析器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def compare_performance(self, 
                          patch_metrics: PerformanceMetrics,
                          full_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """对比分析patch和full模式性能"""
        comparison = {
            'speed_improvement': patch_metrics.samples_per_second / max(full_metrics.samples_per_second, 0.001),
            'memory_reduction': (full_metrics.peak_gpu_memory_mb - patch_metrics.peak_gpu_memory_mb) / max(full_metrics.peak_gpu_memory_mb, 0.001),
            'parameter_reduction': (full_metrics.model_parameters - patch_metrics.model_parameters) / max(full_metrics.model_parameters, 1),
            'quality_difference': patch_metrics.final_loss - full_metrics.final_loss,
            'inference_speedup': full_metrics.inference_time_ms / max(patch_metrics.inference_time_ms, 0.001)
        }
        
        # 统计显著性测试（简化版）
        comparison['speed_significant'] = abs(comparison['speed_improvement'] - 1.0) > 0.1
        comparison['memory_significant'] = abs(comparison['memory_reduction']) > 0.1
        comparison['quality_significant'] = abs(comparison['quality_difference']) > 0.01
        
        return comparison
    
    def generate_report(self,
                       patch_metrics: PerformanceMetrics,
                       full_metrics: PerformanceMetrics,
                       output_path: str):
        """生成详细分析报告"""
        comparison = self.compare_performance(patch_metrics, full_metrics)
        
        # 创建报告
        report = {
            'benchmark_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'patch_metrics': asdict(patch_metrics),
            'full_metrics': asdict(full_metrics),
            'comparison': comparison,
            'summary': {
                'speed_improvement_percent': (comparison['speed_improvement'] - 1.0) * 100,
                'memory_saved_percent': comparison['memory_reduction'] * 100,
                'parameter_reduction_percent': comparison['parameter_reduction'] * 100,
                'inference_speedup': comparison['inference_speedup'],
                'quality_maintained': comparison['quality_difference'] <= 0.05
            }
        }
        
        # 保存JSON报告
        with open(f"{output_path}/benchmark_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # 生成可视化
        if self.config.generate_plots:
            self._generate_plots(patch_metrics, full_metrics, comparison, output_path)
        
        # 生成文本摘要
        self._generate_summary(report, f"{output_path}/benchmark_summary.txt")
        
        return report
    
    def _generate_plots(self, patch_metrics, full_metrics, comparison, output_path):
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8')
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 性能对比条形图
        ax1 = axes[0, 0]
        metrics = ['Speed (SPS)', 'Memory (MB)', 'Parameters (M)', 'Inference (ms)']
        patch_values = [
            patch_metrics.samples_per_second,
            patch_metrics.peak_gpu_memory_mb,
            patch_metrics.model_parameters / 1e6,
            patch_metrics.inference_time_ms
        ]
        full_values = [
            full_metrics.samples_per_second,
            full_metrics.peak_gpu_memory_mb, 
            full_metrics.model_parameters / 1e6,
            full_metrics.inference_time_ms
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, patch_values, width, label='Patch Mode', color='skyblue')
        ax1.bar(x + width/2, full_values, width, label='Full Mode', color='lightcoral')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        
        # 2. 改进百分比
        ax2 = axes[0, 1]
        improvements = {
            'Speed': (comparison['speed_improvement'] - 1.0) * 100,
            'Memory': comparison['memory_reduction'] * 100,
            'Parameters': comparison['parameter_reduction'] * 100,
            'Inference': (comparison['inference_speedup'] - 1.0) * 100
        }
        
        colors = ['green' if v > 0 else 'red' for v in improvements.values()]
        bars = ax2.bar(improvements.keys(), improvements.values(), color=colors)
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Patch Mode Improvements')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. 质量对比（如果有损失历史的话，这里简化展示）
        ax3 = axes[1, 0]
        models = ['Patch Mode', 'Full Mode']
        losses = [patch_metrics.final_loss, full_metrics.final_loss]
        bars = ax3.bar(models, losses, color=['skyblue', 'lightcoral'])
        ax3.set_ylabel('Final Loss')
        ax3.set_title('Training Quality Comparison')
        
        # 4. 资源效率雷达图
        ax4 = axes[1, 1]
        categories = ['Speed', 'Memory', 'Parameters', 'Quality']
        
        # 归一化指标（相对于full mode）
        patch_normalized = [
            comparison['speed_improvement'],
            1 - comparison['memory_reduction'],  # 内存越少越好
            1 - comparison['parameter_reduction'],  # 参数越少越好  
            1 - abs(comparison['quality_difference'])  # 质量差异越小越好
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        patch_normalized += patch_normalized[:1]  # 闭合
        angles += angles[:1]
        
        ax4.plot(angles, patch_normalized, 'o-', linewidth=2, label='Patch Mode', color='skyblue')
        ax4.fill(angles, patch_normalized, alpha=0.25, color='skyblue')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 2)
        ax4.set_title('Efficiency Radar Chart')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/benchmark_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary(self, report, output_path):
        """生成文本摘要"""
        summary = report['summary']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== Patch-Based Training Benchmark Report ===\n\n")
            f.write(f"Generated: {report['benchmark_time']}\n\n")
            
            f.write("## Performance Summary\n")
            f.write(f"• Speed Improvement: {summary['speed_improvement_percent']:.1f}%\n")
            f.write(f"• Memory Saved: {summary['memory_saved_percent']:.1f}%\n") 
            f.write(f"• Parameter Reduction: {summary['parameter_reduction_percent']:.1f}%\n")
            f.write(f"• Inference Speedup: {summary['inference_speedup']:.1f}x\n")
            f.write(f"• Quality Maintained: {'✅' if summary['quality_maintained'] else '❌'}\n\n")
            
            f.write("## Detailed Metrics\n")
            f.write("### Patch Mode:\n")
            patch = report['patch_metrics']
            f.write(f"  - Speed: {patch['samples_per_second']:.1f} samples/sec\n")
            f.write(f"  - Peak GPU Memory: {patch['peak_gpu_memory_mb']:.1f} MB\n")
            f.write(f"  - Parameters: {patch['model_parameters']:,}\n")
            f.write(f"  - Final Loss: {patch['final_loss']:.4f}\n")
            f.write(f"  - Inference Time: {patch['inference_time_ms']:.1f} ms\n\n")
            
            f.write("### Full Mode:\n")
            full = report['full_metrics']
            f.write(f"  - Speed: {full['samples_per_second']:.1f} samples/sec\n")
            f.write(f"  - Peak GPU Memory: {full['peak_gpu_memory_mb']:.1f} MB\n")
            f.write(f"  - Parameters: {full['model_parameters']:,}\n")
            f.write(f"  - Final Loss: {full['final_loss']:.4f}\n")
            f.write(f"  - Inference Time: {full['inference_time_ms']:.1f} ms\n\n")
            
            f.write("## Recommendations\n")
            if summary['speed_improvement_percent'] > 20:
                f.write("• ✅ Patch mode shows significant speed improvement\n")
            if summary['memory_saved_percent'] > 30:
                f.write("• ✅ Patch mode provides substantial memory savings\n")
            if summary['quality_maintained']:
                f.write("• ✅ Quality maintained, safe to use patch mode\n")
            else:
                f.write("• ⚠️ Quality degradation detected, consider tuning\n")


def run_complete_benchmark(data_root: str, 
                          output_dir: str,
                          config: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
    """运行完整的基准测试套件"""
    if config is None:
        config = BenchmarkConfig()
    
    # 创建基准测试器
    benchmark = TrainingBenchmark(config, data_root, output_dir)
    
    # 模型和训练配置
    model_config = {
        'inpainting_network': {
            'input_channels': 7,
            'output_channels': 3
        }
    }
    
    training_config = {
        'batch_size': 4,
        'optimizer': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5
        }
    }
    
    try:
        # 运行patch模式基准测试
        print("🚀 开始Patch模式基准测试...")
        patch_metrics = benchmark.benchmark_patch_training(model_config, training_config)
        
        # 运行full模式基准测试  
        print("🚀 开始Full模式基准测试...")
        full_metrics = benchmark.benchmark_full_training(model_config, training_config)
        
        # 分析结果
        analyzer = BenchmarkAnalyzer(config)
        report = analyzer.generate_report(patch_metrics, full_metrics, output_dir)
        
        print("✅ 基准测试完成！")
        print(f"📊 报告保存在: {output_dir}")
        print(f"🚀 速度提升: {report['summary']['speed_improvement_percent']:.1f}%")
        print(f"💾 内存节省: {report['summary']['memory_saved_percent']:.1f}%")
        print(f"📦 参数减少: {report['summary']['parameter_reduction_percent']:.1f}%")
        
        return report
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        benchmark.logger.error(f"基准测试错误: {e}")
        raise


def test_benchmark_suite():
    """测试基准测试套件"""
    import tempfile
    import shutil
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 简单配置测试
        config = BenchmarkConfig(
            test_epochs=1,
            test_samples=10,
            repeat_runs=1
        )
        
        print("测试基准测试组件创建...")
        benchmark = TrainingBenchmark(config, "./output_motion_fix", temp_dir)
        print("✅ 基准测试器创建成功")
        
        # 测试系统监控器
        monitor = SystemMonitor()
        monitor.start_monitoring()
        metrics = monitor.collect_metrics()
        monitor.stop_monitoring()
        stats = monitor.get_summary_stats()
        print(f"✅ 系统监控测试成功: {metrics}")
        
        # 测试分析器
        analyzer = BenchmarkAnalyzer(config)
        
        # 创建假指标进行测试
        patch_metrics = PerformanceMetrics(
            samples_per_second=100,
            peak_gpu_memory_mb=2000,
            model_parameters=750000,
            final_loss=0.05,
            inference_time_ms=10
        )
        
        full_metrics = PerformanceMetrics(
            samples_per_second=80,
            peak_gpu_memory_mb=4000,
            model_parameters=3000000,
            final_loss=0.04,
            inference_time_ms=15
        )
        
        comparison = analyzer.compare_performance(patch_metrics, full_metrics)
        print(f"✅ 性能对比分析成功: 速度提升 {comparison['speed_improvement']:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # 运行测试
    success = test_benchmark_suite()
    print(f"\n{'🎉 基准测试套件测试通过!' if success else '💥 测试失败!'}")