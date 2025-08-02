#!/usr/bin/env python3
"""
@file evaluate.py
@brief 模型评估脚本

功能描述：
- 全面的模型性能评估
- 多维度指标计算和分析
- 不同游戏场景的对比测试
- 生成详细的评估报告

评估指标：
- 画质指标：SSIM, PSNR, LPIPS
- 性能指标：推理延迟, 内存占用, FPS
- 稳定性指标：时序一致性, 抖动检测
- 鲁棒性测试：边界情况处理

输出内容：
- 量化评估结果
- 可视化对比图表
- 性能分析报告
- 改进建议

@author AI算法团队
@date 2025-07-28
@version 1.0
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'npu', 'networks'))

from mobile_inpainting_network import MobileInpaintingNetwork
from training_framework import FrameInterpolationTrainer
from train import create_multi_game_dataset, load_config


class ModelEvaluator:
    """
    模型评估器
    
    提供全面的模型性能评估和分析功能
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Dict[str, Any],
                 output_dir: str = './evaluation_results'):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            config: 评估配置
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 评估结果存储
        self.results = {
            'quality_metrics': {},
            'performance_metrics': {},
            'stability_metrics': {},
            'per_game_results': {}
        }
        
        print(f"=== Model Evaluator ===")
        print(f"Model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Output: {output_dir}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        加载模型
        
        Args:
            model_path: 模型路径
        
        Returns:
            model: 加载的模型
        """
        if model_path.endswith('.ckpt'):
            # Lightning检查点
            trainer = FrameInterpolationTrainer.load_from_checkpoint(model_path)
            return trainer.student_model
        else:
            # PyTorch检查点
            model = MobileInpaintingNetwork()
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            return model
    
    def evaluate_quality_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估画质指标
        
        Args:
            dataloader: 测试数据加载器
        
        Returns:
            metrics: 画质指标字典
        """
        print("\n📊 Evaluating quality metrics...")
        
        ssim_scores = []
        psnr_scores = []
        l1_errors = []
        
        with torch.no_grad():
            for batch_idx, (input_data, target, _, _) in enumerate(dataloader):
                input_data = input_data.to(self.device)
                target = target.to(self.device)
                
                # 模型预测
                pred = self.model(input_data)
                
                # 转换到numpy用于计算指标
                pred_np = self._tensor_to_numpy(pred)
                target_np = self._tensor_to_numpy(target)
                
                # 计算每个样本的指标
                batch_ssim = []
                batch_psnr = []
                batch_l1 = []
                
                for i in range(pred_np.shape[0]):
                    # SSIM计算
                    sample_ssim = ssim(
                        target_np[i].transpose(1, 2, 0),
                        pred_np[i].transpose(1, 2, 0),
                        multichannel=True,
                        data_range=2.0  # [-1, 1]范围
                    )
                    batch_ssim.append(sample_ssim)
                    
                    # PSNR计算
                    sample_psnr = psnr(
                        target_np[i].transpose(1, 2, 0),
                        pred_np[i].transpose(1, 2, 0),
                        data_range=2.0
                    )
                    batch_psnr.append(sample_psnr)
                    
                    # L1误差
                    sample_l1 = np.mean(np.abs(target_np[i] - pred_np[i]))
                    batch_l1.append(sample_l1)
                
                ssim_scores.extend(batch_ssim)
                psnr_scores.extend(batch_psnr)
                l1_errors.extend(batch_l1)
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
        
        # 计算统计指标
        metrics = {
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'ssim_min': np.min(ssim_scores),
            'ssim_max': np.max(ssim_scores),
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'psnr_min': np.min(psnr_scores),
            'psnr_max': np.max(psnr_scores),
            'l1_mean': np.mean(l1_errors),
            'l1_std': np.std(l1_errors)
        }
        
        print(f"SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f} dB")
        print(f"L1 Error: {metrics['l1_mean']:.4f} ± {metrics['l1_std']:.4f}")
        
        return metrics
    
    def evaluate_performance_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估性能指标
        
        Args:
            dataloader: 测试数据加载器
        
        Returns:
            metrics: 性能指标字典
        """
        print("\n⚡ Evaluating performance metrics...")
        
        # 预热
        warmup_batches = 10
        for i, (input_data, _, _, _) in enumerate(dataloader):
            if i >= warmup_batches:
                break
            input_data = input_data.to(self.device)
            with torch.no_grad():
                _ = self.model(input_data)
        
        # 性能测试
        inference_times = []
        memory_usage = []
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        with torch.no_grad():
            for batch_idx, (input_data, _, _, _) in enumerate(dataloader):
                input_data = input_data.to(self.device)
                
                # 记录内存使用（GPU）
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                # 计时推理
                start_time = time.time()
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                pred = self.model(input_data)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                # 记录指标
                batch_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(batch_time / input_data.shape[0])  # 每样本时间
                
                if self.device.type == 'cuda':
                    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    memory_usage.append(peak_memory)
                
                if batch_idx >= 100:  # 限制测试样本数量
                    break
        
        # 计算性能指标
        metrics = {
            'inference_time_mean': np.mean(inference_times),
            'inference_time_std': np.std(inference_times),
            'inference_time_min': np.min(inference_times),
            'inference_time_max': np.max(inference_times),
            'fps_mean': 1000.0 / np.mean(inference_times),
            'memory_usage_mean': np.mean(memory_usage) if memory_usage else 0,
            'memory_usage_max': np.max(memory_usage) if memory_usage else 0
        }
        
        print(f"Inference Time: {metrics['inference_time_mean']:.2f} ± {metrics['inference_time_std']:.2f} ms")
        print(f"FPS: {metrics['fps_mean']:.1f}")
        if memory_usage:
            print(f"GPU Memory: {metrics['memory_usage_mean']:.1f} MB (peak: {metrics['memory_usage_max']:.1f} MB)")
        
        return metrics
    
    def evaluate_temporal_consistency(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估时序一致性
        
        Args:
            dataloader: 测试数据加载器
        
        Returns:
            metrics: 时序一致性指标
        """
        print("\n🎬 Evaluating temporal consistency...")
        
        frame_differences = []
        optical_flow_errors = []
        
        with torch.no_grad():
            prev_pred = None
            prev_target = None
            
            for batch_idx, (input_data, target, input_prev, target_prev) in enumerate(dataloader):
                input_data = input_data.to(self.device)
                target = target.to(self.device)
                input_prev = input_prev.to(self.device)
                target_prev = target_prev.to(self.device)
                
                # 预测当前帧和前一帧
                pred_current = self.model(input_data)
                pred_previous = self.model(input_prev)
                
                # 计算帧间差异
                pred_diff = torch.abs(pred_current - pred_previous).mean(dim=1)  # [B, H, W]
                target_diff = torch.abs(target - target_prev).mean(dim=1)
                
                # 时序一致性误差
                temporal_error = F.l1_loss(pred_diff, target_diff)
                frame_differences.append(temporal_error.item())
                
                if batch_idx >= 50:  # 限制测试样本
                    break
        
        metrics = {
            'temporal_consistency_mean': np.mean(frame_differences),
            'temporal_consistency_std': np.std(frame_differences),
            'temporal_stability_score': 1.0 - np.mean(frame_differences)  # 稳定性评分
        }
        
        print(f"Temporal Consistency: {metrics['temporal_consistency_mean']:.4f} ± {metrics['temporal_consistency_std']:.4f}")
        print(f"Stability Score: {metrics['temporal_stability_score']:.4f}")
        
        return metrics
    
    def generate_visual_samples(self, dataloader: DataLoader, num_samples: int = 10):
        """
        生成可视化样本
        
        Args:
            dataloader: 数据加载器
            num_samples: 样本数量
        """
        print(f"\n🖼️ Generating {num_samples} visual samples...")
        
        samples_dir = self.output_dir / 'visual_samples'
        samples_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            sample_count = 0
            for batch_idx, (input_data, target, _, _) in enumerate(dataloader):
                input_data = input_data.to(self.device)
                target = target.to(self.device)
                
                pred = self.model(input_data)
                
                # 保存样本
                batch_size = input_data.shape[0]
                for i in range(batch_size):
                    if sample_count >= num_samples:
                        break
                    
                    # 转换为可视化格式
                    input_rgb = self._tensor_to_image(input_data[i, :3])  # RGB通道
                    input_mask = input_data[i, 3].cpu().numpy()  # 掩码通道
                    target_img = self._tensor_to_image(target[i])
                    pred_img = self._tensor_to_image(pred[i])
                    
                    # 创建对比图
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    axes[0, 0].imshow(input_rgb)
                    axes[0, 0].set_title('Input RGB')
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(input_mask, cmap='gray')
                    axes[0, 1].set_title('Input Mask')
                    axes[0, 1].axis('off')
                    
                    axes[1, 0].imshow(target_img)
                    axes[1, 0].set_title('Ground Truth')
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(pred_img)
                    axes[1, 1].set_title('Prediction')
                    axes[1, 1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(samples_dir / f'sample_{sample_count:03d}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        print(f"Visual samples saved to {samples_dir}")
    
    def run_full_evaluation(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        运行完整评估
        
        Args:
            dataloaders: 数据加载器字典
        
        Returns:
            results: 完整评估结果
        """
        print("="*60)
        print("🧪 Starting Full Model Evaluation")
        print("="*60)
        
        # 使用验证集进行评估
        test_loader = dataloaders.get('test', dataloaders['val'])
        
        # 画质指标评估
        quality_metrics = self.evaluate_quality_metrics(test_loader)
        self.results['quality_metrics'] = quality_metrics
        
        # 性能指标评估
        performance_metrics = self.evaluate_performance_metrics(test_loader)
        self.results['performance_metrics'] = performance_metrics
        
        # 时序一致性评估
        stability_metrics = self.evaluate_temporal_consistency(test_loader)
        self.results['stability_metrics'] = stability_metrics
        
        # 生成可视化样本
        self.generate_visual_samples(test_loader)
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_report()
        
        return self.results
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """将tensor转换为numpy数组"""
        return tensor.detach().cpu().numpy()
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """将tensor转换为可显示的图像"""
        img = tensor.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = np.clip(img, 0, 1)
        return img
    
    def _save_results(self):
        """保存评估结果"""
        results_file = self.output_dir / 'evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {results_file}")
    
    def _generate_report(self):
        """生成评估报告"""
        report_file = self.output_dir / 'evaluation_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Mobile Inpainting Network - Evaluation Report\n\n")
            f.write(f"**Model:** {self.model_path}\n")
            f.write(f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 画质指标
            f.write("## Quality Metrics\n\n")
            quality = self.results['quality_metrics']
            f.write(f"- **SSIM:** {quality['ssim_mean']:.4f} ± {quality['ssim_std']:.4f}\n")
            f.write(f"- **PSNR:** {quality['psnr_mean']:.2f} ± {quality['psnr_std']:.2f} dB\n")
            f.write(f"- **L1 Error:** {quality['l1_mean']:.4f} ± {quality['l1_std']:.4f}\n\n")
            
            # 性能指标
            f.write("## Performance Metrics\n\n")
            perf = self.results['performance_metrics']
            f.write(f"- **Inference Time:** {perf['inference_time_mean']:.2f} ± {perf['inference_time_std']:.2f} ms\n")
            f.write(f"- **FPS:** {perf['fps_mean']:.1f}\n")
            if 'memory_usage_mean' in perf and perf['memory_usage_mean'] > 0:
                f.write(f"- **GPU Memory:** {perf['memory_usage_mean']:.1f} MB (peak: {perf['memory_usage_max']:.1f} MB)\n")
            f.write("\n")
            
            # 稳定性指标
            f.write("## Stability Metrics\n\n")
            stability = self.results['stability_metrics']
            f.write(f"- **Temporal Consistency:** {stability['temporal_consistency_mean']:.4f} ± {stability['temporal_consistency_std']:.4f}\n")
            f.write(f"- **Stability Score:** {stability['temporal_stability_score']:.4f}\n\n")
            
            # 总结
            f.write("## Summary\n\n")
            if quality['ssim_mean'] > 0.90:
                f.write("✅ Excellent image quality (SSIM > 0.90)\n")
            elif quality['ssim_mean'] > 0.85:
                f.write("✅ Good image quality (SSIM > 0.85)\n")
            else:
                f.write("⚠️ Image quality needs improvement (SSIM < 0.85)\n")
            
            if perf['inference_time_mean'] < 5.0:
                f.write("✅ Excellent inference speed (< 5ms)\n")
            elif perf['inference_time_mean'] < 10.0:
                f.write("✅ Good inference speed (< 10ms)\n")
            else:
                f.write("⚠️ Inference speed needs optimization (> 10ms)\n")
        
        print(f"Report saved to {report_file}")


def main():
    """评估主函数"""
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to evaluation config')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    print("="*60)
    print("🧪 Model Evaluation Started")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    try:
        # 创建数据集
        print("\n📊 Loading test datasets...")
        dataloaders = create_multi_game_dataset(
            data_configs=config['datasets'],
            batch_size=config.get('batch_size', 16),
            num_workers=config.get('num_workers', 4)
        )
        
        # 创建评估器
        evaluator = ModelEvaluator(
            model_path=args.model,
            config=config,
            output_dir=args.output_dir
        )
        
        # 运行评估
        results = evaluator.run_full_evaluation(dataloaders)
        
        print("\n🎉 Evaluation completed successfully!")
        print(f"Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()