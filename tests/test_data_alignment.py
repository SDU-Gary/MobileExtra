#!/usr/bin/env python3
"""
数据对齐验证脚本
验证warped_rgb与target_rgb之间的像素级对齐质量
这对残差学习的成功至关重要
"""

import os
import numpy as np
import torch
import cv2
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from pathlib import Path


class DataAlignmentVerifier:
    """数据对齐验证器"""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.training_dir = os.path.join(data_root, "training_data")
        
        if not os.path.exists(self.training_dir):
            raise FileNotFoundError(f"训练数据目录不存在: {self.training_dir}")
        
        # 获取数据文件
        self.data_files = [f for f in os.listdir(self.training_dir) if f.endswith('.npy')]
        self.data_files.sort()
        
        if len(self.data_files) == 0:
            raise ValueError(f"没有找到数据文件: {self.training_dir}")
        
        print(f"✅ 找到 {len(self.data_files)} 个数据文件")
    
    def load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载单个样本
        
        Returns:
            warped_rgb: [3, H, W] warped RGB图像
            target_rgb: [3, H, W] 目标RGB图像  
            holes_mask: [1, H, W] 空洞掩码
        """
        data_path = os.path.join(self.training_dir, self.data_files[idx])
        
        # 加载10通道数据
        full_data = np.load(data_path).astype(np.float32)
        
        if full_data.shape[0] != 10:
            raise ValueError(f"数据格式错误: 期望10通道，实际{full_data.shape[0]}通道")
        
        # 分离数据
        warped_rgb = full_data[0:3]      # [3, H, W]
        holes_mask = full_data[3:4]      # [1, H, W] 
        occlusion_mask = full_data[4:5]  # [1, H, W]
        residual_mv = full_data[5:7]     # [2, H, W]
        target_rgb = full_data[7:10]     # [3, H, W]
        
        return warped_rgb, target_rgb, holes_mask, occlusion_mask, residual_mv
    
    def compute_alignment_metrics(self, warped_rgb: np.ndarray, 
                                target_rgb: np.ndarray,
                                holes_mask: np.ndarray) -> Dict[str, float]:
        """计算对齐质量指标"""
        
        # 1. 整体SSIM（结构相似性）
        ssim_score = self._compute_ssim(warped_rgb, target_rgb)
        
        # 2. 非空洞区域的像素差异
        non_hole_mask = (holes_mask[0] < 0.5)  # 非空洞区域
        if np.sum(non_hole_mask) > 0:
            # 计算非空洞区域的MAE
            warped_non_hole = warped_rgb[:, non_hole_mask]
            target_non_hole = target_rgb[:, non_hole_mask]
            non_hole_mae = np.mean(np.abs(warped_non_hole - target_non_hole))
            
            # 计算非空洞区域的MSE
            non_hole_mse = np.mean((warped_non_hole - target_non_hole) ** 2)
        else:
            non_hole_mae = float('inf')
            non_hole_mse = float('inf')
        
        # 3. 空洞区域的像素差异（理论上应该较大）
        hole_mask = (holes_mask[0] >= 0.5)  # 空洞区域
        if np.sum(hole_mask) > 0:
            warped_hole = warped_rgb[:, hole_mask]
            target_hole = target_rgb[:, hole_mask]
            hole_mae = np.mean(np.abs(warped_hole - target_hole))
            hole_mse = np.mean((warped_hole - target_hole) ** 2)
        else:
            hole_mae = 0.0
            hole_mse = 0.0
        
        # 4. 边缘区域对齐质量
        edge_alignment = self._compute_edge_alignment(warped_rgb, target_rgb)
        
        # 5. 计算残差统计
        residual = target_rgb - warped_rgb
        residual_mean = np.mean(np.abs(residual))
        residual_std = np.std(residual)
        
        return {
            'overall_ssim': ssim_score,
            'non_hole_mae': non_hole_mae,
            'non_hole_mse': non_hole_mse,
            'hole_mae': hole_mae,
            'hole_mse': hole_mse,
            'edge_alignment': edge_alignment,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'hole_ratio': np.mean(holes_mask)
        }
    
    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算SSIM（简化版本）"""
        # 转换为灰度图计算
        if img1.shape[0] == 3:  # RGB
            gray1 = np.mean(img1, axis=0)
            gray2 = np.mean(img2, axis=0)
        else:
            gray1 = img1[0]
            gray2 = img2[0]
        
        # 归一化到[0,1]
        gray1 = np.clip(gray1, 0, 1)
        gray2 = np.clip(gray2, 0, 1)
        
        # 计算均值
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        # 计算方差和协方差
        sigma1_sq = np.var(gray1)
        sigma2_sq = np.var(gray2)
        sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM常数
        C1 = 0.01**2
        C2 = 0.03**2
        
        # 计算SSIM
        ssim = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
        
        return float(ssim)
    
    def _compute_edge_alignment(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算边缘对齐质量"""
        # 使用Sobel算子计算边缘
        gray1 = np.mean(img1, axis=0)
        gray2 = np.mean(img2, axis=0)
        
        # 确保数据类型正确
        gray1 = np.clip(gray1 * 255, 0, 255).astype(np.uint8)
        gray2 = np.clip(gray2 * 255, 0, 255).astype(np.uint8)
        
        # 计算边缘
        sobelx1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        sobely1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        edges1 = np.sqrt(sobelx1**2 + sobely1**2)
        
        sobelx2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        sobely2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        edges2 = np.sqrt(sobelx2**2 + sobely2**2)
        
        # 计算边缘相似性
        edge_diff = np.mean(np.abs(edges1 - edges2))
        edge_max = max(np.mean(edges1), np.mean(edges2))
        
        if edge_max > 0:
            edge_alignment = 1.0 - (edge_diff / edge_max)
        else:
            edge_alignment = 1.0
        
        return float(np.clip(edge_alignment, 0, 1))
    
    def verify_batch(self, sample_count: int = 10) -> Dict[str, float]:
        """验证一批样本的对齐质量"""
        
        if sample_count > len(self.data_files):
            sample_count = len(self.data_files)
        
        print(f"🔍 验证 {sample_count} 个样本的数据对齐质量...")
        
        all_metrics = []
        
        for i in range(sample_count):
            try:
                # 加载样本
                warped_rgb, target_rgb, holes_mask, occlusion_mask, residual_mv = self.load_sample(i)
                
                # 计算对齐指标
                metrics = self.compute_alignment_metrics(warped_rgb, target_rgb, holes_mask)
                all_metrics.append(metrics)
                
                print(f"  样本 {i+1}/{sample_count}: SSIM={metrics['overall_ssim']:.3f}, "
                      f"非空洞MAE={metrics['non_hole_mae']:.4f}, "
                      f"空洞比例={metrics['hole_ratio']:.3f}")
                
            except Exception as e:
                print(f"❌ 样本 {i} 验证失败: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("没有成功验证的样本")
        
        # 计算统计结果
        summary = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if not np.isnan(m[metric_name]) and not np.isinf(m[metric_name])]
            if values:
                summary[f"{metric_name}_mean"] = np.mean(values)
                summary[f"{metric_name}_std"] = np.std(values)
                summary[f"{metric_name}_min"] = np.min(values)
                summary[f"{metric_name}_max"] = np.max(values)
        
        return summary
    
    def visualize_alignment(self, sample_idx: int = 0, save_path: str = None):
        """可视化数据对齐情况"""
        
        # 加载样本
        warped_rgb, target_rgb, holes_mask, occlusion_mask, residual_mv = self.load_sample(sample_idx)
        
        # 计算残差
        residual = target_rgb - warped_rgb
        
        # 准备可视化数据
        h, w = warped_rgb.shape[1], warped_rgb.shape[2]
        
        # 转换为可视化格式 [H, W, C]
        warped_vis = np.transpose(np.clip(warped_rgb, 0, 1), (1, 2, 0))
        target_vis = np.transpose(np.clip(target_rgb, 0, 1), (1, 2, 0))
        residual_vis = np.transpose(np.clip(np.abs(residual) * 5, 0, 1), (1, 2, 0))  # 放大残差便于观察
        holes_vis = np.repeat(holes_mask[0:1], 3, axis=0)
        holes_vis = np.transpose(holes_vis, (1, 2, 0))
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'数据对齐验证 - 样本 {sample_idx}', fontsize=16)
        
        axes[0, 0].imshow(warped_vis)
        axes[0, 0].set_title('Warped RGB')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_vis)
        axes[0, 1].set_title('Target RGB')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(residual_vis)
        axes[0, 2].set_title('残差 (×5)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(holes_vis)
        axes[1, 0].set_title('空洞掩码')
        axes[1, 0].axis('off')
        
        # 残差直方图
        residual_flat = residual.flatten()
        axes[1, 1].hist(residual_flat, bins=50, alpha=0.7)
        axes[1, 1].set_title('残差分布')
        axes[1, 1].set_xlabel('残差值')
        axes[1, 1].set_ylabel('频次')
        
        # 对齐质量指标
        metrics = self.compute_alignment_metrics(warped_rgb, target_rgb, holes_mask)
        metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontsize=10, fontfamily='monospace')
        axes[1, 2].set_title('对齐指标')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 可视化结果保存至: {save_path}")
        
        plt.show()
        
        return metrics


def main():
    """主函数"""
    print("🔍 数据对齐验证开始")
    
    try:
        # 创建验证器
        verifier = DataAlignmentVerifier("./output_motion_fix")
        
        # 验证批量样本
        print("\n📊 批量验证结果:")
        summary = verifier.verify_batch(sample_count=20)
        
        print("\n📈 统计摘要:")
        for key, value in summary.items():
            print(f"  {key}: {value:.6f}")
        
        # 评估对齐质量
        print("\n🎯 对齐质量评估:")
        ssim_mean = summary.get('overall_ssim_mean', 0)
        non_hole_mae = summary.get('non_hole_mae_mean', float('inf'))
        hole_mae = summary.get('hole_mae_mean', 0)
        
        print(f"  整体SSIM: {ssim_mean:.3f} ({'优秀' if ssim_mean > 0.8 else '良好' if ssim_mean > 0.6 else '需要改进'})")
        print(f"  非空洞MAE: {non_hole_mae:.4f} ({'优秀' if non_hole_mae < 0.05 else '良好' if non_hole_mae < 0.1 else '需要改进'})")
        print(f"  空洞/非空洞差异: {hole_mae/non_hole_mae:.2f}x ({'合理' if hole_mae > non_hole_mae*2 else '偏小'})")
        
        # 可视化示例
        print("\n🖼️ 生成可视化示例...")
        verifier.visualize_alignment(0, "./data_alignment_verification.png")
        
        print("\n✅ 数据对齐验证完成")
        
        # 返回是否通过验证
        is_good_alignment = (ssim_mean > 0.6 and non_hole_mae < 0.1)
        if is_good_alignment:
            print("🎉 数据对齐质量良好，适合残差学习")
        else:
            print("⚠️ 数据对齐质量需要改进，可能影响残差学习效果")
        
        return is_good_alignment
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


if __name__ == "__main__":
    main()