#!/usr/bin/env python3
"""
残差学习统一工具类
解决跨组件的残差计算不一致问题
"""

import torch
from typing import Tuple

class ResidualLearningHelper:
    """残差学习统一实现工具类（线性HDR版本）"""

    # 全局统一的残差缩放因子（已移除缩放，固定为1.0）
    SCALE_FACTOR = 1.0

    @staticmethod
    def compute_residual_target(target_rgb: torch.Tensor, warped_rgb: torch.Tensor) -> torch.Tensor:
        """
        统一的残差目标计算（线性HDR空间，无范围裁剪）

        Args:
            target_rgb: [C,H,W] 或 [N,C,H,W]，线性HDR预处理后的RGB
            warped_rgb: [C,H,W] 或 [N,C,H,W]，线性HDR预处理后的RGB

        Returns:
            target_residual: 原始残差（不裁剪、不缩放）
        """
        return target_rgb - warped_rgb

    @staticmethod
    def reconstruct_from_residual(warped_rgb: torch.Tensor, residual_pred: torch.Tensor) -> torch.Tensor:
        """
        统一的完整图像重建（线性HDR空间，无范围裁剪）

        Args:
            warped_rgb: [C,H,W] 或 [N,C,H,W]
            residual_pred: [C,H,W] 或 [N,C,H,W]

        Returns:
            reconstructed_rgb: warped + residual（线性HDR范围，可能>1）
        """
        return warped_rgb + residual_pred
    
    @staticmethod
    def validate_residual_data(input_tensor: torch.Tensor, 
                             target_residual: torch.Tensor, 
                             target_rgb: torch.Tensor) -> bool:
        """
        验证残差学习数据的一致性
        
        Args:
            input_tensor: 输入张量 [7, H, W] 或 [N, 7, H, W]
            target_residual: 残差目标 [3, H, W] 或 [N, 3, H, W]
            target_rgb: RGB目标 [3, H, W] 或 [N, 3, H, W]
            
        Returns:
            bool: 数据是否一致
        """
        # 提取warped RGB
        if input_tensor.dim() == 3:
            warped_rgb = input_tensor[:3]
        else:
            warped_rgb = input_tensor[:, :3]
        
        # 重建验证：warped_rgb + residual * scale_factor ≈ target_rgb
        reconstructed = ResidualLearningHelper.reconstruct_from_residual(warped_rgb, target_residual)
        reconstruction_error = torch.mean(torch.abs(reconstructed - target_rgb))
        
        # 允许小的数值误差 (由于clamp操作和精度限制)
        tolerance = 0.01
        is_consistent = reconstruction_error < tolerance
        
        return is_consistent
    
    @staticmethod
    def get_config() -> dict:
        """获取残差学习配置参数"""
        return {
            'residual_scale_factor': ResidualLearningHelper.SCALE_FACTOR,
            'residual_range': [-1.0, 1.0],
            'rgb_range': [-1.0, 1.0],
            'reconstruction_formula': 'warped_rgb + residual * scale_factor'
        }
